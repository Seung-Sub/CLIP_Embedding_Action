"""M-C — 동적 가중 oracle headroom: ζ(α)=[2(1−α)·ζ_m ; 2α·ζ_w] (표준화 통화).

α 그리드(0.05..0.95, 19점)로 h 디코드 오차행렬 E[α, i]를 만든 뒤:
  rung0  : 고정 α 스윕 (α=0.5 = 학습 시 균등 가중 기준선)
  rung1o : grasp-bin별 oracle α* — half-A 선택 → half-B 평가 (게이트 headroom)
  ceil   : per-timestep oracle (천장, 게이트 아님)
  rung1a : analytic gate α_t = clip(a + b·rel_t), rel_t=‖ζ_w‖/(‖ζ_m‖+‖ζ_w‖)
  rung2g : gripper-phase gate (open / closing / closed 3국면별 α)
전 게이트는 per-sample E 열 선택으로 환원 (α는 그리드 스냅).
사전등록 임계 = outputs/week2_dualdyn/prereg_week2_dualdyn.json (실행 전 고정).

  cd ~/clip_ws && WK2_THREADS=12 python3 scratchpad/week2_dualdyn/mc_gate_headroom.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wk2_common import (OUT, GRIP, BIN_NAMES, load_dual_std, build_split,   # noqa: E402
                        policy_zeta, lang_embeddings, decode_batched)
from w0_common import DummyAnchor, split_ids                                # noqa: E402

ALPHA = np.round(np.arange(0.05, 0.951, 0.05), 2)          # 19점 (0.5 포함)
A_GRID = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]
B_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]


def snap(a):
    return np.clip(np.round(a / 0.05) * 0.05, 0.05, 0.95)


def main():
    t0 = time.time()
    ck1, ae, ck2, cfg2, model, use_lang = load_dual_std()
    dm, dw = ck1["dim_main"], ck1["dim_wrist"]

    va, mva, ds = build_split(cfg2, ck1, "val")
    n = len(va["Cf"])
    L = None
    if use_lang:
        files = ds.episode_files()
        val_ids, _ = split_ids(len(files), cfg2["train"]["seed"],
                               cfg2["data"]["val_episodes"])
        sel = [files[i] for i in val_ids]
        main_a = DummyAnchor(ck1["anchor"]["cache_key"], dm)
        ep_lens = [int((mva["ep_id"] == i).sum()) for i in range(len(sel))]
        L = lang_embeddings(ds, main_a, sel, ep_lens)
    zeta_pi, zeta_gt = policy_zeta(model, ae, va, use_lang, ds, cfg2, ck1,
                                   lang_arr=L)
    Y = va["Cf"].astype(np.float64)
    dev_all = ((Y - Y.mean(0)) ** 2).sum()
    print(f"[M-C] val {n}, ζ 준비 ({time.time()-t0:.0f}s)")

    half_A = np.arange(n) % 2 == 0            # 선택
    half_B = ~half_A                          # 평가

    def r2_of(err_sel, mask):
        y = Y[mask]
        dev = ((y - y.mean(0)) ** 2).sum()
        return float(1 - err_sel[mask].sum() / (dev + 1e-12))

    res = {"probe": "mc_gate_headroom", "n_val": n,
           "alpha_grid": ALPHA.tolist(),
           "prereg": "outputs/week2_dualdyn/prereg_week2_dualdyn.json",
           "sources": {}}

    for src, Z in (("policy", zeta_pi), ("oracle", zeta_gt)):
        # ---- 오차행렬 E[α, i] = per-sample SSE ----
        E = np.empty((len(ALPHA), n), np.float64)
        for k, a in enumerate(ALPHA):
            z = Z.copy()
            z[:, :dm] *= 2 * (1 - a)
            z[:, dm:] *= 2 * a
            ahat = decode_batched(ae, z, va["Zc"], va["Zwc"]).astype(np.float64)
            E[k] = ((Y - ahat) ** 2).sum(1)
        i050 = int(np.where(ALPHA == 0.5)[0][0])
        r2_eq_all = float(1 - E[i050].sum() / dev_all)
        r2_eq_B = r2_of(E[i050], half_B)
        sweep = {f"{a:.2f}": float(1 - E[k].sum() / dev_all)
                 for k, a in enumerate(ALPHA)}
        k_best = int(np.argmin(E[:, half_A].sum(1)))
        r2_fixedbest_B = r2_of(E[k_best], half_B)

        # ---- rung1o: bin별 oracle (A 선택 → B 평가) ----
        sel_bin = np.full(n, i050)
        bin_alpha = {}
        for b, name in enumerate(BIN_NAMES):
            m = mva["bin"] == b
            if (m & half_A).sum() < 15:
                bin_alpha[name] = 0.5
                continue
            kb = int(np.argmin(E[:, m & half_A].sum(1)))
            sel_bin[m] = kb
            bin_alpha[name] = float(ALPHA[kb])
        r2_bin_B = r2_of(E[sel_bin, np.arange(n)], half_B)

        # ---- ceil: per-timestep oracle ----
        r2_ceil_all = float(1 - E.min(0).sum() / dev_all)
        r2_ceil_B = r2_of(E.min(0), half_B)

        # ---- rung1a: analytic magnitude gate ----
        nm = np.linalg.norm(Z[:, :dm], axis=1)
        nw = np.linalg.norm(Z[:, dm:], axis=1)
        rel = nw / np.maximum(nm + nw, 1e-9)
        best = None
        for a0 in A_GRID:
            for b0 in B_GRID:
                al = snap(a0 + b0 * rel)
                ki = np.searchsorted(ALPHA, al - 1e-6)
                err = E[ki, np.arange(n)]
                sA = err[half_A].sum()
                if best is None or sA < best[0]:
                    best = (sA, a0, b0, err)
        _, a0, b0, err_an = best
        r2_an_B = r2_of(err_an, half_B)

        # ---- rung2g: gripper-phase gate (open / closing / closed) ----
        closed = mva["g_prev"] > 0.5
        closing = mva["g_closing"] & ~closed
        phases = {"open": ~closed & ~closing, "closing": closing,
                  "closed": closed}
        sel_ph = np.full(n, i050)
        ph_alpha = {}
        for name, m in phases.items():
            if (m & half_A).sum() < 15:
                ph_alpha[name] = 0.5
                continue
            kp = int(np.argmin(E[:, m & half_A].sum(1)))
            sel_ph[m] = kp
            ph_alpha[name] = float(ALPHA[kp])
        r2_ph_B = r2_of(np.take_along_axis(E, sel_ph[None], 0)[0], half_B)

        gains = {"fixed_best_alpha": r2_fixedbest_B - r2_eq_B,
                 "per_bin_oracle": r2_bin_B - r2_eq_B,
                 "per_timestep_ceiling": r2_ceil_B - r2_eq_B,
                 "analytic_gate": r2_an_B - r2_eq_B,
                 "gripper_phase_gate": r2_ph_B - r2_eq_B}
        res["sources"][src] = {
            "r2_equal_all": r2_eq_all, "r2_equal_halfB": r2_eq_B,
            "fixed_alpha_sweep_all": sweep,
            "fixed_best_alpha": float(ALPHA[k_best]),
            "bin_alpha_star": bin_alpha,
            "phase_alpha_star": ph_alpha,
            "phase_n": {k: int(v.sum()) for k, v in phases.items()},
            "analytic_gate_ab": [a0, b0],
            "rel_stats": {"median": float(np.median(rel)),
                          "p10": float(np.percentile(rel, 10)),
                          "p90": float(np.percentile(rel, 90))},
            "r2_halfB": {"equal": r2_eq_B, "fixed_best": r2_fixedbest_B,
                         "per_bin_oracle": r2_bin_B,
                         "per_timestep_ceiling": r2_ceil_B,
                         "analytic_gate": r2_an_B,
                         "gripper_phase_gate": r2_ph_B},
            "gains_halfB": gains}
        print(f"[{src}] eq(all) {r2_eq_all:+.4f} | halfB gains: "
              + " ".join(f"{k}={v:+.4f}" for k, v in gains.items()))

    g = res["sources"]["policy"]["gains_halfB"]
    if g["per_bin_oracle"] < 0.01:
        verdict = "NO_HEADROOM (<+0.01) — 학습형 게이트 폐쇄, param-0 최대"
    elif (g["analytic_gate"] >= 0.005 or g["gripper_phase_gate"] >= 0.005):
        verdict = "PARAM0_ELIGIBLE — analytic/gripper 게이트 셀 자격"
    elif g["per_bin_oracle"] >= 0.02:
        verdict = "LEARNED_GATE_DISCUSSABLE — oracle 있음, param-0 회수 실패"
    else:
        verdict = "MARGINAL (+0.01~0.02 oracle, param-0 미달) — 보류"
    res["verdict"] = verdict
    res["_meta"] = {"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "host": os.uname().nodename, "device": "cpu",
                    "wall_s": round(time.time() - t0)}
    p = OUT / "mc_gate_headroom.json"
    p.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"[M-C] → {verdict}\n저장: {p}")


if __name__ == "__main__":
    main()
