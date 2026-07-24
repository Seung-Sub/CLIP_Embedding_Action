"""M-B — 교차쌍 h 민감도: 분리 정책 2개(ζ_main·ζ_wrist 각각 독립 예측)의 위험 정량.

h([ζ_m; ζ_w], [z_m; z_w]) 디코드 R²를 조건별 비교 (상태는 항상 true):
  true       : (ζ_m_i, ζ_w_i) — jointly generated
  jitter8f   : ζ_w를 같은 ep +8프레임(+4샘플) 시점으로 교체 (시간 비동기)
  nn_task    : ζ_w를 같은 태스크·다른 ep, 상태([z_m;z_w] cosine) 최근접 샘플로 교체
  phase_task : ζ_w를 같은 태스크·다른 ep, 그립-오프셋 최근접(±4f 우선) 샘플로 교체
  cross_task : ζ_w를 다른 태스크 무작위 샘플로 교체
  wrist_zero : ζ_w = 0 (정보 0 바닥)
  (+ 대칭 참고: nn_task_swap_main — ζ_m 교체판)
S_cond = (R²_true − R²_cond)/(R²_true − R²_zero). 사전등록 임계는
outputs/week2_dualdyn/prereg_week2_dualdyn.json (실행 전 고정).

  cd ~/clip_ws && WK2_THREADS=12 python3 scratchpad/week2_dualdyn/mb_crosspair_sensitivity.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wk2_common import (OUT, GRIP, load_dual_std, build_split, policy_zeta,  # noqa: E402
                        lang_embeddings, decode_batched, per_dim_r2)
from w0_common import DummyAnchor, r2_pooled, split_ids                      # noqa: E402
from data.libero import LiberoDataset                                        # noqa: E402


def partners(meta, arrs, rng):
    """조건별 파트너 인덱스 배열 계산 (val 내부)."""
    n = len(meta["ep_id"])
    ep, task, off = meta["ep_id"], meta["task_id"], meta["offset"]
    has = meta["has_grasp"]
    idx = np.arange(n)

    # jitter: 같은 ep 내 +4샘플 (에피소드 경계 클램프)
    jit = np.empty(n, np.int64)
    for e in np.unique(ep):
        ii = idx[ep == e]
        jit[ii] = np.minimum(ii + 4, ii[-1])

    # 상태 NN (같은 태스크, 다른 ep): cosine on [Zc ⊕ Zwc]
    S = np.concatenate([arrs["Zc"], arrs["Zwc"]], 1)
    S = S / np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-9)
    nn = np.empty(n, np.int64)
    ph = np.empty(n, np.int64)
    for t in np.unique(task):
        ii = idx[task == t]
        Si = S[ii]
        D = Si @ Si.T                                   # (m, m)
        same_ep = ep[ii][:, None] == ep[ii][None, :]
        D[same_ep] = -np.inf
        nn[ii] = ii[np.argmax(D, axis=1)]
        # phase-match: |offset 차| 최소 (동률은 랜덤), grasp 없는 샘플은 NN 폴백
        offi = off[ii].astype(np.float64)
        offi[~has[ii]] = np.nan
        for a_local, a_glob in enumerate(ii):
            cand = ~same_ep[a_local]
            if not has[a_glob]:
                ph[a_glob] = nn[a_glob]
                continue
            d_off = np.abs(offi - offi[a_local])
            d_off[~cand] = np.inf
            d_off[np.isnan(d_off)] = np.inf
            mn = d_off.min()
            if not np.isfinite(mn):
                ph[a_glob] = nn[a_glob]
                continue
            pool = ii[d_off <= max(mn, 4)]
            ph[a_glob] = rng.choice(pool)

    # cross-task 무작위
    ct = np.empty(n, np.int64)
    for t in np.unique(task):
        m = task == t
        pool = idx[~m]
        ct[m] = rng.choice(pool, size=m.sum(), replace=True)
    return {"jitter8f": jit, "nn_task": nn, "phase_task": ph,
            "cross_task": ct}


def main():
    t0 = time.time()
    ck1, ae, ck2, cfg2, model, use_lang = load_dual_std()
    dm, dw = ck1["dim_main"], ck1["dim_wrist"]
    n_chunk, act_dim = ck1["n_chunk"], ck1["action_dim"]

    va, mva, ds = build_split(cfg2, ck1, "val")
    n = len(va["Cf"])
    print(f"[M-B] val {n} samples ({time.time()-t0:.0f}s)")

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
    print(f"[M-B] ζ̂/ζ_gt 완료 ({time.time()-t0:.0f}s)")

    rng = np.random.RandomState(0)
    P = partners(mva, va, rng)
    Yva = va["Cf"]

    res = {"probe": "mb_crosspair_sensitivity", "n_val": n,
           "prereg": "outputs/week2_dualdyn/prereg_week2_dualdyn.json (실행 전 고정)",
           "sources": {}}
    for src, Z in (("policy", zeta_pi), ("oracle", zeta_gt)):
        conds = {}

        def run(tag, z):
            ahat = decode_batched(ae, z, va["Zc"], va["Zwc"])
            per = per_dim_r2(Yva, ahat, n_chunk, act_dim)
            conds[tag] = {"r2": r2_pooled(Yva, ahat), "grip_r2": per[GRIP],
                          "arm_r2": float(np.mean(per[:GRIP]))}
            print(f"  [{src}/{tag}] R² {conds[tag]['r2']:+.4f} "
                  f"grip {per[GRIP]:+.4f}")

        run("true", Z)
        for tag, pi in P.items():
            z = Z.copy(); z[:, dm:] = Z[pi, dm:]
            run(tag, z)
        z = Z.copy(); z[:, dm:] = 0
        run("wrist_zero", z)
        z = Z.copy(); z[:, :dm] = Z[P["nn_task"], :dm]     # 대칭 참고
        run("nn_task_swap_main", z)

        span = conds["true"]["r2"] - conds["wrist_zero"]["r2"]
        S = {tag: (conds["true"]["r2"] - conds[tag]["r2"]) / max(span, 1e-9)
             for tag in ("jitter8f", "nn_task", "phase_task", "cross_task")}
        res["sources"][src] = {"conditions": conds, "S": S,
                               "span_true_minus_zero": span}
        print(f"  [{src}] S: " + " ".join(f"{k}={v:+.3f}" for k, v in S.items()))

    s_nn = res["sources"]["policy"]["S"]["nn_task"]
    verdict = ("SEPARATE_SAFE (S_NN<=0.25)" if s_nn <= 0.25 else
               "MIDDLE_GROUND shared-ctx+two-head (0.25<S_NN<=0.60)"
               if s_nn <= 0.60 else "JOINT_REQUIRED (S_NN>0.60)")
    res["verdict"] = verdict
    res["_meta"] = {"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "host": os.uname().nodename, "device": "cpu",
                    "wall_s": round(time.time() - t0)}
    p = OUT / "mb_crosspair_sensitivity.json"
    p.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"[M-B] 주판정 S_NN(policy)={s_nn:+.3f} → {verdict}\n저장: {p}")


if __name__ == "__main__":
    main()
