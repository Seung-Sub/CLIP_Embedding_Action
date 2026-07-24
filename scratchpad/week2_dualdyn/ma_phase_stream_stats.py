"""M-A — 국면분해 스트림 통계: PI 전제("근접·파지 국면에서 wrist 가중↑") 정밀 검정.

그립 온셋-상대 bin(−16..+16f)별로
  (a) ‖Δz_wrist‖/‖Δz_main‖ (raw + per-stream 표준화 후)
  (b) per-stream ridge [Δz ⊕ z] → A_fut 의 gripper-dim / arm-dim R²
를 산출. ridge는 train split에서 fit, val 에서 bin별 평가 (probe_g0 관례 동형).

  cd ~/clip_ws && WK2_THREADS=12 python3 scratchpad/week2_dualdyn/ma_phase_stream_stats.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wk2_common import (OUT, GRIP, BIN_NAMES, load_dual_std, build_split,   # noqa: E402
                        per_dim_r2)

from sklearn.linear_model import RidgeCV                                    # noqa: E402

ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]


def main():
    t0 = time.time()
    ck1, ae, ck2, cfg2, model, use_lang = load_dual_std()
    n_chunk, act_dim = ck1["n_chunk"], ck1["action_dim"]
    s_m = float(ae.dz_std_main.item())
    s_w = float(ae.dz_std_wrist.item())
    print(f"[M-A] dz_std main={s_m:.4f} wrist={s_w:.4f} (ratio {s_m/s_w:.2f}x)")

    tr, mtr, _ = build_split(cfg2, ck1, "train")
    va, mva, _ = build_split(cfg2, ck1, "val")
    print(f"[M-A] samples train {len(tr['Cf'])} / val {len(va['Cf'])} "
          f"({time.time()-t0:.0f}s)")
    no_on_tr = int((~mtr["has_grasp"]).sum())
    no_on_va = int((~mva["has_grasp"]).sum())

    def dz(d):
        return d["Zn"] - d["Zc"], d["Zwn"] - d["Zwc"]

    dzm_v, dzw_v = dz(va)
    nm = np.linalg.norm(dzm_v, axis=1)
    nw = np.linalg.norm(dzw_v, axis=1)
    ratio_raw = nw / np.maximum(nm, 1e-9)
    ratio_std = (nw / s_w) / np.maximum(nm / s_m, 1e-9)

    # ---- (b) per-stream ridge (train fit) ----
    dzm_t, dzw_t = dz(tr)
    arms = {
        "wrist": (np.concatenate([dzw_t, tr["Zwc"]], 1),
                  np.concatenate([dzw_v, va["Zwc"]], 1)),
        "main": (np.concatenate([dzm_t, tr["Zc"]], 1),
                 np.concatenate([dzm_v, va["Zc"]], 1)),
    }
    Ytr, Yva = tr["Cf"], va["Cf"]
    preds = {}
    for tag, (Xtr, Xva) in arms.items():
        mu, sd = Xtr.mean(0), np.maximum(Xtr.std(0), 1e-6)
        rid = RidgeCV(alphas=ALPHAS).fit((Xtr - mu) / sd, Ytr)
        preds[tag] = rid.predict((Xva - mu) / sd)
        g = per_dim_r2(Yva, preds[tag], n_chunk, act_dim)
        print(f"  [{tag}] α={rid.alpha_:g} global grip R²={g[GRIP]:+.4f} "
              f"arm-mean={np.mean(g[:GRIP]):+.4f} ({time.time()-t0:.0f}s)")

    # ---- bin별 산출 ----
    bins = {}
    for b, name in enumerate(BIN_NAMES):
        m = mva["bin"] == b
        n = int(m.sum())
        row = {"n": n}
        if n >= 30:
            row["ratio_raw_median"] = float(np.median(ratio_raw[m]))
            row["ratio_raw_iqr"] = [float(np.percentile(ratio_raw[m], 25)),
                                    float(np.percentile(ratio_raw[m], 75))]
            row["ratio_std_median"] = float(np.median(ratio_std[m]))
            for tag in arms:
                per = per_dim_r2(Yva, preds[tag], n_chunk, act_dim, mask=m)
                row[f"{tag}_grip_r2"] = per[GRIP]
                row[f"{tag}_arm_r2"] = float(np.mean(per[:GRIP]))
            row["adv_wrist_grip"] = row["wrist_grip_r2"] - row["main_grip_r2"]
            row["adv_wrist_arm"] = row["wrist_arm_r2"] - row["main_arm_r2"]
        bins[name] = row
        if n >= 30:
            print(f"  bin {name:>10} n={n:5d} ratio_std={row['ratio_std_median']:.3f} "
                  f"grip W/M {row['wrist_grip_r2']:+.3f}/{row['main_grip_r2']:+.3f} "
                  f"adv {row['adv_wrist_grip']:+.3f}")

    # ---- 집계: near-grasp vs far (사전등록 판정) ----
    aggs = {}
    near = mva["has_grasp"] & (mva["offset"] >= -16) & (mva["offset"] < 16)
    far = mva["has_grasp"] & ~near
    for tag_m, m in (("near_grasp", near), ("far", far)):
        row = {"n": int(m.sum()),
               "ratio_std_median": float(np.median(ratio_std[m]))}
        for tag in arms:
            per = per_dim_r2(Yva, preds[tag], n_chunk, act_dim, mask=m)
            row[f"{tag}_grip_r2"] = per[GRIP]
            row[f"{tag}_arm_r2"] = float(np.mean(per[:GRIP]))
        row["adv_wrist_grip"] = row["wrist_grip_r2"] - row["main_grip_r2"]
        aggs[tag_m] = row
    adv_n, adv_f = aggs["near_grasp"]["adv_wrist_grip"], aggs["far"]["adv_wrist_grip"]
    verdict_true = (adv_n >= 0.05) and (adv_n >= 2 * adv_f)
    verdict = ("PREMISE-REFINED-TRUE" if verdict_true else
               "PREMISE-NOT-CONFIRMED")

    res = {
        "probe": "ma_phase_stream_stats",
        "ckpt": {"p1": "phase1_libero_dualstream_wrist_std.pt",
                 "dz_std_main": s_m, "dz_std_wrist": s_w,
                 "scale_ratio": s_m / s_w},
        "n_train": len(Ytr), "n_val": len(Yva),
        "eps_without_grasp": {"train": no_on_tr, "val": no_on_va,
                              "note": "gripper close 커맨드 미발생 샘플 수(bin=no_grasp)"},
        "global": {
            "ratio_raw_median": float(np.median(ratio_raw)),
            "ratio_std_median": float(np.median(ratio_std)),
            "wrist_grip_r2": per_dim_r2(Yva, preds["wrist"], n_chunk, act_dim)[GRIP],
            "main_grip_r2": per_dim_r2(Yva, preds["main"], n_chunk, act_dim)[GRIP],
            "wrist_arm_r2": float(np.mean(per_dim_r2(Yva, preds["wrist"],
                                                     n_chunk, act_dim)[:GRIP])),
            "main_arm_r2": float(np.mean(per_dim_r2(Yva, preds["main"],
                                                    n_chunk, act_dim)[:GRIP])),
        },
        "bins": bins, "aggregates": aggs,
        "prereg_rule": "정제-참 = near-grasp (W−M) grip R² ≥ +0.05 AND ≥ 2× far 우위",
        "verdict": verdict,
        "_meta": {"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "host": os.uname().nodename, "device": "cpu",
                  "wall_s": round(time.time() - t0)},
    }
    p = OUT / "ma_phase_stream_stats.json"
    p.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"[M-A] near adv {adv_n:+.4f} / far adv {adv_f:+.4f} → {verdict}")
    print(f"저장: {p} ({res['_meta']['wall_s']}s)")


if __name__ == "__main__":
    main()
