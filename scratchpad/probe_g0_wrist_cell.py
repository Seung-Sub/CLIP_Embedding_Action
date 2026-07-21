"""WristCond-v1 학습-전 게이트 프로브 — G0-A (W-A) + G0 (W-B).

DESIGN_wrist_fusion_unified_v1 §5.1 사전등록 스펙:
  G0-A (W-A 선행, AMEND R2): ridge gripper-dim R²
      arm A = [p̄₁..₄(wrist DINOv3 pool2 4토큰 flatten 4096) ⊕ z_w,sig ⊕ z_main]
      arm B = [z_w,sig ⊕ z_main]                     → A_fut(정규화 청크)
      게이트: 설계 무명시 → 사전등록(2026-07-21, 결과 확인 전 고정):
              Δ(gripper-dim R²) ≥ +0.02 → W-A GO / 미달 → W-A 학습 전 폐기.
  G0 (W-B 선행): ridge per-dim R²
      arm W = [Δz̄_w,past ⊕ z̄_w]   (Δz̄_w = patch-mean(t) − patch-mean(t−span), span=16f=0.8s)
      arm M = [Δz̄_main,past ⊕ z_main]  (대조; Δz̄_main = z_main(t) − z_main(t−span))
      게이트(설계 명시): gripper-dim 우위 ≥ +0.05 → W-B GO / 미달 → W-B 학습 없이 폐기.
      + 국면분해(파지창 vs 이송) 동시 산출 (§5.1 비고, suite-위험 §4 방호).

방법론 관례: week0 프로브 동형 — split_ids(train seed=2, val 0.2, RandomState.permutation),
RidgeCV(alphas=[0.1..1e4]) + train 통계 표준화, per-dim R²(시간 pooled, gripper=dim 6).
부수효과: wrist dinov3-vitl16-256-pool2 dense 캐시 생성 (W-A 학습이 그대로 재사용).

실행(원격, GPU 1개 — 캐시 인코딩용, ridge는 CPU):
  cd ~/clip_ws && CUDA_VISIBLE_DEVICES=9 HF_HUB_OFFLINE=1 python3 scratchpad/probe_g0_wrist_cell.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

WS = Path(os.path.expanduser("~/clip_ws"))
sys.path.insert(0, str(WS / "src"))

from core import chunkrep                      # noqa: E402
from core.anchor import get_anchor             # noqa: E402
from data.libero import LiberoDataset          # noqa: E402

import yaml                                    # noqa: E402
from sklearn.linear_model import RidgeCV       # noqa: E402

torch.set_num_threads(int(os.environ.get("W0_THREADS", "12")))
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
CFG = WS / "configs" / "phase2_libero_large256_wristpatch.yaml"
OUT = WS / "outputs" / "wrist_cell"
OUT.mkdir(parents=True, exist_ok=True)
GRIP = 6                                       # LIBERO action dim 7, gripper = 마지막


def split_ids(n_files, seed, val_frac):
    """train_phase{1,2} 공통 split 재현 (w0_common 동형)."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_files)
    n_val = max(1, round(n_files * val_frac)) if val_frac < 1 else int(val_frac)
    return perm[:n_val], perm[n_val:]


def per_dim_r2(y, p, n_chunk, act_dim, mask=None):
    """action 차원별 R² (샘플×시간 pooled) — w0_p3_wrist 동형. mask=국면 서브셋."""
    y = y.reshape(-1, n_chunk, act_dim)
    p = p.reshape(-1, n_chunk, act_dim)
    if mask is not None:
        y, p = y[mask], p[mask]
    out = []
    for k in range(act_dim):
        yk, pk = y[:, :, k].ravel().astype(np.float64), p[:, :, k].ravel().astype(np.float64)
        dev = ((yk - yk.mean()) ** 2).sum()
        out.append(float(1 - ((yk - pk) ** 2).sum() / (dev + 1e-12)))
    return out


def ridge_arm(tag, Xtr, Xva, Ytr, Yva, n_chunk, act_dim, masks):
    mu, sd = Xtr.mean(0), np.maximum(Xtr.std(0), 1e-6)
    rid = RidgeCV(alphas=ALPHAS).fit((Xtr - mu) / sd, Ytr)
    pv = rid.predict((Xva - mu) / sd)
    per = per_dim_r2(Yva, pv, n_chunk, act_dim)
    res = {"dim": int(Xtr.shape[1]), "alpha": float(rid.alpha_),
           "r2_per_dim": per, "r2_gripper": per[GRIP],
           "r2_macro": float(np.mean(per))}
    for mname, m in masks.items():
        pm = per_dim_r2(Yva, pv, n_chunk, act_dim, mask=m)
        res[f"r2_gripper_{mname}"] = pm[GRIP]
        res[f"n_{mname}"] = int(m.sum())
    print(f"  [{tag}] dim={res['dim']} α={res['alpha']:g} "
          f"grip R²={res['r2_gripper']:+.4f} macro={res['r2_macro']:+.4f} "
          f"| grasp {res['r2_gripper_grasp']:+.4f} / transport {res['r2_gripper_transport']:+.4f}")
    return res


def main():
    t0 = time.time()
    cfg = yaml.safe_load(open(CFG))
    ck1 = torch.load(os.path.expanduser(cfg["phase1_ckpt"]), map_location="cpu",
                     weights_only=False)
    n_chunk, act_dim = ck1["n_chunk"], ck1["action_dim"]
    a_mean, a_std = ck1["a_mean"], ck1["a_std"]
    repr_kind = ck1.get("chunk_repr", "time")
    assert act_dim == 7, act_dim
    assert repr_kind == "time", f"per-dim gripper R²는 time repr 전제 ({repr_kind})"

    ds = LiberoDataset(cfg)
    files = ds.episode_files()
    val_ids, tr_ids = split_ids(len(files), cfg["train"]["seed"],
                                cfg["data"]["val_episodes"])
    print(f"episodes: train {len(tr_ids)} / val {len(val_ids)} (seed={cfg['train']['seed']})")

    clip_main = get_anchor(cfg)                                 # siglip2 large256 (pooled 캐시 재사용)
    ganc_cfg = cfg["module"]["grid_obs"]["anchor"]
    ganc = get_anchor({"anchor": ganc_cfg})                     # dinov3 pool2 → dense 캐시 생성/재사용
    gcam = cfg["module"]["grid_obs"]["camera"]
    print(f"anchors: main={clip_main.cache_key} / grid={ganc.cache_key} cam={gcam}")

    print("build_policy_samples (pool2 dense 캐시 인코딩 — 첫 실행은 GPU ~0.3-0.5h)...")
    eps = ds.build_policy_samples(clip_main, files, stride=cfg["data"].get("stride", 2),
                                  obs_anchors=[(ganc_cfg["name"], ganc, gcam)])
    # per-ep 튜플: (Zp, Zc, Zn, Ap, Af, Zw_sig, D_wrist(n,4,1024))
    span, stride = ds.span, cfg["data"].get("stride", 2)
    shift = span // stride                                      # 16/2 = 8 index (t−span)
    print(f"span={span}f stride={stride} → past shift {shift} idx, 인코딩 {time.time()-t0:.0f}s")

    def feats(ep):
        Zp, Zc, _Zn, _Ap, Af, Zw, D = ep
        n = len(Zc)
        pm = D.mean(1)                                          # patch-mean p̄ (n,1024)
        prev = np.maximum(np.arange(n) - shift, 0)              # max(t−span, 0) 설계 관례
        dpw = pm - pm[prev]                                     # Δz̄_w,past
        dzm = Zc - Zp                                           # Δz̄_main,past (Zp=z(max(t−span,0)))
        return dict(A=np.concatenate([D.reshape(n, -1), Zw, Zc], 1),   # G0-A arm A (6144)
                    B=np.concatenate([Zw, Zc], 1),                     # G0-A arm B (2048)
                    W=np.concatenate([dpw, pm], 1),                    # G0 arm W (2048)
                    M=np.concatenate([dzm, Zc], 1),                    # G0 arm M (2048)
                    Af=Af)

    def stack(ids):
        F = [feats(eps[i]) for i in ids]
        return {k: np.concatenate([f[k] for f in F]) for k in F[0]}

    tr, va = stack(tr_ids), stack(val_ids)

    def norm(A):
        a = ((A.reshape(len(A), n_chunk, act_dim) - a_mean) / a_std).astype(np.float32)
        return chunkrep.to_repr(a, repr_kind).reshape(len(A), -1)

    Ytr, Yva = norm(tr["Af"]), norm(va["Af"])
    # 국면분해: 파지창 = 미래 청크 내 gripper 커맨드 변화(raw ±1 스윙) / 이송 = 불변
    g_raw = va["Af"].reshape(-1, n_chunk, act_dim)[:, :, GRIP]
    grasp = (g_raw.max(1) - g_raw.min(1)) > 1.0
    masks = {"grasp": grasp, "transport": ~grasp}
    print(f"samples: train {len(Ytr)} / val {len(Yva)} | 파지창 {grasp.sum()} / 이송 {(~grasp).sum()}")

    res = {"probe": "wrist_cell_G0A_G0", "split_seed": int(cfg["train"]["seed"]),
           "n_tr": len(Ytr), "n_va": len(Yva), "span": int(span), "shift_idx": int(shift),
           "prereg": {"G0A_gate": "Δ(gripper R²) ≥ +0.02 (설계 무명시 → 2026-07-21 사전등록)",
                      "G0_gate": "gripper R²(W−M) ≥ +0.05 (설계 §5.1 명시)"}}

    print("[G0-A] W-A 정보-천장 프로브 (한계 gripper R²)")
    res["G0A_armA_patch_plus_base"] = ridge_arm("A: p̄×4⊕z_w,sig⊕z_main", tr["A"], va["A"],
                                                Ytr, Yva, n_chunk, act_dim, masks)
    res["G0A_armB_base_only"] = ridge_arm("B: z_w,sig⊕z_main", tr["B"], va["B"],
                                          Ytr, Yva, n_chunk, act_dim, masks)
    dA = res["G0A_armA_patch_plus_base"]["r2_gripper"] - res["G0A_armB_base_only"]["r2_gripper"]
    res["G0A_delta_gripper"] = dA
    res["G0A_verdict"] = "GO (>=+0.02)" if dA >= 0.02 else "KILL W-A (<+0.02)"
    print(f"[G0-A] Δgrip R² = {dA:+.4f} → {res['G0A_verdict']}")

    print("[G0] W-B 측정 과거변위 프로브")
    res["G0_armW_wrist_delta"] = ridge_arm("W: Δz̄_w,past⊕z̄_w", tr["W"], va["W"],
                                           Ytr, Yva, n_chunk, act_dim, masks)
    res["G0_armM_main_ctrl"] = ridge_arm("M: Δz̄_main,past⊕z_main", tr["M"], va["M"],
                                         Ytr, Yva, n_chunk, act_dim, masks)
    dW = res["G0_armW_wrist_delta"]["r2_gripper"] - res["G0_armM_main_ctrl"]["r2_gripper"]
    res["G0_delta_gripper"] = dW
    res["G0_verdict"] = "GO (>=+0.05)" if dW >= 0.05 else "KILL W-B (<+0.05)"
    print(f"[G0] grip R²(W−M) = {dW:+.4f} → {res['G0_verdict']}")

    res["_meta"] = {"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "host": os.uname().nodename, "wall_s": round(time.time() - t0)}
    p = OUT / "probe_g0_wrist_cell.json"
    p.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"저장: {p} ({res['_meta']['wall_s']}s)")


if __name__ == "__main__":
    main()
