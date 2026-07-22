"""Capacity sweep 오프라인 probe — h-Jacobian effective rank (docs/PREREG_capacity_sweep.md).

phase1 ckpt 의 동결 디코더 h(ζ, z_t) 에 대해 val 지점들에서 ∂h/∂ζ Jacobian 을 계산,
Roy&Vetterli(2007) effective rank = exp(H(정규화 특이값)) 의 표본 평균을 산출한다.
(src/analysis/alignment_report.py effective_rank 와 동일 정의 — 여기선 행렬이 표본
집합이 아니라 per-point Jacobian 이므로 인라인 구현.)

해석(사전등록 §4): dec R² 가 flat 인데 eff-rank 만 폭에 단조 증가하면 "용량은 표현
여유를 늘리나 과제가 요구하지 않음" — capacity 무죄 서사의 보조 증거.

읽기 전용: ckpt/캐시를 소비만 하고 어떤 학습 산출물도 변경하지 않음.

사용 (clipx env, 캐시 존재 박스):
  python scripts/probe_h_jacobian.py --config configs/phase1_libero_large256_cap_g05.yaml
  python scripts/probe_h_jacobian.py --config ... --ckpt <override.pt> --n 256
출력: outputs/capsweep/hjac_<ckpt stem>.json (+stdout 1줄)
"""
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "src"))

import argparse
import json
import os

import numpy as np
import torch
import yaml

from core import chunkrep  # noqa: F401  (환경 정합 확인용 — phase1 파이프라인과 동일 임포트)
from core.anchor import get_anchor
from data import get_dataset
from models.networks import DeltaAE


def effective_rank(s, eps=1e-12):
    """특이값 벡터 s → exp(엔트로피(정규화 s)) (Roy&Vetterli 2007)."""
    s = np.asarray(s, dtype=np.float64)
    s = s[s > eps]
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="phase1 yaml (val split 재현용)")
    ap.add_argument("--ckpt", default=None, help="train.checkpoint 오버라이드")
    ap.add_argument("--n", type=int, default=256, help="Jacobian 표본 수 (val 앞쪽)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ckpt_path = os.path.expanduser(args.ckpt or cfg["train"]["checkpoint"])
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    p1 = ck["config"]
    m_cfg = p1["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 모델 재구성 (train_phase2/rollout_dataset 재구성 규약 미러, width 노브 포함) ----
    ae = DeltaAE(ck["action_dim"], ck["n_chunk"], m_cfg["latent_dim"],
                 m_cfg["hidden"], m_cfg["layers"], m_cfg["dropout"],
                 m_cfg.get("state_cond", True),
                 m_cfg.get("decoder_state_cond"),
                 m_cfg.get("encoder_state_cond"),
                 align_mode=m_cfg.get("align_mode", "dz"),
                 contrast_w=float(p1.get("loss", {}).get("contrast", 0.0)),
                 contrast_loss=m_cfg.get("contrast_loss", "infonce"),
                 contrast_head=m_cfg.get("contrast_head", False),
                 sigmoid_bias0=m_cfg.get("sigmoid_bias0", -5.5),
                 align_block=m_cfg.get("align_block"),
                 h_mode=m_cfg.get("h_mode", "mlp"),
                 h_flow_steps=m_cfg.get("h_flow_steps", 5),
                 hidden_g=m_cfg.get("hidden_g"),
                 hidden_h=m_cfg.get("hidden_h")).to(device).eval()
    ae.load_state_dict(ck["state_dict"])
    assert ae.h_mode == "mlp", \
        f"h-Jacobian probe 는 결정론 MLP h 전제 (h_mode={ae.h_mode} — flow h 는 표본별 확률적)"

    # ---- val split 재현 (train_phase1.main 과 동일 rng/규약) ----
    ds = get_dataset(cfg)
    files = ds.episode_files()
    rng = np.random.RandomState(cfg["train"]["seed"])
    perm = rng.permutation(len(files))
    v = cfg["data"]["val_episodes"]
    n_val = max(1, round(len(files) * v)) if v < 1 else int(v)
    val_ids = perm[:n_val]
    clip = get_anchor(cfg)
    eps = ds.build(clip, [files[i] for i in val_ids], verbose=False)
    Zt = np.concatenate([e[0] for e in eps])
    Ztn = np.concatenate([e[1] for e in eps])
    Dz = (Ztn - Zt).astype(np.float32)

    n = min(args.n, len(Dz))
    sub = np.random.RandomState(0).permutation(len(Dz))[:n]   # 결정적 표본
    Dz_t = torch.tensor(Dz[sub], device=device)
    Zt_t = torch.tensor(Zt[sub], device=device)

    # ---- per-point Jacobian ∂h/∂ζ @ (Δz, z_t), eff-rank 표본 평균 ----
    ranks, s_first = [], []
    for i in range(n):
        z_i = Zt_t[i:i + 1]
        J = torch.autograd.functional.jacobian(
            lambda zeta: ae.h(zeta, z_i).reshape(1, -1), Dz_t[i:i + 1],
            vectorize=True)                     # (1, out, 1, latent)
        J2 = J.reshape(J.shape[1], -1)          # (n_chunk*act_dim, latent)
        s = torch.linalg.svdvals(J2.float()).cpu().numpy()
        ranks.append(effective_rank(s))
        s_first.append(float(s[0]))

    out = {"ckpt": str(ckpt_path),
           "config": Path(args.config).name,
           "hidden": m_cfg["hidden"],
           "hidden_g": m_cfg.get("hidden_g"),
           "hidden_h": m_cfg.get("hidden_h"),
           "n_points": n,
           "h_jac_effrank_mean": float(np.mean(ranks)),
           "h_jac_effrank_std": float(np.std(ranks)),
           "h_jac_sigma1_mean": float(np.mean(s_first)),
           "ckpt_metrics": ck.get("metrics", {})}
    out_dir = WS / "outputs" / "capsweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"hjac_{Path(ckpt_path).stem}.json"
    out_path.write_text(json.dumps(out, indent=1))
    print(f"h-Jacobian eff-rank = {out['h_jac_effrank_mean']:.2f} ± "
          f"{out['h_jac_effrank_std']:.2f} (n={n}, σ1̄={out['h_jac_sigma1_mean']:.3f}) "
          f"→ {out_path}")


if __name__ == "__main__":
    main()
