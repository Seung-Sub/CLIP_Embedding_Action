"""LIBERO GT 데모 전체 시계열 추론 평가 — 7D 액션 그래프.

  t = 16, 24, ... 마다 (z_{t−16}, z_t, g(A_past)) → f → h → Â_{t:t+16}
  앞 8스텝씩 이어붙여 전체 예측 궤적 구성 (GT 이미지 사용 = 개루프)
  7D = [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper] — 3그룹으로 플롯

사용 (clip_libero env):
  python src/eval_libero/rollout_dataset.py                 # val 첫 데모
  python src/eval_libero/rollout_dataset.py --episode 3     # val 3번째 데모
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import os

import matplotlib
import numpy as np
import torch
import yaml
from matplotlib import font_manager

for _f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    try:
        font_manager.fontManager.addfont(_f)
    except FileNotFoundError:
        pass
matplotlib.rcParams.update({"font.family": ["Noto Sans CJK JP", "sans-serif"],
                            "axes.unicode_minus": False})
import matplotlib.pyplot as plt

from core import chunkrep
from core.anchor import get_anchor
from data.libero import LiberoDataset
from models.networks import DeltaAE
from models.policy import build_policy_from_cfg

WS = Path(__file__).resolve().parents[2]
DIM_NAMES = ["Δx", "Δy", "Δz", "Δroll", "Δpitch", "Δyaw", "gripper"]


def load_models(cfg, device):
    ck1 = torch.load(os.path.expanduser(cfg["phase1_ckpt"]),
                     map_location="cpu", weights_only=False)
    p1 = ck1["config"]
    ae = DeltaAE(ck1["action_dim"], ck1["n_chunk"], p1["model"]["latent_dim"],
                 p1["model"]["hidden"], p1["model"]["layers"],
                 p1["model"]["dropout"],
                 p1["model"].get("state_cond", True),
                 p1["model"].get("decoder_state_cond"),
                 p1["model"].get("encoder_state_cond")).to(device).eval()
    ae.load_state_dict(ck1["state_dict"])
    ck2 = torch.load(os.path.expanduser(cfg["train"]["checkpoint"]),
                     map_location="cpu", weights_only=False)
    m = ck2["config"]["module"]
    use_lang = m.get("lang_token", False)
    use_wrist = m.get("wrist_token", False)

    # ---- F3 관측 융합 (module.obs 있을 때만; 없으면 no-obs 기존 경로와 완전 동일) ----
    # phase2 학습(train_phase2.py)의 앵커·ObsFusion 구성을 그대로 미러링한다.
    obs_anchors, obs_fusion, K = None, None, 0
    if m.get("obs"):
        from core.anchor import get_anchor
        from models.obs_fusion import ObsFusion
        obs_cfg = m["obs"]
        obs_anchors, enc_dims = [], {}
        for enc in obs_cfg["encoders"]:
            anc = get_anchor({"anchor": enc})
            if enc["name"] == "siglip2":
                anc.save_tokens = True                # siglip2: 패치 토큰 반환 활성화
            obs_anchors.append((enc["name"], anc,
                                enc.get("camera", "agentview_rgb")))
            enc_dims[enc["name"]] = anc.patch_dim     # dinov2=1024 / siglip2=1152
        obs_fusion = ObsFusion(enc_dims, d_attn=obs_cfg.get("d_attn", 768),
                               n_query=obs_cfg.get("n_query", 8),
                               out_dim=p1["model"]["latent_dim"],
                               pool=obs_cfg.get("pool", "attn"),
                               unshuffle=obs_cfg.get("unshuffle", 1))
        obs_fusion.load_state_dict(ck2["obs_fusion"])
        obs_fusion = obs_fusion.to(device).eval()
        K = obs_fusion.n_query                        # attn: n_query / mean: 1

    policy = build_policy_from_cfg(
        m, n_tokens=3 + int(use_lang) + int(use_wrist) + K,
        latent_dim=p1["model"]["latent_dim"]).to(device).eval()
    policy.load_state_dict(ck2["state_dict"])
    wrist_cam = ck2["config"]["data"].get("wrist_camera") if use_wrist else None
    # obs_anchors/obs_fusion 는 obs일 때만 non-None; no-obs 호출부는 무시(*_)한다.
    return (ae, policy, ck1["a_mean"], ck1["a_std"], ck1["n_chunk"],
            ck1["action_dim"], use_lang, ck1.get("chunk_repr", "time"),
            wrist_cam, obs_anchors, obs_fusion)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(WS / "configs" / "phase2_libero.yaml"))
    ap.add_argument("--episode", type=int, default=0, help="val 분할 내 인덱스")
    ap.add_argument("--exec-horizon", type=int, default=8)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    (ae, policy, a_mean, a_std, n_chunk, act_dim, use_lang,
     repr_kind, wrist_cam, obs_anchors, obs_fusion) = load_models(cfg, device)
    ds = LiberoDataset(cfg)
    clip = get_anchor(cfg)          # 앵커 config 반영 (무-anchor면 ClipAnchor=ClipWrapper와 동일)

    eps = ds.episode_files()
    rng = np.random.RandomState(cfg["train"]["seed"])
    perm = rng.permutation(len(eps))
    v = cfg["data"]["val_episodes"]
    n_val = max(1, round(len(eps) * v)) if v < 1 else int(v)
    ep = eps[perm[args.episode % n_val]]
    print(f"에피소드: {ds._key(ep)}")
    print(f"지시문: {ds.instruction(ep)}")

    acts = ds.load_actions(ep)
    Z = ds.embeddings(clip, ep)
    Zw = ds.embeddings(clip, ep, wrist_cam) if wrist_cam else None
    # F3: 앵커별 dense patch 토큰 D[t] (Z[t]=z_cur 와 동일 인덱스 정렬)
    D_obs = ([(name, ds.dense_embeddings(anc, ep, cam))
              for name, anc, cam in obs_anchors] if obs_fusion is not None else [])
    lang = torch.tensor(ds.instruction_embedding(clip, ep)[None],
                        device=device) if use_lang else None
    T = len(acts)
    span, H = ds.span, args.exec_horizon

    def norm(a):
        return ((a - a_mean) / a_std).astype(np.float32)

    pred = np.full_like(acts, np.nan)
    t = span
    with torch.no_grad():
        while t + span <= T:
            z_prev = torch.tensor(Z[t - span][None], device=device)
            z_cur = torch.tensor(Z[t][None], device=device)
            past = chunkrep.to_repr(
                norm(ds.resample_chunk(acts[t - span:t])), repr_kind)[None]
            a_emb = ae.g(torch.tensor(past, device=device), z_prev)
            toks = [z_prev, z_cur, a_emb] + ([lang] if use_lang else []) \
                + ([torch.tensor(Zw[t][None], device=device)]
                   if wrist_cam else [])
            if obs_fusion is not None:                # F3: 관측 토큰 K개를 열 끝에 추가
                obs_tok = obs_fusion({name: torch.tensor(D[t][None], device=device)
                                      for name, D in D_obs})       # (1,K,768)
                toks = toks + [obs_tok[:, k] for k in range(obs_tok.size(1))]
            zeta = policy(torch.stack(toks, dim=1))
            ahat = chunkrep.from_repr(ae.h(zeta, z_cur).cpu().numpy()[0],
                                      repr_kind) * a_std + a_mean
            n_exec = min(H, T - t)
            pred[t:t + n_exec] = ahat[:n_exec]
            t += H

    valid = ~np.isnan(pred[:, 0])
    mae = np.abs(pred[valid] - acts[valid]).mean(axis=0)
    grip_acc = ((pred[valid][:, 6] > 0) == (acts[valid][:, 6] > 0)).mean() * 100
    print(f"MAE(정규화 [-1,1] 단위): pos {mae[:3].mean():.3f} | "
          f"rot {mae[3:6].mean():.3f} | 그리퍼 정확도 {grip_acc:.1f}% "
          f"| 추론 구간 {valid.sum()}/{T}")

    fig, axes = plt.subplots(7, 1, figsize=(12, 14), dpi=110, sharex=True)
    tt = np.arange(T) / 20.0
    for d in range(act_dim):
        ax = axes[d]
        ax.plot(tt, acts[:, d], color="#4477AA", lw=1.6, label="GT")
        ax.plot(tt, pred[:, d], color="#EE6677", lw=1.2, ls="--", label="예측")
        ax.set_title(DIM_NAMES[d], fontsize=9)
        ax.grid(color="#E5E7EB", lw=0.5)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f'{ds._key(ep)}\n"{ds.instruction(ep)}"\n'
                 f'pos MAE {mae[:3].mean():.3f} | rot {mae[3:6].mean():.3f} | '
                 f'그리퍼 {grip_acc:.1f}%', fontsize=11)
    fig.tight_layout()
    out = WS / "outputs" / "eval" / f"rollout_dataset_libero_{ds._key(ep)}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
