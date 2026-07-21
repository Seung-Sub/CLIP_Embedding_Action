"""G2 zero-ablation (DESIGN_wrist_fusion_unified_v1 §5.1 G2) — W-A wristpatch ckpt.

val split에서 조건 토큰군 zero-ablation으로 act R²/act L1 변화를 측정:
  (c) intact          : 전 토큰 정상
  (a) grid_zero       : wristpatch×4 (GridObs 출력 토큰) 0화
  (b) sig_zero        : wrist_sig (SigLIP2 손목 조건 토큰) 0화
  (d) both_zero       : (a)+(b) 동시 — 보너스 진단

판정 (사전등록 F2): grid_zero의 R² 하락 < 0.005 → 토큰 미사용(F2 플래그).

경로/스플릿/생성기 시드는 train_phase2.py 평가 블록과 동형 재현
(seed=cfg train.seed, val_episodes 비율 split, torch.Generator(0)) —
intact R²가 ckpt metrics.action_r2(0.6689)를 재현해야 자기검증 통과.

사용: CUDA_VISIBLE_DEVICES=8 python3 scratchpad/g2_zero_ablation_wristpatch.py
"""
import json
import os
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "src"))

import numpy as np
import torch

from core import chunkrep
from core.anchor import get_anchor
from data import get_dataset
from models.networks import DeltaAE
from models.obs_fusion import GridObs
from models.policy import FlowPolicy, build_policy_from_cfg

CKPT = WS / "checkpoints/phase2_libero_large256_wristpatch.pt"
OUT = WS / "outputs/analysis/g2_zero_ablation_wristpatch.json"


def r2(y, yhat):
    dev = ((y - y.mean(0)) ** 2).sum()
    return float(1 - ((y - yhat) ** 2).sum() / (dev + 1e-12))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck2 = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = ck2["config"]
    t_cfg, m_cfg = cfg["train"], cfg["module"]
    assert m_cfg.get("grid_obs") and m_cfg.get("wrist_token") and m_cfg.get("lang_token")
    print(f"[G2] ckpt={CKPT.name} | trained action_r2={ck2['metrics']['action_r2']:.4f} "
          f"| n_val={ck2['metrics']['n_val']}")

    # ---- phase1 동결 모델 (train_phase2.py:327-353 동형) ----
    ck = torch.load(os.path.expanduser(cfg["phase1_ckpt"]),
                    map_location="cpu", weights_only=False)
    p1 = ck["config"]
    n_chunk, act_dim = ck["n_chunk"], ck["action_dim"]
    a_mean, a_std = ck["a_mean"], ck["a_std"]
    repr_kind = ck.get("chunk_repr", "time")
    ae = DeltaAE(act_dim, n_chunk, p1["model"]["latent_dim"],
                 p1["model"]["hidden"], p1["model"]["layers"],
                 p1["model"]["dropout"],
                 p1["model"].get("state_cond", True),
                 p1["model"].get("decoder_state_cond"),
                 p1["model"].get("encoder_state_cond"),
                 align_mode=p1["model"].get("align_mode", "dz"),
                 contrast_w=float(p1.get("loss", {}).get("contrast", 0.0)),
                 contrast_loss=p1["model"].get("contrast_loss", "infonce"),
                 contrast_head=p1["model"].get("contrast_head", False),
                 sigmoid_bias0=p1["model"].get("sigmoid_bias0", -5.5),
                 align_block=p1["model"].get("align_block"),
                 h_mode=p1["model"].get("h_mode", "mlp"),
                 h_flow_steps=p1["model"].get("h_flow_steps", 5)).to(device)
    ae.load_state_dict(ck["state_dict"])
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)

    # ---- val split 재현 (train_phase2.py:323, 356-367 동형; smoke/max-episodes 없음) ----
    rng = np.random.RandomState(t_cfg["seed"])
    torch.manual_seed(t_cfg["seed"])            # (파라미터 init용이었음 — 관례 유지, 소비만)
    ds = get_dataset(cfg)
    files = ds.episode_files()
    perm = rng.permutation(len(files))
    v = cfg["data"]["val_episodes"]
    n_val = max(1, round(len(files) * v)) if v < 1 else int(v)
    val_ids = perm[:n_val]
    print(f"[G2] episodes total {len(files)} / val {n_val}")

    clip = get_anchor(cfg)
    grid_cfg = m_cfg["grid_obs"]
    genc = grid_cfg["anchor"]
    ganc = get_anchor({"anchor": genc})
    grid_anchor = (genc["name"], ganc,
                   grid_cfg.get("camera", genc.get("camera", "agentview_rgb")))

    # val 에피소드만 빌드 (전체 빌드 후 stack(val_ids)와 배열 순서 동일 — val_ids 순서 유지)
    files_val = [files[i] for i in val_ids]
    eps = ds.build_policy_samples(clip, files_val, stride=cfg["data"].get("stride", 2),
                                  obs_anchors=[grid_anchor])
    stack = lambda k: np.concatenate([e[k] for e in eps])
    Zp_va, Zc_va, Zn_va, Ap_va, Af_va = (stack(k) for k in range(5))
    W_va = stack(5)                              # wrist_sig (단일 스트림 6번째)
    D_va = stack(6)                              # grid dense (n, 4, 1024)
    L_va = np.concatenate([
        np.repeat(ds.instruction_embedding(clip, fe)[None], len(e[0]), axis=0)
        for fe, e in zip(files_val, eps)]).astype(np.float32)
    assert len(Zp_va) == ck2["metrics"]["n_val"], \
        (len(Zp_va), ck2["metrics"]["n_val"])    # split 재현 자기검증

    def norm(A):
        a = ((A.reshape(len(A), n_chunk, act_dim) - a_mean) / a_std).astype(np.float32)
        return chunkrep.to_repr(a, repr_kind)
    Cp_va, Cf_va = norm(Ap_va), norm(Af_va)

    with torch.no_grad():
        def embed_past(Cp, Zp):
            out = []
            for i in range(0, len(Cp), 4096):
                out.append(ae.g(torch.tensor(Cp[i:i+4096], device=device),
                                torch.tensor(Zp[i:i+4096], device=device)).cpu())
            return torch.cat(out)
        Ae_va = embed_past(Cp_va, Zp_va).to(device)

    # ---- 모듈 재구성 + ckpt 로드 (train_phase2.py:531-559 동형) ----
    gp_dim = D_va.shape[-1]
    grid_obs = GridObs(patch_dim=gp_dim, out_dim=p1["model"]["latent_dim"],
                       n_tokens=grid_cfg.get("n_tokens", 16),
                       pool=grid_cfg.get("pool", "avg"),
                       d_attn=grid_cfg.get("d_attn", 768),
                       heads=grid_cfg.get("heads", 8),
                       ln=grid_cfg.get("ln", False),
                       tok_drop=grid_cfg.get("tok_drop", 0.0),
                       group_drop=grid_cfg.get("group_drop", 0.0),
                       init_std=grid_cfg.get("init_std")).to(device)
    grid_obs.load_state_dict(ck2["grid_obs"])
    grid_obs.eval()
    Kg = grid_obs.n_tokens
    n_tokens = 3 + 1 + 1 + Kg                     # base3 + lang + wrist_sig + grid4 = 9
    model = build_policy_from_cfg(m_cfg, n_tokens=n_tokens,
                                  latent_dim=p1["model"]["latent_dim"],
                                  action_flat_dim=n_chunk * act_dim).to(device)
    model.load_state_dict(ck2["state_dict"])
    model.eval()
    assert isinstance(model, FlowPolicy)

    val_t = {k: torch.tensor(x, device=device) for k, x in
             dict(zp=Zp_va, zc=Zc_va, cf=Cf_va, lang=L_va, wr=W_va, dobs=D_va).items()}

    def evaluate(zero_grid=False, zero_sig=False):
        """train_phase2.py:782-845 평가 블록 동형 + 토큰군 0화 옵션."""
        with torch.no_grad():
            wr = torch.zeros_like(val_t["wr"]) if zero_sig else val_t["wr"]
            toks = [val_t["zp"], val_t["zc"], Ae_va, val_t["lang"], wr]
            grid_tok = grid_obs(val_t["dobs"])                     # (B,Kg,1024) eval=drop off
            if zero_grid:
                grid_tok = torch.zeros_like(grid_tok)
            toks = toks + [grid_tok[:, k] for k in range(Kg)]
            gen = torch.Generator(device=device)
            gen.manual_seed(0)                                     # 조건 간 동일 노이즈
            zeta = model(torch.stack(toks, dim=1), generator=gen)
            ahat_t = ae.h(zeta, val_t["zc"])
            l_act = torch.nn.functional.l1_loss(ahat_t, val_t["cf"]).item()
            ahat = ahat_t.cpu().numpy()
        act_r2 = r2(Cf_va.reshape(len(Cf_va), -1), ahat.reshape(len(ahat), -1))
        gt = chunkrep.from_repr(Cf_va, repr_kind) * a_std + a_mean
        pr = chunkrep.from_repr(ahat, repr_kind) * a_std + a_mean
        grip_acc = float(((pr[:, :, -1] > 0.0) == (gt[:, :, -1] > 0.0)).mean() * 100)
        # gripper-dim 전용 R² (파지 채널 진단 — 설계가 wrist 가치의 소재로 지목한 채널)
        g_r2 = r2(Cf_va[:, :, -1], ahat[:, :, -1])
        return dict(act_r2=act_r2, act_l1=l_act, grip_acc=grip_acc, grip_r2=g_r2)

    res = {"intact": evaluate(),
           "grid_zero": evaluate(zero_grid=True),
           "sig_zero": evaluate(zero_sig=True),
           "both_zero": evaluate(zero_grid=True, zero_sig=True)}
    for k, m in res.items():
        print(f"[G2] {k:10s} | act R² {m['act_r2']:+.4f} | act L1 {m['act_l1']:.4f} "
              f"| grip acc {m['grip_acc']:.1f}% | grip R² {m['grip_r2']:+.4f}")
    d_grid = res["intact"]["act_r2"] - res["grid_zero"]["act_r2"]
    d_sig = res["intact"]["act_r2"] - res["sig_zero"]["act_r2"]
    f2 = bool(d_grid < 0.005)
    print(f"[G2] ΔR²(grid_zero) = {d_grid:+.4f} | ΔR²(sig_zero) = {d_sig:+.4f}")
    print(f"[G2] F2(grid 토큰 미사용, ΔR²<0.005) = {'FLAG' if f2 else 'no'}")
    sanity = abs(res["intact"]["act_r2"] - ck2["metrics"]["action_r2"])
    print(f"[G2] intact 재현오차 vs ckpt metrics = {sanity:.5f} "
          f"({'OK' if sanity < 0.005 else 'MISMATCH — split/경로 재현 실패 의심'})")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"ckpt": str(CKPT), "n_val": int(len(Cf_va)),
         "trained_action_r2": ck2["metrics"]["action_r2"],
         "results": res,
         "delta_r2_grid_zero": d_grid, "delta_r2_sig_zero": d_sig,
         "F2_flag": f2, "sanity_abs_err": sanity}, indent=2))
    print(f"[G2] 저장: {OUT}")


if __name__ == "__main__":
    main()
