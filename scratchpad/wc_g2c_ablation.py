# W-C G2-C 사전등록 오프라인 게이트: dual ζ_wrist zero-ablation ΔR²(A−B) ≥ 0.27.
#
# week0 w0_p3_wrist.py Part A 패턴의 W-C(std) 적응판 — 차이 3가지만:
#   (1) ckpt = phase{1,2}_libero_dualstream_wrist_std.pt (N3 stream_standardize,
#       DualDeltaAE 를 p1 config 플래그로 재구성 → dz_std buffer 키 복원).
#   (2) 조건 토큰열 = wc_sig: [zp_m, zc_m, a_emb, zw_sig] (+lang), n_tokens=4+lang
#       — zw_sig 는 build_policy_samples(wrist_cond_anchor=main_anchor) 9번째 배열.
#   (3) ζ 통화 = per-stream 표준화 공간: decode 는 표준화 ζ̂ 소비(역변환 불요),
#       oracle ζ_gt = cat(ae.std_dz(Δz_main, Δz_wrist)).
# 게이트 (DESIGN_wrist_fusion_unified_v1 §5.3 사전등록): ΔR²(A−B) ≥ 0.27 GO.
#
#   cd /workspace/CLIP_ws && OMP_NUM_THREADS=8 python3 scratchpad/wc_g2c_ablation.py
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

WS = Path(os.path.expanduser("~/clip_ws"))
sys.path.insert(0, str(WS / "src"))
sys.path.insert(0, str(WS / "scratchpad" / "week0"))

from w0_common import DummyAnchor, r2_pooled, split_ids  # noqa: E402
from core import chunkrep                                # noqa: E402
from data.libero import LiberoDataset                    # noqa: E402
from models.networks import DualDeltaAE                  # noqa: E402
from models.policy import FlowPolicy, build_policy_from_cfg  # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
P1 = WS / "checkpoints" / "phase1_libero_dualstream_wrist_std.pt"
P2 = WS / "checkpoints" / "phase2_libero_dualstream_wrist_std.pt"
OUT = WS / "outputs" / "wrist_cell" / "wc_g2c_ablation.json"
GATE = 0.27


def per_dim_r2(Cf, ahat, n_chunk, act_dim):
    y = Cf.reshape(-1, n_chunk, act_dim)
    p = ahat.reshape(-1, n_chunk, act_dim)
    out = []
    for k in range(act_dim):
        yk, pk = y[:, :, k].ravel(), p[:, :, k].ravel()
        dev = ((yk - yk.mean()) ** 2).sum()
        out.append(float(1 - ((yk - pk) ** 2).sum() / (dev + 1e-12)))
    return out


def main():
    # ---- phase1 dual(std) 재구성 (train_phase2.run_dual 동형) ----
    ck1 = torch.load(str(P1), map_location="cpu", weights_only=False)
    assert ck1.get("dual_stream")
    p1 = ck1["config"]
    dm, dw = ck1["dim_main"], ck1["dim_wrist"]
    dc = dm + dw
    n_chunk, act_dim = ck1["n_chunk"], ck1["action_dim"]
    a_mean, a_std = ck1["a_mean"], ck1["a_std"]
    repr_kind = ck1.get("chunk_repr", "time")
    ae = DualDeltaAE(act_dim, n_chunk, dm, dw, p1["model"]["hidden"],
                     p1["model"]["layers"], p1["model"]["dropout"],
                     p1["model"].get("state_cond", True),
                     stream_standardize=p1["model"].get("stream_standardize", False))
    ae.load_state_dict(ck1["state_dict"])
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    assert ae.stream_standardize, "W-C 게이트: N3 std ckpt 전제"
    print(f"[G2-C] N3 ON: dz_std_main={ae.dz_std_main.item():.4f} / "
          f"dz_std_wrist={ae.dz_std_wrist.item():.4f}")

    ck2 = torch.load(str(P2), map_location="cpu", weights_only=False)
    cfg2 = ck2["config"]
    m_cfg = cfg2["module"]
    assert m_cfg.get("wrist_cond_sig"), "W-C 게이트: wc_sig 토큰열 전제"
    use_lang = bool(m_cfg.get("lang_token", False))

    # ---- phase2 val split 재현 ----
    ds = LiberoDataset(cfg2)
    files = ds.episode_files()
    val_ids, _ = split_ids(len(files), cfg2["train"]["seed"],
                           cfg2["data"]["val_episodes"])
    val_files = [files[i] for i in val_ids]
    main_a = DummyAnchor(ck1["anchor"]["cache_key"], dm)
    wrist_a = DummyAnchor(ck1["anchor_wrist"]["cache_key"], dw)
    print(f"[G2-C] val eps {len(val_files)} (phase2 seed={cfg2['train']['seed']}) 로드...")
    eps = ds.build_policy_samples(main_a, val_files,
                                  stride=cfg2["data"].get("stride", 2),
                                  wrist_anchor=wrist_a,
                                  wrist_cond_anchor=main_a)   # wc_sig: 9번째 = zw_sig
    arr = tuple(np.concatenate([e[k] for e in eps]) for k in range(len(eps[0])))
    Zp, Zc, Zn, Ap, Af, Zwp, Zwc, Zwn, Ws = arr
    L = None
    if use_lang:
        lang_per_ep = [ds.instruction_embedding(main_a, f) for f in val_files]
        L = np.concatenate([np.repeat(lang_per_ep[i][None], len(eps[i][0]), 0)
                            for i in range(len(val_files))]).astype(np.float32)

    def norm(A):
        a = ((A.reshape(len(A), n_chunk, act_dim) - a_mean) / a_std
             ).astype(np.float32)
        return chunkrep.to_repr(a, repr_kind)

    Cp, Cf = norm(Ap), norm(Af)
    n = len(Cf)
    print(f"[G2-C] val samples {n} (ckpt 학습 시 {ck2['metrics']['n_val']})")

    model = build_policy_from_cfg(m_cfg, n_tokens=4 + int(use_lang), latent_dim=dc)
    assert isinstance(model, FlowPolicy) and model.flow_dim == dc
    model.load_state_dict(ck2["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    def _pad(t):
        return t if t.shape[-1] == dc else \
            torch.nn.functional.pad(t, (0, dc - t.shape[-1]))

    zeta_np = np.empty((n, dc), np.float32)
    zeta_gt = np.empty((n, dc), np.float32)
    bs = 1024
    with torch.no_grad():
        for i in range(0, n, bs):
            j = slice(i, min(i + bs, n))
            t = {k: torch.from_numpy(v[j]) for k, v in
                 dict(Zp=Zp, Zc=Zc, Zn=Zn, Cp=Cp, Zwp=Zwp, Zwc=Zwc, Zwn=Zwn,
                      Ws=Ws).items()}
            a_emb = ae.encode(t["Cp"], t["Zp"], t["Zwp"])
            toks = [_pad(t["Zp"]), _pad(t["Zc"]), a_emb, _pad(t["Ws"])]
            if use_lang:
                toks.append(_pad(torch.from_numpy(L[j])))
            gen = torch.Generator(); gen.manual_seed(0)
            zeta_np[j] = model(torch.stack(toks, 1), generator=gen).numpy()
            # oracle ζ_gt = 표준화 공간 (N3: ζ 통화 = std_dz)
            zeta_gt[j] = torch.cat(ae.std_dz(t["Zn"] - t["Zc"],
                                             t["Zwn"] - t["Zwc"]), 1).numpy()

    def decode_r2(zeta):
        outs = []
        with torch.no_grad():
            for i in range(0, n, bs):
                j = slice(i, min(i + bs, n))
                a = ae.decode(torch.from_numpy(zeta[j]),
                              torch.from_numpy(Zc[j]), torch.from_numpy(Zwc[j]))
                outs.append(a.numpy())
        ahat = np.concatenate(outs).reshape(n, -1)
        return {"r2": r2_pooled(Cf.reshape(n, -1), ahat),
                "per_dim_r2": per_dim_r2(Cf, ahat, n_chunk, act_dim)}

    conds = {}
    conds["A_full"] = decode_r2(zeta_np)
    zb = zeta_np.copy(); zb[:, dm:] = 0
    conds["B_zeta_wrist_zero"] = decode_r2(zb)
    zc_ = zeta_np.copy(); zc_[:, :dm] = 0
    conds["C_zeta_main_zero"] = decode_r2(zc_)
    conds["Ap_oracle_full"] = decode_r2(zeta_gt)
    zbo = zeta_gt.copy(); zbo[:, dm:] = 0
    conds["Bp_oracle_wrist_zero"] = decode_r2(zbo)

    dAB = conds["A_full"]["r2"] - conds["B_zeta_wrist_zero"]["r2"]
    dABo = conds["Ap_oracle_full"]["r2"] - conds["Bp_oracle_wrist_zero"]["r2"]
    gate = "GO (>=0.27)" if dAB >= GATE else "FAIL (<0.27)"
    res = {"probe": "wc_g2c_zero_ablation",
           "phase1_ckpt": str(P1), "phase2_ckpt": str(P2),
           "dim_main": dm, "dim_wrist": dw, "n_val": n,
           "ckpt_action_r2_ref": ck2["metrics"]["action_r2"],
           "conditions": conds,
           "delta_r2_A_minus_B": dAB, "delta_r2_oracle": dABo,
           "gate_rule": f"prereg G2-C: dR2(A-B) >= {GATE} GO", "gate": gate}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(f"[G2-C] full {conds['A_full']['r2']:+.4f} | ζw=0 "
          f"{conds['B_zeta_wrist_zero']['r2']:+.4f} | ΔR²(A−B) {dAB:+.4f} → {gate}")
    print(f"       oracle Δ {dABo:+.4f} | 저장: {OUT}")


if __name__ == "__main__":
    main()
