"""R-D0 — W-D pre-training ridge kill-gate (DESIGN_WD_WAprime_v1 §2.4).

Does future Delta z_wrist(t->t+16) in SigLIP2 space contain signal beyond state?
  target  : Zw_sig(t+span) - Zw_sig(t)  (SigLIP2-large256 pooled wrist RAW cache)
  r_state : ridge([z_main(t), z_wrist_sig(t)])            -> target, heldout Rbar^2
  r_full  : ridge([state, past-chunk 112, a_emb=g(A_past,z_prev) 1024]) -> target
  KILL W-D if r_full - r_state < +0.02.
Diagnostic arms: state+chunk112 / state+aemb / state+DINOv3-wrist-pool2-patch4
(design-doc variant). Split seed2 400/100, stride 2, RidgeCV 0.1..1e4,
train-stats standardization, per-dim R^2 uniform mean over 1024 dims.

CPU. RUN (remote):
  cd /workspace/CLIP_ws && OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= \
    HF_HOME=/data2/clip_ws_cache/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python3 scratchpad/pregates/pregate_rd0.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HOME", "/data2/clip_ws_cache/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

WS = Path(os.environ.get("PREGATE_WS", "/workspace/CLIP_ws"))
sys.path.insert(0, str(WS / "src"))
from core import chunkrep                      # noqa: E402
from data.libero import LiberoDataset          # noqa: E402

from sklearn.linear_model import RidgeCV       # noqa: E402
from sklearn.metrics import r2_score           # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
CFG = WS / "configs" / "phase2_libero_large256_matchedbase.yaml"
CACHE = Path("/data2/clip_ws_cache/cache/libero_emb_large256/siglip2-so400m/joint/raw")
DENSE = Path("/data2/clip_ws_cache/cache/libero_emb_large256/dense/"
             "dinov3-vitl16-256-pool2/pre/raw")
OUTD = WS / "outputs" / "pregates"
OUTD.mkdir(parents=True, exist_ok=True)
GATE = 0.02


def main():
    t0 = time.time()
    assert CACHE.exists(), f"SigLIP2 pooled cache missing: {CACHE}"
    assert DENSE.exists(), f"DINOv3 wrist pool2 dense cache missing: {DENSE}"
    cfg = yaml.safe_load(open(CFG))
    ck1 = torch.load(os.path.expanduser(cfg["phase1_ckpt"]), map_location="cpu",
                     weights_only=False)
    n_chunk, act_dim = ck1["n_chunk"], ck1["action_dim"]
    a_mean, a_std = ck1["a_mean"], ck1["a_std"]
    repr_kind = ck1.get("chunk_repr", "time")

    # frozen g for a_emb (same construction as rseries_common.load_phase1)
    from models.networks import DeltaAE
    m1 = ck1["config"]["model"]
    ae = DeltaAE(act_dim, n_chunk, ck1["latent_dim"], m1["hidden"], m1["layers"],
                 m1.get("dropout", 0.0), m1.get("state_cond", True)).eval()
    ae.load_state_dict(ck1["state_dict"])

    ds = LiberoDataset(cfg)
    files = ds.episode_files()
    perm = np.random.RandomState(cfg["train"]["seed"]).permutation(len(files))
    n_val = max(1, round(len(files) * cfg["data"]["val_episodes"]))
    val_ids, tr_ids = set(perm[:n_val].tolist()), set(perm[n_val:].tolist())
    span, stride = ds.span, cfg["data"].get("stride", 2)
    print(f"[split] seed2 train {len(tr_ids)} / val {len(val_ids)} "
          f"| span {span} stride {stride}")

    def norm_chunk(seg):
        ch = ds.resample_chunk(seg)
        ch = ((ch - a_mean) / a_std).astype(np.float32)
        return chunkrep.to_repr(ch, repr_kind)

    def collect(ids):
        ZM, ZW, CP, ZP, DP, Y = [], [], [], [], [], []
        for i in sorted(ids):
            ep = files[i]
            key = ds._key(ep)
            acts = ds.load_actions(ep)
            Zm = np.load(CACHE / f"{key}_agentview_rgb.npz")["Z"].astype(np.float32)
            Zw = np.load(CACHE / f"{key}_eye_in_hand_rgb.npz")["Z"].astype(np.float32)
            D = np.load(DENSE / f"{key}_eye_in_hand_rgb.npz")["D"].astype(np.float32)
            T = min(len(acts), len(Zm), len(Zw), len(D))
            starts = list(range(0, T - span, stride))
            for t in starts:
                past = (np.repeat(acts[0:1], 2, axis=0) if t == 0
                        else acts[max(t - span, 0):t])
                CP.append(norm_chunk(past))
                ZP.append(Zm[max(t - span, 0)])
            ZM.append(np.stack([Zm[t] for t in starts]))
            ZW.append(np.stack([Zw[t] for t in starts]))
            DP.append(np.stack([D[t].reshape(-1) for t in starts]))
            Y.append(np.stack([Zw[t + span] - Zw[t] for t in starts]))
        return (np.concatenate(ZM), np.concatenate(ZW),
                np.stack(CP).reshape(len(CP), -1).astype(np.float32),
                np.stack(ZP), np.concatenate(DP), np.concatenate(Y))

    ZMt, ZWt, CPt, ZPt, DPt, Yt = collect(tr_ids)
    ZMv, ZWv, CPv, ZPv, DPv, Yv = collect(val_ids)
    with torch.no_grad():
        def aemb(CP, ZP):
            out = []
            for i in range(0, len(CP), 4096):
                out.append(ae.g(torch.tensor(CP[i:i + 4096].reshape(-1, n_chunk,
                                                                    act_dim)),
                                torch.tensor(ZP[i:i + 4096])).numpy())
            return np.concatenate(out).astype(np.float32)
        AEt, AEv = aemb(CPt, ZPt), aemb(CPv, ZPv)
    print(f"[data] train {len(Yt)} / val {len(Yv)} samples "
          f"| target dim {Yt.shape[1]} | {time.time()-t0:.0f}s")

    def ridge(tag, Xtr, Xva):
        mu, sd = Xtr.mean(0), np.maximum(Xtr.std(0), 1e-6)
        rid = RidgeCV(alphas=ALPHAS).fit((Xtr - mu) / sd, Yt)
        pv = rid.predict((Xva - mu) / sd)
        r2u = float(r2_score(Yv, pv, multioutput="uniform_average"))
        r2w = float(r2_score(Yv, pv, multioutput="variance_weighted"))
        print(f"  [{tag}] dim={Xtr.shape[1]} a={rid.alpha_:g} "
              f"Rbar2 {r2u:+.4f} (var-w {r2w:+.4f})")
        return {"dim": int(Xtr.shape[1]), "alpha": float(rid.alpha_),
                "r2_uniform": r2u, "r2_var_weighted": r2w}

    S_tr = np.concatenate([ZMt, ZWt], 1)
    S_va = np.concatenate([ZMv, ZWv], 1)
    res = {"gate": "R-D0", "threshold": GATE,
           "target": "Zw_sig(t+16)-Zw_sig(t), raw pooled",
           "n_train": len(Yt), "n_val": len(Yv), "split_seed": 2}
    res["r_state"] = ridge("state z_m+z_w", S_tr, S_va)
    res["r_full"] = ridge("FULL state+chunk112+aemb",
                          np.concatenate([S_tr, CPt, AEt], 1),
                          np.concatenate([S_va, CPv, AEv], 1))
    res["diag_state_chunk"] = ridge("state+chunk112",
                                    np.concatenate([S_tr, CPt], 1),
                                    np.concatenate([S_va, CPv], 1))
    res["diag_state_aemb"] = ridge("state+aemb",
                                   np.concatenate([S_tr, AEt], 1),
                                   np.concatenate([S_va, AEv], 1))
    res["diag_state_dinopatch"] = ridge("state+dino-p4 (design variant)",
                                        np.concatenate([S_tr, DPt], 1),
                                        np.concatenate([S_va, DPv], 1))
    d = res["r_full"]["r2_uniform"] - res["r_state"]["r2_uniform"]
    res["delta_full_minus_state"] = d
    res["verdict"] = "GO (>=+0.02)" if d >= GATE else "KILL W-D (<+0.02)"
    res["_meta"] = {"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "host": os.uname().nodename,
                    "wall_s": round(time.time() - t0)}
    p = OUTD / "rd0_results.json"
    p.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"[R-D0] r_full-r_state = {d:+.4f} -> {res['verdict']} | saved {p}")


if __name__ == "__main__":
    main()
