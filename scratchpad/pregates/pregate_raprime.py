"""R-A' — W-A' patch-informativeness ridge kill-gate (DESIGN_WD_WAprime_v1 §3.4).

uplift_X = gripR2([X-patch4(4096) + z_w,sig + z_main] -> A_fut)
           - gripR2([z_w,sig + z_main] -> A_fut),   X in {SigLIP2-pool2, DINOv3-pool2}
KILL W-A' if uplift_sig < 0.7 x uplift_dino.  probe_g0_wrist_cell.py methodology
(RidgeCV 0.1..1e4, train-stats standardization, per-dim R2, gripper dim 6,
grasp/transport masks). BOTH arms on the SAME subset + split (fairness).

SigLIP2 wrist patch cache does not exist (pool_to unimplemented for
Siglip2Anchor) -> CPU-encode a SUBSET at sample starts only:
  120 episodes = first 96 of train perm + first 24 of val perm (seed2, no new RNG).
Per-episode encodings cached to outputs/pregates/sig2wp2_cache/ (resumable).
GPU FORBIDDEN (W-A confirmation running) — CUDA_VISIBLE_DEVICES forced empty.

CPU. RUN (remote):
  cd /workspace/CLIP_ws && OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= \
    HF_HOME=/data2/clip_ws_cache/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python3 scratchpad/pregates/pregate_raprime.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""            # hard: no GPU borrow
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

from PIL import Image                          # noqa: E402
from sklearn.linear_model import RidgeCV       # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
CFG = WS / "configs" / "phase2_libero_large256_matchedbase.yaml"
CACHE = Path("/data2/clip_ws_cache/cache/libero_emb_large256/siglip2-so400m/joint/raw")
DENSE = Path("/data2/clip_ws_cache/cache/libero_emb_large256/dense/"
             "dinov3-vitl16-256-pool2/pre/raw")
OUTD = WS / "outputs" / "pregates"
SIGC = OUTD / "sig2wp2_cache"
SIGC.mkdir(parents=True, exist_ok=True)
GRIP = 6
N_TR_EP, N_VA_EP = 96, 24
KILL_RATIO = 0.7
FULLSPLIT_DINO_UPLIFT = 0.0332      # probe_g0_wrist_cell.json reference


def sig_encoder():
    """SigLIP2-large256 vision tower, single pass, 16x16 -> 2x2 avg-pool."""
    import torch.nn.functional as F
    from transformers import AutoModel, AutoProcessor
    src = "google/siglip2-large-patch16-256"
    model = AutoModel.from_pretrained(src, dtype=torch.float32).eval()
    proc = AutoProcessor.from_pretrained(src)

    @torch.no_grad()
    def enc(pil_images):
        out = []
        for i in range(0, len(pil_images), 32):
            inp = proc(images=pil_images[i:i + 32], return_tensors="pt")
            v = model.vision_model(pixel_values=inp["pixel_values"])
            t = v.last_hidden_state                      # (B,256,1024) — no prefix
            B, P, D = t.shape
            assert P == 256, f"P={P} (expected 16x16=256, no CLS/registers)"
            x = t.reshape(B, 16, 16, D).permute(0, 3, 1, 2)
            x = F.adaptive_avg_pool2d(x, (2, 2))
            out.append(x.permute(0, 2, 3, 1).reshape(B, 4, D).numpy()
                       .astype(np.float32))
        return np.concatenate(out)
    return enc


def main():
    t0 = time.time()
    cfg = yaml.safe_load(open(CFG))
    ck1 = torch.load(os.path.expanduser(cfg["phase1_ckpt"]), map_location="cpu",
                     weights_only=False)
    n_chunk, act_dim = ck1["n_chunk"], ck1["action_dim"]
    a_mean, a_std = ck1["a_mean"], ck1["a_std"]
    repr_kind = ck1.get("chunk_repr", "time")
    assert repr_kind == "time" and act_dim == 7

    ds = LiberoDataset(cfg)
    files = ds.episode_files()
    perm = np.random.RandomState(cfg["train"]["seed"]).permutation(len(files))
    n_val = max(1, round(len(files) * cfg["data"]["val_episodes"]))
    val_sub = perm[:n_val][:N_VA_EP]
    tr_sub = perm[n_val:][:N_TR_EP]
    span, stride = ds.span, cfg["data"].get("stride", 2)
    print(f"[subset] train {len(tr_sub)} eps / val {len(val_sub)} eps "
          f"(first-of-perm, seed2) | span {span} stride {stride}")

    enc = None

    def collect(ids, tag):
        nonlocal enc
        SIG, DIN, ZW, ZM, AF = [], [], [], [], []
        for j, i in enumerate(sorted(ids.tolist())):
            ep = files[i]
            key = ds._key(ep)
            acts = ds.load_actions(ep)
            Zm = np.load(CACHE / f"{key}_agentview_rgb.npz")["Z"].astype(np.float32)
            Zw = np.load(CACHE / f"{key}_eye_in_hand_rgb.npz")["Z"].astype(np.float32)
            D = np.load(DENSE / f"{key}_eye_in_hand_rgb.npz")["D"].astype(np.float32)
            T = min(len(acts), len(Zm), len(Zw), len(D))
            starts = list(range(0, T - span, stride))
            cpath = SIGC / f"{key}.npz"
            if cpath.exists():
                z = np.load(cpath)
                assert list(z["starts"]) == starts, f"stale cache {cpath}"
                P4 = z["P"]
            else:
                if enc is None:
                    print("[enc] loading SigLIP2-large256 (CPU, fp32)...")
                    enc = sig_encoder()
                frames = ds.load_frames(ep, ds.wrist_camera)
                imgs = [Image.fromarray(frames[t]) for t in starts]
                P4 = enc(imgs)                                # (n,4,1024)
                tmp = cpath.with_name(cpath.name + ".tmp")
                with open(tmp, "wb") as fh:
                    np.savez_compressed(fh, P=P4, starts=np.array(starts))
                tmp.replace(cpath)
                print(f"  [{tag} {j+1}/{len(ids)}] {key}: {len(starts)} frames "
                      f"({time.time()-t0:.0f}s)", flush=True)
            SIG.append(P4.reshape(len(starts), -1))
            DIN.append(np.stack([D[t].reshape(-1) for t in starts]))
            ZW.append(np.stack([Zw[t] for t in starts]))
            ZM.append(np.stack([Zm[t] for t in starts]))
            AF.append(np.stack([ds.resample_chunk(acts[t:t + span]).ravel()
                                for t in starts]))
        return (np.concatenate(SIG), np.concatenate(DIN), np.concatenate(ZW),
                np.concatenate(ZM), np.concatenate(AF))

    Str, Dtr, Wtr, Mtr, Atr = collect(tr_sub, "train")
    Sva, Dva, Wva, Mva, Ava = collect(val_sub, "val")

    def norm(A):
        a = ((A.reshape(len(A), n_chunk, act_dim) - a_mean) / a_std
             ).astype(np.float32)
        return chunkrep.to_repr(a, repr_kind).reshape(len(A), -1)

    Ytr, Yva = norm(Atr), norm(Ava)
    g_raw = Ava.reshape(-1, n_chunk, act_dim)[:, :, GRIP]
    grasp = (g_raw.max(1) - g_raw.min(1)) > 1.0
    masks = {"grasp": grasp, "transport": ~grasp}
    print(f"[data] train {len(Ytr)} / val {len(Yva)} samples | "
          f"grasp {int(grasp.sum())} / transport {int((~grasp).sum())} "
          f"| {time.time()-t0:.0f}s")

    def per_dim_r2(y, p, mask=None):
        y = y.reshape(-1, n_chunk, act_dim)
        p = p.reshape(-1, n_chunk, act_dim)
        if mask is not None:
            y, p = y[mask], p[mask]
        out = []
        for k in range(act_dim):
            yk = y[:, :, k].ravel().astype(np.float64)
            pk = p[:, :, k].ravel().astype(np.float64)
            dev = ((yk - yk.mean()) ** 2).sum()
            out.append(float(1 - ((yk - pk) ** 2).sum() / (dev + 1e-12)))
        return out

    def ridge_arm(tag, Xtr, Xva):
        mu, sd = Xtr.mean(0), np.maximum(Xtr.std(0), 1e-6)
        rid = RidgeCV(alphas=ALPHAS).fit((Xtr - mu) / sd, Ytr)
        pv = rid.predict((Xva - mu) / sd)
        per = per_dim_r2(Yva, pv)
        r = {"dim": int(Xtr.shape[1]), "alpha": float(rid.alpha_),
             "r2_per_dim": per, "r2_gripper": per[GRIP],
             "r2_macro": float(np.mean(per))}
        for mn, m in masks.items():
            r[f"r2_gripper_{mn}"] = per_dim_r2(Yva, pv, m)[GRIP]
        print(f"  [{tag}] dim={r['dim']} a={r['alpha']:g} "
              f"grip {r['r2_gripper']:+.4f} macro {r['r2_macro']:+.4f} "
              f"| grasp {r['r2_gripper_grasp']:+.4f} "
              f"transport {r['r2_gripper_transport']:+.4f}")
        return r

    base_tr = np.concatenate([Wtr, Mtr], 1)
    base_va = np.concatenate([Wva, Mva], 1)
    res = {"gate": "R-A' (W-A' kill-gate)", "kill_ratio": KILL_RATIO,
           "subset": {"train_eps": int(len(tr_sub)), "val_eps": int(len(val_sub)),
                      "rule": "first 96 of train perm + first 24 of val perm (seed2)",
                      "fairness": "both arms same subset/split/targets"},
           "n_train": len(Ytr), "n_val": len(Yva),
           "reference_fullsplit_dino_uplift": FULLSPLIT_DINO_UPLIFT}
    res["armB_base"] = ridge_arm("B: z_w,sig+z_main", base_tr, base_va)
    res["armA_dino"] = ridge_arm("A: dino-p4+base",
                                 np.concatenate([Dtr, base_tr], 1),
                                 np.concatenate([Dva, base_va], 1))
    res["armA_sig"] = ridge_arm("A': sig-p4+base",
                                np.concatenate([Str, base_tr], 1),
                                np.concatenate([Sva, base_va], 1))
    ud = res["armA_dino"]["r2_gripper"] - res["armB_base"]["r2_gripper"]
    us = res["armA_sig"]["r2_gripper"] - res["armB_base"]["r2_gripper"]
    res["uplift_dino"], res["uplift_sig"] = ud, us
    res["ratio_sig_over_dino"] = (us / ud) if abs(ud) > 1e-9 else None
    if ud <= 0:
        res["verdict"] = "INDETERMINATE (dino uplift <=0 on subset — gate ill-posed)"
    else:
        res["verdict"] = ("KILL W-A' (uplift_sig < 0.7 x uplift_dino)"
                          if us < KILL_RATIO * ud else "GO (>= 0.7 x dino)")
    res["_meta"] = {"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "host": os.uname().nodename,
                    "wall_s": round(time.time() - t0)}
    p = OUTD / "raprime_results.json"
    p.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"[R-A'] uplift dino {ud:+.4f} / sig {us:+.4f} "
          f"(ratio {res['ratio_sig_over_dino']}) -> {res['verdict']} | saved {p}")


if __name__ == "__main__":
    main()
