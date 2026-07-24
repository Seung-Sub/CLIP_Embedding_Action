"""R-0 step 1 — build OUR effect bank (train 400 eps) + held-out segment set (100 eps).

Mirrors colleague build_effect_bank.py / train_semantic_adapter.build_segments,
but on OUR substrate (phase1_libero_siglip2_large256 g, large256 RAW pooled cache).
Stores per segment: raw action chunk (execution asset for R-2/R-3), zeta, z_t
(for the G-R0b ridge residualization), dz (visual effect, same cache), netd, grip,
label, seglen, provenance.

CPU-only. RUN (remote):
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= python scratchpad/rseries/r0_build_bank.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rseries_common import (CACHE_Z, CAMERA, HDF5_ROOT, OUT, PHASE1_CKPT,
                            SPLIT_SEED, episode_split, load_banks, load_phase1,
                            preprocess_chunk)


def build_split(keys, bank, cls_id, ae, ck, name):
    by_key = {e["episode"]: e for e in bank["episodes"]}
    chunks, zts, rows = [], [], []
    n_skip = 0
    for key in sorted(keys):
        e = by_key.get(key)
        zp = CACHE_Z / f"{key}_{CAMERA}.npz"
        if e is None or not zp.exists():
            n_skip += 1
            continue
        Z = np.load(zp)["Z"].astype(np.float32)          # (T,1024) RAW pooled
        with h5py.File(HDF5_ROOT / f"{e['task']}.hdf5", "r") as h:
            acts = h[f"data/{e['demo']}/actions"][:].astype(np.float64)
        T = len(Z)
        Ta = min(T, len(acts))
        for s in e["segments"]:
            a, b = int(s["segment"][0]), int(s["segment"][1])
            a = min(a, Ta - 1)
            b = min(max(b, a + 1), Ta)
            seg = acts[a:b]
            if len(seg) < 1:
                continue
            chunks.append(preprocess_chunk(seg, ck))
            zts.append(Z[a])
            be = min(b, T - 1)
            rows.append(dict(
                raw_chunk=seg.astype(np.float32),
                dz=(Z[be] - Z[a]),
                netd=seg[:, :3].sum(axis=0),
                grip=(float(seg[0, 6]), float(seg[-1, 6])),
                lab=cls_id[s["phrase"]],
                seglen=int(len(seg)),
                prov={"task": e["task"], "demo": e["demo"], "seg": [a, b],
                      "key": key}))
    with torch.no_grad():
        zeta = ae.g(torch.tensor(np.stack(chunks)),
                    torch.tensor(np.stack(zts))).numpy().astype(np.float32)
    out = dict(
        zeta=zeta,
        zt=np.stack(zts).astype(np.float32),
        dz=np.stack([r["dz"] for r in rows]).astype(np.float32),
        netd=np.stack([r["netd"] for r in rows]).astype(np.float64),
        grip=np.array([r["grip"] for r in rows], np.float64),
        lab=np.array([r["lab"] for r in rows], np.int64),
        seglen=np.array([r["seglen"] for r in rows], np.int64),
        raw_chunks=[r["raw_chunk"] for r in rows],
        prov=[r["prov"] for r in rows])
    print(f"[{name}] eps kept {len(set(r['prov']['key'] for r in rows))} "
          f"(skip {n_skip}) | segments {len(rows)}")
    return out


def main():
    ae, ck = load_phase1()
    print(f"[frozen] g from {PHASE1_CKPT.name} | latent {ck['latent_dim']} "
          f"n_chunk {ck['n_chunk']} repr {ck.get('chunk_repr')} "
          f"anchor {ck['anchor']}")
    bank, hard, their = load_banks()
    classes = sorted({s["phrase"] for e in bank["episodes"] for s in e["segments"]})
    cls_id = {p: i for i, p in enumerate(classes)}
    train_keys, heldout_keys = episode_split()
    print(f"[split] seed={SPLIT_SEED} train {len(train_keys)} / held-out {len(heldout_keys)}")

    BK = {"classes": classes, "cls_id": cls_id,
          "phase1_ckpt": str(PHASE1_CKPT), "split_seed": SPLIT_SEED,
          "substrate": "large256-single (siglip2-so400m/joint/raw, "
                       "cache libero_emb_large256)",
          "train": build_split(train_keys, bank, cls_id, ae, ck, "train"),
          "heldout": build_split(heldout_keys, bank, cls_id, ae, ck, "heldout")}
    for sp in ("train", "heldout"):
        lab = BK[sp]["lab"]
        print(f"  {sp} class counts:",
              {c: int((lab == i).sum()) for i, c in enumerate(classes)})
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "effect_bank_ours.pkl", "wb") as f:
        pickle.dump(BK, f)
    print(f"[saved] {OUT/'effect_bank_ours.pkl'} "
          f"(train {len(BK['train']['lab'])} / heldout {len(BK['heldout']['lab'])})")


if __name__ == "__main__":
    main()
