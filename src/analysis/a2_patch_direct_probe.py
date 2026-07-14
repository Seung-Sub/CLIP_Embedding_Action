#!/usr/bin/env python3
"""A2 PATCH-DIRECT closure — LOCAL DINOv2 per-frame dense cache (CPU-only, 0 encoding).

Reuses w5_granularity_probe machinery (chunk_targets, eval_arm = RidgeCV + StandardScaler,
same fine/coarse target columns, episode-level split, seed 0). DINOv2 patch ζ_f readout =
pool(ΔF) ⊕ patchnorm(ΔF)  (dim 1024+256 = 1280), at span=4 and span=16, matched-horizon
targets (chunk_targets computes the action delta over the SAME span used for ΔF).
"""
import os, sys, glob
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
sys.path.insert(0, "/home/user/CLIP_ws")
from src.analysis.w5_granularity_probe import chunk_targets, eval_arm

import h5py

DENSE = "/home/user/CLIP_ws/outputs/cache/libero_emb/dense/dinov2-large-nc-reg/pre/norm"
HDF5 = ("/home/user/CLIP/data/libero/libero_spatial/"
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5")
DEMOS = ["demo_0", "demo_1"]
FINE_COLS = ["gripper", "netRot_x", "netRot_y", "netRot_z", "fine_res"]
COARSE_COLS = ["netPos_x", "netPos_y", "netPos_z"]


def load_ep(demo):
    p = os.path.join(DENSE, f"pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo_{demo}_agentview_rgb.npz")
    D = np.load(p)["D"].astype(np.float32)          # (T, 256, 1024) per-frame patch tokens
    with h5py.File(HDF5, "r") as h:
        actions = h[f"data/{demo}/actions"][:].astype(np.float64)
    T = min(len(D), len(actions))
    return D[:T], actions[:T]


def build(span, stride):
    """Per-episode: patch readout at span + matched-horizon targets. Returns list of dicts."""
    eps = []
    for demo in DEMOS:
        D, actions = load_ep(demo)
        starts, coarse, fine, eedelta = chunk_targets(actions, span, stride)
        if len(starts) == 0:
            continue
        ends = starts + span
        dF = D[ends] - D[starts]                     # (n, 256, 1024) same-index patch diff
        meanDF = dF.mean(axis=1)                      # (n, 1024)  pool(ΔF) = ζ_g-ish
        patchnorm = np.linalg.norm(dF, axis=2)        # (n, 256)   spatial "where changed"
        patch_full = np.concatenate([meanDF, patchnorm], axis=1)   # (n, 1280) A1 readout
        eps.append(dict(demo=demo, n=len(starts),
                        meanDF=meanDF.astype(np.float32),
                        patchnorm=patchnorm.astype(np.float32),
                        patch=patch_full.astype(np.float32),
                        coarse=coarse, fine=fine))
    return eps


def split_eval(eps, test_idx):
    """Episode-level split: episodes in test_idx -> test, rest -> train."""
    tr = [e for i, e in enumerate(eps) if i not in test_idx]
    te = [e for i, e in enumerate(eps) if i in test_idx]
    def cat(lst, k): return np.concatenate([e[k] for e in lst], axis=0)
    out = {}
    for arm in ["patch", "meanDF", "patchnorm"]:
        Xtr, Xte = cat(tr, arm), cat(te, arm)
        rc = eval_arm(Xtr, Xte, cat(tr, "coarse"), cat(te, "coarse"))
        rf = eval_arm(Xtr, Xte, cat(tr, "fine"), cat(te, "fine"))
        out[arm] = dict(coarse=rc, fine=rf)
    return out, sum(e["n"] for e in tr), sum(e["n"] for e in te)


def main():
    for stride in [8, 4]:
        print(f"\n{'='*70}\nSTRIDE={stride}\n{'='*70}")
        for span in [4, 16]:
            eps = build(span, stride)
            nchunks = [e["n"] for e in eps]
            print(f"\n--- span={span} (stride={stride}) chunks per ep: {dict(zip([e['demo'] for e in eps], nchunks))} ---")
            # canonical seed-0 episode split (w5 logic: rng.permutation, n_test=max(1,round(0.2*k)))
            rng = np.random.default_rng(0)
            k = len(eps)
            order = rng.permutation(k)
            n_test = max(1, int(round(k * 0.2)))
            test_eps0 = set(order[:n_test].tolist())
            # also both directions (n=2 => single split arbitrary), average for robustness
            results_dirs = []
            for label, tset in [("seed0", test_eps0), ("testE0", {0}), ("testE1", {1})]:
                res, ntr, nte = split_eval(eps, tset)
                results_dirs.append((label, res, ntr, nte))
                pf = res["patch"]["fine"]; pc = res["patch"]["coarse"]
                print(f"  [{label:6s} tr={ntr} te={nte}] PATCH fine R2={pf['r2_macro']:+.3f} "
                      f"(grip={pf['r2_per'][0]:+.3f} rot={np.mean(pf['r2_per'][1:4]):+.3f} "
                      f"fineRes={pf['r2_per'][4]:+.3f}) | coarse R2={pc['r2_macro']:+.3f}")
            # averaged over the two directions (testE0, testE1) -> symmetric use of both eps
            def avg(field, col=None):
                vals = []
                for _, res, _, _ in results_dirs[1:]:
                    r = res["patch"][field]
                    vals.append(r["r2_macro"] if col is None else r["r2_per"][col])
                return float(np.mean(vals))
            print(f"  [AVG(E0,E1)] PATCH fine_macro={avg('fine'):+.3f} "
                  f"grip={avg('fine',0):+.3f} rot={np.mean([avg('fine',i) for i in (1,2,3)]):+.3f} "
                  f"fineRes={avg('fine',4):+.3f} | coarse_macro={avg('coarse'):+.3f}")


if __name__ == "__main__":
    main()
