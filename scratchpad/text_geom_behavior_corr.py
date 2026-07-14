"""cowork-C2: per-task text-geometry <-> behavior correlation (LOCAL, CPU-only).

For each axis (object, action) and tower (CLIP ViT-L/14, SigLIP2-so400m):
  per-task text-distance = mean_p [ 1 - cos(correct, paraphrase_p) ]   (task-mean)
  per-task SR-drop        = correct_baseline - para_SR_task            (recorded)
Then Spearman + Pearson correlation across the 10 tasks.

Recorded per-task para SR (verification_log.md), correct suite-mean baselines:
  CLIP-goal   correct = 88.5 ; SigLIP2-goal correct = 92.0
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # force CPU (no remote/local GPU)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import sys
sys.path.insert(0, "/home/user/CLIP_ws/src")

import numpy as np
from scipy.stats import spearmanr, pearsonr

from eval_libero.libero_para import LIBERO_PARA, BASE
from core.anchor import ClipAnchor, Siglip2Anchor

AXES = ("object", "action")
TASKS = list(range(10))

# --- recorded per-task paraphrase SR (t0..t9) ---
PARA_SR = {
    "CLIP": {
        "object": [70, 65, 45, 35, 90, 55, 35, 25, 50, 40],   # mean 51.0
        "action": [75, 55, 70, 65, 95, 65, 55, 25, 60, 45],   # mean 61.0
    },
    "SigLIP2": {
        "object": [35, 70, 40, 25, 40, 40, 35, 30, 50, 35],   # mean 40.0
        "action": [65, 90, 70, 60, 90, 65, 75, 20, 85, 50],   # mean 67.0
    },
}
CORRECT_BASE = {"CLIP": 88.5, "SigLIP2": 92.0}   # suite-mean (per-task correct NOT recorded)


def per_task_textdist(anchor):
    """returns {axis: [dist_t0..dist_t9]} where dist = 1 - mean_p cos(correct,para)."""
    out = {ax: [] for ax in AXES}
    for tid in TASKS:
        base_emb = anchor.encode_texts([BASE[tid]])["embeds"][0]        # L2-normed
        base_emb = base_emb / np.linalg.norm(base_emb)
        for ax in AXES:
            paras = LIBERO_PARA[tid][ax]
            pe = anchor.encode_texts(paras)["embeds"]
            pe = pe / np.linalg.norm(pe, axis=1, keepdims=True)
            cos = pe @ base_emb                          # (n_para,)
            out[ax].append(float(1.0 - cos.mean()))      # task-mean text-distance
    return out


def main():
    print("loading CLIP ViT-L/14 (CPU)...", flush=True)
    clip = ClipAnchor()
    print("loading SigLIP2-so400m (CPU)...", flush=True)
    try:
        sig = Siglip2Anchor()
        sig_ok = True
    except Exception as e:
        sig, sig_ok = None, False
        print(f"SigLIP2 LOAD FAILED -> UNVERIFIED: {e}", flush=True)

    textdist = {"CLIP": per_task_textdist(clip)}
    if sig_ok:
        textdist["SigLIP2"] = per_task_textdist(sig)

    print("\n=== per-task text-distance (1 - cos), task-mean ===")
    for tower in textdist:
        for ax in AXES:
            vals = textdist[tower][ax]
            print(f"{tower:8s} {ax:7s}: " + " ".join(f"{v:.4f}" for v in vals) +
                  f"   mean={np.mean(vals):.4f}")

    print("\n=== correlation: text-distance  vs  SR-drop (drop = base - para_SR) ===")
    print(f"{'tower':8s} {'axis':7s} {'n':>2s} {'spearman_r':>11s} {'sp_p':>7s} "
          f"{'pearson_r':>10s} {'pe_p':>7s}")
    rows = []
    for tower in textdist:
        base = CORRECT_BASE[tower]
        for ax in AXES:
            td = np.array(textdist[tower][ax])
            drop = base - np.array(PARA_SR[tower][ax], dtype=float)
            sr, sp = spearmanr(td, drop)
            pr, pp = pearsonr(td, drop)
            rows.append((tower, ax, len(td), sr, sp, pr, pp))
            print(f"{tower:8s} {ax:7s} {len(td):2d} {sr:11.3f} {sp:7.3f} "
                  f"{pr:10.3f} {pp:7.3f}")

    # pooled within-tower (object+action, n=20) + z-scored per-cell pool
    print("\n=== POOLED within-tower (object+action, n=20) ===")
    for tower in textdist:
        base = CORRECT_BASE[tower]
        td = np.array(textdist[tower]["object"] + textdist[tower]["action"])
        drop = np.array([base - s for s in PARA_SR[tower]["object"] + PARA_SR[tower]["action"]], float)
        sr, sp = spearmanr(td, drop); pr, pp = pearsonr(td, drop)
        print(f"{tower:8s} pooled n=20  spearman={sr:.3f} (p={sp:.3f})  pearson={pr:.3f} (p={pp:.3f})")

    # object vs action asymmetry (mean text-distance gap per tower)
    print("\n=== object-vs-action asymmetry (mean text-distance) ===")
    for tower in textdist:
        o = np.mean(textdist[tower]["object"]); a = np.mean(textdist[tower]["action"])
        print(f"{tower:8s}: object={o:.4f}  action={a:.4f}  obj-act gap=+{o-a:.4f}")

    import json
    dump = {"textdist": textdist,
            "para_sr": PARA_SR, "correct_base": CORRECT_BASE,
            "corr": [{"tower": t, "axis": ax, "n": n,
                      "spearman_r": sr, "spearman_p": sp,
                      "pearson_r": pr, "pearson_p": pp}
                     for (t, ax, n, sr, sp, pr, pp) in rows]}
    with open("/home/user/CLIP_ws/scratchpad/text_geom_behavior_corr_out.json", "w") as f:
        json.dump(dump, f, indent=2)
    print("\nwrote scratchpad/text_geom_behavior_corr_out.json")


if __name__ == "__main__":
    main()
