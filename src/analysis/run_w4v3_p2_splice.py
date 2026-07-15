"""W4 v3 P2 driver: SpLiCE sparse concept decomposition of LIBERO latent
displacements Dz. Two panels: frozen CLIP ViT-L/14 (768-d) and frozen SigLIP2
so400m (1152-d). The concept dictionary is rebuilt per tower. OFFLINE, CPU-only.

Outputs -> outputs/analysis/w4v3_p2_splice/            (CLIP panel)
           outputs/analysis/w4v3_p2_splice/siglip2/    (SigLIP2 panel)
  dictionary.npz, metrics.json, delta_concepts.json,
  fig_timeline_<task>.png, fig_delta_bars.png
"""
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import glob
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analysis import splice_concepts as sc  # noqa: E402

OUT = "outputs/analysis/w4v3_p2_splice"
ACT_THRESH = 1e-3
EDGE = 3               # avg first/last EDGE frames -> start/end embeddings

TASK_PATTERNS = [
    ("from_table_center", "pick_up_the_black_bowl_from_table_center*"),
    ("on_the_stove", "pick_up_the_black_bowl_on_the_stove*"),
    ("in_the_top_drawer", "pick_up_the_black_bowl_in_the_top_drawer*"),
    ("on_the_wooden_cabinet", "pick_up_the_black_bowl_on_the_wooden_cabinet*"),
    ("next_to_the_plate", "pick_up_the_black_bowl_next_to_the_plate*"),
    ("next_to_the_ramekin", "pick_up_the_black_bowl_next_to_the_ramekin*"),
    ("next_to_the_cookie_box", "pick_up_the_black_bowl_next_to_the_cookie_box*"),
    ("between_plate_ramekin", "pick_up_the_black_bowl_between*"),
]

TOWERS = {
    "clip": {
        "id": sc.MODEL_ID,
        "cache": "outputs/cache/libero_emb",
        "build": lambda c: sc.build_dictionary(c, ensemble=True),
        "out": OUT,
        "alphas": [3e-5, 5e-5, 8e-5, 1e-4, 2e-4],
    },
    "siglip2": {
        "id": sc.SIGLIP_ID,
        "cache": "outputs/cache/libero_emb/siglip2-so400m/joint/norm",
        "build": lambda c: sc.build_dictionary_siglip(c, ensemble=True),
        "out": os.path.join(OUT, "siglip2"),
        "alphas": [1e-5, 2e-5, 3e-5, 5e-5, 8e-5],
    },
}


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def first_agentview(cache, pattern):
    fs = sorted(glob.glob(os.path.join(cache, pattern + "agentview_rgb.npz")))
    return fs[0] if fs else None


def run_tower(tag, cfg, concepts, cats):
    out = cfg["out"]
    os.makedirs(out, exist_ok=True)
    print(f"\n=== TOWER {tag} ({cfg['id']}) ===")
    D = cfg["build"](concepts)
    np.savez(os.path.join(out, "dictionary.npz"),
             D=D, concepts=np.array(concepts, object), cats=np.array(cats, object))

    all_files = sorted(glob.glob(os.path.join(cfg["cache"], "*agentview_rgb.npz")))
    samp = np.stack([np.load(all_files[i])["Z"][j]
                     for i in range(0, min(len(all_files), 60), 3) for j in [0, -1]])

    # calibrate alpha: pick one with median active in [5,15] (prefer ~8), best cos
    calib, best = {}, None
    for a in cfg["alphas"]:
        res = [sc.decompose(z, D, a) for z in samp]
        acts = [sc.active_count(w, ACT_THRESH) for w, _, _ in res]
        med = int(np.median(acts))
        cos = float(np.mean([c for _, _, c in res]))
        calib[f"{a:.0e}"] = {"median_active": med,
                             "mean_recon_rel": round(float(np.mean([r for _, r, _ in res])), 4),
                             "mean_cos_recon": round(cos, 4)}
        if 5 <= med <= 15:
            score = -abs(med - 8) + cos  # prefer ~8 active, then higher cos
            if best is None or score > best[1]:
                best = (a, score)
    alpha = best[0] if best else cfg["alphas"][len(cfg["alphas"]) // 2]

    ceil_cos = []
    for z in samp[:20]:
        w, *_ = np.linalg.lstsq(D.T, z.astype(np.float64), rcond=None)
        ceil_cos.append(float(np.dot(unit(z), unit(D.T @ w))))

    def top_delta(dw, n=7):
        idx = np.argsort(-np.abs(dw))[:n]
        return [{"concept": concepts[i], "delta": round(float(dw[i]), 4)}
                for i in idx if abs(dw[i]) > ACT_THRESH]

    def top_w(w, n=6):
        idx = np.argsort(-w)[:n]
        return [{"concept": concepts[i], "w": round(float(w[i]), 4)}
                for i in idx if w[i] > ACT_THRESH]

    episodes, per_frame_cos = {}, []
    for name, pat in TASK_PATTERNS:
        f = first_agentview(cfg["cache"], pat)
        if f is None:
            continue
        Z = np.load(f)["Z"]
        z0, z1 = unit(Z[:EDGE].mean(0)), unit(Z[-EDGE:].mean(0))
        w0, _, c0 = sc.decompose(z0, D, alpha)
        w1, _, c1 = sc.decompose(z1, D, alpha)
        dw = w1 - w0
        step = max(1, len(Z) // 30)
        cos_traj = [sc.decompose(unit(Z[t]), D, alpha)[2] for t in range(0, len(Z), step)]
        per_frame_cos += cos_traj
        episodes[name] = {
            "file": os.path.basename(f), "T": int(len(Z)),
            "start_top": top_w(w0), "end_top": top_w(w1),
            "delta_up": [d for d in top_delta(dw) if d["delta"] > 0],
            "delta_down": [d for d in top_delta(dw) if d["delta"] < 0],
            "recon_cos_start": round(c0, 4), "recon_cos_end": round(c1, 4),
            "recon_cos_traj_mean": round(float(np.mean(cos_traj)), 4),
        }
        print(f"[{tag}][ep] {name:22s} T={len(Z):3d} cos~{np.mean(cos_traj):.2f} "
              f"UP={[d['concept'] for d in episodes[name]['delta_up'][:2]]} "
              f"DOWN={[d['concept'] for d in episodes[name]['delta_down'][:2]]}")

    metrics = {
        "tower": cfg["id"],
        "embedding": f"joint-space L2-normalized agentview_rgb pooled (T,{D.shape[1]})",
        "n_concepts": len(concepts), "prompt_ensemble": sc.PROMPT_TEMPLATES,
        "alpha_selected": alpha, "active_thresh": ACT_THRESH,
        "calibration_sweep": calib,
        "lstsq_ceiling_cos_mean": round(float(np.mean(ceil_cos)), 4),
        "note_ceiling": ("Unconstrained LS fit over full dictionary = upper bound "
                         "on cosine any text-atom reconstruction can reach; "
                         "1->ceiling is the modality/domain gap, ceiling->sparse "
                         "is the sparsity cost."),
        "sample_frames": int(len(samp)),
        "recon_cos_traj_mean_over_episodes": round(float(np.mean(per_frame_cos)), 4),
    }
    with open(os.path.join(out, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    with open(os.path.join(out, "delta_concepts.json"), "w") as fh:
        json.dump(episodes, fh, indent=2)

    # timeline figures (killer figure)
    for name, pat in [("on_the_stove", "pick_up_the_black_bowl_on_the_stove*"),
                      ("in_the_top_drawer", "pick_up_the_black_bowl_in_the_top_drawer*")]:
        f = first_agentview(cfg["cache"], pat)
        if f is None:
            continue
        Z = np.load(f)["Z"]; T = len(Z)
        W = np.stack([sc.decompose(unit(Z[t]), D, alpha)[0] for t in range(T)])
        dw = W[-EDGE:].mean(0) - W[:EDGE].mean(0)
        track = np.argsort(-np.abs(dw))[:6]
        tt = np.arange(T) / (T - 1)
        plt.figure(figsize=(9, 4.8))
        for i in track:
            plt.plot(tt, W[:, i], lw=2, label=f"{concepts[i]}  (Δ={dw[i]:+.2f})")
        plt.xlabel("normalized trajectory time  (0=start, 1=end)")
        plt.ylabel("SpLiCE concept weight  w")
        plt.title(f"Δ-concept timeline — {name} [{tag}]\n{cfg['id']}, "
                  f"{len(concepts)} concepts, α={alpha:.0e}, "
                  f"recon cos≈{episodes.get(name, {}).get('recon_cos_traj_mean', float('nan')):.2f} "
                  f"(ceiling {metrics['lstsq_ceiling_cos_mean']:.2f})")
        plt.legend(fontsize=7, loc="upper left", framealpha=0.9)
        plt.grid(alpha=0.25); plt.tight_layout()
        p = os.path.join(out, f"fig_timeline_{name}.png")
        plt.savefig(p, dpi=130); plt.close()
        print("[fig]", p)

    # cross-episode delta bar summary
    from collections import defaultdict
    agg = defaultdict(float)
    for ep in episodes.values():
        for d in ep["delta_up"] + ep["delta_down"]:
            agg[d["concept"]] += d["delta"]
    items = sorted(agg.items(), key=lambda kv: kv[1])
    labels = [k for k, _ in items]; vals = [v for _, v in items]
    plt.figure(figsize=(9, max(4, 0.35 * len(labels))))
    plt.barh(range(len(labels)), vals,
             color=["#c0392b" if v < 0 else "#27ae60" for v in vals])
    plt.yticks(range(len(labels)), labels, fontsize=8)
    plt.axvline(0, color="k", lw=0.8)
    plt.xlabel("summed Δ weight across episodes  (green=↑ end, red=↓ end)")
    plt.title(f"Cross-episode Δ-concept summary [{tag}] "
              f"(n={len(episodes)} LIBERO-Spatial pick-place)")
    plt.tight_layout()
    p = os.path.join(out, "fig_delta_bars.png")
    plt.savefig(p, dpi=130); plt.close()
    print("[fig]", p)
    print(f"[{tag}] alpha={alpha:.0e} recon_cos={metrics['recon_cos_traj_mean_over_episodes']} "
          f"ceiling={metrics['lstsq_ceiling_cos_mean']}")
    return metrics


def main():
    concepts, cats = sc.build_vocab()
    print(f"[vocab] {len(concepts)} concepts")
    which = sys.argv[1:] or ["clip", "siglip2"]
    for tag in which:
        run_tower(tag, TOWERS[tag], concepts, cats)


if __name__ == "__main__":
    main()
