#!/usr/bin/env python3
"""W4 v3 P3 part (b) alignment<->SR correlation + (c) ANOVA variance decomposition.
Small-n synthesis of documented per-arm SR + alignment metrics. Suggestive, not confirmatory."""
import json, os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/user/CLIP_ws/outputs/analysis/w4v3_p3_gw_anova"
gw = json.load(open(os.path.join(OUT, "gw_alignment.json")))
bb = gw["backbones"]

# ---- per-backbone alignment metrics (computed from caches, offline) ----
ALIGN = {  # backbone -> alignment metrics
    "CLIP":    {"align_r2": bb["CLIP"]["linear_align_r2_mean"],   "cka": bb["CLIP"]["linear_cka_dz_da"],
                "gw": bb["CLIP"]["gw_distance"],   "mknn": bb["CLIP"]["mutual_knn10"]},
    "SigLIP2": {"align_r2": bb["SigLIP2"]["linear_align_r2_mean"], "cka": bb["SigLIP2"]["linear_cka_dz_da"],
                "gw": bb["SigLIP2"]["gw_distance"], "mknn": bb["SigLIP2"]["mutual_knn10"]},
}

# ---- documented closed-loop correct SR per arm (from docs/NUMBER_CARD, headline_*, C1_*) ----
# fields: arm, backbone, suite, aug, correct_SR, correct_minus_wrong, offline_action_r2 (if documented), seeds
ARMS = [
    dict(arm="CLIP-spatial",          bb="CLIP",    suite="spatial", aug=False, sr=83.5, cmw=75.8, off_r2=None, seeds=3),
    dict(arm="W3.3 CLIP-bothaug",     bb="CLIP",    suite="spatial", aug=True,  sr=87.0, cmw=82.5, off_r2=None, seeds=1),
    dict(arm="SigLIP2 pooled-large256", bb="SigLIP2", suite="spatial", aug=False, sr=86.5, cmw=76.5, off_r2=None, seeds=3),
    dict(arm="SigLIP2 pooled-only(1s)", bb="SigLIP2", suite="spatial", aug=False, sr=88.5, cmw=79.0, off_r2=0.655, seeds=1),
    dict(arm="C1 (SigLIP2 F4)",       bb="SigLIP2", suite="spatial", aug=False, sr=90.0, cmw=78.5, off_r2=0.663, seeds=1),
    dict(arm="noconsist (SigLIP2)",   bb="SigLIP2", suite="spatial", aug=False, sr=92.0, cmw=78.5, off_r2=None, seeds=1),
    dict(arm="SigLIP2-goal",          bb="SigLIP2", suite="goal",    aug=False, sr=92.0, cmw=92.0, off_r2=None, seeds=1),
    dict(arm="CLIP-goal",             bb="CLIP",    suite="goal",    aug=False, sr=None, cmw=88.5, off_r2=None, seeds=1),
]

corr = {}

# (b1) within-backbone: documented offline action-R2 vs correct SR (n=2, SigLIP2 spatial)
r2_arms = [a for a in ARMS if a["off_r2"] is not None and a["sr"] is not None]
# distinct (r2, sr) points
pts = sorted({(a["off_r2"], a["sr"]) for a in r2_arms})
xr = np.array([p[0] for p in pts]); yr = np.array([p[1] for p in pts])
corr["offline_actionR2_vs_SR"] = {
    "points": [{"off_r2": float(x), "SR": float(y)} for x, y in pts],
    "n": len(pts),
    "note": "C1(+0.663,90.0) vs pooled-only(+0.655,88.5) - within SigLIP2, n=2 documented. Direction only; no p (n=2).",
    "direction": "higher offline R2 -> higher SR (consistent, both points)",
}

# (b2) backbone-level alignment metric vs correct SR across all arms with SR (metric constant within bb)
srarms = [a for a in ARMS if a["sr"] is not None]
for metric, better in [("align_r2", "high"), ("cka", "high"), ("gw", "low"), ("mknn", "high")]:
    x = np.array([ALIGN[a["bb"]][metric] for a in srarms])
    y = np.array([a["sr"] for a in srarms])
    r, p = stats.pearsonr(x, y)
    rho, prho = stats.spearmanr(x, y)
    corr[f"backbone_{metric}_vs_SR"] = {
        "n_arms": len(srarms), "n_distinct_x": int(len(np.unique(x))),
        "pearson_r": round(float(r), 3), "pearson_p": round(float(p), 3),
        "spearman_rho": round(float(rho), 3), "spearman_p": round(float(prho), 3),
        "better_when": better,
    }

# scatter figure: backbone align_r2 vs SR, points colored by backbone, marker by suite
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
cols = {"CLIP": "#4C78A8", "SigLIP2": "#F58518"}
mk = {"spatial": "o", "goal": "s"}
for a in srarms:
    ax[0].scatter(ALIGN[a["bb"]]["align_r2"], a["sr"], c=cols[a["bb"]], marker=mk[a["suite"]], s=90,
                  edgecolor="k", linewidth=0.5, zorder=3)
    ax[0].annotate(a["arm"].replace("SigLIP2 ", "").replace("(SigLIP2 F4)", ""),
                   (ALIGN[a["bb"]]["align_r2"], a["sr"]), fontsize=7,
                   xytext=(4, 3), textcoords="offset points")
rr = corr["backbone_align_r2_vs_SR"]
ax[0].set_xlabel("linear alignment R2 (Delta-z -> Delta-action, per backbone)")
ax[0].set_ylabel("closed-loop correct SR (%)")
ax[0].set_title(f"(b) alignment vs SR  r={rr['pearson_r']} p={rr['pearson_p']} "
                f"n={rr['n_arms']} arms / {rr['n_distinct_x']} distinct x", fontsize=9)
ax[0].grid(alpha=0.25)

# within-SigLIP2 offline-R2 vs SR (n=2)
ax[1].scatter(xr, yr, c="#F58518", s=110, edgecolor="k", zorder=3)
for x, y in pts:
    lab = "C1" if abs(x - 0.663) < 1e-6 else "pooled"
    ax[1].annotate(lab, (x, y), fontsize=8, xytext=(5, 2), textcoords="offset points")
ax[1].set_xlabel("documented offline action R2")
ax[1].set_ylabel("closed-loop correct SR (%)")
ax[1].set_title("(b') within-SigLIP2 offline R2 vs SR (n=2, direction only)", fontsize=9)
ax[1].grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "alignment_vs_sr.png"), dpi=140)
plt.close()

# ---- (c) ANOVA variance decomposition ----
def anova_ss(values, groups):
    """one-way SS_between / SS_total for label array `groups`."""
    values = np.asarray(values, float)
    gm = values.mean()
    sst = ((values - gm) ** 2).sum()
    ssb = 0.0
    for g in set(groups):
        v = values[np.array(groups) == g]
        ssb += len(v) * (v.mean() - gm) ** 2
    return ssb, sst

anova = {}

# Design A: balanced 2x2 (backbone x suite) on correct-wrong  [the language-preservation metric]
# cells: CLIP-spatial 75.8, SigLIP2-spatial 76.5, CLIP-goal 88.5, SigLIP2-goal 92.0
cells = {("CLIP", "spatial"): 75.8, ("SigLIP2", "spatial"): 76.5,
         ("CLIP", "goal"): 88.5, ("SigLIP2", "goal"): 92.0}
vals = np.array(list(cells.values()))
gm = vals.mean()
sst = ((vals - gm) ** 2).sum()
suite_m = {s: np.mean([cells[(b, s)] for b in ["CLIP", "SigLIP2"]]) for s in ["spatial", "goal"]}
bb_m = {b: np.mean([cells[(b, s)] for s in ["spatial", "goal"]]) for b in ["CLIP", "SigLIP2"]}
ss_suite = 2 * sum((m - gm) ** 2 for m in suite_m.values())
ss_bb = 2 * sum((m - gm) ** 2 for m in bb_m.values())
ss_int = sst - ss_suite - ss_bb   # no replication -> residual == interaction
anova["designA_2x2_correct_minus_wrong"] = {
    "cells": {f"{b}/{s}": v for (b, s), v in cells.items()},
    "SS_total": round(sst, 2),
    "eta2_suite": round(ss_suite / sst, 4),
    "eta2_backbone": round(ss_bb / sst, 4),
    "eta2_interaction_residual": round(ss_int / sst, 4),
    "interpretation": "correct-wrong (language-preservation gap) variance is ~%.0f%% suite, ~%.0f%% backbone."
                      % (100 * ss_suite / sst, 100 * ss_bb / sst),
}

# Design B: sequential SS on correct-wrong across all documented observations
# observations (cmw, backbone, suite, aug)
obs = [
    (75.8, "CLIP", "spatial", "off"),      # 3-seed agg
    (75.5, "SigLIP2", "spatial", "off"),   # s1
    (79.0, "SigLIP2", "spatial", "off"),   # s2
    (75.0, "SigLIP2", "spatial", "off"),   # s3
    (82.5, "CLIP", "spatial", "on"),       # W3.3 bothaug
    (78.5, "SigLIP2", "spatial", "off"),   # C1
    (78.5, "SigLIP2", "spatial", "off"),   # noconsist
    (88.5, "CLIP", "goal", "off"),
    (92.0, "SigLIP2", "goal", "off"),
]
y = np.array([o[0] for o in obs], float)
gm = y.mean(); sst = ((y - gm) ** 2).sum()
seq = {}
resid = y - gm
# sequential: suite, then backbone, then aug
for fac, idx in [("suite", 2), ("backbone", 1), ("aug", 3)]:
    labels = [o[idx] for o in obs]
    # explained SS of current residual by this factor
    base = resid.copy()
    gm2 = base.mean()
    ssb = 0.0
    for g in set(labels):
        m = base[np.array(labels) == g]
        ssb += len(m) * (m.mean() - gm2) ** 2
        resid[np.array(labels) == g] -= (m.mean() - gm2)
    seq[fac] = ssb
seq_resid = ((resid - resid.mean()) ** 2).sum()
anova["designB_sequential_correct_minus_wrong"] = {
    "n_obs": len(obs), "SS_total": round(sst, 2),
    "frac_suite": round(seq["suite"] / sst, 4),
    "frac_backbone": round(seq["backbone"] / sst, 4),
    "frac_aug": round(seq["aug"] / sst, 4),
    "frac_residual_incl_seed_head": round(seq_resid / sst, 4),
    "note": "sequential (Type-I) SS, order suite>backbone>aug; unbalanced, no replication -> exploratory.",
}

# Design C: correct-SR seed-noise floor vs systematic spread (SigLIP2 pooled-large256 3 seeds)
seed_sr = np.array([87.5, 88.5, 83.5])
spatial_sr = np.array([a["sr"] for a in ARMS if a["sr"] is not None and a["suite"] == "spatial"])
anova["designC_SR_seed_noise_vs_systematic"] = {
    "seed_SR_var_pp2": round(float(seed_sr.var(ddof=1)), 3),
    "seed_SR_sd_pp": round(float(seed_sr.std(ddof=1)), 3),
    "between_arm_spatial_SR_var_pp2": round(float(spatial_sr.var(ddof=1)), 3),
    "between_arm_spatial_SR_sd_pp": round(float(spatial_sr.std(ddof=1)), 3),
    "frac_between_arm_var_above_seed_floor": round(
        float(max(0.0, spatial_sr.var(ddof=1) - seed_sr.var(ddof=1)) / spatial_sr.var(ddof=1)), 4),
    "note": "seed variance (one 3-seed cell) is the noise floor; between-arm spatial SR spread "
            "is only modestly above it -> most spatial-arm SR differences are within seed noise.",
}

out = {"correlation": corr, "anova": anova, "arms_table": ARMS, "align_per_backbone": ALIGN}
json.dump(out, open(os.path.join(OUT, "sr_correlation_anova.json"), "w"), indent=2)
print(json.dumps({"correlation": corr, "anova": anova}, indent=2))
print("\nwrote sr_correlation_anova.json + alignment_vs_sr.png")
