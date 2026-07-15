#!/usr/bin/env python3
"""W4 v3 P3: GW/Wasserstein latent<->action alignment geometry + SR correlation + ANOVA.
OFFLINE, CPU-ONLY. No GPU. Uses cached pooled embeddings (CLIP 768-d, SigLIP2-so400m 1152-d)
for latent displacement Delta-z, and LIBERO libero_spatial action chunks for Delta-action.
"""
import numpy as np, glob, re, os, json, h5py
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
OUT = "/home/user/CLIP_ws/outputs/analysis/w4v3_p3_gw_anova"
os.makedirs(OUT, exist_ok=True)
HDF5_DIR = "/home/user/CLIP_ws/data/libero/libero_spatial"
CACHES = {
    "CLIP":    "/home/user/CLIP/outputs/cache/libero_emb",                              # Z (T,768)
    "SigLIP2": "/home/user/CLIP/outputs/cache/libero_emb/siglip2-so400m/joint/raw",     # Z (T,1152)
}
PAT = re.compile(r"(.+)_demo_(\d+)_agentview_rgb$")
K = 8          # displacement / action-chunk horizon (steps)
STRIDE = 4     # stride between chunk starts
N_GW = 400     # subsample size for Gromov-Wasserstein
MAX_DEMOS = 200  # cap demos loaded per backbone for speed

# ---------------------------------------------------------------- data loading
def build_pairs(cache_dir):
    """Return Dz (n,Dz), Da (n,7): latent displacements & net action chunks."""
    fs = sorted(glob.glob(cache_dir + "/*agentview_rgb.npz"))[:MAX_DEMOS]
    Dz, Da = [], []
    hcache = {}
    for f in fs:
        m = PAT.match(os.path.basename(f)[:-4])
        if not m:
            continue
        stem, did = m.group(1), m.group(2)
        hp = os.path.join(HDF5_DIR, stem + ".hdf5")
        if not os.path.exists(hp):
            continue
        if hp not in hcache:
            hcache[hp] = h5py.File(hp, "r")
        grp = hcache[hp]["data"].get("demo_" + did)
        if grp is None:
            continue
        acts = grp["actions"][:]           # (T,7)
        z = np.load(f)["Z"]                # (T,D)
        T = min(len(z), len(acts))
        for t in range(0, T - K, STRIDE):
            Dz.append(z[t + K] - z[t])
            Da.append(acts[t:t + K].sum(axis=0))   # net motion over the chunk
    for h in hcache.values():
        h.close()
    return np.asarray(Dz, np.float64), np.asarray(Da, np.float64)

# ---------------------------------------------------------------- metrics
def linear_cka(X, Y):
    """Linear CKA between paired sample sets X (n,dx), Y (n,dy)."""
    X = X - X.mean(0); Y = Y - Y.mean(0)
    hsic = np.linalg.norm(X.T @ Y, "fro") ** 2
    return hsic / (np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro"))

def linear_align_r2(X, Y, folds=5, seed=0):
    """Held-out R^2 of linear map X->Y (multi-output ridge), averaged over CV folds.
    Measures how well a single linear transform aligns Delta-z with Delta-action."""
    r = np.random.default_rng(seed)
    n = len(X); idx = r.permutation(n)
    Xc = X - X.mean(0)
    parts = np.array_split(idx, folds)
    lam = 1e-2 * np.trace(Xc.T @ Xc) / Xc.shape[1]
    r2s = []
    for i in range(folds):
        te = parts[i]; tr = np.concatenate([parts[j] for j in range(folds) if j != i])
        xtr, ytr = X[tr], Y[tr]; xte, yte = X[te], Y[te]
        mx = xtr.mean(0); my = ytr.mean(0)
        xtr0 = xtr - mx; ytr0 = ytr - my
        A = xtr0.T @ xtr0 + lam * np.eye(xtr0.shape[1])
        W = np.linalg.solve(A, xtr0.T @ ytr0)
        pred = (xte - mx) @ W + my
        ss_res = ((yte - pred) ** 2).sum()
        ss_tot = ((yte - yte.mean(0)) ** 2).sum()
        r2s.append(1 - ss_res / ss_tot)
    return float(np.mean(r2s)), float(np.std(r2s))

def mutual_knn(X, Y, k=10):
    """Fraction of shared k-nearest-neighbours between the two paired spaces."""
    def knn(M):
        Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        S = Mn @ Mn.T; np.fill_diagonal(S, -np.inf)
        return np.argsort(-S, axis=1)[:, :k]
    a, b = knn(X), knn(Y)
    return float(np.mean([len(set(a[i]) & set(b[i])) / k for i in range(len(X))]))

def cos_cost(M):
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    C = 1.0 - Mn @ Mn.T
    return C / C.mean()          # normalise so cross-backbone GW is comparable

def entropic_gw(C1, C2, eps=0.05, outer=60, inner=60):
    """Entropic Gromov-Wasserstein (squared loss, uniform marginals). Returns (T, gw_cost)."""
    n, m = C1.shape[0], C2.shape[0]
    p = np.full(n, 1.0 / n); q = np.full(m, 1.0 / m)
    C1sq, C2sq = C1 ** 2, C2 ** 2
    constC = C1sq @ np.outer(p, np.ones(m)) + np.outer(np.ones(n), q) @ C2sq.T
    T = np.outer(p, q)
    for _ in range(outer):
        tens = constC - C1 @ T @ (2 * C2).T
        Kmat = np.exp(-tens / eps)
        u = np.ones(n)
        for _ in range(inner):
            u = p / (Kmat @ (q / (Kmat.T @ u + 1e-300)) + 1e-300)
        v = q / (Kmat.T @ u + 1e-300)
        T = u[:, None] * Kmat * v[None, :]
    gw = float(np.sum((constC - 2 * (C1 @ T @ C2.T)) * T))
    return T, gw

def gw_match_acc(T, topk=(1, 5)):
    """With paired samples, how much transport mass lands near the true correspondence."""
    n = T.shape[0]
    order = np.argsort(-T, axis=1)
    return {f"top{k}": float(np.mean([i in order[i, :k] for i in range(n)])) for k in topk}

# ---------------------------------------------------------------- run per backbone
results = {"config": {"K": K, "stride": STRIDE, "N_GW": N_GW, "gw_eps": 0.05,
                      "action_repr": "net_motion_sum_over_chunk",
                      "note": "CLIP=clip-vit-l-14 768d joint; SigLIP2=siglip2-so400m 1152d joint. "
                              "libero_spatial agentview only, offline caches."},
           "backbones": {}}

for name, cdir in CACHES.items():
    Dz, Da = build_pairs(cdir)
    n = len(Dz)
    sel = rng.choice(n, size=min(N_GW, n), replace=False)
    Dzs, Das = Dz[sel], Da[sel]
    Cz, Ca = cos_cost(Dzs), cos_cost(Das)
    Tcoup, gw = entropic_gw(Cz, Ca)
    macc = gw_match_acc(Tcoup)
    # random control: GW of Dz-geometry vs an isotropic-Gaussian action-like cloud
    Rand = rng.standard_normal(Das.shape)
    _, gw_rand = entropic_gw(Cz, cos_cost(Rand))
    cka = linear_cka(Dz, Da)
    r2m, r2s = linear_align_r2(Dz, Da)
    mknn = mutual_knn(Dzs, Das, k=10)
    results["backbones"][name] = {
        "n_pairs": int(n), "dim_z": int(Dz.shape[1]),
        "gw_distance": round(gw, 5), "gw_random_control": round(gw_rand, 5),
        "gw_match_top1": round(macc["top1"], 4), "gw_match_top5": round(macc["top5"], 4),
        "gw_match_chance_top1": round(1.0 / len(sel), 4),
        "linear_cka_dz_da": round(float(cka), 4),
        "linear_align_r2_mean": round(r2m, 4), "linear_align_r2_std": round(r2s, 4),
        "mutual_knn10": round(mknn, 4),
    }
    print(f"[{name}] n={n} dimz={Dz.shape[1]} GW={gw:.4f} (rand {gw_rand:.4f}) "
          f"match@1={macc['top1']:.3f} CKA={cka:.3f} alignR2={r2m:.3f} mknn={mknn:.3f}")

json.dump(results, open(os.path.join(OUT, "gw_alignment.json"), "w"), indent=2)
print("wrote gw_alignment.json")
