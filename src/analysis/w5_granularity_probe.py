#!/usr/bin/env python3
"""W5 granularity diagnostic — GATED-ACCESS-FREE arms (offline probe, F4 go/no-go gate).

Tests the F4 hypothesis: do patch/dense representations improve FINE-action
prediction over pooled? Decompose LIBERO action chunks into COARSE (large EE
translation) vs FINE (gripper open/close + small EE delta), regress each from
candidate visual-displacement representations, report R2/MAE + a Delta-z
LINEARITY probe (EE pose delta from {pooled dz, pool(dF)} -> R2 + dCor).

Arms run here (no gated HF access):
  (1) CLIP ViT-L/14 pooled dz          (768d, cached flat npz key "Z")
  (2) SigLIP2-large-patch16-256 pooled dz   (1024d, MAP head, L2-normed)
  (3) SigLIP2-large-patch16-256 patch dF    (256 patches x 1024, same-index diff)
  (+) hierarchical pooled(+patch) — free bonus (F4 zeta_g + zeta_f design)

Deferred (gated / heavy download, NOT attempted): DINOv3 patch, SigLIP2(+)DINOv3,
RADIO-dense, dinov3-dinotxt, V-JEPA2.

Definitions (cowork COWORK_SigLIP_review_and_granularity_diagnostic + window2 D0):
  chunk over [t, t+span); a = actions[t:t+span] (span,7): 0:3 dPos, 3:6 dRot, 6 gripper.
  net_pos   = sum dPos  (3d)   -> COARSE target (large EE translation; ||net_pos|| = magnitude)
  net_rot   = sum dRot  (3d)
  path_len  = sum ||dPos_i||   (scalar total translational path)
  fine_res  = path_len - ||net_pos||  (scalar; back-and-forth / small translation)
  grip_mean = mean gripper (open/close signal, gripper is binary +/-1)
  FINE target    = [grip_mean, net_rot(3), fine_res]  (5d)
  EE-delta (R2 linearity probe target) = [net_pos(3), net_rot(3)]  (6d)

  Representation dz = z_{t+span} - z_t (pooled, L2-normed embeds).
  dF = tokens_{t+span} - tokens_t (per-patch, agentview static camera assumption).
    pool(dF)     = mean over patches (1024d)           <- R2-probe "pool(dF)" substrate
    patchnorm(dF)= ||dF_p||_2 per patch (256d)         <- spatial "where did it change"
    arm3 patch   = pool(dF) (+) patchnorm(dF) (1280d)  <- headline patch representation
"""
import argparse, glob, json, os, re, sys, time
import numpy as np

SPATIAL_ROOT_DEFAULT = "/workspace/clip_ws/data/libero/libero_spatial"
CLIP_CACHE_DEFAULT   = "/workspace/CLIP_ws/outputs/cache/libero_emb"
OUT_DEFAULT          = "/workspace/CLIP_ws/outputs/analysis/w5_granularity"
SG2_MODEL_DEFAULT    = "google/siglip2-large-patch16-256"


# ----------------------------- data / targets -----------------------------
def list_episodes(root):
    import h5py
    eps = []
    for f in sorted(glob.glob(os.path.join(root, "*.hdf5"))):
        with h5py.File(f, "r") as h:
            demos = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        eps += [(f, d) for d in demos]
    return eps


def subset_episodes(eps, n_target):
    """Spread the subset across tasks (round-robin over demos within each task)."""
    if n_target is None or n_target >= len(eps):
        return eps
    by_task = {}
    for f, d in eps:
        by_task.setdefault(f, []).append(d)
    tasks = list(by_task)
    out, i = [], 0
    while len(out) < n_target:
        t = tasks[i % len(tasks)]
        k = i // len(tasks)
        if k < len(by_task[t]):
            out.append((t, by_task[t][k]))
        i += 1
        if i > len(eps) * 2:
            break
    return out[:n_target]


def clip_cache_path(clip_cache, hdf5_path, demo, camera="agentview_rgb"):
    stem = os.path.basename(hdf5_path)[:-5]  # strip .hdf5
    return os.path.join(clip_cache, f"{stem}_{demo}_{camera}.npz")


def chunk_targets(actions, span, stride):
    """Return dict of per-chunk targets + chunk start indices."""
    T = actions.shape[0]
    starts = list(range(0, T - span, stride))
    coarse, fine, eedelta = [], [], []
    for t in starts:
        a = actions[t:t + span]
        net_pos = a[:, 0:3].sum(0)
        net_rot = a[:, 3:6].sum(0)
        path_len = np.linalg.norm(a[:, 0:3], axis=1).sum()
        fine_res = path_len - np.linalg.norm(net_pos)
        grip_mean = a[:, 6].mean()
        coarse.append(net_pos)
        fine.append(np.concatenate([[grip_mean], net_rot, [fine_res]]))
        eedelta.append(np.concatenate([net_pos, net_rot]))
    return (np.array(starts, dtype=int),
            np.asarray(coarse, np.float64),
            np.asarray(fine, np.float64),
            np.asarray(eedelta, np.float64))


# ----------------------------- SigLIP2 encoding -----------------------------
class Sg2Encoder:
    def __init__(self, model_id, device):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch = torch
        self.device = device
        dt = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = AutoModel.from_pretrained(model_id, dtype=dt).to(device).eval()
        self.proc = AutoProcessor.from_pretrained(model_id)
        # AUDIT FIX (2026-07-13): explicit no-crop for FOV parity with the dense encoders.
        # SigLIP2's default processor already does resize-256 with NO center crop
        # (n_patches=256=16² confirms full grid), so this is a documented NO-OP — arms 1/2/3
        # reproduce bit-identically and the cached SigLIP2 features stay valid.
        if hasattr(self.proc, "do_center_crop"):
            self.proc.do_center_crop = False
        self.dim = self.model.config.vision_config.hidden_size

    def encode(self, pil_images, bs=64):
        import torch
        pooled_all, tok_all = [], []
        for i in range(0, len(pil_images), bs):
            batch = pil_images[i:i + bs]
            inp = self.proc(images=batch, return_tensors="pt").to(self.device)
            px = inp["pixel_values"].to(self.model.dtype)
            with torch.no_grad():
                pf = self.model.get_image_features(pixel_values=px)
                pooled = pf if torch.is_tensor(pf) else pf.pooler_output
                vout = self.model.vision_model(pixel_values=px)
                tok = vout.last_hidden_state          # (B, P, D) raw
            pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)  # match CLIP joint L2
            pooled_all.append(pooled.cpu().numpy().astype(np.float32))
            tok_all.append(tok.float().cpu().numpy().astype(np.float32))
        return np.concatenate(pooled_all), np.concatenate(tok_all)


# ----------------------------- DINOv2 encoding (fusion-PROXY, interim) -----------------------------
class DinoEncoder:
    """DINOv2/DINOv3 patch-token encoder. Returns per-patch tokens with the CLS token AND any
    register/storage tokens dropped (prefix = 1 CLS + num_register_tokens). Raw (geometric)
    tokens, matching the SigLIP2 ΔF convention (no L2-norm).

    DINOv2-large is the ungated INTERIM PROXY (patch14, default 224 -> 16x16 grid, NOT pixel-grid
    aligned to SigLIP2-large-256). DINOv3-L/16 is the TARGET: with force_size=256 it yields a
    16x16@256 patch grid pixel-ALIGNED to SigLIP2-large-patch16-256, and carries 4 register tokens.
    """
    def __init__(self, model_id, device, force_size=None, fp32=False):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch = torch
        self.device = device
        self.force_size = force_size
        # DINOv3 produces NaN patch tokens under fp16 (verified) -> allow fp32 opt-in.
        dt = torch.float32 if (fp32 or not device.startswith("cuda")) else torch.float16
        self.model = AutoModel.from_pretrained(model_id, dtype=dt).to(device).eval()
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.dim = self.model.config.hidden_size
        self.patch = self.model.config.patch_size
        self.n_reg = int(getattr(self.model.config, "num_register_tokens", 0) or 0)
        self.n_prefix = 1 + self.n_reg               # CLS + register/storage tokens
        # AUDIT FIX (2026-07-13): ALWAYS disable center-crop so every dense encoder sees the
        # SAME full field-of-view (mirror Dinov2Anchor in src/core/anchor.py). This removes the
        # DINOv2(default→256 resize + 224 center-crop = zoomed 87.5% FOV) vs DINOv3(full frame)
        # FOV mismatch that confounded the DINO-vs-DINO comparison.
        if hasattr(self.proc, "do_center_crop"):
            self.proc.do_center_crop = False
        if force_size is not None:                   # force square resize (no crop), full frame
            sz = {"height": int(force_size), "width": int(force_size)}
            try:
                self.proc.size = sz
            except Exception:
                pass
            if hasattr(self.proc, "do_resize"):
                self.proc.do_resize = True
            self._call_kwargs = {"size": sz}
        else:
            self._call_kwargs = {}

    def encode_tokens(self, pil_images, bs=16):     # bs=16 (was 64): OOM safety at 512/518 dense res
        import torch
        tok_all = []
        for i in range(0, len(pil_images), bs):
            batch = pil_images[i:i + bs]
            inp = self.proc(images=batch, return_tensors="pt", **self._call_kwargs).to(self.device)
            px = inp["pixel_values"].to(self.model.dtype)
            with torch.no_grad():
                out = self.model(pixel_values=px)
            tok = out.last_hidden_state[:, self.n_prefix:, :]   # drop CLS+registers -> (B,P,D)
            tok_all.append(tok.float().cpu().numpy().astype(np.float32))
        return np.concatenate(tok_all)


class RadioEncoder:
    """C-RADIOv4-SO400M backbone DENSE encoder (off-the-shelf agglomerative fusion of
    SigLIP2 + DINOv3 + SAM teachers). Uses the BACKBONE spatial features (dim 1536), NOT an
    adaptor summary — arm6 is a dense ΔF probe. Loaded via the validated torch.hub recipe
    (src/core/anchor.py RadioAnchor): no adaptor, [0,1] pixels (RADIO's own conditioner
    normalizes), full-frame resize to the nearest supported resolution of FIX_RES."""
    FIX_RES = 512

    def __init__(self, device, version="c-radio_v4-so400m", fix_res=None):
        import torch
        self.torch = torch
        self.device = device
        # AUDIT FIX (2026-07-13): resolution is now configurable so RADIO can be run at 256
        # (→256 patches, matched to the 256-native manual arms) for the fair RADIO-vs-manual
        # comparison, in addition to the existing 512 (→1024 patches).
        fix_res = self.FIX_RES if fix_res is None else int(fix_res)
        self.model = torch.hub.load(
            "NVlabs/RADIO", "radio_model", version=version,
            progress=True, skip_validation=True, trust_repo=True,
        ).to(device).eval()
        self.res = self.model.get_nearest_supported_resolution(fix_res, fix_res)[0]
        self.patch = getattr(self.model, "patch_size", 16)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.res, self.res, device=device)
            summary, feats = self.model(dummy)
        self.dim = int(feats.shape[-1])
        self.n_patches = int(feats.shape[1])

    def _preprocess(self, pil_images):
        import torch, torch.nn.functional as F
        px = []
        for im in pil_images:
            a = torch.from_numpy(np.asarray(im.convert("RGB")).copy()).permute(2, 0, 1).float() / 255.0
            a = F.interpolate(a.unsqueeze(0), size=(self.res, self.res),
                              mode="bilinear", align_corners=False, antialias=True)
            px.append(a)
        return torch.cat(px, 0).to(self.device)

    def encode_tokens(self, pil_images, bs=16):
        import torch
        tok_all = []
        for i in range(0, len(pil_images), bs):
            batch = pil_images[i:i + bs]
            with torch.no_grad():
                _summary, feats = self.model(self._preprocess(batch))   # feats (B, N, 1536)
            tok_all.append(feats.float().cpu().numpy().astype(np.float32))
        return np.concatenate(tok_all)


def dino_episode_features(ep, span, stride, clip_cache, dino_enc, feat_cache_dir):
    """DINOv2 patch-ΔF summary per episode, aligned to the SAME chunks as SigLIP2
    (identical T truncation + chunk_targets starts). Returns dict or None.
    dino_meanDF = mean over patches of ΔF (D); dino_patchnorm = per-patch ‖ΔF‖₂ (P)."""
    import h5py
    hdf5_path, demo = ep
    stem = os.path.basename(hdf5_path)[:-5]
    fc = os.path.join(feat_cache_dir, f"{stem}__{demo}__s{span}_t{stride}.npz")
    if os.path.exists(fc):
        d = np.load(fc)
        return {k: d[k] for k in d.files}

    cpath = clip_cache_path(clip_cache, hdf5_path, demo)
    if not os.path.exists(cpath):
        return None
    clip_z = np.load(cpath)["Z"].astype(np.float32)
    with h5py.File(hdf5_path, "r") as h:
        actions = h[f"data/{demo}/actions"][:].astype(np.float64)
        frames = h[f"data/{demo}/obs/agentview_rgb"][:]
    T = min(len(clip_z), len(actions), len(frames))
    actions, frames = actions[:T], frames[:T]
    starts, _, _, _ = chunk_targets(actions, span, stride)
    if len(starts) == 0:
        return None

    from PIL import Image
    pil = [Image.fromarray(frames[i]) for i in range(T)]
    tok = dino_enc.encode_tokens(pil)                    # (T, P, D)
    P = tok.shape[1]
    ends = starts + span
    dF = tok[ends] - tok[starts]                         # (n, P, D) raw same-index diff
    meanDF = dF.mean(axis=1)                             # (n, D)  pool(ΔF)
    patchnorm = np.linalg.norm(dF, axis=2)               # (n, P)  spatial
    out = dict(dino_meanDF=meanDF.astype(np.float32),
               dino_patchnorm=patchnorm.astype(np.float32),
               n_patches=np.array([P]),
               n_chunks=np.array([len(starts)]))
    os.makedirs(feat_cache_dir, exist_ok=True)
    np.savez_compressed(fc, **out)
    return out


def episode_features(ep, span, stride, clip_cache, sg2_enc, feat_cache_dir):
    """Compute (or load cached) per-episode chunk features + targets. Returns dict or None."""
    import h5py
    hdf5_path, demo = ep
    stem = os.path.basename(hdf5_path)[:-5]
    fc = os.path.join(feat_cache_dir, f"{stem}__{demo}__s{span}_t{stride}.npz")
    if os.path.exists(fc):
        d = np.load(fc)
        return {k: d[k] for k in d.files}

    cpath = clip_cache_path(clip_cache, hdf5_path, demo)
    if not os.path.exists(cpath):
        return None
    clip_z = np.load(cpath)["Z"].astype(np.float32)          # (T,768) normed CLIP joint
    with h5py.File(hdf5_path, "r") as h:
        actions = h[f"data/{demo}/actions"][:].astype(np.float64)
        frames = h[f"data/{demo}/obs/agentview_rgb"][:]       # (T,H,W,3) uint8
    T = min(len(clip_z), len(actions), len(frames))
    clip_z, actions, frames = clip_z[:T], actions[:T], frames[:T]

    starts, coarse, fine, eedelta = chunk_targets(actions, span, stride)
    if len(starts) == 0:
        return None

    from PIL import Image
    pil = [Image.fromarray(frames[i]) for i in range(T)]
    sg2_pooled, sg2_tok = sg2_enc.encode(pil)                 # (T,1024),(T,P,1024)
    P = sg2_tok.shape[1]

    ends = starts + span
    d_clip = clip_z[ends] - clip_z[starts]                    # (n,768)
    d_sg2 = sg2_pooled[ends] - sg2_pooled[starts]             # (n,1024)
    dF = sg2_tok[ends] - sg2_tok[starts]                      # (n,P,1024)
    meanDF = dF.mean(axis=1)                                  # (n,1024) pool(dF)
    patchnorm = np.linalg.norm(dF, axis=2)                    # (n,P) spatial

    out = dict(d_clip=d_clip.astype(np.float32),
               d_sg2=d_sg2.astype(np.float32),
               meanDF=meanDF.astype(np.float32),
               patchnorm=patchnorm.astype(np.float32),
               coarse=coarse.astype(np.float32),
               fine=fine.astype(np.float32),
               eedelta=eedelta.astype(np.float32),
               n_patches=np.array([P]))
    os.makedirs(feat_cache_dir, exist_ok=True)
    np.savez_compressed(fc, **out)
    return out


# ----------------------------- regression / metrics -----------------------------
def eval_arm(Xtr, Xte, ytr, yte):
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_absolute_error
    sx = StandardScaler().fit(Xtr)
    Xtr, Xte = sx.transform(Xtr), sx.transform(Xte)
    alphas = np.logspace(-2, 5, 12)
    reg = RidgeCV(alphas=alphas).fit(Xtr, ytr)
    pred = reg.predict(Xte)
    if pred.ndim == 1:
        pred = pred[:, None]
    r2_macro = r2_score(yte, pred, multioutput="uniform_average")
    r2_per = r2_score(yte, pred, multioutput="raw_values")
    mae = mean_absolute_error(yte, pred)
    return dict(r2_macro=float(r2_macro),
                r2_per=[float(v) for v in np.atleast_1d(r2_per)],
                mae=float(mae), alpha=float(reg.alpha_), dim=int(Xtr.shape[1]))


def distance_correlation(X, Y, cap=2000, seed=0):
    """Biased sample distance correlation between two matrices (rows = samples)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n > cap:
        idx = rng.choice(n, cap, replace=False)
        X, Y = X[idx], Y[idx]
    def dcov_stat(A):
        from scipy.spatial.distance import pdist, squareform
        D = squareform(pdist(A))
        D = D - D.mean(0, keepdims=True) - D.mean(1, keepdims=True) + D.mean()
        return D
    A, B = dcov_stat(X), dcov_stat(Y)
    dcov2 = (A * B).mean()
    dvarx = (A * A).mean()
    dvary = (B * B).mean()
    denom = np.sqrt(dvarx * dvary)
    return float(np.sqrt(max(dcov2, 0.0)) / np.sqrt(denom)) if denom > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_episodes", type=int, default=180)
    ap.add_argument("--span", type=int, default=16)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--gpu", type=str, default="8")
    ap.add_argument("--spatial_root", default=SPATIAL_ROOT_DEFAULT)
    ap.add_argument("--clip_cache", default=CLIP_CACHE_DEFAULT)
    ap.add_argument("--sg2_model", default=SG2_MODEL_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    # fusion-PROXY (DINOv2, interim) arms — off by default so the verified gated-free run is unchanged
    ap.add_argument("--with_dino", action="store_true",
                    help="add arm4-proxy (DINOv2 patch ΔF) + arm5-proxy (SigLIP2⊕DINOv2) + capacity control")
    ap.add_argument("--dino_model", default="facebook/dinov2-large")
    ap.add_argument("--dino_size", type=int, default=518,
                    help="AUDIT FIX: force square no-crop resize for DINOv2 (patch14). 518/14=37x37="
                         "1369 patches — high dense res for the fair DINO-vs-DINO comparison at 512.")
    # TARGET fusion partner (DINOv3-L/16, grid-aligned). Run SEPARATELY from --with_dino (one dense
    # model on GPU at a time; OOM discipline). Produces arm4 (DINOv3 patch ΔF) + arm5 (SigLIP2⊕DINOv3)
    # + capacity control, with arm names tagged _v3.
    ap.add_argument("--with_dinov3", action="store_true",
                    help="add arm4 (DINOv3-L/16 patch ΔF) + arm5 (SigLIP2⊕DINOv3, grid-aligned) + capacity control")
    ap.add_argument("--dinov3_model", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    ap.add_argument("--dinov3_size", type=int, default=256,
                    help="force square resize for grid alignment with SigLIP2-large-256 (256/16=16x16=256 patches)")
    # off-the-shelf agglomerative fusion (C-RADIOv4). Run SEPARATELY (its own process). arm6 dense ΔF.
    ap.add_argument("--with_radio", action="store_true",
                    help="add arm6 (RADIO-dense ΔF; off-the-shelf SigLIP2+DINOv3+SAM backbone fusion, dim 1536)")
    ap.add_argument("--radio_version", default="c-radio_v4-so400m")
    ap.add_argument("--radio_res", type=int, default=512,
                    help="AUDIT FIX: RADIO target resolution (nearest supported). 256->256 patches "
                         "(matched to 256-native manual arms, the fair comparison); 512->1024 (existing).")
    ap.add_argument("--dino_cache", default="/data2/clip_ws_cache/dense",
                    help="HDD dense-feature cache root (keep off the near-full SSD)")
    ap.add_argument("--out_name", default=None,
                    help="results json filename; default depends on which dense arm(s) are on")
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    import torch
    cuda_ok = torch.cuda.is_available()
    print(f"[env] torch.cuda.is_available()={cuda_ok} device_count={torch.cuda.device_count()}")
    if not cuda_ok:
        print("[FATAL] torch.cuda is False — GPU lost. Reporting and exiting (watchdog restarts).")
        sys.exit(2)
    device = "cuda"

    os.makedirs(args.out, exist_ok=True)
    feat_cache = os.path.join(args.out, "feat_cache", os.path.basename(args.sg2_model))

    eps_all = list_episodes(args.spatial_root)
    eps = subset_episodes(eps_all, args.n_episodes)
    print(f"[data] {len(eps_all)} episodes total; using subset of {len(eps)} "
          f"across {len(set(e[0] for e in eps))} tasks; span={args.span} stride={args.stride}")

    t0 = time.time()
    sg2 = Sg2Encoder(args.sg2_model, device)
    print(f"[sg2] loaded {args.sg2_model} dim={sg2.dim} in {time.time()-t0:.1f}s")

    # ----- dense partner / off-the-shelf model(s). Load one dense model at a time in practice
    # (invoke one --with_* flag per process; OOM discipline). Each job = its own HDD feature cache. -----
    dense_jobs = {}   # tag -> {enc, cache, kind, model_id}
    if args.with_dino:
        td = time.time()
        enc = DinoEncoder(args.dino_model, device, force_size=args.dino_size)
        cache = os.path.join(args.dino_cache, os.path.basename(args.dino_model),
                             f"s{args.span}_t{args.stride}_sz{args.dino_size}")
        print(f"[dino] loaded {args.dino_model} (no-crop@{args.dino_size}) dim={enc.dim} patch={enc.patch} "
              f"prefix={enc.n_prefix} in {time.time()-td:.1f}s; cache={cache}")
        dense_jobs["DINOv2_proxy"] = dict(enc=enc, cache=cache, kind="fusion", model_id=args.dino_model)
    if args.with_dinov3:
        td = time.time()
        enc = DinoEncoder(args.dinov3_model, device, force_size=args.dinov3_size, fp32=True)
        cache = os.path.join(args.dino_cache, os.path.basename(args.dinov3_model),
                             f"s{args.span}_t{args.stride}_sz{args.dinov3_size}")
        print(f"[dinov3] loaded {args.dinov3_model} (TARGET grid-aligned) dim={enc.dim} patch={enc.patch} "
              f"reg={enc.n_reg} prefix={enc.n_prefix} force_size={args.dinov3_size} in {time.time()-td:.1f}s; cache={cache}")
        dense_jobs["DINOv3"] = dict(enc=enc, cache=cache, kind="fusion", model_id=args.dinov3_model)
    if args.with_radio:
        td = time.time()
        enc = RadioEncoder(device, version=args.radio_version, fix_res=args.radio_res)
        cache = os.path.join(args.dino_cache, os.path.basename(args.radio_version),
                             f"s{args.span}_t{args.stride}_r{enc.res}")
        print(f"[radio] loaded {args.radio_version} (off-the-shelf) dim={enc.dim} res={enc.res} "
              f"n_patches={enc.n_patches} patch={enc.patch} in {time.time()-td:.1f}s; cache={cache}")
        dense_jobs["RADIO"] = dict(enc=enc, cache=cache, kind="radio", model_id=args.radio_version)

    feats, ep_ids, kept = [], [], 0
    dfeats = {tag: [] for tag in dense_jobs}
    for j, ep in enumerate(eps):
        fdict = episode_features(ep, args.span, args.stride, args.clip_cache, sg2, feat_cache)
        if fdict is None:
            print(f"  [skip] no clip cache / too short: {os.path.basename(ep[0])} {ep[1]}")
            continue
        ddicts, skip = {}, False
        for tag, job in dense_jobs.items():
            dd = dino_episode_features(ep, args.span, args.stride, args.clip_cache, job["enc"], job["cache"])
            if dd is None:
                print(f"  [skip] {tag} none (kept consistent): {os.path.basename(ep[0])} {ep[1]}")
                skip = True
                break
            assert int(dd["n_chunks"][0]) == len(fdict["coarse"]), (
                f"chunk misalignment sg2={len(fdict['coarse'])} {tag}={int(dd['n_chunks'][0])}")
            ddicts[tag] = dd
        if skip:
            continue
        feats.append(fdict)
        for tag in dense_jobs:
            dfeats[tag].append(ddicts[tag])
        ep_ids.append(kept)
        kept += 1
        if (j + 1) % 20 == 0:
            print(f"  [enc] {j+1}/{len(eps)} episodes ({time.time()-t0:.1f}s)")

    P = int(feats[0]["n_patches"][0])
    print(f"[feat] {kept} episodes with features; patches P={P}")

    # episode-level train/test split (no chunk leakage)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(kept)
    n_test = max(1, int(round(kept * args.test_frac)))
    test_eps = set(order[:n_test].tolist())
    tr_mask, te_mask = [], []

    def stack(key):
        return np.concatenate([f[key] for f in feats], axis=0)

    # build sample-level train/test mask
    for i, f in enumerate(feats):
        m = np.full(len(f["coarse"]), i in test_eps)
        te_mask.append(m); tr_mask.append(~m)
    te_mask = np.concatenate(te_mask); tr_mask = np.concatenate(tr_mask)

    d_clip = stack("d_clip"); d_sg2 = stack("d_sg2")
    meanDF = stack("meanDF"); patchnorm = stack("patchnorm")
    y_coarse = stack("coarse"); y_fine = stack("fine"); y_ee = stack("eedelta")
    patch_full = np.concatenate([meanDF, patchnorm], axis=1)  # arm3 headline (1280)
    hier = np.concatenate([d_sg2, patchnorm], axis=1)         # F4 zeta_g(+)zeta_f
    print(f"[split] train samples={tr_mask.sum()} test samples={te_mask.sum()} "
          f"(test episodes={n_test}/{kept})")

    arms = {
        "1_CLIP_pooled_dz":        d_clip,
        "2_SigLIP2_pooled_dz":     d_sg2,
        "3_SigLIP2_patch_dF":      patch_full,
        "3a_SigLIP2_meanDF":       meanDF,
        "3b_SigLIP2_patchnorm":    patchnorm,
        "H_hier_pooled+patchnorm": hier,
    }

    # arm naming: DINOv2 keeps the verified "proxy" labels; DINOv3/RADIO get their own tags.
    ARM_TAGS = {
        "DINOv2_proxy": ("4proxy_DINOv2_patch_dF", "5proxy_SigLIP2+DINOv2_patch_dF", "ctrl_SigLIP2_dimmatched"),
        "DINOv3":       ("4_DINOv3_patch_dF",      "5_SigLIP2+DINOv3_patch_dF",      "ctrl_SigLIP2_dimmatched_v3"),
    }
    dense_n_patches = {}   # tag -> Pd (patch count of the dense partner)
    for tag, job in dense_jobs.items():
        dl = dfeats[tag]
        dmeanDF = np.concatenate([d["dino_meanDF"] for d in dl], axis=0)
        dpatchnorm = np.concatenate([d["dino_patchnorm"] for d in dl], axis=0)
        assert dmeanDF.shape[0] == patch_full.shape[0], f"{tag}/sg2 row misalignment"
        Pd = int(dl[0]["n_patches"][0])
        dense_n_patches[tag] = Pd
        dense_patch = np.concatenate([dmeanDF, dpatchnorm], axis=1)   # pool(ΔF)⊕patchnorm(ΔF)
        if job["kind"] == "radio":
            arms["6_RADIO_dense_dF"] = dense_patch
            print(f"[radio] Pd={Pd} radio_dense dim={dense_patch.shape[1]}")
            continue
        # fusion partner (dinov2 / dinov3): arm4 (partner alone), arm5 (SigLIP2 ⊕ partner), capacity ctrl
        fusion = np.concatenate([patch_full, dense_patch], axis=1)
        # Capacity control: SigLIP2-patch inflated to the SAME total dim as arm5 using SigLIP2
        # information ONLY (seeded Gaussian random projection of arm3 -> partner dim). Isolates the
        # "fusion gain is a dimensionality artifact" concern: any arm5 gain over this control is
        # information the partner carries that SigLIP2 does not.
        rp = np.random.default_rng(12345).standard_normal(
            (patch_full.shape[1], dense_patch.shape[1])).astype(np.float32)
        rp /= np.sqrt(patch_full.shape[1])
        sg2_dimmatched = np.concatenate([patch_full, patch_full @ rp], axis=1)
        a4, a5, actrl = ARM_TAGS[tag]
        arms[a4] = dense_patch
        arms[a5] = fusion
        arms[actrl] = sg2_dimmatched
        print(f"[{tag}] Pd={Pd} partner dim={dense_patch.shape[1]} "
              f"fusion dim={fusion.shape[1]} ctrl dim={sg2_dimmatched.shape[1]}")
    Pd = dense_n_patches.get("DINOv2_proxy") or dense_n_patches.get("DINOv3")
    fine_cols = ["gripper", "netRot_x", "netRot_y", "netRot_z", "fine_res"]
    coarse_cols = ["netPos_x", "netPos_y", "netPos_z"]

    results = {"config": vars(args), "n_episodes_used": kept, "n_patches": P,
               "dino_n_patches": Pd, "dense_n_patches": dense_n_patches,
               "dense_models": {t: j["model_id"] for t, j in dense_jobs.items()},
               "n_train": int(tr_mask.sum()), "n_test": int(te_mask.sum()),
               "n_test_episodes": n_test, "fine_cols": fine_cols,
               "coarse_cols": coarse_cols, "arms": {}}

    for name, X in arms.items():
        Xtr, Xte = X[tr_mask], X[te_mask]
        rc = eval_arm(Xtr, Xte, y_coarse[tr_mask], y_coarse[te_mask])
        rf = eval_arm(Xtr, Xte, y_fine[tr_mask], y_fine[te_mask])
        results["arms"][name] = {"coarse": rc, "fine": rf}
        print(f"[arm {name:26s} dim={X.shape[1]:5d}] "
              f"coarse R2={rc['r2_macro']:+.3f} MAE={rc['mae']:.4f} | "
              f"fine R2={rf['r2_macro']:+.3f} MAE={rf['mae']:.4f} "
              f"(gripR2={rf['r2_per'][0]:+.3f})")

    # R2 linearity probe: EE delta (pos/rot, 6d) from {pooled dz, pool(dF)}
    print("[R2 linearity probe] EE pose-delta (6d) <- {pooled dz, pool(dF)}")
    lin = {}
    lin_arms = {
        "CLIP_pooled_dz": d_clip,
        "SigLIP2_pooled_dz": d_sg2,
        "SigLIP2_pool(dF)": meanDF,
        "SigLIP2_patchnorm(dF)": patchnorm,
    }
    for name, X in lin_arms.items():
        r = eval_arm(X[tr_mask], X[te_mask], y_ee[tr_mask], y_ee[te_mask])
        dcor = distance_correlation(X[te_mask], y_ee[te_mask])
        lin[name] = {"r2_macro": r["r2_macro"], "r2_per": r["r2_per"],
                     "mae": r["mae"], "dcor": dcor, "dim": r["dim"]}
        print(f"  {name:24s} R2={r['r2_macro']:+.3f} dCor={dcor:.3f} MAE={r['mae']:.4f}")
    results["linearity_probe_R2"] = lin

    if args.out_name:
        out_name = args.out_name
    elif args.with_dinov3:
        out_name = "w5_granularity_dinov3_results.json"
    elif args.with_radio:
        out_name = "w5_granularity_radio_results.json"
    elif args.with_dino:
        out_name = "w5_granularity_fusion_proxy_results.json"
    else:
        out_name = "w5_granularity_results.json"
    outjson = os.path.join(args.out, out_name)
    with open(outjson, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[done] wrote {outjson} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
