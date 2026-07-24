"""R-series retrieval port — shared module (R-0/R-1).

Port of the colleague lab's retrieval-based language control onto OUR substrate
(docs/ANALYSIS_colleague_retrieval_control.md §5). READ-ONLY w.r.t. their repo:
the SemanticAdapter / SupCon code below is a verbatim port of
SigLIP/src/models/semantic_adapter.py (colleague reference, unmodified there);
segment extraction mirrors SigLIP/src/training/train_semantic_adapter.py and
SigLIP/lang_adapter_wow/retrieval_videos/build_effect_bank.py.

OUR substrate:
  phase1  = checkpoints/phase1_libero_siglip2_large256.pt  (the phase1 behind
            phase2_libero_large256_matchedbase — verified via its config)
  zeta    = g(resample16(A)_actnorm, z_t)   z_t = SigLIP2-large256 pooled RAW
            (normalize=false, native-best) at segment start, cache libero_emb_large256
  dz      = z[end] - z[start]  same cache (single-stream: visual target == state cache;
            colleague used a separate SigLIP-only cache vs dualavg z_t — we have one stream)
  text    = SigLIP2-large256 text tower via src/core/anchor.py (same normalize=false)

Split: BY-EPISODE RandomState(seed=2).permutation over sorted episode files,
val = perm[:round(N*0.2)] — reproduces the colleague adapter split exactly
(their configs/sem_adapter_heldout.yaml seed=2, same 500 LIBERO-spatial demos,
same sorted-glob episode order) AND our phase2-winner split convention.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WS = Path(os.environ.get("RSERIES_WS", "/workspace/CLIP_ws"))
RS = WS / "scratchpad" / "rseries"
OUT = WS / "outputs" / "rseries"
CACHE_Z = Path("/data2/clip_ws_cache/cache/libero_emb_large256/siglip2-so400m/joint/raw")
HDF5_ROOT = WS / "data" / "libero" / "libero_spatial"
PHASE1_CKPT = WS / "checkpoints" / "phase1_libero_siglip2_large256.pt"
SPLIT_SEED = 2          # colleague sem_adapter_heldout.yaml (== our phase2 winner split)
VAL_FRAC = 0.2
CAMERA = "agentview_rgb"

sys.path.insert(0, str(WS / "src"))


# ---------------------------------------------------------------------------
# SemanticAdapter + SupCon — verbatim port (colleague semantic_adapter.py)
# ---------------------------------------------------------------------------
class Projector(nn.Module):
    """Linear(in_dim -> out) + LayerNorm + L2-normalize.  ~0.26M params."""

    def __init__(self, in_dim=1024, out_dim=256):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, x):
        return F.normalize(self.ln(self.lin(x)), dim=-1)


class SemanticAdapter(nn.Module):
    def __init__(self, in_dim=1024, out_dim=256):
        super().__init__()
        self.P_action = Projector(in_dim, out_dim)
        self.P_visual = Projector(in_dim, out_dim)
        self.P_text = Projector(in_dim, out_dim)


def supcon_crossmodal(anchors, cands, lab_a, lab_c, temp=0.07,
                      hard=None, hard_mask=None, self_pair=False):
    """Multi-positive SupCon (Khosla). Hard negatives enter the denominator only."""
    logits = anchors @ cands.t() / temp
    pos = lab_a[:, None] == lab_c[None, :]
    if self_pair:
        eye = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
        pos = pos & ~eye
        logits = logits.masked_fill(eye, float("-inf"))
    denom_terms = [logits]
    if hard is not None and hard.shape[1] > 0:
        hlog = torch.einsum("ad,ahd->ah", anchors, hard) / temp
        if hard_mask is not None:
            hlog = hlog.masked_fill(hard_mask <= 0, float("-inf"))
        denom_terms.append(hlog)
    denom = torch.logsumexp(torch.cat(denom_terms, dim=1), dim=1, keepdim=True)
    log_prob = logits - denom
    pos_cnt = pos.sum(1)
    valid = pos_cnt > 0
    if not valid.any():
        return anchors.new_zeros(())
    lp = torch.where(pos, log_prob, torch.zeros_like(log_prob))
    per = -(lp.sum(1))[valid] / pos_cnt[valid].clamp(min=1)
    return per.mean()


def supcon_symmetric(qx, qy, lab, temp=0.07):
    return 0.5 * (supcon_crossmodal(qx, qy, lab, lab, temp)
                  + supcon_crossmodal(qy, qx, lab, lab, temp))


# ---------------------------------------------------------------------------
# split / phase1 / preprocessing
# ---------------------------------------------------------------------------
def episode_split():
    """(train_keys, heldout_keys) — '<task_stem>_<demo>' keys, colleague-exact."""
    import h5py
    files = []
    for f in sorted(HDF5_ROOT.glob("*.hdf5")):
        with h5py.File(f, "r") as h:
            demos = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        files += [(f, k) for k in demos]
    perm = np.random.RandomState(SPLIT_SEED).permutation(len(files))
    n_val = max(1, round(len(files) * VAL_FRAC))
    key = lambda ep: f"{ep[0].stem}_{ep[1]}"
    heldout = {key(files[i]) for i in perm[:n_val]}
    train = {key(files[i]) for i in perm[n_val:]}
    return train, heldout


def load_phase1():
    """(ae, ck) — our DeltaAE, eval + frozen."""
    from models.networks import DeltaAE
    ck = torch.load(PHASE1_CKPT, map_location="cpu", weights_only=False)
    m = ck["config"]["model"]
    ae = DeltaAE(ck["action_dim"], ck["n_chunk"], ck["latent_dim"], m["hidden"],
                 m["layers"], m.get("dropout", 0.0), m.get("state_cond", True)).eval()
    ae.load_state_dict(ck["state_dict"])
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae, ck


def resample_chunk(seg, n_chunk):
    src = np.linspace(0, len(seg) - 1, n_chunk)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, len(seg) - 1)
    w = (src - lo)[:, None]
    return seg[lo] * (1 - w) + seg[hi] * w


def preprocess_chunk(seg, ck):
    """raw (L,7) -> our g input (n_chunk, 7): resample -> actnorm -> chunk_repr."""
    from core import chunkrep
    ch = resample_chunk(seg, ck["n_chunk"])
    ch = (ch - ck["a_mean"]) / ck["a_std"]
    return chunkrep.to_repr(ch.astype(np.float32), ck.get("chunk_repr", "time"))


# ---------------------------------------------------------------------------
# text embeddings — OUR anchor (SigLIP2-large256 tower, normalize=false), memoized
# ---------------------------------------------------------------------------
TEXT_CACHE = OUT / "text_emb_cache.npz"


def text_embeddings(phrases):
    """{phrase: (1024,) float32 RAW (unnormalized — our substrate convention)}."""
    phrases = sorted(set(phrases))
    cache = {}
    if TEXT_CACHE.exists():
        z = np.load(TEXT_CACHE, allow_pickle=True)
        cache = {k: e for k, e in zip(list(z["phrases"]), z["emb"])}
    missing = [p for p in phrases if p not in cache]
    if missing:
        import yaml
        from core.anchor import get_anchor
        cfg = yaml.safe_load(open(WS / "configs/phase1_libero_siglip2_large256.yaml"))
        anchor = get_anchor(cfg)          # Siglip2Anchor large256, joint, normalize=false
        print(f"[text] encoding {len(missing)} phrases with {anchor.cache_key}")
        emb = []
        for i in range(0, len(missing), 64):
            emb.append(anchor.encode_texts(missing[i:i + 64])["embeds"])
        emb = np.concatenate(emb).astype(np.float32)
        for p, e in zip(missing, emb):
            cache[p] = e
        allk = sorted(cache)
        TEXT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(TEXT_CACHE, phrases=np.array(allk, object),
                 emb=np.stack([cache[k] for k in allk]).astype(np.float32))
    return cache


# ---------------------------------------------------------------------------
# template sets
# ---------------------------------------------------------------------------
def load_banks():
    bank = json.load(open(RS / "banks/subgoal_phrases_libero_spatial.json"))
    hard = json.load(open(RS / "banks/hard_negative_bank.json"))["phrases"]
    their = json.load(open(RS / "banks/their_templates.json"))
    return bank, hard, their


# OUR independent 3rd template set (G-R0c) — written fresh for this port; checked
# for zero phrase-level overlap with their canonical / para_bank(3) / para_aug(9)
# / gen_seen(6) / unseen(2) / novel3(2) sets.
THIRD_SET_OURS = {
    "approach the black bowl": [
        "guide the arm toward the black bowl",
        "home in on the black bowl",
        "travel to where the black bowl sits"],
    "grasp the black bowl": [
        "clench the gripper around the black bowl",
        "capture the black bowl with the gripper",
        "wrap the gripper around the black bowl and hold it"],
    "move the black bowl backward": [
        "bring the black bowl toward the rear of the table",
        "guide the black bowl backward",
        "scoot the black bowl backward a little"],
    "move the black bowl forward": [
        "bring the black bowl toward the front of the table",
        "guide the black bowl forward",
        "scoot the black bowl forward a little"],
    "move the black bowl left": [
        "bring the black bowl toward the left side",
        "guide the black bowl to the left",
        "scoot the black bowl leftward a little"],
    "move the black bowl right": [
        "bring the black bowl toward the right side",
        "guide the black bowl to the right",
        "scoot the black bowl rightward a little"],
    "place the black bowl on the plate": [
        "seat the black bowl on the plate",
        "deliver the black bowl onto the plate",
        "bring the black bowl down onto the plate"],
    "release the black bowl": [
        "withdraw the gripper from the black bowl",
        "unhand the black bowl",
        "ease the gripper open and leave the black bowl"],
}


# ---------------------------------------------------------------------------
# eval helpers
# ---------------------------------------------------------------------------
def l2n(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def classify(q_a, txtmap, phrase_sets, classes, adapter_Ptext):
    """Nearest-phrase classification of held-out q_a against a template set.

    phrase_sets: {canonical_class: [phrase, ...]}. Returns per-set dict with
    top1 acc, per-class acc, confusion (pred class per max-cos phrase).
    q_a: (N,256) L2. adapter_Ptext: torch module. Scoring rule: class score =
    MAX cos over that class's phrases (nearest phrase); 'mean' variant reported too.
    """
    cls_id = {c: i for i, c in enumerate(classes)}
    texts, tlab = [], []
    for c, ps in phrase_sets.items():
        for p in ps:
            texts.append(p); tlab.append(cls_id[c])
    tlab = np.array(tlab)
    with torch.no_grad():
        q_l = adapter_Ptext(torch.tensor(
            np.stack([txtmap[t] for t in texts]))).numpy()
    S = q_a @ q_l.T                                        # (N, n_texts)
    n_cls = len(classes)
    smax = np.full((len(q_a), n_cls), -np.inf, np.float32)
    smean = np.zeros((len(q_a), n_cls), np.float32)
    for c in range(n_cls):
        m = tlab == c
        smax[:, c] = S[:, m].max(1)
        smean[:, c] = S[:, m].mean(1)
    return smax, smean


def acc_report(smax, lab, classes):
    pred = smax.argmax(1)
    top1 = float((pred == lab).mean())
    per, conf = {}, np.zeros((len(classes), len(classes)), int)
    for i in range(len(classes)):
        m = lab == i
        per[classes[i]] = float((pred[m] == i).mean()) if m.sum() else None
        for j in range(len(classes)):
            conf[i, j] = int(((lab == i) & (pred == j)).sum())
    return top1, per, conf.tolist()
