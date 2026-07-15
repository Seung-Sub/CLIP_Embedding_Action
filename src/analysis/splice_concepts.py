"""SpLiCE-style sparse concept decomposition (NeurIPS'24, 2402.10376).

An image/latent embedding z is approximated as a SPARSE, NON-NEGATIVE linear
combination over a CONCEPT DICTIONARY D, where each dictionary atom is the
L2-normalized text embedding of a concept phrase, encoded by the SAME frozen
tower that produced z:

        z  ~=  sum_k  w_k * d_k ,   w_k >= 0 , w sparse .

We solve   min_w  ||z - D^T w||^2 + lambda ||w||_1 ,  w >= 0
via sklearn Lasso(positive=True) (LASSO with a non-negativity constraint).

For a latent displacement Dz = z_{t+k} - z_t, decomposing z_t and z_{t+k}
separately and differencing their weight vectors gives an interpretable,
text-grounded account of the semantic change (e.g. "bowl-on-table down,
bowl-on-plate up").

Frozen CLIP ViT-L/14. CPU-only, offline. The dictionary MUST be rebuilt per
tower (it is just text embeddings through that tower -- cheap).
"""
import os
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402

MODEL_ID = "openai/clip-vit-large-patch14"
# Prompt ensemble: each atom = L2-normalized mean of these templates (standard
# CLIP zero-shot denoising; reduces template-specific artifacts in the atom).
PROMPT_TEMPLATES = [
    "a photo of {c}",
    "{c}",
    "a rendering of {c}",
    "a tabletop scene with {c}",
]


# ---------------------------------------------------------------------------
# Concept vocabulary (LIBERO-Spatial relevant + generic distractors).
# The relational scene phrases (OBJECTS_IN_PLACE) are the ones that carry the
# manipulation "delta" story; the generic set makes the LASSO selection
# non-trivial (concepts must compete for mass).
# ---------------------------------------------------------------------------

DOMAIN_OBJECTS = [
    "a black bowl", "a bowl", "a white plate", "a plate", "a ramekin",
    "a small ceramic ramekin", "a cookie box", "a red cookie box", "a box",
    "a wooden cabinet", "a cabinet", "a drawer", "an open drawer",
    "a closed drawer", "a stove", "a kitchen stove", "a robot arm",
    "a robot gripper", "a two-finger gripper", "a table", "a wooden table",
    "a kitchen counter", "a wooden desk surface",
]

MATERIALS_COLORS = [
    "wood", "wooden texture", "metal", "ceramic", "plastic", "the color black",
    "the color white", "the color red", "a dark object", "a shiny surface",
    "a matte surface", "a shadow",
]

# Relational / state scene phrases -- the delta-carrying atoms.
OBJECTS_IN_PLACE = [
    "a black bowl on the table",
    "a black bowl on the plate",
    "a black bowl on the stove",
    "a black bowl on the wooden cabinet",
    "a black bowl on the cookie box",
    "a black bowl on the ramekin",
    "a black bowl in the top drawer",
    "a black bowl in a drawer",
    "a black bowl next to the plate",
    "a black bowl next to the ramekin",
    "a black bowl next to the cookie box",
    "a black bowl between the plate and the ramekin",
    "a black bowl near the center of the table",
    "an empty plate on the table",
    "a plate with a bowl on it",
    "a bowl sitting on a plate",
    "a bowl held in a robot gripper",
    "a robot gripper holding a black bowl",
    "a robot gripper reaching for a bowl",
    "a robot gripper above the plate",
    "an empty robot gripper",
    "a robot arm hovering over the table",
    "a bowl lifted in the air",
    "a bowl being placed down",
]

ACTIONS_STATES = [
    "picking up an object", "placing an object", "grasping an object",
    "lifting an object", "reaching for an object", "putting a bowl on a plate",
    "moving a bowl", "an object being pushed", "opening a drawer",
    "closing a drawer", "a hand holding a cup", "stacking objects",
    "an empty scene", "a cluttered table", "a tidy table",
]

SCENE_GENERIC = [
    "a kitchen", "a kitchen scene", "an indoor scene", "a tabletop scene",
    "a laboratory bench", "a robotics workspace", "food on a table",
    "kitchenware", "tableware", "dishes", "a countertop",
]

# Generic distractors so concept selection is meaningful (SpLiCE uses a large
# open vocabulary; a moderate distractor set here suffices to force competition).
DISTRACTORS = [
    "a dog", "a cat", "a horse", "a bird", "a fish", "a car", "a truck",
    "a bicycle", "an airplane", "a boat", "a train", "a mountain", "a beach",
    "a forest", "a river", "the sky", "a cloud", "a tree", "a flower",
    "a person", "a face", "a hand", "a child", "a crowd", "a building",
    "a city street", "a house", "a bridge", "a road", "an apple", "a banana",
    "a pizza", "a sandwich", "a coffee cup", "a wine glass", "a book",
    "a laptop", "a phone", "a keyboard", "a clock", "a chair", "a couch",
    "a bed", "a window", "a door", "a lamp", "a painting", "a mirror",
    "a guitar", "a piano", "a ball", "a toy", "a shoe", "a hat", "a shirt",
    "a bag", "an umbrella", "fire", "water", "snow", "grass", "sand",
    "a mountain landscape", "a sunset", "a night sky", "a computer monitor",
    "a stack of papers", "a pair of scissors", "a bottle", "a can",
    "a spoon", "a fork", "a knife", "a pan", "a pot", "a refrigerator",
    "a microwave", "a sink", "a faucet", "a towel", "a sponge",
    "a desk", "a shelf", "a basket", "a jar", "a tray", "a plate of food",
    "a cutting board", "a wooden floor", "a tiled floor", "a white wall",
    "a gray background", "a plain background", "an office", "a garage",
    "a warehouse", "a workshop", "machinery", "an industrial robot",
    "a mechanical claw", "a conveyor belt", "a control panel", "wires",
    "a green plant", "a potted plant", "a candle", "a bowl of fruit",
    "a wooden bowl", "a metal bowl", "a glass bowl", "a ceramic dish",
    "a saucer", "a mug", "a teapot", "a kettle", "a blender",
]


def build_vocab():
    groups = {
        "domain_objects": DOMAIN_OBJECTS,
        "materials_colors": MATERIALS_COLORS,
        "objects_in_place": OBJECTS_IN_PLACE,
        "actions_states": ACTIONS_STATES,
        "scene_generic": SCENE_GENERIC,
        "distractors": DISTRACTORS,
    }
    concepts, categories = [], []
    for cat, items in groups.items():
        for it in items:
            concepts.append(it)
            categories.append(cat)
    return concepts, categories


# ---------------------------------------------------------------------------
# Dictionary construction (text embeddings through the frozen tower).
# ---------------------------------------------------------------------------

def build_dictionary(concepts, device="cpu", batch=64, ensemble=True):
    """Return D [K, 768] L2-normalized text embeddings (dictionary atoms).

    With ensemble=True each atom is the L2-normalized mean over PROMPT_TEMPLATES.
    """
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()
    proc = CLIPProcessor.from_pretrained(MODEL_ID)
    templates = PROMPT_TEMPLATES if ensemble else PROMPT_TEMPLATES[:1]

    def encode(prompts):
        embs = []
        with torch.no_grad():
            for i in range(0, len(prompts), batch):
                tok = proc(text=prompts[i:i + batch], return_tensors="pt",
                           padding=True, truncation=True, max_length=77).to(device)
                out = model.text_model(input_ids=tok["input_ids"],
                                       attention_mask=tok["attention_mask"])
                t = model.text_projection(out.pooler_output)
                t = torch.nn.functional.normalize(t, dim=-1)
                embs.append(t.float().cpu().numpy())
        return np.concatenate(embs, 0)

    acc = np.zeros((len(concepts), 768), np.float64)
    for tmpl in templates:
        acc += encode([tmpl.format(c=c) for c in concepts])
    acc /= len(templates)
    return (acc / (np.linalg.norm(acc, axis=1, keepdims=True) + 1e-12)).astype(np.float32)


SIGLIP_ID = "google/siglip2-so400m-patch14-384"  # 1152-d; matches cached so400m tower


def build_dictionary_siglip(concepts, device="cpu", batch=64, ensemble=True):
    """Return D [K, 1152] L2-normalized SigLIP2 text embeddings (per-tower dict).

    SigLIP has no separate projection head -- text pooler_output IS the joint
    vector (matches SiglipWrapper / the cached so400m embeddings).
    """
    import torch
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(SIGLIP_ID).to(device).eval()
    proc = AutoProcessor.from_pretrained(SIGLIP_ID)
    templates = PROMPT_TEMPLATES if ensemble else PROMPT_TEMPLATES[:1]

    def encode(prompts):
        embs = []
        with torch.no_grad():
            for i in range(0, len(prompts), batch):
                ins = proc(text=prompts[i:i + batch], return_tensors="pt",
                           padding="max_length", max_length=64, truncation=True).to(device)
                tkw = {"input_ids": ins["input_ids"]}
                if "attention_mask" in ins:
                    tkw["attention_mask"] = ins["attention_mask"]
                p = model.text_model(**tkw).pooler_output
                p = torch.nn.functional.normalize(p, dim=-1)
                embs.append(p.float().cpu().numpy())
        return np.concatenate(embs, 0)

    acc = np.zeros((len(concepts), model.config.text_config.hidden_size), np.float64)
    for tmpl in templates:
        acc += encode([tmpl.format(c=c) for c in concepts])
    acc /= len(templates)
    return (acc / (np.linalg.norm(acc, axis=1, keepdims=True) + 1e-12)).astype(np.float32)


# ---------------------------------------------------------------------------
# Sparse decomposition.
# ---------------------------------------------------------------------------

def decompose(z, D, alpha):
    """Non-negative LASSO decomposition of one embedding z [d] over D [K,d].

    Returns (w [K], recon_rel, cos_recon) where z_hat = D.T @ w.
    """
    from sklearn.linear_model import Lasso

    X = D.T                      # [d, K] design matrix (features = atoms)
    y = np.asarray(z, np.float64)
    m = Lasso(alpha=alpha, positive=True, fit_intercept=False,
              max_iter=20000, tol=1e-6)
    m.fit(X, y)
    w = m.coef_.astype(np.float64)
    z_hat = X @ w
    denom = np.linalg.norm(y) + 1e-12
    recon_rel = float(np.linalg.norm(y - z_hat) / denom)
    zn = z_hat / (np.linalg.norm(z_hat) + 1e-12)
    cos_recon = float(np.dot(y / denom, zn))
    return w, recon_rel, cos_recon


def lstsq_ceiling(z, D):
    """Unconstrained least-squares residual within span(D) = best-case ceiling."""
    X = D.T
    w, *_ = np.linalg.lstsq(X, np.asarray(z, np.float64), rcond=None)
    z_hat = X @ w
    return float(np.linalg.norm(z - z_hat) / (np.linalg.norm(z) + 1e-12))


def active_count(w, thresh=1e-3):
    return int((w > thresh).sum())
