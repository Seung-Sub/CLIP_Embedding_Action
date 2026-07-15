"""MaskCLIP-style training-free text->patch heatmap (ECCV'22, Zhou et al.).

Frozen CLIP ViT-L/14. Bypass the final attention pooling: in the LAST transformer
block take the per-patch VALUE projection (v_proj), apply the block's out_proj as a
per-token linear (i.e. identity attention -- each token attends only to itself),
skip the block residual + MLP, then post_layernorm + visual_projection into the joint
text-image space. Cosine with the CLIP text embedding of a prompt -> per-patch
relevance heatmap. Training-free; frozen encoder used as-is.

CPU by default (small model, a handful of frames). No GPU contention.
"""
import argparse, os, numpy as np, torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_ID = "openai/clip-vit-large-patch14"


def load(device="cpu"):
    model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()
    proc = CLIPProcessor.from_pretrained(MODEL_ID)
    return model, proc


@torch.no_grad()
def maskclip_patch_embeds(model, pixel_values, keep_residual_mlp=False):
    """Return per-patch joint-space embeddings [B, n_patch, joint]."""
    vt = model.vision_model
    L = vt.encoder.layers[-1]
    # Capture the input to the last transformer block via a pre-hook (robust to
    # transformers-version changes in the layer/encoder forward signature).
    cap = {}

    def pre_hook(module, args, kwargs):
        cap["inp"] = args[0] if args else kwargs["hidden_states"]

    h = L.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        vt(pixel_values)
    finally:
        h.remove()
    hidden = cap["inp"]
    x = L.layer_norm1(hidden)
    v = L.self_attn.v_proj(x)              # [B,N,D] per-patch value
    attn_out = L.self_attn.out_proj(v)     # per-token linear (no QK mixing)
    if keep_residual_mlp:
        h = hidden + attn_out
        h = h + L.mlp(L.layer_norm2(h))
    else:
        h = attn_out                       # canonical MaskCLIP: value features only
    h = vt.post_layernorm(h)
    emb = model.visual_projection(h)       # [B,N,joint]
    return emb[:, 1:, :]                    # drop CLS -> [B, n_patch, joint]


@torch.no_grad()
def text_embed(model, proc, prompts, device="cpu"):
    tok = proc(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    pooled = model.text_model(**tok).pooler_output   # EOS-pooled text hidden
    t = model.text_projection(pooled)                # -> joint space (matches image side)
    return torch.nn.functional.normalize(t, dim=-1)


@torch.no_grad()
def heatmap(model, proc, pil_img, prompts, device="cpu", keep_residual_mlp=False):
    """Return dict prompt -> HxW heatmap (raw cosine, before any normalization)."""
    px = proc(images=pil_img, return_tensors="pt").to(device)["pixel_values"]
    patches = maskclip_patch_embeds(model, px, keep_residual_mlp)  # [1,256,768]
    patches = torch.nn.functional.normalize(patches, dim=-1)[0]     # [256,768]
    txt = text_embed(model, proc, prompts, device)                  # [P,768]
    sim = patches @ txt.T                                           # [256,P]
    g = int(round(patches.shape[0] ** 0.5))
    out = {}
    for i, p in enumerate(prompts):
        out[p] = sim[:, i].reshape(g, g).cpu().numpy()
    return out


def upscale(hm, size):
    return np.array(Image.fromarray(hm.astype(np.float32)).resize(size, Image.BICUBIC))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    m, p = load(a.device)
    img = Image.open(a.image).convert("RGB")
    hms = heatmap(m, p, img, a.prompts, a.device)
    for k, v in hms.items():
        print(k, "min", round(float(v.min()), 4), "max", round(float(v.max()), 4),
              "argmax(yx)", np.unravel_index(v.argmax(), v.shape))
