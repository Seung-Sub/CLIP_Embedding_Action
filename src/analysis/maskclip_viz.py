"""Render MaskCLIP heatmap panels for the W4v3-P1 deliverable.

- Sanity panel: distinct-object prompts on a scene (does the value-trick localize?).
- Compositional figure: correct vs swap instruction heatmaps + difference map,
  on a libero_spatial scene that contains two candidate black bowls.
LIBERO agentview_rgb is stored vertically flipped -> flipud for display.
CPU only.
"""
import os, numpy as np, torch, h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from maskclip_heatmap import load, heatmap, upscale

OUT = "/home/user/CLIP_ws/outputs/analysis/w4v3_p1_maskclip"
DATA = "/home/user/CLIP_ws/data/libero/libero_spatial"
FILES = {
    0: "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5",
    1: "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_demo.hdf5",
    6: "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_demo.hdf5",
    7: "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_demo.hdf5",
    8: "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo.hdf5",
}
INSTR = {
    0: "pick up the black bowl between the plate and the ramekin and place it on the plate",
    1: "pick up the black bowl next to the ramekin and place it on the plate",
    6: "pick up the black bowl next to the cookie box and place it on the plate",
    7: "pick up the black bowl on the stove and place it on the plate",
    8: "pick up the black bowl next to the plate and place it on the plate",
}


def scene(tid, demo=0, frame=0):
    with h5py.File(os.path.join(DATA, FILES[tid]), "r") as f:
        img = f[f"data/demo_{demo}/obs/agentview_rgb"][frame]
    return Image.fromarray(np.flipud(img)).convert("RGB")


def overlay(ax, pil, hm, title, vmin=None, vmax=None, mark=True):
    W = pil.size
    up = upscale(hm, W)
    ax.imshow(pil)
    im = ax.imshow(up, cmap="jet", alpha=0.5, vmin=vmin, vmax=vmax)
    if mark:
        yx = np.unravel_index(hm.argmax(), hm.shape)
        g = hm.shape[0]
        ax.plot((yx[1] + .5) * W[0] / g, (yx[0] + .5) * W[1] / g, "w*", ms=14, mec="k")
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    return im


def sanity_panel(model, proc, tids, prompts):
    n = len(prompts)
    fig, axs = plt.subplots(len(tids), n + 1, figsize=(2 * (n + 1), 2 * len(tids)))
    if len(tids) == 1:
        axs = axs[None, :]
    for r, tid in enumerate(tids):
        pil = scene(tid).resize((224, 224), Image.BICUBIC)
        axs[r, 0].imshow(pil); axs[r, 0].set_title(f"scene tid{tid}", fontsize=8); axs[r, 0].axis("off")
        hms = heatmap(model, proc, pil, prompts)
        for c, pr in enumerate(prompts):
            hm = hms[pr]
            overlay(axs[r, c + 1], pil, hm, f'"{pr}"\nmax={hm.max():.2f}')
    fig.suptitle("MaskCLIP (CLIP ViT-L/14, frozen value-trick) sanity: distinct-object localization", fontsize=10)
    fig.tight_layout()
    p = os.path.join(OUT, "sanity_distinct_objects.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    return p


def swap_figure(model, proc, tid_correct, tid_swap, prompt_correct, prompt_swap, tag):
    pil = scene(tid_correct).resize((224, 224), Image.BICUBIC)
    hms = heatmap(model, proc, pil, [prompt_correct, prompt_swap])
    hc, hs = hms[prompt_correct], hms[prompt_swap]
    diff = hc - hs
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.4))
    axs[0].imshow(pil); axs[0].set_title(f"scene tid{tid_correct}\n(2 candidate bowls)", fontsize=8); axs[0].axis("off")
    vmax = max(hc.max(), hs.max()); vmin = min(hc.min(), hs.min())
    overlay(axs[1], pil, hc, f"CORRECT: {prompt_correct}", vmin, vmax)
    overlay(axs[2], pil, hs, f"SWAP: {prompt_swap}", vmin, vmax)
    d = max(abs(diff.min()), abs(diff.max()))
    up = upscale(diff, pil.size)
    axs[3].imshow(pil); im = axs[3].imshow(up, cmap="bwr", alpha=0.55, vmin=-d, vmax=d)
    axs[3].set_title("difference (correct - swap)\nred=toward correct anchor", fontsize=8); axs[3].axis("off")
    fig.suptitle(f"Compositional grounding: heatmap shift correct vs swap [{tag}]", fontsize=10)
    fig.tight_layout()
    p = os.path.join(OUT, f"swap_figure_{tag}.png")
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"[{tag}] correct argmax(yx)={np.unravel_index(hc.argmax(),hc.shape)} "
          f"swap argmax(yx)={np.unravel_index(hs.argmax(),hs.shape)} "
          f"diff range=[{diff.min():.3f},{diff.max():.3f}]")
    return p


if __name__ == "__main__":
    m, p = load("cpu")
    objs = ["a black bowl", "a white plate", "a wooden cabinet", "a cookie box", "a stove"]
    print("sanity:", sanity_panel(m, p, [0, 6, 8], objs))
    # Full-instruction figure (honest: dominated by 'plate')
    print("full-instr fig:", swap_figure(m, p, 6, 7, INSTR[6], INSTR[7], "fullinstr_tid6"))
    # Anchor-scoped figure (target phrase only -> cleaner shift)
    print("anchor fig:", swap_figure(m, p, 6, 7,
          "the black bowl next to the cookie box", "the black bowl on the stove", "anchor_tid6"))
    print("anchor fig2:", swap_figure(m, p, 8, 1,
          "the black bowl next to the plate", "the black bowl next to the ramekin", "anchor_tid8"))
