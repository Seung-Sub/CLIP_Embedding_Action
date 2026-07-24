"""G-D1b — pre-registered FAIL remedial (PAPER_ARCHITECTURE_v2 §3: refit the
0.26M adapter projections on POLICY zeta-hat; script-level, no policy retraining).

G-D1 failed (0.363 < 0.85) with clean decomposition:
  segment->chunk granularity: dataset-zeta chunk-level 0.657 (vs seg-level 0.972)
  zeta-hat vs zeta_GT cos 0.188 == training val lat_cos 0.203 (harness verified)
So the adapter must be fit IN the zeta-hat distribution. Here: collect zeta-hat
on TRAIN episodes with the same replan protocol (t = span, span+8, ...; label =
max-overlap R-0 bank segment), refit SemanticAdapter with the identical R-0
recipe (SupCon paraphrase-9, hard negatives, dz = Z[t+span]-Z[t]), evaluate on
HELD-OUT zeta-hat (same points as gd1_results).  Success criterion inherited
from E2 wording: heldout canonical top-1 >= 0.85 -> demo lives (with refit
adapter); < 0.85 -> demo demoted to offline-readout figure.

CPU. RUN (remote):
  cd /workspace/CLIP_ws && OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= \
    HF_HOME=/data2/clip_ws_cache/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python3 scratchpad/pregates/pregate_gd1b_refit.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HOME", "/data2/clip_ws_cache/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

WS = Path(os.environ.get("PREGATE_WS", "/workspace/CLIP_ws"))
sys.path.insert(0, str(WS / "src"))
sys.path.insert(0, str(WS / "scratchpad" / "rseries"))
import rseries_common as RC                                              # noqa: E402
from r0_train_eval import build_language_bank, train_adapter             # noqa: E402
from core import chunkrep                                                # noqa: E402
from core.anchor import get_anchor                                       # noqa: E402
from data.libero import LiberoDataset                                    # noqa: E402
from models.policy import build_policy_from_cfg                          # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
CFG = WS / "configs" / "phase2_libero_large256_matchedbase.yaml"
OUTD = WS / "outputs" / "pregates"
GATE = 0.85
EXEC_H = 8


def collect_split(ids, files, ds, clip, ae, policy, seg_by_key, classes,
                  a_mean, a_std, repr_kind, tag):
    span = ds.span
    ZH, ZG, DZ, LAB, PURE = [], [], [], [], []

    def norm_chunk(seg):
        ch = ds.resample_chunk(seg)
        ch = ((ch - a_mean) / a_std).astype(np.float32)
        return chunkrep.to_repr(ch, repr_kind)

    for i in ids:
        ep = files[i]
        key = ds._key(ep)
        segs = seg_by_key.get(key)
        if segs is None:
            continue
        acts = ds.load_actions(ep)
        Z = ds.embeddings(clip, ep)
        Zw = ds.embeddings(clip, ep, ds.wrist_camera)
        lang = ds.instruction_embedding(clip, ep)
        T = min(len(Z), len(acts))
        pts, labs, purs = [], [], []
        for t in range(span, T - span + 1, EXEC_H):
            best, best_ov = None, 0
            for s in segs:
                a, b = int(s["segment"][0]), int(s["segment"][1])
                ov = min(t + span, b) - max(t, a)
                if ov > best_ov:
                    best, best_ov = s, ov
            if best is None:
                continue
            pts.append(t)
            labs.append(classes.index(best["phrase"]))
            purs.append(bool(best_ov == span))
        if not pts:
            continue
        with torch.no_grad():
            zp = torch.tensor(np.stack([Z[t - span] for t in pts]))
            zc = torch.tensor(np.stack([Z[t] for t in pts]))
            zw = torch.tensor(np.stack([Zw[t] for t in pts]))
            lg = torch.tensor(np.repeat(lang[None], len(pts), 0))
            cp = torch.tensor(np.stack([norm_chunk(acts[t - span:t])
                                        for t in pts]))
            a_emb = ae.g(cp, zp)
            toks = torch.stack([zp, zc, a_emb, lg, zw], dim=1)
            ZH.append(policy(toks).numpy())
            ZG.append(ae.g(torch.tensor(np.stack(
                [norm_chunk(acts[t:t + span]) for t in pts])), zc).numpy())
        DZ.append(np.stack([Z[min(t + span, len(Z) - 1)] - Z[t] for t in pts]))
        LAB += labs
        PURE += purs
    print(f"[{tag}] {len(LAB)} labeled replan points")
    return (np.concatenate(ZH).astype(np.float32),
            np.concatenate(ZG).astype(np.float32),
            np.concatenate(DZ).astype(np.float32),
            np.array(LAB), np.array(PURE))


def main():
    t0 = time.time()
    cfg = yaml.safe_load(open(CFG))
    ae, ck1 = RC.load_phase1()
    n_chunk, act_dim = ck1["n_chunk"], ck1["action_dim"]
    a_mean, a_std = ck1["a_mean"], ck1["a_std"]
    repr_kind = ck1.get("chunk_repr", "time")

    ck2 = torch.load(os.path.expanduser(cfg["train"]["checkpoint"]),
                     map_location="cpu", weights_only=False)
    policy = build_policy_from_cfg(ck2["config"]["module"], n_tokens=5,
                                   latent_dim=ck1["latent_dim"],
                                   action_flat_dim=n_chunk * act_dim).eval()
    policy.load_state_dict(ck2["state_dict"])

    bank, hard_bank, their = RC.load_banks()
    seg_by_key = {e["episode"]: e["segments"] for e in bank["episodes"]}
    ckA = torch.load(RC.OUT / "adapter_main.pt", map_location="cpu",
                     weights_only=False)
    classes = ckA["classes"]

    ds = LiberoDataset(cfg)
    files = ds.episode_files()
    perm = np.random.RandomState(cfg["train"]["seed"]).permutation(len(files))
    n_val = max(1, round(len(files) * cfg["data"]["val_episodes"]))
    val_ids, tr_ids = perm[:n_val], perm[n_val:]
    clip = get_anchor(cfg)

    ZHt, ZGt, DZt, LABt, _ = collect_split(
        tr_ids, files, ds, clip, ae, policy, seg_by_key, classes,
        a_mean, a_std, repr_kind, "train")
    ZHv, ZGv, DZv, LABv, PUREv = collect_split(
        val_ids, files, ds, clip, ae, policy, seg_by_key, classes,
        a_mean, a_std, repr_kind, "heldout")
    print(f"[data] {time.time()-t0:.0f}s | train class counts "
          f"{np.bincount(LABt, minlength=8).tolist()}")

    # ---- refit adapter on TRAIN zeta-hat, identical R-0 recipe ----
    para_aug = their["para_aug_per_class"]
    hp = dict(their["hparams"])
    hp["warmup"] = hp.pop("warmup", 200)
    need = set(classes)
    for c in classes:
        need.update(para_aug[c])
        need.update(hard_bank.get(c, []))
    txt = RC.text_embeddings(need)
    lbank = build_language_bank(classes, para_aug, hard_bank, txt)
    print("[train] adapter_zetahat (policy zeta-hat, paraphrase-9 recipe)")
    ad = train_adapter(ZHt, DZt, LABt, lbank, hp, "zetahat")

    with torch.no_grad():
        q_txt = ad.P_text(torch.tensor(
            np.stack([txt[c] for c in classes]))).numpy()

    def ev(Zx, lab, pure, tag):
        with torch.no_grad():
            q = ad.P_action(torch.tensor(Zx)).numpy()
        S = q @ q_txt.T
        pred = S.argmax(1)
        part = np.partition(S, -2, axis=1)
        margin = part[:, -1] - part[:, -2]
        acc = float((pred == lab).mean())
        conf = np.zeros((len(classes), len(classes)), int)
        for y, p in zip(lab, pred):
            conf[y, p] += 1
        out = dict(top1=acc,
                   top1_pure=float((pred[pure] == lab[pure]).mean()),
                   top1_boundary=float((pred[~pure] == lab[~pure]).mean()),
                   per_class={classes[c]: float((pred[lab == c] == c).mean())
                              for c in range(len(classes)) if (lab == c).any()},
                   confusion=conf.tolist(),
                   margin_mean=float(margin.mean()),
                   margin_std=float(margin.std()),
                   margin_q=[float(np.quantile(margin, q))
                             for q in (.05, .25, .5, .75, .95)])
        print(f"  [{tag}] top1 {acc:.4f} (pure {out['top1_pure']:.4f} / "
              f"boundary {out['top1_boundary']:.4f}) "
              f"margin {out['margin_mean']:.4f}±{out['margin_std']:.4f}")
        return out

    print("[eval] refit adapter on HELD-OUT zeta-hat")
    res_h = ev(ZHv, LABv, PUREv, "heldout zeta_hat")
    print("[eval] refit adapter on held-out dataset-zeta chunk (transfer check)")
    res_g = ev(ZGv, LABv, PUREv, "heldout zeta_GT")
    # train-fit sanity
    res_tr = ev(ZHt, LABt, np.ones(len(LABt), bool), "train zeta_hat (fit)")

    acc = res_h["top1"]
    verdict = "PASS (refit)" if acc >= GATE else "FAIL (refit)"
    res = {
        "gate": "G-D1b refit (pre-registered remedial, §3)", "threshold": GATE,
        "verdict": verdict,
        "recipe": "R-0 paraphrase-9 SupCon, retrained on policy zeta-hat "
                  "(train episodes, replan-grid chunk labels)",
        "n_train": int(len(LABt)), "n_heldout": int(len(LABv)),
        "train_class_counts": np.bincount(LABt, minlength=8).tolist(),
        "heldout_zeta_hat": res_h, "heldout_zeta_gt_chunk": res_g,
        "train_zeta_hat_fit": res_tr,
        "_meta": {"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "host": os.uname().nodename,
                  "wall_s": round(time.time() - t0)},
    }
    torch.save({"state_dict": ad.state_dict(), "classes": classes,
                "in_dim": int(ZHt.shape[1]), "recipe": hp,
                "trained_on": "policy_zeta_hat_replan_grid"},
               OUTD / "adapter_zetahat.pt")
    p = OUTD / "gd1b_refit_results.json"
    p.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"[G-D1b] heldout zeta-hat top1 {acc:.4f} vs {GATE} -> {verdict} "
          f"| saved {p}")


if __name__ == "__main__":
    main()
