"""G-D1 — demo feasibility pre-gate (PAPER_ARCHITECTURE_v2 §3, pre-registered).

POLICY-GENERATED zeta-hat (not dataset zeta) read through adapter_main.pt:
  policy = phase2_libero_large256_matchedbase.pt (seed2) + phase1 large256 g/h.
  Offline forward passes on held-out episode states (dataset conditioning tokens
  [z_prev, z_cur, a_emb, lang, z_wrist], deterministic x0=past, eval mode).
  Replan grid t = span, span+8, ... (rollout exec-horizon 8 cadence).
  Label = R-0 bank segment with max overlap with [t, t+16). Classify
  P_action(zeta_hat) against canonical-8 phrases via P_text (max-cos).

GATE: canonical top-1 >= 0.85. Also reports margin distribution vs dataset-zeta
(chunk-level g(A_fut,z_t)) margins, cos(zeta_hat, zeta_GT) diagnosis, confusion.

CPU. RUN (remote):
  cd /workspace/CLIP_ws && OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= \
    HF_HOME=/data2/clip_ws_cache/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python3 scratchpad/pregates/pregate_gd1.py
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
from rseries_common import (OUT as RS_OUT, SemanticAdapter, load_banks,  # noqa: E402
                            load_phase1, text_embeddings)
from core import chunkrep                                                # noqa: E402
from core.anchor import get_anchor                                       # noqa: E402
from data.libero import LiberoDataset                                    # noqa: E402
from models.policy import build_policy_from_cfg                          # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
CFG = WS / "configs" / "phase2_libero_large256_matchedbase.yaml"
OUTD = WS / "outputs" / "pregates"
OUTD.mkdir(parents=True, exist_ok=True)
GATE = 0.85
EXEC_H = 8                       # rollout exec-horizon cadence
R1_CORRECT_MARGIN = 0.537        # R-1 baseline (dataset zeta, text->bank)


def main():
    t0 = time.time()
    cfg = yaml.safe_load(open(CFG))
    assert cfg["train"]["seed"] == 2, "matchedbase must be the seed2 ckpt"

    # ---- frozen phase1 (g) — rseries loader (same ckpt as R-0 substrate) ----
    ae, ck1 = load_phase1()
    n_chunk, act_dim = ck1["n_chunk"], ck1["action_dim"]
    a_mean, a_std = ck1["a_mean"], ck1["a_std"]
    repr_kind = ck1.get("chunk_repr", "time")
    latent = ck1["latent_dim"]

    # ---- phase2 flow policy (matchedbase: 5 tokens [zp, zc, a_emb, lang, zw]) ----
    ck2 = torch.load(os.path.expanduser(cfg["train"]["checkpoint"]),
                     map_location="cpu", weights_only=False)
    m_cfg = ck2["config"]["module"]
    assert m_cfg.get("lang_token") and m_cfg.get("wrist_token") \
        and not m_cfg.get("grid_obs") and not m_cfg.get("obs")
    policy = build_policy_from_cfg(m_cfg, n_tokens=5, latent_dim=latent,
                                   action_flat_dim=n_chunk * act_dim).eval()
    policy.load_state_dict(ck2["state_dict"])
    assert policy.source == "past", policy.source
    print(f"[policy] {Path(cfg['train']['checkpoint']).name} seed2 | "
          f"val act R2 {ck2['metrics']['action_r2']:+.3f} | src={policy.source}")

    # ---- adapter + phrase bank ----
    ckA = torch.load(RS_OUT / "adapter_main.pt", map_location="cpu",
                     weights_only=False)
    adapter = SemanticAdapter(in_dim=ckA["in_dim"]).eval()
    adapter.load_state_dict(ckA["state_dict"])
    classes = ckA["classes"]
    txt = text_embeddings(classes)                     # cached npz — no encoder load
    with torch.no_grad():
        q_txt = adapter.P_text(torch.tensor(
            np.stack([txt[c] for c in classes]))).numpy()   # (8,256) canonical

    bank, _, _ = load_banks()
    seg_by_key = {e["episode"]: e["segments"] for e in bank["episodes"]}

    # ---- held-out episodes (split identical to phase2 + R-0: seed2, val 0.2) ----
    ds = LiberoDataset(cfg)
    files = ds.episode_files()
    perm = np.random.RandomState(cfg["train"]["seed"]).permutation(len(files))
    n_val = max(1, round(len(files) * cfg["data"]["val_episodes"]))
    val_ids = perm[:n_val]
    span = ds.span
    clip = get_anchor(cfg)
    print(f"[split] heldout {len(val_ids)} eps | span={span} | replan step {EXEC_H}")

    def norm_chunk(seg):
        ch = ds.resample_chunk(seg)
        ch = ((ch - a_mean) / a_std).astype(np.float32)
        return chunkrep.to_repr(ch, repr_kind)

    rows = []
    n_pts_total = 0
    for i in val_ids:
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
        zh_l, zg_l, meta = [], [], []
        for t in range(span, T - span + 1, EXEC_H):
            n_pts_total += 1
            # segment with max overlap with future window [t, t+span)
            best, best_ov = None, 0
            for s in segs:
                a, b = int(s["segment"][0]), int(s["segment"][1])
                ov = min(t + span, b) - max(t, a)
                if ov > best_ov:
                    best, best_ov = s, ov
            if best is None:
                continue
            zh_l.append((t, norm_chunk(acts[t - span:t])))
            zg_l.append(norm_chunk(acts[t:t + span]))
            meta.append(dict(key=key, t=t, lab=classes.index(best["phrase"]),
                             overlap=best_ov / span,
                             pure=bool(best_ov == span)))
        if not meta:
            continue
        ts = [t for t, _ in zh_l]
        with torch.no_grad():
            zp = torch.tensor(np.stack([Z[t - span] for t in ts]))
            zc = torch.tensor(np.stack([Z[t] for t in ts]))
            zw = torch.tensor(np.stack([Zw[t] for t in ts]))
            lg = torch.tensor(np.repeat(lang[None], len(ts), 0))
            cp = torch.tensor(np.stack([c for _, c in zh_l]))
            a_emb = ae.g(cp, zp)
            toks = torch.stack([zp, zc, a_emb, lg, zw], dim=1)   # train order
            zeta_hat = policy(toks)                               # deterministic
            zeta_gt = ae.g(torch.tensor(np.stack(zg_l)), zc)      # dataset-zeta (chunk)
        for k, mrow in enumerate(meta):
            mrow["zeta_hat"] = zeta_hat[k].numpy()
            mrow["zeta_gt"] = zeta_gt[k].numpy()
            rows.append(mrow)

    print(f"[data] replan points {n_pts_total} | labeled {len(rows)} "
          f"| pure {sum(r['pure'] for r in rows)} | {time.time()-t0:.0f}s")

    ZH = np.stack([r["zeta_hat"] for r in rows]).astype(np.float32)
    ZG = np.stack([r["zeta_gt"] for r in rows]).astype(np.float32)
    lab = np.array([r["lab"] for r in rows])
    pure = np.array([r["pure"] for r in rows])
    cos = (ZH * ZG).sum(1) / (np.linalg.norm(ZH, axis=1)
                              * np.linalg.norm(ZG, axis=1) + 1e-8)

    def classify(Zx):
        with torch.no_grad():
            q = adapter.P_action(torch.tensor(Zx)).numpy()
        S = q @ q_txt.T                                # (N,8) canonical max==mean (1 phrase)
        pred = S.argmax(1)
        part = np.partition(S, -2, axis=1)
        margin = part[:, -1] - part[:, -2]
        return pred, margin, S

    pred_h, marg_h, _ = classify(ZH)
    pred_g, marg_g, _ = classify(ZG)

    def report(pred, margin, tag):
        acc = float((pred == lab).mean())
        accp = float((pred[pure] == lab[pure]).mean())
        accb = float((pred[~pure] == lab[~pure]).mean()) if (~pure).any() else None
        conf = np.zeros((len(classes), len(classes)), int)
        for y, p in zip(lab, pred):
            conf[y, p] += 1
        per = {classes[c]: float((pred[lab == c] == c).mean())
               for c in range(len(classes)) if (lab == c).any()}
        print(f"  [{tag}] top1 {acc:.4f} (pure {accp:.4f} / boundary {accb}) "
              f"margin {margin.mean():.4f}±{margin.std():.4f}")
        return dict(top1=acc, top1_pure=accp, top1_boundary=accb,
                    per_class=per, confusion=conf.tolist(),
                    margin_mean=float(margin.mean()), margin_std=float(margin.std()),
                    margin_q=[float(np.quantile(margin, q))
                              for q in (0.05, 0.25, 0.5, 0.75, 0.95)])

    print("[eval] policy zeta-hat vs adapter canonical-8")
    res_h = report(pred_h, marg_h, "zeta_hat")
    print("[eval] dataset zeta (chunk-level g(A_fut, z_t)) — same points")
    res_g = report(pred_g, marg_g, "zeta_GT")

    acc = res_h["top1"]
    verdict = "PASS" if acc >= GATE else "FAIL"
    # diagnosis rows (used if FAIL, reported always)
    wrong = pred_h != lab
    diag = {
        "cos_zeta_hat_vs_gt_mean": float(cos.mean()),
        "cos_std": float(cos.std()),
        "cos_q05_50_95": [float(np.quantile(cos, q)) for q in (.05, .5, .95)],
        "cos_on_correct": float(cos[~wrong].mean()) if (~wrong).any() else None,
        "cos_on_wrong": float(cos[wrong].mean()) if wrong.any() else None,
        "gt_acc_on_hat_wrong": (float((pred_g[wrong] == lab[wrong]).mean())
                                if wrong.any() else None),
    }
    res = {
        "gate": "G-D1", "threshold": GATE, "verdict": verdict,
        "policy_ckpt": str(cfg["train"]["checkpoint"]),
        "phase1_ckpt": ck1.get("config", {}).get("train", {}).get("checkpoint", ""),
        "adapter": "outputs/rseries/adapter_main.pt",
        "protocol": {"replan_step": EXEC_H, "span": int(span),
                     "label": "max-overlap segment over [t,t+span)",
                     "x0": "past (deterministic, eval mode)"},
        "n_replan_points": int(n_pts_total), "n_labeled": len(rows),
        "n_pure": int(pure.sum()),
        "zeta_hat": res_h, "zeta_dataset_chunk": res_g,
        "diagnosis": diag,
        "reference": {"R0_heldout_canonical_seglevel": 0.972,
                      "R1_correct_margin_text2bank": R1_CORRECT_MARGIN},
        "_meta": {"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "host": os.uname().nodename,
                  "wall_s": round(time.time() - t0)},
    }
    p = OUTD / "gd1_results.json"
    p.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"[G-D1] top1 {acc:.4f} vs gate {GATE} -> {verdict} | saved {p}")


if __name__ == "__main__":
    main()
