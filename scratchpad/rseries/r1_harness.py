"""R-1 — discriminative harness at the RETRIEVAL level (our correct/wrong/
paraphrase protocol + shuffled-text control, colleague demo lacked the latter).

Query: q_l = P_text(SigLIP2-large256 text emb). Bank: q_a = P_action(zeta) over
the TRAIN effect bank (1995-seg class, colleague retrieval_control.py convention).
For each query: top-1 segment (+prov), top-5 consensus, retrieved class =
argmax_c max_{seg in c} cos, class margin = best_class - best_other_class.

Groups:
  correct        canonical command  -> must retrieve its own class
  wrong/swap     other-class command -> must retrieve the OTHER class (matrix)
  paraphrase     unseen templates (theirs 16) + our independent 3rd set (24)
  shuffled       word-shuffled canonicals (3/phrase, seed 0)
  nonsense       jabberwocky strings
  unrelated      commands about other objects/actions (no bank class)

Cross-substrate row: colleague adapter weights are NOT in the clone
(SigLIP/checkpoints absent locally and remotely) -> checked, skipped gracefully.

CPU. RUN: OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= python scratchpad/rseries/r1_harness.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HOME", "/data2/clip_ws_cache/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rseries_common import (OUT, RS, THIRD_SET_OURS, WS, SemanticAdapter,
                            load_banks, text_embeddings)

NONSENSE = [
    "the mimsy borogoves did gyre and gimble in the wabe",
    "colorless green ideas sleep furiously",
    "blorp fizzle quantum banana torque",
    "seven purple concepts of arithmetic rain",
    "flumox the cranberry hypotenuse gently",
    "zork bling table sky unwrap velocity",
    "a of the with under between onto",
    "gripper gripper gripper bowl bowl bowl",
]
UNRELATED = [
    "open the top drawer of the wooden cabinet",
    "turn on the stove",
    "push the plate away from the robot",
    "pick up the ramekin",
    "close the cookie box",
    "wipe the table with a cloth",
]
COLLEAGUE_CKPT_CANDIDATES = [
    WS / "SigLIP/checkpoints/grid/semantic_adapter_paraphrase.pt",
    Path("/home/user/CLIP_ws/SigLIP/checkpoints/grid/semantic_adapter_paraphrase.pt"),
]


def shuffled_variants(phrase, k=3, seed=0):
    import zlib
    rng = np.random.RandomState((seed + zlib.crc32(phrase.encode())) % 2**31)
    words = phrase.split()
    outs, tries = [], 0
    while len(outs) < k and tries < 50:
        tries += 1
        w = list(words)
        rng.shuffle(w)
        s = " ".join(w)
        if s != phrase and s not in outs:
            outs.append(s)
    return outs


def retrieve(q_l, q_a_bank, lab_bank, prov, classes):
    """One query vector -> dict of retrieval facts."""
    sims = q_a_bank @ q_l                                  # (Nbank,)
    order = np.argsort(-sims)
    top1 = int(order[0])
    top5_lab = lab_bank[order[:5]]
    vals, cnts = np.unique(top5_lab, return_counts=True)
    consensus = int(vals[cnts.argmax()])
    best_per_class = np.array([sims[lab_bank == c].max()
                               for c in range(len(classes))])
    rc = int(best_per_class.argmax())
    margin = float(best_per_class[rc] - np.partition(best_per_class, -2)[-2])
    return {"retrieved_class": classes[rc],
            "top1_cos": float(sims[top1]),
            "top1_class": classes[int(lab_bank[top1])],
            "top1_prov": prov[top1],
            "top5_consensus": classes[consensus],
            "top5_consensus_n": int(cnts.max()),
            "class_margin": margin}


def run_group(name, queries, expected, txt, Ptext, q_a_bank, lab_bank, prov,
              classes, results):
    """queries: [(text, expected_class_or_None)]"""
    with torch.no_grad():
        Q = Ptext(torch.tensor(np.stack([txt[t] for t, _ in queries]))).numpy()
    rows, margins, hits = [], [], []
    for (t, exp), q in zip(queries, Q):
        r = retrieve(q, q_a_bank, lab_bank, prov, classes)
        r["query"] = t
        r["expected"] = exp
        if exp is not None:
            r["hit"] = r["retrieved_class"] == exp
            hits.append(r["hit"])
        margins.append(r["class_margin"])
        rows.append(r)
    m = np.array(margins)
    summ = {"n": len(rows),
            "acc": (float(np.mean(hits)) if hits else None),
            "margin_mean": float(m.mean()), "margin_std": float(m.std()),
            "margin_min": float(m.min()), "margin_max": float(m.max()),
            "margins": [round(float(x), 4) for x in m]}
    results["groups"][name] = {"summary": summ, "rows": rows}
    print(f"[{name:16s}] n={summ['n']:3d} acc={summ['acc']} "
          f"margin {summ['margin_mean']:.4f}±{summ['margin_std']:.4f} "
          f"[{summ['margin_min']:.4f},{summ['margin_max']:.4f}]")
    return rows


def main():
    BK = pickle.load(open(OUT / "effect_bank_ours.pkl", "rb"))
    classes = BK["classes"]
    tr = BK["train"]
    ck = torch.load(OUT / "adapter_main.pt", map_location="cpu", weights_only=False)
    adapter = SemanticAdapter(in_dim=ck["in_dim"]).eval()
    adapter.load_state_dict(ck["state_dict"])
    with torch.no_grad():
        q_a_bank = adapter.P_action(torch.tensor(tr["zeta"])).numpy()
    lab_bank, prov = tr["lab"], tr["prov"]
    print(f"[bank] {len(lab_bank)} train segments | classes {len(classes)}")

    bank, hard_bank, their = load_banks()
    unseen = their["unseen_templates"]

    queries = {"correct": [(c, c) for c in classes]}
    queries["paraphrase_unseen_their"] = [(p, c) for c, ps in unseen.items()
                                          for p in ps]
    queries["paraphrase_third_ours"] = [(p, c) for c, ps in THIRD_SET_OURS.items()
                                        for p in ps]
    queries["shuffled"] = [(s, None) for c in classes
                           for s in shuffled_variants(c)]
    queries["nonsense"] = [(s, None) for s in NONSENSE]
    queries["unrelated"] = [(s, None) for s in UNRELATED]

    need = {t for qs in queries.values() for t, _ in qs}
    txt = text_embeddings(need)

    results = {"substrate": BK["substrate"], "adapter": "adapter_main.pt",
               "n_bank_segments": int(len(lab_bank)), "classes": classes,
               "groups": {}}

    for name, qs in queries.items():
        run_group(name, qs, None, txt, adapter.P_text, q_a_bank, lab_bank,
                  prov, classes, results)

    # ---- swap-sensitivity matrix (wrong-command protocol) ----
    # Retrieval depends only on the text, so the 8x8 swap matrix is the map
    # command -> retrieved class; swap-sensitive iff every off-diagonal ordered
    # pair (expected c, issued c') retrieves c' (not c).
    corr = {r["query"]: r["retrieved_class"]
            for r in results["groups"]["correct"]["rows"]}
    swap_ok = sum(1 for c in classes for c2 in classes if c != c2
                  and corr[c2] == c2)
    n_pairs = len(classes) * (len(classes) - 1)
    results["swap_sensitivity"] = {
        "ok_pairs": swap_ok, "n_pairs": n_pairs, "frac": swap_ok / n_pairs,
        "command_to_retrieved": corr}
    print(f"[swap] {swap_ok}/{n_pairs} ordered wrong-command pairs retrieve "
          f"the ISSUED class")

    # ---- per-class confusion (correct + paraphrase groups) ----
    cid = {c: i for i, c in enumerate(classes)}
    conf = np.zeros((len(classes), len(classes)), int)
    for g in ("correct", "paraphrase_unseen_their", "paraphrase_third_ours"):
        for r in results["groups"][g]["rows"]:
            conf[cid[r["expected"]], cid[r["retrieved_class"]]] += 1
    results["confusion_correct_plus_paraphrase"] = conf.tolist()

    # ---- cross-substrate row: colleague adapter weights ----
    found = [str(p) for p in COLLEAGUE_CKPT_CANDIDATES if p.exists()]
    if found:
        results["cross_substrate"] = {"status": "found", "path": found[0]}
        # (would require their zeta substrate to be meaningful — documented)
    else:
        results["cross_substrate"] = {
            "status": "skipped",
            "reason": "colleague semantic_adapter_paraphrase.pt not present in "
                      "clone (SigLIP/checkpoints absent locally and remotely); "
                      "their held-out numbers quoted from e6/e7 JSONs instead"}
    print(f"[cross-substrate] {results['cross_substrate']['status']}")

    json.dump(results, open(OUT / "r1_results.json", "w"), indent=1)
    print(f"[saved] {OUT/'r1_results.json'}")


if __name__ == "__main__":
    main()
