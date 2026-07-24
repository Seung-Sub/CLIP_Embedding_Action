"""R-0 steps 2-3 — train P_text/P_action on OUR zeta + pre-registered gates.

Recipe = colleague paraphrase-winner (e7): SupCon multi-positive, language bank =
canonical + 9 paraphrases/class, per-class template hard-negatives in the
denominator only; loss weights sem_av 0.1 / sem_al 0.05 / sem_vl 0.0, temp 0.07,
AdamW lr 1e-3 wd 1e-4, bs 256, 200 epochs, cosine + 200 warmup steps, seed 0.

Gates (docs/ANALYSIS_colleague_retrieval_control.md §5, pre-registered):
  G-R0a  held-out canonical top-1 >= 0.90
  G-R0b  honesty controls (MANDATORY, same table): text-free MLP classifier on
         zeta; state-residualized variant zeta - r(z_t), r = frozen RidgeCV
         (alpha grid 0.1..1e4, week0 f2_dense_probe pattern) fit on train.
  G-R0c  unseen-template top-1 >= 0.85 AND independent 3rd set >= 0.90
         (3rd set = OURS, written fresh — see rseries_common.THIRD_SET_OURS).

CPU. RUN: OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= python scratchpad/rseries/r0_train_eval.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HOME", "/data2/clip_ws_cache/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rseries_common import (OUT, THIRD_SET_OURS, SemanticAdapter, acc_report,
                            classify, load_banks, supcon_crossmodal,
                            supcon_symmetric, text_embeddings)

GATES = {"G-R0a_canonical": 0.90, "G-R0c_unseen": 0.85, "G-R0c_third": 0.90}


# ---------------------------------------------------------------------------
def build_language_bank(classes, para_aug, hard_bank, txt):
    """(bank_vecs, bank_lab, hard_mat, hard_msk) — colleague build_language_bank,
    paraphrase-winner variant (9 aug paraphrases/class instead of the bank's 3)."""
    cls_id = {c: i for i, c in enumerate(classes)}
    vecs, lab = [], []
    for c in classes:
        for p in [c] + list(para_aug[c]):
            vecs.append(txt[p]); lab.append(cls_id[c])
    hard = {c: list(hard_bank.get(c, [])) for c in classes}
    Hmax = max(len(hard[c]) for c in classes)
    d = vecs[0].shape[0]
    hm = np.zeros((len(classes), Hmax, d), np.float32)
    hk = np.zeros((len(classes), Hmax), np.float32)
    for c in classes:
        for j, p in enumerate(hard[c]):
            hm[cls_id[c], j] = txt[p]; hk[cls_id[c], j] = 1.0
    return (torch.tensor(np.stack(vecs)), torch.tensor(np.array(lab, np.int64)),
            torch.tensor(hm), torch.tensor(hk))


def train_adapter(zeta, dz, lab, lbank, hp, tag):
    """Colleague train_semantic_adapter main loop, ported verbatim (CPU)."""
    torch.manual_seed(int(hp["seed"])); np.random.seed(int(hp["seed"]))
    lbank_v, lbank_l, hard_mat, hard_msk = lbank
    in_dim = zeta.shape[1]
    adapter = SemanticAdapter(in_dim=in_dim, out_dim=256)
    opt = torch.optim.AdamW(adapter.parameters(), lr=float(hp["lr"]),
                            weight_decay=float(hp["wd"]))
    temp, B = float(hp["temp"]), int(hp["bs"])
    sem_av, sem_al, sem_vl = float(hp["sem_av"]), float(hp["sem_al"]), float(hp["sem_vl"])
    epochs = int(hp["epochs"])
    N = len(lab)
    steps_per = max(1, (N + B - 1) // B)
    total_steps = epochs * steps_per
    warmup = int(hp["warmup"])

    def lr_at(step):
        if step < warmup:
            return step / max(1, warmup)
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + np.cos(np.pi * min(p, 1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    zeta_t, dz_t = torch.tensor(zeta), torch.tensor(dz)
    lab_t = torch.tensor(lab)
    t0 = time.time()
    first = last = None
    for ep in range(epochs):
        perm = torch.randperm(N)
        adapter.train()
        for i in range(0, N, B):
            idx = perm[i:i + B]
            zi, li, dzi = zeta_t[idx], lab_t[idx], dz_t[idx]
            q_a = adapter.P_action(zi)
            q_v = adapter.P_visual(dzi)
            L_av = supcon_symmetric(q_a, q_v, li, temp) if sem_av > 0 else zeta_t.new_zeros(())
            hard_i, hmsk_i = hard_mat[li], hard_msk[li]
            lbv = adapter.P_text(lbank_v)
            hb = adapter.P_text(hard_i.reshape(-1, in_dim)).reshape(
                hard_i.shape[0], hard_i.shape[1], -1)
            L_al = (supcon_crossmodal(q_a, lbv, li, lbank_l, temp,
                                      hard=hb, hard_mask=hmsk_i)
                    if sem_al > 0 else zeta_t.new_zeros(()))
            L_vl = (supcon_crossmodal(q_v, lbv, li, lbank_l, temp,
                                      hard=hb, hard_mask=hmsk_i)
                    if sem_vl > 0 else zeta_t.new_zeros(()))
            loss = sem_av * L_av + sem_al * L_al + sem_vl * L_vl
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            first = first if first is not None else loss.item()
            last = loss.item()
        if ep % 50 == 0 or ep == epochs - 1:
            print(f"  [{tag}] ep {ep:3d} loss {last:.4f} "
                  f"(av {float(L_av):.4f} al {float(L_al):.4f})")
    print(f"  [{tag}] {epochs} ep in {time.time()-t0:.0f}s | {first:.4f} -> {last:.4f}")
    adapter.eval()
    return adapter


def train_mlp(X, y, Xh, yh, seed=0, epochs=200, hidden=256, tag="mlp"):
    """Text-free MLP classifier control (G-R0b) — same budget class as adapter."""
    torch.manual_seed(seed); np.random.seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(),
                              torch.nn.Linear(hidden, int(y.max()) + 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt, yt = torch.tensor(X), torch.tensor(y)
    N, B = len(y), 256
    for ep in range(epochs):
        perm = torch.randperm(N)
        for i in range(0, N, B):
            idx = perm[i:i + B]
            loss = torch.nn.functional.cross_entropy(net(Xt[idx]), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(torch.tensor(Xh)).argmax(1).numpy()
    acc = float((pred == yh).mean())
    print(f"  [{tag}] held-out top-1 {acc:.4f}")
    return acc, pred


def eval_adapter(adapter, zeta_h, lab_h, classes, txt, sets):
    """Per-set nearest-phrase (max) + mean-embedding accuracies."""
    with torch.no_grad():
        q_a = adapter.P_action(torch.tensor(zeta_h)).numpy()
    out = {}
    for name, ps in sets.items():
        smax, smean = classify(q_a, txt, ps, classes, adapter.P_text)
        t_max, per, conf = acc_report(smax, lab_h, classes)
        t_mean, _, _ = acc_report(smean, lab_h, classes)
        out[name] = {"top1": t_max, "top1_meanemb": t_mean,
                     "per_class": per, "confusion": conf}
        print(f"  {name:18s} top-1 {t_max:.4f} (mean-emb {t_mean:.4f})")
    return out


# ---------------------------------------------------------------------------
def main():
    BK = pickle.load(open(OUT / "effect_bank_ours.pkl", "rb"))
    classes = BK["classes"]
    tr, ho = BK["train"], BK["heldout"]
    bank, hard_bank, their = load_banks()
    para_aug = their["para_aug_per_class"]
    hp = dict(their["hparams"])          # sem_av .1 sem_al .05 sem_vl 0 temp .07
    hp["warmup"] = hp.pop("warmup", 200)

    # bank(3)/aug(9) paraphrases + eval template sets
    para3 = their["para_bank_per_class"]
    unseen = their["unseen_templates"]
    novel3_theirs = their["novel3_theirs"]
    sets = {"canonical": {c: [c] for c in classes},
            "seen_para3": para3,
            "seen_para9": para_aug,
            "unseen_their": unseen,
            "third_theirs": novel3_theirs,
            "third_ours": THIRD_SET_OURS}

    need = set()
    for ps in sets.values():
        for c, l in ps.items():
            need.update(l)
    need.update(classes)
    for c in classes:
        need.update(hard_bank.get(c, []))
    txt = text_embeddings(need)
    lbank = build_language_bank(classes, para_aug, hard_bank, txt)
    print(f"[lang] bank rows {len(lbank[1])} | hard max/class {lbank[2].shape[1]} "
          f"| text emb {len(txt)}")

    results = {"substrate": BK["substrate"], "phase1_ckpt": BK["phase1_ckpt"],
               "split_seed": BK["split_seed"],
               "n_train_seg": int(len(tr["lab"])), "n_heldout_seg": int(len(ho["lab"])),
               "heldout_class_counts": {c: int((ho["lab"] == i).sum())
                                        for i, c in enumerate(classes)},
               "hparams": hp, "gates_thresholds": GATES}

    # ---- main adapter on OUR zeta ----
    print("[train] adapter_main (zeta, paraphrase-9 recipe)")
    ad_main = train_adapter(tr["zeta"], tr["dz"], tr["lab"], lbank, hp, "main")
    print("[eval] adapter_main")
    results["adapter_main"] = eval_adapter(ad_main, ho["zeta"], ho["lab"],
                                           classes, txt, sets)

    # ---- G-R0b(2): state-residualized zeta - r(z_t), RidgeCV week0 pattern ----
    from sklearn.linear_model import RidgeCV
    ridge = RidgeCV(alphas=np.logspace(-1, 4, 11))
    ridge.fit(tr["zt"], tr["zeta"])
    zres_tr = (tr["zeta"] - ridge.predict(tr["zt"])).astype(np.float32)
    zres_ho = (ho["zeta"] - ridge.predict(ho["zt"])).astype(np.float32)
    r2 = float(ridge.score(ho["zt"], ho["zeta"]))
    print(f"[ridge] alpha {float(ridge.alpha_):.3g} | held-out R2(z_t->zeta) {r2:.4f}")
    results["ridge_zt_to_zeta"] = {"alpha": float(ridge.alpha_), "heldout_r2": r2}

    print("[train] adapter_resid (zeta - r(z_t))")
    ad_res = train_adapter(zres_tr, tr["dz"], tr["lab"], lbank, hp, "resid")
    print("[eval] adapter_resid")
    results["adapter_resid"] = eval_adapter(ad_res, zres_ho, ho["lab"],
                                            classes, txt, sets)

    # ---- G-R0b(1): text-free MLP classifier controls ----
    print("[control] text-free MLP classifiers")
    m1, _ = train_mlp(tr["zeta"], tr["lab"], ho["zeta"], ho["lab"], tag="mlp_zeta")
    m2, _ = train_mlp(zres_tr, tr["lab"], zres_ho, ho["lab"], tag="mlp_zres")
    m3, _ = train_mlp(tr["zt"], tr["lab"], ho["zt"], ho["lab"], tag="mlp_state_only")
    results["controls_textfree"] = {
        "mlp_zeta_top1": m1, "mlp_zeta_resid_top1": m2, "mlp_state_only_top1": m3}

    # ---- gate verdicts ----
    am = results["adapter_main"]
    verdicts = {
        "G-R0a_canonical": {"value": am["canonical"]["top1"],
                            "threshold": GATES["G-R0a_canonical"],
                            "pass": am["canonical"]["top1"] >= GATES["G-R0a_canonical"]},
        "G-R0b_controls_present": {"pass": True,
                                   "note": "mlp_zeta / resid variant in same table"},
        "G-R0c_unseen": {"value": am["unseen_their"]["top1"],
                         "threshold": GATES["G-R0c_unseen"],
                         "pass": am["unseen_their"]["top1"] >= GATES["G-R0c_unseen"]},
        "G-R0c_third_ours": {"value": am["third_ours"]["top1"],
                             "threshold": GATES["G-R0c_third"],
                             "pass": am["third_ours"]["top1"] >= GATES["G-R0c_third"]},
    }
    verdicts["ALL_PASS"] = all(v.get("pass") for v in verdicts.values())
    results["gates"] = verdicts
    results["third_set_ours_prompts"] = THIRD_SET_OURS

    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": ad_main.state_dict(), "classes": classes,
                "in_dim": tr["zeta"].shape[1], "recipe": hp,
                "phase1_ckpt": BK["phase1_ckpt"]}, OUT / "adapter_main.pt")
    torch.save({"state_dict": ad_res.state_dict(), "classes": classes,
                "in_dim": tr["zeta"].shape[1], "recipe": hp,
                "ridge_alpha": float(ridge.alpha_),
                "ridge_coef": ridge.coef_.astype(np.float32),
                "ridge_intercept": ridge.intercept_.astype(np.float32)},
               OUT / "adapter_resid.pt")
    json.dump(results, open(OUT / "r0_results.json", "w"), indent=1)
    print(f"[saved] {OUT/'r0_results.json'} | gates: "
          + ", ".join(f"{k}={'PASS' if v.get('pass') else 'FAIL'}"
                      for k, v in verdicts.items() if k != 'ALL_PASS')
          + f" | ALL={'PASS' if verdicts['ALL_PASS'] else 'FAIL'}")


if __name__ == "__main__":
    main()
