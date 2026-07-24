"""R-0b — is the language<->action-effect alignment genuine, or state-carried?

Train the SemanticAdapter FROM SCRATCH on state-residualized effect vectors
zeta_res = zeta - r(z_t), r = RidgeCV(z_t -> zeta) fit on train segments
(exact R-0 ridge: alpha grid logspace(-1,4,11), week0 f2 pattern), then run the
full R-0/R-1 battery on the residual adapter.

PREMISE CORRECTION (verified in r0_train_eval.py before writing this): R-0's
"adapter, state-residual 0.749/0.543" row was ALREADY a from-scratch training
on zeta_res (train_adapter(zres_tr, ...) eval on zres_ho) — it was NOT a frozen
raw-zeta adapter unfairly evaluated on residuals. R-0b therefore:
  (a) reproduces that run (seed 0 must match canonical 0.7485 bit-for-bit),
  (b) adds a per-dim standardized-residual variant (std stats reported first),
  (c) seed robustness 0/1/2 for both variants,
  (d) the full R-1 retrieval battery (correct/swap/paraphrase/shuffle/nonsense
      margins) on the residual adapters,
  (e) per-class residual-sensitivity table,
  (f) direction-only probe (forward/left/right; backward excluded, n_bank=5
      known-defect class): direction-restricted scoring of the 8-way adapters
      + 3-way adapters trained on direction segments with direction-only ridge.

PRE-REGISTERED INTERPRETATION (fixed BEFORE running the blind parts; the only
number known in advance is variant-A seed-0 canonical = 0.7485 from R-0):
  primary metric = canonical held-out top-1 of the BEST residual variant
  (raw or standardized), mean over seeds 0/1/2.
    >= 0.90      -> alignment is genuinely about action effects; state was
                    redundant, not constitutive -> C3' keeps its strong
                    (state-free) form.
    0.60 - 0.90  -> mixed: effects are partially encoded state-free -> C3'
                    holds with a QUANTIFIED state share (report the gap to the
                    raw-zeta adapter and to the text-free MLP on zeta_res).
    <= 0.60      -> alignment is mostly state-carried -> C3' must be reframed
                    honestly as "scene-effect retrieval".
  secondary (pre-registered): direction classes are predicted to lose LESS
  accuracy under residualization than approach/grasp/place (their label is net
  EE displacement, not trajectory phase); R-1 margins on the residual adapter
  should preserve the correct >> shuffled >> nonsense ordering even if absolute
  margins shrink.

CPU. RUN: OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= \
    python scratchpad/rseries/r0b_resid_adapter.py
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
from rseries_common import (OUT, THIRD_SET_OURS, SemanticAdapter, load_banks,
                            text_embeddings)
from r0_train_eval import (build_language_bank, eval_adapter, train_adapter,
                           train_mlp)
from r1_harness import NONSENSE, UNRELATED, run_group, shuffled_variants

SEEDS = [0, 1, 2]
DIR_CLASSES = ["move the black bowl forward", "move the black bowl left",
               "move the black bowl right"]


def fit_ridge(zt_tr, zeta_tr, zt_ho, zeta_ho, tag):
    from sklearn.linear_model import RidgeCV
    ridge = RidgeCV(alphas=np.logspace(-1, 4, 11))
    ridge.fit(zt_tr, zeta_tr)
    r2 = float(ridge.score(zt_ho, zeta_ho))
    print(f"[ridge:{tag}] alpha {float(ridge.alpha_):.3g} | held-out R2 {r2:.4f}")
    res_tr = (zeta_tr - ridge.predict(zt_tr)).astype(np.float32)
    res_ho = (zeta_ho - ridge.predict(zt_ho)).astype(np.float32)
    return ridge, r2, res_tr, res_ho


def scale_stats(name, X):
    sd = X.std(0)
    st = {"per_dim_std_min": float(sd.min()), "p5": float(np.percentile(sd, 5)),
          "median": float(np.median(sd)), "p95": float(np.percentile(sd, 95)),
          "max": float(sd.max()), "total_var": float(X.var(0).sum())}
    print(f"[scale] {name:10s} per-dim std min {st['per_dim_std_min']:.4g} "
          f"p5 {st['p5']:.4g} med {st['median']:.4g} p95 {st['p95']:.4g} "
          f"max {st['max']:.4g} | total var {st['total_var']:.4g}")
    return st


def r1_battery(adapter, bank_vecs, lab_bank, prov, classes, txt, tag):
    """Exact R-1 groups + swap matrix on a given adapter/bank."""
    with torch.no_grad():
        q_a_bank = adapter.P_action(torch.tensor(bank_vecs)).numpy()
    _, _, their = load_banks()
    unseen = their["unseen_templates"]
    queries = {"correct": [(c, c) for c in classes],
               "paraphrase_unseen_their": [(p, c) for c, ps in unseen.items()
                                           for p in ps],
               "paraphrase_third_ours": [(p, c) for c, ps in THIRD_SET_OURS.items()
                                         for p in ps],
               "shuffled": [(s, None) for c in classes
                            for s in shuffled_variants(c)],
               "nonsense": [(s, None) for s in NONSENSE],
               "unrelated": [(s, None) for s in UNRELATED]}
    res = {"groups": {}}
    print(f"-- R-1 battery [{tag}] --")
    for name, qs in queries.items():
        run_group(name, qs, None, txt, adapter.P_text, q_a_bank, lab_bank,
                  prov, classes, res)
    corr = {r["query"]: r["retrieved_class"]
            for r in res["groups"]["correct"]["rows"]}
    swap_ok = sum(1 for c in classes for c2 in classes if c != c2
                  and corr[c2] == c2)
    n_pairs = len(classes) * (len(classes) - 1)
    res["swap_sensitivity"] = {"ok_pairs": swap_ok, "n_pairs": n_pairs,
                               "frac": swap_ok / n_pairs,
                               "command_to_retrieved": corr}
    print(f"[swap:{tag}] {swap_ok}/{n_pairs}")
    return res


def main():
    BK = pickle.load(open(OUT / "effect_bank_ours.pkl", "rb"))
    classes = BK["classes"]
    tr, ho = BK["train"], BK["heldout"]
    bank, hard_bank, their = load_banks()
    para_aug = their["para_aug_per_class"]
    hp = dict(their["hparams"])
    hp["warmup"] = hp.pop("warmup", 200)
    unseen = their["unseen_templates"]
    sets = {"canonical": {c: [c] for c in classes},
            "seen_para9": para_aug,
            "unseen_their": unseen,
            "third_theirs": their["novel3_theirs"],
            "third_ours": THIRD_SET_OURS}

    need = set(classes)
    for ps in list(sets.values()) + [{c: hard_bank.get(c, []) for c in classes}]:
        for l in ps.values():
            need.update(l)
    need.update(s for c in classes for s in shuffled_variants(c))
    need.update(NONSENSE); need.update(UNRELATED)
    txt = text_embeddings(need)
    lbank = build_language_bank(classes, para_aug, hard_bank, txt)

    r0 = json.load(open(OUT / "r0_results.json"))
    results = {
        "substrate": BK["substrate"], "split_seed": BK["split_seed"],
        "premise_correction": "R-0 adapter_resid was ALREADY trained from "
            "scratch on zeta_res (r0_train_eval.py); its 0.7485 canonical is "
            "the from-scratch number. R-0b adds std-variant, seeds, R-1 "
            "battery, per-class sensitivity, direction probe.",
        "preregistration": {
            "primary": "canonical top-1, best residual variant, mean over "
                       "seeds 0/1/2",
            "bands": {">=0.90": "state redundant -> C3' strong/state-free",
                      "0.60-0.90": "mixed -> C3' with quantified state share",
                      "<=0.60": "state-carried -> reframe as scene-effect "
                                "retrieval"},
            "secondary": "direction classes lose less than approach/grasp/"
                         "place; correct>>shuffled>>nonsense margin ordering "
                         "survives residualization",
            "known_in_advance": "variant resid_raw seed0 canonical 0.7485 "
                                "(from R-0)"},
        "r0_reference": {
            "adapter_main_canonical": r0["adapter_main"]["canonical"]["top1"],
            "adapter_resid_canonical": r0["adapter_resid"]["canonical"]["top1"],
            "mlp_zeta": r0["controls_textfree"]["mlp_zeta_top1"],
            "mlp_zeta_resid": r0["controls_textfree"]["mlp_zeta_resid_top1"],
            "mlp_state_only": r0["controls_textfree"]["mlp_state_only_top1"],
            "ridge": r0["ridge_zt_to_zeta"], "chance": r0["chance"]},
    }

    # ---- step 1: ridge + residuals (R-0 exact) ------------------------------
    ridge, r2, zres_tr, zres_ho = fit_ridge(tr["zt"], tr["zeta"],
                                            ho["zt"], ho["zeta"], "full")
    results["ridge_zt_to_zeta"] = {"alpha": float(ridge.alpha_),
                                   "heldout_r2": r2,
                                   "matches_r0": bool(abs(r2 - r0["ridge_zt_to_zeta"]["heldout_r2"]) < 1e-6)}

    # residual scale check (pre-condition for the std variant)
    results["residual_scale"] = {
        "zeta": scale_stats("zeta", tr["zeta"]),
        "zeta_res": scale_stats("zeta_res", zres_tr)}
    results["residual_scale"]["var_retained_frac"] = (
        results["residual_scale"]["zeta_res"]["total_var"]
        / results["residual_scale"]["zeta"]["total_var"])
    print(f"[scale] residual retains {results['residual_scale']['var_retained_frac']:.3f} of zeta total variance")

    mu, sd = zres_tr.mean(0), np.maximum(zres_tr.std(0), 1e-6)
    zstd_tr = ((zres_tr - mu) / sd).astype(np.float32)
    zstd_ho = ((zres_ho - mu) / sd).astype(np.float32)

    variants = {"resid_raw": (zres_tr, zres_ho),
                "resid_std": (zstd_tr, zstd_ho)}

    # ---- step 2: from-scratch adapters, both variants, 3 seeds --------------
    adapters = {}
    results["variants"] = {}
    for vname, (vtr, vho) in variants.items():
        vres = {"seeds": {}}
        for s in SEEDS:
            hps = dict(hp); hps["seed"] = s
            tag = f"{vname}_s{s}"
            ad = train_adapter(vtr, tr["dz"], tr["lab"], lbank, hps, tag)
            ev = eval_adapter(ad, vho, ho["lab"], classes, txt, sets)
            if s != 0:                      # keep JSON compact: details seed 0 only
                ev = {k: {"top1": v["top1"], "top1_meanemb": v["top1_meanemb"]}
                      for k, v in ev.items()}
            vres["seeds"][s] = ev
            if s == 0:
                adapters[vname] = ad
        cans = [vres["seeds"][s]["canonical"]["top1"] for s in SEEDS]
        uns = [vres["seeds"][s]["unseen_their"]["top1"] for s in SEEDS]
        vres["canonical_mean"] = float(np.mean(cans))
        vres["canonical_std"] = float(np.std(cans))
        vres["unseen_their_mean"] = float(np.mean(uns))
        print(f"[{vname}] canonical {np.mean(cans):.4f}±{np.std(cans):.4f} "
              f"seeds {cans} | unseen {np.mean(uns):.4f}")
        results["variants"][vname] = vres
    results["reproduces_r0_adapter_resid"] = bool(
        abs(results["variants"]["resid_raw"]["seeds"][0]["canonical"]["top1"]
            - r0["adapter_resid"]["canonical"]["top1"]) < 1e-9)

    # ---- step 3: text-free MLP controls on the residuals --------------------
    print("[control] text-free MLPs on residuals")
    m_raw, _ = train_mlp(zres_tr, tr["lab"], zres_ho, ho["lab"], tag="mlp_zres_raw")
    m_std, _ = train_mlp(zstd_tr, tr["lab"], zstd_ho, ho["lab"], tag="mlp_zres_std")
    results["controls_textfree"] = {"mlp_zres_raw": m_raw, "mlp_zres_std": m_std}

    # ---- step 4: R-1 retrieval battery on residual adapters -----------------
    results["r1_battery"] = {}
    for vname, (vtr, _) in variants.items():
        results["r1_battery"][f"{vname}_s0"] = r1_battery(
            adapters[vname], vtr, tr["lab"], tr["prov"], classes, txt,
            f"{vname}_s0")

    # ---- step 5: per-class residual sensitivity (canonical, seed 0) ---------
    sens = {}
    pc_main = r0["adapter_main"]["canonical"]["per_class"]
    for vname in variants:
        pc_v = results["variants"][vname]["seeds"][0]["canonical"]["per_class"]
        sens[vname] = {c: {"main": pc_main[c], "resid": pc_v[c],
                           "delta": (None if pc_v[c] is None
                                     else round(pc_v[c] - pc_main[c], 4))}
                       for c in classes}
    results["per_class_sensitivity"] = sens
    print("[sens] canonical per-class delta (resid_std s0 - main):")
    for c in classes:
        d = sens["resid_std"][c]
        print(f"   {c:38s} main {d['main']:.3f} resid {d['resid']:.3f} "
              f"delta {d['delta']:+.3f}")

    # ---- step 6: direction-only probe (forward/left/right) ------------------
    dmask_tr = np.isin(tr["lab"], [classes.index(c) for c in DIR_CLASSES])
    dmask_ho = np.isin(ho["lab"], [classes.index(c) for c in DIR_CLASSES])
    old2new = {classes.index(c): i for i, c in enumerate(DIR_CLASSES)}
    dlab_tr = np.array([old2new[int(x)] for x in tr["lab"][dmask_tr]])
    dlab_ho = np.array([old2new[int(x)] for x in ho["lab"][dmask_ho]])
    print(f"[dir] train {dmask_tr.sum()} segs, heldout {dmask_ho.sum()} segs "
          f"(counts {np.bincount(dlab_ho)}), majority-chance "
          f"{np.bincount(dlab_ho).max() / len(dlab_ho):.3f}")
    dprobe = {"classes": DIR_CLASSES,
              "n_train": int(dmask_tr.sum()), "n_heldout": int(dmask_ho.sum()),
              "chance_majority": float(np.bincount(dlab_ho).max() / len(dlab_ho))}
    dsets = {"canonical": {c: [c] for c in DIR_CLASSES},
             "unseen_their": {c: unseen[c] for c in DIR_CLASSES},
             "third_ours": {c: THIRD_SET_OURS[c] for c in DIR_CLASSES}}

    # (a) direction-restricted scoring of the 8-way adapters
    ck_main = torch.load(OUT / "adapter_main.pt", map_location="cpu",
                         weights_only=False)
    ad_main = SemanticAdapter(in_dim=ck_main["in_dim"]).eval()
    ad_main.load_state_dict(ck_main["state_dict"])
    rows8 = {"main_zeta": (ad_main, ho["zeta"][dmask_ho]),
             "resid_raw_s0": (adapters["resid_raw"], zres_ho[dmask_ho]),
             "resid_std_s0": (adapters["resid_std"], zstd_ho[dmask_ho])}
    dprobe["restricted_8way_adapters"] = {}
    for rname, (ad, vho) in rows8.items():
        ev = eval_adapter(ad, vho, dlab_ho, DIR_CLASSES, txt, dsets)
        dprobe["restricted_8way_adapters"][rname] = ev
        print(f"[dir:restrict {rname}] canonical {ev['canonical']['top1']:.4f}")

    # (b) direction-only ridge + 3-way from-scratch adapters
    ridge_d, r2_d, dres_tr, dres_ho = fit_ridge(
        tr["zt"][dmask_tr], tr["zeta"][dmask_tr],
        ho["zt"][dmask_ho], ho["zeta"][dmask_ho], "dir-only")
    dprobe["ridge_dir_only"] = {"alpha": float(ridge_d.alpha_),
                                "heldout_r2": r2_d}
    dmu = dres_tr.mean(0)
    dsd = np.maximum(dres_tr.std(0), 1e-6)
    dstd_tr = ((dres_tr - dmu) / dsd).astype(np.float32)
    dstd_ho = ((dres_ho - dmu) / dsd).astype(np.float32)
    lbank_d = build_language_bank(DIR_CLASSES,
                                  {c: para_aug[c] for c in DIR_CLASSES},
                                  {c: hard_bank.get(c, []) for c in DIR_CLASSES},
                                  txt)
    dvariants = {"zeta_raw": (tr["zeta"][dmask_tr], ho["zeta"][dmask_ho]),
                 "resid_dir": (dres_tr, dres_ho),
                 "resid_dir_std": (dstd_tr, dstd_ho)}
    dprobe["adapters_3way"] = {}
    for vname, (vtr, vho) in dvariants.items():
        cans = []
        for s in SEEDS:
            hps = dict(hp); hps["seed"] = s
            ad = train_adapter(vtr, tr["dz"][dmask_tr], dlab_tr, lbank_d, hps,
                               f"dir_{vname}_s{s}")
            ev = eval_adapter(ad, vho, dlab_ho, DIR_CLASSES, txt, dsets)
            cans.append(ev["canonical"]["top1"])
            if s == 0:
                dprobe["adapters_3way"][vname] = {"seed0": ev}
        dprobe["adapters_3way"][vname]["canonical_mean"] = float(np.mean(cans))
        dprobe["adapters_3way"][vname]["canonical_seeds"] = cans
        print(f"[dir:3way {vname}] canonical {np.mean(cans):.4f} seeds {cans}")

    # (c) 3-way text-free MLPs
    dm1, _ = train_mlp(tr["zeta"][dmask_tr], dlab_tr,
                       ho["zeta"][dmask_ho], dlab_ho, tag="dir_mlp_zeta")
    dm2, _ = train_mlp(dres_tr, dlab_tr, dres_ho, dlab_ho, tag="dir_mlp_resid")
    dm3, _ = train_mlp(tr["zt"][dmask_tr], dlab_tr,
                       ho["zt"][dmask_ho], dlab_ho, tag="dir_mlp_state_only")
    dprobe["mlp_3way"] = {"zeta": dm1, "resid_dir": dm2, "state_only": dm3}
    results["direction_probe"] = dprobe

    # ---- save ---------------------------------------------------------------
    OUT.mkdir(parents=True, exist_ok=True)
    for vname, ad in adapters.items():
        torch.save({"state_dict": ad.state_dict(), "classes": classes,
                    "in_dim": tr["zeta"].shape[1], "recipe": hp, "seed": 0,
                    "variant": vname,
                    "ridge_alpha": float(ridge.alpha_),
                    "ridge_coef": ridge.coef_.astype(np.float32),
                    "ridge_intercept": ridge.intercept_.astype(np.float32),
                    "std_mu": (mu if vname == "resid_std" else None),
                    "std_sd": (sd if vname == "resid_std" else None)},
                   OUT / f"r0b_adapter_{vname}.pt")
    json.dump(results, open(OUT / "r0b_results.json", "w"), indent=1)
    best = max(results["variants"], key=lambda v: results["variants"][v]["canonical_mean"])
    bm = results["variants"][best]["canonical_mean"]
    band = (">=0.90 state-free" if bm >= 0.90
            else "0.60-0.90 mixed" if bm > 0.60
            else "<=0.60 state-carried")
    results_verdict = {"best_variant": best, "canonical_mean": bm,
                       "preregistered_band": band}
    print(f"[saved] {OUT/'r0b_results.json'} | best {best} canonical "
          f"{bm:.4f} -> band {band}")
    # re-save with verdict included
    results["verdict"] = results_verdict
    json.dump(results, open(OUT / "r0b_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
