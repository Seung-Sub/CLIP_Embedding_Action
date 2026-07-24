import sys, json, pickle
sys.path.insert(0, "scratchpad/rseries")
import numpy as np, torch
from rseries_common import OUT, SemanticAdapter, THIRD_SET_OURS, load_banks, text_embeddings, l2n
BK = pickle.load(open(OUT/"effect_bank_ours.pkl", "rb")); ho = BK["heldout"]; classes = BK["classes"]
bank, hard, their = load_banks()
sets = {"unseen_their": their["unseen_templates"],
        "third_theirs": their["novel3_theirs"],
        "third_ours": THIRD_SET_OURS,
        "seen_para9": their["para_aug_per_class"]}
need = {p for ps in sets.values() for l in ps.values() for p in l}
txt = text_embeddings(need)
res = {}
for name, ckf in (("adapter_main", "adapter_main.pt"), ("adapter_resid", "adapter_resid.pt")):
    ck = torch.load(OUT/ckf, map_location="cpu", weights_only=False)
    ad = SemanticAdapter(in_dim=ck["in_dim"]).eval(); ad.load_state_dict(ck["state_dict"])
    zeta = ho["zeta"]
    if name == "adapter_resid":
        zeta = zeta - (ho["zt"] @ ck["ridge_coef"].T + ck["ridge_intercept"])
    with torch.no_grad():
        qa = ad.P_action(torch.tensor(zeta.astype(np.float32))).numpy()
    res[name] = {}
    for sname, ps in sets.items():
        K = min(len(v) for v in ps.values())
        accs = []
        for k in range(K):
            with torch.no_grad():
                ql = ad.P_text(torch.tensor(np.stack([txt[ps[c][k]] for c in classes]))).numpy()
            pred = (qa @ ql.T).argmax(1)
            accs.append(float((pred == ho["lab"]).mean()))
        res[name][sname] = {"top1_mean_of_templates": float(np.mean(accs)),
                            "top1_per_template": [round(a, 4) for a in accs],
                            "worst": float(np.min(accs))}
        print(f"{name:14s} {sname:14s} per-template mean {np.mean(accs):.4f} {accs}")
d = json.load(open(OUT/"r0_results.json"))
d["per_template_rule"] = res
d["chance"] = {"heldout_majority": float(max(np.bincount(ho["lab"]))/len(ho["lab"])), "uniform": 1/8}
json.dump(d, open(OUT/"r0_results.json", "w"), indent=1)
print("saved")
