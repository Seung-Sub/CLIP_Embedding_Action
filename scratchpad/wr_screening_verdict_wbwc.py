"""WristCond-v1 W-B/W-C 스크리닝 판정 — 시블링 wr_screening_verdict.py 방법론 동일
(paired per-task bootstrap 10k seed0 주 기준 + 에피소드-단위 민감도 + A4 언어 유보 밴드),
테스트 팔만 {wristdelta(W-B), dualstd(W-C)} 로 확장. 기준선 = 동일 matchedbase JSONL.

출력: outputs/analysis/wr_screening_results_wbwc.json + stdout 표.
"""
import json
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[1]
RUNS = WS / "outputs" / "eval" / "runs"
OUT = WS / "outputs" / "analysis" / "wr_screening_results_wbwc.json"
TEST_ARMS = ["wristdelta", "dualstd"]
BASE = "matchedbase"
MODES = ["correct", "wrong"]
N_TASK, N_EP, N_BOOT, SEED = 10, 20, 10000, 0
GRASP_TASKS = [3, 4, 9]   # 코디네이터 지정 파지-지배 포커스 (시블링 문서 이득 집중 태스크)


def load_merged(tag):
    recs = {}
    dirs = [RUNS / tag] + [RUNS / f"{tag}_r{r}" for r in (2, 3)]
    used = []
    for d in dirs:
        f = d / "episodes.jsonl"
        if not f.exists():
            continue
        used.append(d.name)
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            recs[(r["task_id"], r["episode_index"])] = r
    return recs, used


def paired_bootstrap(a, b, n_boot=N_BOOT, seed=SEED):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(a), size=(n_boot, len(a)))
    diffs = a[idx].mean(1) - b[idx].mean(1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "diff": float(a.mean() - b.mean()),
            "ci95": [float(lo), float(hi)],
            "p_a_gt_b": float((diffs > 0).mean()), "n_units": len(a)}


def lang_verdict(cw):
    if cw >= 75:
        return "PASS"
    if cw >= 65:
        return "DEFER(65-75 유보밴드 → 확증에서 재판정)"
    if cw >= 60:
        return "FAIL(+70 미달; F3 아님)"
    if cw >= 55:
        return "F3-DEFER(55-60 유보밴드)"
    return "F3(언어 화폐 훼손 <+55 즉결 NO-GO)"


def main():
    arms = TEST_ARMS + [BASE]
    data = {}
    for arm in arms:
        for mode in MODES:
            tag = f"{arm}_{mode}_20r_wr"
            recs, used = load_merged(tag)
            per = {t: {} for t in range(N_TASK)}
            for (t, e), r in recs.items():
                per[t][e] = int(bool(r["success"]))
            miss = {t: N_EP - len(per[t]) for t in range(N_TASK) if len(per[t]) < N_EP}
            assert not miss, f"{tag} 미완 태스크 {miss}"
            data[(arm, mode)] = {"per": per, "used": used, "n": len(recs)}
            print(f"{tag}: {len(recs)} eps ({','.join(used)})")

    def task_sr(arm, mode, t):
        return 100.0 * np.mean(list(data[(arm, mode)]["per"][t].values()))

    def mean_sr(arm, mode):
        return float(np.mean([task_sr(arm, mode, t) for t in range(N_TASK)]))

    res = {"runs": {f"{a}_{m}": {"n_eps": data[(a, m)]["n"],
                                 "mean_sr": mean_sr(a, m),
                                 "per_task_sr": {t: task_sr(a, m, t)
                                                 for t in range(N_TASK)}}
                    for a in arms for m in MODES}}

    base_sr = mean_sr(BASE, "correct")
    res["F5_baseline_reanchor"] = {"matchedbase_correct_sr": base_sr,
                                   "band_85_88": bool(85 <= base_sr <= 88)}
    res["lang"] = {}
    for arm in arms:
        cw = mean_sr(arm, "correct") - mean_sr(arm, "wrong")
        res["lang"][arm] = {"c_minus_w": float(cw), "verdict": lang_verdict(cw)}

    res["arms"] = {}
    for arm in TEST_ARMS:
        a_task = [task_sr(arm, "correct", t) for t in range(N_TASK)]
        b_task = [task_sr(BASE, "correct", t) for t in range(N_TASK)]
        pt = paired_bootstrap(a_task, b_task)
        a_ep, b_ep = [], []
        for t in range(N_TASK):
            pa, pb = data[(arm, "correct")]["per"][t], data[(BASE, "correct")]["per"][t]
            for e in sorted(set(pa) & set(pb)):
                a_ep.append(pa[e] * 100.0)
                b_ep.append(pb[e] * 100.0)
        pe = paired_bootstrap(a_ep, b_ep)
        pw = paired_bootstrap([task_sr(arm, "wrong", t) for t in range(N_TASK)],
                              [task_sr(BASE, "wrong", t) for t in range(N_TASK)])
        sig_win, sig_loss = pt["ci95"][0] > 0, pt["ci95"][1] < 0
        lang = res["lang"][arm]["verdict"]
        gate = ("WIN → 확증 후보" if sig_win and lang.startswith(("PASS", "DEFER"))
                else "F4(유의 하락)" if sig_loss
                else "NULL(CI∋0 — 이득 없음)")
        res["arms"][arm] = {"paired_task": pt, "paired_episode": pe,
                            "paired_task_wrong": pw,
                            "verdict": {"paired_sig_win": bool(sig_win),
                                        "paired_sig_loss_F4": bool(sig_loss),
                                        "gate": gate}}

        print(f"\n=== {arm} vs {BASE} — per-task SR (%) correct (wrong 병기) ===")
        print(f"{'task':>4} {'base':>6} {arm[:6]:>6} {'Δ':>6}   {'base_w':>6} {'arm_w':>6}")
        for t in range(N_TASK):
            mark = " ★파지" if t in GRASP_TASKS else ""
            print(f"{t:>4} {task_sr(BASE,'correct',t):>6.1f} {task_sr(arm,'correct',t):>6.1f} "
                  f"{task_sr(arm,'correct',t)-task_sr(BASE,'correct',t):>+6.1f}   "
                  f"{task_sr(BASE,'wrong',t):>6.1f} {task_sr(arm,'wrong',t):>6.1f}{mark}")
        print(f"mean {mean_sr(BASE,'correct'):>6.1f} {mean_sr(arm,'correct'):>6.1f} "
              f"{mean_sr(arm,'correct')-mean_sr(BASE,'correct'):>+6.1f}   "
              f"{mean_sr(BASE,'wrong'):>6.1f} {mean_sr(arm,'wrong'):>6.1f}")
        print(f"paired(task) Δ={pt['diff']:+.2f}pp CI[{pt['ci95'][0]:+.2f},{pt['ci95'][1]:+.2f}] "
              f"p={pt['p_a_gt_b']:.4f}")
        print(f"paired(episode) Δ={pe['diff']:+.2f}pp CI[{pe['ci95'][0]:+.2f},{pe['ci95'][1]:+.2f}] "
              f"p={pe['p_a_gt_b']:.4f} (n={pe['n_units']})")
        print(f"lang {arm}: c−w = {res['lang'][arm]['c_minus_w']:+.1f}pp → {lang}")
        print(f"gate: {gate}")

    print(f"\nlang {BASE}: c−w = {res['lang'][BASE]['c_minus_w']:+.1f}pp")
    print(f"F5 기준선 {base_sr:.1f}% (85-88 밴드: {res['F5_baseline_reanchor']['band_85_88']})")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
