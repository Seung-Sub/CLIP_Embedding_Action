"""W-A 확정실험 판정 (DESIGN_wrist_fusion_unified_v1 확정 티어 + RESULT_wrist_screening §5).

입력: outputs/eval/runs/{matchedbase,wristpatch}_s{1,2,3}_{correct,wrong}_50r[_r2,_r3]/episodes.jsonl
  - 재시도 런 병합: (task_id, episode_index) 키로 dedupe, 나중 런(_r2 < _r3) 우선.
판정 (사전등록):
  1) 주 게이트: paired per-task bootstrap 10k — wristpatch vs matchedbase, 시드 풀링(태스크당
     150ep 평균). CI>0 = 승리. per-seed 표 + 시드-일관성(3시드 전부 Δ>0?) 병기.
     에피소드-단위 paired(시드×태스크×ep, 동일 init_state) 민감도 병기.
  2) 언어 공동기준: wristpatch correct−wrong ≥ +70pp (스크리닝 65.5 유보밴드의 재판정 —
     확정실험은 유보 없음: ≥70 통과 / <70 미달. F3 즉결선 <60/<55 유지 병기).
  3) per-task 표: 설계 파지-지배 [3,4,6] + 스크리닝 이득 집중 [3,4,9] 모두 표시.
  4) F5: matchedbase correct 85~88 재현 (시드별 + 풀링).
  5) 스크리닝(+5.5pp) 대비 안정성.
출력: outputs/analysis/wa_confirm_results.json + stdout 표.
"""
import json
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[1]
RUNS = WS / "outputs" / "eval" / "runs"
OUT = WS / "outputs" / "analysis" / "wa_confirm_results.json"
ARMS = ["matchedbase", "wristpatch"]
MODES = ["correct", "wrong"]
SEEDS = [1, 2, 3]
N_TASK, N_EP, N_BOOT, BSEED = 10, 50, 10000, 0
GRASP_DESIGN = [3, 4, 6]     # 설계서 §5.3 파지-지배 예측
GRASP_SCREEN = [3, 4, 9]     # 스크리닝 이득 집중 (t4 +35, t9 +25, t3 +10)


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


def paired_bootstrap(a, b, n_boot=N_BOOT, seed=BSEED):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(a), size=(n_boot, len(a)))
    diffs = a[idx].mean(1) - b[idx].mean(1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "diff": float(a.mean() - b.mean()),
            "ci95": [float(lo), float(hi)],
            "p_a_gt_b": float((diffs > 0).mean()), "n_units": len(a)}


def main():
    data, complete = {}, True
    for arm in ARMS:
        for s in SEEDS:
            for mode in MODES:
                tag = f"{arm}_s{s}_{mode}_50r"
                recs, used = load_merged(tag)
                per = {t: {} for t in range(N_TASK)}
                for (t, e), r in recs.items():
                    per[t][e] = int(bool(r["success"]))
                miss = {t: N_EP - len(per[t]) for t in range(N_TASK)
                        if len(per[t]) < N_EP}
                if miss:
                    complete = False
                    print(f"[!] {tag}: 미완 태스크 {miss}")
                data[(arm, s, mode)] = {"per": per, "used": used, "n": len(recs)}
                print(f"{tag}: {len(recs)} eps (dirs: {','.join(used) or 'NONE'})")
    if not complete:
        print("[!] 미완 런 존재 — 가용 데이터로 계산, 미완 명시")

    def task_sr(arm, s, mode, t):
        eps = data[(arm, s, mode)]["per"][t]
        return 100.0 * np.mean(list(eps.values())) if eps else float("nan")

    def pooled_task_sr(arm, mode, t):        # 시드 풀링: 태스크당 3×50ep
        vals = []
        for s in SEEDS:
            vals += list(data[(arm, s, mode)]["per"][t].values())
        return 100.0 * np.mean(vals) if vals else float("nan")

    def mean_sr(arm, mode, s=None):
        if s is None:
            return float(np.nanmean([pooled_task_sr(arm, mode, t)
                                     for t in range(N_TASK)]))
        return float(np.nanmean([task_sr(arm, s, mode, t)
                                 for t in range(N_TASK)]))

    res = {"complete": complete, "n_boot": N_BOOT, "bootstrap_seed": BSEED,
           "runs": {f"{a}_s{s}_{m}": {"n_eps": data[(a, s, m)]["n"],
                                      "dirs": data[(a, s, m)]["used"],
                                      "mean_sr": mean_sr(a, m, s)}
                    for a in ARMS for s in SEEDS for m in MODES}}

    # ---- 1) 주 게이트: 풀링 paired per-task bootstrap ----
    a_task = [pooled_task_sr("wristpatch", "correct", t) for t in range(N_TASK)]
    b_task = [pooled_task_sr("matchedbase", "correct", t) for t in range(N_TASK)]
    res["paired_task_pooled"] = paired_bootstrap(a_task, b_task)
    # per-seed paired task bootstrap + 시드 일관성
    res["paired_task_per_seed"] = {}
    for s in SEEDS:
        res["paired_task_per_seed"][s] = paired_bootstrap(
            [task_sr("wristpatch", s, "correct", t) for t in range(N_TASK)],
            [task_sr("matchedbase", s, "correct", t) for t in range(N_TASK)])
    res["seed_consistency_all_positive"] = bool(
        all(res["paired_task_per_seed"][s]["diff"] > 0 for s in SEEDS))
    # 에피소드-단위 paired (시드×태스크×ep — 동일 init_state·동일 시드 짝)
    a_ep, b_ep = [], []
    for s in SEEDS:
        for t in range(N_TASK):
            pa = data[("wristpatch", s, "correct")]["per"][t]
            pb = data[("matchedbase", s, "correct")]["per"][t]
            for e in sorted(set(pa) & set(pb)):
                a_ep.append(pa[e] * 100.0)
                b_ep.append(pb[e] * 100.0)
    res["paired_episode_pooled"] = paired_bootstrap(a_ep, b_ep)
    # wrong 모드 참고
    res["paired_task_pooled_wrong"] = paired_bootstrap(
        [pooled_task_sr("wristpatch", "wrong", t) for t in range(N_TASK)],
        [pooled_task_sr("matchedbase", "wrong", t) for t in range(N_TASK)])

    # ---- 2) 언어 공동기준 (확정실험 = 유보 없음) ----
    res["lang"] = {}
    for arm in ARMS:
        cw_pool = mean_sr(arm, "correct") - mean_sr(arm, "wrong")
        per_seed = {s: mean_sr(arm, "correct", s) - mean_sr(arm, "wrong", s)
                    for s in SEEDS}
        if cw_pool >= 70:
            verdict = "PASS(≥+70)"
        elif cw_pool >= 60:
            verdict = "FAIL(+70 미달; F3 즉결선 60 위)"
        elif cw_pool >= 55:
            verdict = "F3-근접(55-60)"
        else:
            verdict = "F3(<+55 언어 화폐 훼손)"
        res["lang"][arm] = {"c_minus_w_pooled": float(cw_pool),
                            "c_minus_w_per_seed": {k: float(v) for k, v
                                                   in per_seed.items()},
                            "verdict": verdict}

    # ---- 3) per-task / grasp ----
    res["per_task_pooled"] = {
        t: {"base_c": pooled_task_sr("matchedbase", "correct", t),
            "wrist_c": pooled_task_sr("wristpatch", "correct", t),
            "base_w": pooled_task_sr("matchedbase", "wrong", t),
            "wrist_w": pooled_task_sr("wristpatch", "wrong", t)}
        for t in range(N_TASK)}
    for name, ts in [("grasp_design_t3t4t6", GRASP_DESIGN),
                     ("grasp_screen_t3t4t9", GRASP_SCREEN)]:
        res[name] = {
            "delta_mean": float(np.nanmean(
                [res["per_task_pooled"][t]["wrist_c"]
                 - res["per_task_pooled"][t]["base_c"] for t in ts])),
            "per_task": {t: res["per_task_pooled"][t] for t in ts}}

    # ---- 4) F5 + 5) 스크리닝 대비 ----
    base_pool = mean_sr("matchedbase", "correct")
    res["F5_baseline"] = {
        "pooled": base_pool, "band_85_88_pooled": bool(85 <= base_pool <= 88),
        "per_seed": {s: mean_sr("matchedbase", "correct", s) for s in SEEDS}}
    res["vs_screening"] = {
        "screening_delta_pp": 5.5,
        "confirm_delta_pp": res["paired_task_pooled"]["diff"],
        "screening_cw_wristpatch": 65.5,
        "confirm_cw_wristpatch": res["lang"]["wristpatch"]["c_minus_w_pooled"]}

    pt = res["paired_task_pooled"]
    sig_win, sig_loss = pt["ci95"][0] > 0, pt["ci95"][1] < 0
    lang_ok = res["lang"]["wristpatch"]["verdict"].startswith("PASS")
    res["verdict"] = {
        "paired_sig_win": bool(sig_win), "paired_sig_loss_F4": bool(sig_loss),
        "lang_pass": bool(lang_ok),
        "gate": ("CONFIRMED (SR CI>0 AND c−w≥+70)" if sig_win and lang_ok
                 else "SR-WIN / LANG-FAIL" if sig_win
                 else "F4(유의 하락)" if sig_loss
                 else "NULL(CI∋0)")}

    # ---- stdout ----
    print("\n=== per-task SR (%) — pooled 3seeds, correct ===")
    print(f"{'task':>4} {'base':>6} {'wrist':>6} {'Δ':>6}   "
          f"{'base_w':>6} {'wrist_w':>7}")
    for t in range(N_TASK):
        d = res["per_task_pooled"][t]
        mark = (" ★설계파지" if t in GRASP_DESIGN else "") + \
               (" ☆스크리닝이득" if t in GRASP_SCREEN else "")
        print(f"{t:>4} {d['base_c']:>6.1f} {d['wrist_c']:>6.1f} "
              f"{d['wrist_c'] - d['base_c']:>+6.1f}   "
              f"{d['base_w']:>6.1f} {d['wrist_w']:>7.1f}{mark}")
    print(f"mean {mean_sr('matchedbase', 'correct'):>6.1f} "
          f"{mean_sr('wristpatch', 'correct'):>6.1f} "
          f"{mean_sr('wristpatch', 'correct') - mean_sr('matchedbase', 'correct'):>+6.1f}   "
          f"{mean_sr('matchedbase', 'wrong'):>6.1f} "
          f"{mean_sr('wristpatch', 'wrong'):>7.1f}")
    print(f"\n[주 게이트] pooled paired(task) Δ={pt['diff']:+.2f}pp "
          f"CI[{pt['ci95'][0]:+.2f},{pt['ci95'][1]:+.2f}] p={pt['p_a_gt_b']:.4f}")
    for s in SEEDS:
        ps = res["paired_task_per_seed"][s]
        print(f"  seed{s}: Δ={ps['diff']:+.2f}pp "
              f"CI[{ps['ci95'][0]:+.2f},{ps['ci95'][1]:+.2f}] "
              f"p={ps['p_a_gt_b']:.4f} "
              f"(base {ps['mean_b']:.1f} → wrist {ps['mean_a']:.1f})")
    print(f"  시드 일관성(3/3 Δ>0): {res['seed_consistency_all_positive']}")
    pe = res["paired_episode_pooled"]
    print(f"  paired(episode) Δ={pe['diff']:+.2f}pp "
          f"CI[{pe['ci95'][0]:+.2f},{pe['ci95'][1]:+.2f}] "
          f"p={pe['p_a_gt_b']:.4f} (n={pe['n_units']})")
    for arm in ARMS:
        L = res["lang"][arm]
        seeds_str = " ".join(f"s{s}:{L['c_minus_w_per_seed'][s]:+.1f}"
                             for s in SEEDS)
        print(f"[언어] {arm}: c−w={L['c_minus_w_pooled']:+.1f}pp "
              f"({seeds_str}) → {L['verdict']}")
    print(f"[F5] base pooled {base_pool:.1f}% (밴드 85-88: "
          f"{res['F5_baseline']['band_85_88_pooled']}) per-seed "
          f"{ {s: round(v, 1) for s, v in res['F5_baseline']['per_seed'].items()} }")
    print(f"[안정성] 스크리닝 +5.5pp → 확정 {pt['diff']:+.1f}pp; "
          f"c−w 65.5 → {res['lang']['wristpatch']['c_minus_w_pooled']:.1f}")
    print(f"\n종합: {res['verdict']['gate']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float))
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
