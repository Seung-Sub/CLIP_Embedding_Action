"""WristCond-v1 스크리닝 판정 (DESIGN_wrist_fusion_unified_v1 §5.2-4/5, §6).

입력: outputs/eval/runs/{matchedbase,wristpatch}_{correct,wrong}_20r_wr[_r2,_r3]/episodes.jsonl
  - 재시도 런 병합: (task_id, episode_index) 키로 dedupe, 나중 런(_r2 < _r3) 우선.
판정:
  1) paired per-task bootstrap 10k: wristpatch vs matchedbase (correct) — 주 게이트 CI>0.
     (에피소드-단위 paired 부트스트랩도 민감도 분석으로 병기 — init_state paired)
  2) 언어 공동기준: correct−wrong ≥ +70pp (A4 유보 규칙: 게이트 ±5pp = 65~75 → 확증 유보).
     F3: c−w < +60pp → NO-GO (±5pp: 55~60 유보, <55 즉결).
  3) per-task 표 (t3/t4/t6 파지-지배 태스크 강조).
  4) F5: matchedbase correct 85~88 재현 점검.
출력: outputs/analysis/wr_screening_results.json + stdout 표.
"""
import json
import sys
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[1]
RUNS = WS / "outputs" / "eval" / "runs"
OUT = WS / "outputs" / "analysis" / "wr_screening_results.json"
ARMS = ["matchedbase", "wristpatch"]
MODES = ["correct", "wrong"]
N_TASK, N_EP, N_BOOT, SEED = 10, 20, 10000, 0
GRASP_TASKS = [3, 4, 6]        # 설계 예측: wrist 이득 집중 (swap-neither/파지실패 태스크)


def load_merged(tag):
    """tag, tag_r2, tag_r3 순서로 episodes.jsonl 병합 — 나중 런이 같은
    (task_id, episode_index) 키를 덮어씀 (retry 규율)."""
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


def sr_by_task(recs):
    per = {t: {} for t in range(N_TASK)}
    for (t, e), r in recs.items():
        per[t][e] = int(bool(r["success"]))
    return per


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


def main():
    data, complete = {}, True
    for arm in ARMS:
        for mode in MODES:
            tag = f"{arm}_{mode}_20r_wr"
            recs, used = load_merged(tag)
            per = sr_by_task(recs)
            n_missing = {t: N_EP - len(per[t]) for t in range(N_TASK) if len(per[t]) < N_EP}
            if n_missing:
                complete = False
                print(f"[!] {tag}: 미완 태스크 {n_missing}")
            data[(arm, mode)] = {"per": per, "used_dirs": used, "n": len(recs)}
            print(f"{tag}: {len(recs)} eps (dirs: {','.join(used) or 'NONE'})")
    if not complete:
        print("[!] 200ep 미완 런 존재 — 판정은 가용 데이터로 계산하되 미완 명시")

    def task_sr(arm, mode, t):
        eps = data[(arm, mode)]["per"][t]
        return 100.0 * np.mean(list(eps.values())) if eps else float("nan")

    def mean_sr(arm, mode):
        return float(np.nanmean([task_sr(arm, mode, t) for t in range(N_TASK)]))

    res = {"complete": complete,
           "runs": {f"{a}_{m}": {"n_eps": data[(a, m)]["n"],
                                 "dirs": data[(a, m)]["used_dirs"],
                                 "mean_sr": mean_sr(a, m),
                                 "per_task_sr": {t: task_sr(a, m, t)
                                                 for t in range(N_TASK)}}
                    for a in ARMS for m in MODES}}

    # ---- 1) 주 게이트: wristpatch vs matchedbase paired bootstrap (correct) ----
    a_task = [task_sr("wristpatch", "correct", t) for t in range(N_TASK)]
    b_task = [task_sr("matchedbase", "correct", t) for t in range(N_TASK)]
    res["paired_task"] = paired_bootstrap(a_task, b_task)
    # 에피소드-단위 (동일 init_states → (t,e) paired) — 민감도
    a_ep, b_ep = [], []
    for t in range(N_TASK):
        pa, pb = data[("wristpatch", "correct")]["per"][t], \
                 data[("matchedbase", "correct")]["per"][t]
        for e in sorted(set(pa) & set(pb)):
            a_ep.append(pa[e] * 100.0)
            b_ep.append(pb[e] * 100.0)
    res["paired_episode"] = paired_bootstrap(a_ep, b_ep)
    # wrong 모드도 참고 병기
    res["paired_task_wrong"] = paired_bootstrap(
        [task_sr("wristpatch", "wrong", t) for t in range(N_TASK)],
        [task_sr("matchedbase", "wrong", t) for t in range(N_TASK)])

    # ---- 2) 언어 공동기준 c−w (A4 ±5pp 유보 밴드) ----
    res["lang"] = {}
    for arm in ARMS:
        cw = mean_sr(arm, "correct") - mean_sr(arm, "wrong")
        if cw >= 75:
            verdict = "PASS"
        elif cw >= 65:
            verdict = "DEFER(65-75 유보밴드 → 확증에서 재판정)"
        elif cw >= 60:
            verdict = "FAIL(+70 미달; F3 아님)"
        elif cw >= 55:
            verdict = "F3-DEFER(55-60 유보밴드)"
        else:
            verdict = "F3(언어 화폐 훼손 <+55 즉결 NO-GO)"
        res["lang"][arm] = {"c_minus_w": float(cw), "verdict": verdict}

    # ---- 3/4) 종합 ----
    pt = res["paired_task"]
    base_sr = mean_sr("matchedbase", "correct")
    res["F5_baseline_reanchor"] = {"matchedbase_correct_sr": base_sr,
                                   "band_85_88": bool(85 <= base_sr <= 88),
                                   "note": "밴드 밖이면 F5 검토(전 팔 판정 중단 조건)"}
    sig_win = pt["ci95"][0] > 0
    sig_loss = pt["ci95"][1] < 0
    res["verdict"] = {
        "paired_sig_win": bool(sig_win),
        "paired_sig_loss_F4": bool(sig_loss),
        "gate": ("WIN → 확증(50roll×3seed) 후보" if sig_win and
                 res["lang"]["wristpatch"]["verdict"].startswith(("PASS", "DEFER"))
                 else "F4(F3-echo: 유의 하락)" if sig_loss
                 else "NULL(CI∋0 — 이득 없음)")}

    print("\n=== per-task SR (%) — correct ===")
    print(f"{'task':>4} {'base':>6} {'wrist':>6} {'Δ':>6}   {'base_w':>6} {'wrist_w':>7}")
    for t in range(N_TASK):
        mark = " ★파지" if t in GRASP_TASKS else ""
        print(f"{t:>4} {task_sr('matchedbase','correct',t):>6.1f} "
              f"{task_sr('wristpatch','correct',t):>6.1f} "
              f"{task_sr('wristpatch','correct',t)-task_sr('matchedbase','correct',t):>+6.1f}   "
              f"{task_sr('matchedbase','wrong',t):>6.1f} "
              f"{task_sr('wristpatch','wrong',t):>7.1f}{mark}")
    print(f"mean {mean_sr('matchedbase','correct'):>6.1f} "
          f"{mean_sr('wristpatch','correct'):>6.1f} "
          f"{mean_sr('wristpatch','correct')-mean_sr('matchedbase','correct'):>+6.1f}   "
          f"{mean_sr('matchedbase','wrong'):>6.1f} {mean_sr('wristpatch','wrong'):>7.1f}")
    print(f"\npaired(task) Δ={pt['diff']:+.2f}pp CI[{pt['ci95'][0]:+.2f},"
          f"{pt['ci95'][1]:+.2f}] p={pt['p_a_gt_b']:.4f}")
    pe = res["paired_episode"]
    print(f"paired(episode) Δ={pe['diff']:+.2f}pp CI[{pe['ci95'][0]:+.2f},"
          f"{pe['ci95'][1]:+.2f}] p={pe['p_a_gt_b']:.4f} (n={pe['n_units']})")
    for arm in ARMS:
        print(f"lang {arm}: c−w = {res['lang'][arm]['c_minus_w']:+.1f}pp "
              f"→ {res['lang'][arm]['verdict']}")
    print(f"F5 기준선: matchedbase correct {base_sr:.1f}% "
          f"(85-88 밴드: {res['F5_baseline_reanchor']['band_85_88']})")
    print(f"종합: {res['verdict']['gate']}")

    grasp = {t: {"base": task_sr("matchedbase", "correct", t),
                 "wrist": task_sr("wristpatch", "correct", t)} for t in GRASP_TASKS}
    res["grasp_tasks_t3_t4_t6"] = grasp
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float))
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
