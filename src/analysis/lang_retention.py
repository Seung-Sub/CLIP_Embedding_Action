#!/usr/bin/env python3
"""C1 게이트 (2): correct−wrong 언어 유지 집계 (≥ +70pp 게이트).

정책이 언어를 실제로 쓰는지 판별: 같은 씬·코드에서 지시문만 correct(정상) vs
wrong(다른 태스크 지시문)로 바꿔 rollout_sim 을 돌린 결과의 suite-평균 SR 을 비교.
언어를 쓰면 correct ≫ wrong. 게이트 = (correct − wrong) ≥ +70pp.

입력: rollout_sim 을 --instruction-mode correct / wrong 로 돌린 출력 glob.
  현행 저장명:  outputs/eval/rollout_{suite}_correct.txt / _wrong.txt
  (paired_ci 와 동일 로더 재사용 — .txt 의 "taskN: X.X%" 또는 JSON 스키마.)

SR 집계는 repo 관례와 동일: 태스크별 SR 의 suite 산술평균(= rollout_sim 의 "mean").
에피소드별 성공이 있으면 태스크 SR 로 접어(mean) 동일 관례로 집계.

실행 (게이트):
  CUDA_VISIBLE_DEVICES="" python src/analysis/lang_retention.py \
      --correct 'outputs/eval/c1/rollout_libero_spatial_correct.*' \
      --wrong   'outputs/eval/c1/rollout_libero_spatial_wrong.*'
"""
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "src"))

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

from analysis.paired_ci import load_arm  # noqa: E402  (동일 로더 재사용)


def task_sr(pattern):
    """glob → {task_id: SR(percent)}. episode 입도면 태스크당 mean 으로 접음."""
    gran, d, files = load_arm(pattern)
    if gran == "episode":
        d = {t: float(np.mean(v)) * 100.0 for t, v in d.items()}
    return d, files


def main():
    ap = argparse.ArgumentParser(
        description="correct−wrong 언어 유지 집계 (≥ +70pp 게이트)")
    ap.add_argument("--correct", required=True, help="correct 지시문 롤아웃 glob/dir")
    ap.add_argument("--wrong", required=True, help="wrong 지시문 롤아웃 glob/dir")
    ap.add_argument("--threshold", type=float, default=70.0, help="게이트 임계 (pp)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    c, fc = task_sr(args.correct)
    w, fw = task_sr(args.wrong)
    common = sorted(set(c) & set(w))
    if not common:
        raise ValueError("correct/wrong 공통 태스크 없음")

    per_task = {t: {"correct": c[t], "wrong": w[t], "delta": c[t] - w[t]}
                for t in common}
    mc = float(np.mean([c[t] for t in common]))
    mw = float(np.mean([w[t] for t in common]))
    delta = mc - mw
    passed = delta >= args.threshold

    res = {"n_tasks": len(common), "tasks": common,
           "correct_mean_sr": mc, "wrong_mean_sr": mw,
           "delta_correct_minus_wrong": delta, "threshold_pp": args.threshold,
           "gate_pass": bool(passed), "per_task": per_task,
           "files_correct": fc, "files_wrong": fw}

    print(f"=== 언어 유지 (correct − wrong) | 태스크 {len(common)}개 ===")
    for t in common:
        pt = per_task[t]
        print(f"task {t:2d}: correct {pt['correct']:5.1f}%  "
              f"wrong {pt['wrong']:5.1f}%  Δ {pt['delta']:+5.1f}pp")
    print(f"suite-평균: correct {mc:.1f}%  wrong {mw:.1f}%  "
          f"Δ(correct−wrong) {delta:+.1f}pp")
    print(f"GATE (Δ ≥ +{args.threshold:.0f}pp): {'PASS' if passed else 'FAIL'}")

    out = Path(args.out) if args.out else (
        WS / "outputs" / "analysis" / "lang_retention.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
