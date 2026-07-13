#!/usr/bin/env python3
"""C1 게이트 (1): paired 부트스트랩 신뢰구간 — C1(2채널 F4) vs SigLIP2 pooled-단독.

두 팔(arm)은 동일 suite·task·seed·episode 순서(공식 get_task_init_states 고정 =
paired, eval_protocol.md)로 돌린 rollout_sim 결과다. 같은 롤아웃 세팅이므로
부트스트랩을 **paired**로(양 팔 동일 재표집 인덱스) 수행해 SR 차의 95% CI를 낸다.

입력: 두 팔의 rollout_sim 출력 glob (dir 또는 파일 패턴).
  - 현행 rollout_sim 는 태스크별 SR 을 텍스트로 저장한다:
      outputs/eval/rollout_{suite}_{mode}.txt  →  "taskN: X.X%" 라인 + "mean: ..".
    → 이 경우 재표집 단위 = **태스크 SR**(granularity="task"). suite당 태스크 수만큼의
      표본으로 부트스트랩(예: libero_spatial 10 태스크). 이 조립도가 게이트의 실효 입도.
  - 에피소드별 성공(0/1)이 JSON 으로 저장돼 있으면(스키마 아래) 재표집 단위 =
    **에피소드**(granularity="episode")로 더 촘촘하게 낸다.

지원 JSON 스키마(가정 — 저장부가 생기면 자동 사용):
  A) per-episode:  {"per_episode": {"0": [1,0,1,...], "1": [...], ...}}   (task→성공 0/1)
     또는 records:  [{"task_id":0,"success":true}, ...]
  B) per-task SR:  {"results": {"0": 88.5, ...}}  또는  {"0": 88.5, ...}
     (값이 ≤1 이면 fraction 으로 보고 ×100.)

출력: 팔A/팔B suite-평균 SR, 평균 SR 차(pp), 95% CI, p(A>B).  outputs/analysis/ 에 JSON.

실행 (게이트):
  CUDA_VISIBLE_DEVICES="" python src/analysis/paired_ci.py \
      --arm-a 'outputs/eval/c1/rollout_libero_spatial_correct.*' \
      --arm-b 'outputs/eval/pooled/rollout_libero_spatial_correct.*'
"""
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "src"))

import argparse  # noqa: E402
import glob as _glob  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402

import numpy as np  # noqa: E402

_TASK_TXT = re.compile(r"^task(\d+):\s*([-+]?[\d.]+)%\s*$")   # "taskN: X.X%" (swap/mean 제외)


def _norm_task_key(k):
    """'task3' / '3' / 3 → 3 (int). 실패 시 None."""
    if isinstance(k, (int, np.integer)):
        return int(k)
    m = re.search(r"(\d+)", str(k))
    return int(m.group(1)) if m else None


def _to_percent(v):
    """SR 값을 percent(0~100)로 정규화. fraction(≤1)로 보이면 ×100."""
    v = float(v)
    return v * 100.0 if abs(v) <= 1.0 else v


def parse_file(path):
    """단일 rollout 출력 파일 → (granularity, {task_id: payload}).

    granularity="episode": payload = list[int] (에피소드별 0/1)
    granularity="task":    payload = float    (태스크 SR, percent)
    """
    path = Path(path)
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        # per-episode: records 리스트
        if isinstance(data, list):
            ep = {}
            for r in data:
                t = _norm_task_key(r.get("task_id", r.get("task")))
                if t is None:
                    continue
                ep.setdefault(t, []).append(int(bool(r.get("success"))))
            if ep:
                return "episode", ep
        if isinstance(data, dict):
            if "per_episode" in data:
                ep = {}
                for k, v in data["per_episode"].items():
                    t = _norm_task_key(k)
                    if t is not None:
                        ep[t] = [int(bool(x)) for x in v]
                return "episode", ep
            src = data.get("results", data)
            task = {}
            for k, v in src.items():
                t = _norm_task_key(k)
                if t is not None and isinstance(v, (int, float)):
                    task[t] = _to_percent(v)
            return "task", task
        raise ValueError(f"인식 못한 JSON 스키마: {path}")
    # .txt (현행 rollout_sim 출력) — "taskN: X.X%" 만 취함
    task = {}
    for line in path.read_text().splitlines():
        m = _TASK_TXT.match(line.strip())
        if m:
            task[int(m.group(1))] = float(m.group(2))   # 이미 percent
    return "task", task


def load_arm(pattern):
    """glob(dir 또는 패턴) → (granularity, {task_id: payload}). 여러 파일 병합.

    dir 이면 그 안의 rollout_* 파일을 모은다. episode 데이터가 하나라도 있으면
    episode 입도(태스크별 리스트 이어붙임), 아니면 task 입도."""
    p = Path(pattern)
    if p.is_dir():
        files = sorted(_glob.glob(str(p / "rollout_*.json"))
                       + _glob.glob(str(p / "rollout_*.txt")))
    else:
        files = sorted(_glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"매칭 파일 없음: {pattern}")
    ep_all, task_all, has_ep = {}, {}, False
    for f in files:
        g, d = parse_file(f)
        if g == "episode":
            has_ep = True
            for t, v in d.items():
                ep_all.setdefault(t, []).extend(v)
        else:
            task_all.update(d)
    if has_ep:
        return "episode", ep_all, files
    return "task", task_all, files


def _paired_units(ga, da, gb, db):
    """양 팔을 공통 단위의 paired 1D 벡터(a,b; percent 스케일)로 정렬.

    둘 다 episode 면 (task,ep) 단위(양 팔 최소 에피소드 수로 절단), 아니면 태스크 SR."""
    common = sorted(set(da) & set(db))
    if not common:
        raise ValueError("두 팔에 공통 태스크가 없음 (task id 정렬 확인)")
    if ga == "episode" and gb == "episode":
        a, b = [], []
        note = []
        for t in common:
            na, nb = len(da[t]), len(db[t])
            n = min(na, nb)
            if na != nb:
                note.append(f"task{t}:{na}vs{nb}→{n}")
            a.extend(x * 100.0 for x in da[t][:n])   # 0/1 → 0/100
            b.extend(x * 100.0 for x in db[t][:n])
        return "episode", np.array(a, float), np.array(b, float), common, note
    a = np.array([da[t] for t in common], float)
    b = np.array([db[t] for t in common], float)
    return "task", a, b, common, []


def paired_bootstrap(a, b, n_boot=10000, seed=0):
    """paired 부트스트랩: 매 반복 동일 인덱스로 a,b 재표집 → 평균차 분포.

    반환: mean_a, mean_b, diff(=A−B, pp), ci95, p(A>B), n_units."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    assert a.shape == b.shape and a.ndim == 1 and len(a) > 0
    n = len(a)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n, size=(n_boot, n))         # paired: 동일 인덱스 a,b 공유
    diffs = a[idx].mean(1) - b[idx].mean(1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "diff": float(a.mean() - b.mean()),
        "ci95_low": float(lo), "ci95_high": float(hi),
        "p_a_gt_b": float((diffs > 0).mean()),
        "boot_diff_mean": float(diffs.mean()), "boot_diff_std": float(diffs.std()),
        "n_units": int(n), "n_boot": int(n_boot),
    }


def main():
    ap = argparse.ArgumentParser(description="paired 부트스트랩 CI (C1 vs pooled-only)")
    ap.add_argument("--arm-a", required=True, help="C1 결과 glob/dir")
    ap.add_argument("--arm-b", required=True, help="pooled-only 결과 glob/dir")
    ap.add_argument("--label-a", default="C1")
    ap.add_argument("--label-b", default="pooled")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="JSON 저장 경로 (기본: outputs/analysis/paired_ci_<a>_vs_<b>.json)")
    args = ap.parse_args()

    ga, da, fa = load_arm(args.arm_a)
    gb, db, fb = load_arm(args.arm_b)
    gran, a, b, common, note = _paired_units(ga, da, gb, db)
    res = paired_bootstrap(a, b, args.n_boot, args.seed)
    res.update({"granularity": gran, "label_a": args.label_a, "label_b": args.label_b,
                "n_tasks": len(common), "tasks": common,
                "files_a": fa, "files_b": fb, "pair_note": note})

    print(f"=== paired bootstrap CI: {args.label_a} − {args.label_b} ===")
    print(f"granularity   : {gran}  (재표집 단위 {res['n_units']}개, 태스크 {len(common)}개)")
    if note:
        print(f"  [주의] 에피소드 수 불일치 절단: {', '.join(note)}")
    print(f"{args.label_a:>10} SR : {res['mean_a']:.2f}%")
    print(f"{args.label_b:>10} SR : {res['mean_b']:.2f}%")
    print(f"diff (A−B)    : {res['diff']:+.2f} pp")
    print(f"95% CI (paired): [{res['ci95_low']:+.2f}, {res['ci95_high']:+.2f}] pp  "
          f"(n_boot={args.n_boot})")
    print(f"p(A>B)        : {res['p_a_gt_b']:.4f}")
    gate = res["ci95_low"] > 0.0
    print(f"GATE (CI>0, A우세): {'PASS' if gate else 'FAIL'}")
    res["gate_ci_above_zero"] = bool(gate)

    out = Path(args.out) if args.out else (
        WS / "outputs" / "analysis" /
        f"paired_ci_{args.label_a}_vs_{args.label_b}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
