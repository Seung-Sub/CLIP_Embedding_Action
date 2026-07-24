# RESULT — W-A wrist 확증실험 최종 판정 (2026-07-24)

*프로토콜: 3 train-seed × 2 arm(matchedbase / wristpatch) × 2 instruction-mode(correct / wrong) = 12 run,
각 libero_spatial 10 task × 50 rollout = 500 ep/run. 총 6,000 ep. MUJOCO osmesa, retry-supervisor,
per-episode JSONL provenance(`outputs/eval/runs/<run_tag>/episodes.jsonl`). phase1 공유(large256-single),
phase2 seed{1,2,3} 재학습. 판정 수치는 유실된 verdict 스크립트 대신 per-episode 로그에서 직접 재계산.*

## 1. 원 수치 (per-episode 로그 직접 재계산)

| arm | seed | correct SR | wrong SR | c−w |
|---|---|---|---|---|
| matchedbase | 1 | 83.6 | 12.4 | +71.2 |
| matchedbase | 2 | 87.4 | 12.8 | +74.6 |
| matchedbase | 3 | 86.0 | 10.2 | +75.8 |
| wristpatch | 1 | 91.8 | 30.2 | +61.6 |
| wristpatch | 2 | 93.0 | 28.0 | +65.0 |
| wristpatch | 3 | 93.0 | 28.6 | +64.4 |

## 2. 3-seed 풀링 판정

**성능 (correct SR)** — pooled per-task 평균: matchedbase **85.7** / wristpatch **92.6** = **+6.9pp**.
- paired per-task bootstrap 10k, 95% CI **[+4.9, +9.1]** → **SIG > 0 (통과)**.
- per-task 델타 전부 양(t0+3 t1+1 t2+5 t3+7 t4+6 t5+11 t6+8 t7+9 t8+6 t9+13) — **이득이 파지·공간 재참조 태스크(t5/t7/t9)에 집중**, 설계 예측·스크리닝 패턴 재현.
- 3-seed 일관(+8.2/+5.6/+7.0), 스크리닝 +5.5와 정합.

**언어 공동기준 (correct−wrong)** — pooled: wristpatch **+63.7pp** vs matchedbase **+73.9pp**.
- 게이트 +70 **미달**, 사전등록 유보밴드(65–75)의 하단 65도 **하회**.
- wrist가 wrong 지시에서도 ~28–30% 성공(base ~10–13%) = **손목 기하로 언어 없이 파지하는 경로**. 3-seed 일관(61.6/65.0/64.4).

## 3. 사전등록 판정 적용

규칙: SR paired CI > 0 **AND** 언어 c−w ≥ +70pp → "채택 아키텍처".
- SR: **통과** (+6.9pp, CI 하한 +4.9).
- 언어: **미달** (+63.7 < 65 유보밴드 하단).

→ **W-A는 "제안 아키텍처"로 승격되지 않고, SR↔언어 tradeoff 프런티어의 새 점으로 자리한다.**

## 4. 함의 (논문 수납)

1. **캠페인 최초의 확증된 양의 SR 아키텍처 추가** — h-flow/actionflow/crop/W-C가 전부 중립~음성이었던 것과 대비. "복잡화 일관 무익"의 유일 예외 = **조건화-측 wrist 국소 기하**(삽입점 지도의 마지막 조각).
2. **SR↔언어 tradeoff 법칙의 4번째 독립 재현** (융합 다이얼 concat/avg · crop · W-A 스크리닝에 이어 W-A 확증) — C-2 기여의 결정적 강화. tradeoff 프런티어 그림의 확정 점 2개(matchedbase 85.7/+73.9, wristpatch 92.6/+63.7) 추가.
3. **응용 선택지**: 성능 우선 → W-A, 언어 충실 우선 → base. 이 선택 축의 존재가 해석적 기여.
4. **열린 후보 W-A′**: wrist 토큰을 DINOv3→SigLIP2 통일 공간으로. 오프라인 게이트에서 SigLIP2 wrist 패치 = DINO와 파지 정보 동등(R-A′ 1.008배) 확인. 가설 H-L1(같은 언어 타워라 언어 희석 완화) 성립 시 **프런티어를 언어축으로 미는 유일 후보** → 다음 스크리닝.

## 5. 재현 (clone 후)

```bash
# phase2 재학습 (phase1 large256 공유 ckpt 필요)
python src/training/train_phase2.py --config configs/phase2_libero_large256_wristpatch.yaml       # seed2
python src/training/train_phase2.py --config configs/phase2_libero_large256_wristpatch_s1.yaml
python src/training/train_phase2.py --config configs/phase2_libero_large256_wristpatch_s3.yaml
# (matchedbase 동일, config만 교체)
# 롤아웃 (판별 하네스: correct/wrong, per-episode JSONL 자동 기록)
MUJOCO_GL=osmesa python src/eval_libero/rollout_sim.py \
  --config configs/phase2_libero_large256_wristpatch.yaml --suite libero_spatial \
  --episodes 50 --instruction-mode correct --run-tag wristpatch_s2_correct_50r
# 판정 재계산: outputs/eval/runs/*/episodes.jsonl 에서 위 §2 부트스트랩 재실행
```

*판정 스크립트가 컨테이너 재시작 때 유실됐으나 per-episode JSONL로 완전 재계산 가능 —
provenance 하네스(commit 3f0d984, 감사 §5 대응)의 실효성을 스스로 입증한 사례.*
