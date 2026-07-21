# RUNBOOK — P-A 「CROP-AVG / CROP-CONCAT」 (사전등록 §8 P-A, 2026-07-18)

셀: 채택 융합 구조(avg 91.5/+74, concat SR-max 트랙) × 격리 실증 crop 레버(+~5pp,
DINOv2-avg@crop 96.0 vs matched no-crop 90.5 — DESIGN_pipeline_rethink_v1.md §4.2(1)).
crop = 콜리그-EXACT 기하(shortest_edge 256 bicubic resize → 224×224 center-crop, HF
BitImageProcessor 규약 픽셀 동일 검증) 후 각 백본 native 전처리. 구현: `anchor.crop`
옵션 (none|dino|both), src/core/anchor.py.

| arm | config (p1 / p2) | cache_key (하위디렉터리) |
|---|---|---|
| CROP-AVG dino | `phase1_libero_cropavg_dino.yaml` / `phase2_libero_cropavg_dino_noaug.yaml` | `dualfusion-sig1024-dinov3_256-a0.5-crop224dino/joint/norm` |
| CROP-AVG both | `phase1_libero_cropavg_both.yaml` / `phase2_libero_cropavg_both_noaug.yaml` | `dualfusion-sig1024-dinov3_256-a0.5-crop224both/joint/norm` |
| CROP-CONCAT dino | `phase1_libero_cropconcat_dino.yaml` / `phase2_libero_cropconcat_dino_noaug.yaml` | `dualconcat-sig1024-dinov3_256-crop224dino/joint/raw` |

캐시 규율: crop arm 별 cache_key 가 no-crop(`…-a0.5`/`…dinov3_256`) 키와 문자열 수준에서
분리 — 같은 cache_dir 을 써도 오염 불가 (과거 사고 재발 방지 검증 완료, 아래 원장 참조).
phase1/phase2 의 `anchor.crop` 은 반드시 동일해야 함 (z 공간 정의 자체가 다름).

---

## 실행 절차 (원격 = `kist_a6000_ss` 컨테이너 `acfa5157ba25`, GPU 8/9 전용)

공통: `cd ~/clip_ws` (원격), 코드/config 동기화 후 실행. 학습은 GPU 한 장씩 핀 고정,
rollout 은 `MUJOCO_GL=osmesa`(컨테이너 EGL 불가) + `OMP_NUM_THREADS≤6`. osmesa 렌더
간헐 segfault(ep 단위 확률적, 커밋 0842fb1) → 중단 시 동일 커맨드 재착수(캐시/ckpt 보존,
run-tag 재사용 금지 — `_r2` 접미사).

### 1) Phase-1 (DeltaAE 재적합, 캐시 재인코딩 포함 — arm 당 ~수시간)

```bash
# GPU8: avg-dino → avg-both 순차 (같은 cache_dir, cache_key 로 분리)
CUDA_VISIBLE_DEVICES=8 python src/training/train_phase1.py \
  --config configs/phase1_libero_cropavg_dino.yaml
CUDA_VISIBLE_DEVICES=8 python src/training/train_phase1.py \
  --config configs/phase1_libero_cropavg_both.yaml

# GPU9: concat-dino (병렬)
CUDA_VISIBLE_DEVICES=9 python src/training/train_phase1.py \
  --config configs/phase1_libero_cropconcat_dino.yaml
```

**G-off 게이트 (phase2 착수 전 판정)**: phase1 val **dec R² ≥ 0.72** + align cos·a2z 가
매칭 no-crop base(fobsfusion_avg / fobsfusion_concat) 대비 비열화. 미달 arm 은 여기서 중단.

### 2) Phase-2 (no-aug clean regime, seed 2 — winner 설정 복사)

```bash
CUDA_VISIBLE_DEVICES=8 python src/training/train_phase2.py \
  --config configs/phase2_libero_cropavg_dino_noaug.yaml
CUDA_VISIBLE_DEVICES=8 python src/training/train_phase2.py \
  --config configs/phase2_libero_cropavg_both_noaug.yaml
CUDA_VISIBLE_DEVICES=9 python src/training/train_phase2.py \
  --config configs/phase2_libero_cropconcat_dino_noaug.yaml
```

### 3) Rollout — 20롤 스크리닝, correct+wrong 동시, arm 당 --run-tag 각인

```bash
# arm 1 예시 (GPU8) — correct / wrong 을 쌍으로 (G-lang 은 동일 창의 쌍만 유효)
CUDA_VISIBLE_DEVICES=8 MUJOCO_GL=osmesa OMP_NUM_THREADS=6 \
  python src/eval_libero/rollout_sim.py \
  --config configs/phase2_libero_cropavg_dino_noaug.yaml \
  --suite libero_spatial --episodes 20 --instruction-mode correct \
  --run-tag cropavg_dino_correct_20r
CUDA_VISIBLE_DEVICES=8 MUJOCO_GL=osmesa OMP_NUM_THREADS=6 \
  python src/eval_libero/rollout_sim.py \
  --config configs/phase2_libero_cropavg_dino_noaug.yaml \
  --suite libero_spatial --episodes 20 --instruction-mode wrong \
  --run-tag cropavg_dino_wrong_20r

# arm 2 (GPU8): --config …cropavg_both_noaug.yaml  --run-tag cropavg_both_{correct,wrong}_20r
# arm 3 (GPU9): --config …cropconcat_dino_noaug.yaml --run-tag cropconcat_dino_{correct,wrong}_20r
```

per-episode 성공 플래그는 `outputs/eval/runs/<run_tag>/` JSONL 로 자동 각인(UNTRACED 재발
방지). 대조군 창: avg no-crop 91.5 (`phase2_libero_fobsfusion_avg_noaug.pt`) — 필요 시 동일
날짜 창에서 같은 20롤 재실행해 paired 비교.

### 4) 사전등록 게이트 & 판정 (DESIGN_pipeline_rethink §8 P-A)

1. **G-off**: phase1 dec **R² ≥ 0.72** (+ align cos·a2z 비열화) — 미달 arm phase2 금지.
2. **G-cl**: paired bootstrap 10k, **vs avg no-crop 91.5, Δ 95%CI > 0**. 20롤 스크리닝
   통과 arm 만 50롤×3시드 확정.
3. **G-lang**: **correct−wrong ≥ +70pp** 공동기준 (wrong-mode 동시 롤아웃, crop 은 언어
   중립 예상 — 이 게이트가 검정).

**기대**: avg 91.5 → ~95-96, 언어 +74 유지. **실패 모드 예측(사전등록)**: crop 기전 =
"작업공간 중앙 확대" → 테두리-의존 태스크에서 per-task 하락. 발견 시 per-task 분해
(JSONL task_id 별 SR)로 "crop=중앙확대" 기전 절 보고 — 전체 SR 상승과 무관하게 기록.

---

## 검증 원장 (2026-07-18 구현 시점, 로컬 CPU 실물 텐서)

- 기본경로 byte-identity: crop 미지정 config 에서 siglip2/dinov3/dualfusion(구코드 HEAD
  판 대비) 출력 `tobytes()` 완전 동일 + cache_key 문자열 불변 — 31/31 PASS.
- crop 경로: shape 유지, 출력 상이, 외곽 12px 훼손에 불감(crop 실증)·중앙 훼손에 민감,
  dino-arm 은 sig 브랜치 테두리 민감 유지(브랜치 분리 실증), 텍스트 경로 무변화.
- `_crop224` 는 HF dinov2-large 기본 processor(콜리그 판)와 4개 기하(128²/256²/300×200/
  517×723)에서 픽셀 동일.
- 기존 config 전수 스캔: crop 키 보유 파일 = 신규 6개뿐 (기존 arm 전부 crop 경로 미해석).
