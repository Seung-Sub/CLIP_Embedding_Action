# clip_ws — CLIP 잠재공간 Visual-Policy (LIBERO)

frozen CLIP ViT-L/14 잠재공간에서 액션청크를 결합(Phase 1)하고
그 위에서 미래 액션청크를 추론하는 flow matching 정책(Phase 2)을 학습·평가한다. 
LIBERO(Franka Panda, OSC 델타 7D, 20Hz) 벤치마크 기준.

```
Phase 1:  g(A_t, z_t) ≈ Δz  /  h(Δz, z_t) ≈ A_t
Phase 2:  토큰 [z_{t−16}, z_t, g(A_{t−16}), 언어, 손목캠] → flow matching → ζ̂ → h(ζ̂, z_t) = Â_t
폐루프:   16스텝 예측 → 앞 8스텝만 사용 → 재예측 (receding horizon)
```

## 0. 요구사항

- Linux + NVIDIA GPU (VRAM 12GB 이상 권장; RTX 5070 Ti급 Blackwell은 포함된 cu128 torch 필수)
- conda(Anaconda/Miniconda), 디스크 여유 약 20GB (LIBERO 3개 suite 시 추가)

## 1. 설치 (최초 1회)

```bash
git clone https://github.com/kist-pier/CLIP.git ~/clip_ws && cd ~/clip_ws
conda env create -f environment_libero.yml   # env "clip_libero"
conda activate clip_libero && cd ~/clip_ws
# wandb 로깅을 쓰지 않으면 모든 학습 명령에 --set wandb.enabled=false 를 붙인다
# 첫 libero 임포트 시 설정 프롬프트가 뜨면 N 입력 (스크립트에선 printf "N\n" | 파이프)
# 벤치마크 자산은 첫 실행 시 ~/.cache/libero 로 자동 다운로드된다
hf download openai/clip-vit-large-patch14 --local-dir ~/clip_ws/models/clip-vit-large-patch14
# 경로는 configs/config.yaml의 clip.model_dir와 일치시킬 것 (기본값 = 위 경로)
```

---

## 2. 실험 — LIBERO (spatial / object / goal suite)

두 모델을 학습한다 — **phase1(DeltaAE)은 공유**, phase2(정책)만 갈린다:

| | 2-A. 표준(최고성능) | 2-B. 손목캠 제외(Open-loop rollout) |
|---|---|---|
| config | `phase2_libero.yaml` | `phase2_libero_nowrist.yaml` |
| 토큰 | z_prev, z_cur, g(A_past), 언어, **손목캠** | z_prev, z_cur, g(A_past), 언어 |
| 폐루프 성공률(spatial) | **85.2%** | 50.4% (−34.8pp) |
| 용도 | 기본 사용·평가 | `*_serial.py` 다단계 연쇄 진단(§B-8) — 손목캠은 잠재상태 상상(g rollforward) 대상이 아니라서 제외해야 함 |

### B-1. 데모 데이터 다운로드 (suite당 6~7GB, task당 hdf5 1개 × 10)

```bash
python -c "from libero.libero.utils.download_utils import download_from_huggingface; \
           download_from_huggingface('libero_spatial', 'data/libero', check_overwrite=False)"
# object/goal suite도 동일: 'libero_object', 'libero_goal'
```

### B-2. Phase 1 학습 (두 모델 공용, CLIP 인코딩 + 학습, GPU에 따라 수 분~수십 분)

```bash
python src/training/train_phase1.py --config configs/phase1_libero.yaml
# → checkpoints/phase1_libero.pt
```

### B-3. Phase 2 학습 — 2-A(표준) 또는 2-B(손목캠 제외) 중 필요한 쪽, 둘 다 해도 됨

```bash
python src/training/train_phase2.py --config configs/phase2_libero.yaml           # 2-A (~15분)
# → checkpoints/phase2_libero_fm.pt   (flow + 언어 + 손목캠 토큰, 124M)

python src/training/train_phase2.py --config configs/phase2_libero_nowrist.yaml   # 2-B (~15분)
# → checkpoints/phase2_libero_nowrist.pt   (flow + 언어, 손목캠 제외, 123M)
```

### B-4. GT 데모 평가 (7D 액션 플롯, 두 모델 공용 — `--config`로 선택)

```bash
python src/eval_libero/rollout_dataset.py                                          # 2-A(기본)
python src/eval_libero/rollout_dataset.py --config configs/phase2_libero_nowrist.yaml  # 2-B
# → outputs/eval/rollout_dataset_libero_*.png
```

### B-5. 폐루프 suite 평가 (두 모델 공용)

```bash
MUJOCO_GL=egl python src/eval_libero/rollout_sim.py --suite libero_spatial --episodes 20
# 기대(20롤×10태스크 평균): spatial ≈ 85%   (베이스라인 mlp: ≈ 37%) — §7 결과 참조
MUJOCO_GL=egl python src/eval_libero/rollout_sim.py --config configs/phase2_libero_nowrist.yaml --suite libero_spatial --episodes 20
# 손목캠 제외라 기대치 더 낮음(≈ 50%) — §B-8 다단계 연쇄를 쓸 때만 이 모델을 쓸 것
# 옵션: --task-id 0 (단일 태스크) / --save-video 2 / --episodes 10이면 더 빠르지만 표본이 작아 편차가 큼
```

### B-6. 잠재공간 맵핑 시각화 (선택, 대화형 창)

```bash
python src/eval_libero/latent_mapping.py
# phase1 잠재공간에 3인칭 전/후·Δz 화살표·g(액션청크) 화살표·그리퍼 델타·언어 cmd를
# PCA 2D/3D로 표시. 우측에서 태스크/에피소드/시작 시점 선택, [전체 구성]·[3D]·[확대] 토글
```

**object/goal suite도 이미 config가 준비돼 있다** (`configs/{phase1,phase2}_libero_{obj,goal}.yaml`
— `libero_spatial`을 `libero_object`/`libero_goal`로, cache/checkpoint 이름만
바꾼 사본, 표준 레시피 그대로). B-1의 데이터 다운로드만 `'libero_object'`/`'libero_goal'`로
바꿔서 받고, B-2/B-3/B-5를 해당 config로 실행하면 된다:

```bash
python src/training/train_phase1.py --config configs/phase1_libero_obj.yaml
python src/training/train_phase2.py --config configs/phase2_libero_obj.yaml
MUJOCO_GL=egl python src/eval_libero/rollout_sim.py --config configs/phase2_libero_obj.yaml --suite libero_object --episodes 20
# goal도 동일 패턴(_obj → _goal, libero_object → libero_goal)
```

실측 결과: **Object 90.2%**, **Goal 87.2%** (Spatial 85.2%보다도 높음) — 상세는
[`experiments/README.md`](experiments/README.md#다른-suite로-확장--libero-object--libero-goal) 참조.
다른 suite(예: libero_100)로 더 확장하려면 위 config를 템플릿 삼아 같은 sed 패턴
(`libero_spatial` → 원하는 suite명, cache/checkpoint 이름도 같이 치환) 반복.

### B-8. 다단계 연쇄 (월드모델 rollforward, **2-B 모델 필요**)

카메라를 매 재계획마다 다시 인코딩하지 않고, `g(디코딩한 행동, z_cur)`로 잠재상태를
"상상"해 전진시키며 여러 청크를 이어갈 수 있는지 진단하는 도구 2종. 손목캠은 이
전진 대상이 아니라서 B-3의 **2-B(`phase2_libero_nowrist.yaml`) 체크포인트가 있어야
동작**한다 — 2-A 체크포인트로 실행하면 손목캠 토큰 부재로 구조가 안 맞아 에러가 난다.

```bash
python src/eval_libero/rollout_dataset_serial.py            # 대화형 뷰어(오프라인, GT 이미지 기준, 기본 2-B)
# 태스크/에피소드/n(블라인드 청크 수) 슬라이더로 드리프트(cos) 확인

MUJOCO_GL=egl python src/eval_libero/rollout_sim_serial.py --n 4 --episodes 20
# 폐루프: --n = 재조회 주기(청크 단위, 몇 청크마다 카메라를 실제로 씀), 기본 2-B
#   --n 0  최초 1청크만 실측, 이후 끝까지 완전 블라인드(open-loop dead-reckoning) — §7 결과 참조
#   --n 1  매 청크(16스텝)마다 재조회 / --n 4  3청크 블라인드→4번째 재조회 반복
MUJOCO_GL=egl python src/eval_libero/rollout_sim_serial.py --n 0 --episodes 20 --suite libero_spatial
```

### B-9. 언어 페러프레이징 검증 (**2-A 모델**)

libero_spatial은 지시문이 태스크당 1개(총 10개)뿐이라, 정책이 언어를 이해하는 게
아니라 그 문자열을 암기했을 가능성이 있다. `src/eval_libero/paraphrases.py`에
태스크당 3개씩 준비된 페러프레이징(물체·공간관계·목표는 원문과 동일, 어휘·문장
구조만 다름)으로 **원문은 전혀 안 쓰고** 폐루프를 돌려 성공률을 비교한다.

```bash
MUJOCO_GL=egl python src/eval_libero/rollout_sim_paraphrase.py --episodes 20
# 태스크 10개 × 페러프레이징 3개 = 30조건, 조건당 --episodes 롤아웃
# 기대: 85.2%(원문) → 67.5%(페러프레이징만, §7 결과 참조) — 태스크별 편차가 매우 큼(0~96.7%)

python src/eval_libero/recovery_probe_gui.py    # 대화형: 태스크/페러프레이징 선택 + 실시간 관찰(§B-10)
```

### B-10. 실패 복구 관찰 GUI (**2-A 모델**, 데스크톱 세션 필요)

물체를 못 잡고 실패했을 때 복구 시도가 나오는지 직접 관찰하는 대화형 도구.
태스크·지시문(원문/페러프레이징 3종) 라디오버튼, 실시간 카메라 화면, 실행 궤적,
이미지-액션 정렬 cos 그래프. Start를 누르면 에피소드 1개를 끝까지(성공/실패 무관,
기본 스텝 제한 없음) 실행하고 자동으로 멈춘다 — 다음 에피소드는 다시 Start.

```bash
MUJOCO_GL=egl python src/eval_libero/recovery_probe_gui.py
```

---

## 4. 모델 변형과 실험 옵션

- **기본 = flow matching 정책** (권장). **베이스라인(MLP 회귀)** 비교는 config만 교체:
  ```bash
  python src/training/train_phase2.py --config configs/phase2_libero_mlp.yaml
  ```
- 평가 스크립트는 **체크포인트에 저장된 config로 모델 구조를 자동 복원**한다 —
  같은 평가 명령으로 어떤 변형이든 평가된다 (config의 `train.checkpoint`가 평가 대상을 지정).
- 모든 학습 스크립트는 오버라이드/분리 저장을 지원한다:
  ```bash
  python src/training/train_phase2.py --set module.d_model=512 --tag my_run
  # → checkpoints/grid/my_run.pt, outputs/grid/my_run.json (기본 체크포인트를 건드리지 않음)
  ```
- 주요 `module` 키: `name`(mlp|flow) · `d_model` · `layers` · `ctx_layers`(flow 문맥 인코더) ·
  `flow_steps`(Euler 스텝) · `lang_token`(LIBERO 언어) · `wrist_token`+`data.wrist_camera`(LIBERO 손목캠)

## 5. 디렉터리

| 경로 | 내용 |
|---|---|
| `configs/` | phase1/phase2 설정 (`*_libero.yaml` / 베이스라인: `*_mlp.yaml`) |
| `src/core` `src/data` `src/models` `src/training` | CLIP 래퍼 · 로더(임베딩 캐시) · DeltaAE+정책 · 트레이너 |
| `src/eval_libero` | GT 평가(`rollout_dataset.py`) / 폐루프 평가(`rollout_sim.py`) / 다단계 연쇄(`*_serial.py`) / 잠재공간 시각화(`latent_mapping.py`) / 페러프레이징 전용 폐루프(`rollout_sim_paraphrase.py`) / 실패 복구·페러프레이징 관찰 GUI(`recovery_probe_gui.py`) |
| `experiments/` | §7 결과의 원본 데이터(jsonl/txt) |
| `data/` `checkpoints/` `outputs/` | 데이터 / 학습 결과 / 캐시·평가 산출물 (git 제외) |

## 6. 문제 해결

- **mujoco 버전**: 반드시 3.3.2 (environment_libero.yml에 고정됨 — 2.3.x는 로봇 렌더
  붕괴, 3.10+는 크래시). 임의 업그레이드 금지
- **렌더 오류**: 헤드리스/원격에서는 모든 시뮬 명령에 `MUJOCO_GL=egl` 필수
- **데이터를 다시 수집/다운로드한 경우**: 해당 임베딩 캐시를 비울 것 (`rm -rf outputs/cache/libero_emb`)
- **처음부터 완전 재현**: `rm -rf data checkpoints outputs` 후 위 절차를 처음부터

## 7. 결과 (10태스크 평균, 폐루프 성공률 내림차순 — 표기 없으면 libero_spatial 기준)

| 조건 | 폐루프 성공률 | 원본 데이터 |
|---|---|---|
| 2-A, **libero_object suite** | **90.2%** (표준편차 0.2) | [`experiments/object_5rep.jsonl`](experiments/object_5rep.jsonl) |
| 2-A, **libero_goal suite** | **87.2%** | [`experiments/goal_5rep.jsonl`](experiments/goal_5rep.jsonl) |
| **2-A 표준** (최고성능, 손목캠 포함) | **85.2%** | [`experiments/baseline_5rep.jsonl`](experiments/baseline_5rep.jsonl) |
| 2-A, phase1 align 손실을 L1로 교체 | 82.6% (−2.6pp, align_cos는 0.654→0.266로 급락) | [`experiments/l1_align_loss_5rep.jsonl`](experiments/l1_align_loss_5rep.jsonl) |
| 2-A, 디코더에서만 z_t 조건 제거 | 81.8% (−3.4pp) | [`experiments/decoder_nostate_5rep.jsonl`](experiments/decoder_nostate_5rep.jsonl) |
| 2-A, **원문 대신 페러프레이징만 사용** | 67.5% (−17.7pp, 태스크별 0~96.7%로 편차 큼) | [`experiments/paraphrase_only.jsonl`](experiments/paraphrase_only.jsonl) |
| 2-A, 인코더+디코더 둘 다 z_t 조건 제거 | 64.6% (−20.6pp) | [`experiments/encoder_decoder_nostate_5rep.jsonl`](experiments/encoder_decoder_nostate_5rep.jsonl) |
| 2-A, 손목캠 토큰 제외 | 50.4% (−34.8pp) | [`experiments/wrist_excluded_5rep.jsonl`](experiments/wrist_excluded_5rep.jsonl) |
| **2-B + n=0** (손목캠 제외 + 완전 오토리그레시브, 최초 1청크만 실측 후 끝까지 자기예측 되먹임) | **39.0%** | [`experiments/serial_n0.txt`](experiments/serial_n0.txt) |

베이스라인(mlp 회귀, phase2_libero_mlp.yaml)은 ≈37%. 외부 VLA 비교: 같은
LIBERO-Spatial에서 OpenVLA(7B급 VLM) 84.7%, 2026년 SOTA권 97~98%대(APT, OmniVLA-RL) —
우리 2-A(85.2%)는 OpenVLA와 거의 동급. 언어 페러프레이징 강건성은 문헌마다 편차가
큰데, [LIBERO-Para](https://arxiv.org/abs/2603.28301)(VLA 7종, 0.6B~7.5B 평균)는
−22~−52pp를 보고해 우리(−17.7pp)가 오히려 그 범위보다 낫다 — [LIBERO-PRO](https://arxiv.org/html/2510.03827v1)의
OpenVLA 단일 사례(−1pp)가 예외적으로 강건했던 것으로 보임. 상세는
[`experiments/README.md`](experiments/README.md) 참조.

`rollout_sim.py --suite libero_spatial --episodes 20`(2-A/2-B 공통), `rollout_sim_paraphrase.py
--episodes 20`으로 위 절대 성능(2-A, 손목캠 제외, 페러프레이징, 2-B+n=0)은 그대로
재현 가능; z_t 조건 제거·L1 손실 2개는 phase1 모델 옵션을 바꿔 재학습해야 나오는
결과. 전부 5회 반복 평균(페러프레이징·2-B+n=0만 1회) 기준이라 ±2~3pp 편차는 정상.
