# SETUP — 클론에서 실행까지

> 목표: `git clone` 한 새 사람이 **환경을 만들고, 코드가 살아있음을 데이터 없이
> 확인**하기까지. 무엇이 되고 안 되는지 정직하게 적는다. 사용법 상세는
> [`README.md`](README.md), 연구 맥락은 [`ONBOARDING.md`](ONBOARDING.md).

---

## 0. 정직한 전제 — 신선한 클론이 할 수 있는 것/없는 것

`.gitignore` 로 인해 **다음은 클론에 들어있지 않다**:
`/data/` · `/checkpoints/` · `/outputs/` · `/models/` · `/wandb/` · `/docs/` ·
`*.pt *.pth *.hdf5 *.npz` · `/SigLIP/`(콜리그 레포) · `/Related_Papers/`.

| 하려는 것 | 신선한 클론에서 | 필요한 것 |
|---|---|---|
| 코드 읽기·이해 | ✅ 가능 | — (src/ + 최상위 md + reports/ 번들 있음) |
| import·구조 스모크(`--smoke`) | ✅ 가능(모델 자동 다운로드만) | 인터넷(HF) 또는 캐시된 백본 |
| phase1/2 **학습** | ⚠ **데이터 다운로드 후** 가능 | LIBERO HDF5(각 suite ~6~7GB) + VL 백본 |
| 폐루프 **롤아웃** | ⚠ 학습된 ckpt 필요 | 위 학습 산출물(ckpt 는 repo 에 없음) |
| 발표 수치 그대로 재현 | 부분적 | 원격 박스의 캐시/ckpt/`outputs/`(비공개) |
| 최신 문서(rseries/pregates/wrist 확증) 읽기 | ❌ | `docs/`는 gitignore → 원격/로컬 원본 필요(아래 §6) |

즉 **레포 자체엔 데이터·체크포인트가 없다.** 데이터는 아래 절차로 받고, 체크포인트는
직접 학습해서 만든다(수 분~수십 분/모델).

---

## 1. 클론 + conda 환경

```bash
git clone <this-repo-url> ~/clip_ws && cd ~/clip_ws
conda env create -f environment_libero.yml    # env 이름: clip_libero
conda activate clip_libero
```

`environment_libero.yml`의 실제 의존성 이야기:
- **python 3.10**, pip 로 설치.
- **torch / torchvision** — `--extra-index-url .../whl/cu128`(**Blackwell GPU용 CUDA 12.8**).
  README §0 은 RTX 5070 Ti급(Blackwell)에서 cu128 torch 필수라고 명기.
- **transformers, huggingface_hub** — CLIP/SigLIP2/DINOv3 백본 로드.
- **libero** — LIBERO 벤치마크(HuggingFace **재배포판** pypi `libero`; 공식 레포의 2022
  의존성 핀 문제 회피). robosuite==1.4.0 은 패키지가 강제.
- **mujoco==3.3.2** — ★고정. 2.3.x=로봇 텍스처 붕괴 / 3.10+=API 크래시 / **3.3.2=정상**.
  임의 업그레이드 금지.
- h5py, pillow, scikit-learn, pyyaml, imageio[ffmpeg](롤아웃 영상).

> **⚠ CUDA 버전 불일치 주의(정직)**: 위 yml 은 **cu128**(로컬 Blackwell 기준)이다.
> 반면 `PROGRESS.md`(2026-07-09)의 원격 학습 박스(`kist_a6000_ss`, RTX 6000 Ada ×10)는
> **시스템 python + torch cu124** 를 그대로 썼다(yml 로 새로 만들지 않음). 즉 **GPU
> 세대에 맞춰 cu124/cu128 를 고르라** — 두 조합 모두 이 코드에서 검증됐다. 별도
> `environment.yml`(구 CLIP-only 트랙)은 gitignore 라 클론에 없다; LIBERO 트랙은
> `environment_libero.yml` 하나면 된다.

libero 최초 import 시 설정 프롬프트가 뜨면 `N` 입력(스크립트에선 `printf "N\n" |` 파이프).
벤치마크 자산은 첫 실행 시 `~/.cache/libero` 로 자동 다운로드된다.

---

## 2. VL 백본 (모델) 준비

정착 파이프라인(README 레시피)의 기본 앵커는 **CLIP ViT-L/14**:
```bash
hf download openai/clip-vit-large-patch14 --local-dir ~/clip_ws/models/clip-vit-large-patch14
# 경로는 configs/config.yaml 의 clip.model_dir 와 일치시킬 것 (기본값 = 위 경로)
```

F-시리즈/최신 실험이 쓰는 앵커는 **config 의 `anchor.model_dir`(HF repo id)** 로 지정되며
`transformers`/`huggingface_hub` 가 **자동 다운로드**한다(HF_HOME 캐시 재사용):
- **SigLIP2-large256** = `google/siglip2-large-patch16-256`(1024-d, 16×16@256) — large256 기질
- **DINOv3 ViT-L/16** = wrist 패치·C2 fine 채널·융합의 DINO 브랜치
- **DINOv2 / RADIO(C-RADIOv4-SO400M)** — F1 앵커 head-to-head

앵커는 `src/core/anchor.py`의 `get_anchor(cfg)`[anchor.py:584]가 config `anchor:` 섹션으로
선택한다. config 에 `anchor:` 섹션이 없으면 기본 CLIP(=ClipWrapper 비트 동형).

> **오프라인 실행**: HF Hub 504/rate-limit 이력 때문에 실운영은 `HF_HUB_OFFLINE=1` 을
> 기본화했다(FOLLOWUP §9 ops 노트). 백본을 한 번 받아 캐시한 뒤에는 이 변수로 네트워크
> 없이 돌릴 수 있다. `HF_HOME`(또는 `~/.cache/huggingface`)에 캐시가 있어야 함.

---

## 3. 데이터 (LIBERO HDF5 데모)

suite 당 ~6~7GB(task 당 hdf5 1개 × 10 task). 필요한 suite 만 받는다:
```bash
python -c "from libero.libero.utils.download_utils import download_from_huggingface; \
           download_from_huggingface('libero_spatial', 'data/libero', check_overwrite=False)"
# object/goal 도 동일: 'libero_object', 'libero_goal'
```
데이터를 다시 받으면 임베딩 캐시를 비울 것: `rm -rf outputs/cache/libero_emb`
(캐시 경로는 config 의 `data.cache_dir` 로 앵커 `cache_key` 별 자동 분리).

---

## 4. 학습 → 롤아웃 (실 CLI — argparse 실측)

`train_phase1.py` / `train_phase2.py` 공통 플래그: `--config` · `--set KEY=VAL`(오버라이드) ·
`--tag`(분리 저장) · `--smoke`. `rollout_sim.py` 플래그(argparse:40~72):
`--config --suite --task-id --episodes --exec-horizon --max-steps
--instruction-mode {correct,wrong,blank} --checkpoint --ablate-zf
--flow-noise-mode {fresh,walk,locked} --run-tag --save-video`.

```bash
# Phase 1 (두 정책 공용)
python src/training/train_phase1.py --config configs/phase1_libero.yaml       # → checkpoints/phase1_libero.pt

# Phase 2 (2-A 표준: flow + lang + wrist)
python src/training/train_phase2.py --config configs/phase2_libero.yaml       # → checkpoints/phase2_libero_fm.pt

# 폐루프 평가 (성공률) — 컨테이너/헤드리스면 MUJOCO_GL 필수(§5)
MUJOCO_GL=osmesa python src/eval_libero/rollout_sim.py \
  --suite libero_spatial --episodes 20                                        # 기대 ≈ 85%

# 언어사용 판별: correct vs wrong 성공률 차이
MUJOCO_GL=osmesa python src/eval_libero/rollout_sim.py --suite libero_spatial \
  --episodes 20 --instruction-mode wrong --run-tag base_wrong_20r
```
wandb 를 안 쓰면 모든 학습 명령에 `--set wandb.enabled=false` 를 붙인다.

---

## 5. ★ 렌더 gotcha — `MUJOCO_GL` (osmesa vs egl)

- **컨테이너/원격/헤드리스에서는 `MUJOCO_GL=osmesa`(CPU 소프트웨어 렌더) 필수.**
  README 예시는 `egl` 로 쓰여 있으나, 실제 원격 박스는 **nvidia EGL 디스플레이 부재**로
  `egl` 크래시(`EGLGLContext._context` 없음 = 렌더 컨텍스트 생성 실패). 그래서 이
  프로젝트의 모든 실측 롤아웃은 **osmesa** 로 돌렸다(PROGRESS 2026-07-09, FOLLOWUP §11).
- **osmesa 의 알려진 결함(정직)**: **확률적 침묵 세그폴트**(Python traceback 없이 ep
  7/28/67/84 등에서 프로세스 사망). 진단 결과 **RAM-OOM 아님**(148GB 여유), osmesa
  소프트웨어 렌더 자체의 세그폴트로 추정(PROGRESS 최신 커밋, FOLLOWUP §11).
- **워크어라운드 = 재시도-슈퍼바이저 패턴**: 죽은 task 를 ep0 부터 재시도(확률적이라 몇
  회면 완주). W-A 확증 6,000 ep 이 이 방식으로 완주됨. 근본 수리(osmesa/GL 스택 교체)는
  박스 관리자 권한 필요.
- 데스크톱 GUI 도구(`latent_mapping.py`, `recovery_probe_gui.py`)는 실제 디스플레이 필요.

---

## 6. 최소 스모크 — 데이터·GPU 없이 env 확인

`--smoke` 는 2 에피소드·3 epoch 로 코드 경로만 점검한다(단, LIBERO 데이터가 조금은
있어야 로더가 도는 셀도 있음). **데이터가 전혀 없어도** 도는 순수 코드-무결성 스모크는
`scratchpad/test_*_smoke.py`:

```bash
# 토큰 canonical 순서 + 비트 동형(신규 옵션이 기본 경로를 안 흔드는지) — 데이터 불요
python scratchpad/test_wa_token_order_smoke.py     # W-A 토큰 순서 규약
python scratchpad/test_libero_byteid.py            # 로더 비트 동형
python scratchpad/test_wc_standardize_smoke.py     # W-C 표준화 가드
python scratchpad/test_f4_build.py                 # F4 fine 채널 빌드/무효과 init

# 데이터가 조금 있으면: 학습 코드 경로 점검(2 eps)
python src/training/train_phase1.py --smoke
python src/training/train_phase2.py --smoke
```
이들이 통과하면 **환경(torch/transformers/모델 로드/모듈 배선)이 살아있다**. 전량
데이터·GPU 없이 "clone → env 성립"을 증명하는 가장 값싼 관문이다.

---

## 7. 문제 해결 요약
- **mujoco 버전**: 반드시 3.3.2(yml 고정). 다른 버전 = 렌더 붕괴/크래시.
- **렌더 오류**: 헤드리스는 `MUJOCO_GL=osmesa`(egl 은 이 박스에서 크래시). 확률적
  세그폴트는 재시도-슈퍼바이저로 우회.
- **HF 504/rate-limit**: `HF_HUB_OFFLINE=1`(백본 1회 캐시 후).
- **데이터 재수집**: `rm -rf outputs/cache/libero_emb`(임베딩 캐시 무효화).
- **완전 재현**: `rm -rf data checkpoints outputs` 후 §1부터.
- **최신 문서 부재**: `docs/`가 gitignore. `reports/2026-07-18_wrist_fusion_session/`
  번들은 커밋돼 있으니 감사·설계·wrist 스크리닝은 거기서 읽고, rseries/pregates/wrist
  확증(`docs/RESULT_*`)은 원본 보관처에서 별도로 받아야 함(§0 표).
