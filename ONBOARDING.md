# ONBOARDING — 처음 온 사람을 위한 입구 문서

> 이 파일은 **레포를 처음 클론한 사람**이 (a) 연구 전체 서사와 코드를 이해하고,
> (b) 무엇을 어떻게 돌리는지 알고, (c) 방법이 올바르게 수행됐는지 검증할 수 있게
> 하는 **단일 진입점**이다. 설치·실행은 [`SETUP.md`](SETUP.md), 실험 전수 대장은
> [`EXPERIMENTS_INDEX.md`](EXPERIMENTS_INDEX.md), 사용법 상세는 [`README.md`](README.md).
>
> **⚠ 문서 경로 규약**: `docs/`는 작업 폴더라 `.gitignore` 처리되어 **클론에는 없다**.
> 아래에서 인용하는 `docs/RESULT_*`·`docs/AUDIT_*`·`docs/DESIGN_*` 원문은 전부
> **`reports/<세션>/` 번들에 tracked 사본**으로 들어 있다 — 리뷰어는 그쪽을 읽으면 된다:
> [`reports/2026-07-18_wrist_fusion_session/`](reports/2026-07-18_wrist_fusion_session/)(감사·재설계·wrist 스크리닝 19편),
> [`reports/2026-07-22_retrieval_capacity_session/`](reports/2026-07-22_retrieval_capacity_session/)(retrieval·W-A 확증·논문 v2 10편). 각 번들 `README.md`가 인덱스다.

---

## (i) 연구 한 문단 요약

**동결(frozen)된 Vision-Language 인코더의 의미 잠재공간에서, 변위(displacement)
Δz = z_{t+k} − z_t 에 로봇 액션을 접지(grounding)한다.** 이미지를 액션으로 바로
회귀시키는 대신, "카메라 잠재가 얼마나·어느 방향으로 움직였는가"라는 **변위 벡터**를
액션청크의 표현으로 삼는다. 이렇게 하면 (1) frozen VL 공간의 **언어·이미지·액션이 한
공간에서 의미를 교환**할 잠재적 여지가 생기고, (2) 정책이 지시문 언어를 **실제로 사용하며
보존**하는지를 `correct − wrong instruction` 성공률 차이로 **판별**할 수 있으며,
(3) 변위와 액션의 대응이 **해석 가능(interpretable)** 해진다. LIBERO(Franka Panda,
OSC 델타 7D, 20Hz) 벤치마크에서 검증한다. 핵심 발견은 세 가지다 — ① 정책은 언어를
결정적으로 사용한다(언어 지우면 성공률 **+75~92pp** 붕괴), ② 시각기하 정보는 **어디에
넣느냐(삽입점)** 가 전부다(관측/조건화-측은 양성, 타깃/코드-측은 일관 음성), ③ 성능과
언어충실도 사이에 **단조 tradeoff 법칙**이 있다. 프레임의 값은 복잡한 아키텍처가 아니라
**변위 접지라는 삽입점 + 전처리**의 단순성에 있다는 것이 전체 실험의 통합 결론이다.
자세한 프레이밍: [`FOLLOWUP_experiments.md`](FOLLOWUP_experiments.md) §0, [`README.md`](README.md) §7,
[`KICKOFF.md`](KICKOFF.md), [`DESIGN_fusion_dense_latent_action_v1.md`](DESIGN_fusion_dense_latent_action_v1.md).

---

## (ii) 파이프라인 그림 (텍스트 다이어그램)

2-스테이지, **인코더·디코더 동결**. 표기: `A` = 액션청크(16스텝×7D), `z_t` = frozen
VL 앵커가 인코딩한 시각 잠재, `Δz` = z_{t+16} − z_t, `ζ`(zeta) = 학습된 변위-액션 코드.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ ANCHOR (동결)   src/core/anchor.py                                             │
│   get_anchor(cfg)  [anchor.py:584]  → CLIP / SigLIP2 / DINOv2·v3 / RADIO /     │
│                                        DualFusion(avg) / DualConcat(concat)     │
│   이미지 프레임 → z_t (pooled 잠재)  ·  지시문 텍스트 → lang 임베딩             │
└──────────────────────────────────────────────────────────────────────────────┘
        │                                                            │
        ▼  z_t, z_{t+16}                                             ▼ lang
┌──────────────────────────────────────────────┐
│ PHASE 1 — DeltaAE   src/models/networks.py    │   학습: src/training/train_phase1.py
│   class DeltaAE            [networks.py:152]   │        [main: train_phase1.py:277]
│   ┌──────────────────────────────────────┐    │
│   │ g = ChunkEncoder   [networks.py:15]   │    │   g:  액션청크 A (+ z_t)  → ζ ≈ Δz
│   │     (1D-CNN)                          │    │       (액션 → 변위)
│   │ h = ChunkDecoder   [networks.py:41]   │    │   h:  Δz (+ z_t)          → A
│   │     (MLP)                             │    │       (변위 → 액션)
│   └──────────────────────────────────────┘    │
│   손실 losses() [networks.py:237]:             │   align(MSE+cos) + recon(L1) + cycle(L1)
│     + 선택 HY03 언어정렬 info_nce()            │     align_mode = dz | direct | hybrid
│       [networks.py:213] (모션문장 대조)        │     (hybrid = dz + λ·InfoNCE, motion_lang.py)
│   → checkpoints/phase1_*.pt (g, h 동결 저장)   │
└──────────────────────────────────────────────┘
        │  동결된 g, h 를 phase2 가 상속(latent_dim 주입)
        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2 — FlowPolicy   src/models/policy.py        학습: train_phase2.py [main:353]│
│   class FlowPolicy   [policy.py:50]   build_policy_from_cfg [policy.py:154]      │
│                                                                                  │
│   조건 토큰열  [ z_prev, z_cur, g(A_past), lang, wrist ]   ← 토큰 순서 규약 고정  │
│                  0       1        2(A_EMB)  3     4         [policy.py:60]        │
│                        (Z_CUR_IDX=1, A_EMB_IDX=2)                                 │
│        │                                                                         │
│        ▼  조건부 flow matching (ζ 공간 속도장 v, Euler 6스텝, source=past)        │
│      ζ̂  (미래 변위 코드 추론)                                                     │
│        │                                                                         │
│        ▼  h(ζ̂, z_cur)   ← phase1 의 동결 디코더 재사용                            │
│      Â_t  (미래 액션청크)                                                          │
└──────────────────────────────────────────────────────────────────────────────┘
        │
        ▼  폐루프(receding horizon):  16스텝 예측 → 앞 8스텝만 실행 → 재예측
┌──────────────────────────────────────────────────────────────────────────────┐
│ ROLLOUT / 평가   src/eval_libero/rollout_sim.py  (argparse: rollout_sim.py:40~)  │
│   --suite / --episodes / --instruction-mode {correct,wrong,blank} ← 언어사용 판별│
│   --config (ckpt 의 저장 config 로 모델 구조 자동 복원) / --run-tag (JSONL 기록)  │
│   → 성공률(SR) · reward · 영상 · per-episode JSONL provenance                     │
└──────────────────────────────────────────────────────────────────────────────┘
```

**손목캠(wrist) 갈래**: 표준(2-A)은 wrist 토큰 포함(spatial 85.2%), 제외판(2-B,
`phase2_libero_nowrist.yaml`)은 다단계 연쇄 진단용. wrist 를 **추론 스트림**으로
승격한 `DualDeltaAE` [networks.py:287] 도 있으나 폐루프 무익(§EXPERIMENTS_INDEX wrist 절).

---

## (iii) 코드 지도 (`src/` 모듈별 한 줄)

| 파일 | 한 줄 설명 |
|---|---|
| `core/anchor.py` | **다중 백본 앵커 레지스트리** `get_anchor(cfg)` — CLIP/SigLIP2/DINOv2/DINOv3/RADIO + DualFusion(avg)/DualConcat(concat). 이미지→z_t, 텍스트→lang 임베딩. 기본 clip 은 ClipWrapper 와 비트 동형 |
| `core/clip_wrapper.py` | frozen CLIP ViT-L/14 pooled 768 임베딩 + 패치토큰 |
| `core/chunkrep.py` | 액션청크 표현 변환(time/basis) — `to_repr`/`from_repr` |
| `core/config.py` | `configs/config.yaml` 로더 |
| `models/networks.py` | **Phase1 DeltaAE**(g=ChunkEncoder, h=ChunkDecoder) + 언어정렬 InfoNCE + 변형(ChunkFlowDecoder, ResidualFlowDecoder, DualDeltaAE) |
| `models/policy.py` | **Phase2 FlowPolicy**(조건부 flow matching 헤드) + MLPConcat 베이스라인. 토큰순서 규약 고정 |
| `models/obs_fusion.py` | dense patch → K개 관측토큰 attention-pool (F3/grid-token 관측 융합) |
| `models/f4.py` | F4 fine 채널(텍스트쿼리 cross-attn → 96d 병목 → tanh 게이트) — C1/C2 타깃-측 코드 |
| `data/libero.py` | LIBERO HDF5 로더: 임베딩 캐시, (z_t, z_{t+16}, 청크) 쌍, 정책용 삼중쌍 |
| `data/motion_lang.py` | 청크 모션 문장 생성기(지배축×방향×크기+그리퍼) — HY03 언어정렬 타깃 |
| `training/train_phase1.py` | Phase1 DeltaAE 학습(`--smoke`/`--set`/`--tag`/`--config`) |
| `training/train_phase2.py` | Phase2 정책 학습(같은 CLI) — 토큰열 조립·flow 타깃·dual-stream 배선 |
| `eval_libero/rollout_sim.py` | **시뮬 폐루프 평가**(성공률) + `--instruction-mode` 판별 하네스 + `--run-tag` JSONL |
| `eval_libero/rollout_dataset.py` | GT 데모 전체 시계열 추론 → 7D 액션 그래프 + MAE |
| `eval_libero/rollout_sim_serial.py`, `rollout_dataset_serial.py` | 다단계 연쇄(월드모델 rollforward, `--n` 재조회 주기; 2-B 모델 필요) |
| `eval_libero/rollout_sim_paraphrase.py`, `paraphrases.py` | 페러프레이징 전용 폐루프 + 사전 |
| `eval_libero/rollout_sim_libero_para.py`, `libero_para.py` | 공개 LIBERO-Para 벤치 배선 |
| `eval_libero/recovery_probe_gui.py` | 실패 복구·페러프레이징 관찰 GUI(대화형) |
| `eval_libero/latent_mapping.py` | 잠재공간 PCA 시각화(대화형) |
| `diagnosis/f2_dense_probe.py` | F2 dense 디코더빌리티 오프라인 프로브(RidgeCV/MLP, held-out R²) |
| `analysis/*` | 정렬·언어보존·splice·granularity 등 오프라인 분석(paired_ci, lang_retention, splice_concepts, maskclip_*, w5_granularity_probe …) |

---

## (iv) 문서 지도 & 읽는 순서

### 최상위(공개 레포에 커밋됨)
| 문서 | 성격 |
|---|---|
| `ONBOARDING.md` (이 파일) | 입구 — 서사·파이프라인·코드/문서 지도·핵심 결과·검증법 |
| `SETUP.md` | 클론 → 환경 구축 → 스모크. 무엇이 되고 안 되는지 정직하게 |
| `EXPERIMENTS_INDEX.md` | 프로젝트 전 실험 셀 대장(테마별, 판정·근거문서 포함) |
| `README.md` | **정착 파이프라인 사용법** — 설치·phase1/2 학습·롤아웃·§7 결과표 |
| `DESIGN_fusion_dense_latent_action_v1.md` | F-시리즈 연구 설계도("무엇을·왜", 단계 F0–F7) |
| `KICKOFF.md` | 실행자(Claude Code) 착수 지시·**불변식**(폐루프가 유일 심판·사전등록·검증근거·축 분리) |
| `PROGRESS.md` | 개발 devlog(무엇을/어떻게/결과, 최신 순) |
| `FOLLOWUP_experiments.md` | 연구자 토의용 실험 팔로업(§1–15, 통합 서사) |
| `reports/2026-07-18_wrist_fusion_session/` | 감사·재설계·wrist 캠페인 **상세 보고서 번들 19편**(README.md 가 인덱스) |

### `docs/` (⚠ **gitignore — 공개 클론엔 없음**, 아래 (검증)·(가장 최근 결과) 참조)
`AUDIT_`(신뢰성 감사) · `DESIGN_`(설계) · `RESULT_`(결과 판정) · `LIT_`(문헌) ·
`ANALYSIS_`(분석) · `WEEK0/1_`(오프라인 게이트) · `PREREG_`(사전등록) · `PAPER_ARCHITECTURE_v2`.
가장 최신(rseries retrieval · pregates · wrist 확증 · capacity sweep)은 **여기에만** 있다 →
[Report §"가장 최근"] 및 아래 (v)의 근거문서 참조. **이 폴더를 클론에 넣으려면 `.gitignore`
의 `/docs/` 라인 처리 필요**(현재 로컬 전용).

### 새 합류자 읽는 순서
1. **`ONBOARDING.md`(이 파일)** → 전체 지도 잡기
2. **쓰려면** → `README.md`(설치·학습·평가·결과표) + `SETUP.md`
3. **연구 맥락 이해** → `DESIGN_fusion_dense_latent_action_v1.md` → `PROGRESS.md`(로그) →
   `FOLLOWUP_experiments.md`(통합 서사 §0–15)
4. **무엇을 다 돌렸나** → `EXPERIMENTS_INDEX.md`
5. **근거 깊이 파기** → `reports/2026-07-18_wrist_fusion_session/README.md`(19편 인덱스) → 개별 보고서

---

## (v) 핵심 결과 한눈에

| 결과 | 수치 | 근거 문서 |
|---|---|---|
| **언어 following (헤드라인 양성)** | correct−wrong = CLIP **+75.8pp**(3시드) / SigLIP2-large256 **+76.5±1.8pp**(3시드) / CLIP-goal **+88.5** / SigLIP2-goal **+92.0** | `PROGRESS.md`(1a·1d-goal), `FOLLOWUP_experiments.md` §1 |
| **구성적(공간) 접지** | 씬내 타깃-스왑: Faithful **48%** / Biased **2.5%**(VLA OFT/π0 Biased 45–79% ≫ 우리) | `FOLLOWUP_experiments.md` §2, `PROGRESS.md`(1c) |
| **삽입점 지도** | 관측/조건화-측 = 양성(S1 avg 91.5·concat 97.5, W-A wrist +5.5); 타깃/코드-측 = 일관 음성(F1 앵커·C1/C2 게이트·S1b 역할분리·S2 h-flow·dual-stream) | `FOLLOWUP_experiments.md` §3–7, §15 통합; `README.md` §7 |
| **SR↔언어 tradeoff 법칙** | 조건화의 DINOv3 함량↑ → SR↑·언어↓ 단조: concat 97.5/+69 → avg 91.5/+74 → S1b 86/+78. wrist 확증에서 4번째 독립 재현 | `FOLLOWUP_experiments.md` §5, `docs/RESULT_wrist_confirmation.md` §4 |
| **wrist W-A 확증** | 3seed×50roll: correct SR base **85.7** → wristpatch **92.6** = **+6.9pp**, paired 95%CI **[+4.9,+9.1]** SIG>0; 언어 c−w **+63.7** vs base +73.9 (+70 게이트 미달) → tradeoff 프런티어 새 점(미채택) | `docs/RESULT_wrist_confirmation.md`(2026-07-24), `docs/RESULT_wrist_screening.md`(스크리닝 +5.5) |
| **의미교환 천장** | zero-shot 벡터 주입(T-0) 죽음(cos 0.592≈셔플 0.598), 그러나 검색-매개 형태(C3′)는 약하게 부활 | `FOLLOWUP_experiments.md` §13, `docs/WEEK1_gate_results.md` |
| **retrieval C3′** | "주입은 죽고 검색은 산다" 이중해리 우리-기질 재현: R-0 canonical 0.972, R-1 correct 8/8·스왑 56/56·셔플 붕괴. 단 G-R0c-2(독립 3rd셋) 0.773 FAIL — paraphrase 불변성은 학습 어휘 반경 내로 한정. R-A′: SigLIP2 wrist 패치 = DINO 동등(1.008) | `docs/ANALYSIS_colleague_retrieval_control.md`, `docs/RESULT_rseries_R0R1.md`, `docs/RESULT_pregates.md` |

*모든 수치는 별도 명기 없으면 libero_spatial, 10태스크, correct-instruction, no-aug 클린,
MUJOCO osmesa 기준. 콜리그(외부) 수치는 다른 프로토콜(50롤/EGL)이라 **절대비교 불가·방향만 참조**.*

---

## (vi) 어떻게 검증하나 (방법 재현·감사)

- **Provenance JSONL**: 롤아웃이 `outputs/eval/runs/<run-tag>/episodes.jsonl` 에
  per-episode 결과를 상시 기록(`rollout_sim.py --run-tag`, commit `3f0d984`). 판정
  스크립트가 유실돼도 이 로그에서 부트스트랩 CI를 **완전 재계산** 가능 —
  W-A 확증(6,000 ep)이 실제로 이 방식으로 재산출됨(`docs/RESULT_wrist_confirmation.md` §5).
  단 `outputs/` 는 gitignore(원본은 원격 `kist_a6000_ss:/workspace/CLIP_ws/outputs/` 및 wandb `clipvp`).
- **스모크 테스트**(전량 데이터·GPU 없이 코드 무결성 검증): `scratchpad/test_*_smoke.py` —
  `test_libero_byteid.py`(비트 동형), `test_wa_token_order_smoke.py`(토큰 canonical 순서),
  `test_wc_standardize_smoke.py`(W-C 표준화), `test_pb_langselpool_smoke.py`, `test_f4_build.py`.
- **비트 동형(byte-identity) 규율**: 모든 신규 옵션은 **가드 안**에 넣어 기본값이면
  기존 경로와 텐서 단위 `torch.equal`(F0: phase1 22/22 + phase2 51/51 PASS). 새 기능이
  기존 재현 수치를 흔들지 않음을 매번 게이트.
- **사전등록(pre-registration)**: 실행 **전** 예측·게이트 임계를 `docs/PREREG_*` /
  각 RESULT 문서 헤더에 mtime 규약(결과 JSON보다 먼저 기록)으로 고정 → 사후 기준 수정 금지.
  `KICKOFF.md` 불변식 + `docs/upgrade_ledger.md`(비공개 예측 장부)로 적중·반증 정직 기록.
- **폐루프가 유일한 심판**: 오프라인 R²로 백본/설정을 선정하지 않는다(offline≠SR, 여러 번
  실증). 우열 주장은 **3시드 + paired 차분 CI**, CI 겹치면 "구별 불가".
- **신뢰성 감사 트랙**: `docs/AUDIT_*` 가 문서·콜리그 주장을 코드·1차 아티팩트로
  재검증(headline 10/10 재추적 → 7 EXACT, 게이트 α 소수 7자리 일치 = 날조 아님 확정).

---

*질문이 생기면: 사용법은 `README.md`/`SETUP.md`, 왜 그렇게 했나는 `DESIGN_*`/`KICKOFF.md`,
무엇이 언제 어떻게 됐나는 `PROGRESS.md`/`EXPERIMENTS_INDEX.md`/`reports/`.*
