# WEEK1 통합 CPU/캐시-only 게이트 배터리 — 결과 (2026-07-18 실행, 07-20 완결)

*사전등록 출처: `docs/DESIGN_grounding_space_v1.md` §3.4(T-0 G-T0-a/b/c)·§3.1-1(M6),
`docs/PORTFOLIO_divergent_architectures_v1.md` §1–§7(개념별 kill-gate)·§11(CPU-gate 일괄 배치),
`docs/ANALYSIS_clip_language_limits_v1.md` §2c-i(등급-언어)·Stage 0. 실행: 원격 kist_a6000_ss
**CPU 전용**(`CUDA_VISIBLE_DEVICES=""` — GPU 8/9 P-A 학습 무접촉, OMP/torch 8스레드, nice 10),
스크립트 `~/clip_ws/scratchpad/week1_gates/w1_*.py`(신규 — repo 소스 무변경), 결과 JSON
`outputs/week1_gates/`(원격+로컬 동기). WEEK0 하네스 관례 승계. 배터리 7게이트 전부 실행 — 스킵 0.*

## 0. 기질·재현 무결성 (sanity)

- **기질**: large256 단독 — `phase1_libero_siglip2_large256.pt`
  (SigLIP2-large-patch16-256, 1024d, `normalize=False` RAW pooled; 캐시
  `/data2/clip_ws_cache/cache/libero_emb_large256`). DESIGN §3.4 지정 기질
  (dim_text=dim). 텍스트 = 동일 모델 텍스트타워 CPU 로드, 캐시와 동일 규격
  (raw, padding=max_length).
- **split/정규화 재현**: train 22014 / val 5243 (ckpt 기록 일치), 재계산
  align cos **+0.6470** · dec R² **+0.6781** = ckpt metrics 소수 4자리 일치.
- **방향 프록시 검증**: 청크 순 EE 변위(정규화 해제 OSC Δpos 16스텝 합) vs HDF5
  `obs/ee_pos` 실변위 — cos 중앙값 **+0.995** → T-0 방향 판정에 액션-합 프록시 정당.
- 정직 기록: 재인코딩 지시문 vs 기존 lang 캐시 max|Δ| = 9.5e-3 (노름 ~16 대비 미미;
  스레드/fp 비결정성 수준 — 방향 지표에 무영향).

## 1. T-0 (A2 LangSubgoalDisp kill-gate) — `w1_t0.json` → **FAIL / A2 KILL**

**설정**: 10태스크 상태문장 쌍(form A canonical + B/C paraphrase, JSON에 원문 보존;
goal 은 suite 구조상 전 태스크 공통 "the black bowl is on the plate" — 판별력은 init
문장에서만 나옴을 사전 명기). `s = median‖Δz_train‖/‖Δz_text‖` (4.079 / 12.1–19.7,
s≈0.21–0.34). `â = h(s·Δz_text, z_t)` 를 val 5243 상태에서 복호, 순 EE Δpos 방향 cos
vs GT 데모 청크. 대조군: 문장-셔플(derangement 5시드), **h(0,z_t) 상태-단독**, oracle
h(Δz_GT,z_t).

| 조건 | 방향 cos 중앙값 | frac>0 |
|---|---|---|
| correct 문장 | **+0.592** | 0.717 |
| 문장-셔플 | +0.598 | 0.709 |
| **h(0, z_t) 상태-단독** | **+0.798** | 0.789 |
| oracle h(Δz_GT) | +0.985 | 0.980 |

- **G-T0-a (기준: 중앙값>0 AND paired corr−shuf CI95 하한>0)**: 중앙값은 +0.59지만
  **셔플과 동일**(paired 중앙값 −0.011, CI [−0.019, −0.005] — 오히려 미세 음수) →
  **FAIL**. 결정타: **상태-단독 h(0,z_t)가 +0.80으로 텍스트 주입보다 높다** — 방향
  정합의 전부가 상태 지름길(P-zg1/zg3 RED 와 정합)이고, 텍스트 델타는 신호가 아니라
  교란으로 작동. 그리퍼 부호 일치도 correct 0.870 vs zero 0.852 (언어 기여 ≈ 0).
- **G-T0-b (스왑 민감도)**: wrong-task 스왑 Δâ 중앙값 0.374 vs paraphrase 노이즈
  0.097 (**3.85×**) → 기계적 PASS — h 는 텍스트 벡터 방향에 민감하게 반응*하지만*
  (T0-a에 의해) 그 반응이 올바른 태스크 방향이 아님. "주입 경로는 살아있고 내용이
  틀렸다"는 진단.
- **보정 사다리 ② (LOTO dual-ridge, A/B/C 27쌍)**: 예측→과제평균 Δz_img cos
  0.82–0.94 로 보이나(과제평균들 자체가 강한 공유성분을 가져 부풀려짐), 복호 방향 cos
  중앙값 +0.520, paired vs 셔플 CI [−0.016, +0.026] → **무우위, ladder② FAIL**.
- oracle +0.985 → **h 측은 완전 건강** — 실패는 전적으로 Δz_text↔Δz_img 방향 대응
  부재(§2 M6-b)에 귀속.

**판정: G-T0-a FAIL + ladder② FAIL + M6-b FAIL = 사전등록 kill 조건 전부 성립 →
A2 LangSubgoalDisp KILL** (raw/스칼라/선형 보정 범위에서). PC-CLIP식 학습 보정기 ③은
본 배터리 범위 밖 — ANALYSIS §5 "Stay-but-reframe/Migrate" 분기의 남은 폴백으로만 존치.

## 2. M6 modality-gap 소거 — `w1_m6.json` → **M6-a PASS / M6-b FAIL**

**설정**: 절대공간(z_img val vs 상태문장·지시문 43개) vs 델타공간(청크 Δz vs 텍스트
델타 30개, s=0.258 스케일 매칭). 조작적 정의(사전 고정): rel_gap = ‖μ_A−μ_B‖/√평균분산,
소거 = 델타 rel_gap < 0.5×절대. 대각 우세 = argmax 일치 ≥4/10 (이항 p<0.05).

- **M6-a (1차 상수 오프셋 소거)**: rel_gap **2.505 → 0.695 (비율 0.277 < 0.5)** →
  **PASS — cancellation 실재 정량 확인**. gap_norm 24.5 → 2.6. ANALYSIS §3.2 의
  "Δ는 1차 gap 성분을 정의상 소거" 주장이 실측으로 닫힘 (논문 방어 1문단 확보).
- **M6-b (G-T0-c, 방향 대응)**: confusion 대각 top-1 = **1/10** (ep·chunk, form
  A/B/C/평균 전부 1/10) → **FAIL**. 전 태스크의 Δz_text 가 이미지측 태스크 0/1 열로
  쏠림(argmax 고유값 2), diag_mean ≈ offdiag_mean ≈ 0.00 — **태스크 입도의 방향
  대응이 사실상 0**. 문장-스왑이 argmax 를 바꾸지 못함.
- **M6-c (스펙트럼)**: Δz_text 에너지의 img-Δz 상위 8/32 PC 포획률 1.6%/7.4%
  (무작위 0.8%/3.1%, 이미지 자기 자신 48%/72%) — 무작위의 겨우 2×. 주각 cos 최대 0.30.

**함의**: 소거되는 것은 **상수 오프셋(1차)**뿐이고, 남는 공유 스팬에서 **방향 구조
(2차)는 공유되지 않는다** — SIMAT/PC-CLIP 경고(2404.07983 의 "오프셋 너머 다른 조직")
그대로. C3 "공짜 상속" 서사의 오프라인 반증. 델타 문법 자체(M6-a)는 논문 자산으로 회수.

## 3. A1 ConceptBasis / A9 MotionWordVQ — `w1_concept.json` → **둘 다 KILL**

**설정**: 사전 = SpLiCE 201어휘(4-템플릿 앙상블) + 모션어 40, 전체평균 중심화 후 L2
정규화 원자(사전 고정 조작 정의), 중첩 K∈{32,79,114,241}; 비음수 LASSO(평균 활성
6.5–7.8 ≈ k=8); 판정 = 층화 3000쌍에서 frozen-h 복호 R² vs 동일 표본 baseline
h(Δz,z_t)=**0.672**. A9 = 모션 40 코드북 soft-VQ top-m (τ=0.05 train 선택, 스케일
= median‖Δz‖ 전역).

| 사전 | recon R² | recon cos | h(D·w) R² | drop |
|---|---|---|---|---|
| K32 | −0.006 | 0.147 | 0.218 | **0.454** |
| K79 | +0.012 | 0.206 | 0.231 | 0.441 |
| K114 | +0.025 | 0.230 | 0.240 | 0.432 |
| K241 | +0.036 | 0.254 | 0.242 | **0.430** |
| A9 m=8 | (cos 0.158) | — | 0.167 | **0.505** |

- **A1: 최소 drop 0.430 ≫ 기준 0.05 → KILL (통화 변형 사망)**. 결정적 진단:
  h(D·w) R² 0.24 ≈ h(0,z_t) 0.20 — 텍스트-앵커 사전의 재구성은 상태-단독 복호에
  거의 아무것도 더하지 못한다. Δz 분산의 96%가 텍스트 원자 스팬 밖 (recon R² 0.04).
  E5(eff-rank 5) 순풍 가설은 "5축이 존재한다"였지 "그 축이 텍스트 방향이다"가
  아니었음이 판명 — read-out(해석 사영) 변형도 정량 통화로는 무의미, W4v3-P2 식
  질적 Δw 서사 용도로만 존치.
- **A9: m=8 drop 0.505 ≫ 0.10 → KILL**. 질적 모션어 시퀀스(JSON `a9_wordseq`)는
  국소적으로 그럴듯한 구간(reach→grasp→transport→lower)이 있으나 노이즈가 지배 —
  코드북이 제어 정보를 담지 못함.
- 질적 top-concept 표에서도 distractor("a blender", "a clock")가 상위 — 사전 자체가
  Δz 를 설명하지 못함을 재확인.

## 4. A4 RetrievalCond — `w1_retrieval.json` → **KILL**

**설정**: 인덱스 = train 22014쌍, key = unit(z_t) cosine (및 [unit(z_t);unit(lang)]),
top-1/top-5 평균으로 Δz·액션청크 예측. 기준 = **기질-일치 ridge(z_t→Δz) R² 0.4638**
(week0 프로토콜 동형 재계산, α=1; 문서 인용 0.54 는 concat 기질 — 병기). 2차 게이트 =
same-task 제외.

| key | Δz R² top1 / top5 | act R² top1 / top5 |
|---|---|---|
| z_t | −0.092 / +0.321 | +0.127 / +0.437 |
| [z_t; lang] | −0.071 / **+0.332** | +0.204 / +0.478 |
| z_t (same-task 제외) | −0.471 / +0.024 | −0.603 / −0.119 |

- **최고 검색 R² 0.332 ≤ ridge 0.464 → 사전등록 KILL** (0.54 대비 −0.208).
- top-1 은 음수(개별 이웃의 고분산), top-5 평균도 선형 상태예보기에 완패 — frozen
  비모수 메모리는 이 기질/키에서 선형 ridge 이상을 주지 못함.
- same-task 제외 시 완전 붕괴(0.024) — 검색 이득의 사실상 전부가 **태스크
  선택(top1 same-task 87–100%) + 동일 태스크 궤적 위상**의 암기 성분. E7 통과라는
  설계 서사와 무관하게 예측력 자체가 kill 기준 미달.

## 5. A7 LangVerifier — `w1_verifier.json` → **KILL**

**설정**: 상태 1500(태스크 층화), 후보 K=8 (GT + same-task 2 + other-task 5),
Score = cos(g(a,z_t), goal). goal 3종: text(s·Δz_text) / oracle-img(train 과제평균
Δz) / gt-Δz(해당 쌍 GT — 절대 상한). chance 12.5%, 유의 상회 문턱 13.9%.

| goal | top-1 acc | pairwise GT우위 |
|---|---|---|
| text | 0.129 | 0.506 |
| text(wrong 문장) | 0.109 | 0.481 |
| oracle-img (과제평균) | 0.121 | 0.508 |
| gt-Δz (상한) | **0.669** | **0.901** |

- **text·oracle 모두 chance** → 사전등록 KILL (mechanism_alive=False).
- 기전 해부: gt-Δz goal 에서는 0.67/0.90 — **g-공간 채점기 자체는 청크 입도에서
  작동한다**. 죽은 것은 goal 표현: 과제-수준 변위(텍스트든 이미지 평균이든)는 0.8s
  청크 변위와 **시간 입도가 달라** 후보 판별 신호가 없다 (T-0 의 phase 문제와 동근원).
  상태소거 주장(E4 회피)은 확인됨(corr(gap, state-floor) −0.17/−0.12 ≈ 저상관) —
  "argmax 상태 소거" 논리는 맞았으나 남는 액션-성분에 goal-정렬 신호가 없음.

## 6. A5 CycleAlign 전초 (사영 프로브) — `w1_cyclealign.json` → **조건부 PASS**

**설정**: RidgeCV 선형 p: 소스→지시문 임베딩(1024d), val 최근접(cos) 지시문 top-1
태스크 acc (chance 10%). 셔플 대조 = train 표본-레벨 라벨 치환 5시드 재적합.

| 소스 | acc | 셔플 | 우위 |
|---|---|---|---|
| **ζ = g(a,z_t)** | **0.952** | 0.123 | +0.829 |
| raw Δz | 0.415 | 0.122 | +0.293 |
| z_t (상태 천장) | 0.941 | 0.127 | +0.813 |

- 규칙상 **PASS** (ζ 셔플 대비 압도적 우위) — "ζ에서 지시문이 선형으로 읽힌다"는
  성립. **단 정직 조건**: z_t 천장 0.941 과의 격차 +0.011 뿐 — 읽히는 것의 대부분이
  ζ에 실린 **상태 성분**(P-zg4a R² 0.95 와 정합)일 개연이 높다. raw Δz 0.415 는
  "변위 자체도 태스크 정체성을 일부 운반"의 하한 증거.
- **함의**: in-loss CycleAlign 승격 전에 **상태-잔차화 사영 프로브**(ζ − r(z_t) 로
  재검) 1개가 선행 조건 — 잔차에서 acc 가 chance 로 떨어지면 A5 도 실질 사망.

## 7. 등급-언어 기하 프로브 — `w1_graded.json` → **FAIL (천장 확인, 예측대로)**

**설정**: 3축(좌우/상하/전후) × 강도 5단계("a tiny bit"→"very far"), 축 = bare 반대쌍
차분, 판정 = ≥4/6 방향 |Spearman|≥0.9. 문헌상 실패가 기본 예측(사전 기재).

- **0/6 방향 단조** (ρ +0.5, +0.2, +0.6, −0.6, +0.6, −0.9 부호 뒤죽박죽). bare 형이
  4/6 방향에서 최대 사영 — 수식어는 임베딩을 크게 움직이나(노름 10–21) **강도
  축으로 정렬되지 않음**. PI의 gradation 질문에 대한 데이터 답변: **SigLIP2 텍스트
  공간에 "slightly/far" 등급 구조는 없다** — 이 축은 정책이 아니라 기질의 천장
  (ANALYSIS §3.4 지도의 "비관" 행 실측 확정).
- 소득 1건: **서수**("first/second/third/fourth bowl from the left")는 사영 단조
  ρ=1.0 (−9.78→−1.43) — 서수 참조축은 약하게나마 존재 (후속 참고용 각주).

## 8. 종합 — 포트폴리오 판정과 남는 것

| 게이트 | 사전등록 기준 | 실측 | 판정 | 개념 운명 |
|---|---|---|---|---|
| T-0 G-T0-a | 방향cos>0 + 셔플 우위 | corr=shuf(−0.011), h(0,z)가 우월 | **FAIL** | **A2 KILL** |
| T-0 ladder② | 선형보정 후 셔플 우위 | CI [−0.016,+0.026] 무우위 | FAIL | (A2 kill 확정 요건) |
| M6-a 소거 | rel_gap 비율<0.5 | 0.277 | **PASS** | Δ-문법 방어 자산 |
| M6-b 대각 | ≥4/10 | **1/10** (전 form) | **FAIL** | A2·A7 kill 입력 |
| A1 | drop≤0.05 | **0.430** (recon R² 0.04) | **KILL** | ConceptBasis 통화 사망 |
| A9 | m=8 drop≤0.10 | **0.505** | **KILL** | MotionWordVQ 사망 |
| A4 | 검색 R²>ridge 0.464 | 0.332 (제외시 0.02) | **KILL** | RetrievalCond 사망 |
| A7 | 랭킹>chance | text 0.129·oracle 0.121 (gt상한 0.67) | **KILL** | LangVerifier 사망 |
| A5 전초 | 셔플 우위 | acc 0.952 (단 z_t 천장 0.941) | **PASS(조건부)** | 잔차화 재검 후 결정 |
| 등급-언어 | ≥4/6 단조 | 0/6 (서수만 ρ=1.0) | **FAIL(예측대로)** | 천장 지도 확정 |

**핵심 판독 3줄**:
1. **h·g 기전은 건강하다** (oracle 방향 0.985, gt-Δz 랭킹 0.90) — 죽은 것은 전부
   "**텍스트/과제-수준 벡터 ↔ 청크-수준 이미지 변위**"의 대응이며, 원인은 (i) 방향
   대응 부재(M6-b)와 (ii) 시간 입도 불일치(과제 변위 vs 0.8s 청크)의 이중.
2. 포트폴리오의 언어-확장 개념군(A2·A7·A1·A9)과 검색 개념(A4)은 **오프라인에서 전원
   사망** — 큐에 남는 유효 후보는 기존 검증 트랙(P-A crop, P-B, R-Δ, wrist Stage1,
   innovation-grounding/잔차화)뿐. 언어축 개선의 남은 경로는 ANALYSIS §5 순서대로
   (a) paraphrase-증강 (b) 보정기 ③(PC-CLIP식 학습 정렬기 — "공짜 상속" 하향 정직
   보고 전제) (c) Structure-CLIP/E5-V 앵커 브리지.
3. 순수 소득: M6-a(오프셋 소거 실증), 서수축 존재, A5 전초 양성, "상태-지름길이
   방향 지표까지 지배(h(0,z_t) cos 0.80)"라는 진단 — 전부 논문 §분석/§한계 재료.

*wall-clock: prep(쌍 재구성+텍스트 1,146문장 CPU 인코딩) ≈ 1.7h + 게이트 7본 ≈ 1.5h,
스킵 없음. 결과 JSON 7본 = `outputs/week1_gates/w1_{t0,m6,concept,retrieval,verifier,
cyclealign,graded}.json` (원격 `~/clip_ws/outputs/week1_gates/` 동일본).*
