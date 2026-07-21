# ANALYSIS — CLIP 계열 잠재공간의 언어 한계와 우리 논지의 천장: "task-selection이 아닌 semantic use"는 가능한가 (v1)

*작성: 2026-07-18, Claude Code 리서치 에이전트. PI 질문에 대한 개념 분석.*
*모든 arXiv ID는 본 세션에서 WebSearch/WebFetch로 abstract 대조 검증(4개 병렬 검증 트랙). 검증 불가 항목은 **UNVERIFIED** 명시. 내부 수치 출처: `FOLLOWUP_experiments.md`, `PROGRESS.md`, `docs/W4v3_P2_splice_concept_2026-07-14.md`, `docs/F1_text_geom_behavior_correlation_2026-07-13.md`, `docs/LIT_POSITIONING_2026-07-18.md`, `docs/DESIGN_grounding_space_v1.md`(T-0).*

**PI 질문 (충실 번역)**: 언어를 바꾸면 거동이 달라지는 것은 확인했으나, 이는 "task 선택"처럼 느껴질 뿐 "언어를 제대로 사용한다"는 느낌이 아직 없다. CLIP 계열은 이미지-언어 쌍 대조학습으로 retrieval만 하도록 학습되었는데, 우리가 크게 활용할 지식이 애초에 잠재공간에 있긴 한가? 동작 원리 자체의 한계인가? CLIP 계열을 유지하며 잠재공간 활용법을 더 파고드는 것이 맞나, 아니면 다른 계열/조합의 더 풍부한 잠재공간으로 가는 것이 취지에 맞나?

**한 문단 요약 (전체 결론)**: 대조학습이 cross-modal 코사인 판독(readout)을 "retrieval-shaped"로 만든다는 PI의 직관은 이론·실증 양쪽에서 옳다 — 심지어 대조 목적함수를 *완벽히* 최적화한 인코더가 구성적(compositional) 섭동에 무감각할 수 있음이 증명되어 있다(§1.1). 그러나 "잠재공간에 지식이 없다"는 결론은 따라오지 않는다: (i) 구성/결합(binding) 정보는 uni-modal 임베딩 안에는 존재하며 cross-modal 정렬이 그것을 읽어내지 못할 뿐이고(§1.3), (ii) 우리의 Δz(변위) 설계는 modality gap·도메인 오프셋 같은 대조학습의 병리 성분을 *정확히 상쇄*하는, 이 공간에서 살아남는 부분을 쓰도록 되어 있으며(§3), (iii) "task-selection vs semantic use"는 실재하는 구분이고 **반증 가능한 실험 사다리**가 존재한다(§2) — 그 최저가 실험(T-0, `docs/DESIGN_grounding_space_v1.md` §3.4)은 학습 0·롤아웃 0으로 이미 스펙이 나와 있다. 권고는 "exploit → probe ceiling → migrate only on evidence": 현 스택에서 T-0/C3 델타-주입과 novel-composition 검정을 먼저 돌리고, 실패 시에만 앵커 레지스트리를 통한 E5-V/VLM2Vec 브리지로 이동한다(§5).

---

## 1. 대조 사전학습이 CLIP 계열 공간에 확실히 넣는 것과 빠뜨리는 것

### 1.1 목적함수가 요구하는 것: "쌍 판별에 충분한 특징"까지만 — 이론

- **Wang & Isola (2005.10242)**: InfoNCE는 점근적으로 정확히 두 성질만 최적화한다 — positive-pair **alignment** + hypersphere 위의 feature **uniformity**. 구성적 구조·인과적 결합은 목적함수의 어디에도 없다.
- **Zimmermann et al. (2102.08850)** "Contrastive Learning Inverts the Data Generating Process": 강한 생성모형 가정 하에서는 InfoNCE가 진짜 잠재를 affine/orthogonal 변환까지 복원. 단 가정이 강해 CLIP 실제 학습에 그대로 이전되지 않음 — "잘 되면 의미구조가 생길 수도 있다"의 상한선 논거.
- **★ 2510.26302** "Understanding Hardness of Vision-Language Compositionality (token-level causal lens)": **대조 목적함수를 완벽히 최적화하면서도 SWAP/REPLACE/ADD 섭동에 증명 가능하게 무감각한 "pseudo-optimal" 텍스트 인코더가 존재**함을 보임 — 구성 실패는 목적함수와 *모순되지 않고 정합*("composition non-identifiability"). **PI 직관의 가장 강한 이론적 형식화**: 스케일·데이터로는 원리적으로 고쳐지지 않을 수 있다.
- **ARO (2210.01936)**의 실증판 동일 논거: 캡션/패치를 셔플해도 retrieval 성능이 거의 유지 → 표준 대조 retrieval 학습은 어순·구성 정보를 *필요로 하지 않으므로* 배우지 않는다.

**함의**: "retrieval만 하도록 학습됐다"는 PI의 표현은 정확히는 "**쌍 판별에 충분한 통계까지만 보증**된다"이다. 그 이상(구성, 결합, 가산적 의미구조)은 목적함수의 보증 밖이며, 있다면 *부수적으로*(inductive bias·데이터 통계로) 생긴 것이다. §1.3이 보여주듯 실제로 부수적으로 꽤 생긴다 — 문제는 그것을 어떤 연산으로 읽어내느냐다.

### 1.2 실증된 실패 축 (bag-of-words와 그 이후)

| 벤치마크 | ID | 핵심 수치 | 함의 |
|---|---|---|---|
| Winoground | 2204.03162 | CLIP ViT-B/32 group **8.0%** (chance 16.7, 인간 85.5); 이미지·그룹 점수는 chance *이하* | 어순만 다른 캡션-이미지 결합 불가 |
| ARO | 2210.01936 | VG-Relation **59%**/VG-Attribution **62%** (chance 50), COCO Order 46% (chance 20) | 관계·속성·순서에 bag-of-words |
| CREPE | 2212.07796 | 미학습 화합(compound)에서 R@1 최대 −12%, 복잡도↑ 시 chance로 붕괴 | 체계성·생산성 부재 |
| SugarCrepe | 2306.14610 | 편향 제거 hard negative에서도 SWAP-object/attribute 인간 대비 −27~−50pt; **모델·데이터 스케일과 거의 무관** | 스케일이 안 고침 |
| SugarCrepe++ | 2406.11171 | 어휘 변경(≠의미 변경)과 의미 변경을 혼동; 속성·공간관계 최악 | **어휘 표면형 민감** — 우리 paraphrase 결과의 직계 문헌 |
| MMVP "Eyes Wide Shut" | 2401.06209 | CLIP-blind pair; 오픈소스 MLLM들 chance(25%) 이하; MLLM 실패가 CLIP 실패와 강상관 | CLIP 인코더가 하류 병목이 되기도 |
| Hard Positive Truth | 2409.17958 | hard-*positive* 추가 시 CLIP −12.9%, NegCLIP류 −38.7% | hard-negative 파인튜닝 이득은 부분적 아티팩트 |

**언어의 세부 축별 약점** (우리 §2(c) graded-language 설계의 근거):
- **공간 전치사**: What's Up (2310.19785) — 공간관계만 다른 통제 사진에서 인간 99% vs 18개 VLM 전멸(VQA 파인튜닝 BLIP도 ~56%, 쌍별 chance 근처). 단순 데이터 보정도 거의 무효.
- **수량/계수**: Teaching CLIP to Count (2302.12066) — vanilla CLIP은 counting 인코딩 실패(CountBench), 전용 파인튜닝 필요.
- **동사/시간**: Verbs in Action (2304.06708) — CLIP류 비디오 모델은 "동사 이해 제한적, 명사에 광범위 의존". Test of Time (2301.02074) — before/after 시간순서조차 실패, 사후학습(TACT) 필요.
- **서수/등급**: OrdinalBench (2603.07786, **UNVERIFIED** — 검색 결과만 확인, 원문 미확인) — "N번째 물체" 일반화 한계. "the second bowl from the left"류 서수 참조를 CLIP 텍스트공간에서 다룬 직접 문헌은 미발견.

### 1.3 그럼에도 존재하는 선형 의미구조 — 그리고 우리에게 결정적인 반전

- **★ 2502.03566** "CLIP Behaves like a Bag-of-Words Model **Cross-modally but not Uni-modally**" (ICLR 2026): 속성-객체 **결합(binding) 정보는 uni-modal CLIP 임베딩 안에 존재**하며, 실패하는 것은 cross-modal 정렬이다. 텍스트 임베딩에 **학습된 선형사상 하나**를 걸면 상당 부분 복원. → **우리의 Δz = z_{t+k} − z_t 는 uni-modal(이미지−이미지) 연산**이다. bag-of-words 비판의 본체는 cross-modal 코사인 판독을 겨냥하며, 우리의 주 연산 경로를 직접 때리지 않는다. (단 Δz_text↔Δz_image 교환(C3/T-0)은 cross-modal이므로 이 비판이 적용됨 — §3.2에서 선형 보정으로 재론.)
- **Linear Spaces of Meanings (2302.14383)**: "ideal words" — 합성 개념의 CLIP 임베딩이 소수 인자 기저벡터의 합으로 근사 분해되며, 대조학습이 왜 이런 근사-가산 구조를 유도하는지 기하·확률적 논증. 가산(parallelogram) 구조 존재의 최적 인용.
- **SpLiCE (2402.10376)**: CLIP 임베딩 = 희소 비음수 개념결합, training-free. **TextSpan (2310.05916)**: attention head별 텍스트 라벨된 의미 부분공간(위치/모양/색 전담 head). → 해석 도구는 실재하고 우리 W4v3에서 이미 가동.
- **SIMAT (2112.03162)**: 단 — **vanilla CLIP의 delta 벡터 이미지 변환은 잘 안 되며**, COCO 가벼운 파인튜닝으로 극적 개선. **PC-CLIP (2409.09721)** "Finetuning CLIP to Reason about Pairwise Differences": vanilla CLIP은 word2vec식 산술 성질이 "없다"고 명시, 이미지쌍 차이↔차이설명 텍스트 정렬 파인튜닝으로 부여. → **delta 의미론은 stock CLIP에서 잠재적으로 약하게 존재하고, 저렴한 보정으로 열린다**는 일관된 형태(§3.1).
- LRH (2311.03658)는 개념-방향 가설의 형식화이나 **LLaMA-2 실증(LLM 전용)** — CLIP 증거로 인용하지 말 것.

### 1.4 SigLIP2는 retrieval 이상을 갖고 있는가 — 부분적으로, 그리고 검증 공백

**SigLIP2 (2502.14786) 사전학습 레시피 (검증됨)**: sigmoid 대조손실 + **LocCa (2403.19596)식 캡셔닝 디코더**(전체 캡션 + bbox 예측 + 영역 캡션) + **SILC (2310.13355)/TIPS (2410.16512)식 자기증류·마스크 예측** + 온라인 데이터 큐레이션. 릴리스는 **인코더만**(텍스트 디코더 미공개). localization·dense prediction은 SigLIP1 대비 유의 개선 명시. (SigLIP 원 논문 2303.15343은 sigmoid loss를 효율/배치 스케일 문제로 규정 — 표현 구조 차이 주장 없음.)

즉 SigLIP2 인코더는 **captioning gradient를 이미 통과한** 인코더다. 이것이 중요한 이유:
- **CapPa (2306.07915)**: 캡셔닝 사전학습은 대조 대비 ARO Attribution **85.7 vs 62.7**, Relation **86.7 vs 58.7**, COCO Order **98.8 vs 49.5**; SugarCrepe에서 NegCLIP·OpenCLIP-G까지 상회. **목적함수가 bag-of-words의 원인**이라는 가장 깨끗한 증거.
- 따라서 "SigLIP2 = 순수 대조"는 부정확하다. 구성 신호가 인코더에 일부 주입되었을 개연성이 있다.
- **그러나**: SigLIP2 논문은 ARO/SugarCrepe/Winoground를 **보고하지 않으며**, 2025-26 제3자 평가도 미발견(**UNVERIFIED**). 인접 자료로 SigLIP-B16 Winoground group ~10 = 여전히 chance 근처. **SigLIP2의 캡셔닝 목적이 구성성 벤치를 실제로 올렸는지는 공개 증거 공백** — 우리의 week-0 프로브(§5)에서 직접 잴 수 있고, 재면 그 자체가 작은 기여다.
- 우리 내부 관측과의 정합: SigLIP2 텍스트타워의 per-task 거리↔SR 하락 상관이 CLIP보다 강함(pooled Spearman +0.607 p=0.005 vs CLIP +0.435 p=0.055; `F1_text_geom_behavior_correlation`) — SigLIP2 기하가 행동과 더 정량적으로 결합되어 있다는 약한 방증.

### 1.5 Modality gap — 기하학 정밀 특성화 (§3에서 재사용할 핵심)

- **Mind the Gap (2203.02053)**: 이미지·텍스트가 서로 다른 좁은 **cone**에 상주; 초기화 cone 효과 + 저온도 대조 목적이 분리를 *보존*. gap의 조작적 정의 자체가 모달리티 평균(centroid) 차이 벡터.
- **★ GR-CLIP (2507.19054)**: gap은 "**이미지·텍스트 임베딩 부분공간에 근사 직교하는 상수 벡터**"로 근사 가능; 모달리티별 평균 차감만으로 gap이 즉시 붕괴하며 mixed-modality retrieval NDCG@10 최대 **+26pt**. → **1차 근사: gap = 공유 상수 오프셋.**
- **C3 (2401.08567)** "Connect, Collapse, Corrupt": "Collapse" 단계가 문자 그대로 모달리티별 평균 차감(e′ = e − E[e]); 잔여 오정렬은 등방 노이즈로 모델링 — 상수-오프셋 그림이 1차 근사임을 그들 스스로 인정하는 구조. (주의: 이 문헌 약칭 C3는 우리 기여주장 C3와 무관한 우연 일치.)
- **2차 보정 (정직한 한계)**: Two Effects One Trigger (2404.07983) — gap은 **소수 차원에 집중**되고 두 모달리티 공간은 오프셋 이상으로 "다르게 조직"됨(원인 = 이미지·캡션 정보 불균형). It's Not a Modality Gap (2405.18570) — 단일 모달리티 이중 인코더에서도 gap 발생 = "contrastive gap"(저 uniformity). Cross the Gap (2502.04263, **에이전트 보고로만 확인**) — 평균 이동으로 못 고치는 intra-modal 오정렬 존재.
- **종합 판정**: gap의 **상수-오프셋 성분은 실재하고 지배적**(1차)이나, **완전한 임베딩 호환성**은 성립하지 않음(2차). — 이 비대칭이 §3.2의 논거가 된다: *차분(Δ)은 1차 성분을 정확히 상쇄하지만, 절대 임베딩 교환은 2차 문제를 그대로 만난다.*

### 1.6 §1 결론 — PI 질문 1에 대한 답

"활용할 지식이 잠재공간에 있긴 한가?" — **있다, 그러나 비대칭적으로**: (i) 객체/장면 수준 의미, 근사-가산 개념구조, uni-modal 결합 정보는 있음. (ii) 어순·관계 구성, 등급/서수/수량, 동사 뉘앙스는 cross-modal 판독에서 신뢰 불가(일부는 공간 자체에 부재 개연). (iii) "동작 원리 자체의 한계인가?"에는 **부분 긍정** — 2510.26302가 보이듯 대조 목적함수는 구성 민감성을 강제하지 않으므로, *cross-modal 코사인으로 읽는 한* 원리적 한계다. 단 우리는 그 판독 연산을 쓰는 게 아니라 **uni-modal 변위 + 학습된 디코더(h)**로 읽는다 — 한계의 상당 부분이 우리 경로를 비껴간다는 것이 §3의 논지다.

---

## 2. "task-selection vs semantic use"는 실재하는 구분인가 — 그리고 어떻게 반증하는가

### 2.1 구분의 정식화

우리의 correct−wrong +75.8~92pp는 원리상 두 가설 모두와 정합한다:

- **H_select (task-index 가설)**: 지시문 임베딩은 N개 학습 태스크에 대한 **연속 인덱스 키**로만 기능한다. 정책은 "가장 가까운 학습 태스크의 궤적 분포"를 호출할 뿐, 문장의 *내부 구조*(객체·관계·정도)는 사용하지 않는다. 이 경우도 wrong 지시문에서 SR은 무너진다(다른 인덱스를 호출하므로) — **correct−wrong 지표 단독으로는 H_select를 기각할 수 없다.**
- **H_use (semantic-use 가설)**: 지시문 임베딩의 **구성적 성분**(어떤 객체, 어떤 관계, 어느 정도)이 행동의 **해당 성분**을 인과적으로 결정한다. 판별 기준은 "학습 분포에 없는 조합/정도에 대한 체계적 외삽"과 "언어 벡터 연산이 시연 없이 행동을 생성"하는 것.

중요한 정직성: H_select도 "언어를 안 쓴다"가 아니다 — CLIP 텍스트공간의 **거리 구조를 이용한 selection**이며, 이것만으로도 paraphrase 51-61%(임베딩 이웃이 같은 태스크로 사상)와 1c Faithful 48%(형제 태스크 지시문 = 학습에서 본 문장)가 설명된다. 1c의 스왑 지시문이 **학습 시 본 형제-태스크 문장**이라는 점이 핵심 약점이다: 재타깃 성공은 "공간 언어 사용"의 증거지만, "본 적 있는 인덱스로의 전환"과 아직 구분되지 않는다. **PI의 불만은 정확히 이 갭을 짚고 있다.**

이미 H_select **순수형**(이산 인덱스)을 넘는 내부 증거 하나: per-task 텍스트-거리↔SR-하락의 **연속 상관**(SigLIP2 pooled Spearman +0.607, p=0.005) — 이산 태스크 분류기는 만들 수 없는 **등급적(geometric-graded) 결합**이다. 그러나 이는 "연속 retrieval"과는 여전히 구분되지 않는다. 즉 현 증거 위치는: **이산 선택 < 우리 ≤ 연속 선택**, 목표는 **연속 선택 < 우리**를 보이는 것.

### 2.2 반증 실험 사다리 — 각 셀의 양방향 판정 기준

| 실험 | 설계 | "just retrieval"이면 | "semantic structure"면 | 기존 계획 대비 |
|---|---|---|---|---|
| **(a) Novel composition** | 학습에서 안 본 (공간관계 × 객체) 재조합 지시문. libero_object/goal의 이종 객체로 1c 확장(1c-object) + 관계어 교차("black bowl"→"ramekin" × "next to"→"on"). 지표 = LIBERO-CF (2602.17659) Grounding Rate 프로토콜 | 최근접 학습태스크 행동으로 붕괴(Biased↑ 또는 neither) — GR ≈ CF 기존선(4.7-30.8%) | 미학습 조합에서 GR이 chance·기존선을 체계적으로 상회; 성분별 오류 분해 시 객체·관계 축이 독립적으로 옳음 | LIT §3.3에 **부분 포함**(1c-object는 FOLLOWUP §2 한계로 예고됨) — **관계×객체 교차 셀은 신규** |
| **(b) Δz_text 산술 zero-demo 실행** | **= T-0** (`DESIGN_grounding_space_v1.md` §3.4, 사전등록 완료 — 여기서 재설계하지 않음): `a = h(s·Δz_text, z_t)`, h가 true Δz로 recon 학습된 점을 이용해 **학습 0·롤아웃 0 오프라인 게이트**(G-T0-a/b/c: 데모 액션 대비 방향 정합, wrong-문장 대조 민감도, M6 matched-대각) 통과 시에만 폐루프 주입 | 오프라인 게이트 실패·주입 SR≈0: 텍스트 델타는 h 입력분포 밖(H_select는 시연 없는 과제에 아무 예측도 못 만듦 = **완전 실패 예측**) | 시연 0으로 유의미 방향 정합·GR(목표: CF 기존선 2배 ≥~60%). **성공 시 H_select 최강 기각** — 인덱스로는 원리적으로 불가능한 행동 생성 | **= LIT C3 ① + T-0.** 보정 사다리(§3.2)의 s-스케일은 T-0에 이미 포함; 선형사상·PC-CLIP식 보정기는 T-0 실패 시 폴백(DESIGN §3.3-5와 동일 순서) |
| **(c) 등급/정량 언어** | "move slightly left / much further left", "the second bowl from the left". 2단: (c-i) **텍스트타워 기하 프로브**(CPU, week-0): ‖Δz_text("slightly X"→"far X")‖·방향이 단조·정렬인가; 서수 1st/2nd/3rd가 선형 분리되는가. (c-ii) 통과 시에만 폐루프 | (c-i)에서 이미 퇴화(문헌 예측: What's Up·CountBench·OrdinalBench상 **퇴화가 기본 예측**) → 이 축은 *우리 정책이 아니라 CLIP 계열의 천장*으로 귀속 | (c-i) 통과 + 폐루프에서 변위 크기가 수식어와 단조 → 등급적 semantic use의 강한 증거 | **신규.** 기대치 관리: 실패해도 논지 사망 아님 — "천장 지도"의 한 축으로 보고(§3.4) |
| **(d) 지시문-조건 attention/개념 궤적** | (d-i) SpLiCE Δw 궤적에서 **지시된 개념만 상승·spurious 평탄·swap 시 개념도 swap**(LIT C3 ③); (d-ii) MaskCLIP/TextSpan으로 지시 객체 위치·의미축 소재; (d-iii) 정책 attention이 지시문 토큰→해당 관측 토큰으로 흐르는지 | 개념 궤적이 지시문과 무관하게 동일(태스크 공통 grasp 서사만) — 실제로 W4v3-P2의 현 상태가 정확히 이 경계: **grasp 이벤트는 읽히나 지시-특이 개념(bowl-on-plate)은 미검출** | swap 시 SpLiCE 활성 개념이 지시 방향으로 이동; TextSpan 관계축 활성이 관계어와 공변 | **부분 포함**: W4v3-P1/P2가 (d-i,ii)의 절반. **swap-조건 개념궤적 대비**가 신규 잔여분 |

**판정 규칙의 비대칭성 (미리 등록)**: (b)=T-0가 성공하면 (a)(c)의 부분 실패에도 논지("교환 통화")는 성립한다 — 통화가 실재하되 표현력이 제한된 것. 역으로 (b)가 보정 폴백까지 실패하면, (a)(d)가 아무리 좋아도 "구조는 있으나 *교환*은 안 되는" 공간이며, 이는 §5의 migrate 트리거다. **(c)는 논지의 성패 축이 아니라 천장의 측량 축**이다 — 문헌상 실패가 기본 예측이므로 실패를 결과로 보고하는 실험으로 설계한다(negative result 재활용).

### 2.3 blank 0%의 위치

blank-instruction 0%는 H_select/H_use 어느 쪽 증거도 아니다: ""의 임베딩은 의미적으로 적재된 앵커(null-text guidance 문헌; 2305.06710 — null-text 섭동이 체계적 스타일 이동 유발)이지 "언어 없음"의 중립점이 아니며, 학습분포 밖 OOD 입력이다. correct−blank는 "언어 채널 의존도"의 방증으로만 인용하고 semantic use 논증에서는 제외하는 것이 정직하다 (현 FOLLOWUP의 취급과 일치).

---

## 3. 우리 논지의 정직한 천장 분석

### 3.1 Δz는 "retrieval-shaped" 임베딩의 차분으로서 더 구성적인가, 덜 구성적인가

**차분이 상쇄하는 것 (For)**:
1. **에피소드-상수 성분의 정확한 소거**: 도메인/스타일/렌더링 오프셋(sim LIBERO는 CLIP 텍스트공간에 OOD — W4v3-P2 실측 LS ceiling cos 0.53-0.57, 이미지↔지시문 cos ~0.17)은 한 에피소드 내 근사-상수 → Δz에서 소거. **실측 확인**: 절대 임베딩의 SpLiCE 재구성은 빈약(cos 0.18-0.32)하나 **Δw는 8 에피소드 × 2 타워에서 안정·해석 가능**(reach/rest→grasp/hold 서사) — "Δ가 gap offset 상쇄하고 작동"(PROGRESS W4v3-P2). 이론 예측과 실측이 이미 맞물린 지점.
2. **선례 기제**: StyleCLIP (2103.17249) global direction과 StyleGAN-NADA/DiffusionCLIP (2110.02711)의 **directional CLIP loss** — 1 − cos(ΔI, ΔT), 즉 이미지-임베딩 변위를 변화 의미의 담지자로 삼아 텍스트 델타와 맞추는 연산은 생성/편집 문헌에서 **검증된 표준 기제**다. 우리의 Δz_image↔Δz_text 교환은 이 기제의 제어 버전.
3. **uni-modal 우회**: §1.3 (2502.03566) — 결합 정보는 uni-modal 임베딩에 존재. Δz_image는 uni-modal 연산이고, 그 해독기 h는 *학습된* 판독(코사인 아님)이므로, bag-of-words의 본체 비판(cross-modal 코사인 판독)을 비껴간다. h R² +0.66~0.75와 폐루프 85-97.5%는 "h가 Δz에서 행동 정보를 실제로 판독한다"는 실증.

**차분이 상쇄하지 못하는 것 (Against, 정직)**:
1. **PC-CLIP·SIMAT의 경고**: vanilla CLIP의 (의미적) 이미지쌍 차분은 차이-텍스트와 자연 정렬되지 않으며 보정 파인튜닝이 필요했다. 단, 두 논문의 설정은 **서로 다른 장면의 의미적 차이**이고 우리는 **같은 장면의 시간적 변위**(공유 성분이 훨씬 커서 상쇄 이득이 큼) — 우리 쪽이 유리한 설정이나, T-0 raw 실패 가능성은 실계획에 반영되어 있다(DESIGN §3.3 보정 사다리).
2. **CLIP4IDC (2206.00629)의 경고**: 독립 인코딩된 CLIP 임베딩은 미세 시각 변화를 잘 담지 못함(그래서 그들은 인코더를 적응학습) — pooled Δz의 **입도(granularity) 한계**. 우리 실측과 정합: SpLiCE Δ가 grasp *이벤트*는 읽지만 "bowl-on-plate" 같은 미세 관계 착지는 못 읽음. **Δz의 변화-의미는 이벤트 입도이지 관계 입도가 아직 아니다.**
3. **차분해도 없는 것은 없다**: 등급·서수·수량이 임베딩에 애초 없으면(§1.2) Δz에도 없다. 차분은 노이즈 소거 연산이지 정보 생성 연산이 아니다.

**판정**: Δz는 절대 임베딩보다 **명확히 더 유리한 객체**다 — 대조학습의 두 병리(gap, 저조한 절대-공간 해석성)를 구조적으로 소거하고, 살아남는 변화-의미(이벤트 입도)를 h가 판독한다. 단 "더 구성적"이라기보다 "**덜 오염된**" 것이며, 구성성의 상한은 여전히 기저 공간이 결정한다.

### 3.2 Modality gap과 Δz_text↔Δz_image 교환 — 엄밀 분석

gap의 1차 모형(§1.5): z_img = s_img + γ_img, z_txt = s_txt + γ_txt (s = 공유-스팬 성분, γ = 모달리티별 근사-상수 오프셋, γ ⟂ 공유 스팬; GR-CLIP 2507.19054).

- **Δz_image = s_img(t+k) − s_img(t)** (γ_img 소거), **Δz_text = s_txt(목표) − s_txt(현재)** (γ_txt 소거). → **두 델타 모두 공유 스팬에 산다.** 절대 임베딩 교환이 반드시 만나는 gap 문제를, 델타 교환은 1차에서 *정의상* 만나지 않는다. **이것이 displacement grounding의 가장 강한 선험적 논거**이며, C3-문헌 (2401.08567)의 mean-collapse가 uni-modal 학습→cross-modal 수행을 가능케 한 것과 동일한 기제의 변위판이다. 레포 실측(W4v3-P2)은 이 논거의 국소 확인이고, DESIGN_grounding_space §2가 지적하듯 **cross-modal 델타쌍에서의 직접 측정은 미실측** — §5 Stage 0 프로브 2가 그 측정이다.
- **2차 잔여 (교환이 여전히 실패할 수 있는 이유, 사전 등록)**: (i) 2404.07983 — 두 모달리티 공간은 오프셋 너머 "다르게 조직"(스팬 내 기저 불일치) → 같은 스팬에 있어도 **회전/스케일 불일치** 가능. (ii) 2405.18570 — 저 uniformity로 유효 스팬 자체가 협소. (iii) ‖Δz_text‖(과제 전체 의미 이동) vs ‖Δz‖(0.8s 청크 변위)의 스케일 불일치 — T-0의 s 스칼라가 이것의 1차 처방. → **보정 사다리 (T-0/DESIGN §3.3과 정렬)**: ⓪ raw → ① 스칼라 s(T-0 포함) → ② **단일 선형사상**(2502.03566이 정확히 이 처방으로 결합 복원; Procrustes/ridge, 시연 델타쌍 수백 개로 적합) → ③ PC-CLIP식 경량 보정기(최후 폴백; "공짜 상속" 서사 약화를 정직 보고 — LIT C3 위험 기재와 동일). ②까지는 "frozen 공간의 성질을 드러내는 판독 보정"으로 논지 훼손 없이 방어 가능 — SpLiCE도 사전(dictionary)이라는 판독 장치를 쓰듯, 보정 선형사상은 판독 장치이지 공간 재학습이 아니다.

### 3.3 객체-명사 취약(−37.5pp)은 CLIP 한계인가 데이터 한계인가

증거 분해:
- **기질(substrate) 기여 — 실측**: 텍스트타워가 object-paraphrase를 action-paraphrase보다 ~2배 멀리 배치(1−cos: CLIP 0.0995 vs 0.0561, SigLIP2 0.1166 vs 0.0598)하고, 그 거리가 SR 하락을 예측(pooled Spearman CLIP +0.435/SigLIP2 +0.607) — **하락의 상당분은 정책 암기가 아니라 텍스트타워 기하에서 상속**된다. 문헌 정합: Verbs in Action의 "명사 의존"은 명사 *토큰*이 유사도를 지배한다는 뜻 — 명사 의존적 공간에서는 명사를 동의어로 바꾸는 것이 임베딩을 가장 크게 움직인다. 즉 "명사에 의존하지만 명사-어휘에 불변이 아닌" 공간: SugarCrepe++ (2406.11171)의 어휘-의미 혼동과 동일 현상. **이 부분은 CLIP 계열의 한계가 맞다.**
- **데이터 기여 — 미검정**: 우리 정책의 조건화 학습은 태스크당 사실상 1문장(10개 점)만 본다. h/flow가 텍스트 매니폴드의 극히 희소한 표본 위에서 학습되어, 기질이 보존한 이웃 구조조차 활용 못 했을 수 있다. **판별 실험(저렴)**: 학습 시 GPT-paraphrase 증강 조건화(기질 동결 유지) → paraphrase SR이 크게 회복되면 데이터 한계 우세, 회복이 텍스트-거리 상관을 그대로 따라가면 기질 한계 우세. 부수 이득: W3.3(aug가 언어를 세탁하지 않음)과 같은 형태의 "언어 강건화 레시피" 셀이 된다.
- **판정**: **혼합이되 분리 가능** — 기질 성분은 실측으로 이미 입증(상관), 데이터 성분은 1개 저비용 실험으로 격리 가능. "CLIP 한계라서 어쩔 수 없다"고 결론 내리기 전에 데이터 성분을 소진해야 한다.

### 3.4 천장의 지도 (요약)

| 능력 축 | CLIP 계열 Δz-접지에서 기대 | 근거 |
|---|---|---|
| 태스크/객체 수준 선택·재지향 | **지원됨 (실증)** | correct−wrong +75.8~92pp, 1c Faithful 48/Biased 2.5 |
| 이벤트 입도 변화-의미 (grasp/place) | **지원됨 (실증)** | W4v3-P2 Δw 서사, h R², directional-loss 선례 |
| 어휘-불변 의미 (paraphrase) | **부분 — 기질 기하 상속 + 데이터 희소** | §3.3; SugarCrepe++ |
| 관계 입도 착지 (bowl-*on-plate*) | **경계 — 현재 미검출** | W4v3-P2 한계; CLIP4IDC; ARO/What's Up |
| 미학습 조합 외삽 | **미검정 (§2a)** | — |
| 텍스트-델타 zero-demo 실행 | **미검정 (§2b=T-0) — 논지의 성패 축** | SIMAT/PC-CLIP 경고 vs gap-상쇄 논거 |
| 등급/서수/수량 언어 | **비관 (문헌상 부재 개연)** | 2310.19785, 2302.12066, 2304.06708 |

또한 SR↔언어 단조 tradeoff(FOLLOWUP §5: 조건화 DINOv3 함량↑ → SR↑·언어↓)는 **조건화 채널 구성의 성질**이지 기질 품질의 성질이 아니다 — 어떤 대안 공간으로 가도 비언어 시각용량을 조건화에 더하면 같은 희석이 재현될 것이라는 것이 기본 가설이며, 이 다이얼 자체가 우리 기여(C5)다.

---

## 4. 대안 잠재공간 계열 — 언어구조 / 제어 증거 / 이주비용 / 해석도구 보존 / 취지 적합성

취지(불변 기준): **frozen 공유 공간에서 언어-이미지-행동의 해석 가능한 교환**. 해석 도구(SpLiCE/TextSpan/text-delta 주입)는 "이미지 임베딩과 같은 공간에 텍스트를 인코딩할 수 있는 frozen 타워"를 요구한다.
**모든 대안에 걸린 내부 제약 (중요)**: F1 head-to-head에서 **SR은 앵커에 3중 null** — 이주의 기대이득은 SR이 아니라 **언어축**(paraphrase 회복, T-0/C3 성공, novel-composition GR)에서만 정의되어야 한다.

### (a) CLIP 계열 유지 + 더 잘 쓰기
- **OTTER (2503.03734, ICML 2025)**: frozen CLIP 비전+텍스트 유지, **text-aware visual token pooling**만 학습 — LIBERO unseen 59% vs OpenVLA-ft 29%, 실세계 unseen pick-place 62% vs 9%. "frozen CLIP 의미로 제어가 된다"의 최강 외부 증거이자, 우리 F3(학습형 attention-pool, 음성)와의 결정적 차이 = **풀링 쿼리가 텍스트 조건**이라는 점. 우리 관측-융합 삽입점에 text-conditioned pooling을 넣는 변형은 F3 음성과 기전적으로 구분되는 미검토 셀. **취지: 완전 부합** (공간·도구 전부 보존).
- **SigLIP2 patch-level**: dense 능력 자체는 실재(§1.4)하나 로봇 정책 전용 활용의 독립 문헌 없음(**UNVERIFIED**); 내부 grid-token OOM·F3 음성 전례 → 낮은 우선순위. **취지: 부합하나 내부 증거가 반대 방향.**
- **텍스트타워 prompt-tuning (CoOp 2109.01134)**: CLIP 전체 동결 + 연속 프롬프트 벡터만 학습 — paraphrase 축 보정(어휘 변형을 프롬프트가 흡수)에 정확히 맞는 저비용 레버, 로봇 폐루프 선례는 산발적. **취지: 부합** ("frozen" 주장에 각주 필요: 프롬프트는 입력이지 가중치가 아님).
- **PC-CLIP식 델타 보정 (2409.09721) / 단일 선형사상 (2502.03566)**: §3.2 보정 사다리 ②③ = T-0 실패 시 폴백. **취지: 부합** (판독 장치로 정직 보고 시).

### (b) 생성형 VLM 은닉상태/임베딩
- **증거의 강도**: CapPa ARO 85.7 vs 62.7 (§1.4); **2411.05195** — *동일 비전 인코더*에서 생성형 판독이 CLIP 판독을 구성·공간·세립 축에서 상회(정보는 인코더에 있고 판독이 문제라는 기제 증거); diffusion classifier Winoground 38.5 vs OpenCLIP-H 33.0 (2303.16203, **텍스트 인코더가 같은데도** 생성 경로가 이김). → **생성형 사전학습 공간이 더 구성적이라는 주장은 검증됨.**
- **임베딩화 경로**: **E5-V (2407.12580)** — MLLM에 "summarize in one word" 프롬프트로 이미지·텍스트를 **같은 공간에** 임베딩(modality gap을 프롬프트로 붕괴), 텍스트쌍만으로 학습, composed retrieval zero-shot 강력. **VLM2Vec (2410.05160)** — MMEB 62.9 vs CLIP 37.8. 중간층이 최종층보다 임베딩으로 우수(2502.02013 — LLM 실증, MLLM 전용 증거는 얇음). **MLLM 임베딩을 로봇 정책 조건/앵커로 쓴 선행 연구는 부재 확인(정직 탐색 후)** — 성공 시 그 자체가 신규 기여.
- **해석도구**: E5-V류는 이미지·텍스트가 한 공간 → **SpLiCE 사전 재구축 가능**(개념 어휘를 같은 모델로 인코딩), TextSpan은 CLIP-ViT 구조 특정적이라 **상실**. Δz·Δz_text 문법은 그대로 성립.
- **이주비용 (구체)**: `src/core/anchor.py`의 `BaseAnchor` 서브클래스 1개(~150줄, encode_images/encode_texts/cache_key) + config — **앵커 레지스트리 덕에 구조 변경 없음**(PROGRESS 2026-07-10 `get_anchor(cfg)` 일반화가 폐루프까지 이미 지원). 비용의 본체는 (i) 캐시 재추출(임베딩 차원 ~4096, LLaVA-NeXT-8B급 = SigLIP2-so400m 대비 추론 ~10-20×; 데이터셋 1회 추출 수 시간-GPU), (ii) phase1+phase2 재학습(기존 파이프라인, 1-2일), (iii) 롤아웃 임베딩 추출 지연 증가(8B 모델; 우리 평가는 비실시간 sim이라 수용 가능). **취지: 조건부 부합** — frozen·공유공간·Δz 문법은 유지되나 "경량 대조 인코더의 공짜 상속" 서사가 "생성형 사전학습의 상속"으로 바뀜.
- **π0/OpenVLA 방증**: 필드는 생성형 VLM 백본(PaliGemma 2407.07726; π0 2410.24164, OpenVLA 2406.09246)으로 이동했으나, 생성형-vs-CLIP 백본의 **언어추종 통제 비교는 부재**(UNVERIFIED) — 우리 브리지 실험이 이 공백을 정확히 찌른다.

### (c) Diffusion 잠재/cross-attention 공간
- 구성 증거는 실재(2303.16203; 2303.15233 — CLEVR 결합에서 CLIP chance vs Imagen 최대 100/86%; 단서 2505.17955: 공간관계는 diffusion 우세·counting은 CLIP 우세, in-domain 편중). DIFT (2306.03881) 대응점 특징. 로봇 활용은 **픽셀 경유**(SuSIE 2310.10639 — 언어→편집 이미지 서브골) 또는 특징장(F3RM 2308.07931 — 이것은 사실 frozen **CLIP**의 3D 리프팅; D³Fields 2309.16118은 텍스트 아닌 참조-이미지 목표).
- **치명 결격**: pooled 벡터공간 + frozen 텍스트타워라는 우리 문법이 없음 — Δz 정의 불가, SpLiCE/TextSpan 불가, 교환 통화 부재. 언어는 cross-attention *과정*에 있지 *공간*에 있지 않다. **취지: 부적합** (구성성은 부럽지만 우리 논지의 형태가 아님).

### (d) 3D/flow-정렬 공간
- **DynaFLIP (2605.30350, 실재·내용 확인)**: 이미지-언어-3D-flow 삼중항 simplex-volume 최소화 사전학습, 실세계 OOD 최강 정적 백본 대비 +22.5%. **그러나 배치(deploy) 시 단일 RGB만 사용하고 텍스트 부재** — 공유 텍스트-이미지 공간은 학습 시 구성물이지 배치 인터페이스가 아님 → text-delta 주입·SpLiCE 불가. **취지: 부적합** (인코더 후보로는 F1-null 교훈상 기대이득 없음).
- **SpatialVLM (2401.12168)/SpatialRGPT (2406.01584)**: 정량 공간언어("30cm 왼쪽")를 생성형 VLM에 주입 — §2(c) 등급-언어 천장의 해법이 생성형 경로에 존재함을 보이는 방증. **취지: (b)에 흡수.**

### (e) 명시적 구성/구조 임베딩
- **Structure-CLIP (2305.06152, AAAI 2024)**: scene-graph 유도 hard negative + 지식강화 인코더로 **CLIP-형 공유공간을 유지한 채** VG-Attribution +12.5%/VG-Relation +4.1%. CLIP 아키텍처 체크포인트 → **앵커 스왑 = 사실상 config 변경**으로 우리 파이프라인에 들어옴. 제어 증거는 전무(retrieval만). NegCLIP (2210.01936 부산물)도 동일 논리의 더 저렴한 후보(단 2409.17958의 hard-positive 취약 경고). **취지: 완전 부합 — 가장 싼 "구성성 강화" 브리지.**
- 참고: AM-RADIO (2312.06709)는 다중교사 증류 후에도 언어정렬 헤드를 보존(우리 `RadioAnchor`로 이미 등록됨); Theia (2407.20179)는 텍스트 헤드 보존 여부 **UNVERIFIED**.

### 요약표

| 계열 | 언어구조 | 제어 증거 | 이주비용 | SpLiCE/TextSpan/Δ주입 | 취지 적합성 (한 줄) |
|---|---|---|---|---|---|
| (a) CLIP 유지+활용 | 기존+보정 | OTTER 강함 | ~0 | 전부 보존 | **부합 — 기본값** |
| (b) 생성형 VLM 임베딩 (E5-V/VLM2Vec) | 구성성 검증됨 | 없음(공백=기회) | 앵커 1클래스+재학습, 추론 10-20× | SpLiCE 재구축 가능/TextSpan 상실 | **조건부 부합 — 유일한 진지한 이주 후보** |
| (c) Diffusion 잠재 | 과정에 있음(공간 아님) | 픽셀 경유만 | 파이프라인 재설계 | 전부 상실 | **부적합** |
| (d) 3D/flow 정렬 (DynaFLIP) | 학습시만 | 인코더로서만 | 앵커 스왑 | 배치시 텍스트 부재 | **부적합 (F1-null이라 무익)** |
| (e) Structure-CLIP/NegCLIP | 구성 강화 | 없음 | ~config | 전부 보존 | **부합 — 최저가 브리지** |

---

## 5. 권고 — 단계적 결정 규칙 (opinionated, falsifiable)

**총론**: 현 결과를 버리는 선택지는 없다 — correct−wrong 헤드라인, insertion-point 지도, SR↔언어 다이얼은 **기질-상대적 주장**이라 기질을 바꿔도 살아남고, 오히려 기질 비교의 기준선이 된다. 프레임은 **exploit → probe ceiling → migrate only on evidence**.

### Stage 0 — Week-0 프로브 (CPU/저비용, 훈련 0, 즉시)
1. **T-0 오프라인 게이트** (`DESIGN_grounding_space_v1.md` §3.4 스펙 그대로): `a = h(s·Δz_text, z_t)`의 G-T0-a/b/c — **존재하는 가장 싼 C3 실험**, 학습 0·롤아웃 0. 본 분석의 §2b 판정 기준을 그 게이트에 그대로 건다.
2. **오프라인 cross-modal Δ정렬 측정** (T-0의 G-T0-c/M6와 통합): cos(Δz_text, Δz_image) matched vs mismatched (CKA, LIT §3.3-④), 보정 사다리 ⓪①② 각각에서 — DESIGN §2가 "미실측"으로 지목한 바로 그 수치.
3. **등급-언어 기하 프로브** (§2c-i): "slightly/further", 서수의 Δz_text 단조성·분리성 — 기존 `scratchpad/text_geom_behavior_corr.py` 하네스 확장으로 반나절.
4. **구성성 벤치 자가 측정**: 우리 타워들(CLIP/SigLIP2 + 후보 E5-V/Structure-CLIP)을 SugarCrepe(+What's Up)에 직접 — SigLIP2 공백(§1.4)을 메우는 부수 기여이자 이주 결정의 사전 순위표.

### Stage 1 — 판별 실험 본대 (현 스택, GPU 수일)
- **T-0 폐루프 주입**(오프라인 게이트 통과 시) + **novel-composition 1c-object/관계교차** (§2a) + **paraphrase-증강 학습 대조** (§3.3 데이터-성분 격리) + **swap-조건 SpLiCE 개념궤적** (§2d 잔여분).
- **결정 규칙**:
  - **Stay-and-headline**: raw 또는 스칼라/선형보정(사다리 ②까지) T-0 주입이 GR ≥ ~2× LIBERO-CF 기존선(4.7-30.8%), novel-composition GR chance 상회 → CLIP 계열 유지 확정, C3가 헤드라인. (c) 등급-언어 실패는 "천장 지도"로 동반 보고.
  - **Stay-but-reframe**: 주입이 보정 ③(PC-CLIP식 보정기)에서만 성공 → 유지하되 "공짜 상속"을 "저비용 판독 보정"으로 정직 하향; 보정기 크기·데이터 요구를 표로 (LIT C3 위험 기재 이행).
  - **Migrate 트리거**: (i) 보정 ③까지 주입 실패 **그리고** 오프라인 Δ정렬(Stage 0-2)도 낮음(공간 자체에 교환 구조 부재 판정), 또는 (ii) paraphrase-증강으로도 객체-명사 축 미회복(기질 기하 한계 확정), **그리고** Stage 0-4에서 대안 타워가 텍스트 기하·구성성에서 유의 우위.
- **paraphrase-증강 셀은 결과와 무관하게 이득**: 회복되면 레시피 기여, 안 되면 기질-한계 증거 = migrate 근거.

### Stage 2 — 최저가 브리지 (트리거 발동 시에만)
1. **Structure-CLIP/NegCLIP 앵커 스왑** (~config 수준 + phase1/2 재학습 1-2일): 구성-강화 CLIP이 언어축을 올리는지. 여기서 오르면 이주 없이 해결.
2. **E5-V(또는 VLM2Vec) 앵커 브리지**: `BaseAnchor` 서브클래스 + 캐시 재추출 + phase1/2 재학습 + 20롤 스크리닝 — **총 ~2-4일, 코드 구조 변경 없음**(anchor registry·cache_key 설계가 이미 지원). **측정은 언어축만**: paraphrase SR(+15pp 이상 회복이면 이주 정당), T-0 주입 성부, correct−wrong 유지. SR은 F1-null이므로 판정 기준에서 제외.
3. 이주 확정 시 서사: "동일 Δz 문법을 기질만 바꿔 재검 — 언어축 우위가 생성형 사전학습에서 온다"는 **기질 비교 논문**으로 현 결과 전부가 baseline으로 재활용됨.

### 소견 (opinionated)
PI의 불만("selection처럼 느껴진다")은 **현 증거 상태의 정확한 진단**이다 — correct−wrong·1c·paraphrase는 모두 "연속 retrieval-selection" 가설을 아직 기각하지 못한다(§2.1). 그러나 그 답은 기질 교체가 아니라 **판별 실험**이다: H_select를 기각할 수 있는 가장 싼 실험(T-0, novel-composition)이 아직 안 돌았고, T-0은 학습 0·롤아웃 0으로 오늘 돌 수 있다. 이론·문헌은 우리 설계에 유리한 비대칭을 말한다 — 대조공간의 병리는 절대 임베딩과 cross-modal 코사인에 집중되어 있고, 우리의 uni-modal 변위 + 학습 판독 + gap-상쇄는 그 병리를 구조적으로 비껴가는 조합이다(§3). 실패한다면 그것은 깨끗한 부정 결과이고, 그때의 이주(E5-V 브리지)는 이미 준비된 config 거리에 있다. **지금 이주하는 것은 증거 없이 반나절-수일 실험을 건너뛰고 2-4주 비용을 사는 것**이므로 권하지 않는다.

---

## 부록 — 검증 상태 요약

**VERIFIED (본 세션 arXiv abstract 대조)**: 2005.10242, 2102.08850, 2103.17249, 2109.01134, 2110.02711, 2112.03162, 2203.02053, 2204.03162, 2206.00629, 2210.01936, 2212.04089, 2212.07796, 2301.02074, 2302.12066, 2302.14383, 2303.15233, 2303.15343, 2303.16203, 2304.06708, 2305.06152, 2305.06710, 2306.03881, 2306.07915, 2306.14610, 2308.07931, 2309.16118, 2310.05916, 2310.10639, 2310.13355, 2310.19785, 2311.03658(LLM-only), 2312.06709, 2401.06209, 2401.08567, 2401.12168, 2402.10376, 2402.19119, 2403.19596, 2404.07983, 2405.18570, 2406.01584, 2406.09246, 2406.11171, 2407.07726, 2407.12580, 2407.20179, 2409.09721, 2409.17958, 2410.05160, 2410.16512, 2410.24164, 2411.05195, 2502.02013, 2502.13923, 2502.14786, 2503.03734, 2503.11093, 2505.15659, 2505.17955, 2507.19054, 2510.26302, 2605.30350(DynaFLIP — 실재·내용 확인).
**UNVERIFIED / 부분**: SigLIP2의 ARO/SugarCrepe/Winoground 성적(미보고·제3자 평가 미발견), OrdinalBench 2603.07786(검색결과만), "연속 CLIP 프레임 델타가 동사를 인코딩"류 단독 결과(부재 추정), Theia의 언어정렬 텍스트 헤드 보존 여부, SigLIP2 dense의 로봇 정책 전용 활용, 생성형-vs-CLIP 백본의 VLA 언어추종 통제 비교(부재 = 우리 기회), 2502.04263(Cross the Gap — 에이전트 보고로만 확인), PC-CLIP 세부 수치(abstract 범위 밖).
**내부 수치 출처**: FOLLOWUP_experiments.md §1-11, PROGRESS.md(1a/1b/1c/1d-goal, W4v3), docs/W4v3_P2_splice_concept, docs/F1_text_geom_behavior_correlation, docs/LIT_POSITIONING_2026-07-18 §3.3, docs/DESIGN_grounding_space_v1.md(T-0 §3.4, 보정 사다리 §3.3).
