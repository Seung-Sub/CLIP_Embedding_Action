# DESIGN — 파이프라인 전면 재검토 v1 (first-principles review, 2026-07-18)

*작성: 수석 리뷰어 에이전트(읽기 전용 감사 — 코드/설정 무변경). 근거 = FOLLOWUP_experiments.md 전문 ·
AUDIT_flow_crosslab / AUDIT_negative_results / LIT_POSITIONING (2026-07-18 3부작) · HANDOFF.md ·
NUMBER_CARD.md · `src/models/networks.py` · `src/models/policy.py` · `src/core/anchor.py` ·
`src/training/train_phase{1,2}.py` · `src/eval_libero/rollout_sim.py` · configs/. 수치는 문서 기재값
기준(감사 §5: 로컬 1차 아티팩트 10/10 UNTRACED — 원격 회수 전까지 조건부).*

**리뷰 질문(사용자 제기) 3건**: ① concat이 맞는 구조인가? ② SigLIP2+DINOv3가 맞는 모델인가?
③ g에 z_t가 들어가는 것은 "반칙"인가? — 아래 7축 분석 후 TOP-3 후보 구성으로 답한다.

---

## 0. 판정 요약표

| 축 | 판정 | 한 줄 근거 |
|---|---|---|
| 1. g의 z_t 조건 | **KEEP + 진단 의무** | 물리적으로 필요(Δz=Env(z_t,a)) — 반칙 아님. 단 "상태 지름길" 정량화 프로브 4종은 미실행 → 주장 방어에 필수 |
| 2. 액션 정규화 (z-score vs quantile) | **베이스 KEEP / action-flow 셀에만 quantile 이식** | h(MLP+L1)엔 무차별 예상; flow가 수송하는 변수일 때만 분포 자체가 바뀜(감사 원인후보 1위) |
| 3. Phase1 손실 | **구조 KEEP + 블록균형 로깅 추가** | MSE+cos는 실증 승자(C8 DZ 87.0). concat/dual의 블록 스케일 불균형은 미완화(dual 감사 지적) — 전면 whitening은 반대 |
| 4. 인코더 선택 | **SigLIP2-L256 + DINOv3 KEEP**; center-crop 즉시 적용; OTTER-pooling 1셀 신규 | SR은 백본 무차별(F1 3중 null + v2≈v3 매칭전처리) — 레버는 백본이 아니라 전처리와 삽입점 |
| 5. Phase2 구조 | **토큰셋 KEEP**; "middle cell"(JOINT-ζA) 사전등록 스펙 확정 | g(A_past)는 누수 아님(과거 정보 + x0 결합 = 캠페인 승자). 접지 유지 action-flow가 유일한 미검정 중간지대 |
| 6. h 디코더 | **MLP 평균 KEEP (settled)**; 확률적 h는 JOINT-ζA 내부에서만 1회 | 전면교체 33/37, 잔차 48–65, 콜리그 M7 무이득 3중 수렴. eff-rank≈5.5는 "h 확장"이 아니라 "통화 압축(PCA-k)" 신호 |
| 7. 2-stage 구조 | **KEEP**; 반증 팔(JOINT-FT) 1셀만 후순위 예약 | frozen 통화 = C3 해석가능성 + ARM-AE −7.4pp/actionflow −12~21pp = 접지가 하중 지지 실증 |

---

## 1. g 안의 z_t — "반칙"인가? (사용자 우려 정면 분석)

### 1.1 무엇이 걱정인가를 두 가지로 분리

현 구조(`networks.py:34-38`): `g(a_chunk, z_t) → ζ ≈ Δz`. z_t는 CNN 풀링 뒤 head 입력에 concat.

**(a) 누수(leakage) 의미의 반칙 — 아니다.** z_t는 추론 시점에 정책도 보는 값이다. phase2 타깃
`lat_target = ae.g(A_fut, z_cur)`(policy.py:171)에서 미래 정보는 A_fut(지도 타깃, 모방학습의 정의)뿐이고
상태 입력은 현재 z_cur — train/test 정보 비대칭이 없다. LAM 문헌 전반(LAPA/Genie/IGOR의 IDM,
DINO-WM의 FDM)도 상태조건 인코더가 표준이다. 참고로 우리 g는 IDM이 아니라 **FDM(액션→시각변위
전방모델)**, h가 IDM(변위→액션 역모델)에 해당 — "상태조건 전방모델"은 반칙이 아니라 물리다.

**(b) 의미 훼손(semantic adulteration) 의미의 반칙 — 실재하는 위험, 미정량.** ζ=g(a,z_t)는 z_t의
임의 함수를 실어나를 수 있다. 최악 시나리오: 데모는 전문가 정책이라 a≈π*(z_t)이므로 **g가 액션을
무시하고 z_t만으로 Δz를 예측**해도 align 손실이 상당 부분 내려간다(BC 결정론 지름길). 그 경우
ζ는 "행동 통화"가 아니라 "상태 예보"가 되고, phase2는 자기 입력(z_cur)의 함수를 타깃으로 회귀하는
자명 문제가 되며, 논문 서사("ζ≈Δz가 액션 통화")의 근간이 흔들린다.

### 1.2 이미 있는 반대 증거 (완전 붕괴 시나리오는 배제됨)

1. **D4 다봉성**: z-조건부 액션 분산이 스크립트 3배, 결정론 상한 R²≈0.59 (HANDOFF §4.1/§6.1)
   → 상태만으로 액션이 결정되지 않는 데이터. 상태-지름길 g는 cycle L1(h∘g≈a)을 그 상한 이하로
   낮출 수 없는데, 실측 dec R² 0.655–0.749 — cycle 경로가 액션 정보를 실제로 통과시키고 있다.
2. **ARM-AE 대조군**(align=0): 오프라인 recon 동급(0.679)인데 폐루프 −7.4pp CI분리 (HANDOFF §6.5)
   → Δz 정렬 자체가 폐루프 인과 기여. 정렬이 자명해였다면 이 차이가 없어야 한다.
3. **correct−wrong +75~92pp**: 순수 상태-피드백 정책이었다면 wrong 지시문에서도 SR이 유지됐어야
   한다(언어 무시) — 실제로는 붕괴 → 조건화 경로가 살아 있다.
4. **C0 프로브**(`decoder_state_cond=False`, networks.py:161): docstring 기재 "실측: 거의 무손실"
   — h가 z_t 없이도 액션 복원 → ζ 단독으로 액션 정보를 담고 있다는 방증. **단 이 수치의 1차
   아티팩트를 찾지 못했다(UNVERIFIED)** — 재확인 대상.

### 1.3 그래도 반드시 실행할 정량 프로브 4종 (week-0, 학습 거의 불요)

`encoder_state_cond` 플래그는 이미 구현되어 있으나(networks.py:155-163) **결과 문서가 없다** —
"C1 프로브"는 설계만 되고 실행되지 않은 것으로 판단(문서 전수 grep 무발견).

| ID | 프로브 | 측정 | 적신호 기준 |
|---|---|---|---|
| P-zg1 | **액션-셔플**: g(a_π(i), z_i) vs g(a_i, z_i)의 align cos/MSE (배치 내 permute, 학습 불요) | g가 액션을 실제로 읽는 정도 | 셔플 cos ≥ 0.5×원본 → 상태 지름길 |
| P-zg2 | **state-free g′ 재적합**: `encoder_state_cond=false` 1회 phase1 (기구현, config 1줄) | align/a2z retrieval 하락폭 = z_t의 정보 기여 | 하락 "없음"이면 오히려 이상(Δz가 상태 무관?) — 어느 쪽이든 논문 수치 |
| P-zg3 | **상태-단독 상한**: RidgeCV로 z_t→Δz R² (f2_dense_probe.py 재사용 가능) | align R² 중 상태만으로 설명되는 비율 | g의 align R² ≈ ridge R² → g가 상태예보기 |
| P-zg4 | **I(ζ; z_t) 프록시**: ζ→z_t 선형 프로브 R² + h(0, z_t) vs h(ζ, z_t) 절제 | ζ에 실린 상태 중복량 / h의 ζ 의존도 | h(0,z_t) R²가 h(ζ,z_t)에 근접 → 통화 공동화 |

### 1.4 빼면 무엇이 깨지나 + 대안 서열

**빼면 깨진다**: Δz = Env(z_t, a) — 같은 밀기 액션도 씬 배치에 따라 다른 시각변위를 만든다
(libero_spatial은 정확히 "배치만 다른" suite). state-free g의 align 타깃은 환원 불가능한 조건부
분산을 갖게 되어 align cos 하락 + ζ의 Δz 의미 약화(= ARM-AE 방향으로 후퇴, −7.4pp 전례)가 예상.

**대안 서열 (P-zg1~4가 적신호일 때만 발동)**:
1. **잔차화 타깃(innovation grounding)** — align 타깃을 `Δz − r(z_t)` (r = z_t→Δz ridge 예측기,
   동결)로 교체: 상태-예측가능 성분을 통화에서 빼고 "액션이 만든 초과 변위"만 접지. 상태 지름길을
   구조적으로 차단하면서 z_t 조건은 유지 — 가장 원리적.
2. **FiLM 후기 조건화** — z_t를 concat이 아니라 마지막 층 scale/shift로만 주입(상태 대역폭 병목).
3. state-free g + z_t는 h에만 — 물리 위반이라 최후순위(P-zg2가 "무손실"로 나올 때만).

**판정: KEEP.** "반칙" 우려는 (b)형이고, 현 간접 증거는 완전 붕괴를 배제하지만 **정량 방어가 없다**.
P-zg1~4는 총 반나절 비용으로 리뷰어의 "state shortcut" 공격에 대한 선제 방어 + 논문 부록 1절이 된다.

---

## 2. 액션 정규화 — quantile [-1,1] (콜리그) vs z-score (우리)

**사실관계**: 콜리그 = 1/99pct→[-1,1] 클립, gripper(dim6)만 meanstd 분리(`SigLIP/src/core/actnorm.py:31-41`
— gripper 이진분포에서 quantile이 왜곡되기 때문). 우리 = 전차원 z-score(`train_phase1.py:98-102`),
quantile 경로 자체가 없음. 크로스랩 감사는 이것을 **actionflow 격차의 원인후보 1위**로 지목:
잠재-flow에선 무관한 차이가, **flow가 액션을 직접 수송할 때는 수송 대상 분포 자체의 차이**가 된다
(경계 있는 콤팩트 분포 vs heavy-tail).

**우리 내부 전례**: quantile은 이미 한 번 기각됐다 — 단 그 기각은 "오프라인 R² 0.767 = 지표 착시
(동일공간 재측정 0.674)"(HANDOFF §4.4), 즉 **평가공간 착시 문제였지 학습 유해성 판정이 아니었다.**
두 결론은 충돌하지 않는다.

**판정**:
- **베이스 파이프라인(h=MLP+L1, flow=latent): 이식 불요.** L1 회귀 타깃의 단조 재매개변수화는
  분포 꼬리 가중치만 바꾸며, 현 z-score로 97.5%까지 나왔다. 기전 없는 재개는 금지 원칙 위반.
- **action-space flow 셀(§5 JOINT-ζA): 이식 필수.** 감사 권고 (1)항 그대로 — quantile은 "포트
  불충실"로 분류 가능한 유일 요소. config 플래그(`actnorm: quantile|meanstd`) + gripper 분리 +
  롤아웃 역변환 정합(cache-key에 정규화 판 각인)으로 ~40줄.
- **주의**: quantile 이식 후에도 기대치는 "콜리그 base 동률"이지 +레버가 아니다(그들 3-seed 96.2 ≈
  base 96.4 — 97.2는 single-seed 허상으로 자체 강등됨).

---

## 3. Phase1 손실 설계 (align MSE+cos · recon · cycle · HY03)

### 3.1 align: raw Δz에 대한 MSE — 대체로 옳으나 concat/dual에서 블록 불균형 미완화

앵커는 z를 단위구에 놓는다(anchor.py `_post`; concat은 블록별 unit). 따라서 Δz 노름은 작고 프레임
유사도에 반비례 — MSE는 **큰-변위 샘플에 자동 가중**(움직임이 큰 순간을 더 정확히 접지)이고 cos가
방향을 잡는다. 이 조합이 C8 절제의 폐루프 승자(DZ 87.0 ≥ HY01/DA)였으므로 구조는 KEEP.

**실질 문제는 다중블록 기질**: concat 2048의 MSE는 sig-블록 + dino-블록 오차의 단순합, cos는 노름 큰
블록이 지배한다. 두 인코더의 시간 평활도가 다르면 Var(Δz_sig) ≠ Var(Δz_dino) → 한 블록이 align을
독식. **dual-stream 감사가 정확히 이 문제(스케일² 비례 → main 지배)를 "표기만 하고 방치"로 지적했고,
concat도 같은 구조 위험을 가진다(측정 무).** 처방(비용 순):
1. per-block ‖Δz‖·align cos **로깅**(수 분) — 불균형 2× 미만이면 종결.
2. 불균형 시: 블록-표준화 align(블록별 std로 나눈 MSE + 블록별 cos 평균) 1셀 — offline 게이트만으로
   판정 가능(폐루프 불요, dec R²·a2z 동공간 비교).

**ζ whitening/정규화 — 반대.** 콜리그 eff-rank(∂h/∂ζ ≈ 5.5/1024) 발견은 유효 방향이 극소수라는 뜻
— 전차원 whitening은 노이즈 축을 증폭한다(2605.22493의 "과도 정규화 latent는 모드 정보 상실" 경고와
같은 방향). 압축이 맞다면 §6의 PCA-k 통화가 옳은 실험이지 whitening이 아니다.

### 3.2 cycle — 유지 (phase2 디코딩 경로의 학습기)

cycle = h(g(a,z_t),z_t)≈a 는 장식이 아니라 **h를 g-출력 다양체 위에서 훈련하는 유일한 항**이다.
추론 시 h가 받는 것은 진짜 Δz가 아니라 ζ̂≈g-다양체 샘플(policy.py:176)이므로, cycle을 빼면
h는 train(Δz)/test(ζ̂) 분포 불일치를 안는다. 가중 0.25 KEEP — 절제는 셀 낭비.

### 3.3 HY03 InfoNCE — "어느 기질의 어느 블록에, 어떤 투영으로"가 전부

증거 정리: **CLIP-768 raw ζ에선 양성**(HY03 폐루프 87.0 무손실 + t2a/zero-shot 획득 — C8),
**fused-concat ζ의 SigLIP2 블록에선 강한 음성**(S1b-hybrid 67.5%, phase1 a2z retrieval 55.7→4.5
붕괴 — NUMBER_CARD). 음성 감사 판정은 "under-diagnosed"(λ0.3 무튜닝 이식 + phase1 건강지표 일부
미기록)이나, retrieval 붕괴 자체가 기록돼 있으므로 **원인 불문 "이 레시피 그대로는 불가"는 확정**.

원리적 배치(재시도 시 필수 조건 3):
1. **투영헤드 경유** — `contrast_proj`가 구현돼 있으나 미사용(networks.py:204-205). InfoNCE의
   노름/온도 압력이 ζ 기하를 직접 재배치하지 않도록 SimCLR 원리대로 분리. (S1b-hybrid 붕괴의
   가장 유력한 기전 = raw 블록 직접 재배치.)
2. **offline 게이트 선행** — a2z retrieval·dec R²가 dz-only 대비 비열화일 때만 폐루프 진입
   (S1b-hybrid는 이 게이트가 없어서 200롤을 태웠다).
3. λ 스윕 0.05–0.3 (0.3 단일점 이식 금지).
배치 우선순위: C3(semantic interchange) 패키지가 Δz_text 치환 보정을 요구할 때 **avg 기질**에서
1셀 — 그 전까지는 dz-only가 기본. (avg에서 HY03은 미검증 상태다.)

---

## 4. 인코더 선택 공간

### 4.1 이미 settled — 재개 금지

- **"더 좋은 백본 = 더 좋은 SR" 기각**: F1 head-to-head SR 3중 null(matched-reg), DINOv2≈DINOv3
  매칭전처리 동급(90.5 vs 91.5), "DINOv2 우위"는 center-crop confound로 철회(§8). SR을 위해
  백본을 갈아끼우는 셀은 신규 기전 없이 열지 않는다.
- **SigLIP2-large256 vs so400m**: so400m@384 계열은 구캠페인에서 80.2 ≈ CLIP 81.0(HANDOFF §5),
  large256은 16×16@256 그리드가 DINOv3@256과 정합(융합 캐시/전처리 통일의 실리). F1-null 위에서
  so400m 재검은 기대값 0 — **large256 KEEP**, so400m은 후보 아님.

### 4.2 열려 있는 레버 — 우선순위 순

1. **center-crop 전처리 (+~5pp, 미적용)** — DINOv2-avg@crop 96.0 vs 매칭 no-crop 90.5의 격차가
   백본이 아니라 crop이었다는 §8 격리 결과. **주의(긴장 관계)**: 초기 앵커감사는 "crop이 시뮬 렌더
   테두리 12.5% 삭제 = 로봇 관행 위반"으로 no-crop을 채택했었다(anchor.py Dinov2Anchor v2 주석).
   즉 +5pp의 기전은 "작업공간 중앙 확대"이고, 테두리 정보를 쓰는 태스크에서 역효과 가능 → crop을
   DINO 브랜치만/양 브랜치 2팔로 사전등록(TOP-1 후보, §8).
2. **OTTER형 text-aware pooling (2503.03734)** — DINO를 "추가"하는 대신 SigLIP2 **자신의 patch**를
   텍스트 어텐션으로 풀링해 관측 본류에 공급. 왜 신규 기전인가: (i) F3 음성은 학습형 어텐션풀
   +120ep 레짐 + KV-LN 부재의 naive 구현(음성 감사 §3)이라 일반화 불가 판정 — OTTER 풀링은
   frozen 유사도 기반(무학습/저학습)이라 다른 세포. (ii) C5 tradeoff(SR↑↔언어↓)의 원인이 "조건화의
   비언어 시각용량"이라면, **언어-지향 시각 디테일**은 tradeoff 축 밖에 있어야 한다 — 이 가설의
   직접 검정. 성공 시 "DINO가 필요했던 게 아니라 관측 본류의 공간 디테일이 필요했고, 언어-지향
   디테일이면 언어도 안 잃는다"로 서사 강화; 실패 시 C5 다이얼 주장의 대조점 확보. 양쪽 다 논문 가치.
3. **RADIO — 폐루프 불요 (현상 유지).** 상태: `RadioAnchor` 구현 완료(anchor.py:270-348,
   siglip2-g 어댑터 웹검증), granularity 프로브까지만 투입 — 해상도 confound 적발 후 "증류 통일
   공간은 의미·기하 축이 얽혀 Δz 읽기에 부적합" 판정, 편의-베이스라인 역할로 격하
   (RESEARCH_OVERVIEW §5). phase1/2 폐루프 미실행. F1-null 위에서 폐루프 기대값 0 — 리뷰어
   질문("왜 RADIO 안 쓰나")에는 기존 프로브 실측으로 답변 가능. 셀 배정 반대.

### 4.3 LARY(2604.11689) 대응 — "언어접지 지표가 랭킹을 뒤집는다"의 입증 방법

LARY 주장 = LAM 품질(오프라인 재구성류) 기준 DINOv3 > SigLIP. 우리 데이터는 정확히 그 랭킹이
**축에 따라 뒤집힘**을 이미 담고 있다: 오프라인 dec R²는 DINOv2 최상(0.764)인데 폐루프는 언어정렬
앵커 우위(G2 +7.8pp), 언어사용(correct−wrong)은 조건화 SigLIP 함량에 단조(+78→+74→+69).
**입증물 = 인코더축 3열 테이블 1장**: 행 {SigLIP2-only, avg, concat, DINO-중심}, 열 {오프라인
dec R² / 폐루프 SR / correct−wrong (+t2a)} — 열별 랭킹 역전을 그대로 보인다. 신규 실험 거의 불요
(기존 수치 조립 + DINO-only 폐루프 1셀 보강이면 완결). 이것이 C1 기여의 "LARY가 놓친 축" 방어선.

### 4.4 head-to-head 쇼트리스트 (비용 포함)

| 셀 | 내용 | 비용(대략) | 기대 |
|---|---|---|---|
| E-1 crop-avg / crop-concat | §8 TOP-1 | 캐시 재인코딩 + p1/p2 재적합 + 200롤 ×2팔 ≈ 2 GPU-일 | SR +3~5pp |
| E-2 OTTER-pool (b1 조건토큰판) | §8 TOP-2 | dense 캐시 기존(f4 경로 재사용) + p2만 재학습 ≈ 1.5 GPU-일 | tradeoff 파괴 검정 |
| E-3 DINOv3-only 폐루프 (LARY 테이블 보강) | dz phase1+2 | ≈ 1 GPU-일 | 언어축 최하 확인(예측) |
| ~~so400m / RADIO / DINOv2 재대결~~ | — | — | settled, 배정 금지 |

---

## 5. Phase2 구조 — 토큰셋 비판과 "middle cell" 스펙

### 5.1 토큰셋 [z_prev, z_cur, g(A_past), lang, wrist] — KEEP, 비판 2건 기각 1건 보류

- **g(A_past) 누수 우려 — 기각.** A_past는 이미 실행된 액션(추론 시 합법 가용), 상태조건은
  z_prev(train_phase2 `embed_past(Cp, Zp)` = 롤아웃 rollout_sim.py:278 동형 — train/test 정합 확인).
  미래 정보 없음. 게다가 이 토큰은 flow x0(source=past 결합)를 겸하는 **캠페인 승자 부품**이고
  past_noise 0.05가 폐루프 오차 누적을 모사한다. 굳이 지적하면 g(A_past,z_prev)≈z_cur−z_prev라
  (z_prev,z_cur) 쌍과 부분 중복 — 그러나 x0 역할 때문에 제거 불가.
- **z_prev/z_cur vs Δz 토큰 직접 — 보류(저순위).** Δz=z_cur−z_prev 토큰은 파라미터-0 추가지만,
  절대 상태(z_cur)는 어차피 필요(장애물/배치 인지)해서 대체가 아닌 추가만 가능. LayerNorm 뒤
  선형층이 차분을 못 배울 이유가 없어 기대값 소. 셀 배정 안 함.
- **조건 dropout 부재** — 콜리그 레시피(projector dropout 0.2)와의 차이는 의도적(aug/-13pp 감사
  판정, no-aug 클린 밴드). KEEP.

### 5.2 "middle cell" — JOINT-ζA: ζ 접지를 유지하는 action-space flow (사전등록 스펙)

**동기**: actionflow(우리 76/80%, ζ 폐기)와 latent-flow(97.5, h 경유)의 중간지대 — "h 우회가
문제인가, ζ 접지 상실이 문제인가"를 분리하는 유일한 설계. 콜리그 결과(actionflow ≈ base 동률)와
우리 결과(−12~21pp)의 기질 상호작용 가설(감사 원인후보 2위)의 결정 실험이기도 하다.

**설계 (구현 스케치)**:
- flow 변수: `x = [ζ (dim z) ; a_q (112, quantile-정규화)]` — FlowPolicy.flow_dim만 확장, ctx 불변.
- x0 = `[g(A_past, z_prev) ; A_past_q]` (양쪽 다 past 결합 = 기존 승자 관례 유지).
- 손실: `l_fm(CFM, joint)` + `l_act = L1(â_q, A_fut_q)` + **접지 끈** `l_tie = L1(h(ζ̂, z_cur).detach_free, â)`
  (또는 대안: ζ̂ 절반에 기존 lat 손실 `MSE+cos(ζ̂, g(A_fut,z_cur))` 유지 — 이쪽이 더 보수적, 기본팔).
- 추론: â 직접 실행(h 우회) — 단 ζ̂가 접지 손실로 묶여 있어 "통화"는 유지.
- 재계획 안정화(2026 툴킷, 이 셀에서만 의미 있음 — 베이스 잠재-flow는 추론 결정론이라 해당 없음):
  ① **prefix/inpainting**(RTC 2506.07339): 새 청크의 겹침 구간(실행 중 8스텝의 잔여)을 이전 계획으로
  고정하고 나머지만 수송. ② **PAINT**(2606.19774) 노이즈 선택: 이전 청크 backward-Euler 역산으로
  x0 선정 — "--flow-fixed-noise가 실은 고정이 아니었다"(감사 버그)의 성숙한 대체물. ③ BID는
  샘플링-시 폴백(구현 최소). 사전등록: ①만 1차팔, ②는 ①이 경계 불연속을 남길 때만.
- 필수 선행: §2의 quantile actnorm 이식.

**기대값 — 정직하게 낮다.** 콜리그의 actionflow조차 base 동률(96.2≈96.4)이었으므로 SR 상승 기대는
0에 가깝다. 이 셀의 가치는 **기전 증명**이다: JOINT-ζA가 우리 기질에서 actionflow의 −12~21pp를
지우고 latent-flow 동률을 회복하면, "값의 원천 = ζ 접지"가 소거법이 아니라 **구성적으로** 입증된다
(FOLLOWUP §9 해석의 결정판). 회복 못 하면 "h 결정 디코딩 자체가 필수"로 서사가 더 좁고 강해진다.
어느 쪽이든 논문 §접지 절의 최종 문단.

---

## 6. h 디코더 — MLP 평균이 옳다는 증거는 견고; 남은 질문은 "축소"

**settled**: 전면교체 h-flow 33/37%(mode-switching 실증), residual-flow 48–65%, 콜리그 M7(gate
1.0 강제에도 94.0 = 무붕괴·무이득)·Q2 90.4·Q3 미실행. 디코더-측 확률화는 3랩-수렴 음성.

**locked-noise 재론**: 감사가 옳다 — `--flow-fixed-noise`는 에피소드당 1회 시딩 후 generator가
재계획마다 전진(rollout_sim.py:184-186→254)이라 **진짜 모드-락은 시험된 적 없다.** 그러나 재실행
가치는 낮다: (i) 콜리그의 올바른 det_boot M7도 무이득, (ii) 진짜 모드-락의 성숙형이 PAINT이고
그것은 §5.2 JOINT-ζA에 이미 배정됐다. **확률적 h 단독 셀 반대 — 확률성의 마지막 기회는 JOINT-ζA
내부 1회로 한정**(같은 셀에서 prefix+PAINT+quantile이 함께 있어야 공정한 시험).

**eff-rank≈5.5/1024(콜리그) — "h를 키우지 말고 통화를 줄여라" 신호.**
1. week-0: 우리 h의 ∂h/∂ζ Jacobian SVD로 eff-rank 실측(수 분, ckpt만 필요). 콜리그와 동급(≤10)이면:
2. **PCA-k 통화 셀(1개)**: 학습셋 Δz의 PCA 상위 k=32 부분공간에 align/flow를 정의(g 출력·flow_dim·
   h 입력 = k). 문헌 정합: PF-DAG/2605.22493 라인 "compact latent + 준결정 디코딩"이 정확히 이
   처방이고, 우리는 frozen 공간의 주성분이라 해석가능성(각 축의 SpLiCE 개념 분해)도 유지된다.
   **위험**: 언어 민감도가 저분산 축에 실려 있을 가능성 → 언어 공동기준(≥+70pp)이 이 셀의 1차
   게이트다(오프라인 게이트: k=32 재구성 Δz 분산비 ≥95% + dec R² 비열화).

---

## 7. 2-stage vs 대안 (joint / end-to-end latent FM)

**2-stage가 사는 것 (증거 딸린 것만)**:
1. **frozen 통화 = C3 전체**: ζ가 frozen VL 공간에 있어야 SpLiCE/TextSpan 분해·Δz_text 치환·
   역방향 캡셔닝(LIT §3.3)이 성립. joint로 g/h가 움직이는 순간 이 패키지 전부 상실.
2. **접지의 인과 기여 실증**: ARM-AE −7.4pp(정렬 제거), actionflow −12~21pp(h 우회) — 2-stage의
   두 고정점(align 타깃, frozen h)을 각각 빼 봤고 둘 다 손해였다. "2-stage는 제약"이라는 비판에
   대한 실측 반례가 이미 두 개 있는 셈.
3. **모듈성/게이트 경제**: phase1 오프라인 게이트가 폐루프 200롤 이전에 불량 셀을 걸러 왔다
   (S1b-hybrid가 게이트 부재로 태운 비용이 반면교사).
4. **앵커 교환 과학**: 삽입점 지도(C4) 자체가 "통화 고정 + 성분 교환" 설계라서 가능했다.

**"2-stage는 한계다"의 반증 조건(사전등록 가능)**: JOINT-FT 팔 — phase2 학습 중 g/h를 lr×0.1로
해동하되 align+cycle 손실 유지. 판정: SR paired Δ CI>0 **AND** correct−wrong ≥ +70pp면 한계 인정.
예측: align이 정책 그래디언트와 경합하며 ζ가 Δz 다양체를 이탈 → 언어축 침식(HY03-fused 붕괴와
같은 부호). 우선순위 낮음(헤드라인 확정 후 1셀) — end-to-end latent FM(+접지 aux)은 VITA 영토의
재발명이고 frozen-공간 주장을 포기하므로 우리 차선이 아니다.

---

## 8. TOP-3 풀-파이프라인 후보 (thesis-fit × 기대이득 × 비용 순)

공통 사전등록 게이트: **G-off**(phase1: dec R²·align cos·a2z가 해당 base 대비 비열화; phase2 val)
→ **G-cl**(paired bootstrap 10k, 동일 창 matched base 대비 Δ 95%CI>0, 20롤 스크리닝→50롤/3시드 확정)
→ **G-lang**(correct−wrong ≥ **+70pp** 공동기준, wrong-mode 동시 롤아웃). 셀당 per-episode 성공
플래그 저장(감사 §5의 UNTRACED 재발 방지 — 결과 파일명에 실험명 각인).

### P-A 「CROP-AVG」 (+ 자매팔 CROP-CONCAT) — 1순위
- **구성**: 현 avg 융합(SigLIP2-L256 + DINOv3, α=0.5) + center-crop 전처리. 팔 2: crop을 DINO
  브랜치만 / 양 브랜치. concat판 동시 실행(SR-max 트랙).
- **근거**: avg = 이중기준 유일충족 제안 아키텍처(91.5/+74); crop = 격리 실증된 +~5pp 레버(§8
  DINOv2 96.0 vs 90.5). 유일하게 "이미 검증된 레버 × 이미 선택된 구조"의 곱.
- **기대**: avg 91.5→~95-96, 언어 +74 유지(crop은 언어 중립 예상 — G-lang이 검정).
- **비용**: 최저(캐시 재인코딩 + p1/p2 재적합 + 롤아웃, ~2 GPU-일).
- **사전등록**: G-off(dec R² ≥ 0.72) → G-cl(vs avg-no-crop 91.5, CI>0) → G-lang(≥+70). 실패 모드
  예측: 테두리-의존 태스크에서 per-task 하락 발견 시 "crop=중앙확대" 기전 절로 보고.

### P-B 「OTTER-POOL」 — 2순위 (tradeoff 파괴 시도)
- **구성**: SigLIP2-L256 patch 토큰을 지시문 텍스트 어텐션으로 풀링(OTTER 2503.03734 방식,
  frozen 유사도 기반)한 토큰을 관측 본류에 공급. b1: 조건 토큰 +1 (phase1 무변경, 최저가) /
  b2: z에 융합(phase1 재적합) — b1 선행, b1 양성일 때만 b2.
- **근거**: 유일하게 남은 "관측측 + 무게이트 + 신기전" 셀(FOLLOWUP §열린항목의 L1+L4 관측측
  patch 트랙의 원리적 완성형). F3 음성은 naive 구현+저데이터 레짐 판정(감사)이라 봉인 안 됨.
  C5 다이얼의 기전 가설("비언어 용량이 언어를 희석") 직접 검정 — 성공/실패 모두 논문 기여.
- **기대**: SR ≥ 91.5 AND 언어 ≥ +74 동시(= free-lunch 반례) 또는 깨끗한 음성(C5 강화).
- **비용**: 중(dense 캐시는 f4 경로 재사용 가능, ~1.5-4 GPU-일).
- **사전등록**: 예측을 먼저 박는다 — "tradeoff가 용량 희석이면 b1은 언어 비열화 + SR 상승,
  tradeoff가 근본적이면 b1도 단조선 위에 떨어진다." G-off → G-cl(vs avg) → G-lang.

### P-C 「JOINT-ζA」 — 3순위 (기전 증명, §5.2 스펙)
- **구성**: [ζ; a_q] 결합 수송 + 접지 손실 + quantile actnorm + RTC prefix(+조건부 PAINT).
- **기대**: SR 이득 ≈ 0 (목표 = actionflow −12~21pp 소거·latent 동률 회복). 가치 = "ζ 접지가
  값의 원천"의 구성적 입증 + 확률 디코딩/재계획 툴킷의 공정한 마지막 시험.
- **비용**: 최고(~1주: actnorm 이식 + flow_dim 확장 + 롤아웃 경로).
- **사전등록**: 1차 판정 기준을 "vs latent-flow 동률(비열등, CI 하한 > −5pp)"로 — 우월 검정이
  아님을 명시. G-lang 공동기준 동일 적용.

**순위 논리**: P-A = 확실성(검증 레버×채택 구조), P-B = 정보량(가설 검정 양방향 가치),
P-C = 서사 완결(기대 SR 0이지만 thesis 방어력 최대). 셋은 직교라 P-A 진행 중 P-B 준비 가능.

---

## 9. 재개 금지 목록 (settled — 신규 기전 없이 셀 배정 불가)

| 셀 | 종결 근거 | 재개 조건 |
|---|---|---|
| 백본 교체로 SR 추구 (so400m/RADIO/DINOv2 재대결) | F1 3중 null + 매칭전처리 v2≈v3 + RADIO 프로브 | 없음 (언어축 테이블 보강용 DINO-only 1셀만 예외, §4.4 E-3) |
| h 전면교체 flow / 디코더-측 잔차 flow | 33/37·48-65 + 콜리그 M7/Q2/Q3 수렴 | JOINT-ζA 내부 1회로 대체 소진 |
| raw actionflow (ζ 폐기) | 우리 −12~21pp; 콜리그도 base 동률로 자체 강등 | P-C가 상위 호환 |
| S1b-noalign 재실행 | 감사 BEST-SHOT 판정 | 없음 |
| camera-aug/dropout on 이 기질 | −13pp 재기준선 감사 | 기질이 바뀌면(예: OTTER-pool) 재검 허용 |
| wrist dual-stream | 감사 INCONCLUSIVE(NS) — 단 "재개"가 아니라 **진단 미완**: ζ_wrist zero-ablation(5분) 선행 없이는 어느 방향 결론도 금지 | zero-ablation이 기여>0일 때만 |
| C1/C2 게이트 fine 채널 | ∂L_act/∂α≡0 구조 결함 — "이 설계 무이득"만 성립 | 감사 처방 5종 전부 반영 시에만 (P-B가 사실상 대체) |

## 10. week-0 실행 목록 (학습 전, 총 ≤1일)

1. **원격 아티팩트 회수** (감사 §5 — 10/10 UNTRACED 해소, 논문 전 최우선).
2. P-zg1~4 (z_t 지름길 정량, §1.3) + C0 "거의 무손실" 1차 근거 재확인.
3. h Jacobian eff-rank 실측 (§6).
4. concat per-block ‖Δz‖/align 통계 (§3.1).
5. ζ_wrist zero-ablation (§9 wrist 행).
6. rollout 결과 파일 실험명 각인 + per-episode 플래그 저장 패치(1시간, 이후 모든 셀의 전제).

---
*판정의 성격: §1·§3.2·§5.1·§6(MLP)·§7·§9는 현 증거로 종결 가능. §2(quantile)·§3.3(HY03 배치)·
§4.2(crop/OTTER)·§5.2·§6(PCA-k)는 사전등록 실험 대상. 이 문서의 어떤 항목도 기존 no-aug 클린
밴드·paired bootstrap·언어 공동기준 프로토콜을 변경하지 않는다.*
