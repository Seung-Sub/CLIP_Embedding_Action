# PORTFOLIO — 발산적 아키텍처 후보군 v1 (divergent architecture ideation)

*작성: 2026-07-18, 발산 아이디에이션 리서치 에이전트 (읽기 전용 — 본 문서 외 무변경).*
*PI 지시: 현 구조(DeltaAE g/h + FlowPolicy + SigLIP2/DINOv3 융합)의 반복 개선을 멈추고,
테제 — **"언어정렬 frozen 공간의 변위 = 언어·이미지·행동의 공용 통화, 해석가능한 로봇 정책"** —
를 더 잘 실현할 수 있는 **구조적으로 다른** 아키텍처 개념을 생성·비판·서열화하라.*

*증거 규율: 모든 개념은 §0의 확립된 음성지도와 적신호에 대해 (i) 모순하지 않음 또는
(ii) 의도적 도전 + 기전 차이를 명시하고, (iii) 최저가 오프라인 kill-gate를 갖는다.
근거 문서: `FOLLOWUP_experiments.md` · `docs/WEEK0_probe_results.md` ·
`docs/DESIGN_grounding_space_v1.md` · `docs/ANALYSIS_clip_language_limits_v1.md` ·
`docs/DESIGN_wrist_fusion_unified_v1.md`.*

**문헌 검증 태그**: **[V]** = 기존 문서(DESIGN/ANALYSIS/LIT)에서 abstract 대조 검증 승계 ·
**[S]** = 본 세션 WebSearch 스니펫 수준 확인(초록 전문 대조 아님 — 채택 전 abstract 재검증 요) ·
**UNVERIFIED** = 미확인(기억/추정).

---

## 0. 증거 제약 요약 — 모든 개념이 통과해야 하는 필터

| # | 확립된 사실 | 출처 | 개념 설계에의 함의 |
|---|---|---|---|
| E1 | 삽입점 지도: **관측/조건화-측 = 유일 양성**(S1 concat 97.5/avg 91.5), 타깃/코드-측 4중 음성(C1·C2·S1b·dual-wrist) | FOLLOWUP §3–6, §11 | 새 정보는 조건 토큰으로; 타깃-측 재진입은 기전 차이 입증 필요 |
| E2 | 디코더/정책 복잡화 일관 음성: h-flow 33–37, residual-flow 48–65, actionflow 76–80(ζ 접지 폐기 = 손해) | FOLLOWUP §7, §9–10 | 용량 추가·확률화 금지; 통화 재정의(R-Δ류)는 별개 클래스 |
| E3 | HY03(InfoNCE 모션문장, ζ 언어정렬)이 fused ζ에서 역효과(67.5%) | FOLLOWUP §6 | ζ에 대조손실 재도입 금지; alignment-only(반발항 없는) 기전은 별개 |
| E4 | P-zg1/zg3 **RED**: align cos의 65–70%가 상태성분, ridge(z_t→Δz)가 g를 능가; P-zg4a: ζ가 z_t를 R² 0.90으로 운반 | WEEK0 §1–3 | "align 높음=액션 접지" 인용 금지; 상태성분이 소거되는 연산(후보 간 argmax, 잔차화, 개념사영)은 가산점 |
| E5 | h Jacobian eff-rank ≈ 4–5 / 전체차원; offline≠SR | WEEK0 §4, FOLLOWUP §9 | 소차원 통화(개념 basis, PCA-k, 코드북)가 원리적으로 가능; 모든 판정은 폐루프+오프라인 이중 |
| E6 | SR↔언어 단조 tradeoff(조건화 비언어 시각용량↑ → correct−wrong↓); 공동기준 c−w ≥ +70pp | FOLLOWUP §5 | 비언어 조건 토큰 추가 개념은 언어 공동기준을 kill 조건에 포함 |
| E7 | 정책-생성 토큰은 조건 정보의 재부호화일 뿐 신규 정보 0 (Phase-B 폐루프 널의 기전 설명) | DESIGN_wrist §1 | 조건 토큰은 **측정·검색·언어 유래**여야 함 — 자기 예측 토큰 금지 |
| E8 | 추론은 완전 결정론(x0=past); 주입 슬롯은 (a) 조건 토큰 (b) h-측 둘뿐 | DESIGN_grounding §7–8 | 노이즈/프라이어 조작류 개념 기각 |
| E9 | h는 recon 항으로 true Δz를 직접 학습 → raw-Δz 좌표는 h의 in-distribution | DESIGN_grounding §1·§3 | 학습-0 kill-gate의 만능 지렛대: 임의 후보 통화 x를 h(x, z_t)로 즉시 디코드 검정 가능 |
| E10 | Δ(변위)는 modality gap 1차 성분(상수 오프셋)을 정의상 소거; 절대 임베딩 교환은 2차 문제 직면 | ANALYSIS §1.5, §3.2 | 언어↔이미지 교환 개념은 반드시 델타-대-델타로 설계 |
| E11 | 박스 상태: CUDA 신규 프로세스 불가(복구 대기), osmesa 롤아웃 확률 세그폴트 | WEEK0 §0, FOLLOWUP §11 | **CPU-only 오프라인 게이트를 가진 개념이 지금 당장 착수 가능** — 큐 배치의 실질 변수 |

기존 큐(불변 전제): P-A crop(진행), wrist 3-arm(W-A/B/C, 착수 대기), T-0/M1–M6 배터리,
innovation-grounding(잔차화 align), R-Δ(raw-Δz 통화), P-B(LangSelPool).

---

## 1. A2 — **LangSubgoalDisp**: 언어-생성 서브골 변위 추적 (seed S2)

**기전.** T-0의 아키텍처 승격판. 소형 보정기 `C`(사다리: ⓪ raw → ① 스칼라 s → ② 선형
Procrustes/ridge — ANALYSIS §3.2)가 `Δz_text = E_text(goal) − E_text(now)`를 이미지-델타
다양체로 사상하고, 그 결과를 **goal-displacement 조건 토큰** `Δz_goal = C(Δz_text)`로
정책에 상시 주입한다(학습 시 데모 태스크 문장에서 계산 — 배관은 lang 토큰과 동형).
롤아웃에서는 (b-슬롯 병용) `h((1−β)·ζ̂ + β·Δz_goal, z_cur)` 블렌드 β 스윕이 가능(E8의
허용 슬롯 a·b 정확히 둘 다 사용). 시연 없는 새 과제 = 문장만 갈아끼우면 Δz_goal이 바뀜 —
**zero-demo 전이가 프로브가 아니라 상시 기능**이 된다.

- **Thesis-fit 10/10**: "언어·이미지·행동의 공용 통화" 주장에서 아직 비어 있는 방향
  (언어→행동, 시연 무)을 아키텍처로 채움. H_select(retrieval-selection 가설)의 최강 기각
  장치 내장(ANALYSIS §2.2-b: 인덱스로는 원리적으로 불가능한 행동 생성).
- **증거지도 정합**: 조건화-측 삽입 = 유일 양성 삽입점(E1). 토큰이 **언어 유래**라 E6
  tradeoff의 비언어 희석이 아니라 오히려 언어 함량 증가 방향. E7 통과(자기 예측 아닌 언어
  유래). E8 슬롯 규약 준수. E10의 델타-대-델타 설계 그대로. 모순하는 음성 결과 없음.
- **신규성**: SuSIE [V 2310.10639]=픽셀 서브골(diffusion 편집), PC-CLIP [V 2409.09721]=
  정렬 학습만·제어 없음, "Latent Goal Prediction from Language for Model-Based Planning"
  [S 2606.20627]=**최근접 이웃**(언어→잠재 goal 예측, 플래닝용 — 단 변위 통화·frozen 역모델
  h 경유·gap-소거 논증 부재로 보임, abstract 재검증 요), zero-shot task spec [S 2204.11134]=
  유사도 보상만. **변위-통화 + frozen h 복호 + 조건토큰 상시화** 조합은 미발견 — 신규성 中上.
- **최저가 kill-gate**: 이미 사전등록된 **T-0 G-T0-a/b/c + M6**(DESIGN_grounding §3.4)이
  그대로 이 개념의 gate다 — 학습 0, 롤아웃 0, GPU <0.5h. **Kill**: M6 matched-대각 우세
  부재 AND 보정 사다리 ②(선형)까지 방향 cos 중앙값 ≤ 0(셔플 대조 대비 무우위) → 개념 사망
  (그 실패 자체가 ANALYSIS §5의 migrate 트리거 입력으로 회수됨).
- **증분 비용**: 보정기 = 수백 델타쌍 ridge(CPU 분); phase2 재학습 1런(+1토큰, ctx +1.6M,
  ~2 GPU-h); 캐시·h·phase1 전부 재사용. 폐루프 스크리닝 0.5 GPU-일.
- **Steelman**: *T-0이 성공하는 순간 이 설계는 "언어가 시연 없이 행동을 생성하는" 능력을
  정책의 상시 입력으로 만든다 — 논문 헤드라인(C3)이 곧 아키텍처인 유일한 후보.*

## 2. A4 — **RetrievalCond**: 판례-인용 검색증강 조건화 (seed S4)

**기전.** frozen 공간이 공짜로 주는 non-parametric 메모리. 학습 세그먼트 전체(~22k)를
key = [z_t; Δz_past; lang]로 인덱싱(단일 행렬, 1024–2048d — FAISS 불요, matmul <1ms).
재계획마다 top-k(k=4–8) 이웃의 **(Δz_fut, g(a))를 조건 토큰으로 append**(고정 선형 사영,
학습형 풀링·게이트 없음). 검색 결과는 매 결정마다 로그 → 정책이 "어느 데모의 어느 구간을
판례로 인용했는지"가 감사 가능. PI의 "selection처럼 느껴진다"는 불만을 **명시적·감사가능한
selection 채널**로 반전시키고, flow 정책은 그 위의 잔차(의미적 보정)를 담당한다.

- **Thesis-fit 7/10**: 통화 교환축(언어→행동)을 직접 늘리진 않으나, "해석가능한 로봇 정책"
  축을 결정-단위 인용으로 강화. frozen-공간 테제의 실용 배당(재학습 없는 메모리) 실증.
- **증거지도 정합**: 조건화-측(E1 양성). **E7을 통과하는 희소한 방법** — 검색 토큰은 정책이
  못 만드는 **외부(훈련셋) 정보**를 시험 시점에 주입한다(정책-생성 토큰과 정반대). F3
  음성과의 차이 = 학습형 attention-pool 부재(F3 실패 기전), C1/C2와의 차이 = 게이트 부재
  (잔차 게이트 굶음 회피). **정직한 위험**: 검색 토큰은 비언어 용량 → E6 tradeoff의 wrist/
  DINOv3 축과 동형의 언어 희석 가능 — c−w ≥ +70pp를 kill 조건에 명시. novel-composition
  과제에서 잘못된 판례로의 붕괴 위험 → 인용 로그가 그 실패를 그대로 가시화(진단 도구 겸용).
- **신규성**: 밀집 분야 — STRAP [S 2412.15182](학습-데이터 검색 증강), Retrieve-Don't-
  Retrain [S 2606.15631](VLA test-time 검색, cross-embodiment 풀), ReMoBot [S 2408.15919]
  (검색 자체가 정책), RoVer 계열, VINN [UNVERIFIED 2112.01511](kNN 모방). **차별점 =
  언어정렬 변위공간 키 + 잠재-flow 정책의 조건 토큰 + 판례 인용 해석가능성 주장** —
  패러다임 신규성은 中, "system+interpretability" 포지셔닝 권고.
- **최저가 kill-gate (CPU-only, 학습 0 — E11 하에서 즉시 실행 가능)**: held-out val에서
  검색된 이웃의 Δz_fut(k-평균)로 실제 Δz_fut를 예측한 R² vs **P-zg3 ridge 상한(0.54)**.
  **Kill**: 검색 R² ≤ ridge(z_t→Δz) R² → 메모리가 선형 상태예보기 이상을 못 준다 = 개념
  사망(비용 CPU 반나절). 2차 게이트: 검색 R²의 이득이 same-task 이웃 제외 시에도 잔존하는지
  (암기-selection 성분 분리).
- **증분 비용**: phase1 재학습 0, 인덱스 = 기존 캐시 재배열, phase2 재학습 1런(+k토큰,
  ctx +k×1.6M), 롤아웃 +matmul 1회. 총 ≤ 0.5 GPU-일 + CPU.
- **Steelman**: *frozen 공간의 진짜 배당은 인코더 재학습 불요가 아니라 "훈련 경험 전체가
  시험 시점에 주소지정 가능"하다는 것 — 판례를 인용하는 최초의 변위-접지 정책.*

## 3. A7 — **LangVerifier**: 언어-일관성 심판자 (자체 제안 #1)

**기전.** 학습 0, 롤아웃-측 전용. 재계획마다 후보 액션청크 K개를 생성(기본 = x0 섭동
K-샘플, 또는 A4의 검색 후보)하고, 각 후보를 frozen 공간에서 채점:
`Score(a) = cos(g(a, z_cur), Δz_goal)`, Δz_goal = C(Δz_text)(A2 보정기) 또는 태스크 문장
델타. argmax 실행 + 히스테리시스(직전 승자 대비 score 우위 δ 미만이면 유지 — mode-switching
차단). 채점 로그 = "왜 이 행동인가"의 해석가능한 결정 궤적. K=1이면 현행과 비트 동형.

- **Thesis-fit 9/10**: 언어가 행동 **결정의 심판**이 되는 최소 기계 — semantic-use를 표현
  수준이 아니라 결정 수준에서 실현. 통화(g-공간)를 채점기로 재사용 = 테제 순환 완성.
- **증거지도 정합**: 학습 무접촉(E1–E3 무관). E8 결정론 존중(선택은 결정론적 argmax;
  h-flow 음성의 원인이던 "궤적 샘플 실행"이 아니라 "후보 평가 후 1개 실행" + 히스테리시스).
  **E4(RED)의 우아한 회피**: g 채점의 상태성분(65–70%)은 고정 z_cur에서 후보 간 **상수** —
  argmax에서 정확히 소거되고 액션-기여 ~30% 성분만이 서열을 결정한다. a2z retrieval
  29.9–37.7%(chance의 ~2,000×, DESIGN_grounding §2.1)가 "g 점수가 액션을 판별한다"의 기존
  실측 방증.
- **신규성**: RoVer [S 2510.10975] = **학습된** process reward model을 VLA test-time
  verifier로(최근접 이웃) — 우리는 **무학습 frozen-공간 언어일관성 점수**라는 점이 차별.
  DINO-WM [V 2411.04983]·V-JEPA-2-AC [V 2506.09985] = frozen 잠재 MPC이나 **언어 비용함수
  부재**. "language terminal cost in frozen VL delta space, training-free" 조합 미발견 —
  신규성 中上.
- **최저가 kill-gate (CPU)**: val 상태에서 GT 액션 vs 셔플 액션 N개를 Score로 랭킹 —
  top-1 정확도가 chance 대비 유의 우위인지 + Δz_goal 스왑(wrong 문장) 시 랭킹이 유의하게
  붕괴하는지(언어 의존성 확인). **Kill**: 랭킹 ≈ chance(액션 판별 실패) 또는 M6 실패
  (보정된 Δz_goal 부재 — A2와 운명 공유).
- **증분 비용**: rollout_sim.py 후보 루프 + 채점 함수 ~100줄; K배 flow 추론(ms 단위,
  병목은 어차피 렌더 — E11). A2 보정기에 무임승차.
- **Steelman**: *selection-vs-use 논쟁을 끝내는 가장 싼 방법은 언어를 심판석에 앉히는 것 —
  correct−wrong이 아니라 "후보별 채점표"가 증거물이 된다.*

## 4. A1 — **ConceptBasis**: 개념-기저 통화 (seed S1)

**기전.** SpLiCE [V 2402.10376]식 텍스트-앵커 사전 `D ∈ R^{1024×K}`(모션/관계 어휘 —
W4v3 모션문장 v3 + SpLiCE 관계 어구 재사용, K≈64–256)에 대한 희소 비음수 계수로 통화를
재정의: `w = SparseCode_D(Δz)`, 정책은 w를 수송, h′는 (Dw 또는 w, z_t)에서 액션 복호.
ζ가 문자 그대로 "개념 가중치"가 되어 언어-이미지-행동 교환이 **구성상(by construction)**
성립. 2단계 도입: **(i) read-out 변형** — 통화 불변, w는 해석 사영(위험 0), **(ii) 통화
변형** — flow_dim=K로 완전 교체.

- **Thesis-fit 10/10**: "공용 통화"의 가장 문자적 실현 — 해석가능성이 사후분석이 아니라
  타입 시스템이 됨.
- **증거지도 정합**: 통화 재정의 클래스 = R-Δ/PCA-k와 같은 축(E2의 "복잡화"가 아니라
  차라리 **단순화**: 2048→K). E5(eff-rank ~5)가 강한 순풍 — h가 읽는 유효방향이 ~5개라면
  K=64 개념 기저가 정보를 담을 여지가 실재. E3(HY03) 비모순: 희소코딩은 재구성 기반,
  대조 반발항 없음. E4 가산점: 모션-델타 어휘 사전은 상태(씬 구성) 성분을 표현할 기저가
  부족 → ζ의 상태 운반(P-zg4a R² 0.90)이 구조적으로 절단될 개연 — 잔차화와 같은 방향.
  **정직한 의존성**: 이 개념의 착수 판단은 **R-Δ 결과에 조건부**가 옳다(R-Δ가 "raw Δz
  통화 가능"을 보이면 개념사영 통화는 그 직계 후속; R-Δ가 "압축·정제 필요"로 나오면
  개념 basis가 바로 그 정제의 해석가능판 후보로 승격 — 어느 쪽이든 살지만 설계가 달라짐).
- **신규성**: SpLiCE/LaBo/DN-CBM/CB-SAE [S 2512.10805] = 분류·해석 전용; **Event-Grounded
  SAE for VLA policies [S 2605.17204] = 최근접 이웃**(VLA 내부의 SAE 사후 분석 — 단
  통화가 아니라 분석 도구로 보임, abstract 재검증 필수). **텍스트-앵커 희소 기저를 제어
  통화 자체로** 쓴 발표물 미발견 — 신규성 上.
- **최저가 kill-gate (CPU, 학습 0 — E9 지렛대)**: frozen h로
  `R²[h(D·w(Δz), z_t)] vs R²[h(Δz, z_t)]`, K∈{32,64,128,256} 스윕 + gripper-dim 별도.
  **Kill**: K≤256에서 R² 하락 > 0.05 → 개념 병목의 정보 손실 과다(통화 변형 사망, read-out
  변형만 생존). 보조: M1(SpLiCE ζ vs Δz Jaccard — 기등록)이 사전 겸용.
- **증분 비용**: 사전 구축 = `src/analysis/splice_concepts.py` 재사용(CPU 시간); read-out
  변형 학습 0; 통화 변형 = phase2 재학습 + h′ 재적합(~3 GPU-h).
- **Steelman**: *eff-rank 5짜리 통화라면, 그 5축이 익명의 PCA 축일 이유가 없다 — 이름 있는
  개념 축이어도 되는지가 이 개념의 전부이고, 그 검정은 오늘 CPU로 된다.*

## 5. A5 — **CycleAlign**: 생성형 왕복 정렬 (seed S5)

**기전.** phase1의 언어 연결을 InfoNCE(HY03, 음성)가 아니라 **alignment-only 왕복**으로:
사영기 `p: ζ → text-space`가 모션문장 임베딩을 **회귀**(MSE+cos, negative 없음)하고
역사영이 Δz를 재구성(cycle). Wang & Isola [V 2005.10242] 분해에 근거한 기전 차이 주장:
InfoNCE = alignment + **uniformity(반발)**이고, HY03의 fused-ζ 손상은 유사 모션 간 반발이
ζ 기하를 왜곡한 것으로 해석 가능 — cycle/회귀는 반발항이 0이라 실패 기전을 공유하지 않는다.
도입 순서: **(0) 사후(post-hoc) 프로브 먼저** — frozen ζ 위에 p만 적합해 텍스트-복호
가능성을 측정(phase1 무접촉). 성공하면 해석 배당은 공짜이고 in-loss 판은 불필요할 수 있음.

- **Thesis-fit 7/10**: ζ가 "문장으로 읽히는" 통화가 되는 경로 — 통화 교환의 역방향
  (행동→언어) 담당.
- **증거지도 정합**: E3에 대한 **의도적 도전**이며 기전 차이(반발항 유무)를 명시 — 단
  in-loss 판은 fused-ζ SR 비열등을 승격 조건으로 강제. E4 주의: 모션문장이 상태와 상관이면
  회귀도 상태 지름길 학습 가능 → 문장-셔플 대조 필수.
- **신규성**: directional CLIP loss 계열 [V 2110.02711]은 생성/편집 도메인; 잠재-액션
  모델에 caption-consistency를 건 발표물 미확인(UNVERIFIED) — 신규성 中.
- **최저가 kill-gate (CPU)**: post-hoc p 적합 → held-out cos(p(ζ), E_text(모션문장)) vs
  문장-셔플 기준선. **Kill**: 셔플 대비 무우위 → ζ의 텍스트-복호 가능성 부재, in-loss 판
  자동 기각(개념 종료, "ζ는 언어로 안 읽힌다"는 정직한 한계 데이터로 회수).
- **증분 비용**: post-hoc = CPU 수 시간; in-loss = phase1 1런(~2 GPU-h).
- **Steelman**: *HY03의 사망 원인을 목적함수 해부 수준에서 특정하고 고친 재도전 — 성공 시
  통화의 왕복(행동↔언어) 양방향이 처음으로 닫힌다.*

## 6. A9 — **MotionWordVQ**: 텍스트-앵커 변위 코드북 (자체 제안 #2)

**기전.** 코드북 엔트리 = **frozen 텍스트 임베딩 그 자체**(모션 단어/구 어휘를 A2 보정기로
이미지-델타 좌표에 사상). ζ/Δz를 top-m 소프트 양자화(+소형 스칼라 잔차), 정책은 코드
로짓을 예측, h는 코드 임베딩(+잔차)에서 복호. 실행 궤적이 "동작 단어의 문장"으로 그대로
출력된다 — 공용 통화 테제의 가장 문자적(이산·상징) 실현. A1의 이산 자매 개념.

- **Thesis-fit 8/10**: 해석가능성 최상(궤적=단어열); 단 이산화가 연속 제어 정보(특히 크기)
  를 깎을 구조적 위험.
- **증거지도 정합**: 통화 재정의 클래스이나 **이산 병목 + 잔차 채널 = 복잡화 냄새**(E2
  경계) — C1/C2의 "잔차 채널 굶음" 전례가 잔차 스칼라에 재현될 위험 명시. E5 순풍(eff-rank
  5면 소형 코드북 가능). E3 비모순(양자화는 대조 아님).
- **신규성**: LAPA [L 2410.11758]·Moto [L 2412.04445]·UniVLA [L 2505.06111] = **학습형**
  VQ(언어 앵커 없음); RT-H [S 2403.01823] = VLM이 언어 모션을 **생성**(frozen-임베딩
  양자화 아님); CLAP [S] = proprio 정렬 코드북. **"코드북 = frozen 텍스트 임베딩" 미발견**
  — 신규성 上, 기전 위험도 上.
- **최저가 kill-gate (CPU, 학습 0 — E9)**: `R²[h(softVQ_m(Δz), z_t)]` (m=1..8, 어휘 코드북)
  vs full. **Kill**: m=8 소프트 혼합에서도 R² 붕괴(>0.10 하락) → 어휘 코드북이 제어 정보를
  못 담음. A1 게이트와 **같은 스크립트로 동시 실행 가능**(사전만 교체).
- **증분 비용**: A1·A2 기계(사전, 보정기) 재사용; 통화판은 phase2+h′ 재학습.
- **Steelman**: *"로봇이 지금 무엇을 하는지"를 로그가 아니라 통화 자체가 영어로 말하는
  정책 — 데모 영상 옆에 단어열 자막이 달리는 아키텍처.*

## 7. A8 — **PrecedentDecoder**: 비모수 판례 디코더 (자체 제안 #3)

**기전.** h를 국소가중회귀(LWR/kNN)로 교체 또는 병치: `a = Σ_i softmax(−‖Δz−Δz_i‖/T)·a_i`
(훈련쌍 캐시, k≈16). 디코드 수준까지 판례 인용이 관통 — "이 행동은 데모 i의 t구간 보간"
이라는 완전 추적 가능 정책. 동시에 과학적 절제: h의 기능이 국소 메모리 보간에 불과한지,
그 이상인지를 분리.

- **Thesis-fit 6/10**: 해석축 강화 + h 이해의 과학 가치; 통화 교환축 기여는 없음.
- **증거지도 정합**: **디코더-측 변경 — E2의 4중 음성 전례가 정면에 있음**을 자인. 단
  기전이 정반대(용량 추가·확률화가 아니라 용량 제거·결정론·비모수) — 도전 사유 성립하되
  승산은 보수적으로. E5(eff-rank 5)와 E9(h의 raw-Δz 수용)가 국소 선형 구조 가설을 지지.
- **신규성**: VINN [UNVERIFIED] = 관측공간 kNN 정책; 변위공간 kNN **디코더**(상류는 flow
  정책 유지)는 미확인 — 신규성 中下, 가치는 주로 해석·절제 과학.
- **최저가 kill-gate (CPU, 학습 0)**: val에서 kNN-h R² vs MLP-h(concat 0.749) + gripper
  acc ≥ 92%. **Kill**: R² < h − 0.05 → 폐루프 진입 금지, "h는 보간 이상을 한다"는 결과로
  회수(그 자체가 논문 한 문단).
- **증분 비용**: 0 학습; 롤아웃 kNN matmul.
- **Steelman**: *h R² 0.75가 22k 판례의 보간으로 재현된다면, 통화 해석가능성 주장이
  인코더-정책-디코더 전 구간에서 닫힌다.*

## 8. A3 — **MultiScaleDisp**: 다중 스케일 변위 위계 (seed S3) — **정직 판정: KILL (독립 셀로는)**

**기전(제안대로)**: coarse 장스팬 Δz(전역 계획) + fine 단스팬 Δz(국소 제어) 2층 + 스케줄러.
**비판**: (i) coarse 모듈이 정책과 같은 입력에서 공학습되면 그 출력 토큰은 조건 정보의
재부호화 — **E7이 Phase-B dual-wrist 널을 설명한 바로 그 기전**이 시간축에서 재현될 것이
기본 예측. (ii) A2-프로브의 span-16>span-4는 "타깃 스팬이라는 **하이퍼파라미터**"의 증거이지
2층 기계의 증거가 아님 — 스팬 스윕은 아키텍처 없이 오프라인 1셀로 소진 가능. (iii) 스케줄러
+2헤드 = 팔당 변경 다수(E2 복잡성 프라이어 정면 위반). **구제 경로**: coarse 변위가 **신규
정보원**에서 올 때만 살아난다 — 언어에서 오면 = A2, 검색에서 오면 = A4. 즉 이 개념의 유효
성분은 이미 A2/A4에 흡수된다. **잔여 액션**: span 스윕(오프라인, phase2 타깃 스팬 {8,16,32}
val 비교)만 P-A 편승 셀로 등재. 만약 그래도 2층을 시험한다면 kill-gate: teacher-forced
coarse 토큰(오라클) vs 자기생성 coarse 토큰의 val act-R² 격차 — 자기생성이 오라클 이득을
전부 반납하면(예측) 즉사.

## 9. A10 — **TriModalHub**: 3-모달 변위 허브 (자체 제안 #4) — **정직 판정: PARK (보류)**

**기전**: 어댑터 3개(E_a: 액션, E_t: Δz_text, E_i: Δz_img)가 공유 마이크로 통화 u(dim≈32,
eff-rank 근거)로 수렴, 디코더가 교차 복원(9칸 cross-decode 행렬 학습). 완전한 교환 행렬 =
테제의 최대주의 실현. **비판**: (i) u는 **학습 공간** — "frozen 공간의 변위 = 통화"라는
테제 문장을 스스로 폐기하는 방향(테제-fit이 높아 보이지만 실은 테제 이탈: 통화가 frozen이
아니게 됨). (ii) E2 복잡성 프라이어 정면(어댑터 6개 동시 학습). (iii) 싼 전초기지가 이미
큐에 있음 — PCA-k=32(§WEEK0 권고 ②)와 R-Δ가 "소차원 통화 가능성"과 "frozen 충실도"를 먼저
판정한다. **재개 조건**: PCA-k 성공 AND R-Δ가 "압축 필요" 방향 AND A2의 보정기 ②(선형)가
부족으로 판명 — 세 조건 동시 성립 시에만 허브가 정당화됨. kill-gate(그때): cross-decode
행렬 9칸 중 text→action 칸의 오프라인 R²가 T-0 경로 대비 우위인가.

## 10. A6 — **DualViewFlow**: per-view 이중 flow (seed S6, 사용자 제안) — 조건부 슬롯 유지

재개발하지 않음(지시대로). 정의: ζ_main/ζ_wrist 2헤드 flow, shared-τ 공동 수송 vs 독립
수송 비교. **발동 조건 = DESIGN_wrist §6-F1**: W-C(표준화 dual 재판)가 paired SIG 승리하여
"타깃-측 wrist 무익" 명제가 반증될 때만. 그 전까지 셀 없음. kill-gate(발동 시): G2-C
zero-ablation ≥0.27 + 공동 수송이 독립 수송 대비 val act-R² 우위.

---

## 11. 서열화 포트폴리오 테이블

*채점: fit = 테제 실현도(/10) · 정합 = 음성지도/적신호 충돌 여부 · 신규 = 문헌 대비 ·
gate$ = kill-gate 비용 (CPU = GPU 불가 기간(E11)에도 착수 가능).*

| 순위 | ID | 개념 | fit | 정합 | 신규 | gate$ | kill 기준(요약) | 판정 |
|---|---|---|---|---|---|---|---|---|
| 1 | **A2** | LangSubgoalDisp | 10 | 충돌 0 (조건화-측, 언어 유래) | 中上 [S 2606.20627 재검증 요] | ~0 (T-0/M6 기등록) | M6 대각 우세 무 + 사다리②까지 방향 cos ≤ 0 | **풀 스펙 권고 ①** |
| 2 | **A4** | RetrievalCond | 7 | 충돌 0 (E7 통과, 단 E6 언어 희석 감시) | 中 (분야 밀집) | CPU 반나절 | 검색 R² ≤ ridge 상한 0.54 | **풀 스펙 권고 ②** |
| 3 | **A7** | LangVerifier | 9 | 충돌 0 (학습 무접촉, E4 상수 소거) | 中上 (무학습 verifier 미발견) | CPU (랭킹 프로브) | GT-랭킹 ≈ chance 또는 M6 실패 | A2 부속 라이더로 동시 착수 |
| 4 | **A1** | ConceptBasis | 10 | 충돌 0 (통화 재정의 클래스) | 上 | CPU (frozen-h 디코드) | K≤256에서 R² −0.05 초과 하락 | **R-Δ 결과 조건부** — gate는 지금, 통화판은 R-Δ 후 |
| 5 | A5 | CycleAlign | 7 | E3 도전(기전 차이 명시) | 中 | CPU (post-hoc p) | 문장-셔플 대비 무우위 | post-hoc 프로브만 선실행 |
| 6 | A9 | MotionWordVQ | 8 | E2 경계(잔차 채널 위험) | 上 | CPU (A1과 동일 스크립트) | soft-m=8에서 R² −0.10 초과 | A1 gate에 편승, 통화판은 A1 뒤 |
| 7 | A8 | PrecedentDecoder | 6 | E2 도전(용량 제거 방향) | 中下 | CPU (kNN swap) | R² < h−0.05 | 과학 절제로 1회 실행 가치 |
| 8 | A6 | DualViewFlow | 5 | E1 예외 조항 대기 | — | — | W-C 승리 시에만 발동 | 조건부 슬롯(F1) 유지 |
| 9 | A10 | TriModalHub | 5(테제 이탈 위험) | E2 정면 | 中 | — | PCA-k·R-Δ·보정기② 3조건 후 | **PARK** |
| 10 | A3 | MultiScaleDisp | 4 | E7 정면 위반 예측 | 下 | — | (span 스윕만 P-A 편승) | **KILL** (A2/A4에 흡수) |

**CPU-gate 일괄 배치 제안**: A4·A7·A1·A9·A5·A8의 kill-gate는 전부 CPU/캐시-only —
GPU 불능 기간(E11)에 **한 배치(WEEK0 후속과 동일 형식)로 6개 게이트를 동시 소진** 가능.
이 배치가 끝나면 포트폴리오의 생존자가 데이터로 확정된다.

## 12. 권고 — 풀 스펙 대상 2건과 근거

**① A2 LangSubgoalDisp** (+ A7 라이더). 근거: (a) kill-gate가 이미 큐에 있는 T-0/M6
그 자체라 **추가 게이트 비용 0** — 통과 시 스펙만 있으면 즉시 아키텍처로 승격, 실패 시
migrate 트리거 입력으로 전량 회수(어느 쪽도 낭비 없음). (b) 유일 양성 삽입점(조건화) +
언어 유래 토큰이라 E6 tradeoff의 반대 방향 — 현 스택에서 언어축을 **올릴** 수 있는 유일한
구조 후보. (c) PI 불만(selection 느낌)에 대한 결정 실험(H_select 기각)을 내장. (d) A7이
같은 보정기 위에서 학습 0으로 동승 — 1 스펙에 2 개념.

**② A4 RetrievalCond**. 근거: (a) 진행/대기 중인 큐 전체(P-A, wrist 3-arm,
innovation-grounding, R-Δ, P-B)와 **완전 직교** — phase1 무접촉, 어느 승자 기질 위에도
얹힘. (b) kill-gate가 CPU-only라 GPU 불능인 지금 즉시 착수 가능하고, ridge-상한(0.54)이라는
기성 기준선 덕에 판정이 자동. (c) E7(신규 정보 조건화)을 통과하는 드문 후보 — Phase-B
실패 기전을 정면으로 회피하는 설계라는 서사 가치. (d) 해석가능성(판례 인용)이 테제의
"interpretable policy" 절반을 채운다.

*차순위 명시: A1은 fit 만점이나 통화-재정의 착수 판단이 R-Δ(큐 3순위, ≤1.5 GPU-일)의
결과에 걸려 있어 "풀 스펙"이 아니라 "CPU gate 선실행 + R-Δ 후 분기"가 옳다 — R-Δ보다
먼저 스펙을 굳히면 절반은 다시 쓰게 된다.*

---

## 부록 — 본 세션 신규 확인 문헌 (전부 [S] = 스니펫 수준, 채택 전 abstract 대조 필수)

- 2605.17204 Event-Grounded Sparse Autoencoders for VLA Policies (A1 최근접)
- 2512.10805 Concept Bottleneck Sparse Autoencoders (CVPR 2026)
- 2412.15182 STRAP: Sub-Trajectory Retrieval for Augmented Policy Learning
- 2606.15631 Retrieve, Don't Retrain: VLA test-time retrieval
- 2408.15919 ReMoBot: Retrieval-based few-shot IL
- 2603.02688 Retrieve-Reason-Act / 2603.29419 RAAP (검색증강 주변부)
- 2606.20627 Latent Goal Prediction from Language for Model-Based Planning (A2 최근접)
- 2204.11134 Zero-shot task specification (CLIP 유사도 목표)
- 2510.10975 RoVer: Reward model as test-time verifier for VLA (A7 최근접)
- 2403.01823 RT-H: Action Hierarchies Using Language (A9 인접)
- 2606.18955 Motion-Focused Latent Action / CLAM / 2409.18707 Discrete Policy (A9 주변부)

기존 검증 승계 [V]/[L]: SpLiCE 2402.10376, PC-CLIP 2409.09721, SuSIE 2310.10639,
DINO-WM 2411.04983, V-JEPA-2-AC 2506.09985, LAPA 2410.11758, Moto 2412.04445,
UniVLA 2505.06111, Wang&Isola 2005.10242, directional CLIP loss 2110.02711 등
(`docs/DESIGN_grounding_space_v1.md`·`docs/ANALYSIS_clip_language_limits_v1.md` 검증 목록).

*본 문서는 어떤 기존 셀·게이트·재개금지 목록도 변경하지 않는다. 모든 신규 gate는
사전등록 후보이며, 실행 전 PI 승인 대상.*
