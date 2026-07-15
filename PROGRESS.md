# PROGRESS — CLIP Embedding Action (F-시리즈)

CLIP 잠재공간 기반 액션 표현 연구의 개발 로그. 각 항목 = **무엇을 / 어떻게 / 결과**.
설계도: `DESIGN_fusion_dense_latent_action_v1.md`, 착수 지시: `KICKOFF.md`.
> 내부 규율 문서(예측 장부·검증 로그·NUMBER_CARD)는 비공개(gitignore). 본 파일은 팔로업용 공개 로그.

---

## 🔄 현재 진행 중
- (완료) **방향1 언어사용 굳히기**: 1a 3시드(correct−wrong +75.8pp)·1d-goal(+88.5pp)·1b(LIBERO-Para)·1c(씬내 타깃-스왑) — 아래 로그.
- (완료) **F3 dense-obs = 음성 종결**(a 50.0 > b 31.5 > c 15.5, 게이트 실패) → "richer obs 조건화가 폐루프 악화". F6 불요.
- (완료) **anchor 폐루프 경로 일반화**(commit 7ad1950) → 비-CLIP 앵커(SigLIP2/RADIO) 폐루프 head-to-head(F1) 언블록.

---

## 로그 (최신 순)

### 2026-07-15 · C2 게이트 UNSAFE·HOLD — 대조팔 회귀 (cowork AUDIT_ORDER)
- **무엇을**: 사용자 우려(융합 음성이 당황스럽다) → C2 구현 점검 → cowork가 결정적 이상 포착: **C2 대조팔(pooled+aug) correct 80.5%가 동일 substrate no-aug 88.5%보다 8pp↓** = 증강이 +레버인데 대조가 자기 기준선 밑 = **신규 레짐(조립/aug/예산/eval) 회귀**로 C2 판정 오염.
- **판정**: **C2 게이트 = UNSAFE(보류).** "타깃코드 무이득·ζ_f 유해·α 0.0015" 결론을 논문/HEADLINE에서 HOLD 플래그로 전환(공유 버그가 fine 채널에 비대칭 피해 가능). **단 SigLIP2-fine(C1/noconsist)은 구 레짐·검증 대조(88.5)라 음성 유효.** 언어보존 축 무관.
- **어떻게**: 감사 착수(cowork §2, verify-first) — #6 재기준선(no-aug 88.5 ckpt vs c2_control 80.5 동일 harness 재롤아웃 → 회귀 실재/노이즈 판별) + per-task 대조 + aug-뱅크 검증 + 학습예산. 감사 후 대조 87-90 복귀 시 C2 재실행.
- **함의**: 콜리그 avg 94.8 판에서 우리 대조조차 80.5 = 아이디어 실패보다 **측정 실패 우선**(과거 DINOv2 65.2→공정 79.2 교훈 재적용). CC 병행 발견: C2는 관측-레벨 융합(외부 94.8)이 아닌 gated 타깃 액션코드 = 다른 종류; 게이트 α=β=0 cold-start도 요인 후보. docs/C2_regression_audit(진행).
- **부수**: HEADLINE/NUMBER_CARD/NOVELTY C2 행 HOLD 플래그. R4는 감사 종결 후. 언어보존 논문 초안 병행 계속.

### 2026-07-15 · C2 게이트 음성 확정 — SigLIP2+DINOv3 융합 fine 채널 무이득·유해
- **무엇을**: PI 융합 질문 — SigLIP2+DINOv3 cross-attention 융합(규제 포함 제대로 구현)의 폐루프 실성능? fast-track §3 게이트.
- **어떻게**: C2-full(ζ_g SigLIP2-large256 pooled ⊕ ζ_f DINOv3-L/16@256 fine, w_consist=0, 규제스택 전체) vs control(pooled+aug). 5세트 롤아웃(correct/wrong/ζ_f-절제 + 대조).
- **결과**: **음성 확정.** C2-full correct **78.3 = control 78.3(정확 동일)**, paired +0.00 CI[−17.3,+17.3](유의차 없음). **ζ_f 절제 +6.7pp 향상 = fine 채널 유해.** **게이트 α 0.0015**(C1 0.027/noconsist 0.0056 — 3기질 중 최소, 미개방). correct−wrong +76.1(언어보존 유지). → **위치정보 최강 substrate(DINOv3@256 topk8 0.343)조차 fine 채널 못 살림** = **fine 채널 3기질 종결**(SigLIP2 C1/noconsist + DINOv3 C2).
- **함의**: **PI 융합 답 = "융합 fine 채널은 폐루프 무이득·유해, 값은 언어보존."** 시각 풍부화 음성축 완결(관측 F3·앵커 F1·타깃코드 C1+C2 substrate 일반화). 헤드라인=언어보존(2 suite×2 backbone +75.8~92pp) 유지. R4(wrist) 계승. docs/C2_gate_rollout + cowork 회신. HEADLINE/NOVELTY/NUMBER_CARD 갱신.

### 2026-07-14 · C2-full 학습 완료 — DINOv3-fine 게이트 미개방 (α 0.0015)
- **무엇을**: SigLIP2+DINOv3 융합(C2-full) 학습 완료 + 게이트 α/β 추출(cowork §2.3 1급 기전).
- **결과**: C2-full val 0.3817 / **R² +0.665**(C1 +0.663·control과 동급) / gripper 93.3%. **f4 게이트 α = 0.0015(tanh 0.00147)** — C1 0.027·noconsist 0.0056보다 **더 닫힘(3개 중 최소)**. → **위치정보 최강 substrate(DINOv3@256 topk8 0.343)조차 fine 채널 게이트를 못 엶.** 오프라인도 무이득. = cowork 사전선언 "α~0 → fine 채널 3기질 음성(SigLIP2-fine C1/noconsist·DINOv3-fine C2)→R4" 방향. PI 질문("융합 실성능") 답 trending: **fine 채널 substrate 무관 미사용(task 미보상).** 
- **다음**: fast-track §3 폐루프 롤아웃(5세트: C2-full correct/wrong/ablate-zf + control correct/wrong)으로 SR 확정 — paired CI(vs control)·correct−wrong·ζ_f 절제. α가 이미 음성 예측(C2-full≈control 기대). 양성 시 3시드+goal 승급, 음성 시 정직 종결+R4. docs/C2_gate_rollout(진행).

### 2026-07-14 · C2-full GO + 2팔 학습 착수 (SigLIP2+DINOv3 융합)
- **무엇을**: cowork DECISION_C2_GO(§2① 자기정정 — SigLIP2-fine 게이트 미개방으로 DINOv3-fine 보류는 과대외삽; DINOv3 미검증이 PI 질문). C2-full 착수.
- **어떻게**: ζ_g=SigLIP2-large256 pooled ⊕ ζ_f=**DINOv3-L/16@256 patch ΔF**(K=8 텍스트쿼리 cross-attn, 96d 병목, tanh-gate, w_consistency=0, D3-A flow), 규제 스택 전체(both-aug 3v + dropout0.2 + noise). Dinov3Anchor 신규(fp32/registers 드롭/no-crop). 대조=pooled+aug(f4 off). 2팔 병렬(C2-full GPU9 + control GPU8).
- **해상도 결정(§2.2, model-usage)**: **DINOv3@256 R_topk8=0.343 > @512 dense-optimum 0.319 > SigLIP2 0.157** → @256 그리드-정합에서 DINOv3 위치정보 강점 유지(오히려 상회) → @256 native 채택(폴백 불요). = "각 모델 최적 활용" 데이터 확인.
- **결과**: 착수 성공(dim-assert OK, 크래시 없음, 어셈블리 진행). 판정 예정(fast-track §3): paired CI(C2-full vs control)·correct−wrong≥+70pp·ζ_f 절제·**게이트 α/β**(DINOv3-fine이 여는지=1급 기전: 열면 "위치정보가 열쇠", 안 열면 fine-채널 3기질 음성 완결). 기준선 avg-fusion ~94.8(UNVERIFIED-external). 외부부하로 wall-clock 지연(타이밍). docs/C2_prestage.

### 2026-07-14 · W4v3 정렬-입증 그림 3종 + 논문 골격 v1
- **무엇을**: cowork W4v3 스펙(정렬 입증 오프라인 분석) P1-P3 + SCOPE 감량(F1/paramfree/A3/T4/F7 DROP) 반영 + 논문 초안 골격.
- **결과**: **P1 MaskCLIP 히트맵**(큰물체 착지✓/fine bowl diffuse — 정성 보조, A3 흡수) · **P2 SpLiCE**(Δz=grasp 이벤트 해석, 재구성 저=domain gap → "Δz가 gap offset 상쇄 작동" 통찰) · **P3 GW/ANOVA**(★언어보존 분산 ~97% suite/backbone무관, arm간 SR차이 ~80% 시드노이즈 = 음성축 정량지지). PAPER_SKELETON_v1 작성(양성 2×2 + 3중음성 + W4v3 + 차별선). C2-full은 cowork §2①(게이트 미개방) 재판정 대기. docs/W4v3_P{1,2,3}·PAPER_SKELETON_v1.

### 2026-07-14 · C1 장 종결 — F4 구조절제 판정 (a) 진짜 3번째 음성
- **무엇을**: cowork F4_revival_rule 4조건으로 F4 구조절제(w_consistency=0)가 fine 채널을 살리는지 판정.
- **어떻게**: phase2_libero_c1_noconsist(w_consist=0) 학습+롤아웃(correct/wrong/ζ_f-절제) + 게이트 α/β ckpt 실측.
- **결과**: **(a) 확정** — 4조건 중 ①(paired +3.5pp CI[−0.43,+7.43] 0포함)②(ζ_f 절제 −0.8pp 무시)④(**게이트 α 0.027→0.0056 미개방**) 미달, ③(correct−wrong +78.5)만 통과(비차별). **L_consistency 제거해도 게이트 미개방 = 폐루프 태스크가 fine patch 채널 미보상**(설계결함보다 깊은 구조적 음성). → **"시각 풍부화 3중 음성(F3 관측·F1 앵커·C1 타깃코드)" 기전 확증.** ⚠ SigLIP2-fine 결과 — DINOv3-fine(C2 fast-track) 미검증, cowork 재판정 대기(precondition_conflict). docs/C1_noconsist_ablation_rollout.

### 2026-07-14 · 언어보존 3시드 + C1 진단 + 논문 조립
- **무엇을**: C1 NO-GO 후 cowork 판정(a 헤드라인굳히기/b C2 조건부/c F4 확장강등) 반영 — (a) 3시드 + ζ_f 실패진단 3종 + 논문 헤드라인 조립.
- **어떻게**: pooled-only large256 seed{1,3} 학습+롤아웃(seed2와 3시드). 진단: cos/CKA(ζ_f,ζ_g)·attribution ∂â/∂ζ_f·실패모드. 조립: HEADLINE_SUMMARY·NOVELTY v2·NUMBER_CARD.
- **결과**: **언어보존 3시드 견고** — SigLIP2-large256 correct−wrong **+76.5±1.8pp**(s1/s2/s3 +75.5/79.0/75.0, 밴드 75-79 전부 ≥+70; CLIP 3시드 +75.8과 수렴 = backbone-agnostic). **C1 진단**: ζ_f 실패 = **구조적**(L_consistency가 ζ_f를 coarse Δp로 붕괴, CKA 0.871; 게이트 미개방 α 0.027; 정책 무시 Jacobian 0.069) → **C2 무용(기질무관)**. **서사 확정**: "언어보존 접지가 하중 지탱, 시각 풍부화는 F3(관측)·F1(앵커)·C1(코드) 전 삽입점서 실패(기전 있음)". F4→확장 장, 구조절제+R4(wrist)=future. docs/{HEADLINE_SUMMARY,NUMBER_CARD,headline_3seed,C1_zetaf_failure_diagnostics}.

### 2026-07-14 · C1 폐루프 게이트 — F4 fine 채널 NO-GO (1시드)
- **무엇을**: C1(F4 ζ_g⊕ζ_f fine) vs SigLIP2-large-256 pooled-only 폐루프 비교 + ζ_f 절제(병목-효능). libero_spatial 10×20롤.
- **어떻게**: 3-arm 롤아웃(C1 correct/wrong, C1 --ablate-zf, pooled correct/wrong) → paired 비교 + correct−wrong + full-vs-ablate. 게이트 도구 `src/analysis/{paired_ci,lang_retention}.py`, `rollout_sim.py --ablate-zf`.
- **결과**: **fine 채널 폐루프 이득 없음.** C1 correct **90.0** vs pooled 88.5 (**+1.5pp**, per-task SD 8.08 → CI 0 포함=유의 안 함). ζ_f 절제 **91.0 > full 90.0**(ζ_f **−1.0pp**, 끄면 근소 향상=병목-효능 없음). correct−wrong C1 +78.5 / pooled +79.0(언어보존 공통·유지). 오프라인은 C1 근소 우위(val 0.3866/R²+0.663 vs 0.3887/+0.655) → **폐루프로 미전환**.
- **판정/함의**: "값은 code construction(F4)"이 폐루프 미지지. **견고한 것은 언어보존**(전 arm +78~82pp). 3중 정합: offline≠closed-loop · richer-info 폐루프 무익 · 값은 언어보존 기질(F3·E-series와 동일). caveat: 1시드(엄밀엔 3시드), single-suite. f4 조립 pathological(~45-58GB 단일스레드, ~6h/arm). A2 span-16 유지(moot). 상세 `docs/C1_gate_report_2026-07-14.md`.

### 2026-07-13 · C1(F4 2-채널) substrate 착수 + 폐루프 게이트 인프라
- **무엇을**: F4 학습형 latent action(ζ_g 풀드 언어정렬 ⊕ ζ_f patch 미세채널)의 C1 빌드 시작 + 게이트 판정 도구 구현.
- **어떻게**: `src/models/f4.py`(F4FineChannel: text-query K=8 cross-attn, 96d 연속 병목, tanh-gate α=β=0 초기 무효과,
  frozen 디코더+게이트드 잔차+L_consistency), `configs/phase1_libero_siglip2_large256.yaml`(SigLIP2-large-256, latent 1024) +
  `phase2_libero_c1.yaml`. 게이트 3종: `src/analysis/paired_ci.py`(C1 vs pooled-only paired bootstrap CI),
  `lang_retention.py`(correct−wrong≥+70pp), `rollout_sim.py --ablate-zf`(ζ_f 제로화=병목-효능 프로브).
- **결과**: f4 α=β=0 초기 **무효과 검증**, `--ablate-zf` **no-flag 시 bit-identical** + f4=None 경로 무영향,
  게이트 self-check **17/17 PASS**. phase1 large-256 학습 중(GPU9, latent 1024). 원격 동기화 대상 3파일
  (f4.py 원격 부재 발견 → 동기화 목록 교정). D3-A 이탈(별도 게이트드 ζ_f flow 분기·공유 τ) cowork 승인.

### 2026-07-13 · 설계 대안 A1–A4 사전등록 + A2·F1 오프라인 프로브
- **무엇을**: cowork 4대안(A1 무학습 크기게이트/A2 2-시간축 ζ_f/A3 Δ-attention/A4 no-mix concat) 사전등록 + 오프라인 검증.
- **어떻게**: A2=ζ_f 변위 span 16→4 fine R² 비교, A1=‖ΔF‖ top-M patch 선별(0-파라미터=지름길 구조적 제거),
  F1=텍스트기하(1−cos)↔per-task SR-하락 상관(CLIP·SigLIP2 towers, n=10/pooled n=20).
- **결과**: A2 — dense 캐시가 span-16 pre-differenced 저장이라 patch 헤드라인은 GPU 인코딩 1패스 필요(**이연**);
  offline pooled 프록시는 span 길수록 fine R² 단조 증가(gripper span16 +0.700>span4 +0.165) → **span-16 유지(잠정)**.
  F1 — geometry→behavior 다리 **축-대조 수준 확증**(pooled SigLIP2 r_s=+0.607 p=0.005, SigLIP2>CLIP 사전등록대로)
  but fine within-axis per-task gradient 불성립(SigLIP2 object 셀 반대부호, n=10 저검정력 — 정직 표기). A1 진행 중.

### 2026-07-10 · 1c: 씬내 타깃-스왑 프로브 (PARTIAL 구성적 접지 · 최종)
- **무엇을**: 지시문을 **같은 씬에 실재하는 다른 태스크의 타깃**(bowl_2)으로 스왑 — cross-task wrong보다 인과적
  (새 지시 타깃이 씬에 실제 존재하므로, 새 타깃으로 가면 구성적 접지 / 그냥 실패면 태스크선택자).
- **어떻게**: SWAP 맵 + 3-way **instructed(=Faithful, 새 타깃 bowl_2)/orig(=Biased, 원래 학습 타깃 bowl_1)/neither
  (둘 다 실패, 대부분 timeout)** 검출기(env `_eval_predicate 'on'`), LIBERO-spatial, 20롤/task·10태스크.
  판별 평가(correct/wrong/swap) 자체는 **우리 발명이 아님** — 동시대 벤치 **LIBERO-CF(2602.17659)·ICBench(2603.06001)**가
  "language-ignoring" VLA 실패 모드를 확립(기존 VLA `correct−wrong` ≤ **+19.4pp**, swap에서 Biased 45–78%).
- **결과 (최종)**: **instructed(Faithful) 48.0% / orig(Biased) 2.5% / neither 49.5%.** per-task instructed:
  t5 **100** · t9 **90** · t8 **75** · t1·t2 **70** · t7 **60** 강 vs t4·t6 **0**(어려운 공간 재참조). **판정 =
  PARTIAL·위치의존 구성적(COMPOSITIONAL) 접지, 태스크선택자 아님** — 정책이 스왑 지시를 따라 새 타깃으로 가고
  (Faithful 48%) 학습 타깃 회귀는 거의 없음(**Biased 2.5%**); 실패는 "neither"(미완)이지 학습-타깃 편향 아님
  (태스크선택자라면 Biased가 높아야 하는데 정반대). **LIBERO-CF 리프레임**: 기존 VLA OFT/π0/π0.5 Biased
  78.6/45.0/60.9% ≫ **우리 2.5%** → 암기 궤적 재생이 아니라 (스왑)지시 실제 추종 → "eval 발명"이 아니라
  "frozen-Δz가 VLA 실패하는 구성적-접지 stress test 통과". caveat: 태스크완료 술어 기준·neither 대부분 timeout,
  Faithful 48% < correct-SR ~80%는 예상. (완전 CF-Spatial bddl 이식 1c-ii는 가능한 후속.)
  스왑 실행가능성 감사 통과(10/10 feasible, 아티팩트 반증 → 판정 확정), 단 일부 "neither"(t3/t4/t6=0%)는
  파지난이도 교락(재지향-후-미실행)이지 접지 실패로 입증된 것은 아니라는 정직한 caveat를 함께 기록.

### 2026-07-10 · anchor 일반화: 폐루프 경로에 get_anchor(cfg) (commit 7ad1950)
- **무엇을**: 폐루프 앵커 head-to-head(F1)를 언블록 — 비-CLIP 앵커(SigLIP2/RADIO)를 **전체 폐루프 파이프라인**에 흐르게.
- **어떻게**: phase2 학습 + sim/dataset 롤아웃이 `ClipWrapper()` 하드코딩 → **`get_anchor(cfg)`**로 교체(이전엔
  phase1+obs-fusion만 앵커 일반화, 폐루프는 CLIP 고정). 추가 수정: **지시문(언어) 임베딩 캐시를 앵커 `cache_key`로 키잉**
  (CLIP 768d vs SigLIP2 1152d가 아니면 충돌). **CLIP 기본 경로 비트동형 보존.**
- **결과**: CLIP 기본 비트동형 유지, 비-CLIP 앵커가 이제 폐루프까지 흐름 → **F1(RADIO/SigLIP2 substrate) 폐루프
  head-to-head를 판별 하네스와 함께 실행 가능**하게 언블록. 다음: 원격 여유 시 2b 언어축 head-to-head.

### 2026-07-10 · 1b: LIBERO-Para 객체 vs 동작 어휘 (부분적 의미 이해)
- **무엇을**: 언어 사용이 표면형(문자열 암기) vs 의미(패러프레이즈 강건)인지 분해.
- **어떻게**: LIBERO-Goal 정책, LIBERO-Para(HF, MIT) 패러프레이즈 객체-어휘/동작-어휘, 20롤/task. baseline goal correct 88.5%.
- **결과**: object-어휘 **51.0%**(−37.5pp) / action-어휘 **61.0%**(−27.5pp), Δ차 10pp(사전등록 표면형 임계 15pp **미만**). **판정 = 부분적 의미 이해**: 패러프레이즈에서 51-61% 유지(wrong=0%와 대비 → **의미 접지, 순수 암기 아님**), 단 **객체 명사가 최약축**(표면형 잔여, LIBERO-Para VLA 범위 내). correct 88.5 → paraphrase 51-61 → wrong 0 그라디언트가 (부분적) 진짜 접지 지지.

### 2026-07-10 · 1d-goal: 언어사용 일반성 (goal correct−wrong +88.5pp)
- **무엇을**: 언어사용(1a)이 goal suite에도 일반화되는가.
- **어떻게**: 신규 LIBERO-Goal 정책(phase1-goal dec 0.754 / phase2-goal R² 0.768), goal correct/wrong/blank 폐루프(20롤/task).
- **결과**: goal correct **88.5%** / wrong **0.0%** / blank **0.0%** → **correct−wrong +88.5pp** (spatial +75.8pp보다 강함). goal은 초기 씬 고정·언어가 유일 태스크 단서라 의존 더 큼 → 틀린/빈 지시문에 완전 전멸(0%). **언어사용 일반성 확정**(사전등록 sub "spatial 최대 델타"는 반증 — goal이 더 큼; 핵심 예측은 오히려 강하게 확정). 다음: 1b(객체 vs 동작 어휘) 실행 중.

### 2026-07-10 · 1a: 언어사용 3시드 확정 (correct−wrong +75.8pp)
- **무엇을**: −74.5pp(정책이 언어를 실제 사용)를 3시드로 확정 (헤드라인).
- **어떻게**: full-data phase2 s0/s1/s2(동결 full phase1 공유), correct/wrong 폐루프(20롤/task, osmesa, GPU 8/9 샤드).
- **결과**: correct−wrong = **s0 +76.5 / s1 +76.5 / s2 +74.5pp → 평균 +75.8pp**, 3시드 전부 CI로 0 분리(사전등록 적중 ✅). correct 82.7% / wrong 6.8%(3시드 평균). **언어보존·사용 헤드라인 확정.** blank는 OOD 보조. 다음: 1c(씬내 타깃스왑) + LIBERO-Goal 정책으로 1b(LIBERO-Para 객체/동작)+1d(goal 일반성).

### 2026-07-10 · F1 2a: RadioAnchor 구현 (Direction 2, 병행)
- **무엇을**: 언어정렬 richer 앵커 후보 RADIO — F3로 관측축 닫혀 이제 **타깃/앵커 축**(이기는 축).
- **어떻게**: `anchor.py`에 RadioAnchor = C-RADIOv4-SO400M(torchhub) + **`siglip2-g` 어댑터**(스펙의 `siglip`은 미존재 → 자료조사로 정정) summary=언어정렬 임베딩 + SigLIP2-g 텍스트 타워. dim 1536, res 512 whole-frame(no-crop), dense 불요(§F3 닫힘). 서브에이전트 구현+로컬 테스트, 실행자 검증.
- **결과**: dim 1536, 이미지·텍스트 동일 공간(cos 정상), get_anchor 등록, 기존 앵커 불변, deps(timm/open_clip) 설치. 다음(원격 여유 시): **2b 언어축 head-to-head**(CLIP vs SigLIP2 vs RADIO) — 1차 산출은 SR 아닌 언어축(§0.5).

### 2026-07-10 · F3 폐루프 판정: dense-obs **음성** (게이트 실패)
- **무엇을**: dense obs 융합이 폐루프 SR을 개선하는가 — 진짜 dense go/no-go.
- **어떻게**: 동일 120ep 공정 subset·동결 full phase1(dec 0.682) 공유, 3 arm 폐루프(20롤/task, osmesa): a(no-obs)/b(mean-patch)/c(DINOv2-reg attnpool).
- **결과**: **a 50.0% > b 31.5%(−18.5pp) > c 15.5%(−34.5pp)**. dense-obs가 폐루프를 해침 — 표현력 클수록 더(단조). 게이트(c가 a·b 양쪽 이겨야) **실패**. 단일 mean 토큰(b)조차 −18.5pp → dense **정보 자체가 해로운 shortcut**(토큰수/ctx 아티팩트 아님). 오프라인(a>b>c)과 방향 일치, 폐루프가 harm 증폭. **"풍부한 정보 추가→폐루프 악화" 패턴 3번째 확증**(proprio −28 · DINOv2앵커 −21.8 · dense-obs). 서사 변경 → cowork escalate, 다음 방향 PI/cowork 판단.

### 2026-07-09 · F3 통합: obs 융합 → phase2 (Task 2+4)
- **무엇을**: dense obs 토큰을 정책(phase2) 학습에 통합.
- **어떻게**: `libero.build_policy_samples`가 obs 앵커 dense patch를 subset materialize(캐시키 분리), `train_phase2`가 `module.obs` 게이트 하에 ObsFusion 빌드 + obs 토큰 K개 토큰열 뒤 append + 옵티마이저/체크포인트 통합. 신규 config `phase2_libero_obsc.yaml`(arm c). *(정밀 스펙+하드 게이트로 서브에이전트 구현, 실행자 독립 검증.)*
- **결과**: **게이트 통과** — (A) no-obs `--smoke` val_parts가 불변값과 **완전 일치**(비트동형, 독립 재검증), (B) obs arm(DINOv2-reg) 빌드+학습+저장 OK(124.16M→133.60M, n_tokens 5→13). 설계결정: full dense(~24GB) 비현실 → F3 초기는 동일 subset 공정비교, full은 lazy-loading 후속. 다음: rollout 통합 → arm b/e config → 원격 subset 폐루프 비교.

### 2026-07-09 · F3 착수: dense obs 융합 기반 (앵커 + ObsFusion)
- **무엇을**: F3(진짜 dense go/no-go) 구현 착수 — 계획(`docs/F3_PLAN.md`) 후 독립 기반 2개.
- **어떻게**: (1) `src/core/anchor.py`에 DINOv2-registers 변형(레지스터/CLS 제거→patch-only tokens, `-reg` 캐시키; 기존 앵커·기본 `dinov2-large` 경로 불변), (2) `src/models/obs_fusion.py` 신규 — 인코더별 patch→공통차원 사영→K=8 학습쿼리 cross-attention→K개 obs 토큰(768d), mean/pixel-unshuffle 지원. 둘 다 서브에이전트 구현+단위테스트, 실행자 diff 검토.
- **결과**: DINOv2-reg tokens `(N,256,1024)` 검증(261→[:,5:]로 patch만) / ObsFusion 4종 shape 테스트 통과. **no-obs 기본 경로 불변**(비트동형 보존). 다음: phase2/rollout 통합(no-obs 비트동형 게이트) + config(a/b/c/e) + P6/P7 포팅 → 원격 arm×3시드 폐루프.

### 2026-07-09 · 폐루프 결과: 포트 재현 검증 + 언어사용 판별 (§0.6 R2)
- **무엇을**: clean 리팩터 코드의 폐루프 재현 검증 + "정책이 언어를 실제 쓰는가" 판별평가.
- **어떻게**: 원격 GPU(osmesa), libero_spatial 20롤/task·1시드. correct(정상)/wrong(다른 태스크 지시문)/blank(빈 문자열) 3모드 폐루프.
- **결과**: **correct 80.0%**(문서 81-85 밴드 내 → **포트 회귀 없음 검증**), **wrong 5.5%**, **blank 0.0%** → correct−wrong **−74.5pp**, correct−blank **−80.0pp**. **§0.6 R2 통과: 정책이 언어를 결정적으로 사용**(언어 토큰이 손목캠 −34.8pp보다 큰 레버). 신뢰성=방향적 확정 — **wrong가 핵심 증거**(유효 지시문 불일치의 능동적 오도), blank는 OOD 혼입 주의. 확정은 3시드 + LIBERO-Para. (코드 경로 감사=버그 없음.)

### 2026-07-09 · cowork 검토 통합 (외부 이론 파트너)
- **무엇을**: cowork 검토노트 반영 — 구현 감사·문헌 재검증·우선순위 재정렬. (소통은 `docs/` 폴더.)
- **어떻게**: **F2 재판정** "부분확정"→**"필요조건 통과·dense 미검증"**(patch-mean=pooling이라 dense 구조적 미테스트, clsmp≈cls=전역표현 이점). F3·wrong/blank **사전등록**, arXiv ID 보정 3 + 경쟁논문 6 반영, SigLIP2 crop 확인(no-crop=정상), paraphrase 언어취약성(task2 100%→11.7%)을 판별지표로 승급. 회신 `docs/COWORK_REPLY_2026-07-09.md`.
- **결과**: 신규성 재정의(action-grounding × 언어보존 × dense-latent-bottleneck 결합; DynaFLIP이 frozen SOTA라 경계 좁아짐). 조치 큐 확정: F3에 **DINOv2-registers**, F7에 **sim-렌더 confound**, **LIBERO-Para** 공개벤치 배선, F1 RADIO(C-RADIOv4 2601.17237). (상세 내부 규율문서는 비공개.)

### 2026-07-09 · 리포 정리 (팔로업 가능성)
- **무엇을**: 새 합류자가 헷갈리지 않게 문서 정리.
- **어떻게**: README에 **문서 맵**(README=사용법 / DESIGN=설계 / KICKOFF=실행 / PROGRESS=로그)
  + F-시리즈 옵션·디렉터리 갱신, `src/README.md` 동기화(anchor·motion_lang·diagnosis 추가,
  policy `mlp/cls/pma`→`mlp/flow` 오기 수정), KICKOFF에 내부문서 caveat. (전담 서브에이전트 감사 후 적용.)
- **결과**: 정착 파이프라인 vs 활성 F-시리즈 구분 명확화, 코드-문서 일치.

### 2026-07-09 · 원격 파이프라인 구축 + EGL 렌더 이슈 해결
- **무엇을**: 외부 GPU 서버(`kist_a6000_ss`, RTX 6000 Ada ×10)에서 실학습·폐루프 실행 환경 구축.
- **어떻게**: GitHub push → 원격 `/workspace/CLIP_ws` 클론, data/models 심링크(기존 스냅샷 재사용),
  `~/clip_ws` 재지정. 시스템 python(libero+mujoco3.3.2+torch cu124) 사용. GPU 8·9.
- **결과**: phase1·phase2 실학습 **성공**(체크포인트 저장). 단 폐루프에서 `MUJOCO_GL=egl` 크래시
  — 컨테이너에 nvidia EGL 부재(+device_id 충돌). **`MUJOCO_GL=osmesa`(CPU 렌더)로 수정 → 정상**
  (task0 렌더·성공 확인). 학습 파이프라인은 원격에서 검증됨.

### 2026-07-09 · SigLIP2 토크나이저 정합성 검증 (서브에이전트)
- **무엇을**: SigLIP2 로드 시 `bos/eos_token_id 49406/49407`(CLIP 토큰) 경고가 텍스트 경로를
  오염시키는지 (F1 언어비교 유효성).
- **어떻게**: 전담 서브에이전트가 로컬에서 토크나이저 클래스·토큰 id·텍스트 임베딩 판별력 검사.
- **결과**: **정상(CORRECT)**. 토크나이저 = GemmaTokenizer(vocab 256000, eos=1). 경고는 config.json의
  CLIP 잔재 필드가 정적 range 검증에만 걸린 것으로 토큰화·`get_text_features` 미사용. 임베딩
  결정론적·판별적(동일 cos≈1.0, 상이 0.59~0.66). **§3 무오염, 수정 불요.**

### 2026-07-09 · F2 — dense 디코더빌리티 프로브 (go/no-go)
- **무엇을**: dense 표현이 CLIP-pooled보다 액션을 더 잘 디코딩하는지(오프라인) — 전체 dense 가설 관문.
- **어떻게**: `src/diagnosis/f2_dense_probe.py`. 인코더별 상태조건부 `[Δ표현, z_t]` → GT action chunk를
  RidgeCV+얕은 MLP로 회귀, held-out R². z-score 표준화(차원 confound 제거). 3시드·60ep libero_spatial.
  (초판이 음의 R²로 퇴화 → 측정 감사 후 상태조건부+RidgeCV로 수정 = 유효 영역 복원.)
- **결과** (CLIP-pooled 대비 R² gap, 3시드 평균): **DINOv2-cls +0.145 / clsmp +0.151 (견고)**,
  SigLIP2 +0.048(불안정), fusion +0.113(<best-single). 판정 **✅부분**. 단서: (a) 오프라인≠폐루프
  (E-series에서 DINOv2 오프라인 우세가 폐루프 −21.8pp 패배), (b) clsmp≈cls → patch 특이적 아님
  (전역 CLS 이점, 진짜 dense는 F3 attention-pool 필요), (c) 차원 confound 배제. → **F3 진행 정당**.

### 2026-07-09 · wrong/blank-instruction 판별 하네스 (§0.6 R2)
- **무엇을**: 정책이 언어를 실제로 쓰는지 측정(표준 SR로는 안 보임).
- **어떻게**: `rollout_sim.py`에 `--instruction-mode {correct,wrong,blank}` 추가. wrong=다른 태스크
  지시문(순환 오프셋), blank=빈 문자열. 기본 correct는 불변.
- **결과**: 로직 로컬 검증(correct≠wrong, blank=""). 실행은 원격(현재 진행 중). 측정 = correct−wrong/−blank.

### 2026-07-09 · HY03 하이브리드 언어정렬 이식
- **무엇을**: phase1에 언어정렬(1급 불변식) 복원 — 모션문장 대조로 언어축 유지.
- **어떻게**: `networks.py` `DeltaAE`에 `align_mode{dz,direct,hybrid}` + `info_nce`(SupCon 다중양성,
  학습형 온도). hybrid = dz 손실 + λc·InfoNCE(모션문장 타깃, `motion_lang.py`). train_phase1 배선.
- **결과**: **dz 기본 경로 비트 동형**(신규 파라미터는 가드 안, 손실 합산식 불변) + hybrid --smoke 실행 확인.

### 2026-07-09 · 앵커 추상화 이식 (F1 전제)
- **무엇을**: 다중 백본(CLIP/SigLIP2/DINOv2) 공통 인터페이스.
- **어떻게**: `src/core/anchor.py`(get_anchor 레지스트리). train_phase1가 앵커 선택, `latent_dim=anchor.dim`,
  libero 캐시 하위호환 폴백(기본 CLIP=기존 평면 캐시).
- **결과**: anchor=clip 기본 **비트 동형** + 3앵커 로컬 shape 검증(CLIP 768/1024, SigLIP2 1152, DINOv2 1024/2048).

### 2026-07-09 · F0 — latent_dim 일반화 리팩터
- **무엇을**: `policy.py`의 `LATENT=768` 하드코딩 제거 + dense 캐시 경로 신설 (이후 전 단계 전제).
- **어떻게**: `latent_dim` 파라미터화, phase2/eval이 phase1 체크포인트에서 주입. `dense_embeddings()` 추가.
- **결과**: **비트 동형 게이트 PASS** (phase1 22/22 + phase2 51/51 tensor `torch.equal`). anchor_proj 층은 F1로 연기.

### 2026-07-08 · 워크스페이스 마이그레이션
- **무엇을**: 정리된 LIBERO 전용 코드베이스를 신규 레포로 이전.
- **어떻게**: `github.com/Seung-Sub/CLIP_Embedding_Action` 신규 생성, 단일 커밋(Seung-Sub 단독), data/models 심링크.
- **결과**: 실행 가능한 clean 워크스페이스 확립.

---

## 다음 (계획, KICKOFF 순)
- 롤아웃 결과 → 재현 검증(≈85%) + 언어사용 판정 → 기록
- **F3** dense obs 융합(학습 attention-pool, 폐루프+P6 게이트) · **F1** RADIO 앵커 head-to-head
- 이후 F4(학습형 latent action) · F5(통합) · F6(아키텍처) · F7(frozen vs LoRA)
