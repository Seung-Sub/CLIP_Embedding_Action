# RESULT — R-시리즈 검색 포트 R-0/R-1 (오프라인 어댑터 재현 + 판별 하네스)

*실행: 2026-07-21, kist_a6000_ss CPU 전용 (GPU 8/9 = W-A 확증 캠페인 점유, 미접촉).*
*사전등록 스펙/게이트: `docs/ANALYSIS_colleague_retrieval_control.md` §5. 코드: `scratchpad/rseries/` (로컬+원격 동일), 결과 JSON: `outputs/rseries/` (원격 생성 → 로컬 복사).*

---

## 0. 한 문단 요약

동료 랩의 retrieval 기반 언어 제어를 **우리 기질**(large256-single phase1의 ζ = g(a, z_t), SigLIP2-large256 텍스트타워, 우리 전처리)로 이식했다. **분할까지 동료와 정확히 일치**(seed=2 에피소드 분할 → train 1995 seg / held-out 497 seg, 그들 수치와 동일)한 상태에서: canonical held-out top-1 **0.972**(그들 0.974)로 **G-R0a 통과**, 그들의 unseen 템플릿 **0.952**로 **G-R0c-1 통과**. 그러나 **우리가 새로 작성한 독립 3rd 템플릿셋은 0.773으로 G-R0c-2 실패**(최근접-문구 규칙 기준; 평균-임베딩 규칙으론 0.944) — 실패는 사실상 **grasp 한 클래스**(0.04)에 집중되며, 원인은 학습 분포 밖 동사("unhand", "capture", "scoot", "bring toward the rear")의 텍스트 임베딩이 이웃 클래스로 붕괴하는 **표현 불변성의 반경 한계**다(그들 스스로 기록한 nudge/seize 실패와 같은 계열 — 그들의 3rd셋 0.974는 학습 동사를 재사용한 "가까운" 셋이었고, 우리 셋이 이 경계를 드러냈다). 정직 대조(G-R0b)는 그들 경계를 우리 기질에서 재확인: **텍스트-무관 MLP 0.968 ≈ 어댑터 0.972**(분류기 등가), **상태-잔차화 ζ−r(z_t) 시 0.749/0.543으로 급락**(ζ의 상태-운반 성분이 정확도의 상당 부분을 실어 나름). R-1 판별 하네스는 검색 레벨에서 **correct 8/8 (마진 0.54), 스왑 민감 56/56, 셔플 텍스트는 마진 1/3로 붕괴(0.18), 넌센스 0.06** — "주입은 죽고 검색은 산다"의 우리-기질 재현이 성립한다. *(후속: 잔차 효과벡터로 처음부터 학습한 §R-0b(§6)은 0.742로 사전등록 "혼합" 대역 — canonical 신호의 70%는 상태-무관, 일반화의 ~55%는 상태-운반, 클래스별 비대칭이 핵심.)*

---

## 1. 설정 (검증된 사실)

| 항목 | 값 |
|---|---|
| phase1 | `checkpoints/phase1_libero_siglip2_large256.pt` — `phase2_libero_large256_matchedbase.pt`의 config가 참조하는 그 ckpt (로드로 확인). latent 1024, n_chunk 16, chunk_repr time |
| ζ | g(resample16(A)·actnorm, z_t), z_t = SigLIP2-large256 pooled **RAW**(normalize=false, native-best) 세그먼트 시작 프레임, 캐시 `/data2/clip_ws_cache/cache/libero_emb_large256/siglip2-so400m/joint/raw` |
| 시각효과 Δz | z[end]−z[start] 동일 캐시 (단일 스트림 — 동료는 dualavg z_t + SigLIP-only Δz로 분리; 우리는 한 스트림이 둘을 겸함. **기질 차이로 문서화**) |
| 텍스트 | SigLIP2-large256 텍스트타워, `src/core/anchor.py` 경유, 동일 normalize=false |
| 분할 | 에피소드 단위 RandomState(**seed=2**) permutation, val 20% — 동료 `sem_adapter_heldout.yaml`과 동일 규약, 같은 500 데모·같은 정렬 순서 → **train 400ep/1995seg, held-out 100ep/497seg = 그들 수치와 정확히 일치** (분할 재현 확증) |
| 어댑터 레시피 | 동료 paraphrase-승자 판 그대로: SupCon multi-positive, 언어은행 = canonical+9 paraphrase/클래스(80행), 클래스별 템플릿 hard-negative 분모 주입, sem_av 0.1/sem_al 0.05/sem_vl 0, τ 0.07, AdamW lr 1e-3 wd 1e-4, bs 256, 200ep, cosine+warmup 200, seed 0 |
| 은행 자산 | 그들 `banks/subgoal_phrases_libero_spatial.json` + `hard_negative_bank.json` 재사용(무변경), 템플릿셋은 그들 e7/e7b 결과 JSON에서 추출 |
| 실행 은행 (R-2 대비) | `outputs/rseries/effect_bank_ours.pkl` — train 1995 seg의 raw action chunk + ζ + z_t + Δz + netd/grip + 라벨 + prov 저장 완료 |

채점 규칙 주의: 다중 템플릿 셋의 클래스 점수 = **클래스 내 최근접 문구 max-cos**(1차, 아래 표 기준). 평균-임베딩 대안치를 병기(JSON에 둘 다 저장). 동료의 채점 스크립트는 클론에 없어 규칙 동일성은 미확인 — canonical(문구 1개)에선 두 규칙이 동치라 G-R0a 비교는 안전.

---

## 2. R-0 결과 — **정직 대조 표** (held-out 497 seg, top-1)

| 판 | canonical | seen-para(9) | **unseen(그들 16)** | 3rd(그들) | **3rd(우리 24)** |
|---|---:|---:|---:|---:|---:|
| **어댑터 (우리 ζ)** | **0.972** | 0.972 | **0.952** | 0.970 | **0.773** (평균-emb 0.944) |
| 어댑터, 상태-잔차 ζ−r(z_t) | 0.749 | 0.745 | 0.543 | 0.604 | 0.626 |
| 텍스트-무관 MLP (ζ) | **0.968** | — | — | — | — |
| 텍스트-무관 MLP (ζ−r(z_t)) | 0.857 | — | — | — | — |
| 텍스트-무관 MLP (z_t 단독) | 0.863 | — | — | — | — |
| *참고: 동료(그들 기질)* | *0.974* | *0.974* | *0.96* | *0.974* | *(해당 없음)* |

보조: RidgeCV(z_t→ζ, α그리드 0.1–1e4, week0 f2 패턴) held-out R² = **0.790** (α=3.16). 우연 수준: majority 0.201 / uniform 0.125 (그들 e6 무작위 사영 0.147±0.032).

**채점 규칙 강건성** (동료 e6은 "템플릿별 정확도의 평균" 규칙 사용 — 사후 확인, `r0_results.json.per_template_rule`에 병기): unseen(그들) max 0.952 / 템플릿별-평균 **0.930** — 어느 규칙으로도 G-R0c-1 PASS. 우리 3rd셋 max 0.773 / 템플릿별-평균 **0.742**(최악 템플릿열 0.384) — 어느 규칙으로도 G-R0c-2 FAIL. 참고로 그들 novel3조차 템플릿별-평균으론 0.869(<0.90) — 판정은 규칙에 좌우되지 않으나, 그들 발표치(0.974)와의 1:1 비교엔 규칙 명시가 필수.

**정직 판독 (그들 경계의 우리-기질 재확인 + 강화)**:
1. **분류기 등가**: 어댑터 0.972 vs 텍스트-무관 MLP 0.968 — canonical 정확도 자체는 언어의 공로가 아니다. 어댑터의 고유 가치는 (그들 주장대로) paraphrase-불변 + 언어 인터페이스뿐.
2. **상태-운반 성분**: z_t 단독으로 이미 0.863이 나오고(세그먼트 위상은 장면 상태로 크게 예측됨 — approach/grasp/place는 궤적 앞·중·뒤), ζ의 79%가 z_t로 선형 예측된다. 잔차화하면 어댑터 0.749/unseen 0.543, MLP 0.857 — **ζ의 판별력 중 상당분은 액션이 아니라 상태에서 온다**. 이것은 우리 A5 통과의 조건부("상태-잔차 대조 의무")가 옳았음을 보인다. 단 잔차판 MLP 0.857은 여전히 우연(majority 0.201)을 크게 상회 — 액션 고유 신호도 실재한다.
3. **불변성 반경**: unseen(그들) 0.952는 통과하지만 **우리 3rd셋 0.773 실패는 grasp 붕괴(0.04, 100 중 94가 release로 흡수) + backward(0/4, 내재 결함 클래스) + right(6/8)에 집중**. 문구별 진단(`r0_results.json.third_ours_phrase_diagnostic`): "unhand the black bowl"(release용)이 grasp 잠재에 더 가깝고(+0.505), "capture the black bowl with the gripper"는 approach로(+0.704), "bring the black bowl toward the rear/front"는 place로(+0.44), "bring … toward the right side"는 **left**로 붕괴. 즉 paraphrase-불변성은 **학습 paraphrase 은행의 어휘 반경 안에서만** 성립하고, 반경 밖 동사는 SigLIP2 텍스트 기하의 임의 이웃으로 떨어진다. 그들 novel3(0.970)가 통과하는 이유는 grab/take/hold 등 **학습 동사를 재조합한 셋**이기 때문 — 3rd셋의 독립성 수준이 결과를 좌우한다는 것 자체가 이번 포트의 신규 발견.

### 게이트 판정 (사전등록 임계 대비)

| 게이트 | 임계 | 실측 | 판정 |
|---|---|---:|---|
| G-R0a canonical | ≥ 0.90 | 0.972 | **PASS** |
| G-R0b 정직 대조 병기 | 의무 | 표 §2 | **PASS (이행)** |
| G-R0c-1 unseen 템플릿 | ≥ 0.85 | 0.952 | **PASS** |
| G-R0c-2 독립 3rd셋(우리 작성) | ≥ 0.90 | **0.773** (평균-emb 0.944) | **FAIL** |
| 종합 | — | — | **3/4 — G-R0c-2 실패로 전체 FAIL** |

우리 3rd셋 프롬프트 전문은 `scratchpad/rseries/rseries_common.py`의 `THIRD_SET_OURS` 및 `r0_results.json.third_set_ours_prompts`에 문서화(클래스당 3문구, 그들 canonical/para3/para9/gen_seen/unseen/novel3와 문구 단위 중복 0).

---

## 3. R-1 결과 — 검색 레벨 판별 하네스 (은행 = train 1995 seg, 질의 = P_text)

| 그룹 | n | 정답률 | 클래스 마진 (best−2nd best) |
|---|---:|---:|---|
| **correct** (canonical) | 8 | **8/8** | **0.537 ± 0.089** [0.33, 0.63], top-5 합의 전부 5/5 |
| **wrong/스왑** (다른 클래스 명령) | 56쌍 | **56/56** — 발화된 클래스를 검색 | (correct에서 유도) |
| paraphrase unseen(그들) | 16 | 10/16 (0.625) | 0.223 ± 0.206 |
| paraphrase 3rd(우리) | 24 | 18/24 (0.750) | 0.283 ± 0.152 |
| **셔플 텍스트** (단어 순서 파괴 ×3/문구) | 24 | — | **0.185 ± 0.095** (correct의 ~1/3) |
| 넌센스 문장 | 8 | — | **0.056 ± 0.041** |
| 무관 명령 (다른 객체/행위) | 6 | — | 0.097 ± 0.085 |

- **이중 해리의 우리-기질 확인**: correct 8/8 + 스왑 56/56 + 큰 마진 vs 셔플/넌센스의 마진 붕괴 — 검색은 텍스트 의미를 따라가고, 의미 없는 텍스트는 저마진·불안정 검색(넌센스 8건 중 4건이 release, 3건이 backward 같은 저빈도/흡수 클래스로 떨어짐)이 된다. 사전등록한 "셔플-텍스트 대조군"(그들 데모에 없던 통제) 결과: **셔플 24건 중 21건은 원문 클래스를 유지** — SigLIP2 텍스트타워의 bag-of-words 성향으로 방향·동사 단어가 순서 없이도 검색을 지배하되, 마진은 1/3로 줄어 신뢰도 신호로는 구별 가능.
- **검색 레벨은 분류 레벨보다 어렵다**: 같은 unseen 셋이 분류(§2) 0.952 → 검색 0.625. 방향이 반대(문구→세그먼트)면 클래스당 수백 세그먼트의 max-cos 극값이 오검색을 유발(nudge류 5/6 실패 — 그들 원본 어댑터의 unseen 검색 11/16=0.69와 같은 급). 마진이 작은 실패(m<0.07)가 대부분이라 top-5 합의/마진 임계로 걸러낼 수 있음.
- per-class 혼동행렬(정답+paraphrase 합산)과 문구별 상세는 `outputs/rseries/r1_results.json`.
- **교차-기질 행(그들 어댑터 가중치 재실행): SKIP** — `semantic_adapter_paraphrase.pt`가 클론(로컬)·원격 어디에도 없음(`SigLIP/checkpoints` 부재 확인). 그들 수치는 e6/e7 결과 JSON 인용으로 갈음.

---

## 4. 다음 단계 권고

**R-2 (zero-demo 방향 명령 실행, 통계 보강판) — GPU 필요, 지금은 미착수.**
- 준비 완료 자산: `outputs/rseries/effect_bank_ours.pkl`의 raw_chunks(실행용) + `adapter_main.pt`.
- 스펙(사전등록 유지): 4방향+2그리퍼 × ≥3 장면 × ≥3 init (명령당 n≥9), 오픈루프 32스텝 타일링 재생. 1차 지표 = 반대쌍 분리(left−right |Δy|>0.05m 등, 규약-무관), 2차 = **카메라-frame 재라벨 후** 사람-기준 방향 정확도(그들 0/4 결함을 닫는 우리 차별점). backward는 사전 제외(은행 5 seg, 내재 라벨 결함). right는 저빈도(39 seg) 유의.
- **착수 조건: W-A 확증 캠페인의 GPU 8/9 점유 해제 후.** 런처는 기존 rollout 인프라(`src/eval_libero`) + 렌더 재시도 슈퍼바이저(osmesa 세그폴트 이력) 재사용.
- R-1의 함의로 실행 프로토콜에 **마진 임계**(예: class margin < 0.1이면 "명령 불확실" 거부)를 넣을 것 — 셔플/넌센스와 정상 명령이 마진으로 분리됨을 §3에서 확인했다.

**R-3 (retrieval-conditioned decoding)**: 검색된 ζ*를 우리 h(ζ*, z_t)에 조건화(매 상태 재복호) — R-2와 같은 롤아웃 셋업에서 armed 비교 가능. R-2 이후.

**G-R0c-2 실패의 처리**: 논문 서사에서 "paraphrase-불변" 주장은 **"학습 paraphrase 반경 내 불변"**으로 한정할 것. 반경 확장 실험(문구 은행 증강 20/eff는 그들이 이미 포화 확인)보다, 실패 문구가 **저마진으로 자기표식**된다는 R-1 관찰을 방어 논리로 쓰는 편이 정직하다.

---

## 5. 산출물 색인

| 경로 | 내용 |
|---|---|
| `scratchpad/rseries/{rseries_common,r0_build_bank,r0_train_eval,r1_harness}.py` | 포트 코드 (로컬+원격 동일본, 동료 레포 무변경) |
| `outputs/rseries/effect_bank_ours.pkl` | 우리 효과 은행 (train 1995 + heldout 497: raw_chunks/ζ/z_t/Δz/netd/grip/lab/prov) |
| `outputs/rseries/adapter_main.pt`, `adapter_resid.pt` | P_action/P_visual/P_text 가중치 (본판/상태-잔차판, ridge 계수 포함) |
| `outputs/rseries/r0_results.json` | R-0 전 수치 + 게이트 + 문구별 진단 |
| `outputs/rseries/r1_results.json` | R-1 전 질의 상세 + 혼동행렬 + 마진 분포 |
| `outputs/rseries/text_emb_cache.npz` | SigLIP2-large256 텍스트 임베딩 메모이즈 (185 문구) |
| `scratchpad/rseries/r0b_resid_adapter.py` | §R-0b 스크립트 (사전등록 해석 헤더 포함, 로컬+원격 동일본) |
| `outputs/rseries/r0b_results.json`, `r0b_run.log` | §R-0b 전 수치 (변형×시드, R-1 배터리, 방향 프로브) |
| `outputs/rseries/r0b_adapter_resid_{raw,std}.pt` | 잔차 어댑터 가중치 (seed 0, ridge 계수/표준화 파라미터 포함) |

---

## 6. §R-0b — 잔차 효과벡터로 처음부터 학습: 정렬은 진짜인가, 상태-운반인가

*실행: 2026-07-21, kist_a6000_ss CPU 전용 (GPU 미접촉). 코드 `scratchpad/rseries/r0b_resid_adapter.py`, 결과 `outputs/rseries/r0b_results.json`.*

### 6.0 전제 정정 (착수 전 코드 검증으로 확인)

R-0b의 발제 전제("R-0의 0.749는 raw-ζ 어댑터를 잔차에 불공정 평가한 수치")는 **사실과 다르다**. `r0_train_eval.py`의 G-R0b(2) 단계는 이미 ζ_res = ζ − r(z_t)로 어댑터를 **처음부터 학습**했고(`train_adapter(zres_tr, …)` → `zres_ho` 평가), 0.749/0.543이 바로 그 from-scratch 수치다. R-0b 재실행(seed 0)은 canonical **0.7485를 비트 단위로 재현**(`reproduces_r0_adapter_resid: true`). 따라서 R-0b의 신규 기여는 (a) 이 재현 확인, (b) 표준화 변형, (c) 시드 강건성, (d) 잔차 어댑터에 대한 R-1 검색 배터리, (e) 클래스별 잔차 민감도, (f) 방향-전용 프로브다.

### 6.1 사전등록 해석 (실행 전 스크립트 헤더에 고정; 사전 인지 수치는 resid_raw seed0 0.7485뿐)

1차 지표 = 최고 잔차 변형의 canonical held-out top-1, 시드 0/1/2 평균. **≥0.90** → 상태는 잉여, C3′ 강형(state-free) 유지 / **0.60–0.90** → 혼합, 상태 지분을 정량 병기한 C3′ / **≤0.60** → 대부분 상태-운반, "장면-효과 검색"으로 정직하게 재구성. 2차(사전등록): 방향 클래스가 approach/grasp/place보다 잔차화 손실이 작을 것; correct≫shuffled≫nonsense 마진 서열은 유지될 것.

### 6.2 결과 — **판정: 0.742 → 혼합(0.60–0.90) 대역**

잔차 스케일 점검: ζ_res는 ζ 총분산의 **13.6%**만 보유(차원별 std 중앙값 0.031 vs 0.083, 퇴화 없음 — LayerNorm이 전역 스케일을 흡수하므로 표준화 효과는 예상대로 미미).

| 판 (from-scratch, 시드 3개 평균±std) | canonical | unseen(그들) | 3rd(우리) | 텍스트-무관 MLP |
|---|---:|---:|---:|---:|
| raw ζ (R-0 기준선) | 0.972 | 0.952 | 0.773 | 0.968 |
| **ζ_res raw** | **0.7425 ± 0.006** | 0.541 | 0.635 | 0.857 |
| ζ_res 표준화 | 0.7418 ± 0.002 | 0.527 | 0.622 | **0.883** |
| 참고: z_t 단독 MLP / 우연(majority) | 0.863 / 0.201 | — | — | — |

- **정량화된 상태 지분**: canonical 기준 우연-상회 신호의 **70% 보존** ((0.742−0.201)/(0.972−0.201)), 즉 ~30%가 상태-운반. 단 **일반화(unseen)는 상태 의존이 더 크다**: (0.541−0.201)/(0.952−0.201) = **45% 보존**, ~55% 상태-운반. paraphrase-불변성이 canonical 판별보다 상태에 더 기대고 있었다는 뜻.
- 잔차 MLP 상한 0.883 vs 잔차 어댑터 0.742 — 언어 정렬 비용 ~0.14는 raw판(0.968 vs 0.972, ~0)보다 크다: 상태를 빼면 남는 액션-고유 신호는 SigLIP2 텍스트 기하에 정렬하기 더 어렵다.

**클래스별 잔차 민감도** (canonical, seed 0, main→resid_std Δ): approach **−0.450**, place **−0.430** ≫ left −0.184, grasp −0.110, forward −0.105, release **−0.031**, right/backward ±0. 상태-운반 성분은 **approach/place(궤적 위상 클래스)에 집중**되고, **gripper 이벤트(grasp/release)는 잔차화를 거의 무손실 통과** — 우려하던 grasp는 0.87로 생존. 액션 자체에 서명이 있는 클래스(그리퍼 개폐)가 가장 state-free다.

**R-1 배터리 (잔차 어댑터, 검색 레벨)**: correct **8/8**, 스왑 **56/56** — 이중 해리는 잔차에서도 성립. 마진은 전반 축소: correct 0.248(raw판 0.537), shuffled 0.120(21/24 원 클래스 유지), nonsense 0.045. correct/shuffled 비 2.9×→2.1×로 좁아져 **R-2에 제안한 마진-임계 거부 규칙은 잔차 기질에선 여유가 줄어든다**(서열 자체는 사전등록대로 유지).

**방향-전용 프로브 — 2차 사전등록 가설은 반증**: 3-way(forward/left/right, heldout 95, majority 0.516)에서 z_t-단독 MLP가 이미 **0.895** — LIBERO-spatial에서 "어느 방향으로 옮기나"는 시작 장면(그릇/접시 배치)이 거의 결정한다. 방향-전용 ridge(R² 0.789)로 잔차화하면 3-way 어댑터 0.488(raw)/0.593(std)로 우연 인근 붕괴(잔차 MLP 0.758은 상회 — 어댑터가 못 쓰는 액션-고유 신호는 남아 있음). 8-way 어댑터의 3-way 제한 채점은 main 0.926 → resid 0.842/0.832로 완만. 즉 **"방향이 더 state-free"가 아니라, 방향 라벨이 이 벤치마크에서 가장 장면-결정적**이고, state-free 서열은 실제로 **release/grasp > forward/left > approach/place**다.

### 6.3 사전등록 해석 적용 (판정)

**혼합 대역(0.60–0.90) 확정** — C3′는 다음 한정으로 유지한다: *"언어↔행동효과 정렬의 canonical 판별력은 대부분(70%) 상태-무관 액션 신호로 성립하지만, paraphrase 일반화의 절반 이상(~55%)과 approach/place 클래스 판별은 장면-상태가 실어 나른다."* 순수 "장면-효과 검색"으로의 격하는 부당(잔차만으로 0.74/0.88, 우연 0.20)하고, 강형 state-free 주장도 부당(특히 unseen 0.54). 논문 서사에는 클래스별 비대칭(그리퍼 이벤트 state-free, 궤적-위상·방향 클래스 상태-의존)을 표로 병기하는 것이 가장 정직하다. R-2 실행 프로토콜에는 방향 명령의 장면-결정성(z_t 단독 0.895)이 **교란변수**로 작용함을 명시할 것 — zero-demo 방향 실행의 성공이 "언어가 방향을 지정했다"의 증거가 되려면 같은 장면에서 반대 방향 명령을 주는 반사실 셋업(이미 스펙에 있는 반대쌍 분리 지표)이 필수다.
