# EXPERIMENTS_INDEX — 전 실험 셀 대장

> "무엇을 다 돌렸고, 제대로 했고, 무엇이 나왔나"의 마스터 레퍼런스. 테마별로 묶었다.
> 판정 태그: **양성**(positive) / **음성**(negative) / **널**(null=차이 없음) /
> **부분**(partial) / **미결**(inconclusive) / **인프라/대기**. 수치는 별도 명기 없으면
> libero_spatial · 10태스크 · correct-instruction · no-aug 클린 · MUJOCO osmesa.
>
> **근거문서 경로**: 인용된 `docs/*`는 `.gitignore`(작업 폴더)라 클론에 없다 — tracked 사본은
> `reports/2026-07-18_wrist_fusion_session/`·`reports/2026-07-22_retrieval_capacity_session/` 번들에 있다.
> 외부(콜리그) 수치는 다른 프로토콜(50롤/EGL) = 방향만 참조, 절대비교 불가.
> 근거문서 중 `docs/*` 는 gitignore(로컬/원격 전용), `reports/*`·최상위 md 는 공개.

---

## A. 언어보존·언어사용 (헤드라인 양성)

| 셀 | 무엇을 | config/방법 | 결과 | 판정 | 근거 |
|---|---|---|---|---|---|
| **R2 판별 하네스(초판)** | 정책이 언어를 실제 쓰나 | `rollout_sim.py --instruction-mode {correct,wrong,blank}`, 1시드 | correct 80.0 / wrong 5.5 / blank 0.0 → c−w **−74.5pp** | 양성 | PROGRESS 2026-07-09 |
| **1a 언어사용 3시드** | −74.5pp 확정 | full-data phase2 s0/s1/s2, 동결 phase1 공유 | c−w s0 +76.5 / s1 +76.5 / s2 +74.5 → **평균 +75.8pp**, 3시드 CI로 0 분리 | 양성(헤드라인) | PROGRESS 1a, FOLLOWUP §1 |
| **SigLIP2-large256 3시드** | backbone-agnostic? | pooled-only large256 seed{1,2,3} | c−w **+76.5±1.8pp**(75/79/75) = CLIP과 수렴 | 양성 | PROGRESS 2026-07-14, FOLLOWUP §1 |
| **1d-goal 일반성** | goal suite 로 일반화? | 신규 LIBERO-Goal 정책 | correct 88.5 / wrong 0 / blank 0 → **+88.5pp**(spatial보다 강함) | 양성 | PROGRESS 1d-goal |
| SigLIP2-goal | 〃 backbone×suite | | c−w **+92.0** | 양성 | FOLLOWUP §1 |
| **1b LIBERO-Para** | 문자열 암기 vs 의미 | LIBERO-Goal 정책, 객체/동작 어휘 페러프레이즈 | object **51.0**(−37.5) / action **61.0**(−27.5), Δ 10pp < 15 임계 | 부분(의미 이해, 객체명사 최약축) | PROGRESS 1b |
| **1c 씬내 타깃-스왑** | 지시 타깃 충실 vs 편향 | 같은 씬 형제 태스크 지시로 스왑, instructed/orig/neither | Faithful **48%** / Biased **2.5%** / neither 49.5 (VLA OFT/π0 Biased 45–79 ≫) | 부분 양성(공간 구성적 접지, spatial-only) | PROGRESS 1c, FOLLOWUP §2 |
| 페러프레이징 폐루프(2-A) | 원문 미사용 강건성 | `rollout_sim_paraphrase.py`, 30조건 | 85.2(원문) → **67.5**(페러프)−17.7pp (task별 0~96.7 편차) | 부분(LIBERO-Para −22~−52보다 나음) | README §7 |

---

## B. 시각-풍부화 삽입점 지도 (핵심 서사)

핵심 질문: DINO 계열 기하를 **어디에** 넣어야 폐루프 SR↑? → 관측/조건화-측만 양성.

| 셀 | 삽입점 | 방법 | 결과 | 판정 | 근거 |
|---|---|---|---|---|---|
| **F0** | (인프라) | `latent_dim` 하드코딩 768 제거·dense 캐시 경로 | 비트 동형 게이트 PASS(phase1 22/22 + phase2 51/51) | 인프라/PASS | PROGRESS F0 |
| **F2** | (오프라인 프로브) | dense 디코더빌리티 `f2_dense_probe.py`, RidgeCV+MLP, 3시드 | DINOv2-cls **+0.145**/clsmp +0.151(견고), SigLIP2 +0.048, fusion +0.113(<best-single) | 부분(오프라인만, clsmp≈cls=전역 CLS 이점) | PROGRESS F2 |
| **F3** 관측 덧대기 | **관측(dense)** | 앵커 + ObsFusion K토큰 → phase2, 3 arm | a(no-obs) **50.0** > b(mean) 31.5 > c(attnpool) 15.5 (단조 악화) | **음성** | PROGRESS F3, FOLLOWUP §3.1 |
| **F1** 앵커 교체 | **앵커** | CLIP/SigLIP2/DINOv2/RADIO head-to-head(matched-reg) | SR 3중 null(encoder 품질 ≠ SR); 언어사용은 backbone-agnostic | 음성(SR) | PROGRESS F1, FOLLOWUP §3.2 |
| **C1** fine 채널 | **타깃/코드** | F4 ζ_g(pooled)⊕ζ_f(SigLIP2 patch cross-attn 96d tanh게이트) | 게이트 α **0.027 미개방**, paired +1.5(CI 0 포함), ζ_f 절제 시 +1.0 | 음성 | PROGRESS C1, FOLLOWUP §3.3 |
| **C2** fine 채널 | **타깃/코드** | 〃 ζ_f=DINOv3-L/16@256 patch | α **0.0015**(3기질 최소), paired +0.00, ζ_f 절제 **+6.7**(유해) | 음성 | PROGRESS C2 |
| ↳ C2 감사 재기준선 | (감사) | no-aug 85.0 vs c2_control+aug 71.8 | −13pp 실재하나 버그 아닌 **aug+dropout 정규화 confound** → 이후 no-aug 클린 밴드 | 방법 정정 | FOLLOWUP §3.3 감사 |
| **S1 avg** 관측융합 | **관측 본류(z_t)** | `DualFusionAnchor` avg 1024, phase1 재적합 | correct **91.5** / c−w +74.0 / paired **+6.5pp CI[+0.5,+15]** / R²+0.735 | **양성** | PROGRESS S1, FOLLOWUP §4 |
| **S1 concat** | **관측 본류** | `DualConcatAnchor` concat 2048 no-mix | correct **97.5** / c−w +69.0 / paired **+12.5pp CI[+4.5,+22]**(견고) / R²+0.749 | **양성(SR 최고)** | PROGRESS S1, FOLLOWUP §4 |
| **S1b** 역할분리 | 조건화=SigLIP2 / 앵커·코드=융합 | `cond_anchor` flag, align_block=1024 | noalign correct **86.0**(SR 이득 상실)/c−w +78; hybrid 67.5 | 음성(반증) | PROGRESS S1b, FOLLOWUP §6 |
| **S2 h-flow** 디코더 | 디코더-측 flow | `ChunkFlowDecoder` CFM, concat 융합 | 오프라인 다봉성 실증(K=32 R² +0.714) but 폐루프 **33%(naive)/37%(fixed-noise)** = mode-switching | 음성 | PROGRESS S2, FOLLOWUP §7 |
| **actionflow** | 정책-측 flow(h 우회) | `flow_space=action`, raw 액션 직접 수송 | af-concat **76** / af-avg **80**(SR·언어 양축 하락; 콜리그 97.2와 반대부호) | 음성(우리 기질) | FOLLOWUP §9 |
| **residual-h-flow** | 디코더 잔차 flow(Q2/M7) | MLP 평균 + 잔차 flow | ~**48–65%**(폐루프 무익, 콜리그 M7/Q2와 정합) | 음성 | FOLLOWUP §10 |
| **grid-token** | 관측 패치 무게이트 | DINOv3 patch 관측 토큰 | **OOM-사망**(무결과, F3-echo 위험) | 미결(음성 방향) | FOLLOWUP §10 |
| DINOv2 vs DINOv3 | 백본 confound | 매칭 전처리(256-no-crop) | DINOv2-avg 90.5 ≈ DINOv3-avg 91.5 = **백본 동급**; 격차는 center-crop 전처리(+5pp) | 중립(confound 해소, "DINOv2 우위" 철회) | PROGRESS 2026-07-16, FOLLOWUP §8 |

---

## C. 전처리 레버 (P-A center-crop)

| 셀 | 방법 | 결과 | 판정 | 근거 |
|---|---|---|---|---|
| **P-A crop** | `anchor.crop {none,dino,both}`, 3팔 20롤×correct/wrong | cropavg_dino 91.0(−0.5 NS) / cropconcat_dino 94.0(**−3.5, CI[−7,−0.5] 유의 하락**) / cropavg_both 93.0(+1.5 NS) — G-cl 전부 탈락 | **음성**("+5pp 전처리 레버" 철회) | `docs/RESULT_PA_crop_screening.md`, `reports/.../RESULT_PA_crop_screening.md`, FOLLOWUP §14 |

wrong SR 상승(17.5→25, 28.5→34.5) = 시각정보↑→언어의존↓ tradeoff 축 재확인.

---

## D. 의미교환 언어축 천장 (week0/week1 오프라인 게이트, 학습 0·전량 CPU)

| 셀 | 무엇을 | 결과 | 판정 | 근거 |
|---|---|---|---|---|
| **T-0** Δz_text 주입 | 텍스트 델타를 h 에 직접 주입(zero-shot) | 방향 cos **0.592 ≈ 셔플 0.598**(무우위). 원인=텍스트델타↔이미지델타 방향 대응 부재(1/10)+시간입도 불일치. h·g 배관은 건강(oracle Δz cos 0.985) | **음성(KILL)** | `docs/WEEK1_gate_results.md`, FOLLOWUP §13 |
| 발산 포트폴리오 10개 | kill-gate 일괄 소진 | 개념기저 통화 KILL / 동작단어 VQ KILL / 검색 조건화 KILL(이득=암기) / 언어 심판자 KILL(≈chance) / 다중스케일 설계-KILL | 음성(언어군 전멸) | `docs/WEEK1_gate_results.md`, `PORTFOLIO_divergent_architectures_v1.md` |
| 등급-언어 단조 | 서수 언어 이해 | 0/6 단조(서수만 ρ=1.0) | 음성 | `docs/WEEK1_gate_results.md` |
| **M6-a** gap 상쇄 | modality gap 이 Δ에서 상쇄되나 | rel_gap **2.505→0.695**(cos 0.277) 실증 = "Δ-문법" 이론 방어선 | 양성(생존 자산) | FOLLOWUP §13, `ANALYSIS_clip_language_limits_v1.md` |
| **A5** 검색-정렬 | ζ̂에서 지시문 검색 acc | **0.952**(검색 수준 텍스트↔ζ 정렬 존재) — 조건부 PASS | 양성(조건부) | `docs/WEEK1_gate_results.md` |
| **P-zg** 상태 지름길 | align 이 상태 성분인가 | **RED** — align cos ~2/3 상태 성분, 상태단독 ridge(z_t→Δz)가 g 상회 → innovation-grounding 발동 | 진단(주의 신호) | `docs/WEEK0_probe_results.md` |
| h eff-rank | 디코더 병목 | eff-rank ~5 재현(콜리그 정합) | 진단 | `docs/WEEK0_probe_results.md` |

**해석**: 현 기질 언어사용 = **태스크-선택 수준**(그래서 c−w +76pp 강함), 청크-수준
벡터 의미교환은 부재. zero-shot 의미교환 주장 폐기, 천장 지도 자체가 해석적 기여.

---

## E. Retrieval 포트 (C3′ — 약형 의미교환 부활)

| 셀 | 무엇을 | 결과 | 판정 | 근거 |
|---|---|---|---|---|
| **콜리그 retrieval 분석** | 언어→검색→시연 재생 기전 검증(read-only) | "주입은 죽고 검색은 산다" 랩내 이중해리(E5 주입 0.58 FAIL vs 검색 8/8). camera-frame 0/4·실행 n=1 등 한계 정직 목록 | 분석(C3-강 사망·C3-약 부활) | `docs/ANALYSIS_colleague_retrieval_control.md` |
| **R-0** 어댑터 재현 | 우리 기질 P_text/P_action(SupCon+paraphrase) | canonical held-out **0.972**(그들 0.974), unseen **0.952**. 텍스트-무관 MLP 0.968=분류기 등가. 상태-잔차화 시 0.749로 급락 | 부분(G-R0a·c1 PASS) | `docs/RESULT_rseries_R0R1.md` |
| **G-R0c-2** 독립 3rd셋 | 우리 작성 신규 템플릿 | **0.773 FAIL**(평균-emb 0.944) — grasp 붕괴. paraphrase 불변성은 **학습 어휘 반경 내** 한정 | 음성(전체 FAIL 3/4) | `docs/RESULT_rseries_R0R1.md` §2 |
| **R-1** 판별 하네스 | 검색 레벨 correct/wrong/셔플 | correct **8/8**, 스왑 **56/56**, 셔플 마진 붕괴(1/3), 넌센스 0.056 = 이중해리 우리-기질 재현 | 양성(기전) | `docs/RESULT_rseries_R0R1.md` §3 |
| **R-0b** 잔차 어댑터 | 정렬이 진짜인가 상태-운반인가 | canonical **0.742**(혼합 대역 0.60–0.90): ~70% 상태-무관, unseen ~55% 상태-운반. 그리퍼 이벤트 state-free | 부분(혼합) | `docs/RESULT_rseries_R0R1.md` §6 |
| **R-2/R-3** | zero-demo 방향 실행 / retrieval-conditioned decoding | 자산 준비 완료(effect_bank_ours.pkl), **GPU 대기·미착수** | 대기 | `docs/RESULT_rseries_R0R1.md` §4 |

---

## F. Wrist 캠페인 (조건화-측 첫 양성, 타깃-측 종결)

| 셀 | 구조 | SR | paired Δ | c−w | 판정 | 근거 |
|---|---|---|---|---|---|---|
| Phase-B dual-stream(초판) | `DualDeltaAE` main=SigLIP2 + wrist=DINOv3-CLS 별도 인코더, 추론 스트림 | 부분판독 66.7/76.2/80.6 | — | — | 음성(초판) → 감사서 **통계 미성립 재분류**(절단창 비독립·matched baseline 0회 실행) | PROGRESS 2026-07-16, FOLLOWUP §11, `AUDIT_negative_results` |
| **matchedbase**(최초 실행) | — | 87.0 | ref | +75.5 | 재현 정상(종전 비교는 대조군 경로버그로 0회 실행이었음) | `docs/RESULT_wrist_screening.md`, FOLLOWUP §15 |
| **W-A** | DINOv3 wrist 패치 2×2→**4토큰, 조건화-측**, phase1 불변(+7.4M) | **92.5** | **+5.5**(task-CI[−2.5,+14.5]) | +65.5(유보밴드) | **양성(캠페인 최초)** — 이득이 파지태스크 집중(t4 +35) | `docs/RESULT_wrist_screening.md`, FOLLOWUP §15 |
| **W-B** | 측정된 wrist 변위 입력토큰(param 0) | 91.5 | +4.5 NS | +63.0 미달 | 널(W-A 하위호환) | 〃 |
| **W-C** | wrist 변위 **추론**(g_wrist+확장 h+결합 flow, 스케일 표준화) | 82.5 | −4.5 NS | +73.5 정상 | **음성(타깃측 영구 종결)** — 표준화 작동해도 무이득 | 〃 |
| **W-A 확증** | 3seed×2arm×2mode×50roll = **6,000 ep** | base 85.7 → **92.6 = +6.9pp**, paired CI **[+4.9,+9.1] SIG>0** | +6.9 | wristpatch **+63.7** vs base +73.9(+70 미달) | **양성(SR)·미채택**(SR↔언어 tradeoff 새 점) | `docs/RESULT_wrist_confirmation.md`(2026-07-24) |
| PI 제안 판정 | 스트림 분리 정책 / 동적 가중 | — | — | — | 분리=**금지**(교차쌍 h민감도 0.662 JOINT_REQUIRED); 동적가중=**무익**(oracle headroom +0.000025) | `DESIGN_dualpolicy_dynamic_weighting_v1.md`, FOLLOWUP §15 |
| 사전 프로브(week0) | ζ_wrist ablation / ego-motion | ΔR² **0.179**(게이트 9배)=정보 실재; ego-motion 기각(R² 0.177); wrist=그리퍼 채널 상시 우월(R² 0.881 vs 0.725) | GO | `docs/WEEK0_probe_results.md` |

**wrist 서사**: "복잡화 일관 무익"의 **최초 예외 = 조건화-측 wrist 국소 기하**. 삽입점
지도 완성 — 기하는 관측/조건화만 양성, 타깃/코드-측은 백본·스케일·수리 불문 음성.

---

## G. 사전게이트 (2026-07-22, CPU-only, W-A 확증과 동시)

| 게이트 | 질문 | 기준 | 실측 | 판정 | 근거 |
|---|---|---|---|---|---|
| **G-D1** 의도판독 데모 | 정책 ζ̂ 를 adapter 로 읽나 | canonical ≥0.85 | 0.363(재적합 0.786) | **FAIL**(E2 데모 → 오프라인 그림 강등) | `docs/RESULT_pregates.md` §1 |
| **R-D0** W-D 킬게이트 | 미래 Δz_wrist 에 state 이상 신호? | r_full−r_state <+0.02 → KILL | **+0.0227**(아슬아슬) | GO(대기, PI E8 권고 존치) | `docs/RESULT_pregates.md` §2 |
| **R-A′** W-A′ 킬게이트 | SigLIP2 wrist 패치 = DINO 만큼 그리퍼 정보? | 비율 <0.7 → KILL | uplift 1.008(동등) | **GO** — "DINO 기하 특이" 가설 오프라인 기각 | `docs/RESULT_pregates.md` §3 |

---

## H. 용량 스윕 (사전등록, 오프라인, GPU 대기)

| 셀 | 무엇을 | 상태 | 근거 |
|---|---|---|---|
| **capacity sweep** | phase1 g/h hidden 폭 {0.5,1,2,4}× 병목인가(8팔). 판정축=dec R²·cycle R², 게이트: flat이면 무죄·+0.03 단조면 폐루프 개설 | **사전등록 완료·GPU 대기(미실행)**. config 8종 작성됨, `hidden_g`/`hidden_h` 노브 비트 동형 | `docs/PREREG_capacity_sweep.md`, `configs/phase1_libero_large256_cap_*.yaml` |

---

## I. 신뢰성 감사 (2026-07-18, 코드·1차 아티팩트 재검증)

| 감사 | 발견 | 지위 변경 | 근거 |
|---|---|---|---|
| flow 크로스랩 | 콜리그 "actionflow 97.2" = single-seed, 3-seed 96.2≈base 강등; decoder-측 h-flow 양성은 애초 없음. 우리 결함=quantile actnorm 미이식 + `--flow-fixed-noise` 모드락 아님 | actionflow "양성" 오인 정정, `--flow-noise-mode {fresh,walk,locked}` 신설 | `docs/AUDIT_flow_crosslab_2026-07-18.md`, `reports/...` |
| 음성결론 5건 | C1/C2 게이트 ∂L_act/∂α≡0 구조결함(재개방), F3 naive+120ep confound(재개방), wrist dual-stream 통계 미성립(재개방), S1b noalign만 BEST-SHOT | 4건 재개방, 헤드라인 10/10 UNTRACED → **7 EXACT 재계산 + α 7자리 일치**(날조 아님) | `docs/AUDIT_negative_results_2026-07-18.md`, `reports/...` |
| provenance | 로컬 헤드라인 미추적 → 원격 로그 회수 | 하네스에 per-episode JSONL 상시 기록(commit 3f0d984) | PROGRESS 2026-07-18~21 |

---

## J. 정착 파이프라인 결과(참고, README §7 축약)

| 조건 | SR | 근거 |
|---|---|---|
| 2-A, libero_object suite | **90.2%**(SD 0.2) | `experiments/object_5rep.jsonl` |
| 2-A, libero_goal suite | **87.2%** | `experiments/goal_5rep.jsonl` |
| 2-A 표준(손목캠 포함) | **85.2%** | `experiments/baseline_5rep.jsonl` |
| align 손실 L1 교체 | 82.6(−2.6) | `l1_align_loss_5rep.jsonl` |
| 디코더만 z_t 제거 | 81.8(−3.4) | `decoder_nostate_5rep.jsonl` |
| 인코더+디코더 z_t 제거 | 64.6(−20.6) | `encoder_decoder_nostate_5rep.jsonl` |
| 손목캠 토큰 제외(2-B) | 50.4(−34.8) | `wrist_excluded_5rep.jsonl` |
| 2-B + n=0 완전 오토리그레시브 | 39.0 | `serial_n0.txt` |
| 베이스라인 MLP 회귀 | ≈37% | `phase2_libero_mlp.yaml` |

외부 비교: LIBERO-Spatial OpenVLA 84.7(우리 85.2 동급), Object/Goal 은 OpenVLA·Octo·DP
전부 앞섬. 2026 SOTA(MemoryVLA/APT) 96~99 는 격차. 상세 `experiments/README.md`.

---

## 통합 결론 (한 줄)
값은 **언어보존 frozen-Δz 접지**(+75.8~92pp)에 있고, **시각기하는 삽입점이 전부**(관측/조건화
양성, 타깃/코드-측 일관 음성), **SR↔언어 단조 tradeoff**가 지배하며, 복잡한 아키텍처는
폐루프에서 일관 무익 — **유일 예외가 조건화-측 wrist 국소 기하(W-A)**. 상세 서사는
`FOLLOWUP_experiments.md` 통합 결론 + §12–15.
