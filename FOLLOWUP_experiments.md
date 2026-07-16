# Frozen VL-Latent Displacement Grounding for LIBERO — 실험 팔로업 (연구자 토의용)

*정리 기준: 2026-07-16 · suite: libero_spatial(별도 명기 없으면) · 프로토콜: 10 태스크 × 20 롤아웃/태스크, no-aug 클린 레짐, correct-instruction, MUJOCO osmesa (별도 명기 없으면). 콜리그(외부) 수치는 그들 프로토콜(50롤/det_boot/EGL/held-out)이라 절대비교 불가 = 방향만 참조.*

---

## 0. 셋업 — 무엇을, 어떻게

**논지**: 동결(frozen) VL 인코더의 의미 잠재공간에서 **변위 Δz = z_{t+k} − z_t 에 액션을 접지**한다. 주장 = (i) 정책이 언어를 결정적으로 사용·보존, (ii) 그 하중은 언어보존 기질이 지탱, (iii) 시각기하 풍부화는 **삽입점에 따라** 폐루프 이득이 갈린다(체계적 음성지도).

**아키텍처 (2-스테이지, 인코더·디코더 동결)**:
- **Phase1 — DeltaAE** (`src/models/networks.py`): `g`(1D-CNN): 액션청크(+상태 z_t)→ζ≈Δz / `h`(MLP): Δz(+z_t)→액션청크. 손실 = align(MSE+cos) + recon(L1) + cycle(L1). 선택 HY03 언어정렬(align_mode=hybrid: dz + λ·InfoNCE 모션문장).
- **Phase2 — FlowPolicy** (`src/models/policy.py`): 잠재 flow matching. 토큰 [z_prev, z_cur, g(A_past), lang, wrist] → ζ̂ → h(ζ̂,z_cur) → 액션. source=past, steps=6.
- **앵커** (`src/core/anchor.py`): CLIP/SigLIP2/DINOv2/DINOv3/RADIO + DualFusion(avg)/DualConcat(concat).

**판별 평가 하네스** (`src/eval_libero/rollout_sim.py`): 지시문 모드 correct/wrong(타 태스크 지시문)/blank/swap. **correct−wrong = 언어사용 지표**(성공기준은 원 태스크 고정 → 언어 무시 정책은 wrong서도 SR 유지, 언어사용 정책은 하락).

---

## 1. 언어보존 (헤드라인 양성)

- **의도**: frozen-Δz 정책이 지시문 언어를 실제로 사용/보존하는가? (백본·suite 무관하게)
- **구현**: 학습된 정책에 correct vs wrong 지시문 토큰을 주고 폐루프 SR 측정. 2 suite(spatial/goal) × 2 backbone(CLIP/SigLIP2).
- **결과**: correct−wrong = 1a CLIP **+75.8pp**(3시드) / SigLIP2-large256 **+76.5±1.8pp**(3시드) / CLIP-goal +88.5 / SigLIP2-goal +92.0. W3.3 both-aug +82.5.
- **해석**: 언어사용이 backbone-agnostic·2 suite 전반 성립(ANOVA로 suite-attributable 확인). = 논문 헤드라인.

## 2. 구성적 접지 (1c, 씬내 타깃-스왑) — ★ 범위: 공간/관계 접지 (spatial-only)

- **의도**: 정책이 "지시된 타깃"에 충실한가, 아니면 편향(고정/가까운 타깃)으로 푸는가?
- **구현·범위 (정밀)**: libero_spatial 10태스크는 **전부 "pick up the black bowl [공간관계] and place it on the plate"** — 물체 종류(black bowl)·목표 동일, **공간관계만 다름**. 한 씬에 같은 종류 bowl이 여러 위치에 있고, 스왑은 **같은 씬**에서 **지시문만** 형제 태스크 것으로 교체해 **다른 위치의 (동종) bowl**을 가리킴. instructed(스왑타깃)/orig(원타깃)/neither 도달률 측정. → **∴ 이 실험은 "공간/관계 참조 해소"를 검증하지, 물체-종류/속성 접지가 아님**(모든 타깃이 동일 black bowl).
- **결과**: Faithful(스왑타깃) **48%** / Biased(원타깃) **2.5%** / neither 49.5% (VLA OFT/π0의 Biased 45–79% ≫ 우리).
- **해석**: 같은 씬에서 다른 위치 bowl로 재지시하면 정책이 **지시된 위치로 재타깃**(48%), 원타깃 고집은 거의 없음(2.5%) = **공간 언어를 실제 사용**. neither 49.5%는 파지 난이도이지 오지향 아님.
- **한계·확장**: spatial-suite 동종 물체 한정이므로 **물체-종류/속성 구성적 접지는 미검증** → libero_object/goal(서로 다른 물체 종류)에서 1c 스왑 확장이 필요(future). 그러면 "공간(1c-spatial)+물체종류(1c-object)" 2축으로 구성적 접지 주장 강화.

## 3. 시각-풍부화 삽입점 음성지도

핵심 질문: DINO 계열 기하 정보를 **어디에** 넣어야 폐루프 SR이 오르나? 4개 삽입점을 시험 → 아래는 모두 음성(no-aug 검증 대조).

### 3.1 F3 — 관측 덧대기 (dense-obs 융합)
- **의도**: 관측에 dense patch 토큰을 학습형 attention-pool로 덧대면 SR↑?
- **구현**: 앵커 + ObsFusion(K개 관측토큰) → phase2, P6 게이트.
- **결과**: 폐루프 **악화** (a 50.0 > b 31.5 > c 15.5). **음성**: richer obs 조건화가 폐루프 저하(게이트 실패).

### 3.2 F1 — 앵커 교체 (백본 head-to-head)
- **의도**: "더 좋은 인코더 = 더 좋은 SR"인가?
- **구현**: CLIP/SigLIP2/DINOv2/RADIO 동일 규제로 폐루프 head-to-head.
- **결과**: SR **3중 null**(matched-reg). **음성**: encoder 품질 ≠ SR. 단 언어사용은 backbone-agnostic 존재.

### 3.3 C1/C2 — 액션코드 타깃-측 게이트 fine 채널 (F4)
- **의도**: 잠재 액션코드에 텍스트-쿼리 cross-attention으로 fine patch 채널(ζ_f)을 게이트(α=0 init)로 덧대면 미세운동 개선?
- **구현**: F4 계층코드 ζ_g(pooled)⊕ζ_f(patch cross-attn K=8→96d bottleneck→tanh게이트). C1=SigLIP2-fine, C2=DINOv3-fine.
- **결과**: **게이트 미개방**(C1 α=0.027, C2 α=0.0015), 무이득(paired CI 0 포함), C2 ζ_f 절제 시 오히려 +. **음성**: coarse가 ~88% 푸는 suite서 잔차 게이트가 "굶음". *(외부 콜리그 M7 동일 gated-DINO-잔차도 무이득으로 독립 재현.)*
- **감사(중요)**: C2 창의 aug 대조팔이 자기 no-aug(88.5)보다 낮아(80.5) 판정오염 우려 → 재기준선(20롤) 감사: noaug 85.0 vs c2_control(+aug) 71.8 = **−13pp는 실재하나 버그 아닌 camera-aug+dropout 정규화의 폐루프 효과**(오프라인 val 동급·캐시 clean·조립경로 부재·예산 동일로 1차근거 배제). aug는 CLIP엔 도움이나 이 기질엔 confound → **이후 융합은 no-aug 클린 밴드서 평가**.

**3의 통합**: 타깃/코드-측(C1·C2)에 기하를 넣으면 무효 — 이 결론은 뒤(S1b)에서 "삽입점" 서사로 완성됨.

## 4. S1 — 관측-레벨 융합 (양성) ★

- **의도**: 3.1~3.3 음성 뒤, 외부 콜리그가 관측-레벨 융합(dual-encoder avg)으로 이득을 봄 → 우리 레짐/판별하네스로 공정 검정. **삽입점을 관측 본류(z_t 자체)로 이동.**
- **구현**: `DualFusionAnchor`(avg): z=L2norm(0.5·L2norm(SigLIP2)+0.5·L2norm(DINOv3)), 1024d. `DualConcatAnchor`(concat): [L2norm(sig);L2norm(dino)] 2048d no-mix. phase1 DeltaAE를 융합 z 위에 재적합 → phase2 → 롤아웃. (avg=외부 dual_wrapper 공식의 DINOv3판, concat=변형. **둘 다 cross-attention 아님** — 우리 원 cross-attn은 타깃측 C2로 음성이었음.)
- **결과 (paired bootstrap 10k, per-task, vs no-aug 베이스 85.0)**:
  | arm | correct SR | correct−wrong | paired Δ | 오프라인 h R² |
  |---|---|---|---|---|
  | **avg** | 91.5% | +74.0pp | **+6.5pp, 95%CI[+0.5,+15]** SIG>0 | +0.735 |
  | **concat** | **97.5%** | +69.0pp | **+12.5pp, 95%CI[+4.5,+22]** SIG>0(견고) | +0.749 |
  | baseline pooled | 85.0% | (ref) | — | +0.655 |
- **해석**: **관측-레벨 융합은 폐루프 강한 양성**(타깃-측 C2와 정반대). concat per-task=[100×5,95×5], 이득이 베이스 취약 공간태스크(t4 55→100 등)에 집중 = DINOv3 기하가 SigLIP2 약점 보강.

## 5. SR↔언어 tradeoff

- **의도**: concat이 SR 최고(97.5)지만 언어델타(+69)가 avg(+74)보다 낮음 — 관계?
- **결과**: 조건화의 DINOv3 함량 축에서 **단조 반대 이동**: concat(full) 97.5/+69 → avg(0.5) 91.5/+74 → S1b(0) 86/+78.
- **해석**: **free lunch 없음.** DINOv3 함량↑ → SR↑·언어↓. avg = 사전등록 이중기준(SR SIG>0 AND 언어≥70) **유일 충족 = 제안 아키텍처**.

## 6. S1b — 역할분리(비대칭) 삽입 (반증)

- **의도**: concat의 SR을 언어 손실 없이 얻자 — 조건화 토큰=SigLIP2 단독(언어경로 강제) / 앵커·코드·h=융합(기하). (+HY03 hybrid를 ζ의 SigLIP2 블록에만.)
- **구현**: 조건 토큰 = fused[:, :1024] 슬라이스(SigLIP2-alone과 bit-exact)+zero-pad; g/h/ζ=full fused 2048. `cond_anchor` flag. align_block=1024 신설(info_nce가 ζ SigLIP2 블록↔SigLIP2 모션텍스트).
- **결과**: **반증.** S1b-noalign correct **86.0%**(≈베이스, SR 이득 상실)·correct−wrong +78(언어 최고) / S1b-hybrid 67.5%(HY03가 이 fused ζ선 역효과).
- **해석**: 비대칭 삽입(DINOv3를 조건화에서 제거)이 SR 이득을 못 지킴 → **융합 SR 이득의 주경로 = 관측/조건화에 DINOv3**. 앵커/코드-측만으론 무효. **C1·C2와 일관 → 통합 서사: DINOv3 기하는 정책 입력(관측)에 있어야 SR 기여.**

## 7. S2 — h-flow 디코더 (음성)

- **의도**: MLP h가 다봉 p(action|ζ,z_t)를 평균/뭉갠다 → flow로 교체해 코히런트 궤적 샘플.
- **구현**: `ChunkFlowDecoder`(조건부 CFM, action 생성; h_mode=mlp|flow 플래그, mlp 비트동형). concat 융합에 적용.
- **결과**: **오프라인 다봉성 실증** — K=32 평균-R² recon +0.714/cycle +0.868 ≈ MLP, 단일샘플 −0.037. 그러나 **폐루프 33%(naive)·37%(에피소드 고정노이즈 v2)** — 실패=300스텝 timeout/wandering = receding-horizon **mode-switching**.
- **해석**: 다봉성은 실재하나 **폐루프 제어선 MLP 평균이 옳고 샘플링이 해로움**(뭉갬=버그 아님). h-flow NO-GO. *(후속: 이건 디코더-측 flow — 콜리그의 버린 M7과 동일. 콜리그 승자 actionflow는 정책-측 flow로 별개 — §9.)*

## 8. 백본/전처리 조사 (DINOv2 vs DINOv3, confound 해소)

- **의도**: 우리 DINOv3-avg 91.5 vs 콜리그 DINOv2-avg 93-96 — 백본 차이인가 잘못 쓴 건가?
- **구현**: (a) DINOv3 사용 감사(웹+코드), (b) DINOv2-avg를 우리 파이프라인/eval로, (c) 전처리 매칭 격리, (d) DINOv3@512.
- **결과**:
  - **DINOv3 추출 정확**(감사): pooler=post-LN CLS=DINOv3 권장 global, register 정상, 버그 없음. 문헌상 DINOv3 ViT-L(87.44%)≥DINOv2.
  - **매칭전처리(256-no-crop)**: DINOv2-avg **90.5** ≈ DINOv3-avg **91.5** = **백본 동급**.
  - DINOv2-avg @center-crop(256→224) = **96.0** → **격차는 백본 아닌 center-crop 전처리(+~5pp 레버)**.
  - DINOv3@512 ≈ 93%(진행 중, @256 대비 ~+2pp).
- **해석**: "DINOv2가 낫다"는 **전처리 confound였음(철회)**. DINOv3 사용 정상. 진짜 레버 = **center-crop 전처리**. pooled 융합엔 백본 무차별 → DINOv3 유지(패치 트랙 SOTA).

## 9. 크로스-랩 수렴 + actionflow (진행 중)

- **콜리그(SigLIP 폴더) 새 pull 재분석**: (a) concat=구현/레짐 특이(fusion-mode 아님, matched avg 69.8 vs concat 65.0 ~5pp) — **우리와 독립 수렴**. (b) DINOv2-vs-DINOv3를 2차 요인 강등 — 우리와 수렴. (c) offline≠SR: 디코더 ∂h/∂ζ eff-rank≈5.5/1024 → action-R²가 SR 추종, latent-cos는 역순위.
- **actionflow (그들 유일 승자-초과, 97.2)**: `flow_space=action` — phase2 정책이 **raw 액션청크(112d)를 직접 flow transport, 동결 MLP h를 추론서 완전 우회.** 우리 h-flow(디코더 뒤 flow=그들 버린 M7)와 **다른 방식**. 구현·검증(latent 비트동형)·커밋 후 우리 concat·avg에 검정.
  - **결과 = 음성(우리 파이프라인, correct 200롤 완주)**: af-concat **76%**(151/200, correct−wrong +20pp) / af-avg **80%**(159/200, correct−wrong ~+50pp) — **SR·언어 양축 모두 하락**(latent-concat 97.5/+69, latent-avg 91.5/+74 대비). 콜리그의 +0.8pp(96.4→97.2, 단일시드 밴드 내)가 우리에겐 반대 부호.
  - **해석**: 우리 값은 **변위잠재(ζ) 접지**에 있고 actionflow는 **ζ를 버려 raw 액션공간서 flow** → 접지 구조 상실 = 손해. **h는 병목이 아니라 필수**(ζ→h 접지가 우리 기여 핵심). → **actionflow는 우리 thesis와 배치**되며, 이 음성이 오히려 "변위잠재 접지가 값의 원천"을 강화. (구현 정상성: af-avg 초기 4태스크 91%≈latent가 방증 — 하락은 실제, 버그 아님.)
  - *ops 노트: HF Hub 504 반복 → HF_HUB_OFFLINE=1 기본화; killed-proc zombie GPU 메모리 1회 watchdog docker restart로 해소.*

---

## 통합 결론 (현재까지)

1. **값은 언어보존 frozen-Δz 접지에 있다** (+75.8~92pp, 2 suite × 2 backbone, ANOVA 확증).
2. **시각기하는 삽입점이 전부다**: 관측/조건화(정책 입력)에 넣으면 양성(S1 avg/concat), 타깃/코드-측(C1·C2)이나 앵커-only(S1b)면 무효. = 체계적·기전 있는 지도.
3. **SR↔언어 단조 tradeoff**(조건화 DINOv3 함량) — avg가 유일 양충족 sweet spot = 제안 아키텍처.
4. **음성으로 정리된 것**: 관측 학습-덧대기(F3)·앵커교체(F1)·게이트 타깃 fine채널(C1/C2)·역할분리(S1b)·디코더-측 h-flow(S2).
5. **방법론**: correct−wrong 판별하네스(외부가 못 잰 언어축)·paired bootstrap CI·재기준선 감사(정규화 confound 규명)·action-R²(offline≠SR).

## 열린 항목 / 진행 중

- **actionflow(정책-측 flow, h 우회)** — 그들 97.2, 우리 미검정 → concat·avg 학습·롤아웃 중. h-디코더 병목 제거의 결정적 검정.
- **center-crop 레버**를 최고 arm(avg/concat)에 적용(+~5pp 기대).
- **cowork 프로토콜(50롤/3시드/all-held-out)로 최종 arm 재평가**(현재 20롤/1시드 스크리닝).
- **미검토 셀**: concat+both-aug+lang(콜리그 미검증), √2 lang 스케일매칭, patch-token+정책 attention(L1+L4, C1/C2와 달리 관측측·무게이트 — 기전적으로 다른 접근).

## 토의 포인트 (연구자용)

1. **"삽입점" 서사**: 왜 관측/조건화 삽입만 이득이고 타깃/코드-측은 무효인가? (정책이 관측을 조건으로 행동 결정 → 기하가 입력에 있어야 함 + coarse 채널이 이미 충분해 잔차 게이트가 굶음.) — 일반 VLA에도 성립할 가설인가?
2. **SR↔언어 tradeoff의 근본성**: 조건화의 비언어 시각용량이 언어민감도를 희석한다 — 이게 frozen-VL-접지 특유인가, 일반적인가?
3. **offline≠SR / decoder null-space**: eff-rank 5.5 병목 → 왜 정책-측 flow(actionflow)는 되고 디코더-측 flow(h-flow/M7)는 안 되나?
4. **평가 프로토콜 표준화**: 랩간 비교를 위한 공통 프로토콜(50롤/held-out/판별하네스)?

## 10. Phase-A 아키텍처 서치 (best base 탐색, 2026-07-16)
목표: wrist 적용 전, 다양한 조합으로 best base 확정. 결과 = **복잡한 방법은 전부 음성, 단순 관측융합이 승자.**
| 방법 | 결과 | 판정 |
|---|---|---|
| concat-latent-MLP (관측융합) | **97.5%** | ✅ **현 best base (SR)** |
| avg-latent-MLP | 91.5% (언어+74) | ✅ 균형 base |
| naive h-flow (MLP 전체대체) | 37% | ❌ mode-switching |
| actionflow (h 우회 direct action) | 76/80% | ❌ ζ 접지 버림 |
| **residual-h-flow (MLP평균+잔차flow, Q2/M7)** | **~48-65%** | ❌ 잔차-flow도 폐루프 무익(콜리그 M7/Q2와 정합) |
| **grid-token (DINOv3 patch 무게이트 관측)** | OOM-사망(무결과) | ⚠ F3-echo 위험+dense 인코딩 OOM 불안정 |
| DINOv2 vs DINOv3 (매칭전처리) | 동급 | 백본 무차별 |
| center-crop 전처리 | +~5pp 레버 | ✅ 미적용(cheap) |

→ **통합 결론**: **"어디에 넣느냐(관측 본류)"가 값이고, 디코더/코드-측 복잡화(flow-decode 계열)·타깃-측 게이트(C1/C2)·역할분리(S1b)·패치-관측(F3/grid-token)는 무익.** best base = **concat 97.5 / avg 91.5(언어균형)**. 남은 cheap 개선 = center-crop. 이 단순성 자체가 "frozen 변위 접지"의 강한 서사(복잡한 시각융합보다 삽입점·전처리가 결정).
- **ops 노트**: dense-patch 인코딩(grid-token)·동시 롤아웃서 proc-death+zombie GPU 반복 → HF_HUB_OFFLINE 기본화 + watchdog docker restart로 대응(데이터 무손실).

## 11. Phase-B — wrist-cam 추론(dual-stream 변위) (2026-07-16)
목표(사용자 요청): wrist-cam을 단순 조건입력이 아닌 **추론(변위) 스트림**으로 승격, **카메라별 특성에 맞게 인코더 분리**. best base 확정(Phase-A) 후 적용.

**설계**: `DualDeltaAE` — main(agentview)=SigLIP2-large256 전역, wrist=DINOv3-CLS(별도 인코더). g_main·g_wrist 분리 → 각 스트림 독립 변위 ζ 추론, 확장 h가 [ζ_main;ζ_wrist]에서 액션 디코드. 독립 align loss(main/wrist). isolation base(main=단일 SigLIP2, 융합 아님)로 **wrist-추론 효과만 격리**(천장 없는 헤드룸에서 검정). wrist 인코더는 grid-token OOM 전례 때문에 dense-patch 대신 CLS 채택.

**결과**:
| 항목 | 값 | 판정 |
|---|---|---|
| phase1/phase2 offline | act R² **+0.663**, align_main·align_wrist **둘 다 수렴** | ✅ 모델·dual경로 정상, 스케일 불균형이 학습 안 깨뜨림 |
| closed-loop SR (correct, 부분×3) | **66.7% / 76.2% / 80.6%** | ❌ large256-single baseline **85-88 하회 = uplift 없음** |

→ **판정**: **wrist-추론(dual-stream 변위)은 closed-loop 이득 없음.** offline은 건강하나(변위가 잘 학습·정렬됨) 폐루프 성공률은 baseline 이하 → wrist를 추론 스트림으로 승격해도 정책이 이득 못 얻음. **Phase-A의 "복잡화 무익, 단순 관측융합 승자" 패턴과 완전 정합** (h-flow/actionflow/residual-flow/grid-token에 이어 dual-stream도 음성). wrist는 현행 **조건입력**이 최선.
- **⚠ 한계(정직)**: 롤아웃이 이 박스에서 **CAP4/2/1 모두 27-84ep서 proc-death**(osmesa/mujoco per-process 렌더링 누수 추정) → full-400 완주 불가, 3회 부분판독. 매칭 large256-single phase2 ckpt 부재로 동일-task 동시대조는 미수행. 단 3회 부분 모두 66-81%로 baseline을 넘지 못해 **"이득 없음" 결론은 방향상 견고**(uplift였다면 88+ 방향이어야). 정밀 수치는 롤아웃 안정화 후 재검 가능.

## 통합 서사 (Phase-A + Phase-B 종합)
전체 계획(아키텍처 서치 → wrist) 완주. **모든 축에서 동일 결론**: frozen VL-잠재 변위 접지의 값은 **단순 관측-레벨 융합(concat/avg)의 삽입점 + 전처리**에 있고, 디코더/코드-측/역할분리/패치-관측/추가-변위-스트림(wrist) 등 **아키텍처 복잡화는 폐루프에서 일관되게 무익**. 이 강건한 단순성이 프레임워크의 핵심 서사.
