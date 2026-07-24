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
- **⚠ 한계(정직)**: 롤아웃이 이 박스에서 **CAP4/2/1 모두 27-84ep서 proc-death** → full-400 완주 불가, 3회 부분판독. 매칭 large256-single phase2 ckpt 부재로 동일-task 동시대조는 미수행. 단 3회 부분 모두 66-81%로 baseline을 넘지 못해 **"이득 없음" 결론은 방향상 견고**(uplift였다면 88+ 방향이어야). 정밀 수치는 롤아웃 안정화 후 재검 가능.
- **🔧 불안정 근본원인 진단(2026-07-16)**: 침묵사(Python traceback 없음, ep 7/28/67/84 = 확률적) = **osmesa 소프트웨어-렌더 세그폴트**가 유력. **RAM-OOM 아님**(503GB 중 148GB 여유). **EGL 전환 시도→실패**(`EGLGLContext._context` 없음 = 컨테이너 EGL 디스플레이 부재, import은 되나 렌더 컨텍스트 생성 불가). → 남은 워크어라운드: 죽은 task 재시도-슈퍼바이저(재시도는 ep0 재시작이라 비효율적이나 확률적 크래시라 몇 회 재시도로 완주 가능). 근본수리(osmesa 버전/GL 스택 교체)는 박스 관리자 개입 필요.

## 통합 서사 (Phase-A + Phase-B 종합) — ⚠ §12-15에서 부분 갱신됨
전체 계획(아키텍처 서치 → wrist) 완주. **모든 축에서 동일 결론**: frozen VL-잠재 변위 접지의 값은 **단순 관측-레벨 융합(concat/avg)의 삽입점 + 전처리**에 있고, 디코더/코드-측/역할분리/패치-관측/추가-변위-스트림(wrist) 등 **아키텍처 복잡화는 폐루프에서 일관되게 무익**. 이 강건한 단순성이 프레임워크의 핵심 서사.
*(주의: "전처리 레버"는 §14에서 철회, "wrist 무익"은 §15에서 조건화-측 한정 반전, C1/C2·F3·h-flow 결론 일부는 §12 감사로 지위 변경.)*

---

# 2026-07-18~21 세션 (감사 → 재설계 → wrist 캠페인) — 상세 원문: `reports/2026-07-18_wrist_fusion_session/`

## 12. 신뢰성 감사 — 기존 결론의 지위 변경 (중요)

- **의도**: 문서·콜리그 주장을 그대로 믿지 않고 코드·1차 아티팩트로 재검증 (PI 지시).
- **결과 (지위 변경 목록)**:
  | 기존 결론 | 감사 후 지위 |
  |---|---|
  | "콜리그는 h-flow 계열 양성" | **오인** — decoder-측은 콜리그도 M7 무이득 폐기·Q3 미실행; actionflow 97.2는 single-seed, 3-seed 96.2≈base 자체 강등 |
  | h-flow 37% = "모드락도 실패" | **보류 강등** — `--flow-fixed-noise`는 생성기 전진(모드락 아님). 진짜 모드락은 `--flow-noise-mode locked` 신설로 이제야 검정 가능 |
  | C1/C2 "게이트 미개방 = fine 정보 무가치" | **일반화 철회** — ∂L_act/∂α≡0 배선(액션 손실이 게이트를 열 경로 없음) + CFM target stop-grad 부재 + L_consist가 붕괴 유도. "이 설계 무이득"만 성립 |
  | F3 "richer obs 유해" | **레짐 한정** — KV-LN 부재·init 1.0·규제 전무 naive 모듈 + 120ep 저데이터. full-data 공정판 = §15 W-A가 사실상 수행 |
  | Phase-B wrist "baseline 85-88 하회" | **통계 미성립** — 결정론 롤아웃의 절단창 3개는 독립표본 아님(실효 64/84, CI[66,84] NS) + matched baseline이 경로버그로 **0회 실행**이었음 |
  | 헤드라인 수치 provenance | 로컬 10/10 UNTRACED → 원격 per-task 로그 회수로 **7 EXACT 재계산**, 게이트 α 3종 ckpt 실측 소수 7자리 일치. 하네스에 per-episode JSONL 상시 기록 추가 |

## 13. 언어 축 천장 지도 (오프라인 게이트 배터리, 학습 0)

- **의도**: "언어를 제대로 쓰는가"를 값싸게 판별 + 발산 아키텍처 10후보의 kill-gate 일괄 소진.
- **결과**:
  - **T-0 FAIL**: Δz_text를 h에 직접 주입 → 방향 cos 0.592 ≈ 셔플 문장 0.598 (무우위). 원인 분해 = ① 태스크간 텍스트델타↔이미지델타 방향 대응 부재(argmax 1/10) ② 시간입도 불일치(문장=과제 전체 vs Δz=0.8초). **h·g 배관은 건강**(oracle Δz면 cos 0.985).
  - 발산 포트폴리오 언어군 **전멸**: 개념기저 통화(KILL, h(D·w) drop 0.430), 동작단어 VQ(KILL), 검색 조건화(KILL — 이득이 태스크 암기), 언어 심판자(KILL ≈chance). 다중스케일 계층은 설계 단계 KILL(재부호화 기전).
  - **생존 자산**: **M6-a — modality gap이 변위(Δ)에서 상쇄**(rel_gap 2.505→0.695) 실증 = "Δ-문법" 선험 논거의 정량 증명. 등급-언어 0/6 단조(서수만 ρ=1.0). 사영 프로브 조건부 PASS.
  - week-0 병행: **P-zg RED** — align cos의 ~2/3가 상태 성분, 상태단독 ridge(z_t→Δz)가 g를 상회 → 사전등록 대안(innovation-grounding: align 타깃을 Δz−r(z_t)로 잔차화) 발동 조건 충족. h eff-rank ~5(콜리그 재현).
- **해석**: 현 기질의 언어 사용 = **태스크-선택 수준**(그래서 correct−wrong +76pp는 강함), 청크-수준 벡터 의미교환은 부재. zero-shot 의미교환 주장은 폐기, 잔여 사다리 = 학습 보정기 → 생성형 임베딩(E5-V) 브리지(비용 시 한정). 이 천장 지도 자체가 해석적 기여.

## 14. P-A center-crop — 음성 종결 ("+5pp 전처리 레버" 철회)

- **의도**: DINOv2 격리 실험의 +5pp crop 레버를 채택 구조(avg/concat)에 적용.
- **결과 (20롤 스크리닝, 3팔 × correct/wrong 각 200ep)**: cropavg_dino 91.0(−0.5 NS) / cropconcat_dino 94.0(**−3.5, CI[−7,−0.5] 유의 하락**) / cropavg_both 93.0(+1.5 NS) — **G-cl 전부 탈락, 확정실험 진출 없음**. wrong SR 상승(17.5→25 / 28.5→34.5) = 시각정보 확대→언어의존 감소, tradeoff 축 재확인. 테두리-의존 태스크 국소 하락(사전등록 실패모드 부분 확인).

## 15. wrist 캠페인 — **조건화-측 W-A 첫 양성(+5.5pp), 타깃-측 추론(W-C) 영구 종결**

- **의도**: §11의 "wrist 무익"을 감사(§12) 후 공정 재판 — 문헌수집→내부증거 브리프→통합설계→적대검증(AMEND 5건: GridObs 무음 no-op 차단 등)→구현의 5단 파이프라인.
- **사전 프로브**: ζ_wrist ablation ΔR² 0.179(게이트 9배) = 정보 실재; ego-motion 혼입 기각(R² 0.177); wrist = **그리퍼 채널 상시 우월 관측**(R² 0.881 vs main 0.725, 파지 집중은 1.2배뿐 — "파지에서만"이 아니라 "항상+접근시 격차 확대").
- **결과 (matched 스크리닝, 팔당 400ep — 이 프로젝트 최초의 진짜 matched wrist 비교)**:
  | 팔 | 구조 | SR | paired Δ | c−w | 판정 |
  |---|---|---|---|---|---|
  | matchedbase | — | 87.0 | ref | +75.5 | 재현 정상 |
  | **W-A** | DINOv3 wrist 패치 2×2→**4토큰, 조건화-측**, phase1 불변, +7.4M | **92.5** | **+5.5** (task-CI[−2.5,+14.5] / ep-CI[0,+11]) | +65.5 (유보밴드) | **확증 진출** — 이득이 파지태스크 집중(t4 +35/t9 +25/t3 +10) |
  | W-B | 측정된 wrist 변위 입력토큰(param 0) | 91.5 | +4.5 NS | +63.0 미달 | 널 — W-A 하위호환 |
  | **W-C** | wrist 변위 **추론**(g_wrist+확장 h+결합 flow), 스케일 표준화 수리판 | 82.5 | −4.5 NS | +73.5 정상 | **널 — 타깃측 종결** (표준화는 작동: align 균형 수렴·G2-C 1.4배 개선 — 그래도 무이득 = 스케일 변명 소진) |
- **PI 제안 2건의 측정-선행 판정**: 스트림별 분리 정책 — **금지**(교차쌍 h 민감도 0.662>0.60: h는 두 변위를 결합 코드로 읽음, 분리 예측 쌍은 다양체 이탈). 동적 가중 — **무익**(oracle headroom +0.000025 = kill 기준의 1/400; N3 표준화 공간에서 wrist 상대크기가 파지 직전 +33% 자연 상승 = 원하던 동작을 데이터가 이미 수행).
- **서사 갱신**: "복잡화 일관 무익"의 **최초 예외 = 조건화-측 wrist 국소 기하**. 삽입점 지도 완성형 — 기하 정보는 (i) 관측/조건화에서만 양성(S1 융합, W-A wrist), (ii) 타깃/코드-측은 백본·스케일·수리 불문 일관 음성(C1/C2·S1b·dual-stream·W-C). 진행 중: W-A 확증 50롤×3시드 — 관건은 언어 +70 회복(wrist 기하로 언어 없이 푸는 경로 = F3-방향 리스크 감시).

## 16. 콜리그 retrieval 제어 검증 + C3′ 재편 + R-시리즈 (2026-07-22~24)

*상세: `reports/2026-07-22_retrieval_capacity_session/{ANALYSIS_colleague_retrieval_control,RESULT_rseries_R0R1}.md`.*

- **의도**: 동료 랩(SigLIP@ed54d17)의 "retrieval 기반 제어 = 진짜 wow(언어→검색→실행이 조향을 이김)" 헤드라인이 우리 T-0 사망(§13)과 모순인가 — 코드·1차 JSON으로 재검증(PI 지시, READ-ONLY 무변경) 후 우리 기질로 이식.
- **구현**: (a) 그들 `lang_adapter_wow/`·`banks/*.json`·결과 JSON 코드-레벨 재구성. (b) 어댑터를 **우리 기질**(large256-single ζ=g(a,z_t), SigLIP2-large256 텍스트타워)로 포트 — R-0(오프라인 어댑터 재현), R-1(correct/wrong/셔플 검색 판별 하네스), §6 R-0b(잔차 효과벡터 from-scratch 어댑터). banks 재사용, 코드 `scratchpad/rseries/`.
- **결과**:
  - **기전 정정**: 그들 wow = 벡터 주입이 아니라 **언어 코사인 top-1 → 녹음된 시연 action chunk 오픈루프 재생**(정책·h·Δz 슬롯 미경유). 요구 성질은 방향 대응(T-0의 사인)이 아니라 **판별적 정렬**뿐 + 은행이 서브골 조각 입도라 T-0의 두 사망 원인(방향 대응 부재·입도 불일치)을 구조적으로 우회.
  - **랩 내 이중 해리**: 같은 어댑터로 주입(E5 잔차 조향 camera dir-acc 0.29→0.58, 게이트 0.85 FAIL) vs 검색(8/8) — 실패 경계선이 "정렬 품질"이 아니라 **"주입이냐 인덱스냐"**. 우리 WEEK1(T-0/M6-b 사망 + A5 통과)의 실행-측 확증.
  - **R-0(우리 기질)**: 분할 동료-정확 일치(train 1995 / held-out 497 seg). canonical top-1 **0.972**(그들 0.974) G-R0a PASS, unseen 템플릿 0.952 G-R0c-1 PASS. **우리 독립 3rd셋 0.773 G-R0c-2 FAIL**(평균-emb 0.944) — 실패는 grasp 붕괴(0.04, 94/100 release 흡수)에 집중 = paraphrase-불변성의 **어휘-반경 한계**(학습동사 밖 "unhand/capture/scoot"는 SigLIP2 텍스트기하 임의이웃으로 붕괴). 그들 novel3(0.970)가 통과한 건 학습동사 재조합이라 = 3rd셋 독립성 수준이 결과 좌우.
  - **정직 대조(G-R0b)**: 텍스트-무관 MLP 0.968 ≈ 어댑터 0.972(**분류기 등가** — canonical 정확도는 언어 공로 아님, 고유가치=paraphrase-불변+언어 인터페이스뿐), 상태-잔차화 ζ−r(z_t) 시 0.749/unseen 0.543 급락(A5 조건부 "상태-잔차 대조 의무"가 옳았음).
  - **R-1 검색 판별**: correct **8/8**(마진 0.537) + 스왑 **56/56** vs 셔플 마진 0.185(1/3 붕괴)·넌센스 0.056 = 이중 해리 우리-기질 재현. 셔플 21/24가 원 클래스 유지(SigLIP2 bag-of-words 성향)하되 마진으로 분리 → R-2에 마진-임계 거부 규칙 권고.
  - **R-0b(상태-지분 정량)**: 잔차 어댑터 canonical 0.742 = **혼합 대역** — 우연-상회 신호의 **70% 상태-무관 / 45%만 unseen 보존(=일반화 ~55% 상태-운반)**. 클래스 비대칭: 그리퍼 이벤트(grasp/release)는 잔차화 거의 무손실 = state-free, approach/place(궤적 위상)는 −0.45 상태-의존. 방향 라벨은 z_t 단독 MLP 0.895 = 벤치마크에서 가장 장면-결정적(2차 사전등록 가설 반증 — R-2 방향 실행은 반대쌍 반사실 필수).
- **해석**: T-0 사망이 "C3 전체의 죽음"에서 **"C3-강(zero-demo 벡터산술) 죽음 + C3-약(검색-매개 선택) 경계 확정"**으로 재프레임. C3′ 정직 문장 = *"canonical 판별력은 대부분(70%) 상태-무관 액션 신호지만 일반화의 절반 이상은 장면-상태가 실어 나르고, 유효성은 학습 paraphrase 어휘 반경 안으로 한정되며 반경 밖 실패는 저마진으로 자기표식"*. WEEK1의 A5·M6-a가 각주에서 헤드라인 예언으로 승격. R-2(반대쌍 zero-demo 실행)·R-3(retrieval-conditioned decoding)은 GPU 해제 후 폐루프 다리.

## 17. W-A 확증 최종 + tradeoff 프런티어 (2026-07-24)

*상세: `reports/2026-07-22_retrieval_capacity_session/RESULT_wrist_confirmation.md`. 스크리닝 근거: §15.*

- **의도**: §15 스크리닝의 W-A(+5.5pp, 언어 +65.5 유보밴드)를 사전등록 규칙(SR paired CI>0 **AND** 언어 c−w≥+70pp → "채택 아키텍처")으로 확증 — 관건은 언어 공동기준의 +70 회복.
- **구현**: 3 train-seed × 2 arm(matchedbase/wristpatch) × 2 mode(correct/wrong) = 12 run, 각 libero_spatial 10task × 50rollout = 500ep/run = **총 6,000ep**. phase1 large256-single 공유, phase2 seed{1,2,3} 재학습. MUJOCO osmesa + retry-supervisor. 판정은 유실된 verdict 스크립트 대신 **per-episode JSONL(`outputs/eval/runs/*/episodes.jsonl`)에서 직접 부트스트랩 재계산**(commit 3f0d984 provenance 하네스 자기입증).
- **결과**:
  - **성능(correct SR)**: pooled per-task matchedbase **85.7** / wristpatch **92.6 = +6.9pp**, paired per-task bootstrap 10k **95% CI[+4.9, +9.1] = SIG>0 통과**. per-task 델타 전부 양(t0+3…**t5+11 t7+9 t9+13**) = 이득이 파지·공간 재참조 태스크 집중(설계·스크리닝 패턴 재현), 3-seed 일관(+8.2/+5.6/+7.0), 스크리닝 +5.5와 정합.
  - **언어 공동기준(c−w)**: pooled wristpatch **+63.7pp** vs matchedbase **+73.9pp** — 게이트 +70 미달, 유보밴드(65–75) 하단 65도 **하회**. wrist가 wrong 지시에서도 ~28–30% 성공(base ~10–13%) = **손목 기하로 언어 없이 파지하는 경로**(3-seed 일관 61.6/65.0/64.4).
  - **사전등록 판정**: SR 통과 · 언어 미달 → **W-A는 "제안 아키텍처"로 승격 안 함, SR↔언어 tradeoff 프런티어의 새 점**으로 자리.
- **해석**:
  1. **캠페인 최초의 확증된 양의 SR 아키텍처 추가** — h-flow/actionflow/crop/W-C가 전부 중립~음성이던 것과 대비. "복잡화 일관 무익"의 유일 예외 = **조건화-측 wrist 국소 기하**(삽입점 지도의 마지막 조각).
  2. **SR↔언어 tradeoff 법칙의 4번째 독립 재현**(융합 다이얼·crop·W-A 스크리닝에 이어 확증) = C-2 기여 결정적 강화. 프런티어 확정 점 2개 추가(matchedbase 85.7/+73.9, wristpatch 92.6/+63.7).
  3. **응용 선택 축**: 성능 우선→W-A, 언어 충실 우선→base. 이 선택 축의 존재 자체가 해석적 기여.
  4. **열린 후보 W-A′**(§18): wrist 토큰을 DINOv3→SigLIP2 통일 공간으로 — H-L1(같은 언어 타워라 언어 희석 완화) 성립 시 프런티어를 **언어축으로 미는 유일 후보**.

## 18. 미검정 예약 큐 — W-A′/P-B/W-D/capacity-sweep/demo-redesign (2026-07-22, launch-ready)

*상세: `reports/2026-07-22_retrieval_capacity_session/{DESIGN_WD_WAprime_v1,RESULT_pregates,PREREG_capacity_sweep,PAPER_ARCHITECTURE_v2}.md`. GPU가 W-A 확증에 점유되어 전부 사전등록·CPU 킬게이트까지만 완료, 폐루프 착수 대기.*

- **의도**: PI 잔존 직관·의심을 닫힌 음성 지도(§15 W-C 종결·M-B JOINT_REQUIRED·M-C 무-headroom·Phase-A 프라이어) 위반 없이, 착수-전 킬게이트로 지출을 캡한 상태로 예약. "지는 쪽도 산출물이 있는 실험만 연다" 규율.
- **구현·결과 (사전등록 셀별)**:
  - **W-D "AuxΔw"** (손목 추론의 마지막 미검정 기전 = 손실-측): 미래 손목변위 Δz_wrist(t→t+16, SigLIP2 공간)를 ctx 트렁크 보조 헤드로 예측 — **h/액션 경로에 단 1비트도 미진입**(재부호화 논증·M-B 무관). 학습-전 킬게이트 **R-D0 GO 아슬아슬**(r_full−r_state +0.0227 ≥ +0.02, 마진 0.0027 — 초과신호 거의 a_emb=g 임베딩). E8 "보류/킬" 권고 존치(어느 기여도 W-D 미요구).
  - **W-A′ "SigPatch"** (기전 귀속 + 의미 접근성): W-A의 DINOv3 wrist 패치를 SigLIP2-large256 패치로 **인코더 정체성 1개만 교체**(파라미터-정확 매치). 킬게이트 **R-A′ GO 동등**(SigLIP2 uplift +0.0459 ≈ DINO +0.0456, 비율 **1.008** → "W-A 이득=DINO 기하 특이" 오프라인 기각, 손목 패치정보 인코더-불문). 언어 양가설 사전등록(H-L1 희석완화/H-L2 의미간섭, 판별 readout=t0/t2/t4 wrong-모드 95/85/65 기준). §17 확증이 "SR 승리·언어 미달"로 끝났으므로 **W-A′가 결정 셀로 승격**(언어 회복 유일 저가 후보).
  - **P-B LangSelPool(B1)** (관측측 patch 활용): 텍스트-쿼리 patch pooling(kv-LN/pos-emb/tok+group drop/attn-entropy 로깅, F3 결함 전면 수정판) 구현·8단 스모크 PASS(commit 093cc1b). 롤아웃은 `instruction_for(tid)` 쿼리로 wrong/blank 판별평가에 언어 인과 유지. B2는 STUB(사전등록 조건부). 폐루프 대기.
  - **g/h 용량 스윕** (PI "phase1 용량 부족?" 종결): DeltaAE hidden_g/hidden_h 독립 노브(기본 비트동형) + config 8종(폭 0.5–4×, g/h 귀속 분리) + probe_h_jacobian. 판정=전팔 ±0.01 무죄(영구 종결) / (4×−0.5×)≥+0.03 단조 폐루프 개설. 오프라인 4–8 GPU-h, launch-ready.
  - **의도-판독 데모 재설계** (G-D1 FAIL 대응): 정책 ζ̂ top-1 0.363(데이터 ζ와 cos 0.188, 학습시 lat_cos 0.203과 정합 = 버그 아닌 **분포이동+입도**), ζ̂-재적합 후 0.786 < 0.85 → **E2 데모 "오프라인 판독 그림"으로 강등**(PAPER v2 §5 E2 조항 발동). 부분 구제 수치: 마진-게이팅 0.856@커버리지80%, place∪release 병합 7→6클래스 0.871 — PI 재가 시 제한 데모 부활. `adapter_zetahat.pt`가 어느 판이든 필수 기반.
- **해석**: 논문 삽입점 지도의 마지막 공란 = "손실-측"(W-D 결과로 충전, 널이면 타깃/관측/손실 3-기전 소진 = "wrist 추론" 종결 각주). W-A′ 결과 = 기전 귀속 문장(DINO 기하 vs 인코더-불문 공간 디테일) 승패 무관 수록 + H-L1 실측 시 프런티어를 안쪽으로 미는 유일 점. 전 셀 착수-전 CPU 킬게이트로 반론("복잡화 0승인데 또 여는가")의 지출을 ≤1 GPU-h로 캡.
