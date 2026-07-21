# 연구 포지셔닝 보고서: Frozen VL-Embedding Displacement (Δz) 정책의 문헌 지형과 기여 주장 (2026-07-18)

*작성: Claude Code 리서치 에이전트. 모든 arXiv ID는 조사 세션에서 소스 검증, 불가 항목 UNVERIFIED 명시.*

## 1) Latent Action Model 지형

| 논문 (ID) | 잠재공간 | 비전 인코더 | 액션 디코딩 | 언어 분석 |
|---|---|---|---|---|
| Genie (2402.15391) | 이산 codebook(\|C\|=8) | from scratch | 없음 | 없음 |
| LAPA (2410.11758) | NSVQ 이산 | quantizer 학습 | real-action head 재학습 | unseen instruction 일반화(적대적 probe 없음) |
| Moto (2412.04445) | VQ-VAE(128) | frozen ViT+M-Former | co-finetune MLP | CALVIN 분류 probe만 |
| IGOR (2411.00785) | VQ(N=4,\|C\|=32) | **frozen DINOv2** IDM | low-level policy | counterfactual video 정성 |
| villa-X (2507.23682) | VQ(32)+proprio FDM | LAM 학습 | joint flow-matching | 최소 |
| UniVLA (2505.06111) | task-centric 이산(DINOv2 공간 FDM) | 사실상 동결 | LLM latent→디코더 | instruction-sensitivity ablation 없음 |
| UniAct (2501.10105) | 범용 이산 primitive | VLM 튜닝 | embodiment별 MLP | 없음 |
| GO-1 (2503.06669) | LAPA류 planner | LAM 학습 | flow expert | 없음 |
| CoMo (2505.17006) | **연속** latent motion | 학습 | pseudo-action | 없음 |
| GR00T N1 (2503.14734)/N1.5 | codebook 없음 | N1.5: VLM 전체 동결 | flow-matching DiT | N1.5 language-following 46.6→93.3%(FLARE 2505.15659) |

기타: GR-2(2410.06158, LAM 아님), Seer(2412.15109), AdaWorld(2503.18938), DreamVLA(2507.04447), CLAM(2505.04999), From Pixels to Tokens(2605.04678 — frozen DINOv2 LAM, 이산>연속 +2.7%), **LARY(2604.11689 — LAM 품질 DINOv3>SigLIP)**, LatBot(2511.23034).

### 1.2 "Frozen 임베딩 displacement" 최근접 이웃 — **핵심: 우리 조합은 미발표**
frozen VL(언어정렬) 임베딩의 Δz 자체를 액션 표현으로 삼는 2-stage(Δz-AE + latent FM)는 2026-07 기준 미발표. 근접 순서:
1. **WALA (2607.11397)**: frozen DINOv3 미래 delta를 LAM 디코더 *재구성 타깃*으로 — delta가 액션 표현은 아님, vision-only, 언어 부재. **스쿱 위협 1순위.**
2. **StaMo (2510.05057)**: "state token 차이=latent action" — 인코더가 학습형(frozen VL 아님).
3. **Delta-JEPA (2606.31232)**: latent difference action decoder — 명시적으로 frozen 회피, FM 없음, 언어 없음.
4. DINO-WM(2411.04983)/V-JEPA 2-AC(2506.09985): 절대 잠재 + MPC 플래닝 — Δz·디코더·언어 없음.
5. ALAM(2605.10819): transition+action joint FM — self-trained 공간.
6. VITA(2507.13231): latent AE+FM — latent가 action-AE이지 임베딩 변위 아님.
7. DynaFLIP(2605.30350): Δz 정책 아님 — 인코더 사전학습 백본 논문.
8. Genie류 "LAPP": UNVERIFIED(2504.15472는 preference-RL; LAPA 혼동 추정).

**언어 분석 공백**: LAM 전체에서 wrong-instruction/counterfactual probe 수행 사례 전무. RoboSemanticBench(2606.02277): VLA 과제실패의 ~96%가 잘못된 타깃 선택 — 우리 biased 2.5%와 정면 대비 가능.

## 2) Flow/Diffusion Action Head 폐루프 안정성

- 헤드: Diffusion Policy(2303.04137, "jittery action" 명명), ACT(2304.13705, temporal ensembling), π0(2410.24164)/π0.5(2504.16054), VITA(2507.13231, ICLR'26 — 실재 검증), Consistency Policy(2405.07503), Streaming DP(2406.04806), OneDP(2410.21257), ManiCM(2406.01586), Streaming Flow Policy(2505.21851 — 마지막 실행 액션에서 flow 시작=연속성 구조적).
- 실패 모드 공식 명명: **BID(2408.17355)** "oscillate between strategies"+reactivity-coherence tradeoff 형식화; RTC(2506.07339) chunk 경계 pause/jerk; Legato(2602.12978) "spurious multimodal switching"; PF-DAG(2602.21684) "mode bouncing"; 이론 2605.22493 — 분리된 모드는 저-Lipschitz transport로 표현 불가 → **모드 선택을 잠재변수로 분해하고 준결정적으로 디코딩하라**.
- 해법 계열: ① 평균화(ACT TE — BID가 "모드 간 평균 유해" 실증) ② 거절샘플링/가이드(BID, SGAD 2508.12189, ACG 2510.22201) ③ **inpainting/prefix 조건화 = 현재 대세**(RTC, training-time RTC 2512.05964, Legato, A2C2 2509.23224 +7pp LIBERO-Spatial, REMAC 2601.20130) ④ 노이즈-공간 일관성(**PAINT 2606.19774** — 이전 chunk를 backward-Euler 역산해 초기 노이즈 선택 = "노이즈 고정" 아이디어의 성숙형; RTI 2508.05396, RA-DP 2503.04051, D3P 2508.06804, STEP 2602.08245) ⑤ 증류(지연만 해결).
- **우리 접점**: "compact latent 생성+준결정 디코딩=시간일관성" 라인 형성 중(VITA, CoLA-Flow 2601.23087, PF-DAG, LAFP 2606.10517, 이론 2605.22493) — **누구도 frozen VL 공간 latent를 안 씀**. 우리 ζ-FM→h 결정 디코딩이 이 처방과 정확히 일치 → BID식 oscillation 지표로 폐루프 모드-스위칭 감소를 측정하면 저비용 부수 기여. (경고 2605.22493: 과도 정규화 latent는 모드 정보 상실 — 반례 실험으로 방어.)
- UNVERIFIED: "A careful examination of action chunking"(정확 제목, LBM 2507.05331 혼동 추정), "SmoothOperator".

## 3) 언어 접지 평가·해석가능성

### 3.1 벤치마크 좌표계
- **LIBERO-CF (2602.17659)**: Grounding Rate/SR 분리. OpenVLA-OFT: CF 4.7%GR/0.4%SR vs biased 83.6/78.6; π0: 28.8/9.6 vs 63.0/45.0; π0.5: 30.8/13.2 vs 65.6/60.9. → 우리 "biased 2.5%"의 공인 좌표계.
- **ICBench (2603.06001)**: "linguistic blindness" 격리, 해법 IGAR(train-free).
- **LIBERO-Plus (2510.13626)**: 언어 섭동 SR 하락 OpenVLA −49.7pp, π0 −33.2pp, 강한 OFT계 −14.0~−17.4pp; **blank-instruction에서 성능 거의 불변 = instruction blindness 결정적 증거**.
- **"전형 VLA ≤+19pp" 상수는 단일 출처 없음 → 폐기.** 안전한 서술: "공표된 correct-vs-perturbed 격차 ~0–50pp; 최신 강력 VLA는 paraphrase ≤~17pp, blank ≈0pp(LIBERO-Plus); CF grounding 4.7–30.8%(LIBERO-CF)". 우리 +75~92pp는 이 좌표계에서 4-5배 격차.
- 보조: VLATest(2409.12894), INT-ACT(2506.09930), CAST(2508.13446), LangGap(2603.00592), shortcut(2508.06426), InSpire(2505.13888), BYOVLA(2410.01971), RoboArena(2506.18123), ReSteer(2603.17300), 2606.11906.

### 3.2 해석 도구
- **SpLiCE (2402.10376)**: CLIP 임베딩→희소 개념 분해, training-free, **Δz 직접 적용 가능**. **TextSpan (2310.05916)**: 의미 축 분해. MaskCLIP(2112.01071), CLIP Surgery(2304.05653).
- 델타 산술: StyleCLIP(2103.17249); **SIMAT (2112.03162) — 경고: vanilla CLIP delta transport는 잘 안 되며 가벼운 정렬 파인튜닝으로 크게 개선** → Δz_text↔Δz_image 치환에 보정 단계 필요 예고. **PC-CLIP (2409.09721)**: 이미지쌍 임베딩 차이↔차이-설명 텍스트 정렬 = "Δz에서 언어 읽기"의 원시 연산.
- 이미지-차이 캡셔닝: CLIP4IDC(2206.00629), VIXEN(2402.19119), OmniDiff(2503.11093). "CLIP-Diff" UNVERIFIED.
- 기하: LRH(2311.03658), Platonic(2405.07987 fm), vec2vec(2505.12540), task arithmetic(2212.04089 fm). CKA 선례 VLA-Trace(2605.30117), 주의 2210.16156.
- 무시연 언어→행동: **SuSIE (2310.10639)** — 언어→편집 이미지 서브골→정책(픽셀 공간; 우리는 잠재 직행), RT-2(2307.15818 fm), BC-Z(2202.02005 fm), UniPi(2302.00111 fm), **OTTER (2503.03734)** — frozen CLIP+text-aware pooling = "frozen VL 의미가 제어에 충분"의 최강 기존 근거.

### 3.3 Semantic Interchange 입증 패키지 (권장 조합 = ①+③)
① **Δz_text 치환 zero-demo 실행**(헤드라인): Δz_text = E_text(목표문)−E_text(현상태문)을 Δz 슬롯에 주입, 미시연 과제 롤아웃. 지표: 신규 과제 SR + LIBERO-CF Grounding Rate(기존선 4.7-30.8%의 2배 목표). 보정은 PC-CLIP 방식, 실패 대비는 SIMAT delta-calibration.
② **역방향 읽기**: 롤아웃 중 Δz(t) 캡셔닝(IDC류), 지시문 일치도가 시간에 따라 단조 증가·CF 지시에서 발산하는 곡선.
③ **기계적 증명**: SpLiCE로 Δz 궤적 개념 분해 — 지시된 물체/목표 개념만 상승, spurious 개념 평탄, object-swap 시 활성 개념 스왑. TextSpan 의미축 + MaskCLIP 공간 소재. → shortcut-learning 비판의 직접 해독제.
④ **기하적 증명**: 과제쌍별 Δz_text↔Δz_image CKA — matched 높음/mismatched 낮음.
⑤ 대조군: blank/paraphrase(LIBERO-Plus 프로토콜) — 우리는 전형 VLA(0-17pp)보다 훨씬 크게 무너져야 하며 그것이 역설적 양성.

## 4) 기여 주장 후보 5건

- **C1 [아키텍처]** "frozen 언어-정렬 임베딩 공간의 변위 Δz 자체를 액션 접지 표현으로 삼는 최초의 조작 정책(Δz-AE + latent FM 2-stage)". 최근접: WALA/StaMo/Delta-JEPA/UniVLA/DINO-WM/VITA/ALAM. 차별: 변위=액션표현 그 자체 + 언어정렬 공간(언어접지 공짜 상속) + FM prior. 입증: SigLIP2-Δz vs DINOv3-Δz vs 학습 인코더-Δz head-to-head(LARY의 "DINOv3>SigLIP"이 언어접지 지표에선 역전됨을 보여야 완성). 위험: **스쿱 높음**(WALA→SigLIP 한 걸음). "to our knowledge, first … language-aligned" 한정 표현.
- **C2 [평가]** "counterfactual에서 biased 2.5%(VLA 45-79%), correct−wrong +75~92pp — 추론시 개입 없이 아키텍처만으로". 최근접: LIBERO-CF/ICBench/LIBERO-Plus/OTTER(사후 완화 vs 구조적 상속; OTTER는 CF 지표 부재). 입증: **자체 probe를 LIBERO-CF Grounding Rate·ICBench 프로토콜로 재측정**(공인 좌표계 전환이 채택률 좌우) + blank + GPT paraphrase. 위험 낮음.
- **C3 [테제]** "언어-델타·이미지-델타·액션의 상호 교환(Δz_text 주입 zero-demo 실행 + Δz 역방향 언어 복원)". 최근접: SuSIE(픽셀 경유)/RT-2(기전 불명)/PC-CLIP/SIMAT/OTTER. 차별: 교환 공간·연산(벡터 치환)이 명시적·검사가능. 입증: §3.3 패키지. **위험 최고** — vanilla delta transport 실패 가능(SIMAT), 보정기 필요 시 "공짜 상속" 서사 약화 → 보정기 크기·데이터 요구 정직 보고, C1·C2와 분리 성립 구조로.
- **C4 [설계 지도]** insertion-point map. 최근접: Prismatic(2402.07865 — 관측측 channel-concat 원조, DINOv2+SigLIP 유의/DINOv2+CLIP 비유의 = pairing 특이; VLM 벤치선 동결>파인튜닝), OpenVLA(2406.09246 — fused vs SigLIP-only 폐루프 ablation 원문 부재 UNVERIFIED), Eagle/Eagle-2(2408.15998/2501.14818 — 최종 레시피서 DINOv2 탈락), Spatial Forcing(2510.12276 — 깊이 스윕이지 삽입점 아님), DeepVision-VLA(2603.15618 — 심층 주입 이득 **반례**), OpenVLA-OFT(2502.19645 — FiLM 유무가 언어 following 결정 = "삽입 방식이 스트림 사용을 결정" 선례), VER(2510.05213), AutoFly(2602.09657). 판정: **관측 concat 자체는 표준(신규 아님); 신규는 latent-변위 맥락 + 비관측 삽입점 체계 음성 + frozen 유지 + 언어축 동시 보고.** 단독 논문보다 C1 설계 원리 섹션으로 흡수 권장. 방어: 우리 음성은 "심층 주입 일반"이 아니라 "Δz 공간의 언어 정렬을 깨는 삽입은 실패" 가설로 통합.
- **C5 [다이얼]** SR↔언어-민감도 단조 tradeoff — 한 모델의 연속 다이얼(기존 문헌은 모델 간 비교 진단뿐). 입증: 융합 비율 3-5점 스윕, y=(SR, CF-gap, Grounding Rate). 위험 낮음(점 수 확보 필요).

**권장 패키징**: 주 논문 = C1+C2+C4(흡수)+C5, 헤드라인 실험 = C3 ①+③. 테제 한 문장: *"언어-정렬 frozen 공간의 변위는 언어·이미지·행동이 교환 가능한 공용 통화이며, 이 통화를 훼손하지 않는 삽입점에서만 기하 정보가 이득이 된다."*

## 5) SigLIP계+DINO계 융합 신규성 점검
관측 수준 concat 융합 자체는 Prismatic/OpenVLA 표준 — 신규 아님. C-RADIOv4(SigLIP2-g+DINOv3+SAM3 증류)의 로봇 정책 채택 UNVERIFIED. Theia(2407.20179). 우리 insertion-point map과 정확히 겹치는 발표물 없음.

## 부록: UNVERIFIED/주의
"CLIP-Diff"·"SmoothOperator"·"A careful examination of action chunking"(정확 제목)·Genie류 "LAPP": 미발견/혼동. OpenVLA fused-vs-SigLIP-only 폐루프 delta: 원문 미보고. "≤+19pp" 상수: 출처 없음. from-memory ID: RT-2, UniPi, BC-Z, Platonic, task arithmetic, RoboCLIP(2310.07899). **2510 이후 ID 다수는 최신 프리프린트 — 투고 직전 WALA/StaMo/Delta-JEPA/CoLA-Flow 인접 재검색 필수(주 단위로 움직이는 영역).**
