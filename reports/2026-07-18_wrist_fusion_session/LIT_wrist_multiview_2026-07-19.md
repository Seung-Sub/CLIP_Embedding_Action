# LIT — wrist/multi-view 문헌 수집 (wrist v2 Stage 2 설계 근거)

*작성 2026-07-19. 목적: `DESIGN_wrist_v2.md` §Stage 2 후보(4a EE-frame 변위 — ego-motion
프로브 기각 R²=0.177로 강등 / 4c 국소 기하 패치 토큰 — 1순위 / 4b 국면 게이트 / 4d h-상태)에
대한 찬반 증거 수집. 모든 arXiv ID는 웹검색/arxiv.org 페이지로 대조했고, 미확인은
**UNVERIFIED** 명기. 검증 기준: arxiv.org에서 ID·제목 일치 확인 = VERIFIED.*

*우리 파이프라인 맥락 요약: frozen VL 임베딩 변위(Δz)를 액션 화폐로 쓰는 LIBERO 정책.
wrist는 현재 조건 토큰 1개(SigLIP2, ablation −34.8pp)로만 기여. Phase-B dual-stream
(ζ_wrist를 flow 타깃에, DINOv3-CLS 조건 교체)은 INCONCLUSIVE-널. 실측: Δz_main/Δz_wrist
std 비 6.5×, DINOv3-CLS wrist 스트림 오프라인 ΔR²=0.179, ego-motion 프로브 R²=0.177.*

---

## 1. SOTA 조작 정책의 wrist/eye-in-hand 카메라 사용 방식

### 1.1 아키텍처별 wrist 투입 방식 (융합 토폴로지)

| 정책 | ID (검증) | wrist 투입 방식 | 융합 토폴로지 |
|---|---|---|---|
| ACT/ALOHA | **2304.13705** VERIFIED | 4캠(top, front, 좌·우 wrist) 각각 ResNet18 → 피처를 트랜스포머 인코더 시퀀스로 | **뷰별(per-view) CNN 인코더 + 토큰 시퀀스 concat, 단일 트랜스포머가 교차-뷰 어텐션** (mid fusion) |
| Diffusion Policy | **2303.04137** VERIFIED | 뷰마다 **독립 ResNet-18**(비사전학습, spatial softmax, GroupNorm)로 인코딩 후 latent **concat** | **per-view 인코더 + late concat** — 교차-뷰 어텐션 없음 |
| Octo | **2405.12213** VERIFIED | workspace/wrist 이미지를 각각 토큰화, 트랜스포머 입력 토큰으로 삽입; 카메라 구성이 로봇마다 달라도 "어떤 토큰을 넣느냐"만 바꿈 | **shared 토크나이저 스택 + 토큰-레벨 융합**; wrist 데이터는 사전학습의 27%뿐 |
| π0 / π0.5 | **2410.24164 / 2504.16054** VERIFIED | 임베디먼트별 가변 뷰 수(1 base + 1–2 wrist)를 **모든 프레임의 이미지 토큰을 한 시퀀스로 concat** 후 언어 토큰과 이어붙임 | **shared ViT(VLM 백본) over views + 토큰 concat** (early-mid fusion) |
| OpenVLA-OFT | **2502.19645** VERIFIED | 3인칭 + wrist 이미지를 **같은 SigLIP+DINOv2 융합 인코더**에 통과, 토큰을 LLM 컨텍스트에 concat; **FiLM**으로 언어 임베딩이 각 ViT 블록의 시각 피처를 scale/shift 변조 | **shared dual-encoder + 토큰 concat + 언어-FiLM**. 논문 주장: 멀티뷰(특히 ALOHA) 입력에서 FiLM 없으면 **spurious 시각 피처에 과적합** — 언어 조건이 뷰 융합의 안정화 장치 |
| GR00T N1 / N1.5 | **2503.14734** VERIFIED | 멀티뷰 RGB를 Eagle-2 VLM(SigLIP-2 인코더)이 프레임당 64토큰으로 인코딩 → DiT 액션 모듈이 cross-attention | **shared VLM 인코더 + 후단 DiT cross-attn** (System2/System1 분리). N1.5: VLM 동결 + FLARE(미래 latent 정렬 목적함수) 추가 |
| RoboFlamingo | **2311.01378** VERIFIED | static + gripper 카메라 RGB를 OpenFlamingo VLM에 투입, policy head가 이력 모델링 | **shared VLM 인코더 + resampler/cross-attn** |

**공통 패턴**: 2023 이후 대세는 **"shared 인코더 over views + 토큰-레벨 concat, 융합은
트랜스포머 어텐션에 위임"**. per-view 별도 인코더는 ACT(경량 CNN)·Diffusion Policy 계열의
소형 정책에서만 표준. **뷰별로 다른 백본을 쓰는 SOTA 정책은 발견하지 못함** (§2.3 참조 —
우리 설계의 차별점이자 선례 부재 리스크).

→ **우리 설계 함의**: 우리의 phase2 조건 토큰 열(agentview 토큰 + wrist 토큰)은 정확히
"토큰 concat + 정책 내부 어텐션" 계열로, SOTA 토폴로지와 정합. 4c(wrist 패치 4토큰 추가)는
이 계열 안의 보수적 확장이며 토폴로지 신설이 아님 — 선례 리스크 낮음.

### 1.2 wrist 기여를 정량화한 ablation (핵심 증거)

- **robomimic "What Matters in Learning from Offline Human Demonstrations"
  (2108.03298, VERIFIED)**: wrist 관측 제거 시 성공률 **10–45pp 하락** (예: 73.3→43.3%).
  BC 계열 조작에서 wrist가 최대 단일 관측 기여라는 가장 인용되는 정량 근거.
  → 우리 −34.8pp ablation과 크기·방향 일치. 우리 수치가 이상치가 아니라 문헌 표준 범위.
- **"Vision-Based Manipulators Need to Also See from Their Hands" (2203.12677,
  VERIFIED, ICLR 2022, Hsu et al.)**: hand-centric(eye-in-hand) 단독이 3인칭 단독보다
  학습 효율·OOD 일반화 모두 우수 — 단 **hand-centric 가시성이 충분할 때만**. 3인칭을
  같이 쓰면 학습엔 필요하나 OOD를 해침 → **3인칭 스트림에 variational information
  bottleneck을 걸어** 해결(스트림-비대칭 규제의 원형).
  → 함의 (i) wrist는 "보조"가 아니라 종종 주 스트림, (ii) **스트림별 비대칭 규제**(한쪽만
  IB/dropout)의 직접 선례 — DualDeltaAE에서 main/wrist에 다른 규제 강도를 주는 근거.
- **ACT/ALOHA (2304.13705)**: 4캠 전부 사용이 기본; 논문 자체는 캠별 ablation 표를
  제공하지 않음(정밀 양팔 태스크에서 wrist 2캠 포함이 설계 전제).
- **OpenVLA-OFT (2502.19645)**: ALOHA류 멀티뷰 입력 확장 시 FiLM 부재가 실패 요인이라
  보고 — **"뷰가 늘수록 언어 접지가 흐려진다"**는 우리 §5 언어 희석(correct−wrong)
  tradeoff와 같은 현상의 다른 표현.
  → 4c 반증 조건에 correct−wrong 기준을 둔 것은 문헌과 정합. wrist 토큰 추가 시 언어
  변조(우리로 치면 언어 조건 경로) 보존을 함께 감시해야 함.
- **어느 국면이 이득인가**: 문헌 정설은 "접촉·삽입·정밀 파지 국면, mm-수준 시각 피드백이
  성패를 가르는 구간"(robomimic 계열 + Look Closer의 정밀 태스크 이득). 단, 국면별로
  분해한 정량 ablation(approach vs grasp)은 **주요 논문에서 직접 보고된 것을 찾지 못함**
  — 우리 Stage 0 프로브 4(파지창 집중도)가 문헌 공백을 메우는 측정이 됨.

---

## 2. 제어를 위한 multi-view 표현 학습

### 2.1 교차-뷰 일관성/예측 손실

- **MV-MWM "Multi-View Masked World Models" (2302.02408, VERIFIED, Seo et al.
  ICML 2023)**: 멀티뷰 마스크드 오토인코더가 **무작위로 마스킹된 시점의 픽셀을 다른
  시점으로부터 재구성**(cross-view prediction) → 그 표현 위에 world model. 뷰 랜덤화
  강건성, 카메라 무보정 sim2real.
  → 함의: 교차-뷰 예측은 "표현 학습" 단계 장치. 우리는 인코더 동결이라 직접 이식 불가하나,
  **Δz_main↔Δz_wrist 상호 예측 가능성(교차-뷰 변위 일관성)을 보조 프로브/손실**로 쓰는
  아이디어의 근거. 예측 불가능한 잔차 = 뷰-고유 정보라는 분해 논리도 여기서 나옴.
- **Look Closer (2201.07779, VERIFIED, Jangir et al. RA-L 2022)**: 3인칭+wrist를
  **cross-view attention**(뷰 간 상호 spatial attention)으로 융합한 RL. hammer 태스크
  75% vs 단순 멀티뷰 38% vs 단일뷰 13% — **융합 방식 자체가 2배 차이**를 만든 드문 정량.
  → 함의: 우리 조건 토큰은 정책 내부 self-attention에 융합을 맡기는데, Look Closer는
  "명시적 교차-뷰 어텐션"이 단순 병렬 투입보다 낫다는 증거. phase2에서 wrist 토큰과
  agentview 토큰 간 어텐션이 실제로 형성되는지(어텐션 맵 진단)가 값싼 후속 체크.
- **TCN "Time-Contrastive Networks" (1704.06888, VERIFIED, Sermanet et al. 2017)**:
  동시각 다중 시점을 임베딩에서 끌어당기고 시간 이웃을 밀어내는 triplet — **뷰-불변
  (view-invariant) 표현의 고전적 원형**.
  → 함의: 뷰-불변만 추출하면 wrist 고유 정보(파지 잔차)는 정의상 버려진다 — 우리가
  원하는 것은 반대(뷰-**고유** 성분 보존)임을 명확히 해주는 대조 준거.

### 2.2 뷰-불변 vs 뷰-고유 인수분해

- **VILA "Learning to Act Robustly with View-Invariant Latent Actions"
  (2601.02994, VERIFIED-제목, 2026)**: 전이 패턴(dynamics) 기반 뷰-불변 latent action.
- 멀티뷰 표현을 **view-consistent / view-specific 인자로 분해해 별도 학습**하는 2단계
  패러다임은 멀티뷰 분류 문헌(예: MVFD 2501.06524, VERIFIED-제목)에는 확립되어 있으나,
  **조작 제어에서 "view-specific 스트림을 따로 정렬(align)하는" 직접 선례는 확인하지
  못함** — 우리 DualDeltaAE의 align_main/align_wrist 분리는 이 문헌 공백 지대에 있음.
  → 함의: ζ_wrist를 "뷰-고유 성분 전용 채널"로 해석하는 우리 프레임은 이론 문헌과
  정합하지만 제어 실증 선례는 없다 — 논문화 시 novelty 겸 리스크로 서술할 것.

### 2.3 뷰별 인코더 선택(view-specific encoder choice) — 선례 조사

- **직접 선례(뷰마다 다른 frozen 백본) 발견 실패.** 조사 범위에서 모든 SOTA 정책은 뷰 간
  shared 인코더(§1.1) 또는 동일 아키텍처 복수 인스턴스(ACT/DP). "3인칭=의미 인코더,
  wrist=기하 인코더" 형태의 **백본 이질(heterogeneous backbone per view) 설계는 논문
  수준 선례를 찾지 못함** (UNVERIFIED-부재: 전수조사는 아님).
- 가장 가까운 것: ① OpenVLA(2406.09246, VERIFIED)의 SigLIP+DINOv2 **채널 융합** — 단
  이는 **같은 이미지**에 두 인코더(멀티-인코더 단일-뷰)이지 뷰별 분담이 아님. ablation은
  "융합 인코더 이득 존재하나 데이터 다양성 효과보다 작음". ② Diffusion Policy류의 뷰별
  독립 ResNet — 아키텍처는 같고 가중치만 다름. ③ DINOBot(2402.13181, VERIFIED-제목):
  **wrist-camera 관측에 DINO-ViT 피처**로 검색·정렬 — "wrist 근접 뷰에는 DINO 기하
  피처"라는 우리 week-0 발견(DINOv3 wrist ΔR²=0.179)과 방향 일치하는 방증.
  → 함의: 우리의 "SigLIP2(main, 언어 정렬) + DINOv3(wrist, 기하)" 뷰별 기질 분담은
  **문헌상 신규 조합**. 지지 방증(§3.3 DINO 근접뷰 우위 + DINOBot)은 있으나 직접 검증
  선례가 없으므로, 4c의 c0(SigLIP2 유지 + DINOv3 병기)처럼 **점진 추가**로 검정하는
  현 설계가 리스크 관리상 옳다.

---

## 3. Eye-in-hand 특수성

### 3.1 ego-motion 처리

- **기전 문헌**: wrist 뷰의 프레임간 변화 = ego-motion(카메라 이동) ⊕ 장면 변화. 이를
  명시 분해하는 조작-정책 논문은 드묾. 가장 가까운 계열:
  - **camera/EE-frame 액션 표현** 계열 — Qwen-RobotManip Technical Report
    (**2606.17846**, VERIFIED — 설계문서의 UNVERIFIED 2건 중 1건 확정): 모든 EE 액션을
    **camera-frame delta pose**로 통일 + 카메라 기하를 positional encoding으로 액션
    전문가에 주입 — "시각적으로 비슷한 액션 = 수치적으로 가까운 액션" 논리를 명시.
    cVLA (**2507.02190**, VERIFIED-제목): camera-space VLA. From Fixed to Free Cameras
    (**2607.05396**, VERIFIED-제목): 액션을 **카메라 좌표계에서 native 예측** 후 hand-eye
    행렬로 base-frame 변환. Astribot (**2507.17141**, 기검증): EE-frame delta 우위.
  - **경고(설계문서 정정)**: 설계문서가 UNVERIFIED로 남긴 **2512.11218은 camera-frame
    delta pose 논문이 아님** — 실제로는 "Seeing to Act, Prompting to Specify: A Bayesian
    Factorization of Vision Language Action Policy" (VLA의 베이지안 인수분해, VERIFIED).
    KICKOFF의 해당 인용은 **오귀속**이므로 설계문서에서 삭제/교체 요망.
  → **4a에 대한 판정**: 문헌은 "액션을 카메라 프레임에 정렬하면 좋다"를 지지하지만,
  이들은 전부 **액션 표현** 논문이지 "frozen 임베딩 변위의 ego-motion 오염" 논문이 아님.
  우리 Stage 0 프로브가 **ego-motion이 Δz_wrist를 R²=0.177밖에 설명 못함**을 실측한
  이상(G3b 기각), 4a의 전제(ego 지배)는 우리 기질에서 성립하지 않음 — 문헌 지지가
  프로브 기각을 뒤집을 수 없다. **4a 강등은 문헌과 모순되지 않음** (문헌은 액션 좌표계
  이야기, 우리 기각은 임베딩 변위 통계 이야기 — 층위가 다름).
- **action-conditioned view prediction**: GR-1류 비디오 예측 정책, EnerVerse-AC
  (2505.09723, VERIFIED-제목, 액션 조건 환경 생성), Ego-PM (2508.19852, VERIFIED-제목,
  손 궤적 조건 egocentric 예측) — "액션으로 예측 가능한 성분을 분리"하는 우리 ego-잔차
  릿지(Ê(Δz|운동학))와 같은 정신. 단 전부 픽셀/비디오 공간이지 frozen-latent 공간이 아님.
- **WristWorld (2510.07313, VERIFIED-제목)**: 4D world model로 anchor 뷰에서 **wrist
  뷰를 생성**(wrist 데이터 결손 보완). wrist 뷰가 anchor 뷰+기하로 상당히 복원
  가능하다는 것 자체가 "wrist 고유 정보는 잔차에 있다"는 우리 가설의 간접 방증.

### 3.2 파지-국면 조건부 가중/게이팅 (4b 관련)

- 학습 정책에서 **wrist 입력을 국면에 따라 명시 게이팅하는 논문은 확인하지 못함**
  (UNVERIFIED-부재). 존재하는 것: ① 고전 FSM(approach/grasp 상태 전환) 제어, ② HoMeR
  (2506.01185, 기검증)의 국면별 액션 모드 전환(장거리 absolute/근거리 relative EE,
  +29.17pp) — "국면에 따라 **표현을** 바꾼다"는 원리 선례이나 게이팅 대상이 관측이 아니라
  액션. ③ RT-2류 shared-attention 정책이 국면별로 뷰 가중을 **암묵 학습**한다는 관찰
  (블로그/분석 수준, UNVERIFIED).
  → 함의: 4b(비학습 게이트 w(t))는 선례 빈약 + 우리 −34.8pp(상시 기여 큼) → 설계문서의
  "低 승산" 등재가 문헌과도 정합. 어텐션이 이미 soft 게이팅을 학습한다면 명시 게이트는
  중복이라는 반론이 문헌 구조상 우세.

### 3.3 근접(close-up) 뷰 통계와 frozen 인코더 선택 (4c 지지 증거)

- **DINO 계열의 dense/국소 우위는 다수 검증**: DINOv2/v3는 patch-level 목적함수로 dense
  피처가 강하고 (DINOv3 2508.10104, VERIFIED), CLIP 계열은 전역 정렬 목적함수라 공간
  정밀도가 낮다는 벤치마크 다수 (예: frozen segmentation Jaccard DINOv2 0.42 vs CLIP
  0.34; 인스턴스 구별에서 DINOv2 우위; SOCO 2605.31597 대응점 벤치마크에서 DINO 계열
  최상). "wrist 근접뷰 = 텍스처·객체 중심, 언어 의미 희박" 통계에서 DINO가 유리하다는
  **직접 조작 실험은 없으나** 표현 성질 증거는 일관.
  → **4c 핵심 지지**: wrist 뷰에서 CLS(전역 요약)가 아니라 **패치(국소)** 를 쓰라는 것,
  그리고 그 패치는 DINO 계열에서 뽑으라는 것 — 둘 다 표현 문헌 방향과 일치. 우리 week-0
  DINOv3-wrist ΔR²=0.179도 같은 방향. 설계문서의 "CLS는 wrist 뷰에서 거의 상수적 전역
  요약" 주장은 DINO CLS의 객체-중심 emergent 특성 문헌과 합치.
- **주의(4c 반대 방향)**: OpenVLA ablation은 융합 인코더(기하 추가) 이득이 "존재하나
  작음"이라 보고 — 기하 피처 추가의 한계효용은 문헌에서도 크지 않다. +2pp CI 반증
  기준은 현실적.

### 3.4 폐색-구동 뷰 선택

- Viewpoint-Agnostic Manipulation Policies with Strategic Vantage Selection
  (**2506.12261**, VERIFIED-제목), Imagination at Inference: Synthesizing In-Hand Views
  (**2509.15717**, VERIFIED-제목 — 폐색 시 in-hand 뷰를 **합성**해 정책 입력 보강),
  VistaBot (2604.21914, VERIFIED-제목). 능동 시점 선택 계열(Active Vision 2409.17435).
  → 함의: 우리 세팅(고정 2뷰)에는 직접 해당 없음. "폐색 때 wrist가 유일 정보원"이라는
  일반 논거로만 인용 가치.

---

## 4. 다중-스트림 latent 모델의 손실 배치

### 4.1 스트림별 정렬 손실 + 스케일 균형

- **스트림별 latent 정렬 손실을 명시 분리한 조작 논문은 확인하지 못함** — 우리
  align_main/align_wrist 분리는 선례 없음(§2.2와 동일 공백). 스케일 균형의 표준 도구는
  일반 MTL 문헌: **uncertainty weighting (1705.07115, Kendall et al., VERIFIED-표준)**,
  GradNorm, 그리고 **per-stream 통계 정규화**(디퓨전/flow 문헌의 per-dim x0 정규화 관행).
  → **함의(직접 처방)**: 우리 실측 std 비 6.5×는 concat-ζ에 단일 스칼라 x0_std를 쓰는 현
  구현에서 wrist 블록을 사실상 6.5× 과소가중한다. 문헌 표준 처방은 (i) **스트림별
  표준화**(Δz_wrist / σ_wrist) — 학습 파라미터 0, (ii) 손실 자동 가중(uncertainty
  weighting)은 그 다음. Stage 1 "표준화 필수" 설계와 문헌 정합.
- **gradient conflict**: PCGrad (2001.06782, VERIFIED-표준), CAGrad 등은 **멀티태스크**
  용이며, **멀티-뷰 스트림 간 충돌에 적용한 조작 논문은 확인하지 못함**. 우리 상황(align
  cos 손실 2개 + flow 손실 1개)에 선제 도입할 근거 부족 — 스케일 표준화가 먼저.

### 4.2 스트림별 보조 디코더빌리티 손실

- **BC-Z (2202.02005, VERIFIED)**: 보조 목적함수(개루프 궤적 예측) 관행의 원형.
- FoAR류: 접촉 상태 예측기를 보조 손실로(force 문헌). 미래 gripper 2D keypoint 예측
  보조 손실(one-shot imitation 계열 2011.05970).
- **wrist 스트림 → gripper/z축 서브스페이스 전용 감독**의 직접 선례는 확인하지 못함.
  → 함의: "ζ_wrist에서 gripper-dim 액션을 릿지로 디코딩"(Stage 0 프로브 W1)을 학습
  손실로 승격하는 것은 보조-손실 관행의 자연 연장이나 신규 조합. phase1에 붙일 경우
  param-0 프로브에서 이득이 먼저 보일 때만(설계문서 게이트 논리 유지).

### 4.3 뷰 dropout / 정보 병목 (강건성)

- **Akinola et al. "Learning Precise 3D Manipulation from Multiple Uncalibrated
  Cameras" (2002.09107, VERIFIED)**: **sensor dropout을 명시 사용** — 배치 시 카메라
  손실에 강건. (검색 요약의 "1뷰 제거 −4~19%, 2뷰 제거 −53~73%" 수치는 원문 대조 실패
  — **UNVERIFIED 수치**, 인용 금지.)
- **Hsu et al. (2203.12677)**: 3인칭 스트림에만 **variational IB** — 비대칭 스트림 규제 +
  OOD 개선의 가장 깨끗한 선례 (§1.2).
- Octo류 대규모 사전학습은 wrist 결손 데이터(27%만 wrist 보유)를 토큰 마스킹으로
  자연 처리 — 사실상의 뷰 dropout.
  → 함의: DualDeltaAE 재도전 시 **wrist-stream dropout**(학습 중 확률적 ζ_wrist 0화)은
  선례 있는 정칙화이며, 부수 효과로 "wrist 기여의 학습-중 zero-ablation 곡선"이 공짜로
  나와 진단 가치가 있음. 단 Hsu 결과는 방향이 반대일 수 있음을 경고 — 병목을 걸 대상은
  "지배적이고 spurious한 쪽"(그들에겐 3인칭)이지 약한 쪽이 아님. 우리에겐 main이 지배
  스트림이므로, 병목/dropout을 main에 걸어 wrist 활용을 강제하는 변형이 문헌 정신에
  더 가깝다.

---

## 5. CLIP-계열 + DINO-계열 융합 — multi-VIEW 맥락

- **멀티-인코더 단일-뷰(우리 관심의 전 단계)는 확립**: OpenVLA/Prismatic
  (2406.09246 / 2402.07865 VERIFIED-제목) SigLIP+DINOv2 채널 concat; Eagle-2(GR00T
  백본) 멀티-인코더; **Theia (2407.20179, VERIFIED)**: CLIP·DINOv2·ViT 등 복수 VFM을
  단일 소형 모델로 **증류** — 교사들보다 나은 로봇 학습 성능, "다양한 시각 지식의 결합이
  로봇 표현에 이득" 테제의 최강 증거.
- **멀티-뷰에서 뷰별로 CLIP/DINO를 분담시킨 논문은 확인 실패** (§2.3과 동일 결론;
  UNVERIFIED-부재). 주변부: WristWorld(2510.07313)가 wrist 뷰 **생성**에 CLIP 의미
  피처를 조건으로 사용; MV-Actor (2606.10899, VERIFIED-제목) 멀티뷰 의미/공간 정렬;
  DINO Eats CLIP (2604.19432, VERIFIED-제목, 3D 검색 — 조작 아님).
  → **함의**: "S1 concat 기질 상보성(우리 실증) + Theia(단일뷰 다기질 이득) + DINO
  근접뷰 우위(§3.3)" 세 근거를 합치면 4c-c0(SigLIP2 wrist 토큰 유지 + DINOv3 wrist
  병기)는 문헌적으로 정당화 가능한 **최소 신규 조합**이고, 이것이 성공하면 그 자체가
  "view-specific encoder assignment"라는 빈 문헌 셀을 채우는 기여가 됨.

---

## 6. Design ingredients shortlist

| Ingredient | 증거 강도 | 핵심 출처 | 우리 파이프라인 삽입 지점 |
|---|---|---|---|
| wrist 조건 토큰 유지·강화 (조건화 삽입점) | **강** (10–45pp / −34.8pp 상호 재현) | 2108.03298, 2203.12677, 우리 R4 | phase2 조건 토큰 열 — 현행 유지, 4c로 강화 |
| wrist에 DINO-계열 **패치**(국소) 표현 | **중** (표현 성질 다수 + DINOBot 방증, 조작 직접 증거 없음) | 2508.10104, SOCO, 2402.13181, week-0 ΔR²=0.179 | **4c**: DINOv3 패치 2×2 pool → 조건 토큰 4개 |
| 기질 상보(CLIP류+DINO류) 병기 | **중** (단일뷰에선 확립, 멀티뷰 분담은 선례 없음) | 2406.09246, 2407.20179, S1 concat 실증 | **4c-c0**: SigLIP2-wrist + DINOv3-CLS-wrist 병기 |
| 스트림별 Δz 표준화 (σ별 나눗셈) | **중-강** (MTL/디퓨전 정규화 표준 관행; 6.5× 실측) | 1705.07115 계열 관행 | Stage 1: ζ_wrist/σ_wrist, x0_std 스트림별 분리 |
| 언어 변조 보존 감시 (뷰 추가 시) | **중** (OFT: FiLM 없으면 멀티뷰 과적합) | 2502.19645 | 4c 반증 기준의 correct−wrong 이중 기준 유지 |
| main-스트림 병목/dropout (지배 스트림 규제) | **중** (IB로 OOD 개선 선례; 방향 주의) | 2203.12677, 2002.09107 | DualDeltaAE 재도전 시 main-측 dropout/IB |
| 교차-뷰 어텐션 형성 진단 | **중** (융합 방식이 2배 차 사례) | 2201.07779 | phase2 어텐션 맵 검사 (비용 ~0) |
| wrist→gripper-dim 보조 디코더빌리티 손실 | **약-중** (보조손실 관행 연장, 직접 선례 없음) | 2202.02005, 2011.05970 | phase1 align_wrist 옆 보조 릿지-헤드 (W1 프로브 승격) |
| EE/camera-frame 액션 재표현 (4a) | **약(우리 기질 한정)** — 문헌 지지는 액션 표현 층위, 우리 프로브 R²=0.177이 전제(ego 지배) 기각 | 2606.17846, 2507.02190, 2607.05396, 2507.17141 | 4a — **강등 유지**; 액션-측 실험으로 재분류할 때만 부활 |
| 국면 게이트 (4b) | **약** (명시 게이팅 선례 없음, 어텐션 암묵 학습이 통설) | HoMeR(원리만), FSM 고전 | 4b — 低 승산 등재 유지 |
| 교차-뷰 변위 일관성 프로브 (Δz_main↔Δz_wrist 상호 예측) | **약-중** (MV-MWM 논리의 latent 전이, 신규) | 2302.02408 | Stage 0 확장 프로브 (param-0, 뷰-고유 성분 정량화) |
| ζ_wrist를 flow 타깃(코드-측)에 재투입 | **반대 증거 우세** (우리 지도 무효 + 문헌도 조건화/토큰 융합이 주류) | Phase-B, 삽입점 지도 | 재도전 시 Stage 1 표준화 선행 필수 |

**검증 실패/오귀속 기록**: ① **2512.11218 ≠ camera-frame delta pose** (실제: Bayesian
Factorization of VLA Policy) — 설계문서 §0.3 인용 교체 필요. ② Akinola sensor-dropout
정량 수치 UNVERIFIED. ③ "뷰별 이질 백본" 및 "wrist 명시 게이팅" 선례는 부재로 판정
(전수조사 아님 주의). ④ Cortical Policy (2603.21051, ICLR 2026): dual-stream(정적 뷰
=3D keypoint 스트림 / 동적 wrist 뷰=gaze-사전학습 스트림) — 뷰별 **사전학습 목적**을
달리한 가장 가까운 최신 선례로 4c/뷰별 기질 논거에 인용 가치.
