# DESIGN — wrist v2: 손목캠 트랙 재설계 (연구자급, 단계형)

*작성 2026-07-18. 근거: `docs/AUDIT_negative_results_2026-07-18.md` §4, `FOLLOWUP_experiments.md` §10–11,
`configs/phase1_libero_dualstream_wrist.yaml`, `src/models/networks.py:DualDeltaAE`,
`src/training/train_phase{1,2}.py:run_dual`, `src/eval_libero/rollout_sim.py`, KICKOFF R4.
문헌은 본문에 검증 상태(VERIFIED/UNVERIFIED)를 명기.*

**사용자 전제(존중)**: 손목캠은 포기 대상이 아니다 — 파지(grasp) 정보가 가장 많고 EE 움직임이
가장 확대되어 보이는 카메라다. 실측도 이를 지지한다: 손목 조건 토큰 하나를 빼면
**85.2→50.4% (−34.8pp)** (DESIGN_fusion §R4) — 현 파이프라인에서 단일 토큰 기준 가장 값진 입력.
문제는 "손목캠이 쓸모없다"가 아니라 **"손목캠의 추가 가치를 어디에·어떤 좌표계로 넣느냐"**다.

---

## 0. 재정식화 — Phase-B가 실제로 검정한 것과 검정하지 못한 것

### 0.1 Phase-B 판정의 지위 (감사 §4 요약)

Phase-B dual-stream(wrist=추론 변위 스트림)은 **NEGATIVE가 아니라 INCONCLUSIVE**다:

- "baseline 85–88 하회"는 통계적으로 미성립: 최장 판독 64/84=76.2%, Wilson 95% CI **[66.1, 84.0]**,
  vs 85% z≈−1.67 (p≈0.10, NS). 롤아웃이 결정론적(`policy.py:117` source_noise는 train 전용,
  h=MLP, init 고정)이라 3회 부분판독은 독립 표본이 아니라 **같은 결정론 에피소드 열의 절단 창**.
- 태스크 순차 실행으로 84ep = task0–3 전체 + task4 4ep = **커버리지 편향**; matched baseline 부재.
- config가 스스로 "⚠ SCALE" 위험을 표기하고(`phase1_libero_dualstream_wrist.yaml:13-15`)
  완화 장치는 전무: 스트림별 ζ 표준화 없음, concat ζ 전체에 **단일 스칼라 x0_std**
  (`train_phase2.py:155-159`), h의 단일 LayerNorm(`networks.py:50`)이 저분산 wrist 블록 감쇠.
- 결정적 진단(오프라인 ζ_wrist zero-ablation)이 **한 번도 실행되지 않음** — 5분짜리 검사.
- 두 변경이 동시에 들어감: ① wrist 변위 스트림 추가 + ② wrist 조건 기질 교체
  (단일 스트림의 SigLIP2-인코딩 wrist 1토큰 → dual의 DINOv3-CLS prev/cur 2토큰,
  `rollout_sim.py:196-217`). 효과 격리 실패.

### 0.2 삽입점 지도로 다시 읽기 — Phase-B의 널은 지도가 "예측"했던 결과

이 프로젝트의 핵심 실증 지도(FOLLOWUP §3–6, §10):

| 삽입점 | 실험 | 폐루프 판정 |
|---|---|---|
| **관측/조건화 (정책 입력)** | S1 concat/avg, **wrist 조건 토큰(−34.8pp ablation)** | **양성** |
| 앵커-only (조건화 제외) | S1b | 무효 |
| 타깃/코드-측 (ζ, 게이트 잔차) | C1/C2, F4 | 무효 |
| 디코더-측 (h 교체/우회) | h-flow, actionflow, residual-flow | 음성 |

**DualDeltaAE는 wrist를 어디에 넣었나?** ζ_wrist를 **flow 타깃(코드-측)** 에 넣고, h 입력을
넓혔다(디코더-측). 조건화 측면에서는 오히려 검증된 SigLIP2-wrist 토큰을 미검증
DINOv3-CLS로 **교체**했다. 즉 Phase-B는 wrist 정보를 **지도상 일관되게 무효였던 삽입점**으로
승격시키면서, 유일하게 검증된 삽입점(조건화)의 기질을 흔든 실험이다. 결과가 널이었다는 것은
지도와 **정합**이고, "wrist에 추가 가치가 없다"의 증거가 아니다.

**wrist v2의 설계 원칙**: ① 조건화(관측) 삽입점에서 wrist 표현을 풍부화하는 것이 1순위,
② 변위(추론) 스트림 재도전은 "무엇이 wrist 변위인가"를 좌표계 수준에서 바꾼 뒤에만(§4a),
③ 모든 팔은 언어 화폐 보존 이중 기준(SR **AND** correct−wrong)으로 평가 — wrist 토큰은
비언어 조건화 용량이므로 §5의 SR↔언어 tradeoff가 그대로 적용된다.

### 0.3 기전 가설 — 손목 변위의 ego-motion 교란 (v2의 과학적 핵)

손목캠은 EE에 강체 부착이다. LIBERO 액션은 base-frame OSC 델타(7-DoF)이므로:

> **Δz_wrist = f( ego-motion(액션이 만든 카메라 이동) ⊕ 장면 변화(물체 접근·파지·이탈) )**

자유공간 이동 중 frozen 인코더의 Δz_wrist 분산 대부분은 **카메라 자체 이동이 만든 전역
이미지 변화**다. 이것은 액션으로부터 자명하게 예측 가능하므로 align_wrist는 쉽게
수렴하지만(감사 §4b: cos 항 스케일 불변으로 "수렴"이 기여 증거가 아님), 그 성분은
**proprioception과 등가인 정보**라 정책에 새 정보를 주지 않는다. 파지 정보(물체가 그리퍼
개구부에 들어옴, 접촉, 미끄러짐)는 ego-motion을 제거한 **잔차**에 있다. 이 가설이 맞다면:

- Phase-B가 널인 이유가 설명된다 (g_wrist는 "액션→optic flow" 항등에 가까운 함수를 학습).
- 손목캠의 "EE 움직임 확대" 특성은 **양날**이다: 신호도 크지만 ego-motion 교란도 최대.
- 처방이 도출된다: **좌표계 재정렬(§4a) 또는 ego-motion 인수분해**로 잔차를 노출시켜야
  wrist 변위가 비로소 "파지 스트림"이 된다.

이 가설은 Stage 0의 프로브 3(ego-motion R²)으로 **수치 검정 가능**하다 — v2 전체가 이
검정에 게이트된다.

**문헌 근거 (검증 수행)**:
- **VERIFIED — arXiv:2507.17141** (Astribot Suite, "Towards Human-level Intelligence via
  Human-like Whole-Body Manipulation") §4.3: KICKOFF R4가 인용한 이 논문을 재검증했다.
  실제 주장: ① EE-space > joint-space (18/20 vs 5/20), ② delta > absolute (청크 경계
  불연속 0.0032 vs 0.0196), ③ **egocentric(EE)-frame delta > robot(base)-frame delta**
  (전신 조정 태스크 19/20 vs 16/20) — "관찰 시점이 크게 움직일 때 EE-frame delta가 더
  안정·불변인 표현". 단, **주의**: 이 논문은 head-cam 전신로봇이며 손목캠 논문이 아니다.
  "카메라 프레임에 액션을 맞추면 시각-액션 정렬 개선"의 **일반 원리 근거로는 유효**하나
  eye-in-hand 직접 증거는 아님. KICKOFF의 인용은 과잉 일반화가 약간 있었다(정정).
- **VERIFIED — HoMeR (arXiv:2506.01185)**: 장거리=absolute / 미세조작=relative(EE) 하이브리드
  액션 모드 전환, 차선 대비 +29.17pp. "국면에 따라 좌표계/모드를 바꾸는" §4b의 근거.
- **UNVERIFIED — 2606.17846, 2512.11218** (camera-frame delta pose): KICKOFF가 인용하나
  본 설계 작성 시 원문 미확인. Stage 2a 착수 전 확인 요망. (웹검색 스니펫 수준에서는
  "wrist 지배 태스크에서 EE-frame delta ≈ camera-frame 모델링 = 시각-액션 정렬 개선"
  주장이 존재.)

---

## 1. Stage 0 — 오프라인 진단 (수 시간, GPU ≤ 4h, **이후 전 단계의 게이트**)

원칙: 폐루프 1롤아웃도 돌리기 전에, 데이터·기존 ckpt만으로 "wrist 변위에 정보가 있는가,
어디에 있는가, 현 설계가 그걸 왜 못 쓰는가"를 수치화한다. 프로브 2·3·4는 **ckpt 불필요**
(데이터 레벨) — ckpt 회수 실패가 Stage 0 전체를 막지 않는다.

### 1.0 선행: 아티팩트 회수

- 회수 대상: `kist_a6000_ss:~/clip_ws/checkpoints/phase1_libero_dualstream_wrist.pt` +
  대응 phase2 dual ckpt (+ 가능하면 wandb `clipvp` run 로그). 로컬 `checkpoints/`에는
  dual ckpt 부재(구 4개뿐) — 감사 §5의 UNTRACED 문제와 동일 뿌리.
- 회수 실패 시: 프로브 1만 지연. phase1 dual 재학습(50ep, 캐시 재사용 시 ~2-3 GPU-h)으로
  대체 가능 — 단 "그때 그 ckpt"가 아니므로 재학습본 결과는 별도 표기.

### 1.1 프로브 1 — ζ_wrist zero-ablation R² (기존 dual ckpt, ~5분)

**로직** (스크립트: `src/diagnosis/wrist_zero_ablation.py`, 신규 ~80줄):

```
1. phase1 dual ckpt 로드 → DualDeltaAE (dim_main=dim_wrist=1024), phase2 dual ckpt → FlowPolicy.
2. val 삼중쌍 재구성: LiberoDataset.build_policy_pairs(..., wrist_anchor=clip_wrist)
   — train_phase2.run_dual의 val 경로 재사용 (동일 split seed).
3. 4개 조건에서 held-out 액션 R² (train_phase2.run_dual 평가 블록과 동일 산식):
   (A) full        : ζ̂=policy(toks) → ae.decode(ζ̂, z_main, z_wrist)
   (B) ζ_wrist=0   : ζ̂[:, dim_main:]=0 후 decode          (변위 기여 절제)
   (C) ζ_main=0    : ζ̂[:, :dim_main]=0 후 decode          (wrist 변위 단독)
   (D) z_wrist 셔플: decode의 z_wrist를 배치 내 셔플       (상태-조건 기여 분리)
   + oracle 변형: ζ̂ 대신 GT [Δz_main;Δz_wrist]로 (A′)(B′)(C′) — 정책 오차와 h 용량 분리.
4. JSON 아티팩트 저장 (outputs/report/wrist_zero_ablation_{ckpt_tag}.json) — per-action-dim R² 포함.
```

**해석 게이트**:
- ΔR²(A−B) < **0.005** → ζ_wrist는 오프라인에서도 죽은 채널: Stage 1 "기본기 재실행"은
  스킵 가능성 높음(표준화로 살아날 여지는 1.3 결과와 교차 판단), Stage 2로 직행.
- ΔR²(A−B) ≥ **0.02** → 변위 채널에 실질 기여 존재: Stage 1 재실행 가치 있음
  (폐루프 널이 스케일/실행 문제였을 개연성).
- (B′) oracle에서만 기여가 크면 → 정책이 ζ_wrist를 수송하지 못하는 것(단일 x0_std 혐의 강화).

### 1.2 프로브 2 — Δz 스케일 통계 (ckpt 불필요, ~30분)

`train_phase1.py:110-111`이 print만 하고 버리는 통계를 아티팩트로:
Δz_main / Δz_wrist 각각 per-dim std, 전체 std, ‖Δz‖ 분포(중앙값/IQR), 스트림 간 비율.
추가로 ζ 공간(-g 출력, ckpt 있으면)과 x0_std 스칼라 대비 블록별 실제 std.

**게이트**: std 비율이 [0.5, 2] 밖 → 스케일 불일치 "실재" 판정 → Stage 1 표준화 필수 근거.
(예상: main=SigLIP2 raw vs wrist=단위벡터화 CLS이므로 큰 격차 — 예측을 ledger에 등재할 것.)

### 1.3 프로브 3 — 손목 변위 디코더빌리티 + ego-motion 인수분해 (핵심, ckpt 불필요, ~2h)

`f2_dense_probe.py` 골격 재사용 (RidgeCV + 얕은 MLP, held-out R²), 팔 구성:

| 팔 | 입력 → 타깃 | 검정하는 것 |
|---|---|---|
| W1 | [Δz_wrist, z_wrist] → action chunk | 손목 변위의 총 디코더빌리티 |
| M1 | [Δz_main, z_main] → action chunk (기준) | 기존 f2와 정합 sanity |
| **E1** | **action chunk → Δz_wrist** | **ego-motion 지배도** (기전 가설 §0.3) |
| E2 | [ee_pos/ee_ori 델타(proprio)] → Δz_wrist | 순수 운동학→시각변위 설명력 |
| R1 | [Δz_wrist − Ê(Δz_wrist\|E2), z_wrist] → action | **ego-잔차**의 디코더빌리티 |

per-action-dim 분해 필수: **gripper 차원(7번째)·z축 병진(3번째)** vs 나머지.
proprio는 demo HDF5의 `obs/ee_pos, ee_ori, gripper_states`에서 직접 (확인 완료, 존재함).

**게이트 (사전등록)**:
- **G3a (wrist 국소 가치)**: W1의 gripper-dim R²가 M1 대비 **+0.05 이상** → "wrist 변위에
  main이 못 보는 파지 신호 존재" → Stage 2 전체 GO.
- **G3b (ego-motion 지배)**: E1/E2 R² ≥ **0.5** → 기전 가설 채택 → §4a(좌표계 재정렬)
  우선순위 승격. E1 R² < 0.3이면 §4a 근거 약화(교란이 애초에 작음) → §4c 우선.
- **G3c (잔차 가치)**: R1 gripper-dim R² > W1 gripper-dim R² → ego-잔차화가 파지 신호를
  노출시킨다는 직접 증거 = §4a의 오프라인 선행 검증 통과.
- W1이 전 차원에서 M1 이하이고 G3a 실패 → **손목 "변위"는 죽은 방향** — 변위 스트림
  트랙(§4a 포함) 전면 보류, 조건화 풍부화(§4c)와 국면 게이팅(§4b)만 진행.

### 1.4 프로브 4 — WHERE: 국면·태스크별 분석 (ckpt 불필요, ~1h)

- `gripper_states`로 데모를 국면 분할: 접근(그리퍼 열림·EE 하강) / **파지창**(그리퍼 폭 변화
  ±8프레임) / 이송 / 배치. 프로브 3의 W1/M1/R1 R²를 국면별로 재계산.
- 폐루프 근거와 접합: 1c swap의 **neither 49.5% = 파지 난이도**(FOLLOWUP §2), concat조차
  실패가 남는 태스크 — wrist가 이득을 낼 "표적 국면·표적 태스크" 목록화.

**게이트**: 파지창에서 W1(또는 R1)의 상대 우위가 이송 국면 대비 뚜렷(비율 ≥1.5×) →
§4b(국면 게이팅)의 기전 근거 확보. 균일하면 §4b 스킵(게이팅이 자를 국면 구조가 없음).

### Stage 0 사전등록 블록

- **예측 등재**(upgrade_ledger): P0-1 "스케일 비율 > 3×", P0-2 "E1 R² ≥ 0.5 (ego 지배)",
  P0-3 "W1 gripper-dim이 M1을 상회", P0-4 "파지창 집중".
- **반증 조건**: G3a·G3c 모두 실패 + 프로브 1 ΔR² < 0.005 → "wrist 변위에는 조건 토큰을
  넘는 정보가 없다"가 **오프라인에서 확정** — Stage 1/2의 변위 계열 전면 중단,
  §4c(조건화 풍부화)만 잔존. 이것이 이 설계의 최대 지출 절약 장치다.
- **비용**: GPU ≤ 4h (wrist DINOv3 임베딩 캐시 기존재 시 ≤ 2h), 인건 1일.

---

## 2. Stage 1 — 기본기 교정 재실행 (dual-stream을 "공정한 재판"에 세우기)

*조건: Stage 0에서 프로브 1 ΔR² ≥ 0.02 또는 (0.005–0.02 구간이면서 프로브 2 스케일 불일치
확정 — "표준화가 살릴 수 있는" 시나리오)일 때만.*

Phase-B 재판의 목적은 이기는 것이 아니라 **"현 dual 설계의 널"을 통계적으로 성립시키거나
뒤집는 것**이다. 어느 쪽이든 논문 §insertion-map의 한 셀이 확정된다.

### 2.1 교정 목록 (감사 §4 처방의 구현 사양)

1. **스트림별 ζ 표준화**: phase1에서 Δz_main/Δz_wrist를 각자 스칼라 std로 z-score
   (`DualDeltaAE`에 `dz_std_main/dz_std_wrist` buffer 등록, decode에서 역스케일 불필요 —
   h가 표준화 공간에서 학습). align MSE가 자동으로 스트림 균형.
2. **블록별 x0_std**: `FlowPolicy.x0_std` shape (1,)→(dim_cat,) per-dim (또는 최소 2-스칼라
   블록 상수). broadcast라 코드 변경 3줄. (표준화 후엔 사실상 균등하지만 이중 안전판.)
3. **loss 균형 로깅**: align_main/align_wrist/recon 각 항의 epoch별 값 + grad-norm 기여를
   wandb에 — "수렴했다"의 재발 방지 (cos-불변성 착시 차단).
4. **matched baseline**: large256-single(phase1+phase2)을 **같은 주·같은 코드 리비전·같은
   split**에서 재학습. dual과 동일 rollout 창에서 평가.
5. **조건 기질 confound 제거**: dual 팔은 wrist 조건 토큰(§0.1 ②)을 **SigLIP2-wrist로
   유지**하는 변형을 기본으로 (변위 스트림만 DINOv3) — Phase-B의 이중 변경을 격리.
   (원 Phase-B 구성은 부팔로 병행 가능하면 병행.)
6. **retry-supervisor + 아티팩트 위생**: 태스크 단위 서브프로세스, osmesa 세그폴트 시
   해당 태스크 ≤3회 재시도, per-episode 성공 플래그 JSONL, 출력 파일명에 ckpt 태그
   (`rollout_{suite}_{mode}_{ckpt_tag}.jsonl`) — 감사 §5 UNTRACED 재발 방지. 200ep 완주.
7. **프로토콜**: task당 20롤(스크리닝) → paired bootstrap 10k per-task (eval_protocol v1.1),
   correct + wrong 두 모드. 우세 주장 시에만 3시드 확전(불변식 #3).

### 2.2 Stage 1 사전등록 블록

- **예측**: P1-1 "표준화+블록 x0_std로 프로브 1 ΔR²(A−B)가 ≥1.5× 증가",
  P1-2 "폐루프 paired Δ(dual−matched single)는 CI가 0을 포함(널 유지)" — *지도(§0.2)가
  맞다면 기본기를 고쳐도 코드-측 삽입은 무익이어야 한다. 만약 SIG>0이 나오면 그것대로
  지도의 수정(스케일이 유일 병목이었다는 발견)이라 양쪽 다 정보 가치가 있음.*
- **게이트**: paired Δ 95% CI 상한 < +2pp → "현 dual 설계 무이득" **확정** (이번엔 통계적
  성립) → 변위 스트림은 §4a 재정의 경로로만 존속. CI가 +2pp 이상을 포함 → 3시드 확전.
- **반증**: correct−wrong이 baseline 대비 −10pp 이상 악화되면 SR과 무관하게 NO-GO
  (언어 화폐 보존 위반 — wrist 토큰 증가가 언어 희석을 일으키는지 최초 측정 지점).
- **비용**: phase1 dual 2–3 GPU-h + phase2 dual 4–6 + matched baseline 6–8 + 롤아웃
  4팔×200ep(retry 포함) ~12 GPU-h ≈ **총 25–30 GPU-h**, 달력 3–4일(박스 불안정 버퍼 포함).

---

## 3. Stage 2 — 새 과학: 후보 설계 4+1 (기전 차별화 필수)

Phase-A의 프라이어 "복잡화는 일관 음성"을 정면으로 받는다. 각 후보는 **이미 실패한 것과
기전이 어떻게 다른가**를 명시해야 하고, 다르지 않으면 제안 자격이 없다. 공통 원칙:
**추가 용량이 아니라 표현의 좌표계·삽입점·시점(when)을 바꾼다.**

### 4a. EE/카메라-프레임 손목 변위 스트림 (기전 재정의 — 본명 "과학" 후보)

- **설계**: g_wrist의 액션 입력을 base-frame 델타에서 **EE-frame 델타로 회전 변환**
  (demo의 `ee_ori`로 R_base→EE(t) 구성, 병진·회전 델타를 EE 좌표로 재표현; gripper 차원
  불변). 선택 강화판: align 타깃을 ego-잔차 Δz_wrist − Ê(Δz_wrist|운동학)로 교체
  (Ê는 Stage 0 프로브 E2의 릿지 예측기를 동결 재사용 — 학습 파라미터 0).
- **기전 차별화**: 실패한 것들(h-flow/actionflow/residual-flow/게이트)은 전부 **용량·모듈
  추가/교체**였다. 4a는 파라미터를 늘리지 않고 **입력 좌표계를 관찰 카메라와 정렬**한다 —
  "시각적으로 비슷한 손목 변위 = 수치적으로 가까운 액션"이 되도록 접지 자체를 고친다.
  Phase-B와의 차이: Phase-B의 g_wrist는 base-frame 액션→(ego-motion 지배) Δz_wrist라는
  좌표계-교란 맵을 학습했고, 4a는 교란을 입력에서 제거한다.
- **문헌**: 2507.17141 §4.3 (VERIFIED, EE-frame delta 우위 19/20 vs 16/20) + HoMeR
  (VERIFIED). camera-frame delta pose 2건은 착수 전 검증 (UNVERIFIED).
- **선행 게이트**: Stage 0 **G3b(ego 지배) AND G3c(잔차 가치)** 통과 시에만. 오프라인
  프로브(R1)가 이미 4a의 절반을 무료로 검정해 준다는 점이 이 설계의 미덕.
- **반증 조건**: EE-frame 판 프로브(W1-EE)가 W1 대비 gripper-dim R² 개선 없음 →
  폐루프 진입 없이 폐기. 개선 있으면 phase1+phase2 1팔 학습 → Stage 1 프로토콜.
- **비용**: 오프라인 확장 ~2 GPU-h; 폐루프 진입 시 +10–12 GPU-h.

### 4b. 국면 게이트 조건화 (wrist를 "언제" 쓸지)

- **설계**: wrist 조건 토큰(현행 삽입점 유지)에 **비학습** 게이트 w(t)를 곱한다:
  w(t) = σ(k·(파지 근접 신호)) — 신호는 그리퍼 폭 변화율 또는 proprio z-높이+그리퍼 상태의
  고정 휴리스틱. 학습형(α) 변형은 2차(후술 이유로).
- **기전 차별화**: C1/C2 게이트 실패의 뿌리는 **학습형 α에 태스크 그래디언트가 0**이라
  구조적으로 닫힌 것(감사 §1). 4b 기본형은 **학습 파라미터가 없어** 그 실패 모드가 원천
  차단되고, 그래디언트는 스케일된 토큰을 통해 정책 본체로 정상 흐른다. 또한 게이트 위치가
  코드-측 잔차가 아니라 **검증된 조건화 삽입점**이다.
- **근거 요건**: Stage 0 프로브 4에서 "파지창 집중"이 확인될 때만. 손목 정보가 국면
  균일하면 게이팅은 정보 삭제일 뿐이다.
- **반증 조건**: 게이트 on/off paired Δ CI 상한 < +2pp → 폐기 (조건화는 상시가 최선이라는
  결론도 지도에 기입할 가치 있음). **주의**: −34.8pp ablation은 wrist 상시 기여가 이미
  크다는 뜻 — 4b의 사전 승산은 낮게 등재한다(정직한 예측).
- **비용**: phase2 재학습 1팔 + 롤아웃 ≈ 8–10 GPU-h. 구현 반나절 (rollout `w(t)`는
  obs의 gripper_qpos에서 계산).

### 4c. wrist = 국소 기하: CLS 대신 소격자 패치 조건 토큰 (1순위 추천)

- **설계**: 감사가 지적한 "DINOv3-CLS-as-wrist 미검증"을 정면 교체 — wrist 프레임의
  DINOv3 패치 16×16을 **2×2 avg-pool → 4토큰**(param-0)으로 만들어 **조건 토큰 열에 추가**
  (변위 스트림 없음, ζ 불변, h 불변). 변형 c0(최저가): 현 SigLIP2-wrist 토큰을 유지한 채
  DINOv3-CLS wrist 토큰 1개만 **병기** — S1 concat의 "기질 상보성" 발견을 wrist에 이식.
- **기전 차별화**: F3 실패는 agentview + **결함 학습 모듈**(kv-LN 부재, init 1.0, 규제 전무
  — 감사 §3b) + 120ep 레짐이었다. grid-token은 OOM으로 미검정(무결과지 음성 아님).
  4c는 ① 학습 파라미터 0(모듈 결함 여지 없음), ② K=4로 작음(OOM 원인이던 dense 조립을
  스트리밍 pool로 회피), ③ 카메라가 wrist(파지 국소 정보가 실재하는 곳 — CLS는 그리퍼가
  화면 대부분을 채우는 wrist 뷰에서 거의 상수적 전역 요약이라 접촉 순간 정보를 뭉갠다),
  ④ 삽입점이 조건화(검증 양성). **F3-echo 위험은 실재**하므로 정직하게 등재하되, 네 차이
  모두 F3 실패 기전을 직접 겨냥한다.
- **반증 조건**: c0·4c 모두 matched single 대비 paired Δ CI 상한 < +2pp → "wrist 조건화는
  SigLIP2 1토큰으로 포화" 확정(§5 정직 섹션의 결론 채택). correct−wrong −10pp 초과 악화 →
  언어 희석으로 NO-GO (wrist에서도 DINOv3 함량↑=언어↓ tradeoff가 재현되는지가 부산물
  발견이 됨 — §5 tradeoff의 wrist 축 확장).
- **비용**: c0는 임베딩 캐시 + phase2 재학습만 ≈ 6 GPU-h; 4c(패치 pool) 캐시 신규 인코딩
  포함 ≈ 10 GPU-h. **Stage 0 게이트와 독립적으로 실행 가능한 유일 후보** (변위 가설에
  의존하지 않음) — 그래서 1순위.

### 4d. 비대칭 역할: wrist를 h의 상태에만 (조건화 아님) — 저승산, 지도 완결용

- **설계**: ζ는 main 단독 1024 유지(정책·flow 불변), h만 z_wrist를 추가 상태로 받도록 확장
  h(ζ_main, [z_main; z_wrist]). 역방향(조건화-only)은 이미 현행 = 기준선.
- **기전 차별화**: S1b는 조건화에서 기하를 **제거**해 실패했다. 4d는 조건화를 그대로 두고
  **디코딩 상태만** 풍부화 — 지도에서 아직 비어 있는 셀("decode-state 삽입")을 채운다.
  Phase-B와 달리 flow 타깃(코드-측)을 건드리지 않으므로 x0_std/스케일 문제가 아예 없다.
- **정직한 사전 평가**: C0 프로브에서 decoder_state_cond=False가 거의 무손실이었다 —
  h가 상태를 별로 안 쓴다는 기존 증거는 4d에 불리하다. 승산 낮음으로 등재하고, 가치는
  "삽입점 지도의 마지막 셀 확정"이라는 서사 완결성에 있다. 최후순위.
- **비용**: phase1 h 확장 재학습 + phase2 ≈ 8 GPU-h.

### Stage 2 우선순위와 사전등록

**실행 순서 (정보/GPU-h 기준)**: ① 4c/c0 (게이트 독립, 최저가, 조건화-삽입 정합) →
② 4a (Stage 0 G3b·G3c 통과 시 — 유일한 "기전 신규" 후보) → ③ 4b (프로브 4 국면 집중 시)
→ ④ 4d (여력 시). 총 GPU 예산 상한 **50 GPU-h** — 초과 시 미착수 후보는 기록만 남기고 종료.

**전 후보 공통 사전등록**: 예측을 upgrade_ledger에 승산과 함께 등재(4c: 中, 4a: 中,
4b: 低, 4d: 低) / 판정은 폐루프 paired CI만(불변식 #1·#3) / 이중 기준: paired Δ SIG>0
**AND** correct−wrong ≥ 70pp 유지 / 우세 주장 시 3시드 / 모든 팔 no-aug 클린 밴드 /
per-episode JSONL 아티팩트 의무. **suite 주의**: spatial에서 concat 천장이 97.5라
best-base 위 통합 검정은 검정력이 없다 — Stage 2는 **large256-single 기반(헤드룸 ~15pp)**
에서 기전을 확정하고, 승자만 concat-base + (가능하면 libero_goal/object로 확전 — 파지
난이도가 남아있는 곳, 1c neither 49.5% 참조)하여 최종 가치를 검정한다.

---

## 4. 정직 섹션 — "wrist는 이미 최적으로 쓰이고 있다"는 최강 반론

이 트랙에 GPU를 더 태우기 전에, 반대편 최강 논증을 세워 둔다:

1. **−34.8pp는 조건 토큰의 가치이지, 미개발 잔여 가치의 증거가 아니다.** 현 파이프라인은
   wrist에서 이미 가장 큰 단일-토큰 이득을 뽑고 있다. 한계효용 체감이 정상 기대다.
2. **Δz_wrist의 추가 정보 = (ego-motion 성분: proprio 등가, 무가치) + (파지 잔차: 존재하나
   frozen 전역 인코더의 CLS가 포착 못 할 가능성).** 즉 "정보는 있는데 이 표현 계열로는
   원리적으로 못 뽑는" 시나리오 — 그 경우 v2의 어떤 변형도 안 되고, 해법은 frozen 원칙을
   깨는 쪽(별도 접촉 센서/미세 인코더)이라 이 프로젝트의 thesis 밖이다.
3. **복잡화 일관 음성 프라이어**: 8개 방향이 연속 음성/무효였다. 사전 승산을 프라이어로
   깎는 것이 베이즈적으로 정직하다.
4. **천장 문제**: spatial에서 남은 실패는 파지 난이도(neither 49.5%)인데, 그 실패가
   "정보 부족"이 아니라 "저수준 제어 한계(OSC 델타·20Hz·청크 실행)"라면 어떤 관측 개선도
   안 통한다.

**무엇이 이 논쟁을 종결시키는가** (v2가 그 증거를 정확히 생산하도록 설계됨):
- **찬성(v2 무용) 종결 증거**: Stage 0에서 G3a·G3c 동시 실패(파지 신호가 변위에도 잔차에도
  없음) + Stage 2 4c/c0 paired 널(조건화 포화) → "wrist-as-single-condition-token이 이
  frozen 기질에서 최적"을 **오프라인+폐루프 양면에서** 확정. 이 자체가 논문 삽입점 지도의
  강한 한 행이다.
- **반대(잔여 가치 실재) 종결 증거**: G3c 통과(ego-잔차가 gripper-dim 디코더빌리티를 올림)
  → 정보가 존재하고 좌표계가 병목이었다는 것 → 4a 폐루프가 최종 심판.
- 어느 쪽이든 **Stage 0 (≤4 GPU-h)가 논쟁의 8할을 해소한다** — 이것이 이 설계의 요지다.

---

## 5. 요약 실행표

| 단계 | 내용 | 게이트(진입) | 산출물 | GPU-h | 달력 |
|---|---|---|---|---|---|
| 0 | 오프라인 진단 4종 + ckpt 회수 | — (즉시) | zero-ablation/스케일/디코더빌리티/국면 JSON | ≤4 | 1–2일 |
| 1 | 기본기 교정 dual 재판 + matched baseline | 프로브1 ΔR²≥0.02 (또는 0.005–0.02 & 스케일 확정) | 통계 성립한 dual 판정 | 25–30 | 3–4일 |
| 2-4c | wrist 국소 기하 조건화 (c0→패치 4토큰) | 없음 (독립) | 조건화 포화 여부 | 6–10 | 2–3일 |
| 2-4a | EE-frame/ego-잔차 변위 | G3b∧G3c | 좌표계 기전 판정 | 2 (+10–12) | 2–4일 |
| 2-4b | 국면 게이트 | 프로브4 파지창 집중 | when-지도 | 8–10 | 2일 |
| 2-4d | h-상태 wrist | 여력 | 지도 완결 | 8 | 1–2일 |

*총 예산 상한 50 GPU-h (Stage 1 포함 시 80). 모든 예측은 실행 전 upgrade_ledger 등재,
모든 우세 주장은 3시드+paired CI, 모든 롤아웃은 retry-supervisor+per-episode JSONL.*
