# DESIGN — wrist 융합 통합 설계 v1 (**WristCond-v1**) (2026-07-18)

*작성: 통합 아키텍처 설계 에이전트 (파이프라인 3단계 — W1 문헌 → W2 증거 브리프 → 본 설계 → W4 적대 검증).
근거 문서: `docs/BRIEF_wrist_design_inputs.md`(확정 수치·결함·제약 전부) ·
`docs/LIT_wrist_multiview_2026-07-19.md`(검증 문헌·ingredients) · `docs/DESIGN_wrist_v2.md`(단계 계획) ·
`docs/DESIGN_patch_policy_attention_v1.md`(P-A/P-B, F3 결함수정 규격, 메모리 계획) ·
`docs/DESIGN_grounding_space_v1.md` §R-Δ · `docs/WEEK0_probe_results.md`(P-zg1/zg3 RED, ζ_wrist GO, E1 기각) ·
소스 실사: `src/models/networks.py`, `src/models/policy.py`, `src/training/train_phase{1,2}.py`,
`src/eval_libero/rollout_sim.py`, `src/data/libero.py`, `src/core/anchor.py:224-306`, `src/models/obs_fusion.py:83-123`.
모든 제안 요소의 구현 가능성은 file:line 수준으로 대조 완료.*

---

## 0. 결정 요약 — ONE architecture + 2 ablation arms

| | 구성 | 한 줄 정의 | 신규 학습 파라미터 |
|---|---|---|---|
| **MAIN — W-A "WristPatch-Cond"** | 단일 스트림 통화 불변(ζ=ζ_main), wrist는 **조건화-only 풍부화**: 기존 SigLIP2-wrist 토큰 **유지** + DINOv3-L/16 wrist **패치 2×2 pool 4토큰** 추가 (UNGATED) | wrist 정보를 검증된 유일 양성 삽입점(관측/조건화)에서 증폭. ζ/g/h/x0 단 1비트도 불변 | 모듈 1.05M + ctx 확폭 6.3M = **+7.4M** |
| **ABL-1 — W-B "Δ̄w-token"** | W-A + **측정된 과거 손목 변위 1토큰**: Δz̄_w = patch-mean(t) − patch-mean(t−span), 2-스칼라 재척도 | ΔR²=+0.179가 가리킨 "변위 정보"를 타깃이 아니라 **관측측 인과 가용 신호**로 전환 | +0 (버퍼만; ctx +1.6M) |
| **ABL-2 — W-C "dual 표준화 재판"** | Phase-B dual을 결함 ①~⑤ 수리 후 재실행: per-stream dz_std 버퍼, per-block x0_std, 항별 grad 로깅, SigLIP2-wrist 조건 토큰 유지(confound 제거) | 타깃-측 삽입의 널을 **통계적으로 성립**시키거나 뒤집는 지도-완결 팔 (Stage 1 자격 = ΔR² 0.179 GO로 기충족) | 학습 파라미터 +0 (버퍼 3개) |

공통 심판: 폐루프 paired SR (matched large256-single 동시 재학습 기준) **AND** correct−wrong ≥ +70pp.
스크리닝 20roll → 승자만 50roll×3seed 확증. 총 예산: 스크리닝 ≈ 40 GPU-h, 확증 +45, 상한 90.

---

## 1. (c의 핵) 긴장 해소 — "ζ_wrist는 타깃-측에서 실패했는데 오프라인 기여 +0.179는 실재한다"

두 사실은 모순이 아니라 **오프라인 지표의 성격** 때문에 동시에 참이다.

1. **ΔR²(A−B)=+0.179는 "정보 존재" 증명이지 "삽입점 유효성" 증명이 아니다.**
   zero-ablation은 h가 학습한 입력 슬롯을 0으로 꺾는 조작이다. h가 ζ_wrist 슬롯에 의존하도록
   학습된 이상, 그 슬롯을 0으로 만들면 입력이 OOD가 되어 R²가 떨어진다 — 슬롯이 **비중복
   신규 정보**를 담는지와는 별개다. (D 프로브 −0.009는 z_wrist 상태의 중복성만 기각했지
   ζ_wrist의 중복성은 검정하지 않았다.)
2. **롤아웃에서 ζ̂_wrist는 정책의 출력이다.** 폐루프에서 ζ̂_wrist는 조건 토큰들로부터
   정책이 생성한다 — 즉 시험 시점에 ζ̂_wrist가 담을 수 있는 정보는 **조건 토큰에 이미 있는
   정보의 재부호화**뿐이고, 새 관측 정보는 0이다. 타깃-측 ζ_wrist는 h에 디코딩 용량을 더할
   뿐 정보를 더하지 못한다. 오프라인 teacher-forcing은 이 차이를 못 본다(IV6: offline≠SR,
   Phase-B가 정확히 이 패턴: 오프라인 R² 0.663 건강 / 폐루프 널).
3. **따라서 변환 경로는 하나다**: wrist의 실측 정보(oracle gripper-dim 0.890 — 파지 채널은
   손목 변위만으로 거의 완전 디코딩; E1=0.177 — 그 변위는 proprio 등가가 아님)를 **정책이
   관측으로 소비**하게 하라. 삽입점 지도(조건화=유일 양성 −34.8pp / 타깃·코드측=일관 무효)와
   ΔR² 증거가 함께 가리키는 곳은 조건 토큰열이다.

**설계 원칙 도출**: ① ζ 통화는 main 단독 유지(타깃-측 재진입 금지 — W-C만 예외로 지도 완결용),
② wrist의 "상태"는 이미 토큰으로 있으니(−34.8pp) 추가할 것은 (i) CLS가 뭉개는 **국소 기하**
(패치 토큰, 결함 ⑦ 정면 교체)와 (ii) **측정된 과거 변위**(ΔR²=0.179의 관측측 대응물 — 미래
ζ_wrist의 예측이 아니라 과거 Δz_w의 관측이므로 인과적으로 가용), ③ 두 추가 모두 UNGATED
상시(IV4), 학습 부품 최소(Phase-A 프라이어).

---

## 2. 아키텍처 스펙

### 2.1 (b) wrist 인코더 — DINOv3-L/16 패치 2×2-pool 4토큰, SigLIP2-wrist 토큰과 병기

| 결정 | 채택 | 기각 대안과 이유 |
|---|---|---|
| 인코더 | **DINOv3-L/16 @256, 패치 16×16 → `adaptive_avg_pool2d` 2×2 = 4토큰 × 1024d** (`Dinov3Anchor(force_size=256, pool_to=2)` — `anchor.py:256,269-270,301-306` 기존재, cache_key `dinov3-vitl16-256-pool2` 자동 분리) | **DINOv3-CLS**: 결함 ⑦(그리퍼가 화면 대부분인 wrist 뷰에서 CLS≈준상수 전역 요약, 접촉 순간 정보 뭉갬) + DINO CLS 객체-중심 emergent 문헌 합치 — 기각. **SigLIP2-shared**: 이미 있음(유지). 패치의 국소성 근거는 LIT §3.3(DINO dense 우위 일관, DINOBot 2402.13181 방증) + Cortical Policy(2603.21051, 뷰별 사전학습 목적 분담의 최근접 선례) |
| 기존 SigLIP2-wrist 토큰 | **유지** (교체 아님 — 병기) | Phase-B 결함 ⑤(검증된 조건 기질을 미검증 DINOv3-CLS로 교체한 이중 변경)의 재발 방지. LIT §5: "S1 concat 기질 상보성 + Theia + DINO 근접뷰 우위" 세 근거의 최소 신규 조합 = c0∪4c |
| K | **4 (2×2)** | K=16은 언어 희석 예측(패치 문서 §3.2 +55~65pp 탈락 기본 시나리오)과 ctx 파라미터 +25M — wrist 뷰는 main과 달리 타깃-선택 부담이 없어(타깃은 지시문+agentview 소관) 저해상 4셀로 충분하다는 가설. K=16 확전은 W-A 승리 후에만 |
| 전처리 / P-A crop 상호작용 | wrist 프레임 **crop 없음, force_size=256 그대로** | P-A crop 팔은 agentview(main) 카메라 전용 — 카메라·cache_key가 분리돼 wrist 경로와 상호 간섭 0. wrist는 이미 근접뷰라 crop의 기전(원거리 타깃 확대)이 없고, center-crop은 화면 가장자리의 그리퍼 손가락을 자를 위험. P-A 승자가 main에 채택돼도 wrist 설정 무변경 |
| 캐시 계획 | 신규 인코딩 1회: 62,250 프레임 × 4tok × 1024d fp32 ≈ **1.0 GB**, 0.3–0.5 GPU-h. per-frame 4토큰이라 GridObs-OOM 경로(45–58GB, IV5) 원천 부재. 기존 캐시(large256 main, SigLIP2-wrist, DINOv3-CLS-wrist)는 전부 재사용 | dense 16×16 캐시(32.6GB) 불요 — pool이 선형이라 인코딩 시점 풀링과 수학 동일(패치 문서 §3.1 검증 논리) |

### 2.2 (c) g/h 모듈 구조 — 단일 g/h 유지, wrist는 g에도 h에도 넣지 않는다

- **g**: 기존 단일 `ChunkEncoder` (main만). g_wrist 없음 → align_wrist 없음.
- **h**: 기존 단일 `ChunkDecoder` h(ζ_main, z_main). 확장 없음.
- **h 입력 구성 대안 판정**: ① [ζ_main;ζ_wrist] widened h(=Phase-B) — 타깃/디코더-측, §1 논리로
  기각(W-C가 통계 성립판으로만 재판). ② wrist→h-상태만(4d) — [AMEND A3] 기각 근거는
  **DESIGN_wrist_v2 §4d의 낮은 사전 승산 + 예산 순위**(wrist_v2 자체가 최후순위 등재) — 기각(셀 없음).
  (각주: 구판이 인용한 C0("h가 상태를 거의 안 씀")는 WEEK0 §7이 **UNVERIFIED·인용 금지**로
  판정해 제거. 근사 절제 실측 h(Δz,0) R² −0.45~−0.59는 오히려 **반대 방향**(h가 상태를 사용)
  이며, Jacobian eff-rank~5는 ∂h/∂ζ의 성질이지 상태 미사용의 증거가 아니다.)
  ③ **조건-only** — 채택. 부수 배당: phase1 재학습 0회(기존 `phase1_libero_siglip2_large256.pt`
  재사용) → 팔 간 phase1 분산 0, matched 비교 순수성(패치 문서 §2.1 선례).
- **innovation-grounding(§R-Δ·P-zg RED의 잔차화 align) 과의 합성**: W-A/W-B는 phase1을 동결
  재사용하므로 잔차화 셀과 **직교** — 그 셀이 이기면 새 phase1 위에 조건 토큰 구성 무변경으로
  재실행하면 된다(조건화는 align 정의를 건드리지 않음). h eff-rank~5의 PCA-k=32 셀과도 동일하게 직교.

### 2.3 (a) 손실 배치 — 항목별 채택/기각 표

| PI 질문 항목 | 판정 | 스펙 / 기각 사유 |
|---|---|---|
| main align: 잔차화 Δz−r(z_t)? | **본 설계에서 불변** (phase1 동결) | 잔차화는 별도 사전등록 셀(WEEK0 §9 후속 ①)로 진행 중 — wrist 설계를 그 결과에 인질 잡히지 않게 분리. 합성 규칙은 §2.2 |
| wrist align: raw / 잔차 / 폐지? | **폐지 (W-A/W-B)** — g_wrist가 없으므로 align_wrist 자체가 없다 | P-zg1/zg3 적신호(align의 65–70%가 상태 성분)와 정합: 검증 안 된 align 손실을 하나 더 만드는 대신 손실 0개. **W-C 한정**: 표준화 공간 raw Δz_wrist에 MSE+cos (잔차화 아님 — 팔당 변경 하나 원칙; wrist용 상태-지름길 정량은 후속 프로브 zg3-w(ridge z_w→Δz_w)로 먼저 측정 후에만 논의) |
| 스케일 처리 — 정규화 지점 명세 | **채택 (4지점)** | **N1** (W-A): `GridObs`에 사영 전 `LayerNorm(patch_dim)` 추가 (`obs_fusion.py:117` proj 앞; guarded `ln: true`, 기본 false=비트 동형) — 패치 스트림 표준화, F3 kv-LN 결함의 param-최소 대응. **N2** (W-B): Δ̄w 토큰 재척도 `w_tok = Δz̄_w · (σ_ref/σ_Δ)`, σ_Δ=std(Δz̄_w,train), σ_ref=std(z_cur,train) — 두 스칼라를 phase2 ckpt dict에 저장, train/rollout 공용 (ctx 입구 LN(`policy.py:86`)은 flatten 전체 정규화라 토큰 간 상대 스케일을 못 고침 — 결함 ③과 동일 기전이므로 사전 재척도 필수). **N3** (W-C): `DualDeltaAE`에 `dz_std_main`(0.1346)·`dz_std_wrist`(0.0208) buffer 등록, align 타깃·h 입력 모두 z-score 공간 (`train_phase1.py:110-111`의 print를 buffer 주입으로 승격) — 결함 ①·③ 동시 해소. **N4** (W-C): `FlowPolicy.x0_std`를 guarded `x0_per_dim: true` 시 shape (flow_dim,)로 등록(`policy.py:97`; `train_phase2.py:155-159`에서 `lt.std(0)` 주입) — 결함 ② 해소 (N3 후 사실상 균등하지만 이중 안전판, 사전등록 그대로) |
| recon/cycle (확장 h용) | **해당 없음 (W-A/W-B)** — h 불확장 | W-C: recon/cycle 식 불변, 입력이 표준화 공간일 뿐 (`networks.py:331-345` 타깃 텐서만 buffer 나눗셈) |
| 스트림-비대칭 규제 (main-IB/dropout, Hsu 2203.12677) | **main-IB 기각, 신규 wrist 토큰 한정 dropout 채택** | 기각 사유: ① Hsu의 IB 동기는 지배-스트림의 spurious 단서로 인한 **OOD 일반화** 실패 — 우리 실패 모드는 in-distribution SR와 언어 공동기준이고, main 토큰이 언어 화폐의 운반체라 병목은 co-기준 직접 훼손 위험. ② −34.8pp가 이미 "wrist는 충분히 사용됨"을 증명 — "wrist 활용 강제" 동기 부재. ③ Phase-A 프라이어. **채택분**: 신규 wrist 패치 토큰에만 per-token drop p=0.10 + group-drop p=0.10 (패치 문서 §2.3 규격) — 언어 경로 보존 강제 + 학습-중 zero-ablation 곡선 공짜 확보(LIT §4.3). **재개 조건 등재**: P2 사용량 프로브가 "토큰 미사용"인데 G0 프로브가 "정보 실재"라면 main-토큰 dropout 팔을 후속으로 |
| wrist→gripper/z축 보조 감독 | **학습 손실로는 기각, param-0 프로브(G0)로 유지** | LIT §4.2: 직접 선례 없음 + "프로브에서 이득이 먼저 보일 때만" 게이트 논리. G0(§5.1)가 같은 기전을 0원에 검정. 승격 조건: G0 통과 AND W-A/W-B가 파지-지배 태스크(t4류)에서만 실패 잔존 시, phase2에 ridge-head 보조 손실 1팔 |

### 2.4 W-B — 측정 변위 토큰의 정확한 정의

- **정의**: `Δz̄_w(t) = mean_k p_k(t) − mean_k p_k(t−span)`, p_k = wrist DINOv3 pool2 토큰 4개
  (k=1..4), span=16(0.8s). **과거 변위** — 롤아웃에서 인과적으로 가용(PatchDelta 문서 §5.1의
  "과거 변위" 관례를 pooled wrist 스케일로). 신규 인코딩·신규 캐시 0 (W-A의 pool2 캐시에서 유도).
- **왜 이것이 ΔR²=0.179의 올바른 관측측 전환인가**: oracle C′(main=0) gripper 0.890은 손목
  "변위"가 파지 채널을 담는다는 것; E1=0.177은 그 변위가 액션으로 자명하지 않다는 것(=관측
  가치 있음). 다만 0.179는 **미래** 변위(A_fut 정렬) 기준이므로, 과거 변위의 파지-국면 정보량은
  별도 검정이 필요하다 — 그래서 W-B는 G0 프로브(과거 Δz̄_w ridge)를 **선행 게이트**로 갖는다.
  G0 실패 시 W-B는 학습 없이 폐기된다(설계의 최대 절약 장치).
- 콜리그 방증: wrist 토큰 스케일↑ 노브만이 t4(서랍) 60→86을 만든 유일 개입(post93) — 손목
  신호의 **크기**가 폐루프 레버임을 시사, N2 재척도 스칼라가 같은 노브를 원리화한 것.

---

## 3. (d) 텐서 라우팅 표 — 전 홉, dim/정규화/변경 file:line

### 3.1 Phase 1 (W-A/W-B: **변경 0** — 기존 ckpt 재사용)

| 홉 | 텐서 | dim | 정규화 | 코드 |
|---|---|---|---|---|
| 입력 | z_main(t), Δz_main | 1024 | SigLIP2 raw (normalize=false) | 기존 `phase1_libero_siglip2_large256.pt` 그대로 |
| g/h/손실 | 전부 불변 | — | — | **diff 0줄** |

**W-C phase1** (`train_phase1.py run_dual :58-236`):

| 홉 | 텐서 | dim | 정규화 | 변경 |
|---|---|---|---|---|
| 데이터 | (Zt,Ztn,A,Zwt,Zwn) 삼중쌍 | 1024/1024 | main raw / wrist DINOv3-CLS unit-norm | `libero.py:319-324` 불변 |
| 표준화 | Dm/=dz_std_main, Dw/=dz_std_wrist | 1024 각 | z-score(스칼라, train std) | `train_phase1.py:110-111` print→buffer 주입; `networks.py` DualDeltaAE에 `register_buffer` 2개, `losses()`/`encode()`/`decode()` 경계에서 나눗셈 (**N3**) |
| 손실 | 0.5(align_m+align_w)+0.5 recon+0.25 cycle | — | 표준화 공간 MSE+cos | 식 불변 `networks.py:331-345`; **항별 값+grad-norm wandb 로깅 추가** (결함 ④) |
| ckpt | dz_std 2buffer 포함 저장 | — | — | `train_phase1.py:218-226` dict에 자동 포함(state_dict) |

### 3.2 Phase 2 — W-A 토큰열 (`train_phase2.py` **단일-스트림 main 경로**, run_dual 아님)

config 신설 `configs/phase2_libero_large256_wristpatch.yaml`: `phase1_ckpt=large256`,
`module.{name:flow, lang_token:true, wrist_token:true, grid_obs:{anchor:{name:dinov3, force_size:256,
pool_to:2}, camera:eye_in_hand_rgb, n_tokens:4, pool:avg, ln:true, tok_drop:0.1, group_drop:0.1,
init_std:0.02}}`, `data.wrist_camera:eye_in_hand_rgb`, no-aug. **grid_obs seam이 카메라 인자를 이미
지원**하므로 학습·데이터·롤아웃 배관 신규 코드 ≈ 0. [AMEND A1 반영] GridObs의
`ln`/`tok_drop`/`group_drop`/`init_std` guarded 옵션은 **구현 완료**(`obs_fusion.py` GridObs,
기본값 전부 off = 기존 config 비트 동형; train `train_phase2.py`·rollout
`rollout_dataset.load_models` 양측 배선), grid_obs 블록의 **미지원 키는 train_phase2 파싱에서
즉시 assert**(silent no-op 금지 — VERIFY A1).

| # | 토큰/텐서 | dim | 정규화 | 소스 (file:line) |
|---|---|---|---|---|
| 1 | z_prev (main SigLIP2) | 1024 | raw | `libero.py:310-313` → `train_phase2.py:444` |
| 2 | z_cur | 1024 | raw | 동일 |
| 3 | a_emb = g(A_past, z_prev) | 1024 | g-공간 | `train_phase2.py:495-502` (A_EMB_IDX=2 규약 유지 → **x0=past 불변**) |
| 4 | lang | 1024 | SigLIP2 text | `train_phase2.py:474-482` |
| 5 | wrist_sig = z_w,SigLIP2(t) | 1024 | raw (main 앵커) | `libero.py:325-332`(6번째 배열) → `train_phase2.py:449-454` |
| 6–9 | wristpatch×4 = GridObs(pool2 dense) | 1024→1024 | **N1** LN(1024)→Linear(std 0.02 init)→drop | 데이터 `libero.py:333-335`(cam=wrist), 모듈 `train_phase2.py:523-537`, 삽입 `:645-648` |
| flow 타깃 | lat_target = g(A_fut, z_cur) | 1024 | g-공간 (**불변**) | `train_phase2.py:665-667` |
| x0 | g(A_past) 토큰, x0_std 스칼라 | 1024 | 잠재 타깃 std (**불변**) | `policy.py:103-121`, `train_phase2.py:564-570` |
| 디코딩 | ahat = h(ζ̂, z_cur) | 112 | frozen h (**불변**) | `train_phase2.py:668` |
| n_tokens | 3+lang+wrist+Kg = **9** | — | — | `train_phase2.py:540` (산식 기존재) |

**W-B 추가 홉** (guarded `module.wrist_delta: true`):

| # | 토큰 | dim | 정규화 | 변경 |
|---|---|---|---|---|
| 10 | w_tok = (p̄(t) − p̄(t−span)) · σ_ref/σ_Δ | 1024 | **N2** 2-스칼라 재척도 (train 통계, ckpt dict `wrist_delta_std` 저장) | 데이터: `libero.py:333-335` 인접에 guarded 1배열 추가 `np.stack([D[t].mean(0) − D[max(t−span,0)].mean(0)])` (n,1024); 학습: **[AMEND A2 — canonical 순서 확정] base(#1–5) → grid(#6–9) → w_tok(#10, 열 항상 마지막)** — grid 토큰 append **직후**에 1토큰 append (base 리스트에 넣지 않음; 구판의 ":639-641 base 뒤 append" 표기는 #6 해석과 모순이라 폐기). train/rollout 토큰-스택 동형성은 오프라인 스모크 `scratchpad/test_wa_token_order_smoke.py`(CPU, 합성 텐서)로 검증 — G1 전 필수 통과; n_tokens=10 |

### 3.3 롤아웃 (`rollout_sim.py`) — W-A/W-B 경로

| 스텝 | 연산 | dim | 코드 |
|---|---|---|---|
| 재계획마다 | wrist 프레임 → SigLIP2 인코딩(기존) | 1024 | `:185-190` encode_wrist (비dual: main clip — 불변) |
| 재계획마다 | wrist 프레임 → DINOv3 pool2 인코딩 → GridObs → 4토큰 | 4×1024 | `:207-218` grid_toks — **wrist_cam 분기 기존재** (`if wrist_cam and cam == wrist_cam`) |
| W-B | patch-mean 링버퍼 deque(maxlen=span//H+1) 유지, w_tok 계산·재척도 | 1024 | zw_hist 관례 미러(`:246-248`); grid 인코딩 재사용이라 **추가 인코딩 0회** |
| 토큰 스택 | [zp, zc, a_emb, lang, wrist_sig, gp×4(,w_tok)] zero-pad 불요(전부 1024) | 9(10)×1024 | `:285-295` |
| 검증 | 신규 앵커 로드 시 VRAM: SigLIP2-L + DINOv3-L fp32 + 정책 < 6GB | — | 패치 문서 §6.3 실측 계열 |

### 3.4 W-C phase2/롤아웃 (`train_phase2.py run_dual :55-291`, `rollout_sim.py :256-274`)

| 홉 | 텐서 | dim | 정규화 | 변경 |
|---|---|---|---|---|
| 토큰열 | [zp_m, zc_m, a_emb, **zw_sig**, lang] (DINOv3-CLS prev/cur 토큰 **제거**) | 5×2048(pad) | zw_sig=SigLIP2-wrist raw | 결함 ⑤ 격리: 조건화를 단일-스트림 baseline과 동일하게 — 이 팔의 유일 기전 = 타깃-측 ζ_wrist. 데이터: `libero.py:319-324`에 guarded로 SigLIP2-wrist cur 1배열 병기; `train_phase2.py:168-173` 토큰 구성 교체 |
| flow 타깃 | [ζ_m; ζ_w] 표준화 공간 | 2048 | **N3** (phase1 buffer 통과) | `:177-178` 불변 (ae.encode가 표준화 공간 출력) |
| x0_std | per-dim (dc,) | 2048 | **N4** | `:155-159` `lt.std(0)` + `policy.py:97` guarded shape |
| 디코딩 | ae.decode(ζ̂, z_m, z_w) | 112 | h 입구 LN(`networks.py:50`)이 균등 분산 블록을 받음 (결함 ③ 해소) | 불변 |
| 롤아웃 | 토큰 교체 대응(ckpt config로 분기), zw_hist는 h 상태용으로만 유지 | — | — | `:256-274` guarded 수정 |

---

## 4. (e) SigLIP2+DINOv3 기본 융합과의 합성 — 전체 스택 1구성 권고

- **기전 스크리닝 기질 = large256-single** (헤드룸 ~15pp; concat 97.5 위 검정은 검정력 0 —
  BRIEF §4 suite-천장 제약 그대로). W-A/W-B/W-C/matched-base 전부 이 기질.
- **suite 확전 조항 (DESIGN_wrist_v2 §3 복원 — [AMEND R4])**: **승자 확증 시 libero_goal/
  libero_object(파지 난이도 잔존 suite)로 확전 필수**; **전 팔 널 시 goal 1-suite 스팟체크**로
  "spatial-한정 널"과 "일반 널"을 분리 (large256 헤드룸 ~15pp의 실패 구성 미분석 → suite-한정
  false null 위험의 방호).
- **승자 승격 기질 = concat-base(2048)**: phase1 `fobsfusion_concat` 재사용, 토큰은 §3.2와
  동일하되 zero-pad(1024→2048, `train_phase2.py:636-638` 기존 규약), GridObs out_dim=2048
  (+1.05M→2.1M). avg-base 는 언어-우선 폴백: concat+wrist 스택이 co-기준(+70pp)을 못 넘으면
  (concat 단독이 +69로 경계) avg(+74) 위에서 재확증 — **avg vs concat 선택 기준을 "언어
  공동기준 통과 여부"로 사전 고정**한다 (SR만으로 고르지 않음).
- **P-A crop과의 합성**: crop은 main 카메라 전처리·별도 cache_key — wrist 경로와 완전 직교
  (§2.1). 현재 학습 중인 crop 팔의 승자가 나오면 concat-승격 시점에 main 앵커 설정만 갈아끼움.
  wrist는 무crop 고정.
- **P-B LangSelPool과의 합성**: **wrist 패치는 LangSelPool 기계를 재사용하지 않는다(v1 기각)**.
  사유: ① 지시문("black bowl on plate")의 지시 대상은 agentview 소관 — wrist 근접뷰는 언어
  의미 희박(LIT §3.3), 텍스트-쿼리가 고를 것이 없음. ② +5.9M vs +1.05M. ③ 팔당 변경 하나.
  B(main)와 W-A(wrist)가 **각자** 이긴 뒤에만 합성 팔 1개(main=LangSelPool 패치 + wrist=grid
  4토큰; 구현상 `module.obs`/`grid_obs` 상호배타 assert `train_phase2.py:409` 완화 필요 —
  그때 guarded로). W-A의 언어-무관 wrist 토큰은 B의 대조군 역할도 겸한다(패치 문서 §0 논리의
  wrist 판).
- **권고 최종 스택** (모든 게이트 통과 가정): phase1 = concat-2048(crop-승자 반영) /
  phase2 토큰 = [z_prev, z_cur, g(A_past), lang, wrist_sig, wristpatch×4 (+w_tok if W-B GO)] /
  no-aug 클린 밴드 / S1b off / FlowPolicy 불변(d1536, x0=past).

---

## 5. 사전등록 — 게이트, 프로토콜, 예측, 비용

### 5.1 오프라인 게이트 (롤아웃 전, 순서 고정 — 전부 CPU/저가)

| ID | 검정 | 게이트 | 비고 |
|---|---|---|---|
| **G0** (W-B 선행) | ridge [Δz̄_w,past, z̄_w] → A_fut per-dim R², 대조 [Δz̄_main,past, z_main] | gripper-dim 우위 **≥ +0.05** → W-B GO; 미달 → W-B 학습 없이 폐기 | pool2 캐시에서 CPU ~1h; DESIGN_wrist_v2 G3a의 **인과(과거) 변위판**. 국면분해(파지창 vs 이송) 동시 산출 → 4b 근거 무료 확보 |
| **G0-A** (W-A 선행, 학습-전 정보성 — [AMEND R2], 스펙만·스크립트는 착수 시 구현) | ridge 한계 gripper-dim R²: [p̄₁..₄ ⊕ z_w,sig ⊕ z_main] vs [z_w,sig ⊕ z_main] → A_fut | 한계 우위 없음(음성)이면 **W-A 학습 전 폐기 가능**(절약 원칙 확장); 우위 실재 시 W-A GO 보강 | "wrist 패치 4토큰에 기존 토큰을 넘는 정보가 있는가"의 0원 선검정 — pool2 캐시, CPU ~1h, G0와 **같은 배치**로 실행 (VERIFY R2) |
| **G1** | phase2 val act-R² ≥ base−0.01 (large256 0.655) | 미달 → 학습 문제, 롤아웃 금지 | 전 팔 |
| **G2** | wrist 토큰군 zero-ablation @val: wristpatch×4 / wrist_sig / w_tok 각각 0화 → Δ(act loss) | \|Δ\|≈0인 군 = 미사용 → 해당 팔 no-go | **"ζ_wrist 절제의 신설계 재실행"** — 조건화 판 대응물. wrist 교훈(AUDIT §4b) 그대로 |
| **G2-C** (W-C) | 표준화 dual에서 zero-ablation 재실행 | 사전등록 P1-1: ΔR²(A−B) ≥ 1.5×(0.179→≥0.27) | WEEK0 §6 산식·split 동일 재현 |
| **G3** | 언어 민감도: 지시문 스왑 시 ζ̂ 변화량이 wrist 토큰 추가 후에도 유지 | 스왑-Δcos 중앙값이 base 대비 −20% 이상 축소 시 희석 경고 등재 | 1c 형제쌍 프로토콜 오프라인판 |

### 5.2 matched-baseline 프로토콜 (§1 #11 재발 방지 — 이번 캠페인의 헌법)

1. large256-single **phase2를 같은 주·같은 리비전·같은 split(seed 동일)에서 재학습** (phase1은
   전 팔 공유 ckpt라 재학습 불요 — 공유 자체가 pairing. [AMEND A5] 단 이 문장은 **W-A/W-B
   한정** — W-C는 dual phase1을 새로 학습(1시드)하므로 "phase1 공유 = pairing"이 성립하지
   않고 phase1 시드 분산이 confound로 남는다. **W-C 승리(F1 발동) 시 dual phase1 재학습
   1회를 확증(50roll×3seed) 요건에 포함**한다).
2. launch 스크립트에 **ckpt 절대경로 실존 assert** 내장 (`test -f` 실패 시 즉시 중단 — 경로버그
   `checkpoints/` vs `checkpoints/grid/`의 구조적 차단).
3. 롤아웃: 태스크-단위 retry-supervisor(≤3회, osmesa 확률 세그폴트 전제), `--run-tag`에 ckpt 태그
   포함, per-episode JSONL fsync(`rollout_sim.py:96-114`), **arm×mode당 200ep 완주**, 팔 간 동일
   init_states(paired), 실행 창 인터리브(base↔arm 교대)로 박스 상태 confound 제거.
4. 판정: per-task paired bootstrap 10k, 스크리닝 10task×20roll×{correct,wrong} → 이중 기준
   **paired ΔSR SIG>0 AND correct−wrong ≥ +70pp** → 통과 팔만 50roll×3seed 확증.
5. [AMEND A4 — c−w 스크리닝 유보 규칙 (패치 문서 §8-6 이식)] 20roll 스크리닝의 c−w는 노이즈
   ±10pp — **게이트(승리 +70 / §6-F3 +60) ±5pp 이내의 c−w 결과는 pass/fail 판정 유보**하고
   50roll×3seed 확증으로 승격해 재판정한다 (예: c−w 68은 즉시 NO-GO가 아니라 유보 → 확증에서
   판정). kill/pass 즉결은 게이트에서 5pp 초과로 벗어난 경우에만.

### 5.3 예측 (upgrade_ledger 등재용, 승산 포함)

| 팔 | H-win (기전 참) | H-null (포화 참) | 사전 승산 |
|---|---|---|---|
| W-A | SR 85–88 → **+2~+6pp** (파지·근접 국면 태스크 주도), c−w **+70~76** (wrist 패치는 언어-희박이라 희석 완만, group-drop 완충) | ΔCI∋0 AND c−w 유지 → "wrist 조건화는 SigLIP2 1토큰으로 포화" — 4c 종결, §정직 섹션 채택 | 中 (−34.8pp·AUROC 0.89·문헌 10–45pp가 정보 실재를 지지하되, 한계효용 체감이 기본 기대) |
| W-B | W-A 대비 +0~+3pp, t4류 파지 태스크 집중 | G0 탈락(학습 전) 또는 ΔCI∋0 → "측정 변위도 조건화에선 무익" — 변위 서사 종료 | 中-低 |
| W-C | **ΔCI∋0 (널 유지)가 본 설계의 예측** — P1-2: 표준화를 고쳐도 타깃-측은 무익이어야 지도가 참 | (예측 반대로) ΔCI>+2pp → 스케일이 유일 병목이었다는 발견 — §6-F1 발동 | 널 예측 승산 高 |

### 5.4 비용·파라미터 예산

| 항목 | GPU-h | 신규 학습 파라미터 |
|---|---|---|
| pool2 wrist 캐시 (1회) | 0.3–0.5 | — |
| G0–G3 프로브 | ≤0.5 (대부분 CPU) | — |
| matched base phase2 재학습 | 1.5–2 | 0 |
| W-A phase2 | ~2 | +7.4M (모듈 1.05M + ctx 5→9tok 확폭 6.3M; 126.9M 대비 +5.8% — 명시 보고, K=4로 F3의 +9.4M confound 회피) |
| W-B phase2 | ~2 | W-A +1.6M(ctx 1토큰), 모듈 +0 |
| W-C phase1+phase2 | 3+5 ≈ 8 | 학습 파라미터 +0 (buffer 3: dz_std×2, x0 per-dim) |
| 스크리닝 롤아웃 4팔×2모드×200ep | ~24 (CPU-지배, wall ×1.5) | — |
| **스크리닝 합계** | **≈ 40** | — |
| 확증 (승자만 3seed×50roll) | +45 | — |
| **총 상한** | **90 GPU-h** — 초과 시 미착수 팔은 기록만 남기고 종료 | — |

선행 조건: 원격 박스 **CUDA 복구** (WEEK0 §0 — 모든 학습 셀의 게이트). 순서: 캐시+G0(GPU
복구 전 CPU 가능분 선실행) → matched base + W-A 동시 → G1/G2 → W-A 롤아웃 ‖ W-B 학습(G0
통과 시) → W-C(여력·달력 순).

---

## 6. 반증 조건 — 무엇이 이 설계를 죽이는가 (사전 서약)

- **F1 (중심 명제 반증)**: W-C(타깃-측, 수리판)가 paired SIG-승리하고 W-A/W-B가 널이면 —
  "wrist의 가치는 조건화에 있다"는 본 설계의 핵이 틀린 것. 그 경우 dual+표준화를 채택하고
  §1의 논리(출력-재부호화 논증)를 공개 폐기한다. 이것이 W-C를 팔에 포함한 이유다.
- **F2 (사용 실패)**: G2에서 wristpatch 토큰군 \|Δ\|≈0 — 정책이 새 토큰을 안 씀. 롤아웃 진입
  금지, "조건열 자유도는 이미 포화" 기록. (재개 조건: main-토큰 dropout 팔 — §2.3.)
- **F3 (언어 화폐 훼손)**: 어느 팔이든 c−w < +60pp → wrist-DINOv3 함량 증가가 §5 tradeoff
  곡선을 wrist 축에서 재현 — SR과 무관하게 NO-GO, 단일 토큰 구성 회귀. (곡선의 wrist-축
  데이터 포인트로 논문 회수. 단 20roll에서 게이트 ±5pp 이내면 §5.2-5 유보 규칙 적용.)
- **F4 (F3-echo)**: W-A가 base 대비 SIG **하락** — "richer obs 유해"가 모듈 결함 없는 K=4
  wrist에서도 재현되는 것 — F3 일반화의 강한 증거로 격상 기록 (예측된 음성 아님을 명기).
- **F5 (기반 붕괴)**: matched base가 provenance 레짐에서 85–88을 재현 못 하면(UNTRACED 10/10
  이력) 모든 팔 판정 중단 — 기준선 재정박이 최우선.
- **전 팔 널 시 종결 서사**: G0 탈락 + W-A/W-B/W-C 모두 ΔCI⊂(−2,+2) → "이 frozen 기질에서
  wrist-as-single-condition-token이 최적"이 오프라인+폐루프 양면 확정 (DESIGN_wrist_v2 §4의
  종결 증거 산출 조건과 동일; §4 확전 조항의 goal 1-suite 스팟체크 병기로 "spatial-한정 널"과
  구별) — 삽입점 지도의 강한 한 행으로 논문 기여.

## 7. 정직 섹션 — 최강 반론과 본 설계의 응답

반론(DESIGN_wrist_v2 §4): "−34.8pp는 이미 회수된 가치다; Δz_wrist의 잔여 정보는 frozen 전역
인코더로는 원리적으로 못 뽑을 수 있다; 8방향 연속 음성 프라이어; 남은 실패는 저수준 제어 한계."
응답: 본 설계는 그 반론이 **참일 경우의 지출을 최소화**하도록 게이트를 배열했다 — G0는 학습
0원에 W-B를 자르고, G2는 롤아웃 0원에 W-A를 자르며, W-C는 어차피 지도 완결에 필요한 통계
성립판이다. 반론이 참이면 총 지출 ≈ 캐시+프로브+학습 3런(≤15 GPU-h)으로 "포화 확정"이라는
출판 가능한 음성 결과를 얻고, 거짓이면 헤드룸 ~15pp 기질에서 paired 양성을 얻는다. 어느
쪽으로도 정보 손실이 없다는 것이 이 설계의 실질 가치다.

---
*[AMEND A1 리비전 주] A1 코드 배선(+16줄: 키 검증 +8 @:408, 옵션 전달 +8 @:541)으로 본문의
`train_phase2.py` 인용 중 **:407 이후 줄번호가 이동**했다: :433-437(dense 배관)→**:441-445**,
:523-537(GridObs 구성)→**:531-553**, :540(n_tokens 산식)→**:556**, :645-647/:645-648(UNGATED
삽입)→**:661-663**. :405(grid_cfg 파싱) 등 :407 이전 인용과 타 파일 인용은 불변.*

---
*불변식 준수 확인: byte-identity(전 신규 경로 guarded, 기본값 off=비트 동형 — `dual_stream`·
`wrist_token`·`grid_obs` 전례 규약), no-aug 클린 밴드, cache_key 분리(`dinov3-vitl16-256-pool2`
신설), retry-supervisor + run-tag provenance 의무, 폐루프 SR 단독 심판 + 언어 공동기준,
suite-천장 회피(large256 스크리닝 → concat 승격), 팔당 변경 하나(W-A=조건 토큰 추가만 /
W-B=+토큰 1 / W-C=수리 번들, 신기전 0).*
