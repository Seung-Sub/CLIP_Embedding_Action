# DESIGN — W-D(보조 손목-변위 예측) · W-A′(SigLIP2-공간 손목 패치) 사전등록 v1 (2026-07-22)

*작성: 아키텍처 셀 설계 에이전트. 목적: PI의 잔존 직관 두 개 —
① "손목도 **추론**되어야 한다", ② "손목은 view-적합 전처리와 함께 **SigLIP2 통일 공간**에
살아야 한다" — 를 닫힌 음성 지도(W-B/W-C 널, M-B JOINT_REQUIRED, M-C NO_HEADROOM,
Phase-A 프라이어)를 **위반하지 않는 마지막 살아있는 기전**으로 공정하게 검정하는 셀 2개의
사전등록. 근거: `docs/DESIGN_wrist_fusion_unified_v1.md`(W-A 규격) ·
`docs/RESULT_wrist_screening.md`(W-A +5.5 / W-B·W-C 널) ·
`docs/DESIGN_dualpolicy_dynamic_weighting_v1.md`(M-B 0.662 / M-C 폐쇄) ·
`docs/AUDIT_negative_results_2026-07-18.md` · `docs/BRIEF_wrist_design_inputs.md` ·
소스 실사 `src/models/{networks,policy,obs_fusion}.py`, `src/training/train_phase2.py`,
`src/core/anchor.py`, `src/data/libero.py`, `src/eval_libero/rollout_dataset.py` (file:line 대조 완료).*

---

## 0. 결정 요약 — 셀 2개, 각각 "팔당 변경 하나"

| | 구성 | 한 줄 정의 | 검정하는 PI 직관 | 신규 학습 파라미터 |
|---|---|---|---|---|
| **W-D "AuxΔw"** | W-A 불변(토큰열·ζ·g/h·x0 전부) + 정책 ctx 트렁크 위 **보조 헤드**가 미래 손목 변위 **Δz_wrist(t→t+k), SigLIP2 공간**을 예측 — **순수 손실-측**, 출력은 h/액션 경로에 단 1비트도 진입 금지 | "손목 추론"의 유일하게 살아있는 기전 = **표현-형성(representation shaping)**. 타깃-측(W-C)·관측-토큰(W-B)이 아닌 **학습-시 미래 관측 감독** | ① wrist inference | 보조 헤드 ≈ **+1.32M** (학습 전용, 롤아웃 diff 0) |
| **W-A′ "SigPatch"** | W-A의 DINOv3 wrist 패치를 **SigLIP2-large256 wrist 패치 토큰**(16×16 → 2×2 pool → 4토큰)으로 교체 — GridObs 기계·ln/tok_drop/group_drop·토큰 수·파라미터 전부 동일(1024d 정합) | W-A 이득의 기전 귀속: "DINO 기하"인가 "임의 손목 공간 디테일"인가 + 손목 경로의 **의미 접근성**(SpLiCE/TextSpan 프로브 커버리지) | ② SigLIP2 통일 공간 | W-A와 **파라미터-정확 일치** (+7.4M, 인코더 정체성만 교체) |

공통 심판: 폐루프 paired SR AND 언어 공동기준(c−w ≥ +70pp, A4 유보 밴드 65~75) —
헌법(§5.2, DESIGN_wrist_fusion_unified) 그대로. 실행은 **W-A 확증(50roll×3seed) 종료 후**.

---

## 1. 배경 — 무엇이 닫혔고, 무엇이 열린 상처인가

**닫힌 것 (본 설계가 절대 재진입하지 않는 지형):**

1. **타깃-측 ζ_wrist 영구 종결** — W-C(표준화 수리판)조차 폐루프 −4.5pp 널
   (RESULT §6.2). 재부호화 논증(E7; DESIGN_wrist_fusion §1·AUDIT §4): 롤아웃에서
   ζ̂_wrist는 정책 출력이므로 조건 토큰에 이미 있는 정보의 재부호화일 뿐, 신규 관측
   정보 0. 예외 조항 없음(DESIGN_dualpolicy §6-1).
2. **분리 flow → joint h 금지** — M-B S_NN **0.662** > 0.60 = JOINT_REQUIRED. h는
   두 ζ 블록의 결합 정합성을 읽는다. 따로 예측된 조각을 h에 먹이는 어떤 설계도 금지.
3. **동적 재가중 무-headroom** — M-C oracle 이득 +0.000025 (기준 +0.01의 1/400).
4. **측정 변위 토큰(W-B) 널** — W-A의 하위호환(+4.5, c−w +63.0 미달).
5. Phase-A 프라이어: 복잡화 일관 무익, 가치는 관측 삽입점+전처리.

**열린 상처 (본 설계가 겨냥하는 곳):**

- **언어 축**: W-A 스크리닝 c−w +65.5(유보), 확증 **s1 중간판독 correct 91.8 / wrong
  30.2 → c−w +61.6** — 유보 밴드(65~75) **아래**로 내려앉음(원격 중간값, 로컬 1차
  아티팩트 미도착 — s2/s3 완료 시 재판정. 이하 "s1 잠정"으로만 인용). SR 이득은
  유지되는데 언어 격차가 확증에서 무너지는 그림이면, **W-A는 F3(c−w<+60) 근접 위험**.
  → 손목 이득을 지키면서 언어를 회복할 저가 레버가 필요하다. W-A′의 존재 이유 절반,
  W-D의 "언어-중립" 속성이 매력인 이유가 이것.
- **기전 미귀속**: W-A의 +5.5가 "DINO 기하 특이"인지 "아무 손목 공간 디테일"인지
  미분리 — 어느 답이든 논문 문장 (W-A′).
- **"손목 추론" 질문 자체가 미종결**: W-C는 *삽입점*(타깃-측)의 실패지 *감독 신호*
  (미래 손목 변위)의 실패가 아니다. 미래 변위를 **손실-측에서만** 쓰는 기전은 한 번도
  검정된 적 없다 (W-D).

---

## 2. Cell W-D — 보조 손목-변위 예측 (AuxΔw)

### 2.1 기전 논증 — 왜 이것이 재부호화 논증(E7)에 걸리지 않는가 (정밀 서술)

재부호화 논증이 죽이는 것은 "**시험 시점에** 정책이 스스로 생성한 손목 표현을 행동
경로가 소비하는" 설계다(W-C: ζ̂_wrist가 h 입력). W-D는 정확히 그 반대다:

- **보조 타깃 Δz_wrist(t→t+k)는 학습-시 미래 관측**이다 — 데이터셋에는 있으나 조건
  토큰열(z_prev, z_cur, a_emb, lang, wrist_sig(t), wristpatch(t))에는 **없는** 정보.
  즉 보조 손실은 조건화가 갖지 못한 감독을 **그래디언트로만** 주입한다. 이것이 W-C와의
  핵심 기전 차이: W-C도 같은 부류의 미래 신호를 썼지만 그것을 **통화 ζ(=h가 디코딩해야
  할 타깃)에 실었고**, h는 결합 정합성에 민감하며(M-B) 시험 시점 ζ̂_wrist는 재부호화라
  정보 0이었다. W-D는 ζ/g/h/x0/토큰열을 **단 1비트도 바꾸지 않고**, ctx 표현이 "미래
  손목 변위를 선형 근사로 담도록" 기울이기만 한다.
- **시험 시점에 보조 헤드는 실행조차 되지 않는다** (롤아웃 코드 diff 0). 예측이
  틀려도 행동에 영향을 줄 채널이 물리적으로 없다 — M-B의 JOINT_REQUIRED와도 무관
  (새 스트림이 h에 들어가지 않으므로).
- 남는 유일한 위험은 "보조 그래디언트가 ctx를 CFM에 불리한 방향으로 기울이는" 간섭
  — 이것은 G1(act R² 비열화)이 직접 감시한다.

**사전 서약**: W-D의 aux 출력(또는 그 파생물)을 토큰열·h 입력·flow 타깃에 연결하는
어떤 후속 변형도 이 사전등록의 범위 밖이며, 재부호화 논증·M-B 위반으로 **설계 없이
기각**된다.

### 2.2 타깃·캐시 실사 — 신규 인코딩 0회

- **타깃 정의**: `Δz_w,sig(t) = Z_w,sig(t+span) − Z_w,sig(t)`, span=16(0.8s, A_fut·ζ
  타깃과 동일 지평), Z_w,sig = **SigLIP2-large256 pooled wrist**(main 앵커, raw
  normalize=false — 현행 wrist_sig 토큰과 동일 통화). PI 지시("SigLIP2 wrist pooled
  displacement") 그대로 — DINOv3-CLS(미검증, BRIEF 결함 ⑦)는 쓰지 않는다.
- **캐시 존재 확인**: 단일-스트림 경로가 wrist 프레임을 main 앵커로 인코딩·캐시한다
  (`src/data/libero.py:347` `embeddings(clip, ep, self.wrist_camera)`); wristpatch·
  matchedbase 학습(2026-07-20, R² 0.669/0.655)이 `wrist_token: true`로 완주했으므로
  **SigLIP2-large256 wrist pooled 캐시는 원격에 기존재**
  (`/data2/clip_ws_cache/cache/libero_emb_large256/siglip2-so400m/joint/raw/*_eye_in_hand_rgb.npz`).
  캐시는 에피소드 전 프레임 [T,1024]를 담으므로 Z_w(t+span)도 이미 있다. **신규 인코딩
  0회.** 착수 시 launch 스크립트에 캐시 디렉터리 실존 assert(§5.2-2 관례) 포함.
- **표준화**: 타깃은 스칼라 z-score `Δ̃ = Δz_w,sig / σ_Δ`(train std) — 결함 ①의 교훈
  (스케일 방치 금지). σ_Δ 는 ckpt dict `"aux_wdelta_std"`에 저장(W-B `wrist_delta_std`
  관례 미러, `train_phase2.py:932-933`).

### 2.3 손실 배치 — primary (a) ctx 트렁크 / ablation (b) ζ̂ (flow 출력)

**(a) PRIMARY — ctx 트렁크 위 MSE 헤드** (채택, λ=0.1 기본 · λ=0.3 스윕 팔):

```
aux_head = Sequential(LayerNorm(d_model=1536), Linear(1536,512), GELU, Linear(512,1024))  # ≈1.32M
l_aux    = MSE(aux_head(ctx), Δ̃z_w,sig)          # ctx = model.ctx(toks.flatten(1))
total    = 0.5·l_fm + 1.0·l_act + λ_aux·l_aux
```

- **삽입점**: `train_phase2.py` is_flow 잠재 분기 `:753-772`, `zeta, l_fm =
  model.fm_and_sample(...)`(`:755`) 직후. ctx는 `model.ctx(toks.flatten(1))`로 재계산
  (policy.py 무변경 최소-diff; `fm_and_sample`이 내부에서 이미 ctx를 계산하므로
  (`src/models/policy.py:143`) 동일 파라미터에 그래디언트가 정상 누적 — 배치당 ctx
  forward 1회 추가 비용만. 원하면 `fm_and_sample(return_ctx=True)` guarded 확장으로
  제거 가능하나 필수 아님).
- **헤드 구성·옵티마이저**: grid_obs 모듈 구성부(`:595-617`) 뒤에 guarded 생성, Adam
  파라미터 목록(`:662-668`)에 aux_head 추가. ckpt에 `"aux_wdelta"` state 저장(`:928-931`
  dict 확장; off면 키 미생성 = dict 불변). 롤아웃 로더(`rollout_dataset.py`)는 지정
  키만 읽으므로 **롤아웃 코드 무변경**.
- **데이터 배선**: `build_policy_samples`에 guarded kwarg `wrist_fut_delta=True` —
  단일-스트림 wrist 분기(`libero.py:341-348`)에서 이미 로드된 Zw로
  `np.stack([Zw[t+span] − Zw[t]])` (n,1024) 1배열을 **항상 최종 마지막**에 append
  (W-B `obs_delta` 관례 `:355-360` 미러; no-aug 클린 밴드 전제 — `wrist_aug>0`이면
  assert 중단). train_phase2 파싱은 `Wx.pop()` 순서 사전 고정: **aux(최후) → wdelta →
  dense** (`:499-502` 미러). W-D 팔은 wrist_delta off이므로 실충돌 없음.
- **config 키 loud-fail**: 신설 블록 `module.aux_wdelta`는 `_AUX_KEYS = {"lam",
  "placement", "hidden"}` assert를 grid_obs `_GRID_KEYS`(`train_phase2.py:444-448`)와
  동일 규약으로 신설 — VERIFY A1(무음 no-op 금지) 상속.
- **그래디언트 상호작용** (사전 명시): l_aux의 그래디언트는 ctx 트렁크(LN+사영+
  ResidualBlock×2, d1536)와 토큰 생산자(GridObs proj·in_ln)로 흐른다.
  **v_net(속도장)에는 흐르지 않고**(placement=ctx), h·g는 동결이며 a_emb는 no_grad
  (`:789-790`) — **통화(ζ)·디코더·x0 불변 보장**. 항별 grad-norm(wandb: grad_aux/
  grad_lat/grad_act) epoch 로깅 의무 — 결함 ④ 교훈. λ=0.1이면 표준화 공간 MSE가
  주손실 대비 종속 스케일임을 로그로 확인(안 되면 λ=0.03 폴백 1회 허용, 사전 기록).
- **(b) ABLATION — ζ̂(ODE 샘플) 위 헤드**: `aux_head_z(zeta)`(`:755`의 zeta, 그래디언트
  보유) → 같은 타깃. 이 판은 v_net까지 그래디언트가 흘러 **flow 출력 다양체 자체를
  기울인다** — h가 결합 코드를 읽는다는 M-B 소견상 위험이 더 크고, 그래서 primary가
  아니라 ablation이다(기전 대조: "표현-형성은 ctx에서 충분한가, flow까지 가야 하는가").
  (a) λ0.1이 G2-D를 통과하고 SR이 널일 때만 학습(조건부 1런).

### 2.4 게이트·예측·킬 (사전등록)

| ID | 검정 | 게이트 | 비용 |
|---|---|---|---|
| **R-D0** (학습-전, CPU) | ridge 2종 @val split 동일 재현: ① state-only [z_main(t) ⊕ z_w,sig(t)] → Δ̃z_w,sig = **r_state** (G2-D의 기준 수치 — 결과 JSON보다 먼저 prereg 파일 기록, mtime 규약) ② full [① ⊕ DINOv3 wrist pool2 p̄₁..₄] → **r_full** | r_full − r_state < **+0.02** → 조건열이 이미 담은 것 이상을 aux가 배울 신규 정보 없음 → **학습 전 폐기** (G0 관례의 W-D판) | CPU ~1h, 캐시만 |
| **G1** | phase2 val act R² | ≥ **0.66** (W-A 0.669 − 0.01) — 미달 = aux 간섭, 롤아웃 금지. λ0.3이 λ0.1보다 낮으면 λ0.3 즉시 폐기 | 학습 로그 |
| **G2-D** | aux 헤드 val R̄²(1024dim 평균) | > **r_state + 0.02** — 미달 = 헤드가 상태-선형 이상을 못 배움("aux 감독이 비었음") → 롤아웃 금지, "손실-측도 무익" 기록 | 학습 로그 +5분 |
| **G2-D2** (진단, 무게이트) | M-A식 국면분해: aux R²의 파지창 vs 이송 우위 | 기록만 — "grasp-국면 표현 예리화" 서사의 직접 증거/반증 | CPU |
| **G3** | W-A G2 zero-ablation 재실행 (wristpatch4 / wrist_sig 절제 Δ) | W-A(−0.015/−0.025)와 비교해 토큰 사용이 **증가**했는지 — aux의 표현-형성 기전 부수 예측 | CPU |
| 스크리닝 | 20roll×2모드, paired vs matchedbase **AND** vs W-A(같은 창 인터리브) | 승격 = ΔSR(vs base) CI>0 AND c−w ≥ +70(A4 유보 밴드) AND ΔSR(vs W-A) ≥ 0 | ~6 GPU-h |

**예측 (정직한 프라이어 포함):**

- **H-win** (승산 中-低): grasp-국면 표현 예리화 → SR은 W-A 대비 +0~+3pp(t3/t4/t9
  집중), **언어는 중립** — aux는 조건화를 건드리지 않으므로 c−w를 훼손할 채널이
  없다(±3pp 밴드 예측). W-A의 언어 상처가 확증에서 유지된다면, "SR을 더 얹으면서
  언어를 더 깎지 않는" 유일 후보라는 것이 W-D의 매력이다. 단 정직하게: aux는 언어를
  **회복시키지도 못한다** (희석의 원인인 wristpatch 조건 토큰이 그대로이므로) —
  언어 회복은 W-A′의 몫.
- **H-null** (기본 기대, Phase-A 프라이어): G2-D는 통과하되(오프라인 예측력은 실재)
  ΔSR CI∋0 — "표현은 바뀌었으나 폐루프가 보상하지 않음".
- **무엇이 "손목 추론"을 최종 종결하는가 (사전 서약)**: G2-D **통과**(aux가 상태-선형
  이상을 실제로 배움)에도 폐루프 ΔSR(vs W-A) CI∋0이면 — 미래 손목 신호의 3개 기전
  전부가 실측 소진된다: **타깃-측**(W-C, 표준화 수리 후에도 널) · **관측-토큰 측**
  (W-B, 측정 과거 변위 널) · **손실-측**(W-D, 표현-형성 널). 이후 "손목 추론" 제안은
  이 3분류 밖의 신규 기전 클래스를 먼저 명시하지 못하면 설계 없이 기각한다. 이것이
  4c 서사의 완결 행이다. (반대로 G2-D **미달**이면 "종결"은 선언할 수 없고 — 감독
  신호가 전달 실패한 것 — 더 강한 헤드/λ 1회 재시도까지만 허용.)
- **폴백 (조건부 1런)**: W-A 확증이 SR 축까지 전면 실패하면 W-D의 기지는 사라지지만,
  aux는 grid_obs와 직교하므로 **matchedbase + aux**(wristpatch 토큰 없이 wrist_sig
  1토큰 + aux 감독)로 강등 실행 가능 — "순수 손목-추론" 검정으로서 오히려 깨끗함.
  이 폴백도 같은 게이트를 상속한다.

### 2.5 config 스케치 — `configs/phase2_libero_large256_wristpatch_auxwd.yaml`

`phase2_libero_large256_wristpatch.yaml`(W-A)과 **diff 3곳만**:

```yaml
module:
  # … W-A와 동일 (lang_token/wrist_token/grid_obs 블록 그대로) …
  aux_wdelta:            # ★W-D 신설 블록 — train_phase2 _AUX_KEYS loud-fail 필수
    lam: 0.1             #   λ 스윕 {0.1, 0.3} — 팔별 config 분리
    placement: ctx       #   ctx(primary) | zeta(ablation, 조건부)
    hidden: 512          #   1536→512→1024 (≈1.32M, 학습 전용)
train:
  checkpoint: ~/clip_ws/checkpoints/phase2_libero_large256_wristpatch_auxwd01.pt
wandb: {…, run_name: phase2_wristpatch_auxwd01}
```

byte-identity: `aux_wdelta` 블록 부재 시 로더 kwarg·pop·헤드·손실·ckpt 키 전부
미생성 — 기존 config 비트 동형 (guarded 관례; HEAD 대조 스모크 의무).

---

## 3. Cell W-A′ — SigLIP2-공간 손목 패치 토큰 (SigPatch)

### 3.1 동기 — 기전 귀속 + 의미 접근성 (둘 다 결과 무관 논문 문장)

- **기전 귀속**: W-A의 +5.5가 (i) DINOv3 특유의 기하 표현 때문인가, (ii) "어떤
  인코더든 손목 뷰의 공간 디테일 4토큰"이면 충분한가. W-A′는 인코더 정체성 **단 하나**
  만 바꾼 파라미터-정확 매치 팔(둘 다 patch_dim 1024, GridObs proj 1.05M, 9토큰)이라
  이 질문의 깨끗한 판별 실험이다.
- **의미 접근성 (PI 직관 ②)**: wrist 토큰이 SigLIP2 통일 잠재공간에 살면 lang 토큰과
  같은 타워 출신이 되어 SpLiCE/TextSpan/adapter 프로브(기설비:
  `outputs/analysis/w4v3_p2_splice` 계열)가 손목 경로에도 적용된다 — 해석 루프가
  현재 커버하지 못하는 유일한 조건 경로가 손목이다. W-A가 이겨도 남는 "해석 불가
  DINO 블랙박스 토큰 4개"라는 논문 약점을 W-A′ 승리가 무료로 지운다.
- **언어 축 재판**: W-A의 열린 상처(c−w, s1 잠정 61.6)에 대한 양방향 가설이 성립하는
  유일한 저가 개입 (§3.4).

### 3.2 구현 실사 — 현존 seam과 필요한 배선 3건 (전부 소형·guarded)

| 항목 | 현황 (file:line) | 필요 작업 |
|---|---|---|
| SigLIP2 dense 노출 | **기존재** — `Siglip2Anchor.save_tokens`(`anchor.py:142,157-159`): vision tower `last_hidden_state` (N,P,d). train(`train_phase2.py:455-456`)·rollout(`rollout_dataset.py:103-106`) 양측이 grid 앵커 name==siglip2면 자동 활성 | 없음. 주의: SigLIP은 MAP-head 풀링이라 **CLS/register 프리픽스가 없음** — 256토큰 전부 패치, DINOv3식 프리픽스 드롭 불요(착수 시 P==256 assert로 확인) |
| 인코딩-시점 2×2 pool | **부재** — `pool_to`는 `Dinov3Anchor`(`anchor.py:301-306`)에만 있음 | `Siglip2Anchor(pool_to=…)` 신설: tokens에 동일 adaptive_avg_pool2d(16×16→2×2), id 접미사 `-pool2` → dense cache_key `siglip2-so400m-pool2/joint/raw` 자동 분리. pool은 선형이라 인코딩 시점 풀링=사후 풀링 수학 동일(W-A 검증 논리 재사용). `get_anchor`에 `kwargs["pool_to"]=a.get("pool_to")` 배선(`anchor.py:600-601` siglip2 분기) — **주의: get_anchor는 미지원 anchor 키를 무음 무시**하므로 배선 없이 config만 쓰면 풀 없는 65GB dense가 조립되는 사고 경로. 배선+스모크 필수 |
| 롤아웃 patch_dim | **결함 발견** — `rollout_dataset.py:109`가 `ganc.patch_dim`으로 GridObs를 재구성하는데 `Siglip2Anchor.patch_dim`은 클래스 상수 **1152**(so400m). large256 실폭은 1024 → ckpt `proj`(1024→1024)와 shape 불일치, load_state_dict에서 loud-fail | 1줄 수리: `Siglip2Anchor.__init__`에 `self.patch_dim = self.dim` (so400m엔 1152=1152 no-op, 기존 경로 byte-identity). 사전등록 배선 항목으로 명시 — 학습은 데이터 shape(`train_phase2.py:599`)를 쓰므로 **학습만 통과하고 롤아웃에서 죽는** 함정 |
| wrist dense 캐시 | DINOv3 pool2 캐시만 존재(`dense/dinov3-vitl16-256-pool2/pre/raw`) — **SigLIP2 wrist dense 캐시 없음** | 1회 빌드: 62,250 프레임 × 4tok × 1024d fp32 ≈ **1.0 GB**, 0.3–0.5 GPU-h (W-A 캐시와 동일 규모). fp16 인코딩 유지(fp32 규율은 DINOv3-NaN 특이 사항 — W5) |
| 전처리 | wrist 프레임 **crop 없음**, SigLIP2 native 256 resize (W-A 규약 동일 — P-A crop은 main 전용·음성 종결) | 없음 — "view-적합 전처리"는 근접뷰 무crop + native 해상도로 이미 충족 |

이외 전부 W-A와 동일: GridObs(ln/tok_drop 0.1/group_drop 0.1/init_std 0.02),
`_GRID_KEYS` loud-fail(`train_phase2.py:444-448`)은 키 변경이 없어 그대로 통과,
토큰열 9개, phase1 동결 공유, no-aug.

### 3.3 config 스케치 — `configs/phase2_libero_large256_wristpatch_sig.yaml`

W-A config와 **grid_obs.anchor 블록만** diff:

```yaml
  grid_obs:
    anchor:
      name: siglip2                                   # ★유일 변경: 인코더 정체성
      model_dir: google/siglip2-large-patch16-256     #   main 앵커와 동일 모델 (통일 공간)
      normalize: false                                #   dense tokens는 어차피 raw 반환 — cache_key 위생용 명시
      pool_to: 2                                      #   ★신규 배선 (§3.2) — cache_key -pool2 분리
    camera: eye_in_hand_rgb
    n_tokens: 4
    pool: avg
    ln: true                                          # N1 유지 — SigLIP2 패치 스케일도 동일 처리(공정 비교)
    tok_drop: 0.1
    group_drop: 0.1
    init_std: 0.02
train:
  checkpoint: ~/clip_ws/checkpoints/phase2_libero_large256_wristpatch_sig.pt
```

부수 배당: 롤아웃 VRAM에서 DINOv3-L 로드가 빠짐(grid 앵커 = SigLIP2 제2 인스턴스,
fp16 ~0.8GB) — 48GB 박스에서 여유 증가.

### 3.4 게이트·언어 양가설·예측 (사전등록)

| ID | 검정 | 게이트 |
|---|---|---|
| **R-A′** (학습-전 킬게이트, CPU) | 한계 ridge gripper-dim R² @동일 split: uplift_X = R²([X-patch4 ⊕ z_w,sig ⊕ z_main] → A_fut) − R²([z_w,sig ⊕ z_main]), X ∈ {SigLIP2-pool2, DINOv3-pool2} (DINOv3 판 = G0-A 규격 — 미실행분이면 같은 배치로 동시 산출) | **uplift_sig < 0.7 × uplift_dino** (≥30% 열세) → SigLIP2 패치가 파지 정보를 재료적으로 덜 담음 → **학습 전 폐기**, "DINO 기하 특이" 오프라인 판정으로 지도 기록 |
| **G1** | val act R² ≥ 0.66 | 미달 → 롤아웃 금지 |
| **G2** | zero-ablation: sig-patch4 절제 Δ(act R²) ≥ 0.005 (비교점: W-A −0.015) | \|Δ\|≈0 → 미사용, F2 |
| 스크리닝 | 20roll×2모드 paired (matchedbase + W-A 인터리브 3팔 창) | 아래 판정 분기 |

**본판정 분기 (사전 고정):**

- **이득 지속** — ΔSR(vs base) ≥ **+2.75pp**(W-A +5.5의 절반) & CI 규약 통과 →
  기전 = "손목 공간 디테일"(인코더 불문) + 의미 접근성 무료 → **논문 제안 팔을
  W-A′로 교체**(thesis-fit: 통일 SigLIP2 공간 + 해석 루프 완결). K=16 확전 논의는
  이 분기에서만.
- **이득 소멸** — ΔSR < +1pp 또는 CI∋0 → 기전 = "DINO 기하 특이" → W-A 유지 +
  "손목 해석성 비용" 문서화(한 줄 코스트로 논문 수록). 어느 쪽이든 기전 귀속 문장
  확보 — 이 팔은 지는 쪽도 산출물이 있다.

**언어 축 — 양가설 사전등록 (판별 readout 포함):**

- **H-L1 (희석 완화)**: SigLIP2 wrist 토큰은 lang 토큰과 같은 타워 출신 → ctx가
  언어와 **정합적으로** 결합, "장면만 보고 푸는" 언어-우회 경로가 약해짐 →
  c−w ↑ (W-A 대비 +5pp 이상 회복, wrong-모드 성공 하락).
- **H-L2 (의미 간섭)**: 언어-정렬 손목 토큰이 지시문의 **대체물**로 더 쉽게 기능
  (의미 공간이 겹치므로) → 언어-우회가 오히려 강화 → c−w ↓.
- **판별 readout**: ① c−w 총량, ② **태스크별 wrong-모드 성공률** — W-A의 지시-무시
  삼인방 **t0 95 / t2 85 / t4 65**(RESULT §4)가 기준선. W-A′에서 이 세 값이 동반
  하락하며 correct SR 유지 → H-L1; 동반 상승/유지 → H-L2. ③ G3 오프라인 지시문-스왑
  Δcos(W-A 설계서 §5.1 G3 프로토콜 재사용). H-L1이 실측되면 W-A′는 SR이 소폭 열세여도
  **언어 공동기준 통과가 우선**이라는 헌법(§4 avg-폴백 규칙과 동일 논리)에 따라
  확증 후보가 된다 — 이 우선순위를 지금 고정한다.

**정직한 프라이어**: 콜리그 계열 증거(S2 grasp-프로브 AUROC는 wrist **뷰**의 속성,
인코더 불문 0.886+)와 M-A("손목 우위는 그리퍼 채널 상시")는 H-지속 쪽을, LIT §3.3
(DINO dense 근접뷰 우위 일관)과 결함 ⑦ 교체 이력은 H-소멸 쪽을 각각 지지 — 사전
승산은 반반에 가깝고, 그래서 이 실험은 어느 결과든 정보량이 크다.

---

## 4. 공통 프로토콜·순서·예산

### 4.1 불변식 (전부 상속)

matched-baseline 헌법(§5.2: 같은 리비전 재학습·ckpt 절대경로 assert·retry-supervisor·
run-tag provenance·arm×mode 200ep 완주·paired bootstrap 10k·인터리브 창), A4 유보
규칙(c−w 게이트 ±5pp 유보→확증 재판정), F2/F3/F4/F5 즉결 조건, no-aug 클린 밴드,
suite 확전 조항(승자 확증 시 goal/object 확전), byte-identity guarded 플래그 +
HEAD 대조 스모크, 캐시 실존 assert. **기준선 신선도 규칙(신설·사전 고정)**: 스크리닝
시점에 코드 리비전이 wristpatch 학습 시점과 다르거나 2주 초과 경과면 matchedbase
phase2 재학습(1.5-2 GPU-h) + 재롤; 아니면 기존 ckpt 재사용 + **같은 창 재롤만** 의무
(2026-07-20 JSONL 재사용 불가 — 창이 다름).

### 4.2 실행 순서 (W-A 확증 종료가 트리거)

1. **지금 (확증 s2/s3와 병행, GPU 거의 0)**: SigLIP2 wrist pool2 캐시 빌드(0.3-0.5
   GPU-h) → R-D0 + R-A′ ridge를 **같은 CPU 배치**로 산출, prereg JSON 선기록.
   킬게이트 발동 시 해당 셀은 학습 없이 여기서 끝난다(최대 절약).
2. **W-A 확증 판정 후 분기**:
   - **확증 승리 (SR+언어 회복)** → W-A′ 먼저(기전 귀속·thesis-fit 업사이드), W-D
     다음. 순서 근거: W-A′가 이기면 W-D의 기지(base)가 W-A′로 바뀌므로.
   - **SR 승리·언어 미달** (s1 잠정 61.6 추세 지속) → **W-A′가 결정 셀로 승격**
     (H-L1이 손목 이득을 지키며 언어를 회복할 유일한 저가 가설). W-D는 생존 기지
     확정 후.
   - **전면 실패** → W-A′는 그래도 1회 실행 가치(언어 공동기준을 W-A가 못 넘은
     지점에서 H-L1 검정 + 기전 귀속), W-D는 §2.4 폴백(matchedbase+aux)으로 강등.
3. 스크리닝 통과 팔만 50roll×3seed 확증 (+45 GPU-h/팔) + suite 확전 조항.

### 4.3 예산 (스크리닝까지 상한 ~30 GPU-h)

| 항목 | GPU-h | 비고 |
|---|---|---|
| SigLIP2 wrist pool2 캐시 | 0.3–0.5 | 1.0 GB, 1회 |
| R-D0·R-A′·G0-A(미산출분) ridge | ~0 (CPU 1-2h) | prereg 선기록 |
| W-A′ phase2 | ~2 | 파라미터 W-A 동형 |
| W-D phase2 λ0.1 (+λ0.3) | ~2 (+2) | λ0.3은 G1 비교 후 롤 여부 |
| W-D ablation (b) zeta판 | ~2 (조건부) | (a) G2-D 통과·SR 널일 때만 |
| 스크리닝 2팔×2모드×200ep + base/W-A 인터리브 재롤 | ~12–18 | osmesa retry-supervisor 전제 |
| **합계 상한** | **~30** | 초과 시 미착수 팔 기록만 남기고 종료 |

---

## 5. 논문 슬롯 — 결과별 수납 위치

- **삽입점 지도 신규 행**: "**손실-측(보조 예측, 학습-시 미래 관측 감독)**" — 지도의
  마지막 공란. W-D가 어느 쪽으로 끝나도 이 행이 채워지며, 널이면 §2.4의 3-기전 소진
  선언과 함께 "wrist 추론" 종결 각주가 붙는다 (관측/조건화=양성 · 타깃/코드=무효 ·
  손실-측=W-D 결과).
- **기전 귀속 문장 (W-A′)**: "손목 이득의 재료는 {DINO 기하 | 인코더-불문 공간
  디테일}이다" — 승패 무관 수록. 후자면 "의미 접근성 무료" 문단 + SpLiCE/TextSpan
  손목-경로 프로브 결과가 해석 장(章)에 추가된다.
- **Tradeoff frontier 점**: (ΔSR, Δ(c−w)) 평면에 W-A(+5.5, −10), W-B(+4.5, −12.5),
  W-C(−4.5, −2), W-A′, W-D 5점 — "언어 화폐 대 시각 풍부화" 곡선의 wrist 축 완성.
  H-L1 실측 시 W-A′는 frontier를 **안쪽으로 미는 유일한 점**이 된다 — 논문의 제안
  팔 선정 논리(언어 공동기준 우선)가 이 그림 하나로 정당화된다.
- **PI 직관의 최종 대차대조표**: ①추론 직관 — 옳은 절반("손목=그리퍼 채널의 우월한
  관측", M-A로 기고정)과 검정된 절반(추론 기전 3종 실측)으로 분해 수록; ②통일 공간
  직관 — W-A′ 결과가 곧 판정.

## 6. 정직 섹션 — 최강 반론과 응답

**반론 1**: "Phase-A/wrist 캠페인 통산 복잡화 0승인데 셀을 또 여는가." 응답: 두 셀
모두 학습-전 CPU 킬게이트(R-D0/R-A′)가 있어 반론이 참일 때의 지출이 캐시+프로브
(≤1 GPU-h)로 캡핑되고, W-A′는 복잡화가 아니라 **동복잡도 인코더 교체**(파라미터-정확
매치)라 프라이어의 적용 대상이 아니다. W-D는 남은 마지막 기전 클래스의 소진 실험 —
지는 쪽도 지도 완결이라는 정보 이득이 있다.
**반론 2**: "aux 감독은 이미 조건열에 있는 정보의 재학습일 뿐." 응답: 그 경우 R-D0가
r_full−r_state<+0.02로 학습 전에 자르거나 G2-D가 롤아웃 전에 자른다 — 반론 자체가
게이트로 내장돼 있다.
**반론 3**: "W-A′의 언어 가설은 사후 합리화 여지가 있다." 응답: H-L1/H-L2와 판별
readout(t0/t2/t4 wrong-모드 기준값 95/85/65)을 결과 전에 수치로 고정했고, 판정
우선순위(언어 공동기준 > SR 소폭 우세)도 §3.4에서 선서약했다.

---
*불변식 준수 확인: 타깃-측 재진입 없음(W-C 종결 존중) · 분리 flow→h 없음(M-B) ·
게이트망 없음(M-C) · 팔당 변경 하나(W-D=손실 1항 / W-A′=인코더 정체성 1개) · 전 신규
경로 guarded 기본-off byte-identity · cache_key 신설 분리(`siglip2-so400m-pool2`) ·
loud-fail 상속(_AUX_KEYS 신설, _GRID_KEYS 불변) · no-aug · run-tag provenance ·
폐루프 SR + 언어 공동기준 단독 심판.*
