# DESIGN — DINOv3 patch 토큰의 정책-attention 관측 삽입 (L1+L4) v1 (2026-07-18)

*작성: 아키텍처 설계 에이전트. 근거: FOLLOWUP_experiments.md §3/§4/§5/§10, docs/AUDIT_negative_results_2026-07-18.md, docs/LIT_POSITIONING_2026-07-18.md §4(C4)/OTTER(2503.03734)/Prismatic(2402.07865), src/models/{obs_fusion,policy,f4}.py, src/training/train_phase2.py, src/eval_libero/rollout_sim.py, src/core/anchor.py, src/data/libero.py + libero_spatial hdf5 실측.*

**테제 재확인**: 언어정렬 frozen 공간의 변위 Δz는 언어·이미지·행동의 공용 통화다. 기하 정보는 **이 통화를 훼손하지 않는 삽입점(=관측/조건화)** 에서만 이득이 된다. 본 설계의 질문은 하나: *pooled concat(97.5)이 이미 잡은 이득 위에서, fine patch 구조를 관측측에 넣되 언어축(correct−wrong ≥ +70pp)을 지키거나 오히려 개선할 수 있는가?*

---

## 0. 결정 요약 (랭킹)

| 순위 | 후보 | 한 줄 | thesis-fit | info-gain | cost | 언어축 예측 |
|---|---|---|---|---|---|---|
| **1** | **B. LangSelPool** (OTTER-style 텍스트-조건 풀링) | 지시문이 patch 격자에서 "지금 중요한 기하"를 **선택** — 언어가 기하의 상류 | ★★★ (테제 그 자체) | ★★★ (tradeoff 곡선 파괴 가능성) | 중 | **유지~개선 가능** (§4.4 분석) |
| 2 | A. GridToken-v2 (GridObs 완주판, 결함 수정) | 위치보존 저해상 격자 토큰, param-free 풀 — F3/GridObs 부검의 공정 재판 | ★★ | ★★ (F3 "정보 유해" 일반화의 직접 검증) | **저** (스트리밍 불요) | 하락 예상 (+55~65pp 예측, co-기준 위험) |
| 3 | C. PatchDelta (Δpatch 변위 토큰) | 변위 테제를 patch 스케일로 — 조건화에 "fine 변위"를 공급 | ★★★ (미학) | ★ (spatial suite 헤드룸 소진) | 중 | 중립~약한 하락 예측 |

실행 순서 권고: **A(1.5일, 스트리밍 없이 기존 배관으로 즉시) → B2 → B1 → (B가 토큰을 실제 사용하면) C**. A를 먼저 하는 이유는 비용이 아니라 **B의 대조군**이기 때문 — "언어-무관 patch 토큰"(A) vs "언어-선택 patch 토큰"(B)의 correct−wrong 차이가 곧 테제의 직접 증거다.

---

## 1. 설계를 구속하는 확립 사실 → 불변 조건 (IV)

| 확립 사실 (출처) | 도출되는 불변 조건 |
|---|---|
| 관측-레벨 융합만 양성: concat 97.5 / avg 91.5 vs base 85 (FOLLOWUP §4, 20roll/1seed) | **IV1**: patch 토큰은 **정책 조건 토큰열**(관측측)에만 진입. ζ/g/h/디코더에는 절대 넣지 않음 (C1/C2/S2/S1b 음성 재현 금지). |
| 조건화 DINOv3 함량 ↑ → SR↑·언어↓ 단조 (FOLLOWUP §5) | **IV2**: 언어축 co-기준 correct−wrong ≥ +70pp를 **모든 gate에 명시**. 언어-무관 기하 토큰 추가는 이 곡선 위 이동으로 예측하고, 예측과의 편차 자체를 측정 대상으로 삼음. |
| F3 모듈 결함: kv-LN 부재, query init randn(std 1.0), pos-emb 부재, 토큰 dropout/규제 전무, 120ep 레짐 confound (AUDIT §3) | **IV3**: 모든 신규 모듈은 §2.3 결함-수정 체크리스트를 통과해야 함. **full 500ep no-aug 클린 밴드** 고정. |
| C1/C2 게이트 버그: ∂L_act/∂α ≡ 0 (AUDIT §1) | **IV4**: **UNGATED** 상시 삽입 — 태스크 손실이 토큰 경로를 직접 통과 (관측 토큰은 flow ctx에 들어가므로 구조적으로 보장됨). tanh-gate 금지. |
| GridObs OOM: per-sample dense fp32 조립 45–58GB 단일스레드 (AUDIT §1d, §3) | **IV5**: dense 데이터는 §6의 조립 계획(사전 풀링 캐시 또는 fp16 사전할당/memmap)으로만. `np.stack`→`np.concatenate`→`torch.tensor` 3중 복사 경로 금지. |
| offline≠SR, eff-rank 5.5 (FOLLOWUP §9) | **IV6**: 오프라인 probe는 **선별용**(no-go 판정)이지 go 판정이 아님. 폐루프 SR만이 판사. |
| 헤드라인 수치 10/10 UNTRACED (AUDIT §5) | **IV7**: 모든 롤아웃은 per-episode 성공 플래그를 run-name 포함 JSONL로 저장 (덮어쓰기 금지). 본 캠페인부터 강제. |

## 2. 공통 설계

### 2.1 phase1은 무엇을 보는가 — **patch 토큰은 phase1을 완전히 우회한다**

- **DeltaAE(g/h)는 pooled z 위에 그대로**: B2/A/C는 기존 `phase1_libero_fobsfusion_concat.pt`(2048d), B1은 기존 `phase1_libero_siglip2_large256.pt`(1024d)를 **재학습 없이** 사용. ζ 통화의 정의(=pooled frozen 공간의 변위)는 단 1비트도 바뀌지 않음.
- 근거: 통화(ζ)에 patch를 섞는 순간 C1/C2(코드측)·S2(디코더측) 음성 영역으로 들어간다. patch 기하가 살 곳은 IV1에 따라 **정책 flow의 조건 토큰열**뿐이다. 이것이 "L1(관측 grid) + L4(정책 attention)"의 정확한 의미: *선택은 정책측 모듈이 하되, 선택된 결과는 관측 토큰으로만 소비된다.*
- 부수 효과: phase1 고정 → 후보 간·베이스 간 paired 비교에서 phase1 분산이 0. 오프라인 act R²는 phase2 val로만 움직임.

### 2.2 phase2 진입점 (코드 seam — 기존 배관 재사용)

- **학습**: `train_phase2.py`의 grid_obs seam을 일반화. `forward()`의 `toks = base + [module(dobs...)[:, k] for k in range(K)]` (train_phase2.py:645-648) — 신규 모듈은 GridObs와 동일한 `(B, K, latent_dim)` 계약만 지키면 됨. `n_tokens = 5 + K` (base = [z_prev, z_cur, g(A_past), lang, wrist]).
- **롤아웃**: `rollout_sim.py:169-179 grid_toks()` seam 동일 재사용 — 스텝마다 DINOv3 dense 1프레임 인코딩(fp32, ~20-30ms) 후 모듈 통과. B는 여기에 lang 임베딩 인자 1개 추가.
- **데이터**: `libero.py build_policy_samples`의 `obs_anchors` dense 배관 재사용하되 §6 조립 규칙 적용.
- config는 `module.patch_obs:` 신설 (`grid_obs`와 동급, 상호배타 가드). 미설정 시 전 경로 no-op = 기존 비트 동형 (레포 관례 유지).

### 2.3 F3 결함-수정 체크리스트 (모든 후보 공통 규격)

| F3 결함 (AUDIT §3b) | 본 설계 규격 | 비고 |
|---|---|---|
| kv-LN 부재 (raw patch를 MHA에 직접) | `kv = LN(W_kv·patch + pos_emb)` — f4.py:103과 동형 (f4가 이미 사내 검증한 배선) | attention을 쓰는 B·C-attn에 적용 |
| query init `randn` std 1.0 | 모든 학습 임베딩/오프셋 `normal_(std=0.02)` (ViT 관례); Linear weight `normal_(0, 0.02)`, bias 0 | 전 후보 |
| pos-emb 부재 | kv측 학습 2D pos-emb `(P, d_attn)` std 0.02 | **정직한 구분**: attention 풀링에만 필요. A(param-free avg-pool + flat-concat ctx)는 토큰 슬롯 자체가 위치를 인코딩(FlowPolicy ctx는 flatten concat)하므로 pos-emb 불요 — F3의 결함은 "attention이 위치를 모른 채 선택"한 것 |
| 토큰 dropout/규제 전무 | **2단 dropout(학습시만)**: ① per-token zero-drop p=0.10 ② **group-drop p=0.10** (K개 전부 0) | ②가 핵심: 정책이 "기하 토큰 없이도 동작하는 언어 경로"를 유지하도록 강제 → 언어통화 보호 장치 (modality-dropout 관례) |
| 120ep 레짐 confound | **full 500ep**, no-aug 클린 밴드, concat_noaug와 동일 train 블록 (batch 256, lr 1e-4, cosine, 50ep) | 전 후보 |
| (추가) +9.4M 무규제 파라미터 confound | K는 4~16으로 소형 유지, 파라미터 수 명시 보고. K=16·latent 2048이면 ctx 1층이 +50M 됨 → **기본 K=8** (A만 GridObs 선례 재판을 위해 K=16 팔 병행) | Adam·wd 등 최적화기는 베이스와 동일(비교 가능성 우선) |

---

## 3. 후보 A — **GridToken-v2**: 위치보존 저해상 격자 토큰 (GridObs done right)

### 3.1 모듈 스펙

| 항목 | 값 |
|---|---|
| 입력 | DINOv3-L/16 @256 → 16×16=256 patch × 1024d (registers 드롭, fp32 — 기존 Dinov3Anchor 규율) |
| 풀링 | `adaptive_avg_pool2d` 16×16 → **4×4 (K=16)** 및 **2×2 (K=4)** 두 팔 — param-free, 위치 보존 |
| 사영 | `LN(1024)` → `Linear(1024 → latent_dim=2048)`, weight std 0.02 / bias 0. **LN을 사영 앞에 추가** (기존 GridObs에 없던 patch-스케일 정규화 = kv-LN의 param-free 대응물) |
| pos-emb | 불요 (flat-concat ctx의 슬롯 = 위치; §2.3) |
| dropout | per-token 0.10 + group 0.10 |
| 게이트 | 없음 (IV4) — step0부터 상시 |
| 파라미터 | 2.1M (proj) — F3의 9.4M 대비 1/4 |
| phase1 | concat 2048 pooled 그대로 (우회) |
| phase2 토큰열 | [z_prev, z_cur, g(A_past), lang, wrist, grid×K] |

**엔지니어링 핵심 (OOM 원천 제거)**: avg-pool은 선형이므로 **인코딩 시점에 풀어서 캐시**해도 수학적으로 동일 — `Dinov3Anchor(force_size=256, pool_to=4)`가 **이미 코드에 존재** (anchor.py:259-264, cache_key `dinov3-vitl16-256-pool4`로 자동 분리). dense 캐시가 프레임당 16×1024로 줄어 per-sample 조립이 27k×16×1024×4B = **1.8GB** — 기존 코드 경로 그대로, 스트리밍/신규 데이터 코드 **0줄**. (K=4 팔은 pool_to=2.)

### 3.2 예측 — SR↔언어 곡선 위의 이동

- **SR**: concat 베이스(97.5) 위에서 +0~1.5pp (헤드룸 2.5pp뿐). avg 베이스(91.5) 위에서 +2~5pp 예측 — 이득은 위치 정보가 필요한 공간태스크(§4에서 t4류가 이미 개선된 것의 잔여)에서만.
- **언어**: 이 토큰들은 **언어-무관** DINOv3 함량의 순증 → §5 단조 곡선의 오른쪽 이동. concat(+69) 아래로 예측: **+55~65pp** → co-기준(+70) **탈락 가능성이 기본 시나리오**. group-drop 0.10이 완충하겠지만 곡선을 꺾을 기전은 없음.
- **왜 그래도 하나**: (i) F3 "richer obs가 폐루프를 해친다"의 일반화가 모듈결함+120ep confound였는지의 **직접 판결** (AUDIT §3 재실행 처방 그 자체), (ii) B의 대조군 — A(언어무관 선택 없음) vs B(언어 선택)의 언어델타 차이가 테제의 증거, (iii) 비용 최소.

### 3.3 언어통화 훼손 분석

훼손 기전 = 희석(dilution): 조건열 21토큰 중 16개가 비언어 기하 → lang 토큰의 유효 기여 축소, wrong-instruction에서도 기하 토큰만으로 원태스크를 풀 shortcut 여지(단 spatial suite는 지시문이 타깃 위치를 정하므로 기하만으로는 타깃 선택 불가 — biased 2.5% 선례가 방어). **훼손은 예상되지만 통화 자체(ζ 공간)는 무손상** — 희석은 조건화 수준의 현상이고 phase1이 불변이므로, 실패해도 베이스로 롤백 시 잔존 오염 0.

---

## 4. 후보 B — **LangSelPool**: 텍스트-조건 풀링 (OTTER-style) ★ 주력

### 4.1 설계 논리

OTTER(2503.03734)의 text-aware visual pooling을 우리 삽입점 지도에 이식: **지시문 임베딩이 patch 격자를 질의하여 "이 태스크에 유관한 기하" K개 토큰을 추출**, 관측측에 ungated 삽입. C1/C2(f4)도 텍스트-쿼리 cross-attn이었지만 **삽입점(타깃/코드측, gated, ∂L_act/∂α≡0)** 이 달랐다 — B는 같은 연산을 **양성으로 판명된 유일한 삽입점(관측/조건화)** 에, 태스크 손실이 직통하는 ungated 배선으로 옮긴 것. 즉 "L4 정책 attention"의 실체 = *policy conditioning에 들어가기 전, 언어가 기하를 선택하는 cross-attention*.

### 4.2 모듈 스펙 (`LangSelPool`)

| 항목 | 값 |
|---|---|
| kv | DINOv3 patch (B, P, 1024). P=256(full) 또는 64(8×8 pool — §6 절충). `kv = ln_kv(W_kv·patch + pos_emb)`; `pos_emb (P, 768)` std 0.02 |
| query | `q_k = W_q·lang_emb + query_offset_k`, K=8; `W_q: 1024→768` std 0.02, `query_offset (8, 768)` std 0.02 — f4.py:104 배선 동형 (사내 검증된 형태) |
| attention | `nn.MultiheadAttention(768, 8, batch_first=True)` 1층 (깊이 최소화 — Karpathy 규율; F3 실패가 용량 문제였다는 증거 없음) |
| 출력 | `LN(768)` → `Linear(768 → latent_dim)` std 0.02 |
| dropout | per-token 0.10 + group 0.10 (§2.3) |
| 게이트 | 없음. 태스크 손실(flow CFM+act)이 attention 가중치까지 직결 — C1/C2 굶음의 구조적 해소 |
| 파라미터 | ~5.9M (kv_proj 0.8 + pos 0.2 + W_q 0.8 + MHA 2.4 + out 1.6M; latent 2048 기준) |
| 로깅(필수) | attention entropy(per head, per epoch), per-task attention map 8장, ‖token‖ — AUDIT §1e "α 로깅 부재" 교훈 |

**두 sub-arm (phase1 선택)**:
- **B2 (screening 우선)**: concat 2048 phase1 위 — 최고 SR 베이스에서 언어델타 회복 여부 검정.
- **B1 (headline 후보)**: **SigLIP2-large256 단독 phase1(1024)** 위 — 통화는 순수 언어정렬 공간(S1b noalign이 보인 언어 최고 +78 지대), 기하는 **오직 언어-선택 patch 토큰으로만** 조건화에 공급. S1b의 반증("조건화에서 DINOv3를 빼면 SR 상실")과 정합적이면서 그 대우를 검정: *조건화에 DINOv3를 "언어가 거른 형태로" 넣으면 SR과 언어를 동시에 얻는가?* 성공 시 tradeoff 곡선 파괴 = 논문 C5 다이얼 위의 특이점.

### 4.3 예측 — 왜 이것이 tradeoff 곡선을 꺾을 수 있는가 (신중 분석)

§5 단조곡선의 기전 가설은 "조건화의 **언어-무관** 시각용량이 언어민감도를 희석"이다. 곡선을 그린 세 점(concat/avg/S1b)의 DINOv3 성분은 모두 **지시문과 통계적으로 독립인 pooled CLS**였다. B의 기하 토큰은 `tokens = f(patches, lang)` — **언어의 인과적 하류**다. 따라서:

1. wrong instruction → 쿼리가 바뀜 → **기하 토큰 자체가 바뀜** → 정책 행동이 언어에 추가로 민감해짐 → correct−wrong은 희석이 아니라 **증폭** 방향의 기전이 생김. (희석 곡선의 전제 "비언어 용량"이 성립하지 않음.)
2. 단 자동은 아니다 — 반대 기전 두 개를 명시:
   - **쿼리 누출 shortcut**: MHA 출력이 kv가 아니라 쿼리(=텍스트) 내용을 주로 실어 나르면(attention이 평평하고 out이 q의 함수로 수렴) 토큰은 lang 토큰의 복제가 됨 → 언어축 무해하나 SR 이득 0. → 탐지: patch-셔플 probe(§7.2-P3).
   - **entropy 붕괴 실패**: 텍스트(SigLIP2 공간)와 patch(DINOv3 공간)는 선험 정렬이 없어 W_q/W_kv가 21.7k 샘플로 정렬을 학습해야 함. 학습 실패 시 attention ≈ uniform → 토큰 ≈ 전역 avg-pool = avg-fusion의 중복 → SR·언어 모두 베이스 근처. → 탐지: entropy 로깅(uniform=ln256≈5.55 대비), 실패 시 fallback 팔 = kv를 SigLIP2-large256 patch 토큰(save_tokens 배관 기존재)으로 교체해 "정렬 병목이었는지" 분리.
3. **정량 예측 (사전등록)**: B2 — SR 96~99, correct−wrong **≥ +69 (concat 동급 이상)**, 이 둘이 동시 성립하면 곡선 파괴의 1차 증거. B1 — SR 90~96 (base 85-88 대비 유의 상승이면 성공), correct−wrong **+74~80** (avg 이상). B1이 SR≥95 AND 언어≥+74에 도달하면 avg를 대체하는 신규 제안 아키텍처.

### 4.4 언어통화 훼손 분석

- ζ 공간: 무손상 (phase1 우회, B1은 오히려 통화를 더 순수하게 만듦).
- 조건화: 훼손 기전이 구조적으로 약함 — 기하 토큰이 언어의 함수이므로 "언어를 무시하고 기하만 쓰는" 정책은 자기모순(기하를 쓰려면 언어가 쿼리에 있어야 함). 잔여 위험은 **swap-불변 기하 누출**: 10태스크 지시문이 유사해("black bowl … plate") 쿼리 간 차이가 작으면 wrong-instruction에서도 토큰이 거의 동일 → 희석과 등가. → 탐지: 오프라인에서 지시문 스왑 시 토큰 코사인 거리 분포 측정(§7.2-P4). 완화: 쿼리에 lang 임베딩 L2-norm 후 투입(스케일 지배 방지).
- 실패해도 얻는 것: "언어-선택으로도 안 꺾이면 tradeoff는 함량이 아니라 **총 시각용량**의 함수" — C5 다이얼 서사의 기전 규명.

---

## 5. 후보 C — **PatchDelta**: patch 변위 토큰 (변위 테제의 fine 스케일 확장)

### 5.1 모듈 스펙

| 항목 | 값 |
|---|---|
| 입력 | ΔD_t = D_t − D_{t−span} (patch별 동일인덱스 차분, **과거 변위** — 롤아웃에서 인과적으로 가용. f4의 미래 ΔF와 결정적으로 다름: 생성 flow 불요) |
| 정규화 | ΔD는 저스케일·희소(정적 배경≈0) → `LN` 필수 + per-patch ‖ΔD‖로 스케일 로깅 |
| 풀링 | 기본: 4×4 avg-pool → K=16, A와 동일 사영 규격. ablation: LangSelPool의 kv를 ΔD로 교체(언어가 "유관한 움직임"을 선택 = B×C 합성) |
| 롤아웃 | patch 링버퍼 1프레임 유지 (스텝마다 이미 dense 인코딩 — 추가 인코딩 0회). t<span은 ΔD=0 (학습 분포와 일치: past_seg 관례 미러) |
| phase1 | 우회 (동일) |

### 5.2 예측과 정직한 평가

- **기전**: 정책이 ζ(pooled 변위)를 예측하는데 조건에 fine 변위 이력을 줌 = "변위로 변위를 예측" — 테제와 가장 시적으로 정합. 움직인 patch만 활성 → 자기-모션 + 조작 중 물체의 위치 변화가 자연 하이라이트.
- **그러나**: spatial suite의 실패 모드는 **타깃 선택**(운동 개시 전 — 이때 ΔD≈0으로 정보 없음)과 파지 정밀도인데, 전자는 절대 기하(A/B)가 담당, 후자는 wrist 추론(Phase-B)이 이미 음성. ΔD가 이득일 국면(운동 중 servoing)은 concat 97.5가 남긴 2.5pp 안에 거의 없음. **SR 중립 예측**.
- **언어**: ΔD는 물체-종류 의미가 약한 순수 운동 기하 → 희석은 A보다 약할 것(+65~72pp 예측). 훼손 위험 낮음, 이득 근거도 약함.
- **역할**: 단독 승부수가 아니라 (i) A/B가 "patch 토큰이 사용됨(§7.2-P2 양성)"을 보인 뒤의 정보 채널 ablation, (ii) 논문의 "변위 통화는 fine 스케일에서도 성립하나?"라는 질문에 대한 데이터 포인트. **A/B 결과 전 착수 금지.**

---

## 6. 메모리/컴퓨트 엔지니어링 (실측 기반 — 이번엔 죽지 않게)

### 6.1 데이터 실측과 캐시 계산

libero_spatial 실측: **500 episodes, 총 62,250 frames, mean T=124.5** (75~197). span=16(0.8s@20Hz), stride=2 → per-sample starts Σ ≈ (62,250 − 500×16)/2 ≈ **27.1k samples** (train 80% ≈ 21.7k).

| 저장물 | 계산 | 크기 |
|---|---|---|
| **per-frame full-grid 캐시 fp16** (권장 기본) | 62,250 × 256 patch × 1024d × 2B | **32.6 GB** (fp32면 65.3 GB — 현행 npz fp32의 원죄) |
| per-sample 2-frame 조립 fp16 (C 상한) | 27.1k × 2 × 256 × 1024 × 2B | 28.4 GB |
| per-sample 1-frame 조립 fp32 (**구 GridObs 경로**) | 27.1k × 256 × 1024 × 4B | 28.4 GB **×2~3 복사 피크** (에피소드 리스트 + `np.concatenate` + `torch.tensor`) = **57~85 GB → AUDIT의 45–58GB OOM 관측과 정합. 이 경로 금지 (IV5)** |
| A: pool_to=4 캐시 (16 tok) | 62,250 × 16 × 1024 × 2B(fp16) | **2.0 GB** (조립 fp32도 1.8 GB) |
| B-8×8: pool_to=8 캐시 (64 tok) | 62,250 × 64 × 1024 × 2B | 8.2 GB (조립 fp16 3.6 GB / fp32 7.1 GB) |
| B-full: 16×16 조립 fp16 사전할당 | 27.1k × 256 × 1024 × 2B | **14.2 GB** 상주 |

### 6.2 조립 계획 (OOM 3중 복사 제거)

1. **A**: 신규 코드 0줄 — `Dinov3Anchor(force_size=256, pool_to=4)` (기존 anchor.py:259, cache_key 자동 분리 `…-pool4`). avg-pool은 선형이라 풀→사영 = 풀링 후 캐시와 수학 동일. 기존 배관 그대로 1.8GB.
2. **B 1차 (8×8 절충)**: `pool_to=8` — 위치 granularity 32px, 기존 배관으로 7.1GB fp32 조립 (여유). 16×16의 절반 해상도이나 LIBERO 물체 크기(≥30px) 기준 선택엔 충분하다는 가설로 시작.
3. **B-full (16×16 확증 팔)**: `build_policy_samples`에 `dense_dtype=float16` + **사전할당 채움** 옵션: `torch.empty((N_total, P, 1024), dtype=torch.float16)`을 먼저 잡고 에피소드 루프에서 직접 슬라이스 기록 (리스트-then-concatenate 금지) → 피크 14.2GB + 1 에피소드 버퍼(≤50MB). DataLoader는 fp16 텐서를 인덱싱해 배치만 GPU에서 fp32 캐스트 (fp16→fp32 캐스트는 사영 앞 LN이 있어 수치 무해; **DINOv3 인코딩은 fp32 유지**(NaN 회피, W5) — fp16은 저장 시 캐스트만). 잔여 위험 시 fallback: `np.memmap` [62,250, 256, 1024] fp16 + (ep,t) 인덱스 gather Dataset (num_workers=4) — 상주 <2GB.
4. dense 캐시 생성(1회): DINOv3-L fp32 @256, 62.25k frames, 배치 16 → 실측 계열 ~50-80 fps → **0.3–0.5 GPU-h** + npz(fp16) 기록. 기존 fp32 dense 캐시가 있는 카메라는 재인코딩 없이 fp16 변환 스크립트로.

### 6.3 VRAM (RTX 6000 Ada 48GB)

| 소비처 | 추정 |
|---|---|
| 학습: 정책(d_model 1536, ~40M) + patch 모듈(≤6M) + Adam 상태 ×3 | ~1.5 GB |
| 학습: dense 배치 (256 × 256 × 1024 fp32) + MHA 활성 | 0.3 + ~1 GB |
| 학습 합계 (batch 256) | **< 8 GB** — batch 512 여유. K=16·latent2048 시 ctx 1층 +50M(+0.6GB Adam) 주의 |
| 롤아웃: SigLIP2-L fp16 + DINOv3-L **fp32** + 정책 | ~1.6 + ~1.3 + 0.3 → **< 6 GB** |
| 롤아웃 개선: dualconcat의 DINOv3와 grid/patch 앵커 **인스턴스 공유** (현행 rollout_sim은 별도 로드 — 1.3GB 절약, 필수 아님) | — |

병목은 VRAM이 아니라 **osmesa 렌더 안정성**(FOLLOWUP §11) — retry-supervisor 필수 전제, per-episode JSONL(IV7)이 재시도 시 데이터 무손실을 보장.

---

## 7. 평가·사전등록

### 7.1 공통 프로토콜

- 레짐: full 500ep, no-aug 클린 밴드, seed 2, concat_noaug train 블록 동일. 스크리닝 = 10task × 20roll/1seed (correct + wrong), 판정 = paired bootstrap 10k per-task (기존 paired_ci.py), 베이스 arm과 **동일 에피소드 초기상태**.
- **이중 gate (모든 후보 동일)**: ① SR: vs **concat 97.5** paired Δ의 95% CI가 0 미포함 하락이 아닐 것 **AND** vs **avg 91.5** — 후보의 목적에 따라 주장 축 명시(아래) ② 언어: **correct−wrong ≥ +70pp**. 스크리닝 통과 시에만 확증(50roll × 3seed, cowork 프로토콜).
- 산출물(IV7): `outputs/eval/{run_name}_ep.jsonl` per-episode 플래그 + ckpt 해시. wandb `clipvp`.

### 7.2 오프라인 probe 사다리 (싼 것부터; 각 <0.5 GPU-h)

- **P1 건강**: phase2 val act-R² ≥ 베이스−0.01 (concat 0.749 / large256 0.655). 미달 → 학습 문제, 롤아웃 금지.
- **P2 사용량 (C1-굶음 탐지)**: val에서 patch 토큰 zero-ablation → Δ(act loss). |Δ|≈0이면 토큰 미사용 = **즉시 no-go** (롤아웃 불태우지 않음 — wrist 교훈, AUDIT §4b).
- **P3 (B전용) 선택 실재성**: attention entropy vs ln(P); patch-셔플 시 토큰 변화량 (불변 ≈ 쿼리 누출 shortcut); per-task attention map이 지시된 타깃 영역에 질량 집중하는지 육안 8장.
- **P4 (B전용) 언어 민감도**: 지시문 스왑(1c 프로토콜 형제쌍) 시 기하 토큰 코사인 거리 분포 — 중앙값이 same-instruction 노이즈 대비 유의하게 클 것. 작으면 §4.4 swap-불변 누출 → 언어증폭 가설 기각 예고.
- **P5 (A/C) 디코더빌리티**: 16토큰 → 타깃 bowl 픽셀 좌표(시뮬 GT) 선형 probe R² — "위치 정보가 실제로 실려 있는가"의 5분 검사.

### 7.3 후보별 gate·반증 조건·예산

| | **A GridToken-v2** | **B LangSelPool** | **C PatchDelta** |
|---|---|---|---|
| 주장 축 | avg 대비 SR (concat 대비는 비열등) | B2: concat SR 유지 + 언어 ≥+69 회복 / B1: base 85-88 대비 SR 유의↑ + 언어 ≥+74 | 비열등 + 언어 ≥+70 |
| 성공 기준 | SR(avg기반) paired Δ>0 SIG AND c−w ≥ +70 | B2: [SR≥concat−CI0 AND c−w≥+70] / B1: [SR CI>0 vs 85-88 AND c−w≥+74] | c−w ≥ +70 AND SR 비열등, P2 사용량 양성 |
| **반증(명시)** | c−w < +65 → "언어-무관 patch 토큰은 co-기준과 양립 불가" 확정, §5 곡선에 4번째 점 등록 | P3 entropy≈uniform 지속 or P4 스왑-불변 → 기전 실패로 기록 후 SigLIP2-patch kv fallback 1회만; 폐루프 두 sub-arm 모두 gate 미달 → "tradeoff는 함량 아닌 총용량" 채택, 곡선 파괴 주장 폐기 | P2 |Δ|≈0 or 폐루프 미달 → "변위 통화는 pooled 스케일에서 완결" 채택 |
| GPU-h (스크리닝) | 캐시 0.3 + 학습 1.5×2팔 + probe 0.5 + 롤아웃 6×2팔 ≈ **16** | 캐시 0.5 + 학습 1.5×2 + probe 1 + 롤아웃 6×2 ≈ **17** (+fallback 시 +8) | ≈ **9** (1팔) |
| 확증 (통과 시) | 3seed×50roll ≈ +45 | ≈ +45/sub-arm | ≈ +45 |

총 스크리닝 예산 ≈ **42 GPU-h** (+확증 45/통과-arm). 롤아웃은 GPU보다 CPU(osmesa) 지배 — wall-time은 retry 포함 ×1.5 계상.

---

## 8. 리스크 대장 (종합)

1. **[전 후보] 헤드룸 소진**: concat 97.5 기준 SR 이득은 통계적으로 증명 불가능 영역(2.5pp) → SR 주장은 avg/단독 베이스에서, concat에서는 비열등+언어회복으로 주장 축을 사전 고정 (§7.3). 이를 안 하면 어떤 결과도 해석 불능.
2. **[B] 교차공간 정렬 실패** (SigLIP2 텍스트 ↔ DINOv3 patch, 21.7k 샘플): entropy 로깅 + SigLIP2-patch kv fallback으로 "정렬 병목 vs 선택 무익" 분리. 1회로 제한 (garden-of-forking-paths 방지).
3. **[B] F4-echo 오독 위험**: "텍스트-쿼리 cross-attn은 이미 C1/C2에서 음성"이라는 반론에 대한 선제 방어를 문서화 — C1/C2 음성은 (i) 타깃/코드측 삽입점 (ii) ∂L_act/∂α≡0 게이트 버그 (iii) 미래 ΔF 생성 필요, 세 가지가 원인이며 B는 셋 다 제거(관측측·ungated·현재 프레임). 삽입점 테제의 예측은 오히려 "B는 다르게 동작해야 한다"이다.
4. **[A] co-기준 탈락이 기본 시나리오**: 실패해도 §5 곡선의 데이터 포인트 + F3 부검 종결로 회수 — 단 보고서에 "예측된 음성"으로 사전 기록 (사후 스핀 방지).
5. **[운영] osmesa 확률 사망**: retry-supervisor + per-episode JSONL 없이는 어떤 arm도 착수 금지 (Phase-B 판독 불능의 재발 방지).
6. **[통계] 20roll/1seed 스크리닝의 검정력**: c−w gate는 20roll에서 ±10pp 노이즈 — gate 근처(±5pp) 결과는 pass/fail 판정 유보하고 확증으로 승격.

## 9. 실행 순서 (요약)

```
week 1: A (pool_to=4 캐시 → K=16/K=4 학습 → P1/P2/P5 → 롤아웃)   ← B의 대조군 확보
week 1-2: B2 8×8 (캐시 → 학습 → P1-P4 → 롤아웃) → 통과 시 B-full 16×16 확증 팔
week 2-3: B1 (siglip2-large256 phase1 위) — headline 시도
week 3+: (A/B의 P2가 양성일 때만) C
상시: per-episode JSONL + attention/entropy 로깅 + wandb 아티팩트 회수 (AUDIT §5)
```
