# 감사 보고서: flow-decoder / actionflow 재구현 충실성 검증 (2026-07-18)

*작성: Claude Code 감사 에이전트. 범위: `/home/user/CLIP_ws`(우리) vs `/home/user/CLIP_ws/SigLIP`(콜리그 클론). 읽기 전용 감사. 모든 주장에 file:line 또는 UNVERIFIED 표기.*

**핵심 요약 (3줄)**
1. 콜리그의 "97.2% actionflow"는 **single-seed +0.8pp**였고, 이후 3-seed(97.2/95.2/96.2, mean **96.2**)로 **base 96.4와 사실상 동률(±2pp 시드노이즈 내)**로 스스로 강등했다 (`SigLIP/figures4paper/MASTER_RESULTS.md:5,11`). 콜리그가 decoder-측 flow로 양성을 얻은 적은 **한 번도 없다**.
2. 우리 actionflow 포트는 **메커니즘상 충실**(x0=과거청크, CFM+L1, Euler 6스텝, 추론 결정론)하지만, **flow가 수송하는 변수 자체가 다르다**: 콜리그는 quantile-정규화([-1,1] 클립) 액션, 우리는 z-score(meanstd) 액션. 기질(dual_avg vs dualconcat)·정규화(dropout0.2+both-aug vs none)·평가(50롤/EGL vs 20롤/osmesa)도 다르다.
3. 우리 쪽 구체 버그 1건: `--flow-fixed-noise`가 문서상 "재계획 간 동일 노이즈 고정"을 주장하나 실제로는 **에피소드당 1회 시드 후 generator가 재계획마다 전진** → 모드-락 가설은 실검증된 적 없음 (`src/eval_libero/rollout_sim.py:60-63` vs `184-186`, `254`).

---

## Task 1 — 콜리그(SigLIP) repo의 flow-decoder / action-flow 실험 전수 인벤토리

기질 정정: "CLIP-768 기질" 전제는 **구세션**에 해당. `result_report.md`는 CLIP-768 캠페인(base SR ≈83–85, `SigLIP/result_report.md:36-38,124`)이고, actionflow/M7/Q2가 돌아간 **post93 세션의 기질은 dual_avg(SigLIP2-large-256 + DINOv2-large 평균, latent 1024)** (`SigLIP/configs/phase2_dualavg_actionflow.yaml:10-16`). 액션 정규화는 두 세션 모두 **quantile**(`result_report.md:16,124`; `SigLIP/configs/phase1_dualavg_m7.yaml:38-39`).

### 1-A. actionflow (정책-측 action-space flow) — 유일한 "승자 초과" 주장

| 항목 | 내용 | 근거 |
|---|---|---|
| 코드 | `FlowPolicy(flow_space="action")`: flow 변수만 112d(=16×7)로 교체, ctx 토큰은 잠재 유지 | `SigLIP/src/models/policy.py:81-93` |
| x0 | `source='past'` → **x0 = flatten 정규화 과거 액션청크(외부 주입 x0_src)**, 학습 시에만 `source_noise 0.1 × x0_std` 가우시안 가산(`self.training` 가드) → **추론은 완전 결정론** | `policy.py:276-294` |
| ODE | Euler, `flow_steps: 6`, t=0→1 | `policy.py:296-301`, config `:40` |
| 손실 | `lat 0.5`(CFM MSE) + `act 1.0`(생성 청크 L1), wm/lang=0 강제 | `SigLIP/src/training/train_phase2.py:946-956`, config `:52-56` |
| x0_std | 학습 시작 시 액션청크 타깃 std로 fill | `train_phase2.py:558-563`(동일 코드 651-654) |
| 하이퍼 | d_model 1536, layers 4, ctx_layers 2, **dropout 0.2**, lang+wrist 토큰, batch 256, lr 1e-4 cosine, 50ep, seed 2, past_noise 0.05, phase1=dualavg winner(both-view aug variants 3) | `configs/phase2_dualavg_actionflow.yaml` 전문 |
| 롤아웃 | `scratchpad/eval_siglip.py`: `x0_src = past.reshape(1,-1)` 주입, **ae.h 디코딩 없이 reshape→invert** | `eval_siglip.py:107-111, 274-279` |
| 프로토콜 | **eval 50롤/task ×10 task(=500), MUJOCO_GL=egl**, max-steps 300, exec-horizon 8, init-state 자연순 | `scratchpad/run_p2_experiment.sh:11,29`, `eval_siglip.py:72-75` |

**보고 수치와 증거 강도**:
- 97.2 (seed2, single): `report/post93_results.md:329` — 보고서 스스로 "⚠️single-seed(+0.8pp는 ±4pp 시드밴드 내→multi-seed 확인 필요)" 명기.
- 3-seed: 97.2/95.2/96.2 → **mean 96.2** (`report/AUTO_RESULTS.md:12,17`, `figures4paper/MASTER_RESULTS.md:11`). MASTER 결론: "tri_concat 96.0과 **사실상 동률(시드노이즈 ±2pp 내)**" (`MASTER_RESULTS.md:5,38`) — **base winner 96.4 대비 우위 소멸**.
- 대조 노이즈: 같은 winner config 재실행 94.6 (`AUTO_RESULTS.md`: `phase2_dualavg_winner_recheck`) → run-to-run −1.8pp.
- 스택/전이: actionflow+augv2 = 97.2(무이득, `post93_results.md:378-381`), concat_zeropad+actionflow = 95.0 (`AUTO_RESULTS.md:20`).
- 구세션 ±5–7pp 경고는 CLIP-768/20롤 시절(`result_report.md:5`); post93은 50롤 + "winner band 93.4–96.4" 기준.

**판정**: 콜리그의 양성 주장 = **정책-측 actionflow 단독, 최초 single-seed +0.8pp → multi-seed로 base와 동률**. "97.2가 재현되는 강한 레버"라는 근거는 그들 자료 안에서도 성립하지 않는다.

### 1-B. M7 (decoder-측 생성형 잔차 flow) — 폐기(무이득)

- 구조: winner h를 **그대로 앵커로 유지**, 별도 소형 `ResidualFlowHead`(hidden 256/3층/Euler 4스텝, x0=N(0,I))가 잔차 r=a−h를 CFM 학습, **zero-init 학습 스칼라 gate**로 주입 `a = h + gate·δ` (`SigLIP/src/models/networks.py:107-156, 169-180, 193-203`; detach 학습 규약 `:241-256`; config `configs/phase1_dualavg_m7.yaml`).
- 롤아웃: `ae.decode(ζ̂, z_c, generator=dec_gen)`, dec_gen은 **에피소드당 시드(cfg seed+ep) 후 전진**(det_boot=재현성, 모드-락 아님) (`src/eval_libero/rollout_sim.py:91-95,120`).
- 결과: gate 0.1/0.5/1.0 → SR **95.4/94.6/94.0** — 전부 winner band(93.4–96.4) 내 = **무이득, 폐기** (`report/post93_results.md:294,299`; `report/BACKLOG_cut_experiments.md:22`).

### 1-C. Q2 (정책-측 latent MLP평균+잔차 flow) — 무이득

- 구조: **phase2 정책 쪽**에서 `mean_head(ctx)=z̄` 회귀 + 잔차만 flow 수송, `ζ = z̄ + res_gate·∫v`(res_gate=1.0 고정 스칼라), **디코더 h 불변** (`SigLIP/src/models/policy.py:154-162, 305-333`; `configs/phase2_dualavg_q2_mlpresflow.yaml`).
- 결과: SR **90.4** (wandb 5hn8rex7) — winner 이하, 무이득 (`post93_results.md:315,320`; `RESULTS_SUMMARY.md:40`).

### 1-D. Q3 (h 완전교체 풀-flow 디코더) — **실행된 적 없음**

- BACKLOG에만 존재: "(선택) 풀-flow 디코더 … 기대이득 낮음(M7=잔차형이 이미 무이득) — 확인용" (`report/BACKLOG_cut_experiments.md:12`). 전 report/AUTO_RESULTS 검색에서 Q3 결과 0건.

---

## Task 2 — 우리(CLIP_ws) 구현 인벤토리

| 변형 | 코드 | 내용 |
|---|---|---|
| **S2 h-flow** (`h_mode=flow`) | `src/models/networks.py:65-118`(`ChunkFlowDecoder`), DeltaAE 분기 `:177-181` | h를 CFM 디코더로 **완전 교체**. x0 = N(0, x0_std=액션청크 std), Euler 5스텝(`configs/phase1_libero_s2_concat_hflow.yaml:33-34`). 학습: recon/cycle L1→CFM 대체(`networks.py:245-247`) |
| **residual_flow** (`h_mode=residual_flow`) | `networks.py:121-149`(`ResidualFlowDecoder`), 손실 `:248-`, x0_std=잔차 std 주입 `src/training/train_phase1.py:351-361` | `a = mean(ChunkDecoder, L1) + res(ChunkFlowDecoder, 잔차 CFM)`. **gate 없음(상시 full 잔차)**, res 헤드 = full-size(hidden 512/4층/5스텝, `configs/phase1_libero_residflow_concat.yaml:30-44`) |
| **actionflow** (`flow_space=action`) | `src/models/policy.py:74-121`(_x0: x0_src=과거청크, 추론 결정론 — 콜리그와 동형), 학습분기 `src/training/train_phase2.py:316-318, 558-563, 654-665`, val `:788-790` | 콜리그 코드의 사실상 verbatim 포트. config `configs/phase2_libero_actionflow_{concat,avg}_noaug.yaml`(lat 0.5/act 1.0/wm 0, f4 off, steps 6, d_model 1536, seed 2) |
| **롤아웃 드라이버** | `src/eval_libero/rollout_sim.py` | af 분기 `:242-249`(x0_src=past.reshape, ae.h bypass, `*a_std+a_mean` 역정규화); h-flow 분기 `:254`(`ae.h(..., generator=ep_gen)`); `--flow-fixed-noise` 플래그 `:60-63`, ep_gen 시딩 `:184-186`(seed=10000·tid+ep, **에피소드당 1회**) |

우리 보고 수치의 출처: h-flow 33%(naive)/37%(fixed-noise) `FOLLOWUP_experiments.md:86`, `PROGRESS.md:29`; actionflow af-concat **76%**(151/200)/af-avg **80%**(159/200) `FOLLOWUP_experiments.md:104`; residual-h-flow **~48–65%** `FOLLOWUP_experiments.md:140`(§10 표). residual-flow의 per-run 롤아웃 로그/판독 조건(에피소드 수, fixed-noise 사용 여부)은 repo 문서에서 미발견 — **UNVERIFIED**.

---

## Task 3 — 측면별 대조표 (theirs = actionflow 승자 레시피 기준)

| 측면 | 콜리그 (SigLIP) | 우리 (CLIP_ws) | 동일? |
|---|---|---|---|
| flow 공간/타깃 | raw 액션청크 112d flatten, CFM v*=target−x0 (`policy.py:336-338`) | 동일 (`policy.py:140-142`) | ✅ |
| x0 분포·스케일 | **x0 = 과거 정규화 액션청크**(결정론), 학습만 +N(0,(0.1·x0_std)²); x0_std=액션타깃 std | 동일 (`policy.py:103-121`, `train_phase2.py:558-560`) | ✅ |
| ODE | Euler 6스텝, t 좌단 격자 | 동일 | ✅ |
| 조건 토큰 | [z_prev,z_cur,g(A_past),lang,wrist] 잠재 1024, dropout 0.2 projector | 동일 구성이나 잠재 2048(concat, zero-pad 토큰), **dropout 0(감사 판정으로 의도적)** (`configs/phase2_libero_fobsfusion_concat_noaug.yaml:7`) | ⚠️ |
| **액션 정규화** | **quantile 1/99pct→[-1,1] 클립, gripper(dim6)만 meanstd** (`SigLIP/src/core/actnorm.py:31-41`) | **meanstd z-score 전차원, quantile 미구현** (`src/training/train_phase1.py:98-102`, `rollout_sim.py:203`) | ❌ |
| 청크/실행/재계획 | n_chunk 16(0.8s@20Hz), exec-horizon 8, 8스텝마다 재계획 | 동일 | ✅ |
| 재계획 간 노이즈 | actionflow: 노이즈 자체 없음(결정론). M7 잔차: per-episode 시드 generator 전진(det_boot) | actionflow 동일(결정론). h-flow: naive=fresh / `--flow-fixed-noise`=per-episode 시드 후 **전진**(콜리그 det_boot와 동형; "고정"은 이름뿐) | ✅/⚠️ |
| 백본 기질 | dual_avg = avg(SigLIP2-L256, DINOv2-large), latent 1024, **both-view aug(variants 3)** | dualconcat = SigLIP2-L256 ⊕ DINOv3-vitl16 no-mix 2048, **no-aug** | ❌ |
| 학습 예산 | phase2 50ep/batch256/lr1e-4 cosine/seed2; phase1 100ep | phase2 동일; phase1 50ep (`configs/phase1_libero_fobsfusion_concat.yaml:46`) | ⚠️ |
| EMA | 없음(grep 0건) | 없음 | ✅ |
| 손실 가중 | lat 0.5 / act 1.0 / wm·lang 0 | 동일 | ✅ |
| 평가 | **50롤/task×10=500**, MUJOCO_GL=**egl**, max300, 태스크별 고정 init-state 자연순, 단일 train-seed(+3-seed 확정) | **20롤/task×10=200**, **osmesa**(EGL 불가 박스, 확률적 렌더 segfault→retry supervisor; git 0842fb1), 단일 seed | ❌ |

---

## Task 4 — 97.2 vs 76/80 격차 원인 후보 (순위)

0. **[likely-noise-band] 전제 보정**: 설명해야 할 격차는 "97.2 vs 76/80"이 아니라 **"그들 base 대비 ±0pp(3-seed 96.2≈96.4) vs 우리 base 대비 −21.5/−11.5pp"**다. 그들에게 actionflow는 이득도 손해도 아니었고, 우리에게는 명확한 손해였다.
1. **[substrate-difference, 유력] flow 변수의 정규화 체계**: 그들의 flow는 [-1,1] 클립된 quantile-정규화 액션(경계 있는 콤팩트 분포, env 액션공간과 동형)을 수송; 우리는 heavy-tail z-score 액션을 수송. 잠재-flow에서는 무관한 차이가 action-flow에서는 **수송 대상 분포 자체의 차이**가 된다. 증거: `SigLIP/src/core/actnorm.py:8-9`(gripper 왜곡 때문에 분리한다는 명시적 설계), `result_report.md:16,124`(quantile을 base로 채택). 우리 repo에는 quantile 경로 자체가 없다.
2. **[substrate-difference, 유력] "무엇을 버리나"가 기질마다 다름**: actionflow는 ζ→h 접지를 추론에서 우회한다. 그들의 h는 약했고(dec R² 0.824–0.826, `RESULTS_SUMMARY.md:31`), 우리 concat 기질의 이득은 ζ 접지에 집중(`FOLLOWUP_experiments.md:105`; af-avg 초기 4태스크 ~91%≈latent로 구현 정상성 방증). 이 경우 우리의 음성은 부정확한 포트가 아니라 **진짜 기질 상호작용**.
3. **[implementation-difference, 중] 정규화·증강 레짐**: 그들 승자 레시피 = projector dropout 0.2 + both-view aug 3variants; 우리 = 둘 다 없음(우리 기질 감사에서 aug가 −13pp confound로 판정된 의도적 선택, `FOLLOWUP_experiments.md:53`). 112d raw 액션을 직접 생성하는 1536-d flow 헤드는 잠재-flow보다 과적합 민감할 수 있어, dropout/aug 부재가 actionflow에만 비대칭적으로 불리했을 가능성.
4. **[protocol-difference, 소] 평가 조건**: 20롤 vs 50롤(±5–7pp 밴드), osmesa 불안정 vs EGL. 방향성 손해 ~16–20pp를 설명하기엔 부족하나 수치 정밀도는 낮춤.
5. **[likely-noise-band, 소] 시드**: 양쪽 다 phase2 train seed 2 단일. 그들 시드 산포 ±2pp 관측 — 우리 76/80의 오차막대도 최소 그 정도로 봐야 함.

---

## Task 5 — 변형별 충실성 판정

**(a) S2 h-flow (33/37%) — "포트"가 아니라 콜리그가 의도적으로 안 한 실험.**
콜리그는 h 완전교체 flow(Q3)를 **한 번도 실행하지 않았고**(`BACKLOG_cut_experiments.md:12`), 그들의 decoder-측 실험 M7은 (i) winner h 앵커 유지, (ii) zero-init 학습 gate, (iii) 소형 잔차 헤드라는 3중 안전장치가 정의적 특징이다. 우리 33/37%는 그들의 어떤 결과와도 모순되지 않으며, 오히려 그들이 Q3를 자른 판단을 실증한 셈. **버그 1건**: `--flow-fixed-noise`는 도움말이 "재계획 간 일관된 단일 모드 유지"를 주장하지만(`src/eval_libero/rollout_sim.py:60-63`) 구현은 에피소드당 1회 시딩 후 generator가 재계획마다 전진(`:184-186` → `:254`) — 각 재계획은 여전히 서로 다른 x0를 뽑는다. 따라서 "37% = 모드-락도 실패"라는 우리 결론은 **모드-락을 실제로 시험하지 않은 채** 내려진 것. 진짜 모드-락(재계획마다 동일 x0 재시딩 또는 x0 캐시)은 미검증 상태다.

**(b) actionflow (76/80%) — 메커니즘 포트는 충실, 기질 이식은 불충실.**
`_x0`/`fm_and_sample`/학습 분기/롤아웃 x0_src 주입까지 콜리그 코드와 라인 단위 동형(우리 `policy.py:103-143` ≈ 그들 `policy.py:276-341`; 우리 `train_phase2.py:654-665` ≈ 그들 `:946-956`). 구체적 코드 버그는 발견하지 못했다. 다만 (1) **quantile 액션 정규화 미이식**(그들 base의 일부이며 flow 변수 자체를 바꿈), (2) dropout 0.2/aug 미이식 — 두 가지가 "그들 레시피의 재현"으로서는 누락이다. 그리고 재현 목표치 자체가 허상이었다: 그들 최종 자료 기준 actionflow는 base 동률(96.2 vs 96.4)이지 +0.8 레버가 아니다.

**(c) residual-h-flow (~48–65%) — "M7/Q2 동형" 라벨은 부정확.**
M7 대비 누락: **zero-init 학습 gate 없음**(우리는 상시 full 잔차 주입, `networks.py:148-149`; 그들은 `res_gate` zero-init 파라미터 + detach 학습, `SigLIP/networks.py:180,249-253`), 잔차 헤드가 소형(256/3/4)이 아닌 full-size(512/4/5). Q2 대비: Q2는 **정책-측(latent)** 잔차이고 우리는 **디코더-측** — 위치가 다르다(콜리그는 그 위치 구분 자체를 결론으로 삼았다: `BACKLOG:11` "잔차를 디코더(M7, 실패)가 아니라 잠재 쪽에"). 단, 콜리그의 M7 gate=1.0 강제에서도 94.0(무붕괴)이었으므로 gate 부재만으로 우리 48–65%를 다 설명하긴 어렵고, (b)와 같은 정규화/기질 요인이 겹쳐 있을 것. 방향성 결론("잔차-flow 무이득")은 콜리그와 정합.

**명시 질문 답변**: **콜리그는 decoder-측 h-flow(사후 ζ flow-decode)로 양성 결과를 얻은 적이 없다.** M7 = 95.4/94.6/94.0 전부 winner band 내 무이득으로 폐기(`post93_results.md:299`), Q3(완전교체) 미실행, Q2(정책-측 잔차) 90.4 무이득. 양성 주장은 **정책-측 actionflow 단독**이며, 그것도 single-seed 97.2 → 3-seed mean 96.2로 base와 동률로 마감됐다.

---

## 원격 체크포인트 / 누락 아티팩트 (kist_3090에서 추후 확보 필요)

클론에 없는 것들 — 아래 수치들은 현재 **md 보고서 문자열로만 검증됨**:
- `SigLIP/checkpoints/` **완전히 비어 있음**: `phase1_libero_dualavg.pt`, `phase2_dualavg_actionflow{,_seed1,_seed3}.pt`, actionflow_augv2 ckpt, `phase1_dualavg_m7.pt`, `phase2_dualavg_q2_mlpresflow.pt`(configs가 참조하는 `~/wonseok/SigLIP/checkpoints/grid/*` 전부).
- `scratchpad/run_*.log` 0건 — AUTO_RESULTS.md가 수확한 원본 per-task 로그 부재.
- wandb 런 `5hn8rex7`(Q2), `qfmypa4y`, `zn4dp7ti` — 온라인(project `wonseok_clip`), 클론엔 없음.
- 클론 내 실물 ckpt는 구세션 2개뿐: `SigLIP/phase1_libero_siglip2_v2.pt`, `phase2_siglip2_v2_bothaug_drop02_l0.pt`.
- 우리 쪽도 residual-flow/actionflow 롤아웃 원본 로그가 `outputs/eval`에서 미발견(체크포인트·로그는 아마 학습 박스) — 48–65% 판독 조건은 UNVERIFIED.

**권고**: (1) 재대결을 원하면 quantile actnorm을 우리 파이프라인에 이식한 뒤 actionflow 1회만 재검(이것이 유일하게 "포트 불충실"로 분류 가능한 요소), (2) 단 목표치가 96.2≈base임을 감안하면 기대값은 "우리 base 동률" 이상이 아님, (3) h-flow 모드-락은 재시딩 버그 수정 없이는 결론 보류로 강등.
