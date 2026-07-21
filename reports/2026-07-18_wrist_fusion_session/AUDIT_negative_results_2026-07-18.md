# 감사 보고서 — 종결된 NEGATIVE 실험들의 "best shot" 여부 (2026-07-18)

*작성: Claude Code 감사 에이전트. 근거: `/home/user/CLIP_ws` 코드와 로컬 1차 아티팩트만. PROGRESS.md/FOLLOWUP_experiments.md는 "주장 목록"으로만 사용.*

**전제적 발견 (§5): 10개 헤드라인 수치 전부가 이 워크스페이스의 1차 아티팩트로 재계산 불가능(UNTRACED).** 학습·롤아웃은 원격 박스(`kist_a6000_ss`)에서 수행됐고 체크포인트·롤아웃 로그가 로컬에 미동기화. 따라서 각 항목의 "통계적 뒷받침" 판단은 *문서 기재 수치가 정확하다는 가정 하의 조건부 판단*.

---

## 1) C1/C2 게이트 fine 채널 (`src/models/f4.py`) — 판정: **NAIVE-BOLT-ON (구조적으로 게이트가 열릴 수 없는 설계)**

**(a) 태스크 손실 → α 그래디언트가 정확히 0.**
- 학습 시 fine 잔차는 `ahat = ahat + f4.fine_action(zeta, zeta_f, zc)` (`train_phase2.py:676`)인데 `zeta_f`는 **flow의 ODE 샘플**(`f4.py:130-143`)로 x0 노이즈와 flow 파라미터만의 함수 — **α에 의존하지 않는다**(α는 target `zeta_f_tgt = f4.encode(...)`, `train_phase2.py:673`에만 들어감). 따라서 **∂L_act/∂α ≡ 0** — 액션 손실이 게이트를 열 경로가 없다. Flamingo tanh-gate가 열리는 이유는 태스크 손실이 게이트 경로를 직접 통과하기 때문인데 그 전제가 불성립. teacher-forcing(`fine_action(zeta, zeta_f_tgt, zc)`로 학습, 추론만 flow 샘플)이 있었다면 ∂L_act/∂α≠0이었을 것 — 부재.
- α에 그래디언트를 주는 손실은 둘뿐: ① `l_f_fm`(`f4.py:137`) — target이 **stop-grad 없이** CFM 회귀 타깃(메인 정책 타깃은 `train_phase2.py:665-666` no_grad인 것과 대조). 양방향 회귀 = "예측하기 쉬운(저분산) 타깃"으로의 붕괴 압력 = **α를 0으로 누르는 압력**. ② `l_consist`(`f4.py:151-152`, C1만 0.1) — 아래 (b).

**(b) L_consistency는 ζ_f를 Δp로 붕괴시키도록 '설계'되어 있다.** `f4.consistency(zeta_f_tgt, zn − zc)`(`train_phase2.py:677`)는 ζ_f 타깃을 선형 헤드(`f4.py:91`)로 **pooled 변위 Δp에 직접 회귀**. 문서의 "발견" CKA(readout,Δp)=0.871은 이 손실의 **의도된 최적점** — fine 채널의 존재 이유와 정면 충돌하는 유일한 개방-압력 손실. consist=0으로 빼자(C2/noconsist) **α에 남은 그래디언트는 ①의 붕괴 압력뿐** → α 0.027→0.0056→0.0015로 더 닫힌 것은 "폐루프가 fine 정보를 보상 안 함"의 증거가 아니라 **이 배선의 필연**. noconsist "구조 절제" 팔도 공정한 테스트가 아니었다.

**(c) 전문가라면 넣었을 요소 — 전부 부재.** 게이트 전용 lr 그룹/부스트(단일 Adam 단일 lr, `train_phase2.py:582-588`) · 게이트 warmup / α>0 warm-start 팔 · 사용 유도 aux loss(잔차 액션(cf − h(ζ_g).detach())의 ζ_f_tgt 직접 회귀; consist 타깃을 잔차 변위로) · CFM target stop-grad/EMA — 모두 없음. 부수: 인코더(readout) 그래디언트가 tanh(α)≈0.003~0.03 배율로 스케일(`f4.py:110`)되어 인코더가 굶음 → "readout 한계효용 +1.8pp"는 거의 학습 안 된 readout의 측정치로 ΔF 정보량의 하한조차 못 됨.

**(d) 비잔차 주입 공정 대조(GridObs, 무게이트 관측측)는 OOM 사망으로 끝내 미실행.** (`obs_fusion.py:96-99` 주석이 스스로 인정.)

**(e) 통계.** C1 게이트 n=10/task 1시드: +1.5pp CI[−3.5,+6.5]; C2 paired Δ −3.5pp CI[−20.4,+13.4] — 검정력 사실상 0, 게다가 C2 창은 대조팔 오염 창. **"이 게이트 설계로는 무이득"만 지지, "폐루프가 fine 정보를 보상하지 않는다" 일반화는 코드 구조상 도출 불가.**

**재실행 처방**: ① teacher-forced fine head(∂L_act/∂α≠0) + 추론만 flow, ② CFM target stop-grad, ③ consist를 잔차-변위 타깃으로 교체/제거, ④ α warm-start(0.05~0.1)+게이트 lr ×10, ⑤ α/β/‖ζ_f 기여‖ epoch 로깅. 저렴한 대안: GridObs full-data 완주(스트리밍 dense 조립로 OOM 해소).

---

## 2) S1b 역할분리 — 판정: **noalign 팔 BEST-SHOT / hybrid 팔(67.5%) INCONCLUSIVE**

**(a) fused[:, :1024] == SigLIP2-alone bit-exact — 검증됨.** `DualConcatAnchor.encode_images`(`anchor.py:500-509`)는 L2-norm된 SigLIP2와 normalize된 DINO를 **concat 후 재정규화하지 않음**(`anchor.py:507`) → 첫 1024차원은 bit-exact. train/rollout 모두 fused 슬라이스 방식(`train_phase2.py:378-387`, `rollout_sim.py:86-89`)이라 캐시 불일치 배제.

**(b) lang zero-pad 정합.** lang=1024d를 z 폭 2048의 앞블록에 zero-pad, 학습·롤아웃 경로 일치. train/test mismatch 없음.

**(c) hybrid 67.5%는 원인 미분리.** 단일 요인 = phase1_ckpt(InfoNCE λ0.3 refit)로 깔끔하나, ① λ0.3은 타 기질 검증 레시피의 무튜닝 이식, ② **hybrid phase1 AE의 오프라인 건강지표(h R², align cos)가 미기록** — InfoNCE가 ζ의 SigLIP2 블록을 재배치하면 frozen h 디코딩 충실도가 떨어질 수 있는데 미확인. "역할분리+hybrid 반증"인지 "refit AE 품질 저하"인지 분리 불가. **under-diagnosed.**
- 핵심 반증 **noalign 86.0%(이득 상실)** 는 구현 흠결 미발견, 공정한 테스트 — 재실행 불요.

**재실행 처방(조건부, hybrid만)**: s1b_hybrid phase1 ckpt의 recon/cycle R²·align cos 확인 → 저하 시 λ 스윕(0.05~0.3) 또는 contrast_proj(이미 존재 `networks.py:204-205`, 미사용) 활성 팔.

---

## 3) F3 ObsFusion dense-obs — 판정: **NAIVE-BOLT-ON(모듈 품질) + 120ep 레짐 confound — "정보가 해롭다" 결론은 과대해석**

**(a) 팔 간 예산 동등성은 있으나 저데이터 레짐**: 500ep 중 120ep 서브셋, no-obs 기준선 자체가 50.0%, obs 팔은 +9.4M 파라미터 **무규제** — 악화가 "정보 유해"인지 "과적합/최적화 실패"인지 분리 불가. full-data 재검 없음.

**(b) 모듈 결함**: **KV LayerNorm 부재**(`obs_fusion.py:71,77-79` — raw patch 토큰 무정규화 MHA 투입; 이후 만든 f4 readout엔 `ln_kv` 있음 = F3가 1세대 naive 구현이라는 내부 증거) · 쿼리 init `torch.randn`(std 1.0, 관례 0.02 대비 과대) · obs 토큰 dropout/게이트/규제 전무 · pos-emb 부재(공간 선택이 목적인데).

**(c)** 오프라인 val도 a>b>c 동일 방향 → 순수 폐루프-shortcut과는 다르고 120ep 레짐 용량/최적화 문제와 부합. 단 b(mean 1토큰)조차 −18.5pp는 과적합만으론 빠듯 — 진상 규명 실험(full-data, kv-LN 수정판, GridObs 완주) 모두 미실행.

**(d) 통계**: 그 레짐 내 악화 자체는 견고(z 3.8~7.9). 문제는 외적 타당성(120ep→full-data 일반화, "F6 불요" 종결). 인용 산출물 `outputs/report/f3_obs_fusion.json` **미존재**.

**재실행 처방**: full 500ep no-aug 클린 밴드에서 arm a/c + ObsFusion(kv-LN, pos-emb, init 0.02, 토큰 dropout) + GridObs(param-0 avg-pool) 완주.

---

## 4) Phase-B wrist DualDeltaAE — 판정: **INCONCLUSIVE (실행 실패로 결론 미성립; known scale-risk 미완화)**

**(a) 스케일 위험을 스스로 표기하고 방치**: config "⚠ SCALE" 주석(main=SigLIP2 raw vs wrist=DINOv3 CLS 단위벡터)에도 완화 장치 전무 — 스트림별 ζ 표준화 없음, align 동일가중 MSE(스케일² 비례→main 지배), concat ζ 전체 단일 스칼라 x0_std, h의 단일 LayerNorm이 저분산 wrist 블록 감쇠. std 비율은 print만 하고 아티팩트 없음.

**(b)** "align_wrist 수렴"은 cos 항의 스케일 불변성으로 성립 가능; act R² +0.663 ≈ 단일 스트림 0.655 — **ζ_wrist 기여의 증거 아님**. 결정적 진단(오프라인 ζ_wrist zero-ablation R²)이 부재 — 5분짜리 검사.

**(c)** "isolation base"라지만 ① wrist 변위 스트림 추가와 ② wrist 조건 기질 교체(SigLIP2 1토큰→DINOv3-CLS 2토큰)가 동시 변경. rollout-학습 경로 정합, 구현 버그는 미발견.

**(d) "baseline 85-88 하회"는 통계적으로 미성립.** 롤아웃이 결정론적(eval에서 source_noise 미적용, h=MLP, init 고정)이라 3회 부분판독 = **같은 결정론적 에피소드 열의 서로 다른 절단 창(독립 표본 아님)**. 실효 증거 = 최장 판독 64/84 = 76.2%, Wilson 95% CI **[66.1, 84.0]**; vs 85%는 z≈−1.67(p≈0.10, NS). 태스크 순차 실행으로 84ep = task0-3+task4 4ep = **커버리지 편향**, matched baseline 부재. 성립하는 것은 "84ep·앞쪽 4.2개 태스크에서 uplift의 증거 없음"까지.

**재실행 처방**: ④ 오프라인 ζ_wrist zero-ablation 선행(기여 0이면 폐루프 불필요) → ① retry-supervisor 200ep 완주, ② matched large256-single baseline 동시 창 paired 비교, ③ ζ 블록별 표준화 + 블록별 x0_std.

---

## 5) 헤드라인 수치 아티팩트 추적 — **10/10 UNTRACED**

구조적 원인: `rollout_sim.py:319-337`이 결과를 `outputs/eval/rollout_{suite}_{mode}.txt` **단일 파일에 per-task % 집계만** 기록, 실험/ckpt명이 파일명에 없어 **매 실행이 덮어씀**. per-episode 성공 플래그 미저장(`paired_ci.py` 헤더가 스스로 인정). `.gitignore`가 outputs/checkpoints/docs 제외.

| 수치 | 로컬 1차 아티팩트 | 판정 |
|---|---|---|
| concat 97.5 / avg 91.5 | 없음 (config만) | UNTRACED |
| no-aug baseline 85.0/85-88 | `experiments/baseline_5rep.jsonl`(구 CLIP 레짐, 집계만) — 레짐 불일치 | UNTRACED |
| h-flow 33/37 · actionflow 76/80 · residflow 48-65 | 없음 | UNTRACED |
| wrist 66.7/76.2/80.6 | 없음 — 크래시/supervisor 로그 전무 | UNTRACED |
| F3 50/31.5/15.5 | 인용된 `outputs/report/f3_obs_fusion.json` 미존재 | UNTRACED |
| C1/C2 SR·α값 · S1b 86.0/67.5 | markdown 표만; ckpt 로컬 부재 | UNTRACED |

**주의**: "날조"가 아니라 "이 워크스페이스에서 검증 불가"라는 뜻 — wandb(project `clipvp`)와 원격 `~/clip_ws/checkpoints/grid/*.pt`에 사본 존재 가능 → **회수가 최우선 조치**. 게이트 α/β 검증은 ckpt 로드 수 초.

## 종합 판정표

| 항목 | 구현 품질 | negative의 지위 (보고 수치 신뢰 가정) | 재실행 |
|---|---|---|---|
| C1/C2 fine 채널 | NAIVE-BOLT-ON (∂L_act/∂α≡0) | "이 설계 무이득"만 지지; "fine 정보 무가치" 일반화 미지지 | 예 (§1 처방 5개 또는 GridObs 완주) |
| S1b noalign | BEST-SHOT | 방향상 타당(1시드) | 불요 |
| S1b hybrid | INCONCLUSIVE | 원인 미분리 | 조건부 |
| F3 dense-obs | NAIVE-BOLT-ON + 레짐 confound | 레짐 내 악화 견고; "정보 유해" 일반화 과대 | 예 (full-data+수정 모듈) |
| wrist dual-stream | INCONCLUSIVE | "baseline 하회" 미성립(NS) | 예 (ablation 진단 선행) |

**검증 못 한 것**: α/β 실측·CKA 0.871·오프라인 R²·모든 SR 원데이터·Δz std 비율·wandb 로그 — 코드-측 기전은 확인, 수치는 미재현.
