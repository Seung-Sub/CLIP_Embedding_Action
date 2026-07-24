# 세션 보고서 번들 — 2026-07-22 ~ 07-24 (콜리그 retrieval 검증 · R-시리즈 · W-A 확증 · 논문 v2 · 미래 셀 사전등록)

팔로업 연구자용 인덱스. 요약 서사는 루트 `PROGRESS.md`(최상단 2026-07-22~24 블록)와
`FOLLOWUP_experiments.md`(§16–18)에 있고, 본 폴더는 그 근거가 되는 상세 보고서 원문 10편이다.
`docs/` 원본은 gitignore이므로 **본 reports/ 복사본이 공개 공유본**이다. 직전 번들은
`reports/2026-07-18_wrist_fusion_session/`(감사·재설계·wrist 스크리닝). 읽는 순서 권장:

## 1. 콜리그 retrieval 제어 검증 + R-시리즈 (C3′ 재편)
| 파일 | 내용 | 핵심 결론 |
|---|---|---|
| `ANALYSIS_colleague_retrieval_control.md` | 동료 랩(SigLIP@ed54d17) "retrieval 기반 제어 = wow" 헤드라인의 코드-레벨 재구성 + 우리 T-0 사망과의 양립성 분석 (READ-ONLY) | 그들 기전 = **벡터 주입이 아니라 언어→검색→시연 프리미티브 재생**(정책·h·Δz 미경유). **랩 내 이중 해리**: 같은 어댑터로 주입(E5 잔차 0.58 FAIL)/검색(8/8) 갈림 = 우리 WEEK1 판정의 교차랩 재현. **C3-강(벡터 산술)은 사망 보강, C3-약(검색-매개 선택)은 부활**. 정직 한계: 실행 n=1·camera-frame 0/4·8클래스 단일객체. 우리 채택 = R-0~R-3 포트(banks 재사용) |
| `RESULT_rseries_R0R1.md` | 위 어댑터를 **우리 기질**(large256-single ζ)로 이식 — R-0(오프라인 어댑터), R-1(검색 판별 하네스), §6 R-0b(잔차 어댑터) | 분할 동료-정확 일치(1995/497 seg). canonical **0.972**(그들 0.974) G-R0a PASS, unseen 0.952 PASS, **우리 독립 3rd셋 0.773 G-R0c-2 FAIL = 어휘-반경 한계**(grasp 붕괴). 정직 대조: 텍스트-무관 MLP 0.968≈어댑터(분류기 등가), 상태-잔차화 시 0.749 급락. R-1 이중 해리 재현(correct 8/8·스왑 56/56·셔플 마진 1/3 붕괴). R-0b: canonical 신호의 **70% 상태-무관 / 일반화의 ~55% 상태-운반**, 그리퍼 이벤트 state-free·approach/place 상태-의존 |

## 2. wrist 확증 (캠페인 최초 확증 양성)
| 파일 | 내용 | 핵심 결론 |
|---|---|---|
| `RESULT_wrist_confirmation.md` | **W-A 확증 최종 판정** — 3 train-seed × 2 arm × 2 mode × 500ep = 6,000ep, per-episode JSONL 직접 재계산 | **matchedbase 85.7 / wristpatch 92.6 = +6.9pp, paired CI[+4.9,+9.1] SIG>0** — 캠페인 최초 확증된 양의 SR 아키텍처 추가. per-task 전부 양(파지·재참조 t5+11/t9+13 집중). BUT 언어 공동기준 wristpatch **+63.7** vs base +73.9 = 유보밴드(65–75) 하단도 하회 → **채택 아키텍처 아님, SR↔언어 tradeoff 프런티어의 새 점**. SR↔언어 법칙의 **4번째 독립 재현**. 열린 후보 W-A′ |
| `RESULT_wrist_screening.md` | (직전 번들 재수록) 4팔 스크리닝: base 87.0 / W-A 92.5(+5.5) / W-B 91.5 널 / W-C 82.5 널 | 확증의 스크리닝 근거. W-A 확증 진출·타깃측(W-C) 종결. 확증 +6.9는 스크리닝 +5.5와 정합 |

## 3. 논문 아키텍처 (성공률이 아니라 기여)
| 파일 | 내용 | 핵심 결론 |
|---|---|---|
| `PAPER_ARCHITECTURE_v2.md` | PI 질문("SR이 아니라 contribution")에 대한 논문 설계도 v2 — `PAPER_SKELETON_v1` 대체 | 상품 = **3개 법칙급 발견**: C-1(구조적 언어 상속, 개입 없이 Biased 2.5% vs VLA 45–79%), C-2(삽입점 지도 + **SR↔언어 tradeoff LAW를 조직 프레임으로 통합**, 다이얼 3회 재현), C-3=C3′(주입 사망/검색 생존 이중 해리 + 정직한 상태-지분·어휘반경 한계). 시그니처 = **F1 프런티어 그림**(신규 런 0). 얼굴 = **해석-루프 데모**(행동 전 의도 판독). 실험 큐 E1–E9(필수 4건 ≈6–9 GPU-일), LIBERO-CF 좌표 변환 계획 |

## 4. 미래 셀 설계 + CPU 사전게이트 (launch-ready 큐)
| 파일 | 내용 | 핵심 결론 |
|---|---|---|
| `DESIGN_WD_WAprime_v1.md` | PI 잔존 직관 2건("손목도 추론", "손목=SigLIP2 통일 공간")을 닫힌 음성 지도 위반 없이 검정하는 셀 2개 사전등록 | **W-D**(AuxΔw — 순수 손실-측 미래 손목변위 예측, h/액션 경로 미진입)·**W-A′**(SigLIP2-공간 손목 패치, 인코더 정체성 1개만 교체 = 파라미터-정확 매치). 팔당 변경 하나, 전 신규경로 guarded byte-identity, 폐루프 SR+언어 공동기준 단독 심판. 실행은 W-A 확증 종료 후 |
| `RESULT_pregates.md` | 위 셀들의 학습-전 CPU 킬게이트 3종(GPU는 W-A 확증 점유) | **G-D1 데모 FAIL** (정책 ζ̂ top-1 0.363, 재적합 후 0.786<0.85 → E2 데모 "오프라인 판독 그림"으로 강등, 마진게이팅 0.856@80% 부분구제). **R-D0 W-D GO 아슬아슬**(+0.0227≥+0.02, 마진 0.0027). **R-A′ W-A′ GO 동등**(SigLIP2 wrist uplift +0.0459 ≈ DINO +0.0456, 비율 1.008 → "DINO 기하 특이" 오프라인 기각) |
| `PREREG_capacity_sweep.md` | PI 의심("phase1 g/h가 용량 부족?")의 g/h hidden-폭 8팔 스윕 사전등록 (오프라인 전용) | large256-single 상속, g/h 폭 {0.5,1,2,4}× 독립 스윕(귀속 분리). 판정: 전팔 ±0.01 → 무죄(용량 의심 영구 종결) / (4×−0.5×)≥+0.03 단조 → 폐루프 셀 개설. **launch-ready**(config 8종·probe 스크립트 완비), GPU 창 대기 |

## 5. 발산 포트폴리오 후속 (직전 번들 설계문서 재수록)
| 파일 | 내용 | 핵심 결론 |
|---|---|---|
| `DESIGN_dualpolicy_dynamic_weighting_v1.md` | (재수록) PI 제안 2건(분리 정책/동적 가중)의 측정-선행 판정 | M-B 교차쌍 h 민감도 0.662→JOINT_REQUIRED(분리 금지), M-C oracle headroom +0.000025→게이트 폐쇄. 정제 양성: wrist=그리퍼 채널 상시 우월. 본 세션 DESIGN_WD_WAprime의 "닫힌 지형" 근거 |
| `DESIGN_patch_policy_attention_v1.md` | (재수록) 관측측 patch 활용 3후보(LangSelPool/GridToken-v2/PatchDelta), F3 결함 전면 수정 스펙 | 본 세션 커밋 `093cc1b`의 P-B LangSelPool(B1) 구현 근거 문서 |

## 코드 커밋 (main 브랜치)
- `093cc1b` **P-B LangSelPool(B1) + g/h 용량스윕 노브** — 텍스트-쿼리 patch pooling(kv-LN/pos-emb/tok+group drop/attn-entropy 로깅, F3 결함 전면 수정판), `patch_obs` 블록(_PATCH_KEYS loud-fail, 롤아웃은 `instruction_for(tid)` 쿼리로 언어 인과 유지), `DeltaAE` hidden_g/hidden_h 독립 노브(기본 비트동형) + cap 스윕 config 8종 + `probe_h_jacobian.py`. 8단 스모크 PASS, B2는 STUB
- `9244b51` 직전 세션(07-18~21) 팔로업 정리 — 감사 3부작·언어 천장지도·P-A crop 종결·wrist 스크리닝, reports 19편 번들 + PROGRESS/FOLLOWUP §12–15
- `2c68d98` W-B Δ̄w-token + W-C 표준화 재판(N3 per-stream buffer/N4 x0_per_dim/wrist_cond_sig) + 검증 스모크 3종
- `3a3d4fe` GridObs guarded 옵션 4종(ln/tok_drop/group_drop/init_std) + grid_obs 미지원 키 loud-fail + 토큰 canonical 순서 스모크
- `c2c7b06` P-A crop 셀: `anchor.crop {none|dino|both}` 콜리그-EXACT center-crop + config 6종(기본 byte-identity)
- `3f0d984` 롤아웃 provenance(per-episode JSONL) + `--flow-noise-mode {fresh,walk,locked}` — 본 세션 W-A 확증 재계산이 이 하네스로 성립

*R-시리즈 포트 코드는 `scratchpad/rseries/`(로컬+원격 동일본, 동료 레포 무변경), 사전게이트는
`scratchpad/pregates/`, 용량스윕 config는 `configs/phase1_libero_large256_cap_*.yaml`.*

## 원 데이터 소재
per-episode JSONL(W-A 확증 6,000ep)·R-시리즈/사전게이트 결과 JSON·프로브 산출물은 공개 레포 미포함
(`outputs/` gitignore) — kist_a6000_ss `/workspace/CLIP_ws/outputs/{eval/runs/<run_tag>, rseries, pregates, capsweep, analysis}`
및 wandb `clipvp`. W-A 확증 판정은 유실된 verdict 스크립트 대신 `outputs/eval/runs/*/episodes.jsonl`에서
직접 부트스트랩 재계산(commit 3f0d984 provenance 하네스의 실효성 자기입증 사례).
