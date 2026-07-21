# 세션 보고서 번들 — 2026-07-18 ~ 07-21 (감사·재설계·wrist 캠페인)

팔로업 연구자용 인덱스. 요약 서사는 루트 `PROGRESS.md`(로그)와 `FOLLOWUP_experiments.md`(§12–15)에 있고,
본 폴더는 그 근거가 되는 상세 보고서 원문 19편이다. 읽는 순서 권장:

## 1. 신뢰성 감사 (기존 결론의 재검증)
| 파일 | 내용 | 핵심 결론 |
|---|---|---|
| `AUDIT_flow_crosslab_2026-07-18.md` | 콜리그(SigLIP repo) flow 실험 전수 vs 우리 재현의 코드-레벨 대조 | 콜리그 "actionflow 97.2"는 스스로 3-seed 96.2≈base로 강등; decoder-측 h-flow 양성은 애초에 없었음. 우리 포트의 실결함 = quantile actnorm 미이식 + `--flow-fixed-noise`가 진짜 모드락이 아니었던 버그 |
| `AUDIT_negative_results_2026-07-18.md` | 종결된 음성 5건의 "best-shot 여부" 감사 | C1/C2 게이트 ∂L_act/∂α≡0 구조 결함(재개방), F3 naive 모듈+120ep confound(재개방), wrist dual-stream 통계 미성립(재개방), S1b noalign만 BEST-SHOT. 헤드라인 수치 10/10 로컬 UNTRACED → 원격 회수로 7 EXACT + α 실측 7자리 일치 |

## 2. 문헌·포지셔닝
| 파일 | 내용 |
|---|---|
| `LIT_POSITIONING_2026-07-18.md` | LAM 지형, white-space(frozen 언어정렬 Δz=액션표현은 미발표, WALA가 최근접 위협), 기여 후보 C1–C5, 의미교환 입증 패키지 |
| `LIT_wrist_multiview_2026-07-19.md` | eye-in-hand·다중뷰 융합·다중 스트림 손실 문헌(검증 인용). 뷰별 이질 백본은 선례 부재=기회 |
| `ANALYSIS_clip_language_limits_v1.md` | 대조학습 공간의 한계 분석. 핵심: modality gap은 Δ에서 상쇄(이론+실증), binding은 uni-modal에 존재, 유지→천장측정→증거시 이주(E5-V) 전략 |

## 3. 파이프라인 재설계 (설계 문서)
| 파일 | 내용 |
|---|---|
| `DESIGN_pipeline_rethink_v1.md` | 7축 제1원리 재검토: z_t "반칙" 분석(P-zg 프로브 처방), quantile, phase1 손실, 인코더 공간, phase2 토큰, h, 2-stage 정당성. TOP-3 셀(P-A crop/P-B OTTER/P-C JOINT-ζA) + 재개금지 목록 |
| `DESIGN_grounding_space_v1.md` | "g-공간 vs raw Δz" 판정: ζ=저랭크 손실 압축이나 selectivity는 g>dz(농축). T-0(h(Δz_text) zero-shot)·R-Δ 셀 사전등록. FlowPolicy 전 축 KEEP(과적합 갭 2-4×) |
| `DESIGN_patch_policy_attention_v1.md` | 관측측 patch 활용 3후보(LangSelPool/GridToken-v2/PatchDelta), F3 결함 전면 수정 스펙, OOM 정량 해소 |
| `PORTFOLIO_divergent_architectures_v1.md` | 발산 설계 10개 개념 서열(증거필터 E1-E11) — 이후 week-1 게이트에서 언어군 전멸 |

## 4. wrist 캠페인 (설계→검증→실측)
| 파일 | 내용 |
|---|---|
| `DESIGN_wrist_v2.md` → `BRIEF_wrist_design_inputs.md` → `DESIGN_wrist_fusion_unified_v1.md` → `VERIFY_wrist_fusion_v1.md` | 단계형 재설계 → 내부증거 브리프(스케일 6.5×, matched baseline 0회 실행 발견) → 통합 설계(W-A/W-B/W-C) → 적대검증(AMEND A1-A5: GridObs 무음 no-op 차단, 토큰 canonical 순서 등) |
| `DESIGN_dualpolicy_dynamic_weighting_v1.md` | PI 제안 2건(분리 정책/동적 가중)의 측정-선행 판정: M-B 교차쌍 민감도 0.662→JOINT_REQUIRED, M-C oracle headroom +0.000025→게이트 폐쇄. 정제된 양성: wrist=그리퍼 채널 상시 우월(R² 0.881 vs 0.725) |
| `RESULT_wrist_screening.md` | **4팔 판정**: base 87.0 / W-A 92.5(+5.5, 파지태스크 집중 t4+35) / W-B 91.5 널 / W-C 82.5 널(타깃측 종결). W-A 확증(50×3seed) 진출, 언어 +65.5 유보밴드 |

## 5. 오프라인 프로브·게이트 (전부 CPU, 학습 0)
| 파일 | 내용 |
|---|---|
| `WEEK0_probe_results.md` | P-zg RED(상태지름길: ridge가 g를 상회 → innovation-grounding 발동), h eff-rank ~5 재현, ζ_wrist ablation GO(0.179), ego-motion 기각 |
| `WEEK1_gate_results.md` | 발산 언어군 전멸: T-0 FAIL(텍스트 주입≈셔플), A1/A9/A4/A7 KILL, 등급언어 0/6. 생존 자산: M6-a gap 상쇄 실증(0.277), A5 조건부 |

## 6. 실행 기록
| 파일 | 내용 |
|---|---|
| `RUNBOOK_PA_crop.md` | P-A crop 셀 실행 절차 |
| `RESULT_PA_crop_screening.md` | crop 3팔 전부 G-cl 탈락(무이득~유의하락) — "+5pp 전처리 레버" 철회 |

## 코드 (main 브랜치 커밋)
- `3f0d984` 롤아웃 provenance(per-episode JSONL) + `--flow-noise-mode {fresh,walk,locked}`
- `c2c7b06` P-A crop: `anchor.crop {none|dino|both}` + config 6종 (기본 byte-identity 검증)
- `3a3d4fe` GridObs guarded 옵션(ln/tok_drop/group_drop/init_std) + grid_obs 미지원 키 loud-fail + 토큰 canonical 순서 스모크
- `2c68d98` W-B Δ̄w-token + W-C 표준화(N3 per-stream buffer/N4 x0_per_dim/wrist_cond_sig) + 검증 스모크 3종
- 이후 커밋: 확증 시드 config, 프로브 스크립트(`scratchpad/probe_g0_wrist_cell.py`, `g2_zero_ablation_wristpatch.py`), 본 번들

원 데이터(per-episode JSONL, 프로브 JSON)는 공개 레포 미포함(outputs/ gitignore) — kist_a6000_ss `/workspace/CLIP_ws/outputs/{eval/runs,week0_probes,week1_gates,week2_dualdyn,analysis}` 및 wandb `clipvp`.
