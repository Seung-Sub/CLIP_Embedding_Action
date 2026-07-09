# src/

> 정착 파이프라인 + **F-시리즈 연구**(설계 `DESIGN_fusion_dense_latent_action_v1.md`,
> 로그 `PROGRESS.md`) 코드. F-시리즈 항목은 **[F]** 로 표기 — 기본값은 정착 레시피와 동일.

| 패키지 | 파일 | 역할 |
|---|---|---|
| core/ | clip_wrapper.py | frozen CLIP ViT-L/14: pooled 768 임베딩 + 패치토큰 |
|       | config.py | configs/config.yaml 로드 |
|       | chunkrep.py | 액션청크 표현(time/basis) 변환 |
|       | anchor.py | **[F]** 다중 백본 앵커 추상화(CLIP/SigLIP2/DINOv2 공통 인터페이스, `get_anchor` 레지스트리). 기본 clip=기존 ClipWrapper와 출력 동일 |
| data/ | libero.py | LIBERO HDF5 로더: 임베딩 캐시, (z_t, z_{t+16}, 청크) 쌍, 정책용 삼중쌍(경계 포함) |
|       | motion_lang.py | **[F]** 청크 모션 문장 생성기(지배축×방향×크기+그리퍼) — HY03 하이브리드 언어정렬 타깃 |
| models/ | networks.py | Phase1 DeltaAE (ChunkEncoder g / ChunkDecoder h). **[F]** `align_mode={dz,direct,hybrid}` + InfoNCE 언어정렬 |
|         | policy.py | Phase2 f 모듈 2종(mlp/flow) + 손실. `latent_dim` 파라미터화(**[F]** F0, phase1 체크포인트에서 주입; CLIP=768) |
| diagnosis/ | f2_dense_probe.py | **[F]** F2 dense 디코더빌리티 프로브: 인코더별 표현→GT action 회귀(RidgeCV/MLP), held-out R²·MAE (오프라인 go/no-go) |
| training/ | train_phase1.py, train_phase2.py | 학습 (--smoke 점검, --set 오버라이드, wandb) |
| eval_libero/ | rollout_dataset.py | GT 에피소드 전체 시계열 추론 → 7차원 그래프 + MAE |
|              | rollout_sim.py | 시뮬 폐루프 실시간 추론 → 성공률·reward·영상. **[F]** `--instruction-mode {correct,wrong,blank}`(언어사용 판별) |
|              | rollout_sim_serial.py, rollout_dataset_serial.py | 다단계 연쇄(월드모델 rollforward, `--n` 재조회 주기; 2-B 모델) |
|              | rollout_sim_paraphrase.py, paraphrases.py | 페러프레이징 전용 폐루프 + 페러프레이징 사전 |
|              | recovery_probe_gui.py | 실패 복구·페러프레이징 관찰 GUI (대화형) |
|              | latent_mapping.py | 잠재공간 PCA 시각화 (대화형) |
