# configs/
- `config.yaml`        : CLIP 백본 경로·정밀도 (core/clip_wrapper.py가 읽음)
- `phase1_libero.yaml` : delta-AE 학습 — state_cond, pooled Δz, 16청크
- `phase2_libero.yaml` : 정책 f 학습 — flow matching + 언어·손목캠 토큰
- `phase2_libero_mlp.yaml` : 베이스라인(MLP 회귀) 비교용

모든 값에 근거가 주석으로 달려 있음. 실험 시 파일 수정 대신
`python src/training/train_phase*.py --set key=value --tag 이름` 오버라이드 권장.
