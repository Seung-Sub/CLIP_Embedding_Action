#!/bin/bash
# LIBERO-Object / LIBERO-Goal 확장 — 표준 2-A 레시피(flow+wrist+lang) 그대로,
# suite만 바꿔 phase1+phase2 학습 → GT평가 → 폐루프(5rep) 평가
set -u
SUITE=$1   # object | goal
source ~/anaconda3/etc/profile.d/conda.sh && conda activate clip_libero
cd ~/clip_ws
if [ "$SUITE" = "object" ]; then
  export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl
  P1CFG=configs/phase1_libero_obj.yaml; P2CFG=configs/phase2_libero_obj.yaml
  LIBERO_SUITE=libero_object
else
  export CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl
  P1CFG=configs/phase1_libero_goal.yaml; P2CFG=configs/phase2_libero_goal.yaml
  LIBERO_SUITE=libero_goal
fi
LOG=scripts/${SUITE}_run.log
GT=experiments/${SUITE}_gt_results.jsonl
SIM=experiments/${SUITE}_5rep.jsonl
mark() { echo "[$(date +%H:%M:%S)] $1" | tee -a "$LOG"; }
num() { echo "${1:-null}"; }
: > "$LOG"; : > "$GT"; : > "$SIM"

mark "[$SUITE] phase1 학습"
python src/training/train_phase1.py --config "$P1CFG" --set wandb.enabled=false >>"$LOG" 2>&1
grep -E "align cos|디코더 h|params" "$LOG" | tail -3 | tee -a "$LOG"

mark "[$SUITE] phase2 학습"
python src/training/train_phase2.py --config "$P2CFG" --set wandb.enabled=false >>"$LOG" 2>&1
grep -E "관절 MAE|policy\[flow\]" "$LOG" | tail -2 | tee -a "$LOG"

mark "[$SUITE] GT평가 (10에피소드)"
for ep in 0 1 2 3 4 5 6 7 8 9; do
  out=$(python src/eval_libero/rollout_dataset.py --config "$P2CFG" --episode $ep 2>>"$LOG")
  pos=$(echo "$out" | grep -oP 'pos \K[\d.]+'); grip=$(echo "$out" | grep -oP '그리퍼 정확도 \K[\d.]+')
  echo "{\"episode\":$ep,\"pos_mae\":$(num "$pos"),\"grip_acc\":$(num "$grip")}" >> "$GT"
done
mark "[$SUITE]   GT 완료"

mark "[$SUITE] 폐루프 (5회 반복, ${LIBERO_SUITE} 20롤/태스크)"
for rep in 1 2 3 4 5; do
  out=$(python src/eval_libero/rollout_sim.py --config "$P2CFG" --suite "$LIBERO_SUITE" --episodes 20 2>>"$LOG")
  mean=$(echo "$out" | grep -oP '평균 성공률: \K[\d.]+')
  echo "{\"rep\":$rep,\"mean_success\":$(num "$mean")}" >> "$SIM"
  mark "[$SUITE]   rep=$rep -> $(num "$mean")%"
done
mark "${SUITE^^}_DONE"
