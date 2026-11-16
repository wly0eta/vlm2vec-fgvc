#!/bin/bash
# Train (or resume) a run ON THE VAST NODE. Replaces slurm/train.sbatch.
#
#   bash vast/train.sh <run_name> [train.py args...]
#
# * Uses every GPU on the node (NGPUS=... to override) via torchrun.
# * GLOBAL_BATCH=N sets --per_device_train_batch_size to N / NGPUS, so a run keeps the same
#   examples-per-step (and LR schedule) whatever GPU count it lands on -- set it when resuming a
#   Unity run: q2b_inat/mentor_inat used 256, q2b_mixed 32, mentor_mixed 64.
# * Runs detached in a tmux session named after the run (DETACH=0 to stay in the foreground):
#   tmux attach -t <run_name>   to watch, Ctrl-B D to leave it running.
# * Re-running the same command resumes from the newest checkpoint in runs/<run_name>.
# * Crashes are retried (MAX_RETRIES, default 3); `touch runs/<run_name>/STOP` saves and stops
#   cleanly without a retry. A finished run has runs/<run_name>/TRAINING_COMPLETE.
# * Vast instances can be interrupted: checkpoints are written every 30 min (--save_every_minutes);
#   after a restart just run the same command again.
set -uo pipefail
source "$(dirname "$0")/env.sh"
[ $# -ge 1 ] || { sed -n 2,20p "$0"; exit 2; }
NAME=$1; shift
OUT=runs/$NAME

if [ -z "${TMUX:-}" ] && [ "${DETACH:-1}" = 1 ] && command -v tmux >/dev/null; then
  tmux has-session -t "$NAME" 2>/dev/null && { echo "tmux session '$NAME' already exists: tmux attach -t $NAME"; exit 1; }
  tmux new-session -d -s "$NAME" "$(printf '%q ' env DETACH=0 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" GLOBAL_BATCH="${GLOBAL_BATCH:-}" NGPUS="$NGPUS" MAX_RETRIES="${MAX_RETRIES:-3}" bash "$0" "$NAME" "$@"); exec bash"
  echo "started in tmux session '$NAME' on $NGPUS GPU(s)   ->   tmux attach -t $NAME"
  echo "log: $OUT/logs/   curves: $OUT/plots/training_curves.png"
  exit 0
fi

[ -f "$OUT/TRAINING_COMPLETE" ] && { echo "$OUT is already complete ($(cat "$OUT/TRAINING_COMPLETE"))"; exit 0; }
mkdir -p "$OUT/logs"
EXTRA=()
if [ -n "${GLOBAL_BATCH:-}" ]; then
  [ $((GLOBAL_BATCH % NGPUS)) -eq 0 ] || { echo "GLOBAL_BATCH=$GLOBAL_BATCH not divisible by $NGPUS GPUs"; exit 2; }
  EXTRA=(--per_device_train_batch_size $((GLOBAL_BATCH / NGPUS)))
fi

for attempt in $(seq 1 "${MAX_RETRIES:-3}"); do
  LOG=$OUT/logs/train-$(date +%Y%m%d-%H%M%S).log
  echo "[$(date)] attempt $attempt on $NGPUS GPU(s); log: $LOG"
  # train.py: the last occurrence of an option wins, so EXTRA overrides a batch size in "$@"
  torchrun --standalone --nproc_per_node "$NGPUS" train.py --output_dir "$OUT" "$@" ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee "$LOG"
  RC=${PIPESTATUS[0]}
  if [ -f "$OUT/TRAINING_COMPLETE" ]; then echo "[$(date)] $NAME complete"; exit 0; fi
  if [ -f "$OUT/STOP" ]; then echo "[$(date)] stopped on request (rm $OUT/STOP and rerun to resume)"; exit 0; fi
  echo "[$(date)] training exited with $RC; retrying in 30 s from the newest checkpoint"
  sleep 30
done
echo "[$(date)] $NAME failed ${MAX_RETRIES:-3} times -- see $OUT/logs/"
exit 1
