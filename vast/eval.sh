#!/bin/bash
# Evaluate ON THE VAST NODE. Replaces slurm/eval.sbatch + submit_extra_eval.sh.
#
#   bash vast/eval.sh <target> [evaluate.py args...]
#
# <target>:
#   q2b_mixed__cub_full                  a run in runs/ (final adapter, or newest checkpoint)
#   q2b_inat/checkpoint-20990            one checkpoint
#   base:Qwen/Qwen3-VL-2B-Instruct       an untrained model (results -> runs/base_<name>/)
# With no extra args it scores the standard sets (A-OKVQA 4-way and MMEB-style 894-way, MMMU-805, MMBench, CUB English +
# Latin + CUB-164 pair, and iNat val if its images are present). Extra args replace them, e.g.
#   bash vast/eval.sh mentor_mixed --classify configs/cub200_val.jsonl
# Env: EACH_EPOCH=1 (score every epoch checkpoint), GPU=1 (which GPU; run several evals in
# parallel on a multi-GPU node), DETACH=0 (foreground). Results merge into eval_results.json.
set -uo pipefail
source "$(dirname "$0")/env.sh"
[ $# -ge 1 ] || { sed -n 2,17p "$0"; exit 2; }
TARGET=$1; shift

if [[ "$TARGET" == base:* ]]; then
  MODEL=${TARGET#base:}
  NAME=base_$(basename "$MODEL" | tr 'A-Z.-' 'a-z__')
  WHAT=(--base-model "$MODEL" --output "runs/$NAME/eval_results.json")
else
  NAME=${TARGET//\//_}
  WHAT=(--checkpoint "runs/$TARGET")
  [ "${EACH_EPOCH:-0}" = 1 ] && WHAT+=(--each-epoch)
fi
RUN_DIR=runs/${TARGET%%/*}; [[ "$TARGET" == base:* ]] && RUN_DIR=runs/$NAME

if [ $# -eq 0 ]; then
  SETS=(--mc configs/aokvqa_val.jsonl configs/mmmu_val_1img.jsonl configs/mmbench_dev.jsonl
        configs/aokvqa_mmeb.jsonl
        --classify configs/cub200_val.jsonl configs/cub200_val_latin.jsonl
                   configs/cub164_val_latin.jsonl configs/cub164_val_common.jsonl)
  [ -d "$INAT_ROOT/val" ] && SETS+=(configs/inat_val.jsonl)
  SETS+=(--limit 20000)
else
  SETS=("$@")
fi

SESSION=eval-$NAME
if [ -z "${TMUX:-}" ] && [ "${DETACH:-1}" = 1 ] && command -v tmux >/dev/null; then
  tmux new-session -d -s "$SESSION" "$(printf '%q ' env DETACH=0 GPU="${GPU:-0}" EACH_EPOCH="${EACH_EPOCH:-0}" bash "$0" "$TARGET" "${SETS[@]}"); exec bash"
  echo "started in tmux session '$SESSION' on GPU ${GPU:-0}   ->   tmux attach -t $SESSION"
  exit 0
fi

mkdir -p "$RUN_DIR/logs"
LOG=$RUN_DIR/logs/eval-$NAME-$(date +%Y%m%d-%H%M%S).log
echo "log: $LOG"
CUDA_VISIBLE_DEVICES=$(gpu_id "${GPU:-0}") $PYTHON evaluate.py "${WHAT[@]}" "${SETS[@]}" 2>&1 | tee "$LOG"
exit "${PIPESTATUS[0]}"
