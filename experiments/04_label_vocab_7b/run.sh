#!/bin/bash
# Label-vocabulary experiment on CUB-200, ON THE VAST NODE (tmux session 'cub_vocab'):
#   bash experiments/04_label_vocab_7b/run.sh
#
# Three stage-2 fine-tunes of mentor_mixed that differ ONLY in the label text of the same 5,994
# CUB training images:
#   english  runs/mentor_mixed__cub_full   (already trained on Unity: the control)
#   latin    runs/mentor_mixed__cub_latin  every label Latin
#   mixed    runs/mentor_mixed__cub_mixed  half English, half Latin per class (same steps/compute)
# Same recipe as the English run: 5 epochs, global batch 64, lr 1e-4, epoch checkpoints, LoRA r16.
# Then every arm is scored on CUB-200 English + Latin, the CUB-164 English/Latin pair, A-OKVQA
# (4-way and MMEB 894-way), MMMU and MMBench (forgetting), plus the English<->Latin
# name-embedding alignment test. Summary: python tools/collect_results.py runs/mentor_mixed__cub_*
# ~4.5 h per arm on one A100 at the default settings (64 images/step at ~1.9 images/s); run
# vast/bench_speed.sh first on a new GPU type. ~45 min per eval.
set -euo pipefail
source "$(dirname "$0")/../../vast/env.sh"
if [ -z "${TMUX:-}" ]; then
  tmux new-session -d -s cub_vocab "bash $SELF; exec bash"
  echo "started in tmux session 'cub_vocab'   ->   tmux attach -t cub_vocab"
  exit 0
fi
[ -f configs/cub200_train_mixed.jsonl ] || $PYTHON tools/make_cub_mixed.py
[ -f configs/aokvqa_mmeb.jsonl ] || $PYTHON tools/make_mmeb_aokvqa.py
# speed/memory settings from vast/bench_speed.sh (same maths; the Unity control used the defaults)
FAST=()
[ -f runs/_bench/best.env ] && source runs/_bench/best.env && echo "speed settings: ${FAST[*]}"

for arm in latin mixed; do
  DETACH=0 GLOBAL_BATCH=64 bash vast/train.sh "mentor_mixed__cub_$arm" \
    --model_name runs/mentor_mixed --dataset_config "configs/cub200_$arm.yaml" \
    --num_train_epochs 5 --learning_rate 1e-4 --save_strategy epoch --save_every_minutes 60 \
    --dataloader_num_workers 16 \
    --live_eval_mc configs/aokvqa_val.jsonl --live_eval_classify configs/cub200_val.jsonl \
    --live_eval_every_steps 50 ${FAST[@]+"${FAST[@]}"}
done
for arm in latin mixed; do
  DETACH=0 bash vast/eval.sh "mentor_mixed__cub_$arm"
done
$PYTHON tools/name_alignment.py --embed english=runs/mentor_mixed__cub_full latin=runs/mentor_mixed__cub_latin \
  mixed=runs/mentor_mixed__cub_mixed before=runs/mentor_mixed
$PYTHON tools/collect_results.py runs/mentor_mixed runs/mentor_mixed__cub_full runs/mentor_mixed__cub_latin \
  runs/mentor_mixed__cub_mixed
