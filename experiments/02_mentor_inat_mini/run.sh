#!/bin/bash
# Continue the 7B mentor iNat run on iNat train_mini, ON THE VAST NODE, in tmux session
# 'mentor_inat_mini':  bash experiments/02_mentor_inat_mini/run.sh
#
# * Starts from runs/mentor_inat/checkpoint-3210 (the newest Unity checkpoint: Logan's FGVC-SFT
#   7B + 3,210 steps of iNat contrastive training). Its adapter is merged into the base and a
#   fresh LoRA is trained on top, so per-step compute is the same as the Unity run.
# * Same recipe as Unity's mentor_inat (global batch 256, lr 1e-4, GradCache chunk 8, LoRA r16,
#   256 vision tokens) so the speed comparison is like for like: python tools/compare_speed.py
# * 1 epoch of train_mini = 1,953 steps. Downloads iNat val + train_mini first if missing.
set -euo pipefail
source "$(dirname "$0")/../../vast/env.sh"
if [ -z "${TMUX:-}" ]; then
  tmux new-session -d -s mentor_inat_mini "bash $SELF; exec bash"
  echo "started in tmux session 'mentor_inat_mini'   ->   tmux attach -t mentor_inat_mini"
  exit 0
fi
bash vast/get_inat.sh train_mini
DETACH=0 GLOBAL_BATCH=256 bash vast/train.sh mentor_inat_mini \
  --model_name runs/mentor_inat/checkpoint-3210 --dataset_config configs/inat_mini.yaml \
  --num_train_epochs 1 --learning_rate 1e-4 --save_strategy epoch --save_every_minutes 30 \
  --dataloader_num_workers 16 \
  --live_eval_mc configs/aokvqa_val.jsonl --live_eval_classify configs/inat_val.jsonl \
  --live_eval_every_steps 500
