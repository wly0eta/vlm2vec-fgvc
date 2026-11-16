#!/bin/bash
# English-name mapping on the iNat-trained 2B, ON THE VAST NODE (tmux session 'q2b_vocab'):
#   bash experiments/03_label_vocab_2b/run.sh
#
# Start: runs/q2b_inat_ep2 = q2b_inat/checkpoint-20990 (Qwen3-VL-2B, 2 epochs of iNat with Latin
# names): best Latin model (CUB Latin .811, iNat .672) but weak on English names (CUB .347).
# Question: how little CUB data with mixed English/Latin labels teaches it the English names,
# and does it keep the Latin ones? Same images under both names, seeded (tools/make_cub_mixed.py):
#   2shot  400 rows (1 English + 1 Latin image per class)  20 epochs   250 steps
#   4shot  800 rows                                         10 epochs   250 steps
#   10shot 2,000 rows                                        6 epochs   375 steps
#   full   5,994 rows                                        5 epochs   935 steps
# Recipe of the earlier 2B CUB runs: global batch 32, lr 1e-4, LoRA r16 (stacked on the iNat one).
# Then every arm + the start point are scored on CUB English/Latin, CUB-164, A-OKVQA (4-way and
# 894-way), MMMU and MMBench (4 evals in parallel, one per GPU), and the name-alignment test.
set -euo pipefail
source "$(dirname "$0")/../../vast/env.sh"
if [ -z "${TMUX:-}" ]; then
  tmux new-session -d -s q2b_vocab "bash $SELF 2>&1 | tee -a runs/q2b_vocab.log; exec bash"
  echo "started in tmux session 'q2b_vocab'   ->   tmux attach -t q2b_vocab   (log: runs/q2b_vocab.log)"
  exit 0
fi
START=runs/q2b_inat_ep2
[ -f "$START/adapter_model.safetensors" ] || { echo "missing $START (push q2b_inat__checkpoint-20990 there)"; exit 1; }
[ -f configs/cub200_train_mixed_10shot.jsonl ] || $PYTHON tools/make_cub_mixed.py --shots 2 4 10
[ -f configs/aokvqa_mmeb.jsonl ] || $PYTHON tools/make_mmeb_aokvqa.py

# 2B weights + LoRA fit easily in 32 GB: no gradient checkpointing, one GradCache chunk per GPU
FAST=(--gradient_checkpointing false --gc_chunk_size 8 --attn_implementation auto)
ARMS=()
for spec in 2shot:20 4shot:10 10shot:6 full:5; do
  IFS=: read -r k epochs <<< "$spec"
  data=configs/cub200_mixed_$k.yaml; [ "$k" = full ] && data=configs/cub200_mixed.yaml
  run=q2b_inat_ep2__cub_mixed_$k
  ARMS+=("$run")
  DETACH=0 GLOBAL_BATCH=32 bash vast/train.sh "$run" --model_name "$START" --dataset_config "$data" \
    --num_train_epochs "$epochs" --learning_rate 1e-4 --save_every_minutes 30 --dataloader_num_workers 8 \
    --live_eval_classify configs/cub200_val.jsonl --live_eval_every_steps 50 "${FAST[@]}"
done

echo "== evals ($(date))"
pids=()
for i in "${!ARMS[@]}"; do
  GPU=$((i % NGPUS)) DETACH=0 bash vast/eval.sh "${ARMS[$i]}" > /dev/null 2>&1 & pids+=($!)
done
wait "${pids[@]}"
# the start point already has every set from Unity except the 894-way A-OKVQA
DETACH=0 bash vast/eval.sh q2b_inat_ep2 --mc configs/aokvqa_mmeb.jsonl > /dev/null 2>&1 || true

$PYTHON tools/name_alignment.py --output runs/q2b_vocab_alignment.json --embed start=$START \
  $(for r in "${ARMS[@]}"; do printf '%s=runs/%s ' "${r##*mixed_}" "$r"; done)
$PYTHON tools/collect_results.py $START $(printf 'runs/%s ' "${ARMS[@]}")
echo "== done ($(date))"
