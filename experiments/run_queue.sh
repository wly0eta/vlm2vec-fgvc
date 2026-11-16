#!/bin/bash
# Run the node's GPU jobs one after another, unattended (tmux session 'queue'):
#   bash experiments/run_queue.sh
#
#   1. wait for the 2B mixed-label scaling batch (experiments/03_label_vocab_2b/run.sh) to finish
#   2. 2B efficiency benchmark, one GPU per run (how the sweep trains): GradCache chunk and
#      checkpointing at global batch 32 and 128, in parallel on GPUs 0 and 1 -> runs/_bench_2b*/
#   3. zero-shot FGVC suite (Cars, Aircraft, Food, Pets, Dogs, Flowers, iNat-unseen CUB species)
#      on the 2B models (vast/run_zeroshot.sh)
#      + the base chat model's generative zero-shot baseline on the same rows (experiments/05_generative_baseline/run.sh)
#   4. hyperparameter sweep on the 2B, incl. batch size / negatives (experiments/06_hparam_sweep/run.sh)
#   5. 7B speed/memory benchmark at global batch 64 (vast/bench_speed.sh) -> runs/_bench/best.env,
#      then the same 7B at global batch 256 (4x the negatives) to see what it costs per image
#   6. 7B English / Latin / mixed-label CUB experiment (experiments/04_label_vocab_7b/run.sh), with the fastest
#      settings from 5. It keeps the English control's hyperparameters (lr 1e-4, batch 64), NOT
#      the tuned ones from 4: the three arms must differ only in label text.
# Each step is skipped when its result already exists, so rerunning resumes the queue.
# The whole script is one { } block: safe to edit while it runs (bash has already read it).
{
set -uo pipefail
source "$(dirname "$0")/../vast/env.sh"
if [ -z "${TMUX:-}" ]; then
  tmux new-session -d -s queue "bash $SELF 2>&1 | tee -a runs/queue.log; exec bash"
  echo "started in tmux session 'queue'   ->   tmux attach -t queue   (log: runs/queue.log)"
  exit 0
fi
step() { echo; echo "== [$(date '+%F %H:%M')] $*"; }

step "waiting for the 2B scaling batch (q2b_vocab)"
while tmux has-session -t q2b_vocab 2>/dev/null && ! grep -q "== done" runs/q2b_vocab.log 2>/dev/null; do
  sleep 60
done

step "2B efficiency benchmark (1 GPU per run)"
if [ ! -f runs/_bench_2b/best.env ]; then
  export NGPUS=1 MODEL=runs/q2b_inat_ep2 DATA=configs/cub200_mixed_10shot.yaml STEPS=12
  CUDA_VISIBLE_DEVICES=$(gpu_id 0) GLOBAL_BATCH=32 OUT=runs/_bench_2b \
    CONFIGS="nockpt:8:flash_attention_2 nockpt:16:flash_attention_2 nockpt:32:flash_attention_2 ckpt:32:flash_attention_2" \
    bash vast/bench_speed.sh > runs/_bench_2b.log 2>&1 &
  CUDA_VISIBLE_DEVICES=$(gpu_id 1) GLOBAL_BATCH=128 OUT=runs/_bench_2b_128 \
    CONFIGS="nockpt:16:flash_attention_2 nockpt:32:flash_attention_2 ckpt:64:flash_attention_2 ckpt:128:flash_attention_2" \
    bash vast/bench_speed.sh > runs/_bench_2b_128.log 2>&1 &
  unset NGPUS MODEL DATA STEPS
  wait
  cat runs/_bench_2b/results.md runs/_bench_2b_128/results.md
fi

step "zero-shot FGVC suite"
while pgrep -f "vast/get_zerosho[t].sh" > /dev/null; do
  sleep 30   # benchmark data still downloading (vast/get_zeroshot.sh); run_zeroshot.sh rebuilds what's missing
done
grep -q "zs_cars" runs/q2b_inat_ep2/eval_results.json 2>/dev/null || bash vast/run_zeroshot.sh

step "generative zero-shot baseline (base Qwen3-VL-2B as a chat model)"
[ -f runs/gen_qwen3_vl_2b_instruct/results.json ] || bash experiments/05_generative_baseline/run.sh

step "hyperparameter sweep"
[ -f runs/_sweep/best.env ] || bash experiments/06_hparam_sweep/run.sh

step "7B inputs"
for need in "$FGVC_CKPT/adapter_model.safetensors" runs/mentor_mixed/adapter_model.safetensors; do
  until [ -f "$need" ] && [ ! -d "$(dirname "$need")/.rsync-partial" ]; do
    echo "waiting for $need (still uploading?)"; sleep 300
  done
done
SKIP_TESTS=1 bash vast/setup.sh > runs/queue_setup.log 2>&1   # relocates the Unity paths in new runs

# the 7B runs keep the control's global batch 64: it must split evenly over the GPUs (3 healthy
# GPUs -> use 2, 32 rows each)
if [ $((64 % NGPUS)) -ne 0 ]; then
  export CUDA_VISIBLE_DEVICES=$(gpu_id 0),$(gpu_id 1) NGPUS=2 GPU_IDS="$(gpu_id 0) $(gpu_id 1)"
  echo "7B steps on GPUs $CUDA_VISIBLE_DEVICES"
fi

step "7B speed/memory benchmark"
if [ ! -f runs/_bench/best.env ]; then
  bash vast/bench_speed.sh
  # the winner again at 4x the batch: does a bigger step use the GPU better, and what memory?
  best=$(sed -E 's/.*--gradient_checkpointing (true|false) --gc_chunk_size ([0-9]+) --attn_implementation ([a-z_0-9]+).*/\1:\2:\3/' runs/_bench/best.env)
  best=${best/#true/ckpt}; best=${best/#false/nockpt}
  GLOBAL_BATCH=256 OUT=runs/_bench_256 CONFIGS="$best" bash vast/bench_speed.sh
fi

step "7B label-vocabulary experiment"
bash experiments/04_label_vocab_7b/run.sh
step "queue done"
exit 0
}
