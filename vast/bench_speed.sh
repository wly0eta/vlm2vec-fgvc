#!/bin/bash
# Short speed/memory benchmark of training settings that leave the maths unchanged, ON THE NODE:
#
#   bash vast/bench_speed.sh                       # 7B mentor_mixed on CUB, global batch 64
#   STEPS=10 GLOBAL_BATCH=64 MODEL=runs/mentor_mixed bash vast/bench_speed.sh
#   CONFIGS="ckpt:8:sdpa nockpt:8:flash_attention_2" bash vast/bench_speed.sh
#   CUDA_VISIBLE_DEVICES=0 NGPUS=1 MODEL=runs/q2b_inat_ep2 GLOBAL_BATCH=128 OUT=runs/_bench_2b_128 \
#     CONFIGS="nockpt:16:flash_attention_2 nockpt:32:flash_attention_2" bash vast/bench_speed.sh
#
# Each config is checkpointing(ckpt|nockpt):gc_chunk_size:attention. For every one it trains
# STEPS steps into a throwaway run dir, then reports s/step (first 3 steps skipped) and the peak
# GPU memory; an out-of-memory config is reported as OOM. Results: runs/_bench/results.md
# (one table; rerunning appends) and runs/_bench/best.env (highest model TFLOP/s with >= 10%
# memory to spare, which experiments/04_label_vocab_7b/run.sh uses). The whole script is one { } block: safe to edit while running.
{
set -uo pipefail
source "$(dirname "$0")/env.sh"
STEPS=${STEPS:-10}
GLOBAL_BATCH=${GLOBAL_BATCH:-64}
MODEL=${MODEL:-runs/mentor_mixed}
DATA=${DATA:-configs/cub200.yaml}
CONFIGS=${CONFIGS:-"ckpt:8:sdpa ckpt:8:flash_attention_2 ckpt:16:flash_attention_2 nockpt:4:flash_attention_2 nockpt:8:flash_attention_2 nockpt:16:flash_attention_2"}
OUT=${OUT:-runs/_bench}
mkdir -p "$OUT"; rm -f "$OUT/best.tf" "$OUT/best.env"
GPU_NAME=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
{ echo; echo "## $(date '+%F %H:%M') -- $NGPUS x $GPU_NAME, $MODEL, global batch $GLOBAL_BATCH, $STEPS steps"
  echo "| checkpointing | gc_chunk | attention | s/step | images/s | TFLOP/s per GPU (model / hw) | peak GB (alloc / reserved) |"
  echo "|---|---|---|---|---|---|---|"; } >> "$OUT/results.md"

for cfg in $CONFIGS; do
  IFS=: read -r ck chunk attn <<< "$cfg"
  [ "$chunk" -le $((GLOBAL_BATCH / NGPUS)) ] || { echo "skip $cfg: chunk > rows per GPU"; continue; }
  name=$OUT/${ck}_${chunk}_${attn}
  rm -rf "$name"; mkdir -p "$name"
  echo "== $cfg"
  torchrun --standalone --nproc_per_node "$NGPUS" train.py --output_dir "$name" --model_name "$MODEL" \
    --dataset_config "$DATA" --max_steps "$STEPS" --logging_steps 1 --save_every_minutes 0 \
    --per_device_train_batch_size $((GLOBAL_BATCH / NGPUS)) --gc_chunk_size "$chunk" \
    --gradient_checkpointing $([ "$ck" = ckpt ] && echo true || echo false) \
    --attn_implementation "$attn" --dataloader_num_workers 8 > "$name/log.txt" 2>&1
  rc=$?
  if grep -q "OutOfMemoryError\|CUDA out of memory" "$name/log.txt"; then
    row="OOM | | | |"
  elif [ $rc -ne 0 ]; then
    row="failed (rc $rc, see $name/log.txt) | | | |"
  else
    sps=$($PYTHON tools/compare_speed.py "$name/log.txt" --batch "$GLOBAL_BATCH" --total "$STEPS" --skip 3 | tail -1 | cut -d'|' -f5 | tr -d ' ')
    alloc=$(grep -oE "'peak_mem_gb': '?[0-9.]+" "$name/log.txt" | tr -d "'" | awk '{print $2}' | sort -n | tail -1)
    res=$(grep -oE "'reserved_mem_gb': '?[0-9.]+" "$name/log.txt" | tr -d "'" | awk '{print $2}' | sort -n | tail -1)
    # median over the run's log lines, minus the first (its interval includes model loading)
    med() { grep -oE "'$1': '?[0-9.]+" "$name/log.txt" | tr -d "'" | awk '{print $2}' | tail -n +2 | sort -n |
            awk '{v[NR] = $1} END { if (NR) print (NR % 2 ? v[(NR + 1) / 2] : (v[NR / 2] + v[NR / 2 + 1]) / 2) }'; }
    ips=$(awk -v s="$sps" -v b="$GLOBAL_BATCH" 'BEGIN { if (s > 0) printf "%.2f", b / s }')
    row="$sps | $ips | $(med model_tflops) / $(med hw_tflops) | $alloc / $res |"
    # most useful work per second (model TFLOP/s: same batch and data, so it ranks configs exactly
    # like images/s) among configs whose peak ALLOCATED memory stays under 90% of the card ->
    # $OUT/best.env (not reserved: PyTorch's cache grows into whatever is free, ~30 of 32 GB here)
    tf=$(med model_tflops)
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [ -n "$tf" ] && awk -v r="$alloc" -v t="$total" 'BEGIN { exit !(r * 1024 < 0.9 * t) }' && \
       { [ ! -f "$OUT/best.tf" ] || awk -v a="$tf" -v b="$(cat "$OUT/best.tf")" 'BEGIN { exit !(a > b) }'; }; then
      echo "$tf" > "$OUT/best.tf"
      echo "FAST=(--gradient_checkpointing $([ "$ck" = ckpt ] && echo true || echo false) --gc_chunk_size $chunk --attn_implementation $attn)  # $tf model TFLOP/s per GPU, $sps s/step, $res GB reserved" > "$OUT/best.env"
    fi
  fi
  echo "| $ck | $chunk | $attn | $row" | tee -a "$OUT/results.md"
  rm -rf "$name"/checkpoint-* "$name"/adapter_model.safetensors
done
echo; cat "$OUT/results.md"; echo; cat "$OUT/best.env" 2>/dev/null
exit 0
}
