#!/bin/bash
# Hyperparameter sweep ON THE VAST NODE (tmux session 'sweep'):   bash experiments/06_hparam_sweep/run.sh
#
# Every recipe so far used lr 1e-4, temperature 0.02, LoRA r16/alpha 32 untuned. This tunes them
# on a cheap, honest proxy: the 2B q2b_inat_ep2 trained on 10 mixed-label CUB images per class,
# scored on 5 *other* CUB train images per class (English and Latin; tools/make_cub_mixed.py --tune) --
# never on cub200_val, the test split we report. One run per GPU, 4 at a time, each at global
# batch 32 on its single GPU (same 128 candidates per step as the 4-GPU runs).
#   round 1: learning rate 5e-5 / 1e-4 / 2e-4 / 4e-4
#   round 2, at the best lr: temperature 0.01 / 0.05, LoRA r64 (alpha 128), 2x epochs
#   round 3, at the best so far: global batch (= in-batch negatives) 64 / 128 / 256, and 256 at 2x lr.
#     GradCache keeps memory flat in batch size; same epochs, so fewer, larger steps.
# Score = mean of dev top-1 English and Latin. A-OKVQA (300 val rows) is reported to catch
# forgetting but not optimised. Every run also reports its efficiency: median model TFLOP/s per
# GPU, peak memory and training minutes. Results: runs/_sweep/results.md, best: runs/_sweep/best.env
# Speed settings (checkpointing / GradCache chunk / attention) come from runs/_bench_2b/best.env
# when present (experiments/run_queue.sh benchmarks them first); they don't change the maths.
# Caveat: tuned on the 2B with little data; re-check lr on the 7B before trusting it there.
set -uo pipefail
source "$(dirname "$0")/../../vast/env.sh"
if [ -z "${TMUX:-}" ]; then
  tmux new-session -d -s sweep "bash $SELF 2>&1 | tee -a runs/_sweep.log; exec bash"
  echo "started in tmux session 'sweep'   ->   tmux attach -t sweep   (log: runs/_sweep.log)"
  exit 0
fi
while pgrep -f "zero_shot_generativ[e].py" > /dev/null; do sleep 60; done   # the generative baseline uses the GPUs
START=${START:-runs/q2b_inat_ep2}
EPOCHS=${EPOCHS:-6}
OUT=runs/_sweep
mkdir -p "$OUT"
[ -f configs/cub200_mixed_tune.yaml ] || $PYTHON tools/make_cub_mixed.py --tune 10 5
speed() {   # speed <global batch> -> FAST: the benchmarked fastest settings for that batch size
  FAST=(--gradient_checkpointing false --gc_chunk_size 16 --attn_implementation auto)
  local env=runs/_bench_2b/best.env; [ "$1" -gt 32 ] && env=runs/_bench_2b_128/best.env
  [ -f "$env" ] && source "$env"
  [ "${FAST[3]}" -gt "$1" ] && FAST[3]=$1   # a chunk never needs to exceed the batch on one GPU
  return 0
}
DEV=(configs/cub200_tune_dev_english.jsonl configs/cub200_tune_dev_latin.jsonl)

run_one() {   # run_one <gpu> <name> [overrides...]: train, then score on the dev split
  local gpu name=$2 batch=32 args=(); gpu=$(gpu_id "$1"); shift 2
  local dir=$OUT/$name
  [ -f "$dir/eval_results.json" ] && return 0   # done in an earlier attempt
  while [ $# -gt 0 ]; do   # --global_batch N is ours (train.sh's GLOBAL_BATCH); the rest go to train.py
    if [ "$1" = --global_batch ]; then batch=$2; shift 2; else args+=("$1"); shift; fi
  done
  local FAST; speed "$batch"
  CUDA_VISIBLE_DEVICES=$gpu NGPUS=1 DETACH=0 GLOBAL_BATCH=$batch MAX_RETRIES=1 bash vast/train.sh "_sweep/$name" \
    --model_name "$START" --dataset_config configs/cub200_mixed_tune.yaml --num_train_epochs "$EPOCHS" \
    --learning_rate 1e-4 --save_every_minutes 0 --dataloader_num_workers 8 "${FAST[@]}" \
    ${args[@]+"${args[@]}"} > "$dir.train.log" 2>&1
  CUDA_VISIBLE_DEVICES=$gpu $PYTHON evaluate.py --checkpoint "$dir" --mc configs/aokvqa_val.jsonl \
    --limit 300 --no-predictions > "$dir.eval.log" 2>&1
  CUDA_VISIBLE_DEVICES=$gpu $PYTHON evaluate.py --checkpoint "$dir" --classify "${DEV[@]}" \
    --no-predictions >> "$dir.eval.log" 2>&1     # results merge into the same eval_results.json
  rm -f "$dir"/checkpoint-*/optimizer.pt
}

score() {   # score <name> -> "dev_mean english latin aokvqa"
  $PYTHON - "$OUT/$1/eval_results.json" <<'EOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    en, la = d["cub200_tune_dev_english"]["top1"], d["cub200_tune_dev_latin"]["top1"]
    print(f"{(en + la) / 2:.4f} {en:.4f} {la:.4f} {d['aokvqa_val']['accuracy']:.4f}")
except Exception:
    print("nan nan nan nan")
EOF
}

efficiency() {   # efficiency <name> -> "median model TFLOP/s | peak GB | train minutes"
  local log=$OUT/$1.train.log
  local tf mem secs
  tf=$(grep -oE "'model_tflops': '?[0-9.]+" "$log" | tr -d "'" | awk '{print $2}' | tail -n +2 | sort -n |
       awk '{v[NR] = $1} END { if (NR) print v[int((NR + 1) / 2)] }')
  mem=$(grep -oE "'peak_mem_gb': '?[0-9.]+" "$log" | tr -d "'" | awk '{print $2}' | sort -n | tail -1)
  secs=$(grep -oE "'train_runtime': '?[0-9.]+" "$log" | tr -d "'" | awk '{print $2}' | tail -1)
  echo "${tf:-?} | ${mem:-?} | $(awk -v s="${secs:-0}" 'BEGIN { printf "%.0f", s / 60 }')"
}

declare -A SETTINGS   # name -> its train.py overrides, comma-separated
round() {   # round <spec>... where spec = name:override,override  (all in parallel, one GPU each)
  local i=0 pids=() spec name args
  for spec in "$@"; do
    name=${spec%%:*}; SETTINGS[$name]=${spec#*:}; IFS=, read -ra args <<< "${spec#*:}"
    mkdir -p "$OUT/$name"
    run_one $((i % NGPUS)) "$name" ${args[@]+"${args[@]}"} & pids+=($!)
    i=$((i + 1))
    [ $((i % NGPUS)) -eq 0 ] && { wait "${pids[@]}"; pids=(); }   # one run per GPU at a time
  done
  [ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
  for spec in "$@"; do
    name=${spec%%:*}
    read -r mean en la vqa <<< "$(score "$name")"
    echo "| $name | ${spec#*:} | $mean | $en | $la | $vqa | $(efficiency "$name") |" | tee -a "$OUT/results.md"
  done
}

best_of() {   # best_of <name>... -> name with the highest dev mean
  for n in "$@"; do echo "$(score "$n" | cut -d' ' -f1) $n"; done | grep -v '^nan' | sort -rn | head -1 | cut -d' ' -f2
}

{ echo; echo "## $(date '+%F %H:%M') -- start $START, cub200_mixed_tune (10/class), $EPOCHS epochs, global batch 32"
  echo "| run | settings | dev mean | dev English | dev Latin | A-OKVQA (300) | model TFLOP/s/GPU | peak GB | train min |"
  echo "|---|---|---|---|---|---|---|---|---|"; } >> "$OUT/results.md"

round "lr5e-5:--learning_rate,5e-5" "lr1e-4:--learning_rate,1e-4" \
      "lr2e-4:--learning_rate,2e-4" "lr4e-4:--learning_rate,4e-4"
BEST_LR=$(best_of lr5e-5 lr1e-4 lr2e-4 lr4e-4)
LR=${BEST_LR#lr}
echo "== best learning rate: $LR"

round "t0.01:--learning_rate,$LR,--temperature,0.01" "t0.05:--learning_rate,$LR,--temperature,0.05" \
      "r64:--learning_rate,$LR,--lora_r,64,--lora_alpha,128" \
      "ep2x:--learning_rate,$LR,--num_train_epochs,$((EPOCHS * 2))"
BEST=$(best_of "$BEST_LR" t0.01 t0.05 r64 ep2x)
echo "== best after round 2: $BEST"

B=${SETTINGS[$BEST]}
LR2=$(awk -v l="$LR" 'BEGIN { printf "%g", 2 * l }')
round "gb64:$B,--global_batch,64" "gb128:$B,--global_batch,128" "gb256:$B,--global_batch,256" \
      "gb256_2xlr:${B//--learning_rate,$LR/--learning_rate,$LR2},--global_batch,256"
BEST=$(best_of "$BEST" gb64 gb128 gb256 gb256_2xlr)
echo "TUNED=(${SETTINGS[$BEST]//,/ })   # best: $BEST, dev mean / English / Latin / A-OKVQA: $(score "$BEST")" > "$OUT/best.env"
echo "== best overall: $BEST (--global_batch N = GLOBAL_BATCH=N for vast/train.sh)"
cat "$OUT/results.md"; cat "$OUT/best.env"
rm -rf "$OUT"/*/checkpoint-*
