#!/bin/bash
# Generative zero-shot baseline (tools/zero_shot_generative.py) for a base chat model, 4 GPUs in parallel:
#   bash experiments/05_generative_baseline/run.sh                                   # Qwen3-VL-2B, 500 images per set
#   MODEL=Qwen/Qwen2.5-VL-7B-Instruct LIMIT=300 bash experiments/05_generative_baseline/run.sh
# Same sets, label pools and seeded rows as vast/run_zeroshot.sh's embedder evals, so every result
# pairs up image by image:   python tools/compare_paired.py --runs runs/q2b_inat_ep2 runs/gen_qwen3_vl_2b_instruct
# fp32: in bf16 the cached label scores drift ~0.5 nats from an uncached forward (fp32: exact).
set -uo pipefail
source "$(dirname "$0")/../../vast/env.sh"
MODEL=${MODEL:-Qwen/Qwen3-VL-2B-Instruct}
LIMIT=${LIMIT:-500}
OUT=runs/gen_$(basename "$MODEL" | tr 'A-Z.-' 'a-z__')
mkdir -p "$OUT"
# "file:pool" per job; pool '-' = the file's own labels. Four groups of roughly equal cost.
GROUPS_=(
  "configs/zs_cars.jsonl:- configs/zs_aircraft.jsonl:- configs/zs_pets.jsonl:-"
  "configs/zs_food.jsonl:- configs/zs_dogs.jsonl:- configs/zs_flowers.jsonl:- configs/zs_flowers_common.jsonl:- configs/zs_flowers_latin.jsonl:-"
  "configs/cub200_val.jsonl:- configs/cub200_val_latin.jsonl:- configs/cub200_val_unseen20.jsonl:configs/cub200_val.jsonl configs/cub200_val_latin_unseen20.jsonl:configs/cub200_val_latin.jsonl"
  "configs/inat_val_common.jsonl:- configs/inat_val_latin_matched.jsonl:-"
)
pids=()
for g in "${!GROUPS_[@]}"; do
  files=() pools=()
  for job in ${GROUPS_[$g]}; do
    [ -f "${job%%:*}" ] || continue
    files+=("${job%%:*}"); pools+=("${job#*:}")
  done
  [ ${#files[@]} -gt 0 ] || continue
  CUDA_VISIBLE_DEVICES=$(gpu_id "$g") $PYTHON tools/zero_shot_generative.py --model "$MODEL" --limit "$LIMIT" --dtype fp32 \
    --classify "${files[@]}" --class-pool "${pools[@]}" --output "$OUT/results_$g.json" > "$OUT/log_$g.txt" 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
$PYTHON - "$OUT" <<'EOF'
import glob, json, sys
merged = {}
for f in sorted(glob.glob(f"{sys.argv[1]}/results_*.json")):
    merged.update(json.load(open(f)))
json.dump(merged, open(f"{sys.argv[1]}/results.json", "w"), indent=2)
print("| set | classes | rank top-1 (sum) | rank top-1 (mean) | free answer exact | n |")
print("|---|---|---|---|---|---|")
for k, v in merged.items():
    print(f"| {k} | {v['classes']} | {v['rank_sum']:.3f} | {v['rank_mean']:.3f} | {v['gen']:.3f} | {v['n']} |")
EOF
