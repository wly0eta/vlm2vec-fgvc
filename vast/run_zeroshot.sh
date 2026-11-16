#!/bin/bash
# Zero-shot FGVC suite (vast/get_zeroshot.sh) on several models, one GPU each, in parallel:
#   bash vast/run_zeroshot.sh                         # the default 2B comparison set below
#   bash vast/run_zeroshot.sh q2b_inat_ep2 mentor_mixed
# Each model: the six zs_* benchmarks (each scored against its own full class list), then the
# 20 iNat-unseen CUB species against the FULL 200-name pool, English and Latin, then the
# English-vs-Latin pairs from vast/get_lang_sets.sh (same images, only the label language differs):
# Flowers-102 (zero-shot domain) and iNat val (in-domain; a seeded 10k-row subset, the same rows in
# both files since they are row-aligned, scored against every matched species). Results merge into
# runs/<model>/eval_results.json; summary table at the end (tools/collect_results.py).
set -uo pipefail
source "$(dirname "$0")/env.sh"
[ -f configs/cub200_val_unseen20.jsonl ] || bash vast/get_zeroshot.sh
MODELS=("$@")
[ ${#MODELS[@]} -gt 0 ] || MODELS=(q2b_inat_ep2 base:Qwen/Qwen3-VL-2B-Instruct q2b_mixed q2b_inat_ep2__cub_mixed_full)
while pgrep -f "vast/get_lang_set[s].sh" > /dev/null; do sleep 30; done   # language pairs still building
ZS=(configs/zs_cars.jsonl configs/zs_aircraft.jsonl configs/zs_food.jsonl configs/zs_pets.jsonl
    configs/zs_dogs.jsonl configs/zs_flowers.jsonl)

one() {   # one <gpu> <model>
  local gpu=$1 m=$2
  GPU=$gpu DETACH=0 bash vast/eval.sh "$m" --classify "${ZS[@]}"
  GPU=$gpu DETACH=0 bash vast/eval.sh "$m" --classify configs/cub200_val_unseen20.jsonl --class-pool configs/cub200_val.jsonl
  GPU=$gpu DETACH=0 bash vast/eval.sh "$m" --classify configs/cub200_val_latin_unseen20.jsonl --class-pool configs/cub200_val_latin.jsonl
  [ -f configs/zs_flowers_latin.jsonl ] &&
    GPU=$gpu DETACH=0 bash vast/eval.sh "$m" --classify configs/zs_flowers_common.jsonl configs/zs_flowers_latin.jsonl
  [ -f configs/inat_val_common.jsonl ] && [ -d "$INAT_ROOT/val" ] &&
    GPU=$gpu DETACH=0 bash vast/eval.sh "$m" --classify configs/inat_val_common.jsonl configs/inat_val_latin_matched.jsonl --limit 10000
  return 0
}
pids=()
for i in "${!MODELS[@]}"; do
  one $((i % NGPUS)) "${MODELS[$i]}" > "runs/zeroshot-$i.log" 2>&1 & pids+=($!)
  [ $(( (i + 1) % NGPUS )) -eq 0 ] && { wait "${pids[@]}"; pids=(); }
done
[ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
dirs=()
for m in "${MODELS[@]}"; do
  if [[ $m == base:* ]]; then dirs+=("runs/base_$(basename "${m#base:}" | tr 'A-Z.-' 'a-z__')"); else dirs+=("runs/$m"); fi
done
$PYTHON tools/collect_results.py "${dirs[@]}"
