#!/bin/bash
# FGVC transfer without in-domain data (experiments/07_fgvc_transfer/README.md has the questions,
# the runs and the results). One run per call, in tmux session $RUN, log runs/$RUN.log:
#   bash experiments/07_fgvc_transfer/run.sh                         # Logan's 7B + A-OKVQA + Cars
#   GPU_IDS="0 2" RUN=mentor_aokvqa DATA=configs/aokvqa.yaml EPOCHS=3 bash experiments/07_fgvc_transfer/run.sh
#   GPU_IDS="0 1 2 3" RUN=base7b_aokvqa MODEL=Qwen/Qwen2.5-VL-7B-Instruct BASELINE=mentor_aokvqa \
#     DATA=configs/aokvqa.yaml EPOCHS=3 bash experiments/07_fgvc_transfer/run.sh
# RUN (mentor_aokvqa_cars), DATA (configs/mentor_aokvqa_cars.yaml), EPOCHS (2), GPU_IDS (3),
# MODEL (Logan's adapter, merged; or an HF id), BASELINE (mentor_mixed; run for the paired test).
# Global batch 64 must split evenly over the GPUs (1, 2 or 4).
{
set -uo pipefail
source "$(dirname "$0")/../../vast/env.sh"
if [ -z "${TMUX:-}" ]; then
  RUN=${RUN:-mentor_aokvqa_cars}
  tmux new-session -d -s "$RUN" "env CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' GPU_IDS='${GPU_IDS:-3}' EPOCHS='${EPOCHS:-2}' RUN='$RUN' DATA='${DATA:-}' MODEL='${MODEL:-}' BASELINE='${BASELINE:-}' bash $SELF 2>&1 | tee -a runs/$RUN.log; exec bash"
  echo "started in tmux session '$RUN'   ->   tmux attach -t $RUN   (log: runs/$RUN.log)"
  exit 0
fi
step() { echo; echo "== [$(date '+%F %H:%M')] $*"; }
RUN=${RUN:-mentor_aokvqa_cars}
DATA=${DATA:-configs/mentor_aokvqa_cars.yaml}
MODEL=${MODEL:-$FGVC_CKPT}          # starting point: Logan's adapter (merged) or a plain HF model id
BASELINE=${BASELINE:-mentor_mixed}  # run to compare against (paired tests)

step "Stanford Cars train split (GPUs $CUDA_VISIBLE_DEVICES)"
[ -s configs/cars_train.jsonl ] ||
  $PYTHON build_dataset.py classification --dataset tanganke/stanford_cars --split train \
    --instruction "Identify the car model in the image." --out configs/cars_train.jsonl --image-dir images/cars_train
cat > configs/mentor_aokvqa_cars.yaml <<'EOF'
# VQA + non-bird FGVC mixture (experiments/07_fgvc_transfer/run.sh): 50/50 shares of a 17k + 8k epoch
aokvqa:
  data_path: configs/aokvqa_train.jsonl
  weight: 1.0

cars:
  data_path: configs/cars_train.jsonl
  weight: 1.0
EOF
$PYTHON - <<'EOF'
import json
tr = {json.loads(l)["positive"]["text"] for l in open("configs/cars_train.jsonl")}
te = {json.loads(l)["positive"]["text"] for l in open("configs/zs_cars.jsonl")}
print(f"cars train {len(tr)} classes, test {len(te)}, shared {len(tr & te)}")
EOF

step "waiting for the 7B speed benchmark (runs/_bench/best.env)"
until [ -f runs/_bench/best.env ]; do sleep 120; done
FAST=(); source runs/_bench/best.env; echo "speed settings: ${FAST[*]}"

step "train $RUN"
DETACH=0 GLOBAL_BATCH=64 bash vast/train.sh "$RUN" \
  --model_name "$MODEL" --dataset_config "$DATA" \
  --num_train_epochs "$EPOCHS" --learning_rate 1e-4 --save_strategy epoch --save_every_minutes 60 \
  --dataloader_num_workers 16 \
  --live_eval_mc configs/aokvqa_val.jsonl --live_eval_classify configs/cub200_val.jsonl \
  --live_eval_every_steps 50 ${FAST[@]+"${FAST[@]}"}

step "evaluate"
[ -f "runs/$RUN/TRAINING_COMPLETE" ] || { echo "training did not finish"; exit 1; }
DETACH=0 bash vast/eval.sh "$RUN"
DETACH=0 bash vast/eval.sh "$RUN" --classify configs/cub200_val_latin.jsonl
bash vast/run_zeroshot.sh "$RUN"
$PYTHON tools/compare_paired.py --runs "runs/$RUN" "runs/$BASELINE" | tee "runs/$RUN/paired_vs_$BASELINE.md"
$PYTHON tools/collect_results.py "runs/$BASELINE" "runs/$RUN"
step "done"
exit 0
}
