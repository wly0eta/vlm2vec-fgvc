# Running on a Vast.ai node

The training/eval code is the same as on Unity. Only the launch layer changes: `vast/` replaces
`slurm/` (no scheduler, no requeue, no conda modules), and `vast/setup.sh` rewrites the Unity paths
stored in checkpoints and datasets.

| Unity | Vast |
|---|---|
| `slurm/setup_env.sh` | `vast/setup.sh` (on the node) |
| `sbatch slurm/train.sbatch ...` | `bash vast/train.sh <run> ...` (tmux, retries, resume) |
| `sbatch slurm/eval.sbatch ...`, `submit_extra_eval.sh` | `bash vast/eval.sh <run>` |
| `slurm/prep_data.sbatch` | not needed: datasets come from the Unity backup (`vast/push.sh`) |
| `/datasets/ai/inaturalist` | `bash vast/get_inat.sh` -> `/workspace/data/inaturalist` |
| requeue on preemption | rerun the same `vast/train.sh` command (resumes from newest checkpoint) |
| `backup_from_unity.sh` | `bash vast/pull.sh` (node -> Mac) |

Layout on the node: `/workspace/vlm2vec` (this repo + `runs/`, `configs/*.jsonl`, `images/`),
`/workspace/fgvc_sft_ckpt`, `/workspace/data/inaturalist`, `/workspace/hf_cache`.

## 1. Mac -> node (on your Mac)

```bash
cd ~/Downloads/vlm2vec_vast
bash vast/push.sh                 # ~16 GB first time; rerunnable, resumes partial transfers
```
For a new instance: `HOST=root@<ip> PORT=<port> bash vast/push.sh` (both are on the instance's
"Connect" panel; the key is `~/.ssh/unity` unless you set `KEY=`).
Only some runs: `RUNS="q2b_mixed q2b_mixed__cub_full" bash vast/push.sh`. Code only after
edits: `WHAT=code bash vast/push.sh`.

## 2. Set up (on the node)

```bash
ssh -p <port> -i ~/.ssh/<key> -L 8080:localhost:8080 root@<node-ip>   # from the instance's Connect panel
bash /workspace/vlm2vec/vast/setup.sh
bash /workspace/vlm2vec/vast/get_inat.sh          # iNat val (~8 GB), only for iNat evals
```

## 3. Evaluate

```bash
cd /workspace/vlm2vec
bash vast/eval.sh mentor_mixed__cub_full            # standard sets, detached in tmux
EACH_EPOCH=1 bash vast/eval.sh q2b_mixed__cub_full  # every epoch checkpoint
bash vast/eval.sh q2b_inat/checkpoint-20990
bash vast/eval.sh base:Qwen/Qwen3-VL-2B-Instruct    # untrained baseline
GPU=1 bash vast/eval.sh q2b_mixed                   # second GPU, in parallel
python tools/collect_results.py runs/* runs/*/checkpoint-*   # table
```

Every eval also saves each question's outcome to `runs/<run>/predictions/<task>.jsonl` (gold answer,
the model's pick, top candidates with similarities) and logs the most confident mistakes. To read them:
```bash
python tools/show_predictions.py runs/mentor_mixed__cub_full                       # report in the terminal
python tools/show_predictions.py runs/mentor_mixed__cub_full --task cub200_val --examples 20
python tools/show_predictions.py runs/mentor_mixed__cub_full --html runs/report.html   # gallery with images
python tools/show_predictions.py runs/mentor_mixed__cub_full --compare runs/mentor_mixed  # what fine-tuning fixed/broke
```

## 4. Train / resume

`GLOBAL_BATCH` keeps examples-per-step identical to Unity whatever the GPU count (GradCache makes
large per-GPU batches fit), so resumed runs continue the same LR schedule.

```bash
cd /workspace/vlm2vec
# resume the Unity iNat runs (need iNat TRAIN images: bash vast/get_inat.sh train, ~250 GB)
GLOBAL_BATCH=256 bash vast/train.sh q2b_inat --model_name Qwen/Qwen3-VL-2B-Instruct \
  --dataset_config configs/inat.yaml --num_train_epochs 3 --learning_rate 1e-4 \
  --save_strategy epoch --save_every_minutes 30 \
  --live_eval_mc configs/aokvqa_val.jsonl --live_eval_classify configs/inat_val.jsonl --live_eval_every_steps 500
GLOBAL_BATCH=256 bash vast/train.sh mentor_inat --model_name /workspace/fgvc_sft_ckpt \
  --dataset_config configs/inat.yaml --num_train_epochs 2 --learning_rate 1e-4 \
  --save_strategy epoch --save_every_minutes 30 \
  --live_eval_mc configs/aokvqa_val.jsonl --live_eval_classify configs/inat_val.jsonl --live_eval_every_steps 500

# new CUB fine-tune on top of a converted run (no iNat needed)
GLOBAL_BATCH=32 bash vast/train.sh q2b_mixed__cub_full_v2 --model_name runs/q2b_mixed \
  --dataset_config configs/cub200.yaml --num_train_epochs 5 --learning_rate 1e-4 \
  --save_strategy epoch --live_eval_mc configs/aokvqa_val.jsonl \
  --live_eval_classify configs/cub200_val.jsonl --live_eval_every_steps 50
```
Watch: `tmux attach -t q2b_inat` (Ctrl-B D to detach), `tail -f runs/q2b_inat/logs/train-*.log`,
`nvidia-smi`. Stop cleanly: `touch runs/q2b_inat/STOP`. Resume: same command again.

Original Unity settings (for exact reproduction): q2b_inat 3 epochs and mentor_inat 2 epochs at 256
examples/step; q2b_mixed 32, mentor_mixed 64; CUB fine-tunes 32 (2B) / 64 (7B); lr 1e-4, LoRA r16,
temperature 0.02, max_pixels 256*28*28 -- all defaults except those listed.

## 5. Node -> Mac (before destroying the instance!)

```bash
bash vast/pull.sh          # results + adapters;  FULL=1 also brings optimizer state (to resume elsewhere)
```

## Vast-specific notes
* **Disk is per instance.** Destroying an instance deletes `/workspace`; pull first. Stopping keeps it.
* **Interruptible (bid) instances** can be paused at any time: training checkpoints every 30 min;
  rerun the same `vast/train.sh` command after the instance comes back.
* **Port 8080** in the Vast SSH command is usually the instance's Jupyter/portal; don't put your own
  server there. To browse runs: `cd runs && python -m http.server 8888` and connect with
  `-L 8888:localhost:8888`.
* The first run downloads the Qwen weights into `/workspace/hf_cache` (2B ~4.5 GB, 7B ~16 GB).
