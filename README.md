# VLM2Vec: vision-language models as embedding models

Contrastive LoRA fine-tuning that turns a generative vision-language model (Qwen-VL) into an
image+text embedding model, plus dataset builders and embedding-based evaluation (multiple choice,
classification, retrieval). This project uses it to study the **fine-grained (FGVC) vs VQA
trade-off** and the effect of the **label vocabulary** (English vs Latin species names).

**Results: [experiments/README.md](experiments/README.md).** How to run on a GPU node: [vast/README.md](vast/README.md).
Design notes and fixed issues: [docs/DESIGN.md](docs/DESIGN.md).

## Layout

```
vlm2vec/          the library: model + pooling, losses, GradCache, data, trainer, evaluation
train.py          train one run (1 GPU, or torchrun for several)
evaluate.py       score a run on any benchmarks, loading the model once
build_dataset.py  raw dataset -> training/eval JSONL
configs/          training mixtures (YAML) + built JSONL files (not in the repo)
tools/            shared utilities used by several experiments
  compare_paired.py     is A really better than B? (McNemar + bootstrap CI, per image)
  results_db.py         every result in one SQLite file: tables per experiment, SQL queries
  collect_results.py    eval_results.json across runs -> table
  show_predictions.py   what a model got wrong (terminal or HTML gallery)
  name_alignment.py     do English and Latin names of a species embed together?
  zero_shot_generative.py  the base chat model as a classifier (fair baseline)
  make_cub_mixed.py, make_lang_sets.py, make_mmeb_aokvqa.py   experiment datasets
  relocate_paths.py, estimate_cost.py, compare_speed.py      housekeeping
vast/             running on a GPU node: env, setup, push/pull, train.sh, eval.sh,
                  speed benchmark, data downloads, the zero-shot eval suite
experiments/      one folder per question: run.sh + README with results
tests/            pytest suite
```
Base code (`vlm2vec/`, the three entry points, `tools/`, `vast/`) has nothing experiment-specific.
Experiments only call it.

## Quick reference

```bash
python train.py --model_name Qwen/Qwen3-VL-2B-Instruct --dataset_config configs/mixed.yaml \
    --output_dir runs/q2b_mixed --num_train_epochs 1 --per_device_train_batch_size 32
torchrun --standalone --nproc_per_node 4 train.py <same arguments>
python train.py --model_name /workspace/fgvc_sft_ckpt ...     # stack on an adapter (merged first)

python evaluate.py --checkpoint runs/q2b_mixed \
    --mc configs/aokvqa_val.jsonl configs/mmmu_val.jsonl configs/mmbench_dev.jsonl \
    --classify configs/cub200_val.jsonl
python tools/collect_results.py runs/*
python tools/compare_paired.py --runs runs/a runs/b
```

## Results database

```bash
python tools/results_db.py build                           # rebuild runs/results.db from the run dirs
python tools/results_db.py table --experiment 07           # one experiment's runs x tasks
python tools/results_db.py sql "SELECT run, value FROM final WHERE task='cub200_val' ORDER BY value DESC"
python tools/results_db.py sql "SELECT * FROM paired WHERE significant AND run_a='base7b_aokvqa'"
```
SQLite (no server, nothing to install). It is rebuilt from the files the runs already write, so it can't
go out of date with them; `vast/pull.sh` rebuilds the Mac copy after every pull. Tables: `runs`,
`results` (every task of every checkpoint), `paired` (all significance tests), `run_experiments`,
plus the view `final` (each run's final result per task). Schema: `python tools/results_db.py -h`.
Re-running with the same `--output_dir` resumes. All options: `python train.py --help`,
`python build_dataset.py --help`. On the node use the wrappers instead (`vast/train.sh`,
`vast/eval.sh`: tmux, retries, GPU selection).

## Tests

```bash
pytest
python tests/make_tiny_model.py /tmp/tiny-qwen2vl && VLM2VEC_TINY_MODEL=/tmp/tiny-qwen2vl pytest
```
