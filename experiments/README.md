# Experiments

Each folder is one question: a `run.sh` (or script) that trains and evaluates everything it needs,
and a `README.md` with the setup, the exact commands and the results. Scripts run **on the node**
from anywhere (they `cd` to the repo root through `vast/env.sh`), start themselves in a tmux session,
and skip steps whose results already exist, so rerunning one resumes it. Outputs go to `runs/<run>/`;
`runs.txt` lists each experiment's runs for the results database (`tools/results_db.py`).

| # | experiment | question | headline |
|---|---|---|---|
| 01 | [prompt_probe](01_prompt_probe/) | How sensitive are scores to instruction / template / name wording? | not run on Vast yet |
| 02 | [mentor_inat_mini](02_mentor_inat_mini/) | Continue the Unity 7B iNat run on train_mini; speed vs Unity | not run (disk) |
| 03 | [label_vocab_2b](03_label_vocab_2b/) | How few mixed-label CUB images teach a Latin-trained 2B the English names? | 2/class: CUB En .35 → .65; full: .86 En / .88 La |
| 04 | [label_vocab_7b](04_label_vocab_7b/) | English vs Latin vs mixed CUB labels on the 7B | mixed: .89 in both; CUB .63 → .90 with VQA unchanged |
| 05 | [generative_baseline](05_generative_baseline/) | Embedder vs the base chat model answering by name | big in-domain wins; base model often better out of domain |
| 06 | [hparam_sweep](06_hparam_sweep/) | lr / temperature / rank / epochs / batch size on the 2B | all within ±1.5 pts; best lr 2e-4, 12 ep, batch 128 |
| 07 | [fgvc_transfer](07_fgvc_transfer/) | Transfer with no in-domain data; does Logan's FGVC-SFT stage matter? | A-OKVQA alone: CUB .60; SFT stage adds ~0; +Cars helps all FGVC |

[`run_queue.sh`](run_queue.sh) ran 03 → speed benchmarks → zero-shot suite → 05 → 06 → 04
back to back on Nov 7–9 2026; kept as a record and to rerun the whole batch.

## Adding an experiment

```bash
mkdir experiments/08_<name>
cp experiments/07_fgvc_transfer/run.sh experiments/08_<name>/run.sh   # tmux + env + train/eval skeleton
```
Use the shared pieces instead of copying them: `vast/train.sh` and `vast/eval.sh` (one run),
`vast/run_zeroshot.sh` (the zero-shot FGVC suite), `tools/compare_paired.py` (significance),
`tools/collect_results.py` (tables). Write the question and the results into the folder's README,
and list the folder's run names (globs allowed) in `runs.txt`: `tools/results_db.py` uses it to link
runs to experiments (`python tools/results_db.py table --experiment 08`).
