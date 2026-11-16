# 01 · Prompt probe

**Question.** How much do classification scores depend on the exact wording: the query
instruction, the label template ("a photo of a {name}."), and which name is used (English vs Latin)?

No training: score an existing checkpoint under several wordings, plus CLIP-style prompt and
name ensembling.

```bash
python experiments/01_prompt_probe/probe_prompts.py --checkpoint runs/q2b_inat \
    --classify configs/cub200_val.jsonl --paired configs/cub200_val_latin.jsonl
python experiments/01_prompt_probe/probe_prompts.py --collect runs/*/prompt_probe.json
```
Output: `runs/<run>/prompt_probe.json`. Test: `tests/test_probe_prompts.py`.

**Status.** Built and tested on Unity, not yet run on Vast. No results in this repo.
