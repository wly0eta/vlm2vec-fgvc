# 05 · Generative zero-shot baseline

**Question.** An untrained VLM scores near chance *as an embedder*, which says nothing about what
it knows. A fair baseline asks the base chat model (Qwen3-VL-2B) to name the class and scores its
answer against the same label pool (`tools/zero_shot_generative.py`). Same seeded rows as the
embedder evals, so results pair up image by image (`tools/compare_paired.py`).

```bash
bash experiments/05_generative_baseline/run.sh
MODEL=Qwen/Qwen2.5-VL-7B-Instruct LIMIT=300 bash experiments/05_generative_baseline/run.sh
```

## Results: embedder minus generative base, in points (500 images per set; * = McNemar p < .05)

| set | `q2b_inat_ep2` | `q2b_mixed` | `…__cub_mixed_full` |
|---|---|---|---|
| iNat Latin | **+60*** | −2 | **+57*** |
| Flowers Latin / English | **+28*** / −10* | −3 / −11* | **+27*** / −7* |
| CUB unseen-20 En / La | **+19*** / **+26*** | +2 / +1 | **+82*** / **+85*** |
| Cars | −2 | **+17*** | +4 |
| Food | −3 | **+6*** | −4 |
| Pets | −6* | **+11*** | −3 |
| Dogs | −11* | 0 | −9* |
| Aircraft | −4* | −4* | −5* |

**Takeaway.** Contrastive training wins big in the domain it was trained on (and on Latin names,
which the chat model rarely produces). Outside that domain the generative base model is often as
good or better: what the embedder knows comes from its training mixture.
Full tables: `runs/paired_vs_generative.md`.
