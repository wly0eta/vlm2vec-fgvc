# 04 · Label vocabulary, 7B: English vs Latin vs mixed CUB labels

**Question.** Does fine-grained training in one label vocabulary carry over to the other? Three
stage-2 fine-tunes of `mentor_mixed` (Logan's FGVC-SFT 7B + contrastive A-OKVQA + iNat) that differ
**only** in the label text of the same 5,994 CUB training images:

| arm | run | labels |
|---|---|---|
| English | `mentor_mixed__cub_full` (Unity control) | common names |
| Latin | `mentor_mixed__cub_latin` | scientific names |
| mixed | `mentor_mixed__cub_mixed` | half English, half Latin per class |

Same recipe for all three: 5 epochs, global batch 64, lr 1e-4, LoRA r16. Each arm is also scored on
the English↔Latin name-embedding alignment test (`tools/name_alignment.py`).

```bash
bash experiments/04_label_vocab_7b/run.sh        # tmux session 'cub_vocab'
```

## Results (top-1)

| arm | CUB En | CUB La | en→la / la→en alignment | A-OKVQA | MMMU | MMBench |
|---|---|---|---|---|---|---|
| start (`mentor_mixed`) | .629 | .417 | .46 / .47 | .843 | .420 | .790 |
| English | .896 | .443 | .465 / .50 | .844 | .443 | .790 |
| Latin | .695 | .892 | .70 / .705 | .841 | .417 | .789 |
| **mixed** | **.890** | **.889** | **.915 / .885** | .839 | .440 | .786 |

Few-shot English (Unity): 1-shot .771, 5-shot .847, full .896.

**Takeaways.**
* FGVC vs VQA: CUB goes .63 → .90 with A-OKVQA, MMMU and MMBench unchanged.
* Vocabulary does not transfer: English-only training barely moves Latin, and vice versa. Mixed
  labels get about 89% in both at the same compute, and pull the two names of each species
  together in embedding space (alignment .46 → .92).

`report_cub_full_english.html`: prediction gallery of the English arm (`tools/show_predictions.py --html`).
