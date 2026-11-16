# 03 · Label vocabulary, 2B: how little data teaches English names?

**Question.** `q2b_inat_ep2` (Qwen3-VL-2B, 2 epochs of iNat with Latin labels) is strong on Latin
names but weak on English ones. How few CUB images with mixed English/Latin labels does it take to
learn the English names, and does it keep the Latin ones?

Arms (same images under both names, `tools/make_cub_mixed.py`; global batch 32, lr 1e-4, LoRA r16
stacked on the iNat LoRA):

| run | rows | epochs | steps |
|---|---|---|---|
| `q2b_inat_ep2__cub_mixed_2shot` | 400 (1 En + 1 La image per class) | 20 | 250 |
| `…_4shot` | 800 | 10 | 250 |
| `…_10shot` | 2,000 | 6 | 375 |
| `…_full` | 5,994 | 5 | 935 |

```bash
bash experiments/03_label_vocab_2b/run.sh        # tmux session 'q2b_vocab'
```

## Results (top-1)

| run | CUB En | CUB La | A-OKVQA | MMMU | MMBench |
|---|---|---|---|---|---|
| start (`q2b_inat_ep2`) | .347 | .811 | .569 | .297 | .634 |
| 2-shot | .648 | .786 | .557 | .297 | .635 |
| 4-shot | .741 | .821 | .543 | .293 | .630 |
| 10-shot | .804 | .844 | .559 | .309 | .629 |
| full | .859 | .876 | .541 | .294 | .635 |

**Takeaway.** Two images per class nearly double English accuracy (.35 → .65). Full data reaches .86
in English and still improves Latin (.81 → .88). A-OKVQA drops about 3 points at full data.
