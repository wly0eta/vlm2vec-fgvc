# 06 · Hyperparameter sweep (2B)

**Question.** Every recipe used lr 1e-4, temperature 0.02, LoRA r16 untuned. What do the
hyperparameters, including batch size (number of in-batch negatives), actually buy?

Proxy task: `q2b_inat_ep2` trained on 10 mixed-label CUB images per class, scored on 5 *other*
CUB **train** images per class (never `cub200_val`). Score = mean of English and Latin dev top-1.
One run per GPU.

```bash
bash experiments/06_hparam_sweep/run.sh          # tmux session 'sweep'; results: runs/_sweep/results.md
```

## Results (dev mean; A-OKVQA on 300 val rows)

| run | change | dev mean | En | La | A-OKVQA | min |
|---|---|---|---|---|---|---|
| lr5e-5 | lr 5e-5 | .817 | .778 | .855 | .540 | 30 |
| lr1e-4 | default | .823 | .787 | .858 | .550 | 30 |
| lr2e-4 | lr 2e-4 | .830 | .805 | .855 | .530 | 31 |
| lr4e-4 | lr 4e-4 | .826 | .811 | .840 | .567 | 31 |
| t0.01 | lr 2e-4, temp .01 | .816 | .787 | .845 | .537 | 30 |
| t0.05 | lr 2e-4, temp .05 | .834 | .815 | .852 | .553 | 30 |
| r64 | lr 2e-4, LoRA r64 | .825 | .810 | .840 | .573 | 31 |
| ep2x | lr 2e-4, 12 epochs | .841 | .829 | .853 | .550 | 61 |
| gb64 | + batch 64 | .835 | .819 | .850 | .547 | 55 |
| **gb128** | **+ batch 128** | **.847** | .833 | .860 | .540 | 48 |
| gb256 | + batch 256 | .836 | .821 | .850 | .547 | 46 |
| gb256_2xlr | + batch 256, lr 4e-4 | .842 | .832 | .851 | .537 | 46 |

**Takeaway.** Everything sits within about ±1.5 points; training longer helped most (+1.1). Batch
size, rank and temperature are near noise at this data size. Best: lr 2e-4, 12 epochs, batch 128
(`runs/_sweep/best.env`). Tuned on the 2B with little data, so re-check lr before using it on the 7B.
