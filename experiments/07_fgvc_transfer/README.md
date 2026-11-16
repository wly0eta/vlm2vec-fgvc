# 07 · FGVC transfer without in-domain data (7B)

**Questions.**
1. Does the recipe reach birds (CUB-200) **without any bird training data**?
2. Does adding a *different* fine-grained dataset (Stanford Cars) help other fine-grained domains?
3. How much of the result comes from Logan's FGVC-SFT stage rather than the contrastive stage?

Same recipe for every run (global batch 64, lr 1e-4, LoRA r16, live CUB/A-OKVQA eval every 50 steps):

| run | start | training data | epochs |
|---|---|---|---|
| `base7b_aokvqa` | plain Qwen2.5-VL-7B | A-OKVQA | 3 |
| `mentor_aokvqa` | Logan's FGVC-SFT 7B | A-OKVQA | 3 |
| `mentor_aokvqa_cars` | Logan's FGVC-SFT 7B | A-OKVQA + Cars train (8,144 images, 50/50) | 2 |
| `mentor_mixed` (Unity, reference) | Logan's FGVC-SFT 7B | A-OKVQA + iNat (≈1.5k bird species) | 5 |

3 × 17k ≈ 2 × 25k examples, so the runs see about the same number of examples.

```bash
bash experiments/07_fgvc_transfer/run.sh                                     # A-OKVQA + Cars
GPU_IDS="0 2" RUN=mentor_aokvqa DATA=configs/aokvqa.yaml EPOCHS=3 \
  BASELINE=mentor_aokvqa_cars bash experiments/07_fgvc_transfer/run.sh
GPU_IDS="0 1 2 3" RUN=base7b_aokvqa MODEL=Qwen/Qwen2.5-VL-7B-Instruct BASELINE=mentor_aokvqa \
  DATA=configs/aokvqa.yaml EPOCHS=3 bash experiments/07_fgvc_transfer/run.sh
```
Each run ends with the full zero-shot suite (`vast/run_zeroshot.sh`) and a paired test against
`BASELINE`: `runs/<run>/paired_vs_<BASELINE>.md`.

## Results (top-1)

| set | plain 7B + A-OKVQA | Logan + A-OKVQA | Logan + A-OKVQA + Cars | Logan + A-OKVQA + iNat |
|---|---|---|---|---|
| CUB En / La | .599 / .307 | .605 / .280 | .635 / .324 | .629 / .417 |
| CUB unseen-20 En / La | .442 / .272 | .450 / .189 | .488 / .293 | .535 / .350 |
| Cars | .779 | .767 | **.937** | .792 |
| Aircraft | .295 | .308 | .351 | .315 |
| Dogs | .661 | .641 | .689 | .651 |
| Flowers | .773 | .790 | .817 | .767 |
| Food | .844 | .836 | .861 | .840 |
| Pets | .878 | .871 | .887 | .881 |
| iNat | .118 | .113 | .122 | .183 |
| A-OKVQA / MMMU / MMBench | .855 / .419 / .773 | .846 / .443 / .792 | .848 / .427 / .804 | .843 / .420 / .790 |

Logan's checkpoint *before* contrastive training scores .005 on CUB as an embedder.

**Takeaways.**
1. **The contrastive stage does the work.** Training on A-OKVQA alone, with no bird images, takes
   CUB from chance to .60.
2. **Logan's FGVC-SFT stage adds nothing to FGVC embeddings.** Plain Qwen2.5-VL-7B under the same
   recipe matches it on CUB English (.599 vs .605, p = .16) and is *better* on Latin (+2.8, p < 1e-8;
   unseen-20 Latin +8.3). Elsewhere the differences are small and mixed: plain is better on Dogs (+1.9)
   and Cars (+1.2), Logan's on Flowers (+1.7). What the SFT stage does buy is MMBench (+1.9, p = 1e-5).
3. **A second fine-grained domain helps everywhere.** Adding Cars beats A-OKVQA-only on every FGVC
   set (CUB +3.0, CUB Latin +4.4, unseen-20 Latin +10.4, Aircraft +4.4, Dogs +4.8), with VQA unchanged.
   Fine-grained skill partly carries across domains.
