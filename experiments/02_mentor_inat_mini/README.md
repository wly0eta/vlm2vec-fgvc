# 02 · Mentor 7B on iNat train_mini

**Question.** Continue Logan's FGVC-SFT 7B + iNat contrastive run (Unity `mentor_inat`, stopped at
step 3,210) on iNat-2021 train_mini, with the same recipe, and compare per-step speed with Unity.

* Start: `runs/mentor_inat/checkpoint-3210`. Its adapter is merged and a fresh LoRA trained on top.
* Recipe: global batch 256, lr 1e-4, GradCache chunk 8, LoRA r16, 256 vision tokens, 1 epoch = 1,953 steps.
* Speed comparison: `python tools/compare_speed.py`.

```bash
bash experiments/02_mentor_inat_mini/run.sh      # tmux session 'mentor_inat_mini'
```

**Status.** Not run on Vast yet. It needs the iNat train_mini images (`vast/get_inat.sh`), which are
too big for the current node's 32 GB disk.
