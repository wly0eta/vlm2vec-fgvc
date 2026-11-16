# Design notes

How this codebase turns a generative vision-language model into an embedding model, why
each piece is built the way it is, and which problems in the earlier version it fixes.

## 1. What the system does

A vision-language model (Qwen-VL family) reads an image plus text and normally generates
text. We use it as an **encoder**: feed it an input, take one hidden-state vector, and train
it so that a query (an image plus an instruction or question) lands near its correct answer
(a species name, the right option) and far from wrong ones. That is **VLM2Vec**
(Jiang et al., ICLR 2025): contrastive fine-tuning with LoRA.

```
raw dataset ──build_dataset.py──▶ JSONL rows ──MixedDataset──▶ ContrastiveCollator (workers)
   (query image+text, positive, negatives)                          │ tokenized chunks
                                                                    ▼
             VLM2VecTrainer.training_step ──▶ contrastive_step: GradCache + InfoNCE
                                                                    │ adapter + contract
                                                                    ▼
                      evaluate.py ──▶ eval_results.json ──tools/collect_results.py──▶ tables
```

## 2. Module map

Each module owns one concern; dependencies point one way (config ← checkpoint ← modeling,
data; losses/gradcache/distributed are leaf utilities; trainer composes them).

| Module | Responsibility |
|---|---|
| `vlm2vec/config.py` | Every knob as a dataclass; parsed from the CLI (`--help` documents each). |
| `vlm2vec/modeling.py` | Load backbone (merging any adapter chain), add LoRA, `encode()` = decoder → last-token pool → L2-normalize. |
| `vlm2vec/checkpoint.py` | Save/load adapters with verification; the embedding contract; finding checkpoints. |
| `vlm2vec/data.py` | Row format, lazy JSONL access, the epoch-based task mixture, the collator. |
| `vlm2vec/losses.py` | InfoNCE with same-label masking. |
| `vlm2vec/gradcache.py` | Two-pass GradCache with RNG replay. |
| `vlm2vec/distributed.py` | The collectives multi-GPU needs; no-ops on one process. |
| `vlm2vec/trainer.py` | `contrastive_step` (pure, testable) + the HF `Trainer` subclass around it. |
| `vlm2vec/callbacks.py` | Timed checkpoints, stop-for-requeue, live metrics. |
| `vlm2vec/evaluation.py` | Multiple choice, classification, retrieval — tie-aware. |
| `vlm2vec/sources/` | Raw datasets → `Record`s → JSONL (`common.build`). |
| `train.py`, `evaluate.py`, `build_dataset.py`, `tools/collect_results.py` | Thin CLIs. |
| `slurm/` | Cluster scripts: environment, prep, train (requeue lifecycle), eval, the experiment grid. |

## 3. Design decisions

### Model

**LoRA rather than full fine-tuning.** ~0.3–0.8% of parameters train, so a 7B model fits on
one 48 GB GPU, checkpoints are megabytes, and — the reason that matters for this project —
the frozen base keeps its general ability. The research question is how much VQA ability
survives specialization; full fine-tuning would overwrite more of it.

**Last-token pooling.** In a causal model only the last position has attended to the whole
input, so its hidden state is the natural summary; it is also what VLM2Vec uses. The pooling
locates the last *real* token from the attention mask, so it is correct under left or right
padding. (Qwen's tokenizer defaults to left; the old code assumed right and relied on one
attribute being set. Both are now enforced and the pooling no longer depends on it.)

**Skip the LM head.** `encode()` calls the model's inner decoder (`.model`) directly. An
embedding model never needs vocabulary logits; the old code computed them (then trimmed
them with a version-dependent `logits_to_keep` probe) and kept every layer's hidden states.

**bf16 weights, float32 similarities.** Embeddings are cast to float32 before
normalization and dot products. At temperature 0.02 a logit is cosine × 50; bf16 has ~3
significant digits, which would put noise of order 0.1 on every logit, and in evaluation
makes exact ties (see §5) far more common.

### Loss

**InfoNCE over in-batch and explicit negatives.** Candidates are all rows' positives plus
each row's sampled distractors; query *i*'s target is column *i*. With batch 32 and 3
distractors per row, each query sees 127 negatives per GPU.

**Same-label (false-negative) masking.** If two rows share a label — two photos of one
species, two questions answered "yes" — row *j*'s positive is a correct answer for query *i*,
yet InfoNCE pushes them apart. Identical label text also means identical embeddings, so the
loss demanded something impossible. Those logits are set to −∞. Distinct-but-similar labels
stay as negatives: separating confusable species is the point of FGVC. Masking works across
GPUs (labels are gathered with the embeddings).

**Fixed temperature (0.02).** As in VLM2Vec; not changed so results stay comparable.

### Memory and throughput

**GradCache.** More negatives per query improve contrastive training, but a 32-row
multimodal batch's activations do not fit beside a 7B model. GradCache embeds the batch
chunk by chunk without autograd, computes the loss on those (tiny) embeddings, then replays
each chunk with autograd and backprops `embedding · ∂loss/∂embedding`. Peak memory is one
chunk; the gradient equals the full-batch gradient (tested).

**RNG replay in GradCache.** The replay must reproduce the cached forward exactly. LoRA
dropout draws new masks on every call, so each chunk's RNG state is snapshotted before the
cached pass and restored for its replay. The previous trainer did not do this, so its
gradients were computed from embeddings with different dropout masks than the ones the loss
was evaluated on (`tests/test_gradcache.py` reproduces the discrepancy).

**Tokenization in dataloader workers.** Image decoding, resizing, and tokenization moved from
the training step (main process, done twice per step) to the collator, which runs in
`dataloader_num_workers` background processes and overlaps with GPU compute.

**Lazy JSONL.** The 2.7M-row iNat file used to be parsed into Python dicts up front in every
process — several GB, times every worker and every GPU. `JsonlRows` keeps one int64 byte
offset per row (~22 MB) and reads rows on demand; the index is cached next to the file.

### Data

**Real epochs.** The old dataset ignored the requested index and drew a random row (with
replacement) from a randomly chosen task. "One epoch" therefore covered only ~63% of distinct
rows, and under multi-GPU every rank would have drawn the same rows. `MixedDataset` is an
ordinary indexable list built per epoch: each task gets its weighted share, drawn without
replacement; the trainer's sampler shuffles it and gives each GPU a disjoint shard.

**Weights are shares of an epoch.** `mixed.yaml` with weights 1:1 gives each task half of the
epoch regardless of size (A-OKVQA 17k is oversampled, iNat-37k undersampled), which matches
what the old sampler did on average. The draw is redone each epoch, deterministically.

**Never truncate multimodal inputs.** Truncation cuts from the end: it can remove image tokens
(corrupting the sample) or change which token is "last", i.e. what gets pooled. Image inputs
are kept whole and over-length ones logged; only text-only inputs are truncated.

**Dataset building.** One `Record` type, one `build()` pipeline. Images from local iNat are
referenced in place (no copying 2.7M files). Negative sampling is O(k) per row by rejection
sampling (the old O(labels) per row never finished on 2.7M × 10k). Optional taxonomy-aware
hard negatives (`--hard-negative-fraction`) draw confusable species from the same genus or
family; the default stays uniform so new runs remain comparable with existing ones. Files are
written atomically (temp + rename), so an interrupted build never leaves a partial file that
the skip-if-exists logic would mistake for a finished one. CUB-style labels are cleaned
(`001.Black_footed_Albatross` → `Black footed Albatross`).

### Multi-GPU

**One gradient sync per step, done by us.** DDP all-reduces inside `backward()`, but a
GradCache step runs one backward per chunk. Letting DDP sync would mean many all-reduces per
step, or `no_sync` tricks that break on parameters a text-only chunk never touches (the vision
tower). The trainer calls the unwrapped module, accumulates local gradients for the whole step,
and calls `all_reduce_gradients` once. An end-of-run checksum verifies the replicas never drifted.

**Cross-device negatives.** Embeddings and labels are gathered from all GPUs, so each query is
scored against the *global* batch: 4 GPUs × 32 rows → 511 negatives per query. Each rank then
holds the gradient of the same global loss with respect to its own rows, so gradients are
**summed** (exact). Without cross-device negatives each rank has its own loss and gradients are
**averaged**. A two-process test checks the multi-GPU gradient equals the single-process one.

**Learning rate is not auto-scaled** with the global batch; set `--learning_rate` if you change
the GPU count substantially.

### Checkpointing on a preemptible cluster

**Time-based saves** (`--save_every_minutes 30`) replace hand-tuned `save_steps`. The old
setting broke in practice: a 7B run at ~17 s/step reached the 8-hour limit before its first
2000-step save, never checkpointed, and restarted from zero on every requeue.

**Save-and-requeue at the wall.** Five minutes before the limit SLURM signals the job script,
which creates `<run>/STOP`. Training saves at the end of the current step and exits; the script
requeues the job; the next chunk resumes from exactly that step. The script decides from a
`TRAINING_COMPLETE` marker, not an exit code (torchrun reports every failure as 1). A crash is
not requeued (it would crash again). A watchdog requeues anyway if saving overruns, because a
job killed as `TIMEOUT` is not requeued by SLURM.

**Rank-0 decisions are broadcast.** "Save now" and "stop now" are decided on rank 0 and
broadcast, so all GPUs enter checkpointing together; independent per-rank clocks could
deadlock one rank in a collective the others never reach.

### Checkpoints and handoff

**Verified adapter loading.** PEFT 0.20 silently skipped adapter tensors whose keys didn't
match, leaving LoRA at its zero initialization — the "trained" model was the base model, with
no error. Every load now checks for unmatched tensors and for all-zero LoRA-B matrices.

**The embedding contract** (`vlm2vec_config.json`, saved with every adapter) records what the
weights don't: the true base model, image budget, pooling, prompt template. Two failures it
prevents: (1) `adapter_config.json` names only the HF model the merged backbone came from, so a
run trained on the mentor's FGVC checkpoint looked like plain Qwen2.5-VL-7B — evaluating it
that way silently drops the mentor's training; (2) evaluation used the default image budget
even when training used a smaller one. `evaluate.py` reads the contract; old checkpoints
without one need `--base-model`.

**Adapter chains.** `--model_name` may be an adapter directory, including one of our own runs:
its base is loaded (recursively, following contracts), the adapter merged in, and a fresh LoRA
trained on top.

## 4. Evaluation methodology

Every metric is "embed the query, embed the candidates, take the most similar":

| Metric | Candidates | Reported |
|---|---|---|
| `--mc` multiple choice | the row's own options | accuracy, chance |
| `--classify` classification | every label in the file (or `--class-pool`) | top-1, top-5, chance, coverage |
| `--retrieval` | all rows' positives, deduplicated | recall@1/5, MRR |

Safeguards, each of which changed a number in the old code:

- **Ties score 1/k.** `argmax` returns the first maximum, and the old multiple-choice eval put
  the correct option first, so every tie counted as a win — a collapsed model that embeds
  everything identically scored 100%. Tie-aware scoring makes the result independent of option
  order and gives such a model chance accuracy.
- **Duplicate options** equal to the answer are dropped (they cannot be told apart from it).
- **Retrieval deduplicates candidates.** Two images of one species both retrieving its label are
  both correct; the old metric counted the second as a miss.
- **Classification reports coverage**: rows whose label is outside the label space count as
  wrong, and the fraction of such rows is reported rather than hidden in a low score.
- **Live eval uses a seeded random subset**, and classification keeps the file's full label
  set. The old version took the first N rows — often a handful of classes, since files are
  sorted by class — and scored them against only those classes, inflating the curve.
- **Unreadable images** are skipped and counted instead of crashing the run.

Protocol differences to state when reporting:

- **MMBench**: official scoring is CircularEval (correct under every rotation of the options);
  we score each question once. With tie-aware scoring option order cannot matter, but the
  number is per-question accuracy — comparable across our models, not to the leaderboard.
- **MMMU**: multiple-choice questions only, first image only.
- **CUB-200 vs iNaturalist labels**: CUB uses English common names, iNat (our training data)
  Latin binomials. A model trained on iNat is matching unseen vocabulary on CUB, so CUB
  understates FGVC ability; `inat_val.jsonl` is the matched-vocabulary benchmark.
- These are embedding-similarity scores, not the generative protocols used by public
  leaderboards.

## 5. Problems found in the previous version

| # | Problem | Effect | Fix | Test |
|---|---|---|---|---|
| 1 | GradCache replay used different dropout masks than the cached pass | wrong gradients every step | RNG snapshot/restore per chunk | `test_gradcache` |
| 2 | Same-label candidates treated as negatives | loss pushes correct matches apart | −∞ masking, across GPUs | `test_losses` |
| 3 | MC eval: answer always first + first-max argmax | ties counted as wins; degenerate model = 100% | tie-aware scoring | `test_evaluation` |
| 4 | Retrieval: identical labels scored as misses | understated recall | deduplicated candidates | `test_evaluation` |
| 5 | Live eval on first N rows with a shrunken label set | inflated, unrepresentative curves | seeded subset, full label set | — |
| 6 | Eval rebuilt LoRA from defaults and loaded weights unverified | silent no-op / mismatch possible | load via checkpoint's own config + verification | `test_model` |
| 7 | Eval ignored the training image budget | train/eval resolution mismatch | contract file | `test_model` |
| 8 | Sampling with replacement, same RNG on every rank | "1 epoch" ≈ 63% coverage; duplicate batches across GPUs | epoch-based mixture + distributed sampler | `test_data` |
| 9 | Truncation applied to multimodal inputs | could cut image tokens or move the pooled token | never truncate image inputs | — |
| 10 | Similarities in bf16 at temperature 0.02 | noisy logits; more ties | float32 | `test_losses` |
| 11 | All runs wrote `./plots` | parallel jobs overwrote each other's metrics | `<run>/plots` | — |
| 12 | `save_steps` exceeded steps per 8 h on slow runs | never checkpointed, restarted from 0 | time-based saves + stop file | lifecycle test |
| 13 | Requeue on wall relied on an exit code | torchrun reports 1 for everything | completion marker + watchdog | lifecycle test |
| 14 | O(rows × labels) negative sampling | prep never finished at 2.7M | O(k) rejection sampling | `test_sources` |
| 15 | Partial dataset files looked complete | silently truncated training sets | atomic writes | — |
| 16 | Continuing from one of our own runs merged onto the wrong base | mentor stage silently dropped | contract-based base resolution | `test_model` |
| 17 | Every row parsed into memory per process | GBs × workers × GPUs | offset-indexed JSONL | `test_data` |

## 6. Testing

```
pytest                                                    # pure-torch tests, ~10 s
python tests/make_tiny_model.py /tmp/tiny-qwen2vl         # once (downloads the processor)
VLM2VEC_TINY_MODEL=/tmp/tiny-qwen2vl pytest               # + real-model tests
```

Covered: loss and masking maths, GradCache equivalence (with dropout), two-process gradient
equivalence, data mixture semantics, metric behaviour, dataset building, padding-invariant
pooling, checkpoint round-trip and no-op detection, adapter chaining. Verified by running (not
in pytest): single- and two-process training through the HF Trainer, live eval, evaluation,
results collection, and the SLURM stop→requeue→resume lifecycle with a real `USR1` signal.
Not verifiable without the cluster: NCCL on real GPUs, multi-node launches.
