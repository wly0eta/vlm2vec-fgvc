"""The training step: GradCache + InfoNCE, on one GPU or many."""
import time
from collections import defaultdict
from dataclasses import dataclass

import torch
from transformers import Trainer

from . import distributed as dist_utils
from .checkpoint import EmbeddingContract, load_adapter_weights, save_adapter
from .gradcache import encode_without_grad, replay_with_grad
from .losses import contrastive_loss

SIDES = ("query", "positive", "negative")


@dataclass
class StepResult:
    loss: torch.Tensor
    inbatch_accuracy: float
    num_candidates: int


def contrastive_step(encode, chunks: dict, positive_keys: list[str], negative_keys: list[str], *,
                     temperature: float, device: torch.device, mask_false_negatives: bool = True,
                     global_loss: bool = False) -> StepResult:
    """One GradCache contrastive update's forward + backward (no optimizer step).

    1. embed every chunk of queries / positives / negatives with autograd off,
       snapshotting the RNG per chunk;
    2. with `global_loss`, gather every GPU's embeddings and labels so each query is
       scored against the global batch's candidates;
    3. compute InfoNCE (optionally masking same-label false negatives) on detached
       copies and backprop it to get d(loss)/d(embedding);
    4. re-embed each chunk with autograd on, under its RNG snapshot, and backprop that
       gradient into the weights -- one chunk in memory at a time.

    Leaves gradients in `.grad`, local to this process; the caller combines them across
    GPUs (`distributed.all_reduce_gradients`). Kept free of Trainer state so it can be
    tested directly, including multi-process.
    """
    embeddings, rngs = {}, {}
    for side in SIDES:
        if chunks.get(side):
            embeddings[side], rngs[side] = encode_without_grad(encode, chunks[side], device)
    leaves = {side: e.detach().requires_grad_() for side, e in embeddings.items()}

    query, positive = leaves["query"], leaves["positive"]
    # Always a tensor (possibly 0 rows) so every rank joins the same collectives.
    negative = leaves.get("negative", query.new_zeros((0, query.shape[1])))
    if global_loss:
        query = dist_utils.all_gather_with_grad(query)
        positive = dist_utils.all_gather_with_grad(positive)
        negative = dist_utils.all_gather_with_grad(negative)
        positive_keys = [k for part in dist_utils.all_gather_objects(positive_keys) for k in part]
        negative_keys = [k for part in dist_utils.all_gather_objects(negative_keys) for k in part]
    labels = (positive_keys, negative_keys) if mask_false_negatives else (None, None)
    loss, accuracy = contrastive_loss(query, positive, negative, temperature, *labels)
    loss.backward()

    for side, leaf in leaves.items():
        replay_with_grad(encode, chunks[side], rngs[side], leaf.grad)
    return StepResult(loss.detach(), float(accuracy), positive.shape[0] + negative.shape[0])


def count_params(backbone) -> tuple[int, int]:
    """(language-model params, vision-encoder params) that run per token / per image patch.
    Embedding lookups and the LM head (never run: we pool hidden states) cost no matmul FLOPs."""
    vision = language = 0
    for name, p in backbone.named_parameters():
        if "visual" in name:
            vision += p.numel()
        elif "embed_tokens" not in name and "lm_head" not in name:
            language += p.numel()
    return language, vision


def count_work(chunks: dict) -> tuple[int, int]:
    """(text+image tokens through the LM, patches through the vision encoder) in one step's chunks."""
    tokens = patches = 0
    for side in SIDES:
        for c in chunks.get(side, []):
            tokens += int(c["attention_mask"].sum())
            if "image_grid_thw" in c:
                patches += int(c["image_grid_thw"].prod(dim=-1).sum())
    return tokens, patches


class VLM2VecTrainer(Trainer):
    """HF Trainer whose step is `contrastive_step` followed by one gradient all-reduce.

    Everything else -- optimizer, LR schedule, gradient clipping, logging, checkpoint
    rotation, resume -- is stock HF Trainer.
    """

    def __init__(self, *args, contract: EmbeddingContract, processor, **kwargs):
        super().__init__(*args, **kwargs)
        self.contract = contract
        self.processor = processor
        self._metric_sums = defaultdict(float)
        self._metric_steps = 0
        self._params = None                 # (language, vision), counted on the first step
        self._work = [0, 0]                 # tokens, patches since the last log line
        self._last_log = time.perf_counter()

    def training_step(self, model, inputs, num_items_in_batch=None):
        # Bypass the DDP wrapper: it would all-reduce on every one of the step's many
        # backward passes. We sync gradients once ourselves (see distributed.py).
        module = self.accelerator.unwrap_model(model)
        module.train()
        device = self.args.device
        chunks = {side: [c.to(device) for c in inputs[side]] for side in SIDES}
        if not chunks["query"]:
            raise RuntimeError("batch has no readable rows (every image failed to decode)")
        global_loss = self.args.cross_device_negatives and dist_utils.world_size() > 1
        tokens, patches = count_work(chunks)
        self._work[0] += tokens
        self._work[1] += patches

        result = contrastive_step(module.encode, chunks, inputs["positive_keys"], inputs["negative_keys"],
                                  temperature=module.temperature, device=device,
                                  mask_false_negatives=self.args.mask_false_negatives,
                                  global_loss=global_loss)
        # Global loss: each rank holds d(global loss)/d(its own rows) -> the sum is exact.
        # Local loss: ranks hold gradients of different losses -> average them.
        dist_utils.all_reduce_gradients(module.parameters(), average=not global_loss)

        self._metric_sums["inbatch_acc"] += result.inbatch_accuracy
        self._metric_sums["candidates"] += result.num_candidates
        self._metric_steps += 1
        return result.loss

    def log(self, logs: dict, start_time=None) -> None:
        """Attach step-averaged contrastive metrics to HF's regular training log lines."""
        if "loss" in logs and self._metric_steps:
            for name, total in self._metric_sums.items():
                logs[name] = round(total / self._metric_steps, 4)
            self._metric_sums.clear()
            self._metric_steps = 0
            self._log_throughput(logs)
            if torch.cuda.is_available():
                # peak since the last log line, this rank: headroom for gc_chunk_size / no checkpointing
                logs["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
                logs["reserved_mem_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 1)
                torch.cuda.reset_peak_memory_stats()
        super().log(logs, start_time)

    def _log_throughput(self, logs: dict) -> None:
        """Achieved TFLOP/s per GPU since the last log line (wall clock: data loading, live evals
        and saves count against it). model_tflops uses the standard MFU convention, 6 x params x
        tokens (forward + backward); hw_tflops adds what this recipe really runs on top: GradCache's
        extra no-grad forward and, with gradient checkpointing, the recomputed forward."""
        if self._params is None:
            self._params = count_params(self.accelerator.unwrap_model(self.model).backbone)
        now = time.perf_counter()
        seconds, self._last_log = now - self._last_log, now
        (language, vision), (tokens, patches) = self._params, self._work
        self._work = [0, 0]
        forward = 2 * (language * tokens + vision * patches)          # FLOPs of one forward pass
        passes = 4 + (1 if self.args.gradient_checkpointing else 0)   # vs 3 for plain fwd + bwd
        logs["model_tflops"] = round(3 * forward / seconds / 1e12, 1)
        logs["hw_tflops"] = round(passes * forward / seconds / 1e12, 1)
        logs["tokens_per_s"] = round(tokens / seconds)

    # --- checkpoints: adapter + contract only (the base model is never re-saved) ---------------

    def _save(self, output_dir=None, state_dict=None):
        save_adapter(self.model.backbone, self.processor, self.contract, output_dir or self.args.output_dir)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        module = self.accelerator.unwrap_model(model or self.model)
        load_adapter_weights(module.backbone, resume_from_checkpoint)
