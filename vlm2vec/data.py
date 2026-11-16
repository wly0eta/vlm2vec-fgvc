"""Training/eval data: JSONL rows -> tokenized, chunked model inputs.

Row format (one JSON object per line; written by build_dataset.py):

    {"query":       {"text": str, "image": path | null},
     "instruction": str,
     "positive":    {"text": str, "image": path | null},
     "negatives":   [{"text": str, "image": path | null}, ...]}

The query is what gets embedded on one side (an image + question); the positive is the
correct answer/label; negatives are distractors. All three are encoded by the same model.
"""
import json
import logging
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import yaml
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from transformers import AutoProcessor, TrainerCallback

from .checkpoint import hf_model_of
from .config import DEFAULT_INSTRUCTION, QUERY_TEMPLATE

ImageFile.LOAD_TRUNCATED_IMAGES = True   # a few dataset JPEGs are truncated; decode what's there
log = logging.getLogger(__name__)


# --- one side of a pair -------------------------------------------------------------------

@dataclass(frozen=True)
class Item:
    """Something to embed. Frozen (hashable), so identical items can be embedded once."""

    text: str
    image: str | None = None
    instruction: str | None = None   # queries only; None marks a candidate

    @property
    def key(self) -> str:
        """Identity used for false-negative masking: equal keys -> identical embeddings."""
        return self.text if self.image is None else f"{self.text}\x00{self.image}"

    def prompt(self) -> str:
        if self.instruction is None:
            return self.text
        return QUERY_TEMPLATE.format(instruction=self.instruction or DEFAULT_INSTRUCTION,
                                     text=self.text or "").rstrip()


def query_item(row: dict) -> Item:
    return Item(row["query"].get("text") or "", row["query"].get("image"), row.get("instruction") or "")


def positive_item(row: dict) -> Item:
    return Item(row["positive"].get("text") or "", row["positive"].get("image"))


def negative_items(row: dict) -> list[Item]:
    return [Item(n.get("text") or "", n.get("image")) for n in row.get("negatives", [])]


def mask_key(entry: dict, item: Item) -> str:
    """What false-negative masking compares: the entry's "class" if the row file gives one, else
    the item's own text/image. A "class" lets several names of one class count as the same label --
    with mixed English/Latin labels, "American Crow" and "Corvus brachyrhynchos" in one batch must
    not be each other's negatives."""
    return f"class\x00{entry['class']}" if entry.get("class") else item.key


# --- turning items into model inputs -------------------------------------------------------

def load_processor(model_name: str, max_pixels: int):
    """The processor of the underlying HF model, with our image budget and right padding.

    Right padding is also requested on every call (see build_inputs): Qwen's tokenizer
    defaults to left padding, and pooling used to rely on this one attribute being set.
    """
    processor = AutoProcessor.from_pretrained(hf_model_of(model_name), max_pixels=max_pixels)
    processor.tokenizer.padding_side = "right"
    return processor


def image_tokens_for(processor, width: int = 1024, height: int = 1024) -> int:
    """Vision tokens one image of this size costs under the processor's budget (for logs)."""
    image_processor = processor.image_processor
    out = image_processor(images=[Image.new("RGB", (width, height))], return_tensors="pt")
    merge = getattr(image_processor, "merge_size", 1)
    return int(out["image_grid_thw"][0].prod()) // (merge * merge)


def load_image(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


_warned_long = False


def build_inputs(processor, items: Sequence[Item], max_len: int, images: dict | None = None):
    """Tokenize items (and their images) into one padded batch.

    `images` optionally maps path -> already-decoded PIL image (the collator pre-loads them
    so a corrupt file can be dropped instead of crashing the step). Batches containing an
    image are never truncated: cutting a sequence would cut image tokens (corrupting the
    sample) or move the last token that pooling reads.
    """
    global _warned_long
    texts, pil_images = [], []
    for item in items:
        content = ([{"type": "image"}] if item.image else []) + [{"type": "text", "text": item.prompt()}]
        texts.append(processor.apply_chat_template([{"role": "user", "content": content}],
                                                   tokenize=False, add_generation_prompt=False))
        if item.image:
            pil_images.append(images[item.image] if images and item.image in images else load_image(item.image))
    has_images = bool(pil_images)
    batch = processor(text=texts, images=pil_images or None, padding=True, padding_side="right",
                      truncation=not has_images, max_length=None if has_images else max_len,
                      return_tensors="pt")
    if has_images and batch["input_ids"].shape[1] > max_len and not _warned_long:
        log.warning("multimodal input of %d tokens exceeds max_len=%d; kept whole (not truncated)",
                    batch["input_ids"].shape[1], max_len)
        _warned_long = True
    return batch


# --- random access into a big JSONL without loading it ------------------------------------------

class JsonlRows(Sequence):
    """Rows of a JSONL file, read on demand via a byte-offset index.

    The old loader parsed every row into a Python dict up front: for the 2.7M-row iNat
    file that is several GB per process, multiplied by every dataloader worker and every
    GPU. Here each process holds only an int64 offset per row (~22 MB for 2.7M) and the
    OS page cache does the rest. The index is cached next to the file when writable.
    """

    def __init__(self, path: str):
        self.path = path
        self.offsets = self._load_or_build_index()
        self._file = None   # opened lazily: file handles don't survive pickling into workers

    def _load_or_build_index(self) -> np.ndarray:
        stat = os.stat(self.path)
        cache = f"{self.path}.offsets.npz"
        if os.path.exists(cache):
            saved = np.load(cache)
            if int(saved["size"]) == stat.st_size and int(saved["mtime_ns"]) == stat.st_mtime_ns:
                return saved["offsets"]
        offsets, position = [], 0
        with open(self.path, "rb") as f:
            for line in f:
                if line.strip():
                    offsets.append(position)
                position += len(line)
        offsets = np.asarray(offsets, dtype=np.int64)
        try:
            tmp = f"{cache}.{os.getpid()}.tmp.npz"
            np.savez(tmp, offsets=offsets, size=stat.st_size, mtime_ns=stat.st_mtime_ns)
            os.replace(tmp, cache)   # atomic: concurrent ranks can't leave a half-written cache
        except OSError:
            pass                      # read-only data dir: just rebuild next time
        return offsets

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, i: int) -> dict:
        if self._file is None:
            self._file = open(self.path, "rb")
        self._file.seek(int(self.offsets[i]))
        return json.loads(self._file.readline())

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_file"] = None
        return state


# --- the task mixture --------------------------------------------------------------------------

class MixedDataset(Dataset):
    """A weighted mixture of JSONL tasks with real epoch semantics.

    Each task's YAML `weight` is its share of every epoch. An epoch has as many examples
    as all tasks combined; task i contributes round(share_i * total) of them, drawn
    WITHOUT replacement (cycling full permutations when a task is oversampled).

    This replaces a sampler that ignored the index and drew random rows WITH replacement:
    there, "one epoch" covered only ~63% of distinct rows, and because every GPU's copy
    used the same seed, multi-GPU runs would have trained on duplicate batches. Here the
    dataset is an ordinary indexable list, so the trainer's (distributed) sampler shuffles
    it and gives each GPU a disjoint shard; `set_epoch` redraws the mixture per epoch,
    deterministically, so it's identical on every rank and on resume.
    """

    def __init__(self, tasks: dict[str, tuple[Sequence, float]], seed: int = 0):
        if not tasks:
            raise ValueError("no tasks in the mixture")
        self.names = list(tasks)
        self.rows = [tasks[n][0] for n in self.names]
        weights = np.array([tasks[n][1] for n in self.names], dtype=float)
        if (weights <= 0).any():
            raise ValueError(f"task weights must be positive: {dict(zip(self.names, weights))}")
        if empty := [n for n, r in zip(self.names, self.rows) if len(r) == 0]:
            raise ValueError(f"tasks with no rows: {empty}")
        self.total = sum(len(r) for r in self.rows)
        counts = np.floor(weights / weights.sum() * self.total).astype(int)
        counts[np.argmax(weights)] += self.total - counts.sum()   # rounding remainder
        self.counts = counts
        self.seed = seed
        self.set_epoch(0)

    @classmethod
    def from_yaml(cls, path: str, seed: int = 0) -> "MixedDataset":
        with open(path) as f:
            spec = yaml.safe_load(f)
        tasks = {name: (JsonlRows(os.path.expanduser(cfg["data_path"])), float(cfg.get("weight", 1.0)))
                 for name, cfg in spec.items() if isinstance(cfg, dict) and "data_path" in cfg}
        return cls(tasks, seed)

    def set_epoch(self, epoch: int) -> None:
        index = []
        for task_id, (rows, count) in enumerate(zip(self.rows, self.counts)):
            rng = np.random.default_rng([self.seed, epoch, task_id])
            cycles, remainder = divmod(int(count), len(rows))
            picks = [rng.permutation(len(rows)) for _ in range(cycles)] + [rng.permutation(len(rows))[:remainder]]
            ids = np.concatenate(picks)
            index.append(np.stack([np.full_like(ids, task_id), ids], axis=1))
        self.index = np.concatenate(index)
        self.epoch = epoch

    def describe(self) -> str:
        return ", ".join(f"{n}: {c:,} of {len(r):,} rows" for n, c, r in zip(self.names, self.counts, self.rows))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        task_id, row_id = self.index[i]
        return self.rows[task_id][int(row_id)]


class ReseedMixtureCallback(TrainerCallback):
    """Redraw the mixture at the start of each epoch. (accelerate reseeds the sampler but
    does not call the dataset's set_epoch; dataloader workers are re-created per epoch
    unless persistent_workers is set, so they pick up the new draw.)"""

    def __init__(self, dataset: MixedDataset):
        self.dataset = dataset

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = math.floor(state.epoch + 1e-6)   # resumed mid-epoch: state.epoch is fractional
        if epoch != self.dataset.epoch:
            self.dataset.set_epoch(epoch)


# --- batching --------------------------------------------------------------------------------

class ContrastiveCollator:
    """Rows -> GradCache-ready, pre-tokenized chunks, built inside dataloader workers.

    Image decoding, resizing and tokenization used to happen in the training step itself,
    in the main process, twice per step (cached pass and replay). Doing it once here, in
    `dataloader_num_workers` background processes, overlaps it with GPU compute.

    A row whose image can't be decoded is dropped from the batch and logged, rather than
    crashing a multi-day job -- which, after a requeue, would crash on the same batch again.
    """

    def __init__(self, processor, chunk_size: int, max_len: int):
        self.processor = processor
        self.chunk_size = chunk_size
        self.max_len = max_len

    def __call__(self, rows: list[dict]) -> dict:
        images, kept = {}, []
        for row in rows:
            try:
                for item in [query_item(row), positive_item(row), *negative_items(row)]:
                    if item.image and item.image not in images:
                        images[item.image] = load_image(item.image)
                kept.append(row)
            except (OSError, ValueError) as err:
                log.warning("dropping row with unreadable image: %s", err)
        queries = [query_item(r) for r in kept]
        positives = [positive_item(r) for r in kept]
        negatives = [n for r in kept for n in negative_items(r)]
        return {
            "query": self._chunks(queries, images),
            "positive": self._chunks(positives, images),
            "negative": self._chunks(negatives, images),
            "positive_keys": [mask_key(r["positive"], p) for r, p in zip(kept, positives)],
            "negative_keys": [mask_key(entry, n) for r in kept
                              for entry, n in zip(r.get("negatives", []), negative_items(r))],
        }

    def _chunks(self, items: list[Item], images: dict) -> list:
        return [build_inputs(self.processor, items[i:i + self.chunk_size], self.max_len, images)
                for i in range(0, len(items), self.chunk_size)]
