"""Shared machinery for turning a raw dataset into training/eval JSONL.

A source module only has to yield `Record`s; `build()` does the rest: filtering,
per-class capping, saving images, negative sampling, splitting, writing rows.
"""
import json
import logging
import os
import random
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

CLASSIFY_INSTRUCTION = "Identify the species in the image."
MC_INSTRUCTION = "Answer the multiple-choice question about the image by choosing the correct option."


@dataclass
class Record:
    """One example from any source.

    image:    a file path (referenced in place) or a PIL image (saved under image_dir).
    label:    the correct answer / class name.
    choices:  multiple-choice sources only -- all options, including `label`.
    taxonomy: optional grouping used for filtering and hard negatives,
              e.g. {"class": "Aves", "family": "Parulidae", "genus": "Setophaga"}.
    """

    uid: str
    image: object
    label: str
    query_text: str = ""
    choices: list[str] | None = None
    taxonomy: dict = field(default_factory=dict)


def clean_label(label: str) -> str:
    """'001.Black_footed_Albatross' -> 'Black footed Albatross' (CUB-style class names).

    Leading index numbers and underscores are file-system artifacts; leaving them in makes
    the label text the model must match arbitrary instead of linguistic.
    """
    label = re.sub(r"^\d+[._\s-]*", "", label.strip())
    return re.sub(r"\s+", " ", label.replace("_", " ")).strip()


class NegativeSampler:
    """Draws k distinct negative labels for a positive, in O(k) per draw.

    The old converters rebuilt "all labels except the positive" for every row -- O(labels)
    per row, i.e. 2.7M x 10k operations for full iNaturalist, which never finished. Rejection
    sampling from the label list needs ~k draws when there are many labels.

    hard_fraction > 0 takes that share of negatives from the positive's taxonomic group
    (same genus, else same family): confusable species, the negatives that actually teach
    fine-grained distinctions. Default 0 keeps uniformly random negatives, matching how the
    existing checkpoints were trained.
    """

    def __init__(self, labels: Iterable[str], rng: random.Random, hard_fraction: float = 0.0,
                 groups: dict[str, list[str]] | None = None):
        self.labels = sorted(set(labels))
        self.rng = rng
        self.hard_fraction = hard_fraction
        self.groups = groups or {}   # label -> ordered list of group keys, most specific first
        self.members: dict[str, list[str]] = defaultdict(list)
        for label in self.labels:
            for key in self.groups.get(label, []):
                self.members[key].append(label)

    def sample(self, positive: str, k: int) -> list[str]:
        k = min(k, len(self.labels) - 1)
        chosen: list[str] = []
        n_hard = round(k * self.hard_fraction)
        for key in self.groups.get(positive, []):
            if len(chosen) >= n_hard:
                break
            pool = [l for l in self.members[key] if l != positive and l not in chosen]
            chosen += self.rng.sample(pool, min(n_hard - len(chosen), len(pool)))
        excluded = {positive, *chosen}
        while len(chosen) < k:
            candidate = self.labels[self.rng.randrange(len(self.labels))]
            if candidate not in excluded:
                chosen.append(candidate)
                excluded.add(candidate)
        return chosen


def stratified_split(records: list[Record], val_fraction: float, rng: random.Random):
    """Split per label so every label with >= 2 examples lands in both splits.

    A plain random split can put a rare label only in val; the model never trained on it,
    so those rows are guaranteed misses that look like model error. Singletons go to train.
    """
    by_label = defaultdict(list)
    for r in records:
        by_label[r.label].append(r)
    train, val = [], []
    for group in by_label.values():
        rng.shuffle(group)
        n_val = 0 if len(group) < 2 else min(max(round(len(group) * val_fraction), 1), len(group) - 1)
        val += group[:n_val]
        train += group[n_val:]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _save_image(record: Record, image_dir: str) -> str:
    if isinstance(record.image, str):
        return record.image                      # local dataset: reference in place, no copy
    path = os.path.join(image_dir, f"{record.uid}.jpg")
    if not os.path.exists(path):
        record.image.convert("RGB").save(path)
    return path


def _row(record: Record, image_path: str, instruction: str, negatives: list[str]) -> dict:
    return {
        "query": {"text": record.query_text, "image": image_path},
        "instruction": instruction,
        "positive": {"text": record.label},
        "negatives": [{"text": n} for n in negatives],
    }


def _sibling_path(path: str, tag: str) -> str:
    """configs/x_train.jsonl -> configs/x_val.jsonl; configs/x.jsonl -> configs/x_val.jsonl."""
    directory, name = os.path.split(path)
    if "train" in name:
        return os.path.join(directory, name.replace("train", tag))
    stem, ext = os.path.splitext(name)
    return os.path.join(directory, f"{stem}_{tag}{ext}")


def write_jsonl(path: str, rows: Iterable[dict]) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp, n = f"{path}.tmp", 0
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
            n += 1
    os.replace(tmp, path)   # never leave a half-written file that looks complete to resume logic
    return n


def relabel_rows(rows: list[dict], mapping, keep_original_names: bool, num_negatives: int,
                 rng: random.Random) -> tuple[list[dict], dict]:
    """Restrict a classification JSONL to the classes `mapping` covers, optionally renaming them.

    `mapping(label) -> new label or None`. Rows whose label maps to None are dropped. With
    keep_original_names the labels stay as they were -- that gives the *control* set: the same
    classes and images as the renamed set, so comparing the two isolates the effect of the label
    vocabulary alone. Negatives are re-sampled from the kept label set.
    """
    kept = [(row, mapping(row["positive"]["text"])) for row in rows]
    kept = [(row, new) for row, new in kept if new]
    labels = sorted({(row["positive"]["text"] if keep_original_names else new) for row, new in kept})
    sampler = NegativeSampler(labels, rng)
    out = []
    for row, new in kept:
        label = row["positive"]["text"] if keep_original_names else new
        out.append({**row, "positive": {"text": label},
                    "negatives": [{"text": n} for n in sampler.sample(label, num_negatives)]})
    classes_before = len({r["positive"]["text"] for r in rows})
    return out, {"rows": len(out), "rows_before": len(rows), "classes": len(labels), "classes_before": classes_before}


def build(records: Iterable[Record], *, out: str, image_dir: str = "images", limit: int = 0,
          taxon_class: str | None = None, per_class_cap: int = 0, num_negatives: int = 3,
          hard_fraction: float = 0.0, val_fraction: float = 0.0, val_out: str | None = None,
          seed: int = 0, instruction: str = CLASSIFY_INSTRUCTION) -> None:
    """Materialize records into JSONL row files.

    Filtering and capping happen while streaming, before an image is decoded or saved, so
    skipped examples cost nothing. Multiple-choice records keep their own options as
    negatives; classification records get `num_negatives` sampled labels.
    """
    rng = random.Random(seed)
    os.makedirs(image_dir, exist_ok=True)
    kept: list[tuple[Record, str]] = []
    per_label: dict[str, int] = defaultdict(int)
    for record in records:
        if limit and len(kept) >= limit:
            break
        if taxon_class and record.taxonomy.get("class") != taxon_class:
            continue
        if per_class_cap and per_label[record.label] >= per_class_cap:
            continue
        try:
            path = _save_image(record, image_dir)
        except (OSError, ValueError) as err:
            log.warning("skipping %s: %s", record.uid, err)
            continue
        record.image = None   # drop the decoded image; we keep only its path
        kept.append((record, path))
        per_label[record.label] += 1
    if not kept:
        raise RuntimeError("no records survived filtering")

    is_mc = kept[0][0].choices is not None
    sampler = None
    if not is_mc:
        groups = {r.label: [f"genus:{r.taxonomy['genus']}" if r.taxonomy.get("genus") else None,
                            f"family:{r.taxonomy['family']}" if r.taxonomy.get("family") else None]
                  for r, _ in kept}
        groups = {label: [g for g in keys if g] for label, keys in groups.items()}
        sampler = NegativeSampler(per_label.keys(), rng, hard_fraction, groups)

    def to_row(record: Record, path: str) -> dict:
        if is_mc:
            negatives = [c for c in dict.fromkeys(record.choices) if c != record.label]
            return _row(record, path, MC_INSTRUCTION, negatives)
        return _row(record, path, instruction, sampler.sample(record.label, num_negatives))

    if val_fraction > 0:
        train, val = stratified_split([r for r, _ in kept], val_fraction, rng)
        paths = {id(r): p for r, p in kept}
        n_train = write_jsonl(out, (to_row(r, paths[id(r)]) for r in train))
        n_val = write_jsonl(val_out or _sibling_path(out, "val"), (to_row(r, paths[id(r)]) for r in val))
        log.info("wrote %d train / %d val rows (%d labels)", n_train, n_val, len(per_label))
    else:
        n = write_jsonl(out, (to_row(r, p) for r, p in kept))
        log.info("wrote %d rows -> %s (%d labels)", n, out, len(per_label))
