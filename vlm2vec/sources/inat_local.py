"""iNaturalist 2021 from the official COCO-style JSON on local disk
(e.g. /datasets/ai/inaturalist/{train,train_mini,val}.json).

Images are referenced where they already are -- nothing is copied -- which is what makes
building the 2.7M-image train set take minutes instead of a day of downloading.
"""
import json
import os
import random
import re
from collections.abc import Iterator

from .common import Record


def normalize_common_name(name: str) -> str:
    """'Brewer's Blackbird' / 'Brewer Blackbird' / 'Black-footed Albatross' -> comparable keys."""
    name = name.lower().replace("’", "'")
    name = re.sub(r"'s\b", "", name).replace("'", "")
    return re.sub(r"\s+", " ", re.sub(r"[-_]", " ", name)).strip()


def common_name_index(annotations_path: str) -> dict[str, str]:
    """normalized English common name -> scientific name, from iNat's category list."""
    with open(annotations_path) as f:
        categories = json.load(f)["categories"]
    index = {}
    for c in categories:
        if c.get("common_name"):
            index.setdefault(normalize_common_name(c["common_name"]), c["name"])
    return index


def records(root: str, annotations: str, seed: int = 0) -> Iterator[Record]:
    """Yield one Record per annotated image, in a seeded random order.

    Shuffling up front means a per-class cap keeps a random sample of each species
    rather than the first N files of it.
    """
    path = annotations if os.path.isabs(annotations) else os.path.join(root, annotations)
    with open(path) as f:
        data = json.load(f)
    categories = {}
    for c in data["categories"]:
        name = c.get("name") or f"{c.get('genus', '')} {c.get('specific_epithet', '')}".strip()
        categories[c["id"]] = (name, {k: c.get(k) for k in ("class", "order", "family", "genus") if c.get(k)})
    files = {im["id"]: im["file_name"] for im in data["images"]}
    annotations_list = data["annotations"]
    random.Random(seed).shuffle(annotations_list)
    for a in annotations_list:
        file_name = files.get(a["image_id"])
        name, taxonomy = categories.get(a["category_id"], (None, {}))
        if file_name and name:
            yield Record(uid=str(a["image_id"]), image=os.path.join(root, file_name), label=name,
                         taxonomy=taxonomy)
