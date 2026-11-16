"""Label-language eval pairs: the same images and classes, labelled in English vs Latin.

    python tools/make_lang_sets.py --inat-annotations /workspace/data/inaturalist/val.json

* iNat val (in-domain for every iNat-trained model, which only ever saw Latin names):
    configs/inat_val_common.jsonl         English common names
    configs/inat_val_latin_matched.jsonl  the same rows with their Latin names
  Species without an English name in iNat-2021, or whose English name is shared by several
  species, are dropped from both (so neither file has ambiguous labels).
* Flowers-102 (zero-shot domain transfer; CUB-164 is built the same way, build_dataset.py relabel):
    configs/zs_flowers_common.jsonl       original English names, classes iNat can name
    configs/zs_flowers_latin.jsonl        the same rows with iNat's scientific names
  Matched by normalized English name; every match is printed so it can be checked by eye.
"""
import argparse
import json
import random
from collections import Counter

import sys, pathlib  # repo root on the path: vlm2vec/ and evaluate.py live there
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from vlm2vec.data import JsonlRows
from vlm2vec.sources.common import relabel_rows, write_jsonl
from vlm2vec.sources.inat_local import common_name_index, normalize_common_name


def pair(rows: list[dict], mapping, original_name: str, renamed_name: str, seed: int) -> None:
    """Write the original-label and renamed-label versions of the rows `mapping` covers."""
    for keep, out in ((True, original_name), (False, renamed_name)):
        kept, stats = relabel_rows(rows, mapping, keep, 3, random.Random(seed))
        write_jsonl(out, kept)
        print(f"{out}: {stats['rows']}/{stats['rows_before']} rows, "
              f"{stats['classes']}/{stats['classes_before']} classes")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inat-annotations", required=True, help="iNat-2021 val.json (has common names)")
    ap.add_argument("--inat-val", default="configs/inat_val.jsonl")
    ap.add_argument("--flowers", default="configs/zs_flowers.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.inat_annotations) as f:
        categories = json.load(f)["categories"]
    named = [c for c in categories if c.get("common_name")]
    shared = Counter(normalize_common_name(c["common_name"]) for c in named)
    english_of = {c["name"]: c["common_name"] for c in named
                  if shared[normalize_common_name(c["common_name"])] == 1}
    print(f"iNat-2021: {len(english_of)}/{len(categories)} species have a unique English name")
    pair(list(JsonlRows(args.inat_val)), english_of.get,
         "configs/inat_val_latin_matched.jsonl", "configs/inat_val_common.jsonl", args.seed)

    latin_of = common_name_index(args.inat_annotations)
    flowers = list(JsonlRows(args.flowers))
    matches = {}
    for label in sorted({r["positive"]["text"] for r in flowers}):
        latin = latin_of.get(normalize_common_name(label))
        if latin:
            matches[label] = latin
    for label, latin in matches.items():
        print(f"   flowers: {label} -> {latin}")
    assert len(set(matches.values())) == len(matches), "two Flowers classes map to one species"
    pair(flowers, matches.get, "configs/zs_flowers_common.jsonl", "configs/zs_flowers_latin.jsonl", args.seed)


if __name__ == "__main__":
    main()
