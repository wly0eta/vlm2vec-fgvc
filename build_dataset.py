"""Build a training/eval JSONL file from a raw dataset.

    # iNaturalist 2021 from local disk (images referenced in place)
    python build_dataset.py inat-local --root /datasets/ai/inaturalist --annotations train.json \\
        --per-class-cap 100 --out configs/inat_train_1m.jsonl
    python build_dataset.py inat-local --root /datasets/ai/inaturalist --annotations val.json \\
        --out configs/inat_val.jsonl

    # Hugging Face Hub datasets (images saved under --image-dir)
    python build_dataset.py classification --dataset Donghyun99/CUB-200-2011 --split test \\
        --out configs/cub200_val.jsonl --image-dir images/cub200
    python build_dataset.py inat-chat --out configs/inaturalist_train.jsonl --image-dir images/inaturalist
    python build_dataset.py aokvqa --split validation --out configs/aokvqa_val.jsonl --image-dir images/aokvqa
    python build_dataset.py mmbench --out configs/mmbench_dev.jsonl --image-dir images/mmbench
    python build_dataset.py mmmu --out configs/mmmu_val.jsonl --image-dir images/mmmu

    # CUB with iNat's Latin names (164/200 classes match), and the same subset with the
    # original English names -- comparing the two isolates the label-vocabulary effect
    python build_dataset.py relabel --input configs/cub200_val.jsonl \
        --inat-annotations /datasets/ai/inaturalist/val.json --out configs/cub164_val_latin.jsonl
    python build_dataset.py relabel --input configs/cub200_val.jsonl --keep-common-names \
        --inat-annotations /datasets/ai/inaturalist/val.json --out configs/cub164_val_common.jsonl
    python build_dataset.py relabel --input configs/cub200_val.jsonl --aliases configs/cub200_latin_aliases.tsv \
        --inat-annotations /datasets/ai/inaturalist/val.json --out configs/cub200_val_latin.jsonl

Common options (filters, caps, negatives, splits) apply to every source; see --help.
"""
import argparse
import logging
import sys

import random
from collections import defaultdict

from vlm2vec.data import JsonlRows
from vlm2vec.sources import huggingface, inat_local
from vlm2vec.sources.common import build, relabel_rows, write_jsonl


def read_aliases(path: str) -> dict[str, str]:
    """'<label>\\t<new name>[\\t<note>]' lines; '#' starts a comment."""
    aliases = {}
    with open(path) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                label, new = line.rstrip("\n").split("\t")[:2]
                aliases[label] = new
    return aliases


def main() -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", required=True, help="output JSONL")
    common.add_argument("--image-dir", default="images/misc", help="where downloaded images are saved")
    common.add_argument("--limit", type=int, default=0, help="stop after this many kept examples")
    common.add_argument("--taxon-class", help="keep one taxonomic class, e.g. Aves (sources with taxonomy)")
    common.add_argument("--per-class-cap", type=int, default=0, help="at most N examples per label (balancing)")
    common.add_argument("--num-negatives", type=int, default=3, help="sampled distractor labels per row (classification)")
    common.add_argument("--hard-negative-fraction", type=float, default=0.0,
                        help="share of negatives drawn from the same genus/family (needs taxonomy)")
    common.add_argument("--val-fraction", type=float, default=0.0, help="also write a stratified val split")
    common.add_argument("--val-out", help="val JSONL path (default: derived from --out)")
    common.add_argument("--seed", type=int, default=0)

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="source", required=True)
    p = sub.add_parser("inat-local", parents=[common], help="iNaturalist 2021 COCO JSON on local disk")
    p.add_argument("--root", required=True)
    p.add_argument("--annotations", required=True, help="train.json | train_mini.json | val.json")
    p = sub.add_parser("classification", parents=[common], help="HF image + ClassLabel dataset (e.g. CUB-200)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--keep-raw-labels", action="store_true", help="don't strip '001.' prefixes / underscores")
    p.add_argument("--image-field", default="image")
    p.add_argument("--label-field", default="label", help="a ClassLabel column (e.g. FGVC-Aircraft: label = variant)")
    p.add_argument("--instruction", help="task instruction for every row (default: identify the species), "
                                         "e.g. 'Identify the car model in the image.'")
    p = sub.add_parser("inat-chat", parents=[common], help="chat-format species set (sxj1215/inaturalist)")
    p.add_argument("--dataset", default="sxj1215/inaturalist")
    p.add_argument("--split", default="train")
    p = sub.add_parser("aokvqa", parents=[common], help="A-OKVQA multiple choice")
    p.add_argument("--split", default="train")
    p = sub.add_parser("mmbench", parents=[common], help="MMBench (dev split has answers)")
    p.add_argument("--dataset", default="lmms-lab/MMBench_EN")
    p.add_argument("--split", default="dev")
    p = sub.add_parser("mmmu", parents=[common], help="MMMU multiple choice (validation has answers)")
    p.add_argument("--split", default="validation")
    p.add_argument("--subjects", nargs="*", help="subset of subjects (default: all)")
    p = sub.add_parser("relabel", help="rename an existing classification JSONL's labels to iNat scientific "
                                       "names via English common names; classes without a match are dropped")
    p.add_argument("--input", required=True)
    p.add_argument("--inat-annotations", required=True, help="an iNat-2021 json (val.json is enough)")
    p.add_argument("--keep-common-names", action="store_true",
                   help="keep the original names (the control set: same classes/images, original vocabulary)")
    p.add_argument("--aliases", help="TSV of '<label>\\t<new name>' overriding/extending the iNat match "
                                     "(e.g. configs/cub200_latin_aliases.tsv)")
    p.add_argument("--out", required=True)
    p.add_argument("--num-negatives", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", stream=sys.stdout)

    if args.source == "relabel":
        index = inat_local.common_name_index(args.inat_annotations)
        aliases = read_aliases(args.aliases) if args.aliases else {}

        def mapping(label):
            return aliases.get(label) or index.get(inat_local.normalize_common_name(label))

        rows = list(JsonlRows(args.input))
        targets = defaultdict(set)
        for label in {r["positive"]["text"] for r in rows}:
            if mapping(label):
                targets[mapping(label)].add(label)
        merged = {new: sorted(old) for new, old in targets.items() if len(old) > 1}
        if merged:   # two classes renamed to one name would silently merge them
            sys.exit(f"relabel: several labels map to the same name: {merged}")
        rows, stats = relabel_rows(rows, mapping, args.keep_common_names, args.num_negatives,
                                   random.Random(args.seed))
        write_jsonl(args.out, rows)
        logging.info("relabel: kept %d/%d classes, %d/%d rows -> %s", stats["classes"], stats["classes_before"],
                     stats["rows"], stats["rows_before"], args.out)
        return

    records = {
        "inat-local": lambda: inat_local.records(args.root, args.annotations, args.seed),
        "classification": lambda: huggingface.image_classification(
            args.dataset, args.split, label_field=args.label_field, image_field=args.image_field,
            clean=not args.keep_raw_labels),
        "inat-chat": lambda: huggingface.inat_chat(args.dataset, args.split),
        "aokvqa": lambda: huggingface.aokvqa(args.split),
        "mmbench": lambda: huggingface.mmbench(args.split, args.dataset),
        "mmmu": lambda: huggingface.mmmu(args.split, subjects=args.subjects),
    }[args.source]()
    build(records, out=args.out, image_dir=args.image_dir, limit=args.limit, taxon_class=args.taxon_class,
          per_class_cap=args.per_class_cap, num_negatives=args.num_negatives,
          hard_fraction=args.hard_negative_fraction, val_fraction=args.val_fraction,
          val_out=args.val_out, seed=args.seed,
          **({"instruction": args.instruction} if getattr(args, "instruction", None) else {}))


if __name__ == "__main__":
    main()
