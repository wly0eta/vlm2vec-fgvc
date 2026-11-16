"""Build the CUB-200 training sets for the label-vocabulary experiment.

    python tools/make_cub_mixed.py     # -> configs/cub200_train_latin.jsonl, configs/cub200_train_mixed.jsonl
    python tools/make_cub_mixed.py --shots 2 4 10   # + k-shot mixed subsets (configs/cub200_mixed_<k>shot.yaml)
    python tools/make_cub_mixed.py --tune 10 5      # + hyperparameter-tuning split (experiments/06_hparam_sweep/run.sh)

The English->Latin map comes from configs/cub200_val.jsonl vs configs/cub200_val_latin.jsonl,
which are the same images in the same order (checked here): one Latin name per class, built by
`build_dataset.py relabel` with configs/cub200_latin_aliases.tsv.

* latin: every training row labelled with its Latin name (same images as cub200_train.jsonl).
* mixed: same images and the same number of rows (so the same compute as the English run), but
  half of each class's rows use the English name and half the Latin one (alternating within the
  class, seeded). A row's negatives are drawn from the vocabulary of its own positive, so the
  language of a label never gives the answer away.

* mixed k-shot (--shots): k images per class (seeded), half English / half Latin -- k=2 is one
  image under each name. For data-scaling curves; each k also gets a dataset yaml.

* tune (--tune K D): D images per class held out of CUB *train* as a dev set (English and Latin
  versions), and a K-shot mixed training set drawn from the remaining images. Hyperparameters are
  picked on this dev set, never on cub200_val (the test split we report).

Every row and negative carries "class" (the English name), which false-negative masking uses: two
names of one species in a batch are never each other's negatives.
"""
import argparse
import json
import random
from collections import defaultdict


def rows(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write(path: str, data: list[dict]) -> None:
    with open(path, "w") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(data)} rows -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="configs/cub200_train.jsonl")
    ap.add_argument("--val-english", default="configs/cub200_val.jsonl")
    ap.add_argument("--val-latin", default="configs/cub200_val_latin.jsonl")
    ap.add_argument("--num-negatives", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shots", type=int, nargs="*", default=[], help="also write k-shot mixed subsets")
    ap.add_argument("--tune", type=int, nargs=2, metavar=("K", "D"), help="also write the tuning split")
    args = ap.parse_args()

    latin_of = {}
    for e, l in zip(rows(args.val_english), rows(args.val_latin), strict=True):
        assert e["query"]["image"] == l["query"]["image"], "val files are not row-aligned"
        assert latin_of.setdefault(e["positive"]["text"], l["positive"]["text"]) == l["positive"]["text"]
    assert len(set(latin_of.values())) == len(latin_of), "two classes share a Latin name"
    train = rows(args.train)
    english = sorted(latin_of)
    rng = random.Random(args.seed)

    def row(r: dict, cls: str, use_latin: bool, rng: random.Random = rng) -> dict:
        name = (lambda c: latin_of[c]) if use_latin else (lambda c: c)
        others = rng.sample([c for c in english if c != cls], args.num_negatives)
        return {**r, "positive": {"text": name(cls), "class": cls},
                "negatives": [{"text": name(c), "class": c} for c in others]}

    write("configs/cub200_train_latin.jsonl", [row(r, r["positive"]["text"], True) for r in train])
    seen = defaultdict(int)
    order = list(range(len(train)))
    rng.shuffle(order)
    mixed = [None] * len(train)
    for i in order:   # alternate within each class, in a shuffled order: exact half/half per class
        cls = train[i]["positive"]["text"]
        mixed[i] = row(train[i], cls, seen[cls] % 2 == 1)
        seen[cls] += 1
    write("configs/cub200_train_mixed.jsonl", mixed)
    n_latin = sum(r["positive"]["text"] != r["positive"]["class"] for r in mixed)
    print(f"mixed: {n_latin} Latin / {len(mixed) - n_latin} English rows")

    by_class = defaultdict(list)
    for i, r in enumerate(train):
        by_class[r["positive"]["text"]].append(i)
    for k in args.shots:
        rng_k = random.Random(f"{args.seed}-{k}")   # own stream: adding a k never changes the others
        subset = []
        for cls in english:
            picked = rng_k.sample(by_class[cls], min(k, len(by_class[cls])))
            subset += [row(train[i], cls, j % 2 == 1, rng_k) for j, i in enumerate(picked)]
        write(f"configs/cub200_train_mixed_{k}shot.jsonl", subset)
        with open(f"configs/cub200_mixed_{k}shot.yaml", "w") as f:
            f.write(f"# CUB-200, {k} images per class, half English / half Latin labels (tools/make_cub_mixed.py)\n"
                    f"cub200_mixed_{k}shot:\n  data_path: configs/cub200_train_mixed_{k}shot.jsonl\n  weight: 1.0\n")

    if args.tune:
        k, d = args.tune
        rng_t = random.Random(f"{args.seed}-tune")
        dev_en, dev_la, subset = [], [], []
        for cls in english:
            picked = rng_t.sample(by_class[cls], k + d)
            dev_en += [row(train[i], cls, False, rng_t) for i in picked[:d]]
            dev_la += [row(train[i], cls, True, rng_t) for i in picked[:d]]
            subset += [row(train[i], cls, j % 2 == 1, rng_t) for j, i in enumerate(picked[d:])]
        write("configs/cub200_tune_dev_english.jsonl", dev_en)
        write("configs/cub200_tune_dev_latin.jsonl", dev_la)
        write("configs/cub200_train_mixed_tune.jsonl", subset)
        with open("configs/cub200_mixed_tune.yaml", "w") as f:
            f.write(f"# CUB-200 tuning split: {k} train images per class (mixed labels), {d} more held out as dev\n"
                    f"cub200_mixed_tune:\n  data_path: configs/cub200_train_mixed_tune.jsonl\n  weight: 1.0\n")


if __name__ == "__main__":
    main()
