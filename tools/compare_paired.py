"""Is model A really better than model B on a task? Paired comparison on the same images.

    python tools/compare_paired.py runs/q2b_inat_ep2/predictions/zs_cars.jsonl runs/gen_qwen3vl2b/predictions/zs_cars.jsonl
    python tools/compare_paired.py --runs runs/q2b_inat_ep2 runs/gen_qwen3vl2b      # every task both have

Reads two predictions JSONL files (evaluate.py's or tools/zero_shot_generative.py's: one row per image
with "image" and "correct"), keeps the images both scored, and reports:
* each model's accuracy on those shared images,
* the discordant pairs: images only A gets right / only B gets right (the only rows that carry
  information about the difference),
* McNemar's exact test p-value (two-sided binomial on the discordant pairs), and
* a 95% bootstrap interval for the accuracy difference A - B.
A difference is "real" when the interval excludes 0 (equivalently, p < 0.05).
"""
import argparse
import glob
import json
import math
import os
import random


def load(path: str) -> dict[str, bool]:
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "image" in r and "correct" in r and not r.get("skipped"):
                out[r["image"]] = float(r["correct"]) >= 1.0   # tie-aware scores: a shared top-1 is not a win
    return out


def mcnemar_exact(b: int, c: int) -> float:
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def compare(a_path: str, b_path: str, boot: int = 2000, seed: int = 0) -> dict:
    a, b = load(a_path), load(b_path)
    shared = sorted(set(a) & set(b))
    if not shared:
        return {"n": 0}
    pairs = [(a[k], b[k]) for k in shared]
    only_a = sum(x and not y for x, y in pairs)
    only_b = sum(y and not x for x, y in pairs)
    rng = random.Random(seed)
    diffs = sorted(sum(x - y for x, y in (pairs[rng.randrange(len(pairs))] for _ in pairs)) / len(pairs)
                   for _ in range(boot))
    return {"n": len(shared), "acc_a": sum(x for x, _ in pairs) / len(pairs),
            "acc_b": sum(y for _, y in pairs) / len(pairs), "only_a": only_a, "only_b": only_b,
            "p": mcnemar_exact(only_a, only_b), "ci": (diffs[int(0.025 * boot)], diffs[int(0.975 * boot)])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="A.jsonl B.jsonl")
    ap.add_argument("--runs", nargs=2, metavar=("A", "B"), help="two run dirs: compare every shared task")
    args = ap.parse_args()
    if args.runs:
        tasks = sorted({os.path.basename(p) for p in glob.glob(f"{args.runs[0]}/predictions/*.jsonl")} &
                       {os.path.basename(p) for p in glob.glob(f"{args.runs[1]}/predictions/*.jsonl")})
        jobs = [(t.removesuffix(".jsonl"), f"{args.runs[0]}/predictions/{t}", f"{args.runs[1]}/predictions/{t}")
                for t in tasks]
        names = args.runs
    else:
        jobs, names = [("task", args.files[0], args.files[1])], args.files
    print(f"A = {names[0]}\nB = {names[1]}")
    print("| task | n (shared) | A | B | A - B (95% CI) | only A right | only B right | McNemar p |")
    print("|---|---|---|---|---|---|---|---|")
    for task, pa, pb in jobs:
        r = compare(pa, pb)
        if not r["n"]:
            print(f"| {task} | 0 | | | | | | |")
            continue
        lo, hi = r["ci"]
        flag = " *" if r["p"] < 0.05 else ""
        print(f"| {task} | {r['n']} | {r['acc_a']:.3f} | {r['acc_b']:.3f} | {r['acc_a'] - r['acc_b']:+.3f} "
              f"({lo:+.3f}, {hi:+.3f}) | {r['only_a']} | {r['only_b']} | {r['p']:.2g}{flag} |")


if __name__ == "__main__":
    main()
