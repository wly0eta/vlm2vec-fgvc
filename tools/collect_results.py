"""Tabulate eval_results.json across runs (e.g. for the poster's results tables).

    python tools/collect_results.py runs/*                          # final result of each run
    python tools/collect_results.py runs/q2b_mixed/checkpoint-* runs/q2b_mixed   # per-epoch curve
    python tools/collect_results.py runs/* --csv results.csv

Each cell is the dataset's headline metric: accuracy (multiple choice), top-1
(classification) or recall@1 (retrieval).
"""
import argparse
import csv
import glob
import json
import os
import sys

HEADLINE = {"multiple_choice": "accuracy", "classification": "top1", "retrieval": "recall@1"}


def find_results(path: str) -> str | None:
    """The results file for a run dir: top level if training finished, else newest checkpoint-N."""
    direct = os.path.join(path, "eval_results.json")
    if os.path.exists(direct) or path.endswith(".json"):
        return direct if os.path.exists(direct) else path
    found = sorted(glob.glob(os.path.join(path, "checkpoint-*", "eval_results.json")),
                   key=lambda p: int(p.split("checkpoint-")[1].split(os.sep)[0]))
    return found[-1] if found else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run dirs or results JSON files")
    ap.add_argument("--csv", help="also write a CSV here")
    args = ap.parse_args()

    rows, datasets = [], []
    for run in args.runs:
        path = find_results(run)
        if not path:
            print(f"(no eval_results.json under {run})", file=sys.stderr)
            continue
        with open(path) as f:
            results = json.load(f)
        label = os.path.basename(os.path.normpath(run))
        if label.startswith("checkpoint-"):
            label = os.path.basename(os.path.dirname(os.path.normpath(run)))
        run_name = label
        epoch = results.get("_meta", {}).get("epoch")
        if epoch is not None:
            label += f" (epoch {epoch:g})"
        row = {}
        for name, result in results.items():
            if name.startswith("_"):
                continue
            row[name] = result[HEADLINE[result["task"]]]
            if name not in datasets:
                datasets.append(name)
        rows.append(((run_name, epoch if epoch is not None else float("inf")), label, row))
    table = {label: row for _, label, row in sorted(rows, key=lambda r: r[0])}   # by run, then epoch

    print("| run | " + " | ".join(datasets) + " |")
    print("|---|" + "---|" * len(datasets))
    for run, row in table.items():
        print(f"| {run} | " + " | ".join(f"{row[d]:.3f}" if d in row else "–" for d in datasets) + " |")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run", *datasets])
            for run, row in table.items():
                writer.writerow([run, *(row.get(d, "") for d in datasets)])


if __name__ == "__main__":
    main()
