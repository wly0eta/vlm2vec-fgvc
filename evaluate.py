"""Evaluate trained runs (or the untrained base) on any mix of benchmarks.

    python evaluate.py --checkpoint runs/inat_1m \\
        --mc configs/aokvqa_val.jsonl configs/mmmu_val.jsonl configs/mmbench_dev.jsonl \\
        --classify configs/cub200_val.jsonl configs/inat_val.jsonl

    # every epoch of a run (needs --save_strategy epoch at training time), on 10k-row subsets
    python evaluate.py --checkpoint runs/q2b_mixed --each-epoch --limit 10000 --mc ... --classify ...

    # untrained base model, for the "before" column
    python evaluate.py --base-model Qwen/Qwen3-VL-2B-Instruct --mc configs/aokvqa_val.jsonl \\
        --output runs/base_qwen3vl2b/eval_results.json

--checkpoint takes run dirs (final adapter, or the newest checkpoint-N while training is still
running) or specific checkpoint-N dirs. The base model, image budget and prompt format come from
each checkpoint's vlm2vec_config.json; --base-model is needed only for checkpoints trained before
that file existed. Results are printed and merged into <checkpoint>/eval_results.json, which
tools/collect_results.py tabulates.

Every multiple-choice / classification row's outcome (gold answer, the model's pick, its top
candidates with similarities) goes to <checkpoint>/predictions/<task>.jsonl; the log shows the
most confident mistakes, and `python tools/show_predictions.py <checkpoint>` gives the full report
(confusions, per-class accuracy, an HTML gallery with images). --no-predictions turns it off.

Metrics: --mc = multiple-choice accuracy; --classify = top-1/top-5 over the file's full label
set (or --class-pool's), even when --limit subsamples the rows; --retrieval = recall@k / MRR.
All score ties fairly; see vlm2vec/evaluation.py.
"""
import argparse
import datetime
import functools
import json
import logging
import os
import sys

import torch

from vlm2vec.callbacks import checkpoint_epoch, is_epoch_checkpoint, sample_rows
from vlm2vec.checkpoint import resolve_trained_checkpoint
from vlm2vec.data import JsonlRows, load_processor, positive_item
from vlm2vec.evaluation import Embedder, classification, multiple_choice, retrieval
from vlm2vec.modeling import VLM2VecModel

log = logging.getLogger("evaluate")
HEADLINE = {"multiple_choice": "accuracy", "classification": "top1", "retrieval": "recall@1"}


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def expand_targets(paths: list[str], each_epoch: bool) -> list[str]:
    """Run dirs -> checkpoints to evaluate. With each_epoch: every epoch-boundary
    checkpoint-N that precedes the final adapter, then the final adapter itself."""
    targets = []
    for path in paths:
        final = resolve_trained_checkpoint(path)
        if each_epoch and os.path.isdir(path):
            epochs = sorted((c for c in (os.path.join(path, d) for d in os.listdir(path) if d.startswith("checkpoint-"))
                             if is_epoch_checkpoint(c)), key=checkpoint_epoch)
            final_epoch = checkpoint_epoch(final) or float("inf")
            targets += [c for c in epochs if c != final and checkpoint_epoch(c) < final_epoch - 1e-6]
        targets.append(final)
    return targets


def load_tasks(args) -> tuple[list[tuple], list[str]]:
    """[(name, path, metric_fn, rows)] per file, plus missing files. Loaded once, shared by
    every checkpoint evaluated."""
    fixed_pool = sorted({positive_item(r).text for r in JsonlRows(args.class_pool)}) if args.class_pool else None
    tasks, missing = [], []
    for kind, paths in (("mc", args.mc), ("classify", args.classify), ("retrieval", args.retrieval)):
        for path in paths:
            if not os.path.exists(path):
                log.error("missing eval file, skipped: %s", path)
                missing.append(path)
                continue
            all_rows = list(JsonlRows(path))
            rows = sample_rows(all_rows, args.limit)
            name = os.path.splitext(os.path.basename(path))[0]
            if kind == "mc":
                metric = multiple_choice
            elif kind == "classify":
                # label space from the FULL file: scoring a subset only against the labels that
                # happen to appear in it would shrink the task and inflate accuracy
                pool = fixed_pool or sorted({positive_item(r).text for r in all_rows})
                metric = functools.partial(classification, class_pool=pool)
            else:
                metric, name = retrieval, f"{name} (retrieval)"
            tasks.append((name, path, metric, rows))
    return tasks, missing


def save_predictions(records: list, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_mistakes(records: list, k: int = 3) -> None:
    """The k mistakes the model was most sure of (most negative margin)."""
    wrong = sorted((r for r in records if not r.get("skipped") and r.get("correct", 1) < 1),
                   key=lambda r: r.get("margin") or 0)
    for r in wrong[:k]:
        question = f'"{r["question"][:60]}" ' if r.get("question") else ""
        log.info("      x %spredicted %r, gold %r (gold rank %s, margin %+.3f)", question, r["pred"],
                 r.get("gold"), r.get("gold_rank"), r.get("margin") or 0)


def evaluate_one(checkpoint, args, tasks, device, dtype, out: str) -> dict:
    model, contract = VLM2VecModel.for_inference(checkpoint, args.base_model, dtype)
    model.to(device)
    embed = Embedder(model, load_processor(contract.base_model, contract.max_pixels),
                     contract.max_len, args.batch_size)
    epoch = checkpoint_epoch(checkpoint) if checkpoint else None
    log.info("== %s (base %s, max_pixels %d, epoch %s)", checkpoint or "base model",
             contract.base_model, contract.max_pixels, epoch)
    results = {}
    for name, path, metric, rows in tasks:
        records = [] if args.predictions and metric is not retrieval else None
        result = metric(embed, rows, records=records) if records is not None else metric(embed, rows)
        results[name] = {**result, "file": path, "limit": args.limit or None}
        headline = HEADLINE[result["task"]]
        log.info("   %-24s %s=%.4f  chance=%s  n=%d", name, headline, result[headline],
                 f"{result['chance']:.4f}" if "chance" in result else "-", result["n"])
        if records is not None:
            predictions = os.path.join(os.path.dirname(out) or ".", "predictions", f"{name}.jsonl")
            save_predictions(records, predictions)
            results[name]["predictions"] = predictions
            log_mistakes(records)
    results["_meta"] = {"checkpoint": checkpoint, "base_model": contract.base_model, "epoch": epoch,
                        "max_pixels": contract.max_pixels, "dtype": args.dtype,
                        "evaluated_at": datetime.datetime.now().isoformat(timespec="seconds")}
    del model, embed
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def save(results: dict, out: str) -> None:
    previous = json.load(open(out)) if os.path.exists(out) else {}
    previous.update(results)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(previous, f, indent=2)
    log.info("   results -> %s", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", nargs="+", help="run dir(s) or checkpoint-N dir(s); omit for the base model")
    ap.add_argument("--each-epoch", action="store_true", help="also evaluate each run's epoch checkpoints")
    ap.add_argument("--base-model", help="required without --checkpoint, or for pre-contract checkpoints")
    ap.add_argument("--mc", nargs="*", default=[], help="multiple-choice JSONL files")
    ap.add_argument("--classify", nargs="*", default=[], help="classification JSONL files")
    ap.add_argument("--retrieval", nargs="*", default=[], help="retrieval JSONL files")
    ap.add_argument("--class-pool", help="JSONL whose labels define every classification label space")
    ap.add_argument("--limit", type=int, default=0, help="random (seeded) subset of rows per file; 0 = all")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--no-predictions", dest="predictions", action="store_false",
                    help="don't save per-row predictions/<task>.jsonl")
    ap.add_argument("--output", help="results JSON (default: <checkpoint>/eval_results.json)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                        stream=sys.stdout)
    sys.stdout.reconfigure(errors="backslashreplace")   # never die on printing a non-ASCII question
    logging.getLogger("httpx").setLevel(logging.WARNING)   # one line per HF download request otherwise
    if not (args.mc or args.classify or args.retrieval):
        ap.error("give at least one of --mc / --classify / --retrieval")
    if not args.checkpoint and not (args.base_model and args.output):
        ap.error("without --checkpoint, give --base-model and --output")

    targets = expand_targets(args.checkpoint, args.each_epoch) if args.checkpoint else [None]
    if args.output and len(targets) > 1:
        ap.error("--output only works with a single checkpoint")
    tasks, missing = load_tasks(args)
    device, dtype = pick_device(), torch.bfloat16 if args.dtype == "bf16" else torch.float32
    for checkpoint in targets:
        out = args.output or os.path.join(checkpoint, "eval_results.json")
        results = evaluate_one(checkpoint, args, tasks, device, dtype, out)
        save(results, out)
    if missing:
        sys.exit(f"{len(missing)} eval file(s) missing: {missing}")


if __name__ == "__main__":
    main()
