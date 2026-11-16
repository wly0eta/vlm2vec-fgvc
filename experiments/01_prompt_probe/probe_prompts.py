"""Prompt probe for classification: how much do the scores depend on the exact wording?

Three questions, answered without any training:

1. Instruction sensitivity -- score the same images with several paraphrases of the query
   instruction. A model that only works with the instruction it was trained on swings a lot.
2. Label templates -- embed each class name inside several templates ("a photo of a {name}.")
   and score each, plus their ensemble: one prototype per class = mean of its template embeddings
   (CLIP-style prompt ensembling).
3. Name ensembling (with --paired) -- a second, row-aligned copy of the file whose labels are
   other names for the same classes (configs/cub200_val_latin.jsonl for configs/cub200_val.jsonl).
   Scores primary names only, paired names only, and one prototype per class averaging both.

    python experiments/01_prompt_probe/probe_prompts.py --checkpoint runs/q2b_inat \\
        --classify configs/cub200_val.jsonl --paired configs/cub200_val_latin.jsonl \\
        --extra-templates "a photo of a {name}, a type of bird."

    python experiments/01_prompt_probe/probe_prompts.py --collect runs/*/prompt_probe.json runs/*/checkpoint-*/prompt_probe.json

Query embeddings are the expensive part (one image pass per row per instruction); label
embeddings are text-only and cheap. --limit subsamples rows (same rows for every variant), while
the label space stays the file's full label set. Results go to <checkpoint>/prompt_probe.json.
"""
import argparse
import datetime
import json
import logging
import os
import random
import statistics
import sys

import torch

import sys, pathlib  # repo root on the path: vlm2vec/ and evaluate.py live there
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from evaluate import pick_device
from vlm2vec.callbacks import checkpoint_epoch
from vlm2vec.checkpoint import resolve_trained_checkpoint
from vlm2vec.data import Item, JsonlRows, load_processor, positive_item, query_item
from vlm2vec.evaluation import Embedder, score_classification
from vlm2vec.modeling import VLM2VecModel

log = logging.getLogger("probe_prompts")

# The first entry of each list is what training used; the rest are paraphrases.
INSTRUCTIONS = [
    "Identify the species in the image.",
    "What species is shown in this photo?",
    "Which species is this?",
    "Name the organism in the picture.",
    "Classify the image by species.",
    "Represent the query for retrieval.",
]
TEMPLATES = [
    "{name}",
    "a photo of a {name}.",
    "an image of a {name}.",
    "the species {name}.",
]


def mean_prototypes(groups: list[torch.Tensor]) -> torch.Tensor:
    """Average several (classes x d) embedding sets into one unit-length prototype per class."""
    return torch.nn.functional.normalize(torch.stack(groups).mean(0), dim=-1)


def probe_file(embed: Embedder, path: str, paired_path: str | None, instructions: list[str],
               templates: list[str], limit: int, seed: int) -> dict:
    all_rows = list(JsonlRows(path))
    labels = sorted({positive_item(r).text for r in all_rows})          # full label space
    index = {label: i for i, label in enumerate(labels)}
    picked = sorted(random.Random(seed).sample(range(len(all_rows)), limit)) \
        if 0 < limit < len(all_rows) else list(range(len(all_rows)))
    rows = [all_rows[i] for i in picked]
    gold = [index[positive_item(r).text] for r in rows]

    def label_embeddings(names: list[str], template: str) -> torch.Tensor:
        return embed([Item(template.format(name=n)) for n in names])

    def top1(queries, prototypes, g=gold):
        return score_classification(queries, prototypes, g)["top1"]

    out = {"file": path, "n": len(rows), "num_classes": len(labels)}

    # 1. instruction sensitivity (labels as trained: bare names)
    bare = label_embeddings(labels, "{name}")
    per_instruction, trained_queries = {}, None
    for instruction in instructions:
        queries = embed([Item(q.text, q.image, instruction) for q in map(query_item, rows)], cache=False)
        trained_queries = queries if trained_queries is None else trained_queries
        per_instruction[instruction] = top1(queries, bare)
        log.info("   instruction %-40r top1=%.4f", instruction, per_instruction[instruction])
    scores = list(per_instruction.values())
    out["instructions"] = per_instruction
    out["instruction_summary"] = {"trained": scores[0], "mean": statistics.mean(scores), "min": min(scores),
                                  "max": max(scores), "std": statistics.pstdev(scores)}

    # 2. label templates, each alone and ensembled (queries use the trained instruction)
    per_template = {t: label_embeddings(labels, t) for t in templates}
    out["templates"] = {t: top1(trained_queries, p) for t, p in per_template.items()}
    out["template_ensemble"] = top1(trained_queries, mean_prototypes(list(per_template.values())))
    for t, s in out["templates"].items():
        log.info("   template %-43r top1=%.4f", t, s)
    log.info("   template ensemble%35s top1=%.4f", "", out["template_ensemble"])

    # 3. other names for the same classes
    if paired_path:
        paired_rows = list(JsonlRows(paired_path))
        if len(paired_rows) != len(all_rows) or any(
                query_item(a).image != query_item(b).image for a, b in zip(all_rows, paired_rows)):
            raise ValueError(f"{paired_path} is not row-aligned with {path}")
        other = {}
        for a, b in zip(all_rows, paired_rows):
            other.setdefault(positive_item(a).text, set()).add(positive_item(b).text)
        if any(len(v) != 1 for v in other.values()):
            raise ValueError("paired labels are not one-to-one")
        other_names = [next(iter(other[label])) for label in labels]   # aligned with `labels`
        paired_bare = label_embeddings(other_names, "{name}")
        paired_templates = [label_embeddings(other_names, t) for t in templates]
        out["names"] = {
            "primary": top1(trained_queries, bare),
            "paired": top1(trained_queries, paired_bare),
            "both": top1(trained_queries, mean_prototypes([bare, paired_bare])),
            "both_x_templates": top1(trained_queries, mean_prototypes(list(per_template.values()) + paired_templates)),
        }
        for k, s in out["names"].items():
            log.info("   names %-46s top1=%.4f", k, s)
    return out


def collect(paths: list[str]) -> None:
    """One table row per probe file and eval set."""
    print("| run | set | trained instr. | instr. mean | min–max | best template | template ens. "
          "| primary names | paired names | both | both × templates |")
    print("|" + "---|" * 11)
    for path in sorted(paths):
        results = json.load(open(path))
        run = os.path.relpath(os.path.dirname(path))
        epoch = results.get("_meta", {}).get("epoch")
        if epoch:
            run += f" (epoch {epoch:g})"
        for name, r in results.items():
            if name.startswith("_"):
                continue
            s, names = r["instruction_summary"], r.get("names", {})
            best = max(r["templates"].values())
            cells = [f"{s['trained']:.3f}", f"{s['mean']:.3f}", f"{s['min']:.3f}–{s['max']:.3f}", f"{best:.3f}",
                     f"{r['template_ensemble']:.3f}",
                     *(f"{names[k]:.3f}" if k in names else "–" for k in ("primary", "paired", "both", "both_x_templates"))]
            print(f"| {run} | {name} | " + " | ".join(cells) + " |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collect", nargs="+", help="print a table of existing prompt_probe.json files and exit")
    ap.add_argument("--checkpoint", help="run dir or checkpoint-N dir; omit for the base model")
    ap.add_argument("--base-model", help="required without --checkpoint, or for pre-contract checkpoints")
    ap.add_argument("--classify", nargs="+", default=[], help="classification JSONL files")
    ap.add_argument("--paired", nargs="*", default=[],
                    help="row-aligned copies of the --classify files with other names (same order; '-' = none)")
    ap.add_argument("--instructions", nargs="+", default=INSTRUCTIONS, help="first = the trained one")
    ap.add_argument("--templates", nargs="+", default=TEMPLATES, help="label templates with {name}")
    ap.add_argument("--extra-templates", nargs="*", default=[], help="added to --templates (e.g. taxon-specific)")
    ap.add_argument("--limit", type=int, default=2000, help="rows per file (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--output", help="default: <checkpoint>/prompt_probe.json")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                        stream=sys.stdout)
    if args.collect:
        collect(args.collect)
        return
    if not args.classify:
        ap.error("give --classify (or --collect)")
    if not args.checkpoint and not (args.base_model and args.output):
        ap.error("without --checkpoint, give --base-model and --output")
    paired = [None if p == "-" else p for p in args.paired] + [None] * (len(args.classify) - len(args.paired))

    checkpoint = resolve_trained_checkpoint(args.checkpoint) if args.checkpoint else None
    device, dtype = pick_device(), torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model, contract = VLM2VecModel.for_inference(checkpoint, args.base_model, dtype)
    model.to(device)
    embed = Embedder(model, load_processor(contract.base_model, contract.max_pixels), contract.max_len,
                     args.batch_size)
    results = {}
    for path, paired_path in zip(args.classify, paired):
        log.info("== %s on %s", checkpoint or contract.base_model, path)
        results[os.path.splitext(os.path.basename(path))[0]] = probe_file(
            embed, path, paired_path, args.instructions, args.templates + args.extra_templates, args.limit, args.seed)
    results["_meta"] = {"checkpoint": checkpoint, "base_model": contract.base_model,
                        "epoch": checkpoint_epoch(checkpoint) if checkpoint else None,
                        "limit": args.limit, "seed": args.seed,
                        "evaluated_at": datetime.datetime.now().isoformat(timespec="seconds")}
    out = args.output or os.path.join(checkpoint, "prompt_probe.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    log.info("results -> %s", out)


if __name__ == "__main__":
    main()
