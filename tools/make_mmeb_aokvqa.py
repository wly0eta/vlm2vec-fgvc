"""Build configs/aokvqa_mmeb.jsonl: A-OKVQA scored the way VLM2Vec's MMEB benchmark does it.

MMEB (TIGER-Lab/MMEB-eval, A-OKVQA/test) takes the first 1,000 A-OKVQA val questions and ranks
each against ONE shared pool of 894 answer strings (every question's gold answer is in it),
instead of the official 4 options. Its rows are our configs/aokvqa_val.jsonl rows 0-999 in the
same order (checked here: question text and gold answer must match row by row), so our images
are reused and their 7 GB images.zip isn't needed. The instruction stays the one our models were
trained with (rephrasing it moved scores by < 1 point in experiments/01_prompt_probe/probe_prompts.py).

    python tools/make_mmeb_aokvqa.py          # then: python evaluate.py --mc configs/aokvqa_mmeb.jsonl ...

Chance is 1/894 = 0.11%; the VLM2Vec paper's A-OKVQA column is this number.
"""
import argparse
import json

import pandas as pd
from huggingface_hub import hf_hub_download


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", default="configs/aokvqa_val.jsonl")
    ap.add_argument("--out", default="configs/aokvqa_mmeb.jsonl")
    args = ap.parse_args()

    mmeb = pd.read_parquet(hf_hub_download("TIGER-Lab/MMEB-eval", "A-OKVQA/test-00000-of-00001.parquet",
                                           repo_type="dataset"))
    ours = [json.loads(line) for line in open(args.ours)]
    with open(args.out, "w") as f:
        for k, row in enumerate(mmeb.itertuples()):
            question = row.qry_text.split("following question:", 1)[-1].strip()
            mine = ours[k]
            if question != mine["query"]["text"] or row.tgt_text[0] != mine["positive"]["text"]:
                raise SystemExit(f"row {k} doesn't match {args.ours}: {question!r} vs {mine['query']['text']!r}")
            candidates = list(row.tgt_text)          # gold first, then the rest of the shared pool
            f.write(json.dumps({"query": mine["query"], "instruction": mine.get("instruction"),
                                "positive": {"text": candidates[0]},
                                "negatives": [{"text": c} for c in candidates[1:]]}) + "\n")
    print(f"{len(mmeb)} questions x {len(mmeb.tgt_text[0])} candidates -> {args.out}")


if __name__ == "__main__":
    main()
