"""Zero-shot classification with the chat model itself (no embedding): the fair "base model" baseline.

    python tools/zero_shot_generative.py --model Qwen/Qwen3-VL-2B-Instruct --limit 500 \\
        --classify configs/zs_cars.jsonl configs/cub200_val.jsonl --output runs/gen_qwen3vl2b/results.json
    python tools/zero_shot_generative.py ... --check     # first verify the cached scoring (below) is exact

An untrained VLM scores ~chance as an *embedder* because it was never trained to embed, which says
nothing about what it knows. Here it is used as built: image + the row's instruction ("Identify the
car model in the image.") + "Answer with only the name.", and two scores per image:

* rank:  log p(answer = label) for EVERY label in the pool (the file's labels, or --class-pool's),
         top-1 over all of them -- the same task and label space as evaluate.py's classification.
         Reported with the summed log-prob of label + end-of-turn token (rank_sum) and per-token
         mean (rank_mean); the end token stops a label that is a prefix of another from winning.
* gen:   greedy free answer, exact match after normalization (strict: phrasing must match).

Rows are evaluate.py's seeded --limit subset, so per-image results pair up with an embedder's
predictions/<task>.jsonl (tools/compare_paired.py). The image prefix is encoded once per image and its
KV cache reused for every label batch; --check compares that against full uncached forwards.
"""
import argparse
import copy
import json
import logging
import os
import re
import sys

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

import sys, pathlib  # repo root on the path: vlm2vec/ and evaluate.py live there
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from vlm2vec.callbacks import sample_rows
from vlm2vec.checkpoint import DEFAULT_MAX_PIXELS
from vlm2vec.data import JsonlRows, positive_item

log = logging.getLogger("zero_shot_generative")
SUFFIX = " Answer with only the name."


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


class Scorer:
    def __init__(self, model_name: str, max_pixels: int, device: str = "cuda", dtype=torch.bfloat16):
        self.processor = AutoProcessor.from_pretrained(model_name, max_pixels=max_pixels)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name, dtype=dtype, attn_implementation="sdpa").to(device).eval()
        self.tok = self.processor.tokenizer
        self.device = device
        self.end = self.tok.convert_tokens_to_ids("<|im_end|>")

    def prefix(self, image_path: str, instruction: str) -> dict:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction + SUFFIX}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        with Image.open(image_path) as im:
            image = im.convert("RGB")
        return self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)

    def label_ids(self, labels: list[str]) -> list[list[int]]:
        return [self.tok(label, add_special_tokens=False).input_ids + [self.end] for label in labels]

    @torch.no_grad()
    def rank(self, inputs: dict, ids: list[list[int]], batch: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
        """(summed, mean) log p(label tokens | image prefix) for every label, via one prefix pass."""
        out = self.model(**inputs, use_cache=True)
        first = out.logits[0, -1].float().log_softmax(-1)
        cache, P = out.past_key_values, inputs["input_ids"].shape[1]
        sums, means = [], []
        for start in range(0, len(ids), batch):
            chunk = ids[start:start + batch]
            B, L = len(chunk), max(len(x) for x in chunk)
            tokens = torch.full((B, L), self.end, dtype=torch.long, device=self.device)
            valid = torch.zeros((B, L), dtype=torch.bool, device=self.device)
            for i, x in enumerate(chunk):
                tokens[i, :len(x)] = torch.tensor(x, device=self.device)
                valid[i, :len(x)] = True
            past = copy.deepcopy(cache)
            past.batch_repeat_interleave(B)
            mask = torch.cat([torch.ones((B, P), dtype=torch.long, device=self.device), valid.long()], 1)
            # Text after the image continues M-RoPE at P + rope_delta on all 3 axes (what generate()
            # does); given a cache + full mask the model would build positions for all P+L tokens.
            positions = torch.arange(P, P + L, device=self.device) + self.model.model.rope_deltas.to(self.device)
            positions = positions.view(1, 1, L).expand(3, B, L)
            logits = self.model(input_ids=tokens, attention_mask=mask, past_key_values=past,
                                position_ids=positions,
                                cache_position=torch.arange(P, P + L, device=self.device)).logits
            lp = logits[:, :-1].float().log_softmax(-1).gather(-1, tokens[:, 1:, None])[..., 0]
            per_token = torch.cat([first[tokens[:, 0]][:, None], lp], 1) * valid
            sums.append(per_token.sum(1))
            means.append(per_token.sum(1) / valid.sum(1))
        return torch.cat(sums).cpu(), torch.cat(means).cpu()

    @torch.no_grad()
    def rank_uncached(self, image_path: str, instruction: str, label: str) -> float:
        """Reference for --check: one full forward over prefix + label, no cache."""
        inputs = self.prefix(image_path, instruction)
        ids = self.label_ids([label])[0]
        P = inputs["input_ids"].shape[1]
        full = dict(inputs)
        full["input_ids"] = torch.cat([inputs["input_ids"], torch.tensor([ids], device=self.device)], 1)
        full["attention_mask"] = torch.ones_like(full["input_ids"])
        if "mm_token_type_ids" in full:   # appended label tokens are text (type 0)
            full["mm_token_type_ids"] = torch.cat(
                [inputs["mm_token_type_ids"], torch.zeros((1, len(ids)), dtype=inputs["mm_token_type_ids"].dtype,
                                                          device=self.device)], 1)
        lp = self.model(**full).logits[0, P - 1:-1].float().log_softmax(-1)
        return float(lp.gather(-1, torch.tensor(ids, device=self.device)[:, None]).sum())

    @torch.no_grad()
    def generate(self, inputs: dict) -> str:
        out = self.model.generate(**inputs, max_new_tokens=32, do_sample=False)
        return self.tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--classify", nargs="+", required=True)
    ap.add_argument("--class-pool", nargs="*", default=[],
                    help="per --classify file (same order; '-' = the file's own labels)")
    ap.add_argument("--limit", type=int, default=500, help="evaluate.py's seeded subset (0 = all rows)")
    ap.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    ap.add_argument("--output", required=True, help="results JSON; per-row records go next to it")
    ap.add_argument("--check", action="store_true", help="verify cached scoring on a few labels, then exit")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", stream=sys.stdout)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    scorer = Scorer(args.model, args.max_pixels, dtype=torch.float32 if args.dtype == "fp32" else torch.bfloat16)

    results = json.load(open(args.output)) if os.path.exists(args.output) else {}
    os.makedirs(os.path.join(os.path.dirname(args.output) or ".", "predictions"), exist_ok=True)
    for k, path in enumerate(args.classify):
        pool_file = args.class_pool[k] if k < len(args.class_pool) and args.class_pool[k] != "-" else path
        labels = sorted({positive_item(r).text for r in JsonlRows(pool_file)})
        ids = scorer.label_ids(labels)
        rows = sample_rows(list(JsonlRows(path)), args.limit)
        if args.check:
            r = rows[0]
            inputs = scorer.prefix(r["query"]["image"], r["instruction"])
            sums, _ = scorer.rank(inputs, ids)
            for j in list(range(3)) + [labels.index(positive_item(r).text)]:
                ref = scorer.rank_uncached(r["query"]["image"], r["instruction"], labels[j])
                log.info("check %-40s cached %.3f  uncached %.3f", labels[j][:40], float(sums[j]), ref)
            return
        name = os.path.basename(path).removesuffix(".jsonl")
        hits = {"rank_sum": 0, "rank_mean": 0, "gen": 0}
        out = open(os.path.join(os.path.dirname(args.output) or ".", "predictions", f"{name}.jsonl"), "w")
        for i, r in enumerate(rows):
            gold = positive_item(r).text
            inputs = scorer.prefix(r["query"]["image"], r["instruction"])
            sums, means = scorer.rank(inputs, ids)
            answer = scorer.generate(inputs)
            pred_sum, pred_mean = labels[int(sums.argmax())], labels[int(means.argmax())]
            row = {"image": r["query"]["image"], "gold": gold, "pred": pred_sum, "pred_mean": pred_mean,
                   "answer": answer, "correct": pred_sum == gold,
                   "gold_rank": int((sums > sums[labels.index(gold)]).sum()) + 1 if gold in labels else None}
            hits["rank_sum"] += pred_sum == gold
            hits["rank_mean"] += pred_mean == gold
            hits["gen"] += normalize(answer) == normalize(gold)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (i + 1) % 100 == 0:
                log.info("%s %d/%d  rank_sum %.3f", name, i + 1, len(rows), hits["rank_sum"] / (i + 1))
        out.close()
        n = len(rows)
        results[name] = {**{k2: v / n for k2, v in hits.items()}, "n": n, "classes": len(labels),
                         "chance": 1 / len(labels), "file": path, "pool": pool_file, "limit": args.limit or None}
        log.info("   %-28s rank_sum=%.4f rank_mean=%.4f gen_exact=%.4f  (n=%d, %d classes)", name,
                 results[name]["rank_sum"], results[name]["rank_mean"], results[name]["gen"], n, len(labels))
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
