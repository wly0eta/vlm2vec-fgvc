"""Does a model know that an English bird name and its Latin name are the same species -- and
does its *embedding* put them together? Two text-only tests on the 200 CUB classes.

1. Generative knowledge (--generate MODEL ...): ask the chat model for the scientific name of
   each English CUB name and compare with our Latin label (configs/cub200_latin_aliases.tsv +
   iNat). Scored three ways: exact binomial, species epithet (robust to genus reshuffles such as
   Ammodramus -> Centronyx), and genus.
2. Embedding alignment (--embed LABEL=TARGET ...): embed the 200 English and 200 Latin names with
   the embedding model and check, for each English name, whether its nearest Latin name is the
   right one (200-way, chance 0.5%), plus the reverse. TARGET is a run/checkpoint dir,
   base:<model> for an untrained model, or <old checkpoint>|<base model> for pre-refactor runs.

    python tools/name_alignment.py --generate Qwen/Qwen3-VL-2B-Instruct ~/fgvc_sft_ckpt \\
        --embed base2b=base:Qwen/Qwen3-VL-2B-Instruct q2b_inat=runs/q2b_inat/checkpoint-20990

If generation is good but embedding alignment is poor, the model *knows* the pairs but its
embeddings don't use that knowledge -- the case for training with both names.
"""
import argparse
import json
import logging
import re
import sys

import torch
from transformers import AutoProcessor

import sys, pathlib  # repo root on the path: vlm2vec/ and evaluate.py live there
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from evaluate import pick_device
from vlm2vec.checkpoint import hf_model_of
from vlm2vec.data import Item, JsonlRows, load_processor, positive_item
from vlm2vec.evaluation import Embedder
from vlm2vec.modeling import VLM2VecModel, load_backbone

log = logging.getLogger("name_alignment")
PROMPT = ("What is the scientific (Latin binomial) name of the bird species \"{name}\"? "
          "Answer with only the scientific name.")


def name_pairs(english_file: str, latin_file: str) -> list[tuple[str, str]]:
    """(English, Latin) for every class, from two row-aligned eval files."""
    pairs = {}
    for a, b in zip(JsonlRows(english_file), JsonlRows(latin_file)):
        pairs[positive_item(a).text] = positive_item(b).text
    return sorted(pairs.items())


def normalize(text: str) -> list[str]:
    words = re.sub(r"[^a-z ]", " ", text.lower()).split()
    return words[:2]


def score_generation(answer: str, latin: str) -> dict:
    got, gold = normalize(answer), normalize(latin)
    if len(gold) == 1:                       # genus-level CUB class (Fregata, Sayornis)
        hit = bool(got) and got[0] == gold[0]
        return {"exact": hit, "epithet": hit, "genus": hit}
    return {"exact": got == gold,
            "epithet": len(got) == 2 and got[1] == gold[1],
            "genus": bool(got) and got[0] == gold[0]}


@torch.no_grad()
def generative_test(model_name: str, pairs, device, dtype, batch_size: int = 20) -> dict:
    model = load_backbone(model_name, dtype).to(device).eval()
    processor = AutoProcessor.from_pretrained(hf_model_of(model_name))
    processor.tokenizer.padding_side = "left"        # generation continues from the right end
    rows, counts = [], {"exact": 0, "epithet": 0, "genus": 0}
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        texts = [processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": PROMPT.format(name=en)}]}],
            tokenize=False, add_generation_prompt=True) for en, _ in batch]
        inputs = processor(text=texts, padding=True, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        answers = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        for (en, la), ans in zip(batch, answers):
            s = score_generation(ans, la)
            for k in counts:
                counts[k] += s[k]
            rows.append({"english": en, "latin": la, "answer": ans.strip(), **s})
    del model
    torch.cuda.empty_cache() if device.type == "cuda" else None
    n = len(pairs)
    return {"model": model_name, **{k: v / n for k, v in counts.items()}, "n": n, "answers": rows}


@torch.no_grad()
def embedding_test(target: str, pairs, device, dtype) -> dict:
    if target.startswith("base:"):
        model, contract = VLM2VecModel.for_inference(None, target[5:], dtype)
    elif "|" in target:                                # pre-refactor checkpoint|its base model
        ckpt, base = target.split("|", 1)
        model, contract = VLM2VecModel.for_inference(ckpt, base, dtype)
    else:
        model, contract = VLM2VecModel.for_inference(target, None, dtype)
    model.to(device)
    embed = Embedder(model, load_processor(contract.base_model, contract.max_pixels), contract.max_len, 16)
    en = embed([Item(e) for e, _ in pairs])
    la = embed([Item(l) for _, l in pairs])
    sim = en @ la.T                                    # [english, latin]
    gold = torch.arange(len(pairs))
    rank_en = (sim > sim.diag().unsqueeze(1)).sum(1)   # how many Latin names beat the right one
    rank_la = (sim.T > sim.diag().unsqueeze(1)).sum(1)
    off = sim[~torch.eye(len(pairs), dtype=torch.bool)]
    del model, embed
    torch.cuda.empty_cache() if device.type == "cuda" else None
    return {"target": target,
            "en_to_latin_top1": (rank_en == 0).float().mean().item(),
            "en_to_latin_top5": (rank_en < 5).float().mean().item(),
            "latin_to_en_top1": (rank_la == 0).float().mean().item(),
            "matched_pair_cos": sim.diag().mean().item(),
            "unmatched_pair_cos": off.mean().item(),
            "chance_top1": 1 / len(pairs)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--english", default="configs/cub200_val.jsonl")
    ap.add_argument("--latin", default="configs/cub200_val_latin.jsonl")
    ap.add_argument("--generate", nargs="*", default=[], help="chat models (HF id or adapter dir)")
    ap.add_argument("--embed", nargs="*", default=[], help="LABEL=run_dir | LABEL=base:<model> | LABEL=old_ckpt|<base model>")
    ap.add_argument("--output", default="name_alignment.json")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", stream=sys.stdout)
    logging.getLogger("httpx").setLevel(logging.WARNING)   # one line per HF download request otherwise
    pairs = name_pairs(args.english, args.latin)
    device, dtype = pick_device(), torch.bfloat16
    results = {"pairs": len(pairs), "generate": {}, "embed": {}}

    for name in args.generate:
        r = generative_test(name, pairs, device, dtype)
        results["generate"][name] = r
        log.info("GENERATE %-40s exact=%.3f epithet=%.3f genus=%.3f", name, r["exact"], r["epithet"], r["genus"])
        for row in r["answers"][:8]:
            log.info("    %-28s -> %-30r gold %s", row["english"], row["answer"], row["latin"])
    for spec in args.embed:
        label, target = spec.split("=", 1)
        r = embedding_test(target, pairs, device, dtype)
        results["embed"][label] = r
        log.info("EMBED %-24s en->la top1=%.3f top5=%.3f  la->en top1=%.3f  cos matched=%.3f unmatched=%.3f",
                 label, r["en_to_latin_top1"], r["en_to_latin_top5"], r["latin_to_en_top1"],
                 r["matched_pair_cos"], r["unmatched_pair_cos"])
        json.dump(results, open(args.output, "w"), indent=2)
    json.dump(results, open(args.output, "w"), indent=2)
    log.info("results -> %s", args.output)


if __name__ == "__main__":
    main()
