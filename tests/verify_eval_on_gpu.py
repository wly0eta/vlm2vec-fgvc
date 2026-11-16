"""Independent re-check of evaluate.py's numbers on a real checkpoint (run on a GPU node).

    python tests/verify_eval_on_gpu.py runs/q2b_mixed__cub_full [--rows 300]

1. Batch invariance: embeddings at batch size 16 (what evaluate.py uses, with padding) vs batch
   size 1 (no padding at all) must agree -- otherwise padding/pooling leaks into the vectors.
2. Independent scoring: plain argmax over the batch-size-1 embeddings, written from scratch,
   must reproduce evaluation.classification / multiple_choice on the same rows.
3. Controls: shuffling which prototype belongs to which class must drop accuracy to chance;
   the pooled token must be the end of the real sequence, not padding.
"""
import argparse
import random
import sys

import torch

sys.path.insert(0, ".")
from vlm2vec.data import Item, JsonlRows, build_inputs, load_processor, negative_items, positive_item, query_item  # noqa: E402
from vlm2vec.evaluation import Embedder, classification, multiple_choice  # noqa: E402
from vlm2vec.modeling import VLM2VecModel  # noqa: E402


def one_by_one(model, processor, items, max_len):
    out = []
    with torch.no_grad():
        for item in items:
            out.append(model.encode(build_inputs(processor, [item], max_len).to("cuda"))[0].float().cpu())
    return torch.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--rows", type=int, default=300)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    args = ap.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model, contract = VLM2VecModel.for_inference(args.checkpoint, dtype=dtype)
    model.to("cuda")
    processor = load_processor(contract.base_model, contract.max_pixels)
    embed = Embedder(model, processor, contract.max_len, batch_size=16)
    ok = True

    # --- classification (CUB) ---
    all_rows = list(JsonlRows("configs/cub200_val.jsonl"))
    rows = random.Random(1).sample(all_rows, args.rows)
    labels = sorted({positive_item(r).text for r in all_rows})
    official = classification(embed, rows, class_pool=labels)["top1"]

    q16 = embed([query_item(r) for r in rows], cache=False)
    l16 = embed([Item(label) for label in labels])
    q1 = one_by_one(model, processor, [query_item(r) for r in rows], contract.max_len)
    l1 = one_by_one(model, processor, [Item(label) for label in labels], contract.max_len)
    cos_q, cos_l = (q16 * q1).sum(1), (l16 * l1).sum(1)
    print(f"[batch invariance] query cos min={cos_q.min():.5f} mean={cos_q.mean():.5f}; "
          f"label cos min={cos_l.min():.5f} mean={cos_l.mean():.5f}")
    worst = int(cos_q.argmin())
    print(f"[batch invariance] worst query: {query_item(rows[worst]).image}")
    ok &= bool(cos_q.min() > 0.99 and cos_l.min() > 0.99)

    gold = torch.tensor([labels.index(positive_item(r).text) for r in rows])
    sims = q1 @ l1.T
    independent = (sims.argmax(1) == gold).float().mean().item()
    ties = int(((sims == sims.max(1, keepdim=True).values).sum(1) > 1).sum())
    print(f"[CUB top-1] evaluate.py={official:.4f}  independent(bs=1, argmax)={independent:.4f}  rows with ties={ties}")
    ok &= abs(official - independent) <= 2.5 / args.rows   # bf16 batch noise may flip a row or two

    perm = torch.randperm(len(labels), generator=torch.Generator().manual_seed(0))
    shuffled = (sims[:, perm].argmax(1) == gold).float().mean().item()
    print(f"[control] prototypes shuffled across classes: top-1={shuffled:.4f} (chance {1/len(labels):.4f})")
    ok &= shuffled < 0.05

    # what token is pooled? decode the tail of a real batch
    batch = build_inputs(processor, [query_item(rows[0]), Item(labels[0]), Item(labels[-1])], contract.max_len).to("cuda")
    mask = batch["attention_mask"]
    last = mask.shape[1] - 1 - mask.flip(1).int().argmax(1)
    for i in range(3):
        tail = processor.tokenizer.decode(batch["input_ids"][i, max(0, last[i] - 6): last[i] + 1])
        print(f"[pooled token] seq {i}: real length={int(mask[i].sum())}, pooled index={int(last[i])}, tail={tail!r}")
        ok &= int(last[i]) == int(mask[i].sum()) - 1   # right padding: last real token = length-1

    # --- multiple choice (A-OKVQA) ---
    mc_rows = random.Random(1).sample(list(JsonlRows("configs/aokvqa_val.jsonl")), args.rows)
    official_mc = multiple_choice(embed, mc_rows)["accuracy"]
    hits = 0.0
    for r in mc_rows:
        options = [positive_item(r)] + [n for n in negative_items(r) if n.text != positive_item(r).text]
        options = list(dict.fromkeys(options))
        e = one_by_one(model, processor, [query_item(r)] + options, contract.max_len)
        s = e[1:] @ e[0]
        hits += float(s.argmax() == 0 and (s == s.max()).sum() == 1)
    print(f"[A-OKVQA] evaluate.py={official_mc:.4f}  independent(bs=1, strict argmax)={hits / len(mc_rows):.4f}")
    ok &= abs(official_mc - hits / len(mc_rows)) <= 2.5 / args.rows

    print("ALL CHECKS PASSED" if ok else "SOME CHECK FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
