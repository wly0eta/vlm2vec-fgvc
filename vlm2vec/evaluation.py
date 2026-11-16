"""Embedding-based evaluation: multiple choice, classification, retrieval.

Every metric is "embed the query, embed the candidates, pick the most similar". Each one
here also guards against a way that recipe silently gives the wrong number:

* Ties. argmax returns the *first* maximum. The old multiple-choice eval always put the
  correct option first, so any tie -- duplicate option text, bf16-rounded similarities,
  or a collapsed model that embeds everything identically (which would have scored 100%)
  -- was counted as a win. Here a tie among k options scores 1/k: the expected accuracy
  of breaking it at random, independent of option order.
* Duplicate labels. Retrieval used each row's own positive as its only correct target,
  so two images of the same species retrieving the same (identical) label text counted
  as a miss. Candidates are now deduplicated by text.
* Precision. Embeddings are float32 and similarities computed in float32.

Pass `records=[]` to a metric to also get one dict per row (gold answer, the model's pick, its
top candidates with similarities): evaluate.py saves them as predictions/<task>.jsonl and
tools/show_predictions.py turns them into a report.
"""
import logging

import torch

from .data import Item, build_inputs, load_image, negative_items, positive_item, query_item

log = logging.getLogger(__name__)


class Embedder:
    """Embeds Items in batches. Candidate embeddings are cached by Item (class names and
    answer options like "Yes" repeat across rows, so each is encoded once). Items whose
    image can't be decoded come back as NaN rows so the metric can skip and count them."""

    def __init__(self, model, processor, max_len: int, batch_size: int = 16):
        self.model = model
        self.processor = processor
        self.max_len = max_len
        self.batch_size = batch_size
        self.device = next(model.parameters()).device
        self._cache: dict[Item, torch.Tensor] = {}

    @torch.no_grad()
    def __call__(self, items: list[Item], cache: bool = True) -> torch.Tensor:
        todo = list(dict.fromkeys(i for i in items if i not in self._cache))
        computed = {}
        for start in range(0, len(todo), self.batch_size):
            batch = todo[start:start + self.batch_size]
            images, ok = {}, []
            for item in batch:
                try:
                    if item.image and item.image not in images:
                        images[item.image] = load_image(item.image)
                    ok.append(item)
                except (OSError, ValueError) as err:
                    log.warning("unreadable image, skipping: %s", err)
            if ok:
                inputs = build_inputs(self.processor, ok, self.max_len, images).to(self.device)
                for item, vector in zip(ok, self.model.encode(inputs).float().cpu()):
                    computed[item] = vector
        if cache:
            self._cache.update(computed)
        lookup = {**self._cache, **computed}
        dim = next(iter(lookup.values())).shape[0] if lookup else 1
        missing = torch.full((dim,), float("nan"))
        return torch.stack([lookup.get(i, missing) for i in items])


def tie_aware_hit(similarities: torch.Tensor, gold: int) -> float:
    """1/k if `gold` is among the k candidates tied for the maximum, else 0."""
    tied = (similarities == similarities.max()).nonzero().flatten()
    return 1.0 / len(tied) if bool((tied == gold).any()) else 0.0


def _strict_rank(similarities: torch.Tensor, gold: int) -> int:
    """0-based rank of `gold`, placing every tie ahead of it (never optimistic)."""
    return int((similarities >= similarities[gold]).sum()) - 1


def _ranked(similarities: torch.Tensor, names: list[str], k: int) -> list[list]:
    """[[name, similarity], ...] of the k most similar candidates, best first."""
    top = similarities.topk(min(k, len(names)))
    return [[names[i], round(float(v), 4)] for v, i in zip(top.values, top.indices)]


def multiple_choice(embed: Embedder, rows: list[dict], records: list | None = None) -> dict:
    """Accuracy at picking the correct option among each row's options.

    Options identical to the correct answer are dropped (they can't be distinguished
    from it), and duplicate options collapse to one.
    """
    option_lists = []
    for row in rows:
        positive = positive_item(row)
        others = [n for n in dict.fromkeys(negative_items(row)) if n.key != positive.key]
        option_lists.append([positive, *others])
    queries = embed([query_item(r) for r in rows], cache=False)
    embed([o for options in option_lists for o in options])   # one batched pass, cached
    score = chance = 0.0
    n = skipped = 0
    for row, query, options in zip(rows, queries, option_lists):
        if torch.isnan(query).any():
            skipped += 1
            if records is not None:
                records.append({**_describe(row), "skipped": True})
            continue
        similarities = embed(options) @ query
        hit = tie_aware_hit(similarities, 0)
        score += hit
        chance += 1.0 / len(options)
        n += 1
        if records is not None:
            ranked = _ranked(similarities, [o.text for o in options], 10)   # MMEB-style rows have ~900
            records.append({**_describe(row), "gold": options[0].text, "pred": ranked[0][0],
                            "correct": hit, "gold_rank": _strict_rank(similarities, 0) + 1,
                            "margin": _margin(similarities, 0), "ranked": ranked})
    return {"task": "multiple_choice", "accuracy": score / max(n, 1), "chance": chance / max(n, 1),
            "n": n, "skipped": skipped}


def classification(embed: Embedder, rows: list[dict], class_pool: list[str] | None = None,
                   records: list | None = None) -> dict:
    """Top-1 / top-5 over a fixed label space: every row is scored against ALL labels.

    The label space defaults to the labels present in `rows`; pass `class_pool` to fix it
    independently (e.g. the full training label set). Rows whose label is outside the pool
    count as wrong; `coverage` reports how many there were so a low number can't hide them.
    """
    labels = class_pool or sorted({positive_item(r).text for r in rows})
    index = {label: i for i, label in enumerate(labels)}
    prototypes = embed([Item(label) for label in labels])
    queries = embed([query_item(r) for r in rows], cache=False)
    result = score_classification(queries, prototypes, [index.get(positive_item(r).text) for r in rows],
                                  records, labels)
    if records is not None:
        for record, row in zip(records[-len(rows):], rows):
            record.update(_describe(row))
            record.setdefault("gold", positive_item(row).text)
    return result


def score_classification(queries: torch.Tensor, prototypes: torch.Tensor, gold: list,
                         records: list | None = None, labels: list[str] | None = None) -> dict:
    """Top-1 / top-5 of `queries` (n x d) against class `prototypes` (c x d); gold[i] is row i's
    class index, or None if its label is outside the label space (always wrong). NaN queries
    (unreadable images) are skipped and counted. With `records` (and `labels`, the class names),
    appends one dict per row: gold, pred, gold_rank, margin and the top-5 classes."""
    top1 = top5 = 0.0
    n = skipped = covered = 0
    names = labels or [str(i) for i in range(len(prototypes))]
    for query, g in zip(queries, gold):
        if torch.isnan(query).any():
            skipped += 1
            if records is not None:
                records.append({"skipped": True})
            continue
        n += 1
        similarities = prototypes @ query
        if g is None:
            if records is not None:
                ranked = _ranked(similarities, names, 5)
                records.append({"pred": ranked[0][0], "correct": 0.0, "gold_rank": None,
                                "out_of_pool": True, "ranked": ranked})
            continue
        covered += 1
        hit = tie_aware_hit(similarities, g)
        rank = _strict_rank(similarities, g)
        top1 += hit
        top5 += float(rank < 5)
        if records is not None:
            ranked = _ranked(similarities, names, 5)
            records.append({"gold": names[g], "pred": ranked[0][0], "correct": hit, "gold_rank": rank + 1,
                            "gold_score": round(float(similarities[g]), 4),
                            "margin": _margin(similarities, g), "ranked": ranked})
    return {"task": "classification", "top1": top1 / max(n, 1), "top5": top5 / max(n, 1),
            "chance": 1.0 / len(prototypes), "num_classes": len(prototypes), "n": n,
            "coverage": covered / max(n, 1), "skipped": skipped}


def _margin(similarities: torch.Tensor, gold: int) -> float:
    """Gold similarity minus the best other candidate's: > 0 right, < 0 wrong; its size is
    how confident the decision was."""
    others = torch.cat([similarities[:gold], similarities[gold + 1:]])
    return round(float(similarities[gold] - others.max()), 4) if len(others) else 0.0


def _describe(row: dict) -> dict:
    """What the row asked: question text (if any) and image path."""
    query = row.get("query", {})
    return {"question": query.get("text") or "", "image": query.get("image")}


def retrieval(embed: Embedder, rows: list[dict]) -> dict:
    """Query -> positive retrieval over the deduplicated set of all rows' positives."""
    positives = [positive_item(r) for r in rows]
    candidates = list(dict.fromkeys(positives))
    index = {item: i for i, item in enumerate(candidates)}
    candidate_vectors = embed(candidates)
    queries = embed([query_item(r) for r in rows], cache=False)
    ranks, skipped = [], 0
    for positive, query in zip(positives, queries):
        if torch.isnan(query).any():
            skipped += 1
            continue
        ranks.append(_strict_rank(candidate_vectors @ query, index[positive]))
    ranks = torch.tensor(ranks, dtype=torch.float)
    n = max(len(ranks), 1)
    return {"task": "retrieval", "recall@1": float((ranks < 1).sum()) / n,
            "recall@5": float((ranks < 5).sum()) / n, "mrr": float((1 / (ranks + 1)).sum()) / n,
            "num_candidates": len(candidates), "n": len(ranks), "skipped": skipped}
