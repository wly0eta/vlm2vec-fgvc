"""Metric behaviour, with a fake embedder so no model is needed."""
import torch
import torch.nn.functional as F

from vlm2vec.data import Item
from vlm2vec.evaluation import classification, multiple_choice, retrieval, tie_aware_hit


class FakeEmbedder:
    """Maps an Item's text (queries: their image path) to a fixed vector."""

    def __init__(self, vectors: dict[str, torch.Tensor], default=None):
        self.vectors = vectors
        self.default = default

    def __call__(self, items, cache=True):
        out = []
        for item in items:
            key = item.image if item.instruction is not None else item.text
            out.append(self.vectors.get(key, self.default))
        return torch.stack(out)


def mc_row(image, answer, options):
    return {"query": {"text": "q", "image": image}, "instruction": "",
            "positive": {"text": answer}, "negatives": [{"text": o} for o in options]}


def test_tie_aware_hit():
    assert tie_aware_hit(torch.tensor([0.9, 0.1]), 0) == 1.0
    assert tie_aware_hit(torch.tensor([0.5, 0.5, 0.1, 0.5]), 0) == 1 / 3
    assert tie_aware_hit(torch.tensor([0.5, 0.9]), 0) == 0.0


def test_collapsed_model_scores_chance_not_perfect():
    """The old eval put the answer first and used argmax: a model embedding everything
    identically scored 100%. Tie-aware scoring gives chance."""
    same = F.normalize(torch.ones(4), dim=0)
    rows = [mc_row(f"img{i}", "A", ["B", "C", "D"]) for i in range(10)]
    result = multiple_choice(FakeEmbedder({}, default=same), rows)
    assert abs(result["accuracy"] - 0.25) < 1e-9
    assert abs(result["chance"] - 0.25) < 1e-9


def test_option_identical_to_answer_is_dropped():
    e = torch.eye(3)
    emb = FakeEmbedder({"img": e[0], "yes": e[0], "no": e[1]})
    result = multiple_choice(emb, [mc_row("img", "yes", ["yes", "no", "no"])])
    assert result["accuracy"] == 1.0 and abs(result["chance"] - 0.5) < 1e-9


def test_classification_reports_coverage_of_out_of_pool_labels():
    e = torch.eye(3)
    rows = [{"query": {"text": "", "image": f"i{k}"}, "instruction": "", "positive": {"text": lab}}
            for k, lab in enumerate(["a", "b", "zzz"])]
    emb = FakeEmbedder({"i0": e[0], "i1": e[1], "i2": e[2], "a": e[0], "b": e[1]})
    result = classification(emb, rows, class_pool=["a", "b"])
    assert result["num_classes"] == 2
    assert abs(result["top1"] - 2 / 3) < 1e-9         # the out-of-pool row counts as wrong...
    assert abs(result["coverage"] - 2 / 3) < 1e-9      # ...and is visible in coverage


def test_retrieval_counts_shared_label_as_correct_for_every_row():
    """Two images of the same species both retrieving that species' label are both hits
    (the old metric scored the second as a miss)."""
    e = torch.eye(3)
    rows = [{"query": {"text": "", "image": f"i{k}"}, "instruction": "", "positive": {"text": lab}}
            for k, lab in enumerate(["robin", "robin", "jay"])]
    emb = FakeEmbedder({"i0": e[0], "i1": e[0], "i2": e[1], "robin": e[0], "jay": e[1]})
    result = retrieval(emb, rows)
    assert result["num_candidates"] == 2 and result["recall@1"] == 1.0


def test_unreadable_queries_are_skipped_and_counted():
    nan = torch.full((3,), float("nan"))
    e = torch.eye(3)
    emb = FakeEmbedder({"ok": e[0], "A": e[0], "B": e[1], "bad": nan})
    result = multiple_choice(emb, [mc_row("ok", "A", ["B"]), mc_row("bad", "A", ["B"])])
    assert result["n"] == 1 and result["skipped"] == 1 and result["accuracy"] == 1.0


def test_records_describe_each_prediction():
    e = torch.eye(3)
    emb = FakeEmbedder({"img": torch.tensor([0.5, 1.0, 0.0]), "A": e[0], "B": e[1], "C": e[2]})
    records = []
    result = multiple_choice(emb, [mc_row("img", "A", ["B", "C"])], records=records)
    (r,) = records
    assert result["accuracy"] == 0.0
    assert r["gold"] == "A" and r["pred"] == "B" and r["correct"] == 0.0
    assert r["gold_rank"] == 2 and r["margin"] < 0 and r["ranked"][0][0] == "B"

    rows = [{"query": {"text": "", "image": f"i{k}"}, "instruction": "", "positive": {"text": lab}}
            for k, lab in enumerate(["a", "b"])]
    emb = FakeEmbedder({"i0": e[0], "i1": e[0], "a": e[0], "b": e[1]})
    records = []
    classification(emb, rows, records=records)
    assert [(r["gold"], r["pred"], r["image"]) for r in records] == [("a", "a", "i0"), ("b", "a", "i1")]
    assert records[0]["correct"] == 1.0 and records[1]["gold_rank"] == 2
