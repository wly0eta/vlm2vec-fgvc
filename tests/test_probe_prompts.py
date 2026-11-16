"""probe_prompts: instruction sweep, template ensembling and name pairing, with a fake embedder."""
import json

import pytest
import torch
import torch.nn.functional as F

from probe_prompts import mean_prototypes, probe_file

CLASSES = {"Albatross": "Diomedea", "Gull": "Larus", "Tern": "Sterna"}
DIRS = {name: F.one_hot(torch.tensor(i), 4).float() for i, name in enumerate(CLASSES)}


class FakeEmbedder:
    """Queries point at their class only under the trained instruction (else at a junk axis).
    Labels: English names point at their class; Latin names only under a template."""

    def __call__(self, items, cache=True):
        out = []
        for item in items:
            if item.instruction is not None:            # a query; image path = class name
                v = DIRS[item.image] if item.instruction == "trained" else F.one_hot(torch.tensor(3), 4).float()
            else:
                v = torch.zeros(4)
                for name, latin in CLASSES.items():
                    if name in item.text or (latin in item.text and item.text != latin):
                        v = DIRS[name]
            out.append(F.normalize(v + 1e-3, dim=0))
        return torch.stack(out)


def write(path, labels):
    with open(path, "w") as f:
        for cls, label in labels:
            f.write(json.dumps({"query": {"text": "", "image": cls}, "instruction": "trained",
                                "positive": {"text": label}, "negatives": []}) + "\n")


def test_probe_sweeps_instructions_templates_and_names(tmp_path):
    rows = [c for c in CLASSES for _ in range(2)]
    write(tmp_path / "en.jsonl", [(c, c) for c in rows])
    write(tmp_path / "la.jsonl", [(c, CLASSES[c]) for c in rows])
    r = probe_file(FakeEmbedder(), str(tmp_path / "en.jsonl"), str(tmp_path / "la.jsonl"),
                   ["trained", "paraphrase"], ["{name}", "a photo of a {name}."], limit=0, seed=0)
    assert r["instructions"] == {"trained": 1.0, "paraphrase": pytest.approx(1 / 3)}   # ties -> 1/k
    assert r["instruction_summary"]["trained"] == 1.0
    assert r["templates"]["{name}"] == 1.0 and r["template_ensemble"] == 1.0
    assert r["names"]["primary"] == 1.0
    assert r["names"]["paired"] == pytest.approx(1 / 3)       # bare Latin names embed to nothing useful
    assert r["names"]["both"] == 1.0


def test_unaligned_pair_is_rejected(tmp_path):
    write(tmp_path / "en.jsonl", [("Gull", "Gull"), ("Tern", "Tern")])
    write(tmp_path / "la.jsonl", [("Tern", "Sterna"), ("Gull", "Larus")])
    with pytest.raises(ValueError):
        probe_file(FakeEmbedder(), str(tmp_path / "en.jsonl"), str(tmp_path / "la.jsonl"),
                   ["trained"], ["{name}"], limit=0, seed=0)


def test_mean_prototypes_are_unit_length():
    p = mean_prototypes([torch.eye(3), F.normalize(torch.ones(3, 3), dim=1)])
    assert torch.allclose(p.norm(dim=1), torch.ones(3))
