import json
import pickle

import numpy as np
import pytest

from vlm2vec.data import Item, JsonlRows, MixedDataset, query_item


def _write_jsonl(path, n, blank_every=0):
    with open(path, "w") as f:
        for i in range(n):
            f.write(json.dumps({"i": i}) + "\n")
            if blank_every and i % blank_every == 0:
                f.write("\n")
    return path


def test_jsonl_rows_random_access_skips_blank_lines(tmp_path):
    rows = JsonlRows(_write_jsonl(tmp_path / "x.jsonl", 50, blank_every=7))
    assert len(rows) == 50
    assert [rows[i]["i"] for i in (0, 49, 13, 7)] == [0, 49, 13, 7]


def test_jsonl_index_is_cached_and_invalidated_on_change(tmp_path):
    path = _write_jsonl(tmp_path / "x.jsonl", 10)
    JsonlRows(path)
    assert (tmp_path / "x.jsonl.offsets.npz").exists()
    _write_jsonl(path, 12)                       # file changed -> stale cache must be rebuilt
    assert len(JsonlRows(path)) == 12


def test_jsonl_rows_pickle_without_open_handle(tmp_path):
    rows = JsonlRows(_write_jsonl(tmp_path / "x.jsonl", 5))
    rows[0]                                      # opens the file
    clone = pickle.loads(pickle.dumps(rows))     # what DataLoader workers do
    assert clone[4]["i"] == 4


def test_single_task_epoch_visits_every_row_exactly_once():
    ds = MixedDataset({"a": (list(range(100)), 1.0)}, seed=3)
    assert sorted(ds[i] for i in range(len(ds))) == list(range(100))


def test_weights_set_each_tasks_share_of_the_epoch():
    small, big = [("s", i) for i in range(30)], [("b", i) for i in range(70)]
    ds = MixedDataset({"small": (small, 1.0), "big": (big, 1.0)}, seed=0)
    drawn = [ds[i] for i in range(len(ds))]
    assert len(drawn) == 100
    from_small = [x for x in drawn if x[0] == "s"]
    assert len(from_small) == 50
    assert set(from_small) == set(small)          # oversampled task: every row appears
    assert len(set(x for x in drawn if x[0] == "b")) == 50   # undersampled: no repeats


def test_epoch_draw_is_deterministic_and_changes_per_epoch():
    rows = list(range(30))
    a = MixedDataset({"x": (rows, 1.0), "y": (list(range(70)), 1.0)}, seed=0)
    b = MixedDataset({"x": (rows, 1.0), "y": (list(range(70)), 1.0)}, seed=0)
    assert np.array_equal(a.index, b.index)       # same on every rank / after resume
    first = a.index.copy()
    a.set_epoch(1)
    assert not np.array_equal(first, a.index)


def test_empty_task_is_rejected():
    with pytest.raises(ValueError):
        MixedDataset({"x": ([], 1.0)})


def test_query_prompt_matches_trained_checkpoints():
    """The exact text existing checkpoints were trained on -- must not drift."""
    row = {"query": {"text": "", "image": "a.jpg"}, "instruction": "Identify the species in the image.",
           "positive": {"text": "Cardinalis cardinalis"}}
    assert query_item(row).prompt() == "Instruct: Identify the species in the image.\nQuery:"
    assert Item("Cardinalis cardinalis").prompt() == "Cardinalis cardinalis"
    no_instruction = {"query": {"text": "q"}, "positive": {"text": "a"}}
    assert query_item(no_instruction).prompt() == "Instruct: Represent the query for retrieval.\nQuery: q"


def test_class_field_overrides_mask_key():
    """Two names of one class must mask each other; without "class", keys stay the text."""
    from vlm2vec.data import mask_key
    crow, corvus = {"text": "American Crow", "class": "c1"}, {"text": "Corvus brachyrhynchos", "class": "c1"}
    assert mask_key(crow, Item(crow["text"])) == mask_key(corvus, Item(corvus["text"]))
    assert mask_key({"text": "Fish Crow"}, Item("Fish Crow")) == Item("Fish Crow").key
