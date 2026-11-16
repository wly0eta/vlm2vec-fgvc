import json
import random
from collections import Counter

from vlm2vec.sources.common import NegativeSampler, Record, build, clean_label, stratified_split


def test_clean_label():
    assert clean_label("001.Black_footed_Albatross") == "Black footed Albatross"
    assert clean_label("Cardinalis cardinalis") == "Cardinalis cardinalis"
    assert clean_label("  12 Laysan_Albatross ") == "Laysan Albatross"


def test_negatives_are_distinct_and_never_the_positive():
    sampler = NegativeSampler([f"s{i}" for i in range(50)], random.Random(0))
    for _ in range(200):
        negs = sampler.sample("s7", 3)
        assert len(negs) == 3 == len(set(negs)) and "s7" not in negs


def test_negatives_capped_by_label_count():
    assert sorted(NegativeSampler(["a", "b", "c"], random.Random(0)).sample("a", 10)) == ["b", "c"]


def test_hard_negatives_come_from_the_same_genus():
    labels = ["Setophaga a", "Setophaga b", "Setophaga c"] + [f"Other {i}" for i in range(40)]
    groups = {l: ["genus:Setophaga"] if l.startswith("Setophaga") else [] for l in labels}
    sampler = NegativeSampler(labels, random.Random(0), hard_fraction=0.67, groups=groups)
    negs = sampler.sample("Setophaga a", 3)
    assert {"Setophaga b", "Setophaga c"} <= set(negs)


def test_stratified_split_puts_every_multi_example_label_in_both_splits():
    records = [Record(uid=str(i), image="x", label=f"c{i % 5}") for i in range(40)]
    records.append(Record(uid="solo", image="x", label="singleton"))
    train, val = stratified_split(records, 0.2, random.Random(0))
    assert {r.label for r in val} == {f"c{i}" for i in range(5)}
    assert {r.label for r in train} >= {f"c{i}" for i in range(5)} | {"singleton"}
    assert not {r.uid for r in train} & {r.uid for r in val}


def _read(path):
    return [json.loads(line) for line in open(path)]


def test_build_classification_rows_with_cap(tmp_path):
    records = (Record(uid=str(i), image=f"/data/{i}.jpg", label=f"sp{i % 4}") for i in range(40))
    out = tmp_path / "train.jsonl"
    build(records, out=str(out), image_dir=str(tmp_path / "img"), per_class_cap=3, num_negatives=2)
    rows = _read(out)
    assert len(rows) == 12
    assert max(Counter(r["positive"]["text"] for r in rows).values()) == 3
    for r in rows:
        assert r["query"]["image"].startswith("/data/")     # local images referenced in place
        assert len(r["negatives"]) == 2
        assert r["positive"]["text"] not in {n["text"] for n in r["negatives"]}


def test_build_multiple_choice_uses_its_own_options(tmp_path):
    records = [Record(uid="q1", image="/d/1.jpg", label="two", query_text="How many?",
                      choices=["one", "two", "two", "three"])]
    out = tmp_path / "mc.jsonl"
    build(records, out=str(out), image_dir=str(tmp_path / "img"))
    row = _read(out)[0]
    assert row["query"]["text"] == "How many?"
    assert [n["text"] for n in row["negatives"]] == ["one", "three"]   # answer + duplicates removed


def test_build_val_split_path(tmp_path):
    records = [Record(uid=str(i), image="/d/x.jpg", label=f"c{i % 3}") for i in range(30)]
    build(records, out=str(tmp_path / "x_train.jsonl"), image_dir=str(tmp_path / "img"), val_fraction=0.2)
    assert (tmp_path / "x_val.jsonl").exists()


def test_normalize_common_name_matches_cub_and_inat_spellings():
    from vlm2vec.sources.inat_local import normalize_common_name as n
    assert n("Black footed Albatross") == n("Black-footed Albatross")
    assert n("Brewer Blackbird") == n("Brewer's Blackbird") == n("Brewer’s Blackbird")


def test_relabel_keeps_matched_classes_and_builds_matching_control_set():
    from vlm2vec.sources.common import relabel_rows
    rows = [{"query": {"text": "", "image": f"/i/{i}.jpg"}, "instruction": "x",
             "positive": {"text": lab}, "negatives": []} for i, lab in enumerate(["Crow", "Crow", "Jay", "Dodo"])]
    mapping = {"Crow": "Corvus brachyrhynchos", "Jay": "Cyanocitta cristata"}.get
    latin, stats = relabel_rows(rows, mapping, False, 1, random.Random(0))
    common, _ = relabel_rows(rows, mapping, True, 1, random.Random(0))
    assert stats == {"rows": 3, "rows_before": 4, "classes": 2, "classes_before": 3}
    assert [r["positive"]["text"] for r in latin] == ["Corvus brachyrhynchos"] * 2 + ["Cyanocitta cristata"]
    assert [r["query"]["image"] for r in latin] == [r["query"]["image"] for r in common]   # same images
    assert [r["positive"]["text"] for r in common] == ["Crow", "Crow", "Jay"]
    assert all(r["negatives"][0]["text"] != r["positive"]["text"] for r in latin + common)
