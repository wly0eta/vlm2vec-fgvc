"""Hugging Face Hub sources, streamed so only kept images are ever decoded and saved."""
import ast
from collections.abc import Iterator

from .common import Record, clean_label


def _stream(dataset: str, split: str, config: str | None = None):
    from datasets import load_dataset
    return load_dataset(dataset, config, split=split, streaming=True)


def image_classification(dataset: str, split: str, label_field: str = "label", image_field: str = "image",
                         clean: bool = True) -> Iterator[Record]:
    """Any image + ClassLabel dataset (e.g. Donghyun99/CUB-200-2011). Integer labels are
    mapped to the dataset's class names; `clean` strips CUB-style '001.' prefixes and
    underscores."""
    ds = _stream(dataset, split)
    names = getattr(ds.features.get(label_field), "names", None) if ds.features else None
    for i, ex in enumerate(ds):
        if ex.get(image_field) is None or ex.get(label_field) is None:
            continue
        label = names[ex[label_field]] if names else str(ex[label_field])
        yield Record(uid=f"{split}_{i}", image=ex[image_field], label=clean_label(label) if clean else label)


def inat_chat(dataset: str = "sxj1215/inaturalist", split: str = "train") -> Iterator[Record]:
    """Chat-formatted species data: the assistant message is the species name."""
    for i, ex in enumerate(_stream(dataset, split)):
        if not ex.get("images"):
            continue
        species = next((m["content"] for m in ex["messages"] if m["role"] == "assistant"), "").strip()
        if species:
            yield Record(uid=f"{split}_{i}", image=ex["images"][0], label=species)


def aokvqa(split: str, dataset: str = "HuggingFaceM4/A-OKVQA") -> Iterator[Record]:
    """A-OKVQA multiple choice. Official train/validation splits are disjoint; test is unlabeled."""
    for ex in _stream(dataset, split):
        choices, correct = ex["choices"], ex["correct_choice_idx"]
        if correct is None or not 0 <= correct < len(choices):
            continue
        yield Record(uid=str(ex["question_id"]), image=ex["image"], label=choices[correct],
                     query_text=ex["question"], choices=list(choices))


def mmbench(split: str = "dev", dataset: str = "lmms-lab/MMBench_EN") -> Iterator[Record]:
    """MMBench (dev has answers; test is hidden). Options live in columns A-E; the optional
    hint is prepended to the question.

    Protocol note: official MMBench uses CircularEval (a question counts only if answered
    correctly under every rotation of its options). We score each row once; with tie-aware
    scoring option order cannot matter, so this is per-question accuracy -- comparable
    across our models, not to the official leaderboard.
    """
    for i, ex in enumerate(_stream(dataset, split)):
        letters = [L for L in "ABCDE" if _present(ex.get(L))]
        answer = str(ex.get("answer") or "").strip()
        if ex.get("image") is None or answer not in letters:
            continue
        options = [str(ex[L]) for L in letters]
        hint = ex.get("hint")
        question = f"{hint}\n{ex.get('question', '')}" if _present(hint) else ex.get("question", "")
        yield Record(uid=str(ex.get("index", f"{split}_{i}")), image=ex["image"],
                     label=options[letters.index(answer)], query_text=question, choices=options)


def mmmu(split: str = "validation", dataset: str = "MMMU/MMMU", subjects: list[str] | None = None) -> Iterator[Record]:
    """MMMU multiple-choice questions across all subjects (validation has answers).

    Simplifications, stated so the number isn't over-read: open-ended questions are
    skipped, and so are questions that need more than the first image -- ones whose text
    has a second image, or whose options are images ("<image 3>"): we give the model one image,
    so those are unanswerable and only add noise (~40 of the validation MC questions).
    """
    from datasets import get_dataset_config_names
    for subject in subjects or get_dataset_config_names(dataset):
        for ex in _stream(dataset, split, subject):
            if ex.get("question_type") != "multiple-choice" or ex.get("image_1") is None:
                continue
            try:
                options = [str(o) for o in ast.literal_eval(ex["options"])]
            except (ValueError, SyntaxError):
                continue
            answer = str(ex.get("answer", "")).strip()
            if len(answer) != 1 or not 0 <= ord(answer) - ord("A") < len(options):
                continue
            if ex.get("image_2") is not None or any("<image" in o for o in options):
                continue
            yield Record(uid=str(ex["id"]), image=ex["image_1"], label=options[ord(answer) - ord("A")],
                         query_text=ex.get("question", ""), choices=options)


def _present(value) -> bool:
    return value is not None and str(value).strip() not in ("", "nan", "None")
