"""Readable report of what a model predicted vs the right answer, from evaluate.py's
predictions/<task>.jsonl files.

    python tools/show_predictions.py runs/mentor_mixed__cub_full                 # every task, in the terminal
    python tools/show_predictions.py runs/mentor_mixed__cub_full --task cub200_val --examples 20
    python tools/show_predictions.py runs/mentor_mixed__cub_full --html report.html   # gallery with images
    python tools/show_predictions.py runs/mentor_mixed__cub_full --compare runs/mentor_mixed
        # rows one model gets right and the other wrong (what fine-tuning fixed / broke)

A path can be a run or checkpoint dir (reads its predictions/), a predictions/ dir, or .jsonl files.

Per task: accuracy, how many mistakes still had the right answer in the top 5, the most
confident mistakes and the near misses (margin = gold similarity minus the best wrong one), and
for classification the most common confusions (gold -> predicted) and the worst classes.
--html writes one self-contained page (thumbnails embedded) to open on any machine.
"""
import argparse
import base64
import collections
import glob
import html
import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def find_files(path: str) -> dict[str, str]:
    """{task name: predictions file} for a run/checkpoint dir, a predictions dir or a file."""
    if os.path.isfile(path):
        return {os.path.splitext(os.path.basename(path))[0]: path}
    for d in (os.path.join(path, "predictions"), path):
        files = sorted(glob.glob(os.path.join(d, "*.jsonl")))
        if files:
            return {os.path.splitext(os.path.basename(f))[0]: f for f in files}
    newest = sorted(glob.glob(os.path.join(path, "checkpoint-*", "predictions")),
                    key=lambda p: int(p.split("checkpoint-")[1].split(os.sep)[0]))
    return find_files(os.path.dirname(newest[-1])) if newest else {}


def load(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def scored(records: list[dict]) -> list[dict]:
    return [r for r in records if not r.get("skipped")]


def is_classification(records: list[dict]) -> bool:
    return not any(r.get("question") for r in records) and len({r.get("gold") for r in records}) > 20


def fmt_row(r: dict) -> str:
    mark = "OK" if r["correct"] >= 1 else ("~ " if r["correct"] > 0 else "X ")
    q = f'  Q: {r["question"][:90]}\n' if r.get("question") else ""
    top = ", ".join(f"{name} {score:.3f}" for name, score in r.get("ranked", [])[:5])
    return (f"{mark} {os.path.basename(r.get('image') or '-')}\n{q}"
            f"     predicted: {r['pred']}\n     gold:      {r.get('gold')}   (rank {r.get('gold_rank')}, "
            f"margin {r.get('margin', 0):+.3f})\n     top: {top}")


def summarize(name: str, records: list[dict], examples: int) -> dict:
    rows = scored(records)
    wrong = [r for r in rows if r["correct"] < 1]
    n = max(len(rows), 1)
    acc = sum(r["correct"] for r in rows) / n
    in_top5 = sum(1 for r in wrong if r.get("gold_rank") and r["gold_rank"] <= 5)
    print(f"\n=== {name}: accuracy {acc:.3f}  ({len(rows) - len(wrong)}/{len(rows)} right"
          f"{f', {len(records) - len(rows)} skipped' if len(records) > len(rows) else ''})")
    if wrong:
        print(f"    {len(wrong)} mistakes; {in_top5} of them ({in_top5 / len(wrong):.0%}) still had the "
              f"right answer in the top 5")
    confusions, worst = [], []
    if is_classification(rows):
        confusions = collections.Counter((r.get("gold"), r["pred"]) for r in wrong).most_common(10)
        print("\n    most common confusions (gold -> predicted):")
        for (gold, pred), count in confusions:
            print(f"      {count:4d}x  {gold}  ->  {pred}")
        per_class = collections.defaultdict(list)
        for r in rows:
            per_class[r.get("gold")].append(r["correct"])
        worst = sorted(((sum(v) / len(v), len(v), k) for k, v in per_class.items()), key=lambda t: t[0])[:10]
        print("\n    worst classes:")
        for a, count, label in worst:
            print(f"      {a:.2f}  (n={count:3d})  {label}")
    else:
        by_answer = collections.Counter(r["pred"] for r in wrong).most_common(5)
        if by_answer:
            print("    most frequent wrong picks: " + ", ".join(f"{a!r} x{c}" for a, c in by_answer))
    confident = sorted(wrong, key=lambda r: r.get("margin", 0))[:examples]
    near = sorted((r for r in wrong if r.get("margin") is not None), key=lambda r: -r["margin"])[:examples]
    if confident:
        print(f"\n    most confident mistakes:")
        for r in confident:
            print("    " + fmt_row(r).replace("\n", "\n    "))
        print(f"\n    near misses (almost right):")
        for r in near:
            print("    " + fmt_row(r).replace("\n", "\n    "))
    return {"accuracy": acc, "n": len(rows), "wrong": len(wrong), "in_top5": in_top5,
            "confusions": confusions, "worst": worst, "confident": confident}


def compare(name: str, a: list[dict], b: list[dict], label_a: str, label_b: str, examples: int) -> list:
    """Rows (aligned by order: same eval file and --limit) where exactly one model is right."""
    if len(a) != len(b) or any(x.get("image") != y.get("image") for x, y in zip(a, b)):
        print(f"\n=== {name}: rows don't line up (different file or --limit); skipped")
        return []
    pairs = [(x, y) for x, y in zip(a, b) if not x.get("skipped") and not y.get("skipped")]
    fixed = [(x, y) for x, y in pairs if x["correct"] >= 1 and y["correct"] < 1]
    broke = [(x, y) for x, y in pairs if x["correct"] < 1 and y["correct"] >= 1]
    print(f"\n=== {name}: {label_a} right & {label_b} wrong: {len(fixed)}   "
          f"{label_b} right & {label_a} wrong: {len(broke)}   (of {len(pairs)})")
    for title, group in ((f"{label_a} right, {label_b} wrong", fixed), (f"{label_b} right, {label_a} wrong", broke)):
        for x, y in group[:examples]:
            q = f' "{x["question"][:70]}"' if x.get("question") else ""
            print(f"    [{title}] {os.path.basename(x.get('image') or '-')}{q}\n"
                  f"        gold {x.get('gold')!r}: {label_a} said {x['pred']!r}, {label_b} said {y['pred']!r}")
    return fixed + broke


# --- HTML ------------------------------------------------------------------------------------

def thumbnail(path: str | None, size: int = 220) -> str:
    if not path:
        return ""
    for candidate in (path, os.path.join(ROOT, path)):
        if os.path.exists(candidate):
            break
    else:
        return ""
    try:
        from PIL import Image
        image = Image.open(candidate).convert("RGB")
        image.thumbnail((size, size))
        buf = io.BytesIO()
        image.save(buf, "JPEG", quality=80)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def card(r: dict) -> str:
    e = html.escape
    status = "right" if r["correct"] >= 1 else "wrong"
    src = thumbnail(r.get("image"))
    img = f'<img src="{src}" alt="">' if src else '<div class="noimg">no image</div>'
    best = max((s for _, s in r.get("ranked", [])), default=1) or 1
    low = min((s for _, s in r.get("ranked", [])), default=0)
    span = (best - low) or 1
    bars = "".join(
        f'<li class="{"gold" if name == r.get("gold") else ""}"><span class="bar" '
        f'style="width:{25 + 75 * (s - low) / span:.0f}%"></span><span class="lab">{e(name)}</span>'
        f'<span class="val">{s:.3f}</span></li>' for name, s in r.get("ranked", []))
    q = f'<p class="q">{e(r["question"])}</p>' if r.get("question") else ""
    return (f'<div class="card {status}">{img}<div class="body">{q}'
            f'<p><b>Predicted</b> <span class="pill {status}">{e(str(r["pred"]))}</span></p>'
            f'<p><b>Gold</b> <span class="pill gold">{e(str(r.get("gold")))}</span> '
            f'<small>rank {r.get("gold_rank")} · margin {r.get("margin", 0):+.3f}</small></p>'
            f'<ol class="ranked">{bars}</ol></div></div>')


def write_html(out: str, title: str, tasks: dict[str, tuple[list, dict]], per_task: int) -> None:
    e = html.escape
    sections = []
    for name, (records, summary) in tasks.items():
        rows = scored(records)
        wrong = sorted((r for r in rows if r["correct"] < 1), key=lambda r: r.get("margin", 0))[:per_task]
        right = [r for r in rows if r["correct"] >= 1][:max(per_task // 3, 6)]
        confusions = "".join(f"<tr><td>{c}</td><td>{e(str(g))}</td><td>{e(str(p))}</td></tr>"
                             for (g, p), c in summary["confusions"])
        conf_table = (f'<table><tr><th>count</th><th>gold</th><th>predicted</th></tr>{confusions}</table>'
                      if confusions else "")
        sections.append(
            f'<section><h2>{e(name)}</h2><p class="stats"><b>{summary["accuracy"]:.1%}</b> accuracy · '
            f'{summary["n"] - summary["wrong"]}/{summary["n"]} right · {summary["wrong"]} wrong, of which '
            f'{summary["in_top5"]} had the answer in the top 5</p>{conf_table}'
            f'<h3>Mistakes (most confident first)</h3><div class="grid">{"".join(card(r) for r in wrong)}</div>'
            f'<h3>Some correct ones</h3><div class="grid">{"".join(card(r) for r in right)}</div></section>')
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(title)}</title><style>
:root{{--bg:#fafaf9;--fg:#1c1917;--muted:#78716c;--card:#fff;--line:#e7e5e4;--ok:#15803d;--bad:#b91c1c;--gold:#a16207;--bar:#dbeafe}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1c1917;--fg:#f5f5f4;--muted:#a8a29e;--card:#292524;--line:#44403c;--ok:#4ade80;--bad:#f87171;--gold:#facc15;--bar:#1e3a5f}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0 auto;max-width:1300px;padding:16px}}
h2{{border-bottom:1px solid var(--line);padding-bottom:4px;margin-top:40px}} .stats{{color:var(--muted)}}
table{{border-collapse:collapse;margin:8px 0}} td,th{{border-bottom:1px solid var(--line);padding:3px 10px;text-align:left}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}}
.card{{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--ok);border-radius:8px;overflow:hidden}}
.card.wrong{{border-left-color:var(--bad)}} .card img{{width:100%;max-height:220px;object-fit:contain;background:#0001}}
.noimg{{padding:8px;color:var(--muted)}} .body{{padding:8px 10px}} .body p{{margin:4px 0}} .q{{font-style:italic}}
.pill{{border-radius:4px;padding:1px 6px;font-weight:600}} .pill.right{{color:var(--ok)}} .pill.wrong{{color:var(--bad)}} .pill.gold{{color:var(--gold)}}
small{{color:var(--muted)}} .ranked{{list-style:none;padding:0;margin:6px 0 0}}
.ranked li{{position:relative;display:flex;justify-content:space-between;padding:1px 4px;font-size:12px}}
.ranked .bar{{position:absolute;left:0;top:0;bottom:0;background:var(--bar);z-index:0;border-radius:3px}}
.ranked .lab,.ranked .val{{position:relative;z-index:1}} .ranked li.gold .lab{{color:var(--gold);font-weight:700}}
</style></head><body><h1>{e(title)}</h1><p class="stats">Bars: similarity of each candidate to the query
(top candidates, best first); gold answer in yellow. Margin = gold similarity − best wrong one.</p>{"".join(sections)}</body></html>"""
    with open(out, "w") as f:
        f.write(page)
    print(f"\nHTML report -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="run/checkpoint dir, predictions dir or .jsonl file")
    ap.add_argument("--task", nargs="*", help="only these tasks (file names without .jsonl)")
    ap.add_argument("--examples", type=int, default=5, help="examples per list in the terminal")
    ap.add_argument("--html", help="also write a self-contained HTML gallery here")
    ap.add_argument("--per-task", type=int, default=60, help="mistakes shown per task in the HTML")
    ap.add_argument("--compare", help="a second run/checkpoint: list rows where only one is right")
    args = ap.parse_args()

    files = find_files(args.path)
    if args.task:
        files = {k: v for k, v in files.items() if k in args.task}
    if not files:
        sys.exit(f"no predictions under {args.path} (re-run its eval with the current evaluate.py)")
    tasks = {}
    for name, path in files.items():
        records = load(path)
        tasks[name] = (records, summarize(name, records, args.examples))
    if args.compare:
        other = find_files(args.compare)
        label_a, label_b = os.path.basename(args.path.rstrip("/")), os.path.basename(args.compare.rstrip("/"))
        for name, (records, _) in tasks.items():
            if name in other:
                compare(name, records, load(other[name]), label_a, label_b, args.examples)
    if args.html:
        write_html(args.html, os.path.basename(args.path.rstrip("/")), tasks, args.per_task)


if __name__ == "__main__":
    main()
