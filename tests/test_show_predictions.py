"""tools/show_predictions.py on a tiny hand-made predictions file (no model needed)."""
import json

import show_predictions


def test_report_and_html(tmp_path, capsys):
    pred = tmp_path / "run" / "predictions"
    pred.mkdir(parents=True)
    rows = [{"question": "", "image": f"x{i}.jpg", "gold": f"c{i % 25}", "pred": f"c{(i + (i % 2)) % 25}",
             "correct": float(i % 2 == 0), "gold_rank": 1 + i % 2, "margin": 0.1 if i % 2 == 0 else -0.05,
             "ranked": [[f"c{i % 25}", 0.5], ["c0", 0.4]]} for i in range(50)]
    (pred / "cub.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    files = show_predictions.find_files(str(tmp_path / "run"))
    summary = show_predictions.summarize("cub", show_predictions.load(files["cub"]), 2)
    assert summary["wrong"] == 25 and summary["in_top5"] == 25 and summary["confusions"]
    out = tmp_path / "r.html"
    show_predictions.write_html(str(out), "run", {"cub": (rows, summary)}, 10)
    assert "Mistakes (most confident first)" in out.read_text() and "c3" in out.read_text()
    assert "accuracy 0.500" in capsys.readouterr().out
