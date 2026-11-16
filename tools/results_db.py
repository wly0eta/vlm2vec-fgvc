"""Every result in one SQLite file, so results can be queried instead of read across JSON files.

    python tools/results_db.py build                                   # (re)build runs/results.db
    python tools/results_db.py build --runs-dir ~/Downloads/vlm2vec_checkpoints   # the Mac copy
    python tools/results_db.py runs                                    # what's in it
    python tools/results_db.py table --experiment 07_fgvc_transfer     # one experiment's runs x tasks
    python tools/results_db.py table --runs mentor_aokvqa base7b_aokvqa --tasks cub200_val zs_cars
    python tools/results_db.py sql "SELECT run, value FROM final WHERE task='cub200_val' ORDER BY value DESC LIMIT 5"

The database is DERIVED: `build` rebuilds it from scratch out of the files the runs already write
(eval_results.json, the generative baseline's results.json, paired_vs_*.md, run_config.json,
vlm2vec_config.json, TRAINING_COMPLETE), so it can never disagree with them. Delete it at will.
Experiment membership comes from experiments/<NN_name>/runs.txt (one run name or glob per line).

Tables
  runs(run, base_model, start_model, dataset_config, epochs, learning_rate, global_batch, lora_r,
       temperature, max_pixels, finished, path)          start_model etc. only for runs whose
                                                         train.py wrote run_config.json (Nov 16+)
  results(run, checkpoint, epoch, task, kind, value, top5, n, chance, num_classes, file, evaluated_at)
       checkpoint '' = the run's final model; value = the headline metric (MC accuracy,
       classification top-1, retrieval recall@1, generative top-1)
  paired(run_a, run_b, task, n, a, b, diff, ci_lo, ci_hi, only_a, only_b, p, significant, source)
  run_experiments(run, experiment)
View
  final: one row per (run, task), from the final model, else the newest evaluated checkpoint
"""
import argparse
import fnmatch
import glob
import json
import os
import re
import sqlite3
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEADLINE = {"multiple_choice": "accuracy", "classification": "top1", "retrieval": "recall@1"}

SCHEMA = """
CREATE TABLE runs (run TEXT PRIMARY KEY, base_model TEXT, start_model TEXT, dataset_config TEXT,
  epochs REAL, learning_rate REAL, global_batch INTEGER, lora_r INTEGER, temperature REAL,
  max_pixels INTEGER, finished INTEGER, path TEXT);
CREATE TABLE results (run TEXT, checkpoint TEXT, epoch REAL, task TEXT, kind TEXT, value REAL,
  top5 REAL, n INTEGER, chance REAL, num_classes INTEGER, file TEXT, evaluated_at TEXT,
  PRIMARY KEY (run, checkpoint, task));
CREATE TABLE paired (run_a TEXT, run_b TEXT, task TEXT, n INTEGER, a REAL, b REAL, diff REAL,
  ci_lo REAL, ci_hi REAL, only_a INTEGER, only_b INTEGER, p REAL, significant INTEGER, source TEXT,
  PRIMARY KEY (run_a, run_b, task));
CREATE TABLE run_experiments (run TEXT, experiment TEXT, PRIMARY KEY (run, experiment));
CREATE VIEW final AS
  SELECT r.* FROM results r
  WHERE r.checkpoint = (
    SELECT CASE WHEN SUM(x.checkpoint = '') > 0 THEN ''
                ELSE (SELECT y.checkpoint FROM results y WHERE y.run = r.run AND y.task = r.task
                      ORDER BY CAST(SUBSTR(y.checkpoint, 12) AS INTEGER) DESC LIMIT 1) END
    FROM results x WHERE x.run = r.run AND x.task = r.task);
"""


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def split_run(rel: str):
    """'q2b_mixed/checkpoint-1669' -> ('q2b_mixed', 'checkpoint-1669'); '_sweep/gb128' -> ('_sweep/gb128', '')."""
    m = re.match(r"(.*)/(checkpoint-\d+)$", rel)
    return (m.group(1), m.group(2)) if m else (rel, "")


def ingest_results(db, root):
    n = 0
    for path in glob.glob(os.path.join(root, "**", "eval_results.json"), recursive=True):
        rel = os.path.relpath(os.path.dirname(path), root)
        if rel.startswith("old__") or "/predictions" in rel:
            continue
        run, ckpt = split_run(rel)
        data = load_json(path) or {}
        meta = data.get("_meta", {})
        for task, r in data.items():
            if task.startswith("_") or not isinstance(r, dict) or r.get("task") not in HEADLINE:
                continue
            db.execute("INSERT OR REPLACE INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                       (run, ckpt, meta.get("epoch"), task, r["task"], r.get(HEADLINE[r["task"]]),
                        r.get("top5"), r.get("n"), r.get("chance"), r.get("num_classes"), r.get("file"),
                        meta.get("evaluated_at")))
            n += 1
    # generative baseline (tools/zero_shot_generative.py): results.json, top-1 = rank_sum
    for path in glob.glob(os.path.join(root, "gen_*", "results.json")):
        run = os.path.relpath(os.path.dirname(path), root)
        for task, r in (load_json(path) or {}).items():
            if isinstance(r, dict) and "rank_sum" in r:
                db.execute("INSERT OR REPLACE INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                           (run, "", None, task, "generative", r["rank_sum"], None, r.get("n"),
                            r.get("chance"), r.get("classes"), r.get("file"), None))
                n += 1
    return n


def ingest_runs(db, root):
    runs = {row[0] for row in db.execute("SELECT DISTINCT run FROM results")}
    for run in sorted(runs):
        d = os.path.join(root, run)
        cfg = load_json(os.path.join(d, "run_config.json")) or {}
        contract = load_json(os.path.join(d, "vlm2vec_config.json")) or {}
        done = load_json(os.path.join(d, "TRAINING_COMPLETE"))
        epoch = (done or {}).get("epoch") or next(iter(db.execute(
            "SELECT MAX(epoch) FROM results WHERE run=?", (run,))))[0]
        db.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                   (run, contract.get("base_model") or cfg.get("model_name"), cfg.get("model_name"),
                    cfg.get("dataset_config"), cfg.get("num_train_epochs", epoch), cfg.get("learning_rate"),
                    cfg.get("global_batch"), cfg.get("lora_r"), cfg.get("temperature", contract.get("temperature")),
                    cfg.get("max_pixels", contract.get("max_pixels")),
                    int(os.path.exists(os.path.join(d, "TRAINING_COMPLETE"))), d))


def ingest_paired(db, root):
    """compare_paired.py's markdown: 'A = runs/x', 'B = runs/y', then one table row per task."""
    num = lambda s: float(s) if s not in ("", "nan") else None
    n = 0
    for path in glob.glob(os.path.join(root, "**", "paired_vs_*.md"), recursive=True) + \
            glob.glob(os.path.join(root, "paired_*.md")):
        a = b = None
        for line in open(path):
            if line.startswith("A = "):
                a = os.path.relpath(line[4:].strip(), "runs") if line[4:].startswith("runs/") else os.path.basename(line[4:].strip())
            elif line.startswith("B = "):
                b = os.path.relpath(line[4:].strip(), "runs") if line[4:].startswith("runs/") else os.path.basename(line[4:].strip())
            m = re.match(r"\| (\S+) \| (\d+) \| ([\d.]+) \| ([\d.]+) \| ([+-][\d.]+) \(([+-][\d.]+), ([+-][\d.]+)\) "
                         r"\| (\d+) \| (\d+) \| ([\d.e+-]+)( \*)? \|", line)
            if m and a and b:
                g = m.groups()
                db.execute("INSERT OR REPLACE INTO paired VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           (a, b, g[0], int(g[1]), num(g[2]), num(g[3]), num(g[4]), num(g[5]), num(g[6]),
                            int(g[7]), int(g[8]), num(g[9]), int(bool(g[10])), os.path.relpath(path, root)))
                n += 1
    return n


def ingest_experiments(db):
    runs = [r[0] for r in db.execute("SELECT run FROM runs")]
    for txt in glob.glob(os.path.join(REPO, "experiments", "*", "runs.txt")):
        exp = os.path.basename(os.path.dirname(txt))
        for pattern in (l.strip() for l in open(txt)):
            if pattern and not pattern.startswith("#"):
                for run in fnmatch.filter(runs, pattern):
                    db.execute("INSERT OR IGNORE INTO run_experiments VALUES (?,?)", (run, exp))


def build(args):
    if os.path.exists(args.db):
        os.remove(args.db)
    db = sqlite3.connect(args.db)
    db.executescript(SCHEMA)
    n_res = ingest_results(db, args.runs_dir)
    ingest_runs(db, args.runs_dir)
    n_pair = ingest_paired(db, args.runs_dir)
    ingest_experiments(db)
    db.commit()
    n_runs = next(iter(db.execute("SELECT COUNT(*) FROM runs")))[0]
    print(f"{args.db}: {n_runs} runs, {n_res} results, {n_pair} paired comparisons")


def markdown(cur):
    cols = [c[0] for c in cur.description]
    rows = cur.fetchall()
    fmt = lambda v: ("–" if v is None else str(v) if not isinstance(v, float)
                     else f"{v:.1e}" if 0 < abs(v) < 0.001 else f"{v:.3f}")
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def table(db, args):
    if args.experiment:
        runs = [r[0] for r in db.execute("SELECT run FROM run_experiments WHERE experiment LIKE ? ORDER BY run",
                                         (args.experiment + "%",))]
    else:
        runs = args.runs or [r[0] for r in db.execute("SELECT run FROM runs ORDER BY run")]
    if not runs:
        sys.exit("no matching runs (python tools/results_db.py runs lists them)")
    q = ",".join("?" * len(runs))
    tasks = args.tasks or [r[0] for r in db.execute(
        f"SELECT task FROM final WHERE run IN ({q}) GROUP BY task ORDER BY MIN(rowid)", runs)]
    vals = {(r, t): v for r, t, v in db.execute(f"SELECT run, task, value FROM final WHERE run IN ({q})", runs)}
    print("| run | " + " | ".join(tasks) + " |")
    print("|" + "---|" * (len(tasks) + 1))
    for r in runs:
        print(f"| {r} | " + " | ".join("–" if vals.get((r, t)) is None else f"{vals[(r, t)]:.3f}" for t in tasks) + " |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default=os.path.join(REPO, "runs"))
    ap.add_argument("--db", help="default: <runs-dir>/results.db")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("runs")
    t = sub.add_parser("table")
    t.add_argument("--experiment", help="e.g. 07 or 07_fgvc_transfer (prefix match)")
    t.add_argument("--runs", nargs="+")
    t.add_argument("--tasks", nargs="+")
    s = sub.add_parser("sql")
    s.add_argument("query")
    args = ap.parse_args()
    args.runs_dir = os.path.expanduser(args.runs_dir)
    args.db = args.db or os.path.join(args.runs_dir, "results.db")

    if args.cmd == "build":
        return build(args)
    if not os.path.exists(args.db):
        sys.exit(f"{args.db} not found: python tools/results_db.py build" +
                 (f" --runs-dir {args.runs_dir}" if args.runs_dir != os.path.join(REPO, "runs") else ""))
    db = sqlite3.connect(args.db)
    if args.cmd == "runs":
        print(markdown(db.execute(
            "SELECT r.run, GROUP_CONCAT(DISTINCT e.experiment) AS experiments, r.base_model, r.epochs, "
            "r.finished, (SELECT COUNT(*) FROM final f WHERE f.run = r.run) AS tasks "
            "FROM runs r LEFT JOIN run_experiments e ON e.run = r.run GROUP BY r.run ORDER BY r.run")))
    elif args.cmd == "table":
        table(db, args)
    elif args.cmd == "sql":
        print(markdown(db.execute(args.query)))


if __name__ == "__main__":
    main()
