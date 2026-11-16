"""Trainer callbacks: when to checkpoint, when to stop, and live metrics.

Every decision that changes control flow (save now? stop now?) is made on rank 0 and
broadcast, so all GPUs enter checkpointing together. If ranks decided independently --
each reading its own clock -- one could start saving while another moved on to the next
step's collectives, and the job would hang.
"""
import csv
import json
import logging
import os
import random
import shutil
import time

from transformers import TrainerCallback

from .distributed import broadcast_flag, is_main_process

log = logging.getLogger(__name__)


class TimedCheckpointCallback(TrainerCallback):
    """Checkpoint every N minutes of wall time.

    Replaces hand-tuned save_steps. The same 30 minutes suits a 2B model at 5 s/step and a
    7B one at 17 s/step, and survives changing the GPU count (which changes steps/epoch).
    We hit the failure this prevents: a 7B run whose first save_steps checkpoint came after
    the 8-hour wall, so it never saved and restarted from step 0 on every requeue.
    """

    def __init__(self, minutes: float):
        self.interval = minutes * 60
        self.last_save = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        due = self.interval > 0 and is_main_process() and time.monotonic() - self.last_save >= self.interval
        if broadcast_flag(due):
            control.should_save = True

    def on_save(self, args, state, control, **kwargs):
        self.last_save = time.monotonic()


class PruneCheckpointsCallback(TrainerCallback):
    """Keep every epoch-boundary checkpoint plus only the newest other one.

    Epoch checkpoints are what you evaluate (one point per epoch on the learning curve);
    the timed / stop-file ones exist only so a run can resume, and only the newest is ever
    needed for that. HF's save_total_limit can't express this -- it rotates by age, so it
    would delete the early epoch checkpoints first.
    """

    def on_save(self, args, state, control, **kwargs):
        if not is_main_process():
            return
        newest = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        for name in os.listdir(args.output_dir):
            path = os.path.join(args.output_dir, name)
            if name.startswith("checkpoint-") and path != newest and not is_epoch_checkpoint(path):
                shutil.rmtree(path, ignore_errors=True)


def checkpoint_epoch(path: str) -> float | None:
    """Epoch a checkpoint was saved at: checkpoint-N dirs record it in trainer_state.json,
    a finished run's top level in its TRAINING_COMPLETE marker."""
    for name in ("trainer_state.json", "TRAINING_COMPLETE"):
        try:
            with open(os.path.join(path, name)) as f:
                return float(json.load(f)["epoch"])
        except (OSError, KeyError, ValueError, TypeError):
            continue
    return None


def is_epoch_checkpoint(path: str) -> bool:
    epoch = checkpoint_epoch(path)
    return epoch is not None and epoch > 0 and abs(epoch - round(epoch)) < 1e-6


class StopFileCallback(TrainerCallback):
    """Save and stop when `path` exists.

    The SLURM script creates the file a few minutes before the job's time limit. We save
    a checkpoint at the end of the current step and exit; the script then requeues the
    job, which resumes from exactly that step. No work is lost at the wall.
    """

    def __init__(self, path: str):
        self.path = path
        self.stopped = False

    def on_step_end(self, args, state, control, **kwargs):
        if broadcast_flag(is_main_process() and os.path.exists(self.path)):
            log.info("stop file %s found at step %d: saving and exiting for requeue", self.path, state.global_step)
            control.should_save = True
            control.should_training_stop = True
            self.stopped = True


class LiveMetricsCallback(TrainerCallback):
    """Rank 0: append every log line to <output_dir>/plots/metrics.csv, optionally score a
    small eval subset every N steps, and redraw training_curves.png.

    Files are per run (the old version wrote ./plots for every job, so parallel jobs in one
    directory overwrote and then reloaded each other's history). On resume the CSV history
    is reloaded and points past the resumed step are dropped.
    """

    TRAIN_KEYS = ("loss", "inbatch_acc", "learning_rate", "grad_norm")

    def __init__(self, output_dir: str, embedder_factory=None, eval_sets: dict | None = None,
                 every_steps: int = 0):
        self.dir = os.path.join(output_dir, "plots")
        self.embedder_factory = embedder_factory   # model -> Embedder (only called on rank 0)
        self.eval_sets = eval_sets or {}           # name -> (metric_fn, rows)
        self.every_steps = every_steps
        self.train_rows: list[dict] = []
        self.eval_rows: list[dict] = []

    def on_train_begin(self, args, state, control, **kwargs):
        if not is_main_process():
            return
        os.makedirs(self.dir, exist_ok=True)
        self.train_rows = self._reload("metrics.csv", state.global_step)
        self.eval_rows = self._reload("eval_metrics.csv", state.global_step)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not is_main_process() or not logs or "loss" not in logs:
            return
        self.train_rows.append({"step": state.global_step, **{k: logs.get(k) for k in self.TRAIN_KEYS}})
        self._write("metrics.csv", self.train_rows, ["step", *self.TRAIN_KEYS])
        self._render()

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if (not is_main_process() or not self.every_steps or not self.eval_sets
                or state.global_step % self.every_steps):
            return
        was_training = model.training
        model.eval()
        try:
            embed = self.embedder_factory(model)
            for name, (metric_fn, rows) in self.eval_sets.items():
                result = metric_fn(embed, rows)
                score = result.get("accuracy", result.get("top1"))
                self.eval_rows.append({"step": state.global_step, "name": name, "score": score})
                log.info("live eval step %d %s: %.4f", state.global_step, name, score)
        finally:
            model.train(was_training)
        self._write("eval_metrics.csv", self.eval_rows, ["step", "name", "score"])
        self._render()

    # --- files --------------------------------------------------------------------------

    def _reload(self, filename: str, resumed_step: int) -> list[dict]:
        path = os.path.join(self.dir, filename)
        if not os.path.exists(path):
            return []
        with open(path) as f:
            rows = [r for r in csv.DictReader(f) if int(float(r["step"])) <= resumed_step]
        for r in rows:
            for k, v in r.items():
                if k != "name":
                    r[k] = float(v) if v not in ("", "None") else None
        return rows

    def _write(self, filename: str, rows: list[dict], columns: list[str]) -> None:
        with open(os.path.join(self.dir, filename), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def _render(self) -> None:
        from .plotting import render_curves
        steps = [r["step"] for r in self.train_rows]
        series = {k: [r.get(k) for r in self.train_rows] for k in self.TRAIN_KEYS}
        evals: dict[str, tuple] = {}
        for r in self.eval_rows:
            xs, ys = evals.setdefault(r["name"], ([], []))
            xs.append(r["step"])
            ys.append(r["score"])
        try:
            render_curves(os.path.join(self.dir, "training_curves.png"), steps, series, evals)
        except Exception as err:   # plotting must never take down training
            log.warning("plot skipped: %s", err)


def sample_rows(rows: list[dict], n: int, seed: int = 0) -> list[dict]:
    """A seeded random subset. (The old live eval took the first N rows; eval files are
    often sorted by class, so that meant a handful of species and an inflated score.)"""
    if n <= 0 or n >= len(rows):
        return rows
    return random.Random(seed).sample(rows, n)
