"""Training speed from train logs: seconds per step and images per second, side by side.

    python tools/compare_speed.py LOG [LOG ...] [--batch 256]
    python tools/compare_speed.py ~/Downloads/vlm2vec_checkpoints/mentor_inat/logs/train-*65601067.out \\
        runs/mentor_inat_mini/logs/train-*.log

Reads the progress bar lines ("3240/20990 [2:06:25<...]") and fits steps against elapsed time,
skipping the first 20 steps of each attempt (model loading, warm-up, first compile). The fit
includes everything a real run pays for -- live evals, checkpoint saves, data loading -- so it
is wall-clock throughput; "median" is the typical step without those spikes. --batch is the
global batch (examples per step; 256 for the iNat runs).
"""
import argparse
import re
import statistics

BAR = re.compile(r"\|\s*(\d+)/(\d+) \[(?:(\d+):)?(\d+):(\d+)<")
GPU = re.compile(r"NVIDIA [A-Z0-9][\w -]*?(?=,|\s{2}|$)")


def parse(path: str, total: int | None = None) -> tuple[list[tuple[int, float]], str]:
    """[(step, seconds)] of the longest uninterrupted stretch of the training progress bar, and the
    GPU name. Other bars (weight loading, live evals) have different totals and are ignored; a
    resume restarts the bar's clock, which splits the log into attempts."""
    text = open(path, errors="replace").read().replace("\r", "\n")
    bars = [(int(m[1]), int(m[2]), int(m[3] or 0) * 3600 + int(m[4]) * 60 + int(m[5]))
            for m in map(BAR.search, text.splitlines()) if m]
    # the training bar is the longest one, except in short benchmark runs where weight loading's
    # bar is longer -- pass the run's step count then
    total = total or max((b[1] for b in bars), default=0)
    attempts, current = [], []
    for step, t, elapsed in bars:
        if t != total:
            continue
        if current and (elapsed < current[-1][1] or step < current[-1][0]):
            attempts.append(current)
            current = []
        current.append((step, elapsed))
    attempts.append(current)
    gpu = GPU.search(text)
    return max(attempts, key=len), gpu[0].strip() if gpu else "?"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--batch", type=int, default=256, help="global batch (examples per step)")
    ap.add_argument("--gpus", nargs="*", type=int, help="GPU count per log (default: guessed from the log)")
    ap.add_argument("--total", type=int, help="the training bar's step count (needed for short runs)")
    ap.add_argument("--skip", type=int, default=20, help="warm-up steps ignored at the start of each attempt")
    args = ap.parse_args()
    print(f"| log | GPU | steps measured | s/step (wall) | s/step (median) | images/s | 1 epoch of train_mini |")
    print("|---|---|---|---|---|---|---|")
    for i, path in enumerate(args.logs):
        points, gpu = parse(path, args.total)
        points = [p for p in points if p[0] >= points[0][0] + args.skip] if points else []
        if len(points) < 3 or points[-1][1] == points[0][1]:
            print(f"| {path} | {gpu} | too few steps yet | | | | |")
            continue
        (s0, t0), (s1, t1) = points[0], points[-1]
        wall = (t1 - t0) / (s1 - s0)
        gaps = [(b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(points, points[1:]) if b[0] > a[0]]
        median = statistics.median(gaps)
        gpus = args.gpus[i] if args.gpus and i < len(args.gpus) else None
        label = f"{gpus}x {gpu}" if gpus else gpu
        print(f"| {path.split('/')[-1]} | {label} | {s1 - s0} | {wall:.2f} | {median:.2f} | "
              f"{args.batch / wall:.1f} | {500_000 / args.batch * wall / 3600:.1f} h |")


if __name__ == "__main__":
    main()
