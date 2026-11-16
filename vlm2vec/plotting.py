"""Training-curve figure. Pure rendering: takes series, writes a PNG."""
import os


def _ema(values, alpha=0.3):
    out, prev = [], None
    for v in values:
        prev = v if prev is None else alpha * v + (1 - alpha) * prev
        out.append(prev)
    return out


def render_curves(path: str, steps: list[int], series: dict[str, list], eval_series: dict[str, tuple]) -> None:
    """One panel per training series (smoothed if noisy) plus one for eval accuracy.

    series:      {"loss": [...], "inbatch_acc": [...], "learning_rate": [...], "grad_norm": [...]}
    eval_series: {"aokvqa_val": ([steps], [accuracy]), ...}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [(name, values) for name, values in series.items() if any(v is not None for v in values)]
    if eval_series:
        panels.append(("eval accuracy", None))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 3.8), squeeze=False)
    for ax, (name, values) in zip(axes[0], panels):
        if values is None:
            for label, (xs, ys) in eval_series.items():
                ax.plot(xs, ys, marker="o", linewidth=2, label=label)
            ax.set_ylim(0, 1)
            ax.legend(fontsize=8, loc="lower right")
        else:
            points = [(s, v) for s, v in zip(steps, values) if v is not None]
            xs, ys = zip(*points)
            noisy = name in ("loss", "inbatch_acc")
            ax.plot(xs, ys, linewidth=1, alpha=0.35 if noisy else 1.0)
            if noisy and len(ys) > 1:
                ax.plot(xs, _ema(ys), linewidth=2, label="EMA")
                ax.legend(fontsize=8)
        ax.set_title(name)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)
