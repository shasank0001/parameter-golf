"""Generate the README figures from archived metadata and validation logs.

Run from any working directory with Python and matplotlib installed:
    python docs/assets/plot_results.py

No GPU, training libraries, or local-only experiment files are required.
"""
from pathlib import Path
import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RECORDS = ROOT / "records" / "track_non_record_16mb"
RUNS = [
    ("2026-04-04_Full80_Recurrence_StepEmbed_Seq1536_20k", "Recurrence + step embeddings", "D", "#136c66", (12, -25)),
    ("2026-04-03_15L_HybridTail_Seq1536_20k_1x5090", "Ternary + 5-bit tail", "o", "#435bc1", (-155, 14)),
    ("2026-03-27_15L_Ternary_MLPOnly_20k", "Ternary MLP weights", "o", "#79629b", (12, 10)),
]
BACKGROUND = "#fafbf9"
INK = "#25343b"
MUTED = "#53616a"


def metadata(folder):
    return json.loads((RECORDS / folder / "submission.json").read_text())


def style_axis(ax):
    ax.set_facecolor(BACKGROUND)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / name, dpi=160, facecolor=BACKGROUND)
    plt.close(fig)


def plot_results():
    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor=BACKGROUND)
    style_axis(ax)
    for folder, label, marker, color, offset in RUNS:
        data = metadata(folder)
        x, y = data["bytes_total"] / 1_000_000, data["val_bpb"]
        ax.scatter(x, y, s=105, marker=marker, color=color, zorder=3)
        ax.annotate(f"{label}\n{y:.5f} bpb · {x:.2f} MB", (x, y),
                    xytext=offset, textcoords="offset points", fontsize=10,
                    color=INK, linespacing=1.6)
    ax.axvline(16, color="#a64b32", linestyle="--", linewidth=1.5)
    ax.text(15.99, 1.323, "16 MB limit", ha="right", color="#a64b32", fontsize=10)
    ax.set(xlim=(14.9, 16.12), ylim=(1.14, 1.34),
           xlabel="Artifact size (decimal MB; code + compressed weights)",
           ylabel="Final FineWeb bits per byte (lower is better)")
    ax.set_title("All three completed models fit below 16 MB", loc="left", fontsize=16, pad=18, weight="bold")
    ax.grid(alpha=.16)
    fig.text(.105, .025, "◆ Sliding-window evaluation (stride 64)    ● Standard evaluation\n"
             "Source: archived submission.json files. Different recipes and training budgets.",
             fontsize=10, color=MUTED, linespacing=1.5)
    fig.tight_layout(rect=(0, .115, 1, 1))
    save(fig, "results.png")


def validation_points(folder, quantized=False):
    text = (RECORDS / folder / "train.log").read_text()
    metric = r"quant_proxy mode:\S+ " if quantized else ""
    pattern = rf"^step:(\d+)/\d+ {metric}val_loss:[0-9.]+ val_bpb:([0-9.]+).*?$"
    # Anchor to actual log lines; source code printed earlier in the log is ignored.
    points = {int(step): float(value) for step, value in re.findall(pattern, text, re.M)}
    points = sorted((step, value) for step, value in points.items() if step > 0)
    if not points:
        raise ValueError(f"No validation checkpoints found for {folder}")
    return points


def plot_training():
    recurrence = validation_points(RUNS[0][0])
    hybrid = validation_points(RUNS[1][0], quantized=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), facecolor=BACKGROUND)
    panels = [
        (axes[0], recurrence, "Recurrence: 10k → 20k continuation", "Full-precision validation bpb", "#136c66", "o"),
        (axes[1], hybrid, "Hybrid: quantized model improves", "Quantized-proxy validation bpb", "#435bc1", "s"),
    ]
    for ax, points, title, ylabel, color, marker in panels:
        style_axis(ax)
        steps, values = zip(*points)
        ax.plot(steps, values, color=color, marker=marker, markersize=4, linewidth=2)
        ax.set(title=title, xlabel="Training step", ylabel=ylabel)
        ax.title.set_fontsize(12)
        ax.title.set_weight("bold")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k"))
        ax.grid(alpha=.16)
        ax.margins(x=.13, y=.25)
        first_offset = (8, -20) if points is recurrence else (8, 10)
        for index, offset in [(0, first_offset), (-1, (-8, -20))]:
            ax.annotate(f"{values[index]:.4f}", (steps[index], values[index]),
                        xytext=offset, textcoords="offset points", fontsize=10,
                        color=color, weight="bold", ha="left" if index == 0 else "right")
    fig.suptitle("Validation improves during the recorded training stages", x=.06, ha="left", fontsize=16, weight="bold")
    fig.text(.06, .025, "Source: archived train.log validation checkpoints; no smoothing.\n"
             "Panels use different metrics and scales. These are training signals, not final exported-model scores.",
             fontsize=10, color=MUTED, linespacing=1.5)
    fig.tight_layout(rect=(0, .14, 1, .94), w_pad=2)
    save(fig, "training_progress.png")


def plot_budget():
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=BACKGROUND)
    style_axis(ax)
    labels = ["Recurrence", "Hybrid tail", "Pure ternary MLP"]
    for y, (folder, _, _, color, _) in enumerate(RUNS):
        used = metadata(folder)["bytes_total"] / 1_000_000
        free = 16 - used
        ax.barh(y, used, color=color, height=.52)
        ax.barh(y, free, left=used, color="#e2e7e7", edgecolor="#88969e", hatch="////", height=.52)
        ax.text(.35, y, f"{used:.2f} MB used", color="white", va="center", fontsize=11, weight="bold")
        ax.text(16.35, y, f"{free:.2f} MB free", color=INK, va="center", fontsize=11)
    ax.axvline(16, color="#a64b32", linestyle="--", linewidth=1.5)
    ax.text(15.9, -.47, "16 MB cap", ha="right", color="#a64b32", fontsize=10)
    ax.set(yticks=range(3), yticklabels=labels, xlim=(0, 19.8),
           xticks=[0, 4, 8, 12, 16], xlabel="Artifact size (decimal MB; code + compressed weights)",
           ylim=(2.6, -.8))
    ax.grid(axis="x", alpha=.16)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=11)
    ax.set_title("Compressed artifacts leave 0.33–0.96 MB below the cap", loc="left", fontsize=15, weight="bold", pad=18)
    fig.text(.17, .025, "Solid: recorded artifact size    Hatched: unused budget\nSource: archived submission.json files; sizes include counted code.",
             fontsize=10, color=MUTED, linespacing=1.5)
    fig.tight_layout(rect=(0, .14, 1, 1))
    save(fig, "artifact_budget.png")


def main():
    plt.rcParams.update({"font.size": 10, "axes.labelcolor": INK, "text.color": INK})
    plot_results()
    plot_training()
    plot_budget()


if __name__ == "__main__":
    main()
