"""Regenerate the README figure from the archived submission metadata."""
from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RECORDS = ROOT / "records" / "track_non_record_16mb"
RUNS = [
    ("2026-04-04_Full80_Recurrence_StepEmbed_Seq1536_20k", "Recurrence + step embeddings", "D", "#136c66", (12, -25)),
    ("2026-04-03_15L_HybridTail_Seq1536_20k_1x5090", "Ternary + 5-bit tail", "o", "#435bc1", (-155, 14)),
    ("2026-03-27_15L_Ternary_MLPOnly_20k", "Ternary MLP weights", "o", "#79629b", (12, 10)),
]

fig, ax = plt.subplots(figsize=(10, 4.8), facecolor="#fafbf9")
ax.set_facecolor("#fafbf9")
for folder, label, marker, color, offset in RUNS:
    metadata = json.loads((RECORDS / folder / "submission.json").read_text())
    x, y = metadata["bytes_total"] / 1_000_000, metadata["val_bpb"]
    ax.scatter(x, y, s=105, marker=marker, color=color, zorder=3)
    ax.annotate(f"{label}\n{y:.5f} bpb · {x:.2f} MB", (x, y),
                xytext=offset, textcoords="offset points", fontsize=10,
                color="#25343b", linespacing=1.6)
ax.axvline(16, color="#a64b32", linestyle="--", linewidth=1.5)
ax.text(15.99, 1.323, "16 MB limit", ha="right", color="#a64b32", fontsize=10)
ax.set(xlim=(14.9, 16.12), ylim=(1.14, 1.34),
       xlabel="Artifact size (decimal MB; code + compressed weights)",
       ylabel="Final FineWeb bits per byte (lower is better)")
ax.set_title("Completed compression experiments", loc="left", fontsize=17, pad=18, weight="bold")
ax.grid(alpha=.16)
ax.spines[["top", "right"]].set_visible(False)
fig.text(.105, .025, "◆ Sliding-window evaluation (stride 64)    ● Standard evaluation\n"
         "Different recipes and training budgets; this is not a controlled architecture comparison.",
         fontsize=9, color="#53616a", linespacing=1.5)
fig.tight_layout(rect=(0, .105, 1, 1))
fig.savefig(Path(__file__).with_name("results.png"), dpi=170, facecolor=fig.get_facecolor())
plt.close(fig)
