from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 17,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def save_bar(path: Path, labels: list[str], values: list[float], ylabel: str,
             title: str, colors: list[str], ylim: tuple[float, float] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.7)
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=12)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    if ylim:
        ax.set_ylim(*ylim)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.006,
                f"{value:.3f}", ha="center", va="bottom", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


save_bar(
    OUT / "fig_train_cv_rmse.pdf",
    ["Collision-free\nchemistry", "Profile-only\nresponse", "Combined\nours"],
    [0.6440007998, 0.5927771633, 0.5918596891],
    "Macro RMSE (lower is better)",
    "30-target train-only five-fold CV",
    ["#9aa0a6", "#5b8ff9", "#d94841"],
    (0.55, 0.70),
)

save_bar(
    OUT / "fig_literature_cliff_rmse.pdf",
    ["Official\nSVM", "SCAGE\n(2025)", "GraphCliff\n(2025)", "ACANet\n(2026)", "Ours\nCV"],
    [0.751128, 0.73177, 0.757, 0.711, 0.671614],
    "Cliff RMSE (lower is better)",
    "Reported cliff-aware baselines",
    ["#9aa0a6", "#7aa874", "#8c6bb1", "#f0a202", "#d94841"],
    (0.64, 0.80),
)

summary = pd.read_csv(ROOT / "outputs/moleculeace_profile_cv5_combined_v2/summary.cv5.csv")
base = "collision_free_binary_count_kernel"
cand = "combined_positive_signal_kernel_10chem_8profile_4physics_1susceptibility"
pivot = summary.pivot_table(index=["dataset", "fold"], columns="model",
                            values=["rmse", "cliff_rmse"])
gain = pd.DataFrame(index=pivot.index)
gain["rmse"] = (pivot["rmse"][base] - pivot["rmse"][cand]) / pivot["rmse"][base] * 100
gain["cliff"] = (pivot["cliff_rmse"][base] - pivot["cliff_rmse"][cand]) / pivot["cliff_rmse"][base] * 100
gain = gain.groupby(level=0).mean().sort_values("cliff")
fig, ax = plt.subplots(figsize=(10.5, 5.3))
y = np.arange(len(gain))
ax.barh(y, gain["cliff"], color=np.where(gain["cliff"] >= 0, "#d94841", "#9aa0a6"),
        edgecolor="black", linewidth=0.35)
ax.axvline(0, color="black", linewidth=0.9)
ax.set_yticks(y, [name.replace("CHEMBL", "") for name in gain.index])
ax.set_xlabel("Mean relative gain in cliff RMSE (%)")
ax.set_title("Per-target generalization across five folds", pad=12)
ax.grid(axis="x", alpha=0.25, linewidth=0.8)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT / "fig_per_target_cliff_gain.pdf", bbox_inches="tight")
plt.close(fig)

