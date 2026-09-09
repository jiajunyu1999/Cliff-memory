from __future__ import annotations

"""Plot target-level sensitivity to retained response components."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_style import COLORS, clean_axis, panel_label, save_figure, set_publication_style


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "moleculeace_ablation_cv5_seed42" / "summary.cv5.csv"
OUT = ROOT / "reports" / "results" / "component_sensitivity"

ORDER = [
    "collision_free_binary_count_kernel",
    "response_profile_block_kernel",
    "chemistry_profile_kernel_10to8",
    "chemistry_profile_physics_kernel_10to8to4",
    "chemistry_profile_susceptibility_kernel_10to8to1",
    "combined_positive_signal_kernel_10chem_8profile_4physics_1susceptibility",
]
LABELS = [
    "Chemistry only",
    "Response only",
    "Chemistry + response",
    "+ physics",
    "+ susceptibility",
    "Complete model",
]


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    draws = rng.choice(values, size=(20000, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def main() -> None:
    frame = pd.read_csv(SOURCE)
    frame = frame[frame.model.isin(ORDER)].copy()
    if frame.dataset.nunique() != 30 or frame.fold.nunique() != 5:
        raise RuntimeError("component sensitivity requires all 30 targets and five folds")
    # Folds are repeated measurements of a target, not independent bootstrap
    # units. First average within target, then bootstrap the 30 targets.
    target = frame.groupby(["dataset", "model"])[["rmse", "cliff_rmse"]].mean().reset_index()
    rng = np.random.default_rng(20260905)
    rows = []
    for model, label in zip(ORDER, LABELS):
        part = target[target.model.eq(model)]
        for metric in ("rmse", "cliff_rmse"):
            values = part[metric].to_numpy(float)
            lo, hi = bootstrap_ci(values, rng)
            rows.append({"model": model, "label": label, "metric": metric,
                         "mean": float(values.mean()), "lo": float(lo), "hi": float(hi),
                         "targets": int(len(values))})
    result = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT / "summary.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({"targets": 30, "folds": 5,
        "unit": "target-mean across folds", "models": LABELS}, indent=2) + "\n")

    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.85, 2.65), sharey=True)
    x = np.arange(len(ORDER))
    colors = [COLORS["ink"], COLORS["grey"], COLORS["blue"], "#3C5488", "#00A087", COLORS["red"]]
    for ax, metric, ylabel in zip(axes, ["rmse", "cliff_rmse"], ["Overall RMSE", "Cliff RMSE"]):
        part = result[result.metric.eq(metric)]
        for i, (model, color) in enumerate(zip(ORDER, colors)):
            row = part[part.model.eq(model)].iloc[0]
            ax.errorbar(i, row["mean"], yerr=[[row["mean"] - row["lo"]], [row["hi"] - row["mean"]]],
                        fmt="o", ms=5.4, color=color, ecolor=color, elinewidth=1.15,
                        capsize=2.5, markeredgecolor="white", markeredgewidth=.7, zorder=3)
        ax.plot(x, part.set_index("model").loc[ORDER, "mean"], color="#9AA0A6", lw=.85,
                zorder=1, alpha=.9)
        ax.set_xticks(x, LABELS, rotation=42, ha="right")
        ax.set_ylabel(ylabel if ax is axes[0] else "")
        ax.tick_params(axis="x", labelsize=8.2, pad=1)
        ax.tick_params(axis="y", labelsize=9)
        clean_axis(ax, grid="y")
        if ax is axes[1]:
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", left=False, labelleft=False)
        ax.margins(x=.06)
    panel_label(axes[0], "a", x=-.17, y=1.04)
    panel_label(axes[1], "b", x=-.10, y=1.04)
    fig.subplots_adjust(left=.095, right=.985, bottom=.34, top=.93, wspace=.10)
    save_figure(fig, ROOT / "paper" / "figures" / "fig_sensitivity.pdf")


if __name__ == "__main__":
    main()
