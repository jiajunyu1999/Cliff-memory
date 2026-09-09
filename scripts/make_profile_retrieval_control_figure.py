from __future__ import annotations

"""Publication plot for the explicit profile-retrieval controls."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reports/results/extended_evidence/profile_retrieval_controls.csv"
OUT = ROOT / "paper/figures/fig_profile_retrieval_controls.pdf"
PNG = ROOT / "reports/results/extended_evidence/fig_profile_retrieval_controls.png"


def main() -> None:
    d = pd.read_csv(DATA)
    order = ["ECFP-SVM", "Profile-kNN (k=5)", "Profile-KRR", "ECFP + Profile-KRR late fusion"]
    labels = ["ECFP-SVM", "Profile\nkNN", "Profile\nKRR", "Late\nfusion"]
    d["model"] = pd.Categorical(d.model, categories=order, ordered=True)
    colors = ["#6C7A89", "#C79052", "#8D78B8", "#4DBBD5"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.labelsize": 11, "xtick.labelsize": 9,
                         "ytick.labelsize": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0), facecolor="white")
    rng = np.random.default_rng(42)
    for ax, metric, ylabel, letter in zip(
        axes, ["rmse", "cliff_rmse"], ["Overall RMSE", "Cliff RMSE"], ["a", "b"]
    ):
        for x, (name, color) in enumerate(zip(order, colors)):
            values = d.loc[d.model.eq(name), metric].to_numpy(float)
            jitter = rng.normal(0, .045, len(values))
            ax.scatter(np.full(len(values), x) + jitter, values, s=15, color=color,
                       edgecolor="white", linewidth=.3, alpha=.88, zorder=3)
            mean = values.mean()
            ax.plot([x - .18, x + .18], [mean, mean], color="#1E2A30", lw=1.5, zorder=4)
        ax.set_xlim(-.55, 3.55)
        ax.set_xticks(range(4), labels)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", color="#888888", alpha=.28, zorder=0)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(.8); ax.spines["bottom"].set_linewidth(.8)
        ax.text(-.16, 1.03, letter, transform=ax.transAxes, fontweight="bold", fontsize=12)
    fig.tight_layout(w_pad=1.8)
    OUT.parent.mkdir(parents=True, exist_ok=True); PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(PNG, dpi=350, bbox_inches="tight")


if __name__ == "__main__":
    main()
