from __future__ import annotations

"""Publication figure for the completed scaffold-disjoint TAPER run."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reports/results/extended_evidence/scaffold_taper_all30.csv"
GNN = ROOT / "reports/results/extended_evidence/scaffold_gine_all30.csv"
OUT = ROOT / "paper/figures/fig_scaffold_coldstart.pdf"
PNG = ROOT / "reports/results/extended_evidence/fig_scaffold_coldstart.png"


def main() -> None:
    d = pd.read_csv(DATA)
    base = d[d.model.eq("collision_free_binary_count_kernel")].set_index("dataset")
    taper = d[d.model.eq("responsekernel_final")].set_index("dataset")
    gnn = pd.read_csv(GNN).set_index("dataset")
    common = base.index.intersection(taper.index).intersection(gnn.index)
    base, gnn, taper = base.loc[common], gnn.loc[common], taper.loc[common]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelsize": 11, "xtick.labelsize": 9,
                         "ytick.labelsize": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.0), facecolor="white")
    colors = {"base": "#6C7A89", "gnn": "#4DBBD5", "taper": "#00A087"}
    for panel, metric, label in zip(axes, ["rmse", "cliff_rmse"], ["Overall RMSE", "Cliff RMSE"]):
        x = np.array([0, 1, 2])
        for i, target in enumerate(common):
            panel.plot(x, [base.loc[target, metric], gnn.loc[target, metric], taper.loc[target, metric]],
                       color="#B9C0C6", lw=.7, alpha=.65, zorder=2)
        rng = np.random.default_rng(42)
        jitter = rng.normal(0, .018, len(common))
        panel.scatter(np.zeros(len(common)) + jitter, base[metric], s=17,
                      color=colors["base"], edgecolor="white", linewidth=.35,
                      zorder=3, label="Chemistry-only")
        panel.scatter(np.ones(len(common)) + jitter, gnn[metric], s=18,
                      color=colors["gnn"], edgecolor="white", linewidth=.35,
                      zorder=4, label="GINE + NodeNorm")
        panel.scatter(np.full(len(common), 2) + jitter, taper[metric], s=18,
                      color=colors["taper"], edgecolor="white", linewidth=.35,
                      zorder=5, label="TAPER")
        means = [base[metric].mean(), gnn[metric].mean(), taper[metric].mean()]
        panel.plot(x, means, color="#1E2A30", lw=1.35, marker="_", ms=14, zorder=5)
        panel.set_xticks(x, ["ECFP-SVM", "GINE +\nNodeNorm", "TAPER"])
        panel.set_ylabel(label)
        panel.grid(axis="y", linestyle="--", color="#888888", alpha=.28, zorder=0)
        panel.spines["top"].set_visible(False); panel.spines["right"].set_visible(False)
        panel.spines["left"].set_linewidth(.8); panel.spines["bottom"].set_linewidth(.8)
        reduction = 100 * (1 - taper[metric].mean() / base[metric].mean())
        panel.text(.5, .96, f"{reduction:.1f}% below ECFP-SVM", transform=panel.transAxes,
                   ha="center", va="top", fontsize=9, color="#007A67", fontweight="bold")
    axes[0].legend(frameon=False, loc="lower left", fontsize=8)
    axes[0].text(-.16, 1.03, "a", transform=axes[0].transAxes, fontweight="bold", fontsize=12)
    axes[1].text(-.16, 1.03, "b", transform=axes[1].transAxes, fontweight="bold", fontsize=12)
    fig.tight_layout(w_pad=1.8)
    OUT.parent.mkdir(parents=True, exist_ok=True); PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(PNG, dpi=350, bbox_inches="tight")


if __name__ == "__main__":
    main()
