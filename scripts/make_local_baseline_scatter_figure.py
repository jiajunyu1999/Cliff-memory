from __future__ import annotations

"""Draw a local-baseline-only MoleculeACE error scatter for the main text."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from figure_style import COLORS, clean_axis, save_figure, set_publication_style


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "results" / "local_baseline_scatter"

LABELS = {
    "linear_ecfp": "Linear ECFP",
    "random_forest_ecfp": "RF-ECFP",
    "extratrees_ecfp": "ExtraTrees",
    "svm_ecfp": "ECFP-SVM",
    "hist_gradient_boosting_ecfp": "HGB-ECFP",
    "mlp_ecfp": "MLP-ECFP",
    "chemprop": "Chemprop",
    "gine_nodenorm": "GINE-Node",
    "graphcliff": "GraphCliff",
    "mtpnet_released_epoch50": "MTPNet",
    "taper_strict_exclusion": "TAPER",
}

MARKERS = {
    "linear_ecfp": "o",
    "random_forest_ecfp": "s",
    "extratrees_ecfp": "^",
    "svm_ecfp": "D",
    "hist_gradient_boosting_ecfp": "v",
    "mlp_ecfp": "P",
    "chemprop": "X",
    "gine_nodenorm": "<",
    "graphcliff": ">",
    "mtpnet_released_epoch50": "h",
    "taper_strict_exclusion": "*",
}

METHOD_COLORS = {
    "linear_ecfp": "#6C7A89",
    "random_forest_ecfp": "#4E79A7",
    "extratrees_ecfp": "#59A14F",
    "svm_ecfp": "#8E6C8A",
    "hist_gradient_boosting_ecfp": "#F28E2B",
    "mlp_ecfp": "#9C755F",
    "chemprop": "#4DBBD5",
    "gine_nodenorm": "#3C5488",
    "graphcliff": "#E64B35",
    "mtpnet_released_epoch50": "#7E6148",
    "taper_strict_exclusion": "#00A087",
}


def load_summary() -> pd.DataFrame:
    pieces = []
    for path in [
        ROOT / "outputs" / "cross_dataset_regression_baselines" / "summary.csv",
        ROOT / "outputs" / "baseline_runs" / "moleculeace_classical_recent.csv",
        ROOT / "outputs" / "baseline_runs" / "moleculeace_graph_full.csv",
        ROOT / "outputs" / "baseline_smoke" / "mtpnet_epoch50" / "summary.mtpnet_epoch50.csv",
        ROOT / "outputs" / "taper_profile_controls_cv5" / "summary.cv5.csv",
    ]:
        frame = pd.read_csv(path)
        if "collection" in frame:
            frame = frame[frame.collection.eq("MoleculeACE")]
        if "model" in frame and path.name == "summary.cv5.csv":
            frame = frame[frame.model.eq("taper_strict_exclusion")]
        pieces.append(frame[["dataset", "model", "rmse", "cliff_rmse"]])
    target = pd.concat(pieces, ignore_index=True).groupby(
        ["dataset", "model"], as_index=False
    )[["rmse", "cliff_rmse"]].mean()
    summary = target.groupby("model", as_index=False)[["rmse", "cliff_rmse"]].mean()
    summary["label"] = summary.model.map(LABELS)
    summary = summary[summary.label.notna()].copy()
    if summary.model.nunique() != len(LABELS):
        raise RuntimeError(f"incomplete local scatter methods: {summary.model.tolist()}")
    return summary


def draw(summary: pd.DataFrame) -> None:
    set_publication_style()
    fig, ax = plt.subplots(figsize=(5.05, 3.55), facecolor="white")
    ax.set_facecolor("white")

    for row in summary.sort_values("rmse").itertuples(index=False):
        is_taper = row.model == "taper_strict_exclusion"
        ax.scatter(
            row.rmse, row.cliff_rmse,
            s=105 if is_taper else 48,
            marker=MARKERS[row.model],
            facecolor=METHOD_COLORS[row.model],
            edgecolor="#00735E" if is_taper else "#FFFFFF",
            linewidth=1.0 if is_taper else 0.85,
            alpha=0.98 if is_taper else 0.88,
            zorder=4 if is_taper else 3,
        )

    offsets = {
        "TAPER": (8, -2),
        "ECFP-SVM": (-42, -11),
        "ExtraTrees": (-66, 15),
        "RF-ECFP": (-54, -18),
        "MTPNet": (12, 18),
        "HGB-ECFP": (-74, 32),
        "Linear ECFP": (7, 9),
        "MLP-ECFP": (-54, 3),
        "Chemprop": (-8, 13),
        "GraphCliff": (8, -3),
        "GINE-Node": (12, -15),
    }
    for row in summary.itertuples(index=False):
        dx, dy = offsets[row.label]
        ax.annotate(row.label, (row.rmse, row.cliff_rmse), xytext=(dx, dy),
                    textcoords="offset points",
                    ha="left" if dx >= 0 else "right", va="center",
                    fontsize=6.7, color=METHOD_COLORS[row.model],
                    arrowprops={
                        "arrowstyle": "-",
                        "color": METHOD_COLORS[row.model],
                        "lw": 0.45,
                        "alpha": 0.55,
                        "shrinkA": 1.5,
                        "shrinkB": 3.0,
                    })

    ax.plot([0.55, 0.99], [0.55, 0.99], color="#C6C6C6", lw=0.8,
            ls=(0, (3, 2)), zorder=1)
    ax.set_xlabel("Overall RMSE")
    ax.set_ylabel("Cliff RMSE")
    ax.set_xlim(0.55, 1.01)
    ax.set_ylim(0.62, 1.03)
    clean_axis(ax, grid="both")
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.14, top=0.985)
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "summary.csv", index=False)
    fig.savefig(OUT / "fig_local_baseline_scatter.png", dpi=420,
                bbox_inches="tight", facecolor="white")
    save_figure(fig, ROOT / "paper" / "figures" / "fig_inference_efficiency.pdf")


def main() -> None:
    draw(load_summary())


if __name__ == "__main__":
    main()
