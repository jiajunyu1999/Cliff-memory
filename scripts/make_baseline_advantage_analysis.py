from __future__ import annotations

"""Explain where TAPER improves over the strongest locally refit baseline.

All comparisons use target-level units.  Fold-level results are averaged within
each target before a baseline winner is selected, so cross-validation folds are
not treated as independent samples.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from figure_style import COLORS, clean_axis, panel_label, save_figure, set_publication_style


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "results" / "baseline_advantage_analysis"
METRICS = ("rmse", "cliff_rmse", "noncliff_rmse")
METRIC_LABELS = {
    "rmse": "Overall",
    "cliff_rmse": "Cliff",
    "noncliff_rmse": "Non-cliff",
}
BASELINE_LABELS = {
    "linear_ecfp": "Linear ECFP",
    "random_forest_ecfp": "RF-ECFP",
    "extratrees_ecfp": "ExtraTrees-ECFP",
    "svm_ecfp": "ECFP-SVM",
    "hist_gradient_boosting_ecfp": "HGB-ECFP",
    "mlp_ecfp": "MLP-ECFP",
    "chemprop": "Chemprop",
    "gine_nodenorm": "GINE + NodeNorm",
    "graphcliff": "GraphCliff",
    "mtpnet_released_epoch50": "MTPNet",
}


def load_baselines() -> pd.DataFrame:
    paths = [
        ROOT / "outputs" / "cross_dataset_regression_baselines" / "summary.csv",
        ROOT / "outputs" / "baseline_runs" / "moleculeace_classical_recent.csv",
        ROOT / "outputs" / "baseline_runs" / "moleculeace_graph_full.csv",
        ROOT / "outputs" / "baseline_smoke" / "mtpnet_epoch50" / "summary.mtpnet_epoch50.csv",
    ]
    pieces = []
    for path in paths:
        frame = pd.read_csv(path)
        if "collection" in frame:
            frame = frame[frame.collection.eq("MoleculeACE")]
        pieces.append(frame[["dataset", "model", *METRICS]].copy())
    raw = pd.concat(pieces, ignore_index=True)
    raw = raw[raw.model.isin(BASELINE_LABELS)]
    return raw.groupby(["dataset", "model"], as_index=False)[list(METRICS)].mean()


def load_taper() -> pd.DataFrame:
    frame = pd.read_csv(ROOT / "outputs" / "taper_profile_controls_cv5" / "summary.cv5.csv")
    frame = frame[frame.model.eq("taper_strict_exclusion")].copy()
    return frame.groupby("dataset", as_index=False)[
        [*METRICS, "positive_entity_dimensions", "positive_assay_dimensions",
         "selected_entity_dimensions", "selected_assay_dimensions",
         "entity_dimensions", "assay_dimensions"]
    ].mean()


def bootstrap_mean(values: np.ndarray, seed: int, draws: int = 20000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return tuple(float(v) for v in np.quantile(samples, [0.025, 0.975]))


def build_report() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    baselines = load_baselines()
    taper = load_taper().set_index("dataset")
    if baselines.dataset.nunique() != 30 or taper.shape[0] != 30:
        raise RuntimeError("advantage analysis requires all 30 MoleculeACE targets")
    rows = []
    for dataset in sorted(taper.index):
        target_base = baselines[baselines.dataset.eq(dataset)]
        for metric in METRICS:
            winner = target_base.loc[target_base[metric].idxmin()]
            taper_value = float(taper.loc[dataset, metric])
            rows.append({
                "dataset": dataset,
                "metric": metric,
                "best_baseline_model": winner.model,
                "best_baseline_label": BASELINE_LABELS.get(winner.model, winner.model),
                "best_baseline": float(winner[metric]),
                "taper": taper_value,
                "gain": float(winner[metric] - taper_value),
                "relative_gain": float((winner[metric] - taper_value) / winner[metric]),
                "positive_profile_dimensions": float(
                    taper.loc[dataset, "positive_entity_dimensions"]
                    + taper.loc[dataset, "positive_assay_dimensions"]
                ),
                "selected_profile_dimensions": float(
                    taper.loc[dataset, "selected_entity_dimensions"]
                    + taper.loc[dataset, "selected_assay_dimensions"]
                ),
                "available_profile_dimensions": float(
                    taper.loc[dataset, "entity_dimensions"]
                    + taper.loc[dataset, "assay_dimensions"]
                ),
            })
    paired = pd.DataFrame(rows)
    summary_rows = []
    detail: dict[str, object] = {"targets": 30, "unit": "target mean across folds"}
    seed = 20260906
    for metric in METRICS:
        part = paired[paired.metric.eq(metric)]
        gains = part.gain.to_numpy(float)
        lo, hi = bootstrap_mean(gains, seed)
        seed += 1
        stat = wilcoxon(gains, alternative="greater", zero_method="zsplit")
        summary_rows.append({
            "metric": metric,
            "mean_gain": float(gains.mean()),
            "median_gain": float(np.median(gains)),
            "ci_low": lo,
            "ci_high": hi,
            "wins": int((gains > 0).sum()),
            "ties": int((gains == 0).sum()),
            "losses": int((gains < 0).sum()),
            "wilcoxon_p_one_sided": float(stat.pvalue),
        })
    cliff = paired[paired.metric.eq("cliff_rmse")]
    rho, pvalue = spearmanr(cliff.positive_profile_dimensions, cliff.gain)
    detail["profile_evidence_vs_cliff_gain"] = {
        "spearman_rho": float(rho),
        "p_two_sided": float(pvalue),
    }
    return paired, pd.DataFrame(summary_rows), detail


def draw(paired: pd.DataFrame, summary: pd.DataFrame, detail: dict) -> None:
    set_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.45), facecolor="white")

    rng = np.random.default_rng(20260906)
    ax = axes[0]
    positions = np.arange(len(METRICS))
    data = [paired[paired.metric.eq(metric)].gain.to_numpy(float) for metric in METRICS]
    colors = [COLORS["blue"], COLORS["red"], COLORS["grey"]]
    boxes = ax.boxplot(
        data, positions=positions, widths=0.48, patch_artist=True, showfliers=False,
        medianprops={"color": COLORS["ink"], "linewidth": 1.15},
        whiskerprops={"color": "#555555", "linewidth": 0.8},
        capprops={"color": "#555555", "linewidth": 0.8},
        boxprops={"edgecolor": "#555555", "linewidth": 0.8},
    )
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.18)
    for pos, values, color in zip(positions, data, colors):
        ax.scatter(pos + rng.uniform(-0.12, 0.12, len(values)), values, s=10,
                   color=color, alpha=0.45, edgecolor="none", zorder=3)
    ax.axhline(0, color=COLORS["ink"], lw=0.75, ls=(0, (3, 2)), zorder=1)
    ax.set_xticks(positions, [METRIC_LABELS[m] for m in METRICS])
    ax.set_ylabel(r"Best baseline $-$ TAPER RMSE")
    clean_axis(ax, grid="y")
    panel_label(ax, "a", x=-0.20, y=1.04)

    ax = axes[1]
    y = np.arange(len(METRICS))[::-1]
    means = summary.set_index("metric").loc[list(METRICS), "mean_gain"].to_numpy(float)
    low = summary.set_index("metric").loc[list(METRICS), "ci_low"].to_numpy(float)
    high = summary.set_index("metric").loc[list(METRICS), "ci_high"].to_numpy(float)
    xerr = np.vstack([means - low, high - means])
    ax.barh(y, means, xerr=xerr, height=0.50, color=colors, alpha=0.80,
            error_kw={"ecolor": "#30343B", "elinewidth": 0.85, "capsize": 2.0}, zorder=3)
    count_x = float(high.max()) + 0.028
    for yy, metric in zip(y, METRICS):
        row = summary[summary.metric.eq(metric)].iloc[0]
        ax.text(count_x, yy, f"{int(row.wins)}/30",
                ha="center", va="center", fontsize=7.0, color=COLORS["muted"])
    ax.axvline(0, color=COLORS["ink"], lw=0.75, ls=(0, (3, 2)), zorder=1)
    ax.set_yticks(y, [METRIC_LABELS[m] for m in METRICS])
    ax.set_xlabel("Mean paired gain")
    ax.set_xlim(left=min(0, float(low.min()) - 0.015), right=float(high.max()) + 0.070)
    clean_axis(ax, grid="x")
    panel_label(ax, "b", x=-0.18, y=1.04)

    ax = axes[2]
    cliff = paired[paired.metric.eq("cliff_rmse")].copy()
    rho = detail["profile_evidence_vs_cliff_gain"]["spearman_rho"]
    pvalue = detail["profile_evidence_vs_cliff_gain"]["p_two_sided"]
    ax.scatter(cliff.positive_profile_dimensions, cliff.gain, s=19,
               color=COLORS["red"], alpha=0.72, edgecolor="white", linewidth=0.35, zorder=3)
    if cliff.positive_profile_dimensions.nunique() > 1:
        x = cliff.positive_profile_dimensions.to_numpy(float)
        yv = cliff.gain.to_numpy(float)
        coef = np.polyfit(x, yv, deg=1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, coef[0] * xs + coef[1], color=COLORS["ink"], lw=0.9, zorder=2)
    ax.axhline(0, color=COLORS["ink"], lw=0.75, ls=(0, (3, 2)), zorder=1)
    ax.set_xlabel("Positive profile coordinates")
    ax.set_ylabel("Cliff RMSE gain")
    ax.text(0.03, 0.96, rf"$\rho={rho:.2f}$, $p={pvalue:.2f}$",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.2, color=COLORS["muted"])
    clean_axis(ax, grid="y")
    panel_label(ax, "c", x=-0.20, y=1.04)

    fig.subplots_adjust(left=0.085, right=0.990, bottom=0.225, top=0.93, wspace=0.44)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_baseline_advantage_analysis.png", dpi=420,
                bbox_inches="tight", facecolor="white")
    save_figure(fig, ROOT / "paper" / "figures" / "fig_baseline_advantage_analysis.pdf")


def main() -> None:
    paired, summary, detail = build_report()
    OUT.mkdir(parents=True, exist_ok=True)
    paired.to_csv(OUT / "paired_target_gains.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    (OUT / "audit.json").write_text(json.dumps(detail, indent=2) + "\n")
    draw(paired, summary, detail)
    print(json.dumps({
        "figure": str(ROOT / "paper" / "figures" / "fig_baseline_advantage_analysis.pdf"),
        "targets": 30,
        "summary": summary.to_dict(orient="records"),
        **detail,
    }, indent=2))


if __name__ == "__main__":
    main()
