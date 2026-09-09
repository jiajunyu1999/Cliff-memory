from __future__ import annotations

"""Report local MoleculeACE baselines with target-level uncertainty and p-values.

The comparison unit is a target, never an individual CV fold.  For each method
we first average repeated folds within each target, then report the target mean
and target standard deviation.  For each baseline, the one-sided Wilcoxon test
tests whether baseline RMSE is larger than TAPER RMSE on paired targets.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
TAPER_PATH = ROOT / "outputs/taper_profile_controls_cv5/summary.cv5.csv"
SOURCES = (
    ROOT / "outputs/cross_dataset_regression_baselines/summary.csv",
    ROOT / "outputs/baseline_runs/moleculeace_classical_recent.csv",
    ROOT / "outputs/baseline_runs/moleculeace_graph_full.csv",
    ROOT / "outputs/baseline_smoke/mtpnet_epoch50/summary.mtpnet_epoch50.csv",
)
LABELS = {
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
    "taper_strict_exclusion": "TAPER",
}
ORDER = (
    "linear_ecfp", "random_forest_ecfp", "extratrees_ecfp", "svm_ecfp",
    "hist_gradient_boosting_ecfp", "mlp_ecfp", "chemprop", "gine_nodenorm",
    "graphcliff", "mtpnet_released_epoch50", "taper_strict_exclusion",
)


def _load_baselines() -> pd.DataFrame:
    pieces = []
    for path in SOURCES:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "collection" in frame:
            frame = frame[frame.collection.eq("MoleculeACE")]
        if "fold" not in frame:
            frame["fold"] = 0
        if "seed" in frame:
            # The TAPER controls use the official, train-only seed 42 split.
            frame = frame[frame.seed.eq(42)]
        if not frame.empty:
            pieces.append(frame[["dataset", "model", "seed", "fold", "rmse", "cliff_rmse"]])
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _target_values(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(["dataset", "model"], as_index=False)[["rmse", "cliff_rmse"]].mean()


def _format_p(value: float) -> str:
    if value < 1e-3:
        return "***"
    return f"p={value:.3f}"


def _holm_adjust(values: pd.Series) -> pd.Series:
    """Holm family-wise error correction, preserving the original index."""
    finite = values.dropna().sort_values()
    adjusted = pd.Series(np.nan, index=values.index, dtype=float)
    running = 0.0
    total = len(finite)
    for rank, (index, value) in enumerate(finite.items()):
        running = max(running, min(1.0, (total - rank) * float(value)))
        adjusted.loc[index] = running
    return adjusted


def make_report(require_graph: bool) -> tuple[pd.DataFrame, dict]:
    graph_path = SOURCES[-1]
    if require_graph and not graph_path.exists():
        raise FileNotFoundError(f"required full local graph results are missing: {graph_path}")
    if not TAPER_PATH.exists():
        raise FileNotFoundError(f"missing TAPER repeated-CV results: {TAPER_PATH}")

    taper = pd.read_csv(TAPER_PATH)
    taper = taper[taper.model.eq("taper_strict_exclusion")][
        ["dataset", "model", "seed", "fold", "rmse", "cliff_rmse"]
    ]
    baseline = _load_baselines()
    frame = pd.concat([baseline, taper], ignore_index=True)
    target = _target_values(frame)
    taper_target = target[target.model.eq("taper_strict_exclusion")].set_index("dataset")
    rows = []
    details: dict[str, dict] = {}
    for method in ORDER:
        part = target[target.model.eq(method)].set_index("dataset")
        if part.empty:
            continue
        common = part.index.intersection(taper_target.index)
        values = {"method": method, "label": LABELS[method], "n_targets": int(len(part))}
        entry: dict[str, object] = {"n_targets": int(len(part)), "paired_targets": int(len(common))}
        for metric in ("rmse", "cliff_rmse"):
            values[f"{metric}_mean"] = float(part[metric].mean())
            values[f"{metric}_sd"] = float(part[metric].std(ddof=1))
            if method == "taper_strict_exclusion":
                values[f"{metric}_gain_vs_taper"] = 0.0
                values[f"{metric}_p_one_sided"] = np.nan
                continue
            paired = part.loc[common, metric] - taper_target.loc[common, metric]
            # A positive paired value means TAPER has lower error.
            test = wilcoxon(paired, alternative="greater", zero_method="zsplit")
            values[f"{metric}_gain_vs_taper"] = float(paired.mean())
            values[f"{metric}_p_one_sided"] = float(test.pvalue)
            entry[metric] = {
                "mean_gain": float(paired.mean()),
                "median_gain": float(paired.median()),
                "wins_ties_losses": [
                    int((paired > 0).sum()), int((paired == 0).sum()), int((paired < 0).sum())
                ],
                "wilcoxon_p_one_sided": float(test.pvalue),
            }
        rows.append(values)
        details[method] = entry
    report = pd.DataFrame(rows)
    for metric in ("rmse", "cliff_rmse"):
        raw = f"{metric}_p_one_sided"
        report[f"{metric}_p_holm"] = _holm_adjust(report[raw])
        for row in report.itertuples(index=False):
            if row.method in details:
                details[row.method][f"{metric}_p_holm"] = (
                    None if not np.isfinite(getattr(row, f"{metric}_p_holm"))
                    else float(getattr(row, f"{metric}_p_holm"))
                )
    return report, details


def plot(report: pd.DataFrame, path: Path) -> None:
    methods = report.method.tolist()
    labels = report.label.tolist()
    y = np.arange(len(methods))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(8.9, 4.8), sharey=True, facecolor="white")
    colors = ["#00A087" if item == "taper_strict_exclusion" else "#6C7A89" for item in methods]
    metrics = [("rmse", "A", "Overall RMSE"), ("cliff_rmse", "B", "Cliff RMSE")]
    for ax, (metric, panel, label) in zip(axes, metrics):
        ax.set_facecolor("white")
        means = report[f"{metric}_mean"].to_numpy(float)
        sds = report[f"{metric}_sd"].to_numpy(float)
        ax.barh(y, means, xerr=sds, height=.62, color=colors, edgecolor="none",
                error_kw={"ecolor": "#30343B", "elinewidth": .8, "capsize": 2.0}, zorder=3)
        ax.text(.0, 1.015, panel, transform=ax.transAxes, va="bottom", ha="left",
                fontsize=11, fontweight="bold", color="#202124", clip_on=False)
        ax.text(.055, 1.015, label, transform=ax.transAxes, va="bottom", ha="left",
                fontsize=10.5, color="#202124", clip_on=False)
        p_x = float(np.max(means + sds) + .07)
        for yy, mean, sd, p, method in zip(y, means, sds, report[f"{metric}_p_holm"], methods):
            if method == "taper_strict_exclusion" or not np.isfinite(p):
                continue
            ax.text(p_x, yy, _format_p(float(p)), va="center", ha="center",
                    fontsize=8.2, color="#3A3A3A", clip_on=False, zorder=4)
        ax.grid(axis="x", linestyle="--", alpha=.25, color="gray", zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlabel("Target mean RMSE (lower is better)", fontsize=10)
        ax.tick_params(axis="x", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_linewidth(.8)
        # Reserve space for p-value labels instead of letting them collide with bars.
        ax.set_xlim(0, float(np.max(means + sds) + .13))
    axes[0].set_yticks(y, labels, fontsize=8.8)
    axes[0].tick_params(axis="y", length=0)
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    fig.text(.012, .01, "Bars: target mean; error bars: target SD; ***: Holm-adjusted one-sided paired Wilcoxon p<0.001 vs TAPER.",
             fontsize=7.8, color="#4A4A4A")
    fig.tight_layout(rect=(.01, .04, 1, .985))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=350, bbox_inches="tight")
    plt.close(fig)


def write_latex(report: pd.DataFrame, path: Path) -> None:
    """Write the exact values plotted, avoiding manual transcription drift."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Local five-fold MoleculeACE comparison. Values are mean $\mathbin{\pm}$ standard deviation across 30 targets after averaging folds within each target. $P$ values are Holm-adjusted one-sided paired Wilcoxon tests of whether each baseline has larger target-level error than \method.}",
        r"\label{tab:local-cv-significance}",
        r"\scriptsize",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Method & RMSE & Cliff RMSE & $P$ (cliff)\\",
        r"\midrule",
    ]
    for row in report.itertuples(index=False):
        name = r"\textbf{\method}" if row.method == "taper_strict_exclusion" else row.label
        overall = f"{row.rmse_mean:.3f} $\\mathbin{{\\pm}}$ {row.rmse_sd:.3f}"
        cliff = f"{row.cliff_rmse_mean:.3f} $\\mathbin{{\\pm}}$ {row.cliff_rmse_sd:.3f}"
        if row.method == "taper_strict_exclusion":
            overall, cliff, pvalue = rf"\textbf{{{overall}}}", rf"\textbf{{{cliff}}}", "--"
        else:
            pvalue = (r"$<0.001$" if row.cliff_rmse_p_holm < 1e-3
                      else f"{row.cliff_rmse_p_holm:.3f}")
        lines.append(f"{name} & {overall} & {cliff} & {pvalue}\\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "results" / "local_baseline_statistics")
    parser.add_argument("--require-graph", action="store_true")
    args = parser.parse_args()
    report, details = make_report(args.require_graph)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output_dir / "moleculeace_local_baseline_mean_sd_pvalues.csv", index=False)
    (args.output_dir / "moleculeace_local_baseline_pairwise_tests.json").write_text(json.dumps(details, indent=2) + "\n")
    plot(report, args.output_dir / "moleculeace_local_baseline_significance")
    plot(report, ROOT / "paper/figures/fig_local_baseline_significance")
    write_latex(report, args.output_dir / "moleculeace_local_baseline_significance.tex")
    write_latex(report, ROOT / "paper/tables/local_baseline_significance.tex")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
