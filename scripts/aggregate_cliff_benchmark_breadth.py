from __future__ import annotations

"""Create a transparent multi-benchmark activity-cliff result inventory."""

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def summarize_regression(path: Path, benchmark: str, unit: str = "dataset") -> pd.DataFrame:
    frame = pd.read_csv(path)
    # Average correlated folds inside a dataset before reporting cross-dataset SD.
    target = frame.groupby([unit, "model"], as_index=False)[["rmse", "cliff_rmse", "noncliff_rmse"]].mean()
    rows = []
    for model, part in target.groupby("model"):
        for metric in ("rmse", "cliff_rmse", "noncliff_rmse"):
            rows.append({"benchmark": benchmark, "task": "cliff regression", "model": model,
                         "metric": metric, "mean": part[metric].mean(), "sd": part[metric].std(ddof=1),
                         "n_units": len(part), "unit": unit})
    return pd.DataFrame(rows)


def summarize_acnet(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    rows = []
    for feature, part in frame.groupby("feature_mode"):
        for metric in ("auc", "ap", "acc"):
            rows.append({"benchmark": "ACNet", "task": "matched-pair cliff classification",
                         "model": f"SGD-logreg ({feature})", "metric": metric,
                         "mean": part[metric].mean(), "sd": part[metric].std(ddof=1),
                         "n_units": len(part), "unit": "target"})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "reports/results/cliff_benchmark_breadth.csv")
    parser.add_argument("--dablander", type=Path, default=ROOT / "outputs/dablander_external_v2_local/summary.csv")
    args = parser.parse_args()
    inputs = [
        (ROOT / "outputs/responsekernel_final_external_cv5_seed42/summary.cv5.csv", "GraphCliff external"),
        (args.dablander, "Dablander external"),
    ]
    reports = [summarize_regression(path, name) for path, name in inputs if path.exists()]
    acnet = ROOT / "outputs/acnet_all190_action_sgd_v1/summary.csv"
    if acnet.exists():
        reports.append(summarize_acnet(acnet))
    result = pd.concat(reports, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    if args.dablander.exists():
        dab = pd.read_csv(args.dablander)
        unit = dab.groupby(["dataset", "model"])[["rmse", "cliff_rmse"]].mean().reset_index()
        lines = [
            r"\begin{table}[t]", r"\centering",
            r"\caption{Local two-fold evaluation on three external DABlander activity-cliff datasets. Values are mean $\mathbin{\pm}$ standard deviation across datasets after averaging folds within each dataset. Lower is better.}",
            r"\label{tab:dablander-external}", r"\small", r"\begin{tabular}{lrr}",
            r"\toprule", r"Method & RMSE & Cliff RMSE\\", r"\midrule",
        ]
        labels = {"svm": "ECFP-SVM", "calibrated_mmp": "Calibrated MMP"}
        for model in ("svm", "calibrated_mmp"):
            part = unit[unit.model.eq(model)]
            overall = f"{part.rmse.mean():.3f} $\\mathbin{{\\pm}}$ {part.rmse.std(ddof=1):.3f}"
            cliff = f"{part.cliff_rmse.mean():.3f} $\\mathbin{{\\pm}}$ {part.cliff_rmse.std(ddof=1):.3f}"
            lines.append(f"{labels[model]} & {overall} & {cliff}\\\\")
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        (ROOT / "paper/tables/dablander_external.tex").write_text("\n".join(lines) + "\n")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
