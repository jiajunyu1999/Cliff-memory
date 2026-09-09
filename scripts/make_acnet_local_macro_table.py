from __future__ import annotations

"""Write an auditable 190-target ACNet macro table from local predictions."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "linear_ecfp": "Linear ECFP", "random_forest_ecfp": "RF-ECFP",
    "extratrees_ecfp": "ExtraTrees-ECFP", "svm_ecfp": "ECFP-SVM",
    "hist_gradient_boosting_ecfp": "HGB-ECFP", "mlp_ecfp": "MLP-ECFP",
    "chemprop": "Chemprop", "gine_pairnorm": "GINE + PairNorm",
    "graphcliff": "GraphCliff",
}

def fmt(values: pd.Series) -> str:
    return f"{values.mean():.3f} $\\mathbin{{\\pm}}$ {values.std(ddof=1):.3f}"

def main() -> None:
    frame = pd.concat([
        pd.read_csv(ROOT / "outputs/baseline_smoke/acnet_classical_all.csv"),
        pd.read_csv(ROOT / "outputs/cross_dataset_graph_baselines/acnet_gpu0.csv"),
        pd.read_csv(ROOT / "outputs/cross_dataset_graph_baselines/acnet_gpu1.csv"),
        pd.read_csv(ROOT / "outputs/baseline_runs/acnet_chemprop_full.csv"),
    ], ignore_index=True)
    counts = frame.groupby("model").target.nunique()
    if any(counts.get(model, 0) != 190 for model in LABELS):
        raise RuntimeError(f"incomplete ACNet baseline set: {counts.to_dict()}")
    if frame[["auc", "ap"]].isna().any().any():
        raise RuntimeError("ACNet baseline metrics contain missing values")
    lines = [r"\begin{table}[t]", r"\centering",
        r"\caption{Locally executed ACNet comparison over the same 190 targets. Values are target-macro mean $\mathbin{\pm}$ standard deviation. Higher is better.}",
        r"\label{tab:acnet-local-macro}", r"\scriptsize", r"\begin{tabular}{lrr}",
        r"\toprule", r"Method & ROC--AUC & AP\\", r"\midrule"]
    rows = []
    for model, label in LABELS.items():
        part = frame[frame.model.eq(model)]
        lines.append(f"{label} & {fmt(part.auc)} & {fmt(part.ap)}\\\\")
        rows.append({"model": model, "label": label, "n_targets": len(part),
            "auc_mean": part.auc.mean(), "auc_sd": part.auc.std(ddof=1),
            "ap_mean": part.ap.mean(), "ap_sd": part.ap.std(ddof=1)})
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (ROOT / "paper/tables/acnet_local_macro.tex").write_text("\n".join(lines) + "\n")
    pd.DataFrame(rows).to_csv(ROOT / "reports/results/acnet_local_macro.csv", index=False)

if __name__ == "__main__":
    main()
