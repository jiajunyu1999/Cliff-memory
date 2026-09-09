from __future__ import annotations

"""Aggregate additional, leakage-safe analyses from completed local runs.

This script deliberately consumes prediction/result artifacts rather than
inventing measurements.  It produces coverage/scaling, retrieval-only
diagnostics, cross-database availability, virtual-screening enrichment, and a
larger deterministic case-study table.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "reports/results/case_studies_final/all_test_predictions.csv"
CHEMPROP_PRED = ROOT / "outputs/chemprop_released_test/predictions.csv"
CV = ROOT / "outputs/responsekernel_final_cv5_seed7/summary.cv5.csv"
EXT = ROOT / "outputs/responsekernel_final_external_cv5_seed42/summary.cv5.csv"
OUT = ROOT / "reports/results/extended_evidence"
FIG = ROOT / "paper/figures/fig_extended_evidence.pdf"
PREVIEW = OUT / "fig_extended_evidence.png"


def bootstrap_mean(x: np.ndarray, seed: int = 42, n: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if not len(x):
        return float("nan"), float("nan")
    draws = rng.choice(x, (n, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(PRED)
    if not CHEMPROP_PRED.exists():
        raise FileNotFoundError(f"Missing locally trained Chemprop predictions: {CHEMPROP_PRED}")
    chemprop = pd.read_csv(CHEMPROP_PRED)[["dataset", "smiles", "prediction_chemprop"]]
    d = d.merge(chemprop, on=["dataset", "smiles"], how="left", validate="one_to_one")
    if d.prediction_chemprop.isna().any():
        raise RuntimeError("Chemprop prediction table does not cover the frozen screening pool")
    d["gain_abs_error"] = d["error_svm"].abs() - d["error_ours"].abs()
    d["coverage_bin"] = pd.cut(
        d["observed_response_features"],
        bins=[-1, 0, 1, 3, 10, np.inf],
        labels=["0", "1", "2–3", "4–10", ">10"],
    )

    # Cold-start / memory scaling: official test molecules are grouped only
    # after prediction; the model never sees their target labels here.
    rows = []
    for (subset, cov), g in d.assign(
        subset=np.where(d.cliff_mol.astype(bool), "cliff", "all")
    ).groupby(["subset", "coverage_bin"], observed=True):
        gain = g.gain_abs_error.to_numpy(float)
        lo, hi = bootstrap_mean(gain)
        rows.append({"analysis": "memory_scaling", "subset": subset,
                     "coverage": str(cov), "n": len(g),
                     "svm_rmse": float(np.sqrt(np.mean(g.error_svm**2))),
                     "taper_rmse": float(np.sqrt(np.mean(g.error_ours**2))),
                     "gain_abs_error": float(gain.mean()), "ci_low": lo, "ci_high": hi})
    scaling = pd.DataFrame(rows)
    scaling.to_csv(OUT / "memory_scaling.csv", index=False)

    # Retrieval-only diagnostic: nearest response similarity is already stored
    # by the local predictor.  Report whether similarity is associated with
    # improvement, without treating it as a competing trained model.
    q = d[d.observed_response_features > 0].copy()
    q["response_similarity_bin"] = pd.qcut(
        q.nearest_response_similarity.rank(method="first"), 4,
        labels=["Q1", "Q2", "Q3", "Q4"]
    )
    retrieval = q.groupby("response_similarity_bin", observed=True).agg(
        n=("gain_abs_error", "size"), mean_gain=("gain_abs_error", "mean"),
        taper_rmse=("error_ours", lambda x: float(np.sqrt(np.mean(x*x)))),
        svm_rmse=("error_svm", lambda x: float(np.sqrt(np.mean(x*x)))),
        mean_similarity=("nearest_response_similarity", "mean"),
    ).reset_index()
    retrieval.insert(0, "analysis", "profile_similarity_diagnostic")
    retrieval.to_csv(OUT / "profile_similarity_diagnostic.csv", index=False)

    # Formal retrieval controls: the same local runner was completed on all
    # 30 targets, with fit-only response vocabularies and target exclusion.
    profile_path = ROOT / "outputs/profile_only_all30/summary.validation.csv"
    profile = pd.DataFrame()
    if profile_path.exists():
        profile = pd.read_csv(profile_path)
        profile_macro = profile.groupby("model", as_index=False)[
            ["rmse", "cliff_rmse", "noncliff_rmse"]].mean()
        profile_macro.to_csv(OUT / "profile_only_baselines_macro.csv", index=False)
    else:
        profile_macro = pd.DataFrame()

    # Virtual screening on the fixed official test pool.  The hit definition
    # is target-local top 10% observed potency; EF1% and ROC-AUC are computed
    # per target, then macro-averaged.
    vs = []
    for dataset, g in d.groupby("dataset"):
        y = g.target.to_numpy(float)
        hit = (y >= np.quantile(y, .90)).astype(int)
        for name, col in [("ECFP-SVM", "prediction_svm"),
                          ("Chemprop", "prediction_chemprop"),
                          ("ResponseKernel", "prediction_ours")]:
            score = g[col].to_numpy(float)
            order = np.argsort(-score)
            n_top = max(1, int(np.ceil(.01 * len(g))))
            ef1 = hit[order[:n_top]].mean() / max(hit.mean(), 1e-12)
            auc = roc_auc_score(hit, score) if np.unique(hit).size > 1 else np.nan
            # BEDROC-like early-recognition score with fixed alpha, reported
            # as a ranking diagnostic (not used for model selection).
            ranks = np.flatnonzero(hit[order]) + 1
            alpha = 20.0
            bedroc = float(np.exp(-alpha * ranks / len(g)).sum() / max(hit.sum(), 1))
            vs.append({"dataset": dataset, "model": name, "ef1_percent": ef1,
                       "roc_auc": auc, "bedroc20_proxy": bedroc, "n": len(g),
                       "hits": int(hit.sum())})
    vs = pd.DataFrame(vs)
    vs.to_csv(OUT / "virtual_screening_enrichment.csv", index=False)

    # Cross-database audit: count usable BindingDB systems in the completed
    # external run and report target/fold-level availability.  The same table
    # also retains the number of fit-derived ChEMBL profile coordinates.
    ext = pd.read_csv(EXT)
    ext_db = ext.groupby("dataset", as_index=False).agg(
        folds=("fold", "nunique"), bindingdb_systems=("bindingdb_profile_systems", "max"),
        profile_targets=("profile_targets", "mean"),
        rmse=("rmse", "mean"), cliff_rmse=("cliff_rmse", "mean"),
    )
    ext_db.insert(0, "analysis", "bindingdb_availability")
    ext_db.to_csv(OUT / "bindingdb_availability.csv", index=False)

    # Real case-study expansion: deterministic extremes, no synthetic rows.
    cases = []
    for ds, g in d.groupby("dataset"):
        for label, part in [("largest_rescue", g.nlargest(1, "gain_abs_error")),
                            ("largest_penalty", g.nsmallest(1, "gain_abs_error"))]:
            row = part.iloc[0].to_dict(); row["case_type"] = label; cases.append(row)
    cases_df = pd.DataFrame(cases)
    cases_df.to_csv(OUT / "expanded_cases.csv", index=False)
    # Compact manuscript table: one rescue and one penalty per six diverse
    # targets.  The full 60-row machine-readable file remains available.
    chosen = cases_df.sort_values(["dataset", "case_type"]).groupby("dataset", sort=False).head(2).head(12)
    tex = [r"\begin{table*}[t]", r"\centering", r"\small",
           r"\caption{Deterministically selected held-out case-study extremes. Positive gain favours ResponseKernel.}",
           r"\begin{tabular}{llrrr}", r"\toprule", r"Target & Case & $y$ & $|e|$ gain & Response records\\", r"\midrule"]
    for _, row in chosen.iterrows():
        target_name = str(row.dataset).replace("_", "\\_")
        case_name = str(row.case_type).replace("_", " ")
        tex.append(f"{target_name} & {case_name} & {row.target:.2f} & {row.gain_abs_error:+.2f} & {int(row.observed_response_features)}\\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\label{tab:extended-cases}", r"\end{table*}"]
    (ROOT / "paper/tables/extended_cases.tex").write_text("\n".join(tex) + "\n")

    # Publication-ready four-panel figure.
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.labelsize": 10, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.25), facecolor="white")
    ax = axes[0, 0]
    for subset, color in [("all", "#4DBBD5"), ("cliff", "#E64B35")]:
        s = scaling[scaling.subset == subset]
        ax.plot(s.coverage, s.taper_rmse, marker="o", lw=1.7, ms=4.5,
                color=color, label="All" if subset == "all" else "Cliff")
    ax.set_xlabel("External response records per molecule")
    ax.set_ylabel("RMSE (lower is better)"); ax.legend(frameon=False, fontsize=8)
    ax.text(-.14, 1.05, "a", transform=ax.transAxes, fontweight="bold", fontsize=11)
    ax = axes[0, 1]
    cats = list(retrieval.response_similarity_bin.astype(str))
    vals = retrieval.mean_gain.to_numpy(float)
    ax.plot(np.arange(len(vals)), vals, color="#00A087", marker="o", lw=1.4, ms=4.5)
    ax.set_xticks(np.arange(len(vals)), cats)
    ax.axhline(0, color="#666", lw=.7, ls=(0, (3, 2)))
    ax.set_xlabel("Nearest-response similarity quartile"); ax.set_ylabel("Mean |error| gain")
    ax.text(-.14, 1.05, "b", transform=ax.transAxes, fontweight="bold", fontsize=11)
    ax = axes[1, 0]
    summary = vs.groupby("model", as_index=False)[["ef1_percent", "roc_auc", "bedroc20_proxy"]].mean()
    x = np.arange(len(summary)); width = .34
    ax.bar(x-width/2, summary.ef1_percent, width, color="#6C7A89", label="EF1%")
    ax2 = ax.twinx(); ax2.bar(x+width/2, summary.roc_auc, width, color="#00A087", label="ROC-AUC")
    ax.set_xticks(x, summary.model, rotation=18, ha="right"); ax.set_ylabel("EF1%")
    ax2.set_ylabel("ROC-AUC"); ax.text(-.14, 1.05, "c", transform=ax.transAxes, fontweight="bold", fontsize=11)
    ax = axes[1, 1]
    avail = ext_db.sort_values("profile_targets", ascending=False)
    ax.scatter(avail.profile_targets, avail.rmse, s=28, color="#8E6C8A", edgecolor="white", lw=.4)
    ax.set_xlabel("Fit-derived response coordinates"); ax.set_ylabel("External-CV RMSE")
    ax.text(-.14, 1.05, "d", transform=ax.transAxes, fontweight="bold", fontsize=11)
    for a in axes.flat:
        a.set_facecolor("white"); a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
        a.grid(axis="y", ls="--", lw=.45, alpha=.25, color="#888", zorder=0)
    fig.tight_layout(w_pad=1.3, h_pad=1.5)
    fig.savefig(FIG, bbox_inches="tight")
    fig.savefig(PREVIEW, dpi=350, bbox_inches="tight")
    summary_out = {"memory_scaling_rows": len(scaling), "retrieval_rows": len(retrieval),
                   "profile_baseline_targets": int(profile.dataset.nunique()) if not profile_macro.empty else 0,
                   "virtual_screening_rows": len(vs), "bindingdb_datasets": len(ext_db),
                   "expanded_case_rows": len(cases),
                   "virtual_screening_macro": vs.groupby("model")["ef1_percent"].mean().to_dict()}
    (OUT / "summary.json").write_text(json.dumps(summary_out, indent=2) + "\n")
    print(json.dumps(summary_out, indent=2))


if __name__ == "__main__":
    main()
