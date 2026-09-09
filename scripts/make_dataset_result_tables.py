from __future__ import annotations

"""Build manuscript tables from immutable benchmark and result artifacts."""

import json
from pathlib import Path

import pandas as pd

from molcliff.data import (
    EXPANDED_CLIFF_ROOT,
    MOLECULEACE_DATASETS,
    MOLECULEACE_ROOT,
)


ROOT = Path(__file__).resolve().parents[1]
TABLE_ROOT = ROOT / "paper" / "tables"
EXTERNAL_DATASETS = (
    "CHEMBL2311243_IC50",
    "CHEMBL2598_IC50",
    "CHEMBL3024_IC50",
    "CHEMBL4685_IC50",
    "CHEMBL5014_IC50",
)


def tex(value: object) -> str:
    return str(value).replace("_", r"\_")


def dataset_frame(names: tuple[str, ...], root: Path) -> pd.DataFrame:
    return pd.concat(
        [pd.read_csv(root / f"{name}.csv").assign(dataset=name) for name in names],
        ignore_index=True,
    )


def write_dataset_summary() -> None:
    mace = dataset_frame(MOLECULEACE_DATASETS, MOLECULEACE_ROOT)
    external = dataset_frame(EXTERNAL_DATASETS, EXPANDED_CLIFF_ROOT)
    acnet = json.loads((ROOT / "data" / "acnet" / "generated" / "MMP_AC.json").read_text())
    acnet_smiles = {
        row[key]
        for records in acnet.values()
        for row in records
        for key in ("SMILES1", "SMILES2")
    }
    acnet_pairs = sum(len(records) for records in acnet.values())
    acnet_positive = sum(
        sum(int(row["Value"]) for row in records) for records in acnet.values()
    )
    acnet_result = pd.read_csv(ROOT / "outputs" / "acnet_cross_baselines" / "summary.csv")
    acnet_result = acnet_result[
        acnet_result.feature_mode.eq("endpoint")
        & acnet_result.model.eq("extratrees_ecfp")
    ]
    split = acnet_result[["n_train", "n_valid", "n_test"]].sum()

    rows = [
        (
            "MoleculeACE", 30, mace.smiles.nunique(), len(mace),
            int(mace.cliff_mol.astype(bool).sum()),
            int(mace.split.eq("train").sum()), None, int(mace.split.eq("test").sum()),
        ),
        (
            "GraphCliff external", 5, external.smiles.nunique(), len(external),
            int(external.cliff_mol.astype(bool).sum()),
            int(external.split.eq("train").sum()), None, int(external.split.eq("test").sum()),
        ),
        (
            "ACNet", len(acnet), len(acnet_smiles), acnet_pairs, acnet_positive,
            int(split.n_train), int(split.n_valid), int(split.n_test),
        ),
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Dataset statistics. A supervised unit is one activity record for the regression collections and one matched molecular pair for ACNet. The event column denotes cliff molecules for regression and positive pairs for ACNet; percentages are relative to supervised units.}",
        r"\label{tab:datasets}",
        r"\small",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Collection & Targets & Unique mol. & Units & Events (\%) & Train & Validation & Test\\",
        r"\midrule",
    ]
    for name, targets, molecules, units, events, train, valid, test in rows:
        event_cell = f"{events:,} ({100.0 * events / units:.1f})"
        valid_cell = "--" if valid is None else f"{valid:,}"
        lines.append(
            f"{name} & {targets:,} & {molecules:,} & {units:,} & {event_cell} & "
            f"{train:,} & {valid_cell} & {test:,}\\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    (TABLE_ROOT / "dataset_summary.tex").write_text("\n".join(lines) + "\n")


def _parse_scage_table(start_after: int) -> pd.DataFrame:
    supplement = (ROOT / "data" / "scage_supplementary.txt").read_text().splitlines()
    start = next(
        i for i, line in enumerate(supplement)
        if i > start_after and line.startswith("CHEMBL1862_Ki")
    )
    rows: list[list[object]] = []
    for line in supplement[start:]:
        fields = line.split()
        if len(fields) == 9 and fields[0].startswith("CHEMBL"):
            rows.append([fields[0], *map(float, fields[1:])])
        if len(rows) == 30:
            break
    if len(rows) != 30:
        raise RuntimeError("Could not parse all 30 SCAGE supplement rows")
    return pd.DataFrame(
        rows,
        columns=[
            "dataset", "AFP", "CNN", "GAT", "GCN", "MPNN",
            "ImageMol", "GEM", "SCAGE",
        ],
    )


def _mean_sd(values: pd.Series) -> str:
    values = pd.Series(values, dtype=float).dropna()
    return f"{values.mean():.3f} $\\mathbin{{\\pm}}$ {values.std(ddof=1):.3f}"


def _mean_sd_pair(frame: pd.DataFrame) -> tuple[str, str]:
    return _mean_sd(frame.rmse), _mean_sd(frame.cliff_rmse)


def write_cross_dataset_macro() -> None:
    """Write task-separated macro comparisons without cross-task N/A cells."""
    official_ours = pd.read_csv(
        ROOT / "outputs" / "responsekernel_final_official_test_v1" / "summary.test.csv"
    )
    external = pd.read_csv(
        ROOT / "outputs" / "responsekernel_final_external_cv5_seed42" / "per_target.cv5.csv"
    )
    baseline_regression = pd.concat([
        pd.read_csv(ROOT / "outputs" / "cross_dataset_regression_baselines" / "summary.csv"),
        pd.read_csv(ROOT / "outputs" / "cross_dataset_regression_baselines_recent" / "summary.csv"),
        pd.read_csv(ROOT / "outputs" / "baseline_runs" / "moleculeace_classical_recent.csv"),
        pd.read_csv(ROOT / "outputs" / "baseline_runs" / "moleculeace_graph_full.csv"),
    ], ignore_index=True)
    baseline_regression = (
        baseline_regression.groupby(["collection", "model", "dataset"])[
            ["rmse", "cliff_rmse"]
        ]
        .mean()
        .reset_index()
    )
    acnet = pd.concat([
        pd.read_csv(ROOT / "outputs" / "acnet_cross_baselines" / "summary.csv"),
        pd.read_csv(ROOT / "outputs" / "acnet_cross_baselines_recent" / "summary.csv"),
    ], ignore_index=True)
    graph_external = pd.read_csv(
        ROOT / "outputs" / "baseline_runs" / "external_graph_full.csv"
    )
    native_external = pd.read_csv(
        ROOT / "reports" / "results" / "graphcliff_external_native_moleculeace_baselines.csv"
    )
    graph_acnet = pd.concat([
        pd.read_csv(ROOT / "outputs" / "cross_dataset_graph_baselines" / "acnet_gpu0.csv"),
        pd.read_csv(ROOT / "outputs" / "cross_dataset_graph_baselines" / "acnet_gpu1.csv"),
        pd.read_csv(ROOT / "outputs" / "baseline_runs" / "acnet_chemprop_full.csv"),
    ], ignore_index=True)
    expected_graph_models = {
        "chemprop", "gine_residual", "gine_nodenorm", "gine_pairnorm", "graphcliff"
    }
    if set(graph_external.model) != expected_graph_models:
        raise RuntimeError("incomplete GraphCliff-external graph baseline set")
    if graph_external.groupby("model").size().to_dict() != {
        model: 25 for model in expected_graph_models
    }:
        raise RuntimeError("GraphCliff-external graph baselines must contain 5 assays x 5 folds")
    if set(graph_acnet.model) != expected_graph_models:
        raise RuntimeError("incomplete ACNet graph baseline set")
    if graph_acnet.groupby("model").target.nunique().to_dict() != {
        model: 190 for model in expected_graph_models
    } or graph_acnet[["auc", "ap"]].isna().any().any():
        raise RuntimeError("ACNet graph baselines must contain 190 finite target results per model")

    display = {
        "linear_ecfp": "Linear ECFP",
        "svm_ecfp": "RBF-SVM ECFP",
        "hist_gradient_boosting_ecfp": "HistGBM ECFP",
        "mlp_ecfp": "MLP ECFP",
        "extratrees_ecfp": "ExtraTrees ECFP",
        "random_forest_ecfp": "Random forest ECFP",
    }
    shared_regression: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    shared_acnet: dict[str, pd.DataFrame] = {}
    for model, name in display.items():
        mace = baseline_regression[
            baseline_regression.collection.eq("MoleculeACE")
            & baseline_regression.model.eq(model)
        ]
        graphcliff = baseline_regression[
            baseline_regression.collection.eq("GraphCliff external")
            & baseline_regression.model.eq(model)
        ]
        endpoint = acnet[
            acnet.feature_mode.eq("endpoint") & acnet.model.eq(model)
        ]
        shared_regression[name] = (mace, graphcliff)
        shared_acnet[name] = endpoint

    candidate_heads = acnet[acnet.feature_mode.eq("endpoint_action")]
    selected_head = (
        candidate_heads.groupby("model").valid_ap.mean().sort_values(ascending=False).index[0]
    )
    if selected_head != "extratrees_ecfp":
        raise RuntimeError(f"unexpected validation-selected ACNet head: {selected_head}")
    response_acnet = candidate_heads[candidate_heads.model.eq(selected_head)]

    mace_rows = [
        ("Linear ECFP", *_mean_sd_pair(shared_regression["Linear ECFP"][0])),
        ("ExtraTrees ECFP", *_mean_sd_pair(shared_regression["ExtraTrees ECFP"][0])),
        ("Random forest ECFP", *_mean_sd_pair(shared_regression["Random forest ECFP"][0])),
        ("ECFP-SVM", *_mean_sd_pair(shared_regression["RBF-SVM ECFP"][0])),
        ("HistGBM ECFP", *_mean_sd_pair(shared_regression["HistGBM ECFP"][0])),
        ("MLP ECFP", *_mean_sd_pair(shared_regression["MLP ECFP"][0])),
        ("Chemprop", *_mean_sd_pair(baseline_regression[
            baseline_regression.collection.eq("MoleculeACE")
            & baseline_regression.model.eq("chemprop")
        ])),
        ("GINE + NodeNorm", *_mean_sd_pair(baseline_regression[
            baseline_regression.collection.eq("MoleculeACE")
            & baseline_regression.model.eq("gine_nodenorm")
        ])),
        ("GraphCliff", *_mean_sd_pair(baseline_regression[
            baseline_regression.collection.eq("MoleculeACE")
            & baseline_regression.model.eq("graphcliff")
        ])),
    ]
    mace_rows.append((r"\textbf{\method}",
                      r"\textbf{" + _mean_sd(official_ours.rmse) + "}",
                      r"\textbf{" + _mean_sd(official_ours.cliff_rmse) + "}"))

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Task-separated target-macro comparisons. Panel A contains only locally evaluated MoleculeACE methods. Panel B re-runs every Panel-A architecture on the five GraphCliff external assays under the same train-only five-fold protocol, and additionally reports the vendored MoleculeACE AFP, CNN, GAT, GCN, and MPNN architectures. Panel C evaluates a matched-pair \method{}-edit instantiation on ACNet. Mean $\mathbin{\pm}$ target-level standard deviation is reported when per-target values are available. Lower RMSE is better; higher ROC--AUC and AP are better.}",
        r"\label{tab:main}",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{1.02}",
        r"\begin{tabular*}{0.86\textwidth}{@{\extracolsep{\fill}}lrr}",
        r"\toprule",
        r"\multicolumn{3}{l}{\textit{A. MoleculeACE regression (30 targets)}}\\",
        r"Method & RMSE & Cliff RMSE\\",
        r"\midrule",
    ]
    lines.extend(" & ".join(row) + r"\\" for row in mace_rows)
    lines.extend([
        r"\midrule",
        r"\multicolumn{3}{l}{\textit{B. GraphCliff external regression (5 assays)}}\\",
        r"Method & RMSE & Cliff RMSE\\",
        r"\midrule",
    ])
    for name, (_, graphcliff) in shared_regression.items():
        lines.append(f"{name} & {_mean_sd(graphcliff.rmse)} & {_mean_sd(graphcliff.cliff_rmse)}\\\\")
    graph_display = {
        "chemprop": "Chemprop",
        "gine_nodenorm": "GINE + NodeNorm",
        "graphcliff": "GraphCliff encoder",
    }
    for model, name in graph_display.items():
        target = graph_external[graph_external.model.eq(model)].groupby("dataset")[["rmse", "cliff_rmse"]].mean()
        lines.append(f"{name} & {_mean_sd(target.rmse)} & {_mean_sd(target.cliff_rmse)}\\\\")
    native_display = {"afp": "AFP", "cnn": "CNN", "gat": "GAT", "gcn": "GCN", "mpnn": "MPNN"}
    for model, name in native_display.items():
        target = native_external[native_external.model.eq(model)].groupby("dataset")[["rmse", "cliff_rmse"]].mean()
        lines.append(f"{name} & {_mean_sd(target.rmse)} & {_mean_sd(target.cliff_rmse)}\\\\")
    lines.append(r"\textbf{\method} & \textbf{" + _mean_sd(external.responsekernel_final_rmse)
                 + r"} & \textbf{" + _mean_sd(external.responsekernel_final_cliff_rmse) + r"}\\")
    lines.extend([
        r"\midrule",
        r"\multicolumn{3}{l}{\textit{C. ACNet pair classification (190 targets)}}\\",
        r"Method & ROC--AUC & AP\\",
        r"\midrule",
    ])
    for name, endpoint in shared_acnet.items():
        lines.append(f"{name} & {_mean_sd(endpoint.auc)} & {_mean_sd(endpoint.ap)}\\\\")
    acnet_graph_display = {
        "chemprop": "Chemprop",
        "gine_pairnorm": "GINE + PairNorm",
        "graphcliff": "GraphCliff encoder",
    }
    for model, name in acnet_graph_display.items():
        target = graph_acnet[graph_acnet.model.eq(model)]
        lines.append(f"{name} + pair head & {_mean_sd(target.auc)} & {_mean_sd(target.ap)}\\\\")
    lines.append(r"\textbf{\method{}-edit} & \textbf{" + _mean_sd(response_acnet.auc)
                 + r"} & \textbf{" + _mean_sd(response_acnet.ap) + r"}\\")
    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\end{table*}"])
    (TABLE_ROOT / "cross_dataset_macro.tex").write_text("\n".join(lines) + "\n")


def paired_blocks(rows: list[list[str]], header: list[str], caption: str, label: str) -> str:
    midpoint = (len(rows) + 1) // 2
    left, right = rows[:midpoint], rows[midpoint:]
    while len(right) < len(left):
        right.append([""] * len(header))
    alignment = "l" + "r" * (len(header) - 1)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\scriptsize",
        f"\\begin{{tabular}}{{{alignment}@{{\\hspace{{1.7em}}}}{alignment}}}",
        r"\toprule",
        " & ".join(header + header) + r"\\",
        r"\midrule",
    ]
    for first, second in zip(left, right):
        lines.append(" & ".join(first + second) + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def write_moleculeace_results() -> None:
    sources = [
        ROOT / "outputs" / "cross_dataset_regression_baselines" / "summary.csv",
        ROOT / "outputs" / "baseline_runs" / "moleculeace_classical_recent.csv",
        ROOT / "outputs" / "baseline_runs" / "moleculeace_graph_full.csv",
        ROOT / "outputs" / "baseline_smoke" / "mtpnet_epoch50" / "summary.mtpnet_epoch50.csv",
        ROOT / "outputs" / "taper_profile_controls_cv5" / "summary.cv5.csv",
    ]
    pieces = []
    for path in sources:
        frame = pd.read_csv(path)
        if "collection" in frame:
            frame = frame[frame.collection.eq("MoleculeACE")]
        if path.name == "summary.cv5.csv":
            frame = frame[frame.model.eq("taper_strict_exclusion")]
        pieces.append(frame[["dataset", "model", "n_test", "rmse"]])
    local = pd.concat(pieces, ignore_index=True)
    target = (
        local.groupby(["dataset", "model"], as_index=False)
        .agg(n_cv=("n_test", "sum"), rmse=("rmse", "mean"))
    )
    display = {
        "linear_ecfp": "Linear",
        "random_forest_ecfp": "RF",
        "extratrees_ecfp": "ExtraTrees",
        "svm_ecfp": "SVM",
        "hist_gradient_boosting_ecfp": "HGB",
        "mlp_ecfp": "MLP",
        "chemprop": "Chemprop",
        "gine_nodenorm": "GINE-Node",
        "graphcliff": "GraphCliff",
        "mtpnet_released_epoch50": "MTPNet",
        "taper_strict_exclusion": r"\method",
    }
    order = list(display)
    missing = set(order) - set(target.model)
    if missing or target[target.model.isin(order)].dataset.nunique() != 30:
        raise RuntimeError(
            "Table 3 requires selected local MoleculeACE methods on all 30 targets: "
            f"missing {sorted(missing)}"
        )
    wide = target.pivot(index="dataset", columns="model", values="rmse").loc[:, order]
    n_cv = target.groupby("dataset").n_cv.max().astype(int)
    merged = wide.reset_index().sort_values("dataset")
    baseline_columns = order[:-1]
    method_columns = order

    def ranked_cell(value: float, row_values: list[float]) -> str:
        ordered = sorted(set(row_values))
        if value == ordered[0]:
            return f"\\textbf{{{value:.3f}}}"
        if len(ordered) > 1 and value == ordered[1]:
            return f"\\underline{{{value:.3f}}}"
        return f"{value:.3f}"

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Local five-fold RMSE on all 30 MoleculeACE targets. Every method in this table is evaluated locally; fold results are averaged within each target before ranking. Lower is better; bold and underline denote the best and second-best method in each row. Reduction is relative to the strongest local baseline excluding \method.}",
        r"\label{tab:moleculeace_all}",
        r"\tiny",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\renewcommand{\arraystretch}{1.02}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrrrrrrr}",
        r"\toprule",
        r"Target & CV $n$ & Linear & RF & ExtraTrees & SVM & HGB & MLP & Chemprop & GINE-Node & GraphCliff & MTPNet & \method & Reduction (\%) $\uparrow$\\",
        r"\midrule",
    ]
    for _, row in merged.iterrows():
        values = {column: float(row[column]) for column in baseline_columns}
        best = min(values.values())
        ours_value = float(row["taper_strict_exclusion"])
        gain = 100.0 * (best - ours_value) / best
        row_values = [float(row[column]) for column in method_columns]
        baseline_cells = [ranked_cell(value, row_values) for value in values.values()]
        ours_cell = ranked_cell(ours_value, row_values)
        lines.append(
            " & ".join([
                tex(row["dataset"]), f"{int(n_cv[row['dataset']]):,}", *baseline_cells,
                ours_cell, f"{gain:+.1f}",
            ]) + r"\\"
        )
    macro = {column: float(merged[column].mean()) for column in method_columns}
    macro_best = min(macro[column] for column in baseline_columns)
    macro_gain = 100.0 * (macro_best - macro["taper_strict_exclusion"]) / macro_best
    ranks = merged[method_columns].rank(axis=1, method="average")
    wins = merged[method_columns].idxmin(axis=1).value_counts()
    macro_values = [macro[column] for column in method_columns]
    lines.extend([
        r"\midrule",
        f"Macro mean & {int(n_cv.sum()):,} & "
        + " & ".join(ranked_cell(macro[column], macro_values) for column in method_columns)
        + f" & {macro_gain:+.1f}\\\\",
        "Best count & -- & "
        + " & ".join(
            f"\\textbf{{{int(wins.get(column, 0))}}}"
            if int(wins.get(column, 0)) == int(wins.max())
            else f"{int(wins.get(column, 0))}"
            for column in method_columns
        )
        + r" & --\\",
        "Mean rank & -- & "
        + " & ".join(
            f"\\textbf{{{ranks[column].mean():.2f}}}"
            if ranks[column].mean() == ranks.mean().min()
            else f"{ranks[column].mean():.2f}"
            for column in method_columns
        )
        + r" & --\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
    ])
    (TABLE_ROOT / "moleculeace_all_results.tex").write_text("\n".join(lines) + "\n")


def write_external_results() -> None:
    result = pd.read_csv(
        ROOT / "outputs" / "responsekernel_final_external_cv5_seed42" / "per_target.cv5.csv"
    ).sort_values("dataset")
    counts = dataset_frame(EXTERNAL_DATASETS, EXPANDED_CLIFF_ROOT).groupby("dataset").size()
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Five-fold RMSE on every external GraphCliff assay. The baseline is the collision-free chemistry kernel using the same regressor and folds.}",
        r"\label{tab:external_all}",
        r"\scriptsize",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Assay & $n$ & Chemistry & Ours & $\Delta$\%\\",
        r"\midrule",
    ]
    for row in result.itertuples(index=False):
        lines.append(
            f"{tex(row.dataset)} & {counts[row.dataset]:,} & "
            f"{row.collision_free_binary_count_kernel_rmse:.3f} & "
            f"{row.responsekernel_final_rmse:.3f} & {100*row.relative_gain_rmse:+.1f}\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (TABLE_ROOT / "external_all_results.tex").write_text("\n".join(lines) + "\n")


def write_acnet_results() -> None:
    result = pd.read_csv(ROOT / "outputs" / "acnet_cross_baselines" / "summary.csv")
    # Choose one classification head globally from validation macro AP before
    # comparing its endpoint and endpoint-plus-edit views on official tests.
    action = result[result.feature_mode.eq("endpoint_action")]
    selected_head = action.groupby("model").valid_ap.mean().idxmax()
    result = result[result.model.eq(selected_head)].copy()
    if selected_head != "extratrees_ecfp" or result.target.nunique() != 190:
        raise RuntimeError(
            f"unexpected ACNet selection: {selected_head}, "
            f"{result.target.nunique()} targets"
        )
    pivot = result.pivot(index="target", columns="feature_mode", values=["n", "auc"])
    rows = []
    for target, row in pivot.sort_index().iterrows():
        base = float(row[("auc", "endpoint")])
        ours = float(row[("auc", "endpoint_action")])
        rows.append([
            str(target), f"{int(row[('n', 'endpoint')]):,}", f"{base:.3f}",
            f"{ours:.3f}", f"{ours-base:+.3f}",
        ])
    block_size = (len(rows) + 2) // 3
    blocks = [rows[i * block_size:(i + 1) * block_size] for i in range(3)]
    for block in blocks:
        while len(block) < block_size:
            block.append([""] * 5)
    header = r"Target & Pairs & Endpoint & Ours & $\Delta$"
    lines = [
        r"\begin{longtable}{@{}rrrrr@{\hspace{1.5em}}rrrrr@{\hspace{1.5em}}rrrrr@{}}",
        r"\caption{Test ROC--AUC for all 190 ACNet target tasks using the ExtraTrees head selected globally by validation macro average precision. Endpoint fingerprints form the baseline; Ours adds the explicit edit representation.}\label{tab:acnet_all}\\",
        r"\toprule",
        " & ".join([header] * 3) + r"\\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{15}{c}{\tablename\ \thetable{} -- continued}\\",
        r"\toprule",
        " & ".join([header] * 3) + r"\\",
        r"\midrule",
        r"\endhead",
        r"\midrule\multicolumn{15}{r}{Continued on next page}\\\endfoot",
        r"\bottomrule\endlastfoot",
    ]
    lines.extend(" & ".join(a + b + c) + r"\\" for a, b, c in zip(*blocks))
    lines.append(r"\end{longtable}")
    (TABLE_ROOT / "acnet_all_results.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    write_dataset_summary()
    write_cross_dataset_macro()
    write_moleculeace_results()
    write_external_results()
    write_acnet_results()
    print(json.dumps({
        "tables": sorted(path.name for path in TABLE_ROOT.glob("*.tex")),
        "moleculeace_targets": len(MOLECULEACE_DATASETS),
        "external_targets": len(EXTERNAL_DATASETS),
        "acnet_targets": 190,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
