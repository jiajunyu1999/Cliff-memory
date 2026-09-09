from __future__ import annotations

"""Audit chemistry-side neighborhoods for frozen TAPER test cases."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from rdkit import Chem

from figure_style import COLORS, clean_axis, panel_label, save_figure, set_publication_style
from molecule_drawing import render_aligned_triplet_grid
from molcliff.data import load_moleculeace
from validate_moleculeace_offtarget_kernel import molecule_mapping
from validate_moleculeace_operator_action_kernel import exact_kernel


STRATA = (
    ("sparse response", 1, 2),
    ("moderate response", 3, 7),
    ("dense response", 8, np.inf),
)


def select_cases(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions[predictions.cliff_mol.astype(bool)].copy()
    frame["heavy_atoms"] = frame.smiles.map(
        lambda value: Chem.MolFromSmiles(str(value)).GetNumHeavyAtoms()
    )
    frame = frame[
        frame.heavy_atoms.le(30)
        & frame.absolute_error_reduction.gt(0)
        & frame.nearest_chemistry_similarity.ge(0.65)
    ]
    rows = []
    used_targets: set[str] = set()
    for name, lower, upper in STRATA:
        candidates = frame[
            frame.observed_response_features.ge(lower)
            & frame.observed_response_features.le(upper)
            & ~frame.dataset.isin(used_targets)
        ].sort_values(
            ["absolute_error_reduction", "dataset", "smiles"],
            ascending=[False, True, True],
        )
        if candidates.empty:
            raise RuntimeError(f"No retrieval case satisfies the frozen {name} rule")
        chosen = candidates.iloc[0].copy()
        chosen["retrieval_stratum"] = name
        rows.append(chosen)
        used_targets.add(str(chosen.dataset))
    return pd.DataFrame(rows)


def retrieve(case: pd.Series, count: int = 2) -> list[dict]:
    benchmark = load_moleculeace(str(case.dataset)).reset_index(drop=True)
    fit = benchmark[benchmark.split.eq("train")].reset_index(drop=True)
    query = benchmark[
        benchmark.split.eq("test") & benchmark.smiles.eq(str(case.smiles))
    ].reset_index(drop=True)
    if len(query) != 1:
        raise RuntimeError(f"Could not locate unique held-out query for {case.dataset}")
    _, similarity = exact_kernel(fit, query)
    indices = np.argsort(-similarity[0], kind="stable")[:count]
    mapping = molecule_mapping(str(case.dataset))
    query_smiles = str(query.iloc[0].smiles)
    rows = []
    for rank, index in enumerate(indices, start=1):
        neighbor = fit.iloc[int(index)]
        rows.append({
            "dataset": str(case.dataset),
            "retrieval_stratum": str(case.retrieval_stratum),
            "query_molecule_id": mapping.get(query_smiles, ""),
            "query_smiles": query_smiles,
            "query_target": float(case.target),
            "prediction_svm": float(case.prediction_svm),
            "prediction_ours": float(case.prediction_ours),
            "absolute_error_svm": float(abs(case.prediction_svm - case.target)),
            "absolute_error_ours": float(abs(case.prediction_ours - case.target)),
            "absolute_error_reduction": float(case.absolute_error_reduction),
            "observed_response_features": int(case.observed_response_features),
            "rank": rank,
            "retrieved_molecule_id": mapping.get(str(neighbor.smiles), ""),
            "retrieved_smiles": str(neighbor.smiles),
            "retrieved_target": float(neighbor.target),
            "chemistry_kernel_similarity": float(similarity[0, int(index)]),
        })
    return rows


def crop_whitespace(image: Image.Image, padding: int = 8) -> Image.Image:
    array = np.asarray(image.convert("RGB"))
    mask = np.any(array < 248, axis=2)
    if not mask.any():
        return image
    yy, xx = np.where(mask)
    return image.crop((
        max(0, int(xx.min()) - padding),
        max(0, int(yy.min()) - padding),
        min(array.shape[1], int(xx.max()) + padding + 1),
        min(array.shape[0], int(yy.max()) + padding + 1),
    ))


def split_triplet(image: Image.Image) -> list[Image.Image]:
    width, height = image.size
    panel_width = width // 3
    return [
        crop_whitespace(image.crop((i * panel_width, 0, (i + 1) * panel_width, height)))
        for i in range(3)
    ]


def draw(selected: pd.DataFrame, neighbors: pd.DataFrame, output: Path) -> None:
    set_publication_style()
    fig = plt.figure(figsize=(8.55, 4.75), facecolor="white")
    grid = fig.add_gridspec(
        3, 2, left=.035, right=.988, top=.955, bottom=.088,
        width_ratios=[4.35, .72], hspace=.25, wspace=.10,
    )
    max_error = 0.0
    for row in selected.itertuples(index=False):
        max_error = max(max_error, abs(row.prediction_svm-row.target),
                        abs(row.prediction_ours-row.target))
    max_error = np.ceil((max_error + .15) * 2) / 2

    for row_index, case in enumerate(selected.itertuples(index=False)):
        part = neighbors[neighbors.dataset.eq(case.dataset)].sort_values("rank")
        query_id = str(part.iloc[0].query_molecule_id)
        records = list(part.itertuples(index=False))
        mol_ax = fig.add_subplot(grid[row_index, 0])
        image = render_aligned_triplet_grid(
            str(case.smiles), [str(record.retrieved_smiles) for record in records],
            panel_size=(1080, 540),
        )
        mol_ax.set_xlim(0, 3); mol_ax.set_ylim(0, 1.22); mol_ax.axis("off")
        for x0, panel in zip((.015, .350, .685), split_triplet(image)):
            inset = mol_ax.inset_axes([x0, .185, .300, .620], transform=mol_ax.transAxes)
            inset.imshow(panel, interpolation="lanczos", aspect="equal")
            inset.axis("off")

        count = int(case.observed_response_features)
        unit = "response" if count == 1 else "responses"
        target = str(case.dataset).replace("_", " · ")
        mol_ax.text(
            .02, 1.135, "abc"[row_index], transform=mol_ax.transAxes,
            ha="left", va="center", fontsize=8.6, fontweight="semibold", color="#222222",
        )
        mol_ax.text(
            .075, 1.135, f"{target}   |   {case.retrieval_stratum}; {count} {unit}",
            transform=mol_ax.transAxes, ha="left", va="center", fontsize=7.4,
            fontweight="semibold", color="#222222",
        )
        mol_ax.plot([0, 1], [1.065, 1.065], transform=mol_ax.transAxes,
                    color="#BDBDBD", lw=0.65, clip_on=False)
        for x, label in zip((.165, .500, .835), ("Query", "Neighbor 1", "Neighbor 2")):
            mol_ax.text(x, .880, label, transform=mol_ax.transAxes,
                        ha="center", va="center", fontsize=6.9, color="#222222",
                        zorder=8)
            mol_ax.plot([x-.105, x+.105], [.842, .842], transform=mol_ax.transAxes,
                        color="#444444", lw=.55, zorder=8, clip_on=False)

        ids = [query_id, *[str(record.retrieved_molecule_id) for record in records]]
        values = [
            f"observed y {case.target:.2f}",
            f"Kchem {records[0].chemistry_kernel_similarity:.3f}  ·  y {records[0].retrieved_target:.2f}",
            f"Kchem {records[1].chemistry_kernel_similarity:.3f}  ·  y {records[1].retrieved_target:.2f}",
        ]
        for x, identifier, value in zip((.5, 1.5, 2.5), ids, values):
            mol_ax.text(x, .115, identifier, ha="center", va="center", fontsize=6.5,
                        color="#262626")
            mol_ax.text(x, .035, value, ha="center", va="center", fontsize=6.2,
                        color="#666666")

        err = fig.add_subplot(grid[row_index, 1])
        base_error = abs(case.prediction_svm-case.target)
        ours_error = abs(case.prediction_ours-case.target)
        err.text(.02, 1.04, "abs. error", transform=err.transAxes,
                 ha="left", va="center", fontsize=6.8, color="#222222")
        y_base, y_ours = .56, .34
        err.plot([0, max_error], [y_base, y_base], color="#E6E6E6", lw=.65, zorder=0)
        err.plot([0, max_error], [y_ours, y_ours], color="#E6E6E6", lw=.65, zorder=0)
        err.plot([ours_error, base_error], [.45, .45],
                 color="#BDBDBD", lw=1.0, zorder=1)
        err.annotate(
            "", xy=(ours_error, .45), xytext=(base_error, .45),
            arrowprops={"arrowstyle": "-|>", "color": "#BDBDBD", "lw": .8,
                        "mutation_scale": 6.5},
            zorder=2,
        )
        err.scatter(base_error, y_base, s=26, marker="o", facecolor="white",
                    edgecolor="#4B4B4B", linewidth=.85, zorder=3)
        err.scatter(ours_error, y_ours, s=28, marker="o", facecolor="#00A087",
                    edgecolor="#00735E", linewidth=.65, zorder=4)
        if row_index == 0:
            err.text(.02, y_base, "ECFP-SVM", transform=err.get_yaxis_transform(),
                     ha="left", va="center", fontsize=5.9, color="#4B4B4B")
            err.text(.02, y_ours, "TAPER", transform=err.get_yaxis_transform(),
                     ha="left", va="center", fontsize=5.9, color="#00735E")
        err.text(.98, .77, rf"$\Delta |e|=-{case.absolute_error_reduction:.2f}$",
                 transform=err.transAxes, ha="right", va="center", fontsize=6.8,
                 color="#00735E", fontweight="semibold")
        err.text(.98, y_base, f"{base_error:.2f}",
                 transform=err.get_yaxis_transform(), ha="right", va="center",
                 fontsize=6.1, color="#4B4B4B")
        err.text(.98, y_ours, f"{ours_error:.2f}",
                 transform=err.get_yaxis_transform(), ha="right", va="center",
                 fontsize=6.1, color="#00735E")
        err.set_xlim(-.08, max_error); err.set_ylim(.12, .86); err.set_yticks([])
        err.set_xticks(np.arange(0, max_error + .01, 1.0))
        if row_index < 2:
            err.tick_params(axis="x", bottom=False, labelbottom=False)
        else:
            err.set_xlabel("absolute error", labelpad=2, fontsize=7.2)
        clean_axis(err, grid="x")
        err.spines["left"].set_visible(False)
    fig.savefig(output.with_suffix(".png"), dpi=420, bbox_inches="tight", facecolor="white")
    save_figure(fig, output, dpi=360)


def write_table(neighbors: pd.DataFrame, output: Path) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Closest structural neighbors for the held-out examples in Figure~\ref{fig:retrieval-cases}. Ranking uses only the frozen chemistry kernel against official-training molecules.}",
        r"\label{tab:retrieval-cases}",
        r"\small",
        r"\begin{tabular}{llrlrrr}",
        r"\toprule",
        r"Target & Query & Rank & Structural neighbor & $K_{chem}$ & Neighbor $y$ & Query error: base$\to$ours\\",
        r"\midrule",
    ]
    for row in neighbors.itertuples(index=False):
        target = str(row.dataset).replace("_", r"\_")
        lines.append(
            f"{target} & {row.query_molecule_id} & {row.rank} & "
            f"{row.retrieved_molecule_id} & {row.chemistry_kernel_similarity:.3f} & "
            f"{row.retrieved_target:.2f} & {row.absolute_error_svm:.2f}$\\to${row.absolute_error_ours:.2f}\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions", type=Path,
        default=Path("reports/results/case_studies_final/all_test_predictions.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("reports/results/retrieval_case_studies"),
    )
    parser.add_argument(
        "--figure", type=Path,
        default=Path("paper/figures/fig_retrieval_cases.pdf"),
    )
    parser.add_argument(
        "--table", type=Path,
        default=Path("paper/tables/retrieval_case_details.tex"),
    )
    args = parser.parse_args()
    predictions = pd.read_csv(args.predictions)
    selected = select_cases(predictions)
    neighbors = pd.DataFrame(
        row for _, case in selected.iterrows() for row in retrieve(case)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_dir / "selected_queries.csv", index=False)
    neighbors.to_csv(args.output_dir / "retrieved_neighbors.csv", index=False)
    draw(selected, neighbors, args.figure)
    write_table(neighbors, args.table)
    summary = {
        "queries": len(selected),
        "retrieved_neighbors": len(neighbors),
        "targets": selected.dataset.tolist(),
        "mean_absolute_error_reduction": float(selected.absolute_error_reduction.mean()),
        "selection": "best positive cliff rescue per sparse/moderate/dense response stratum; <=30 heavy atoms; chemistry similarity >=0.65",
        "retrieval": "top-2 official-training molecules under the frozen collision-free chemistry kernel",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
