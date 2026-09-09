from __future__ import annotations

"""Generate chemically valid, context-controlled MoleculeACE case studies."""

import argparse
import json
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem

from figure_style import COLORS, clean_axis, panel_label, save_figure, set_publication_style
from molecule_drawing import aligned_molecules, render_aligned_pair_grid
from molcliff.data import load_moleculeace, moleculeace_similarity_matrix


CASE_STRATA = [
    # coverage rule, role, direction of the deterministic ranking
    ("high", "covered rescue", "largest improvement"),
    ("low", "sparse-coverage rescue", "largest improvement"),
    ("none", "no-coverage boundary", "largest degradation"),
]

TRANSFORMATION_LABELS = {
    "CHEMBL234_Ki": "N-propyl → N-methyl",
    "CHEMBL262_Ki": "OH → H",
    "CHEMBL4792_Ki": "I → F",
    "CHEMBL236_Ki": "C=O → CH₂",
}


@lru_cache(maxsize=None)
def benchmark_frame(dataset: str) -> pd.DataFrame:
    return load_moleculeace(dataset).reset_index(drop=True)


def cliff_partner(dataset: str, query_smiles: str, query_y: float) -> dict:
    """Find a true official cliff mate, preferring the most local ECFP/SMILES edit."""
    frame = benchmark_frame(dataset)
    sim, views = moleculeace_similarity_matrix(
        np.asarray([query_smiles]), frame.smiles.astype(str).to_numpy()
    )
    delta = np.abs(frame.target.to_numpy(float) - float(query_y))
    valid = ((sim[0] >= .9) & (delta >= 1.0) &
             frame.smiles.astype(str).to_numpy().__ne__(query_smiles))
    indices = np.flatnonzero(valid)
    if not len(indices):
        raise ValueError(f"No official cliff partner for {dataset}: {query_smiles}")
    ranked = []
    for index in indices:
        _, _, left_atoms, right_atoms, _, _ = aligned_molecules(
            str(frame.loc[index, "smiles"]), query_smiles
        )
        edit_atoms = len(left_atoms) + len(right_atoms)
        ranked.append((edit_atoms,
                       -max(views["ecfp"][0, index], views["smiles"][0, index]),
                       -delta[index], int(index)))
    edit_atoms, _, _, index = min(ranked)
    return {
        "partner_smiles": str(frame.loc[index, "smiles"]),
        "partner_target": float(frame.loc[index, "target"]),
        "partner_split": str(frame.loc[index, "split"]),
        "pair_delta_activity": float(delta[index]),
        "pair_consensus_similarity": float(sim[0, index]),
        "pair_ecfp4_tanimoto": float(views["ecfp"][0, index]),
        "pair_smiles_similarity": float(views["smiles"][0, index]),
        "pair_edit_atoms": int(edit_atoms),
    }


def select_main_cases(merged: pd.DataFrame) -> pd.DataFrame:
    merged = merged.copy()
    merged["heavy_atoms"] = merged.smiles.map(
        lambda s: Chem.MolFromSmiles(str(s)).GetNumHeavyAtoms()
    )
    rows = []
    for coverage, role, ranking in CASE_STRATA:
        subset = merged[merged.cliff_mol.astype(bool) & merged.heavy_atoms.le(32)].copy()
        if coverage == "high":
            subset = subset[subset.observed_response_features.ge(8)
                            & subset.absolute_error_reduction.gt(0)]
        elif coverage == "low":
            subset = subset[subset.observed_response_features.between(1, 2)
                            & subset.absolute_error_reduction.gt(0)]
        else:
            subset = subset[subset.observed_response_features.eq(0)
                            & subset.absolute_error_reduction.lt(0)]
        ascending = ranking == "largest degradation"
        subset = subset.sort_values(
            ["absolute_error_reduction", "dataset", "smiles"],
            ascending=[ascending, True, True],
        )
        chosen = None
        for _, candidate in subset.iterrows():
            partner = cliff_partner(
                str(candidate.dataset), str(candidate.smiles), float(candidate.target)
            )
            if partner["pair_edit_atoms"] <= 2:
                chosen = candidate.to_dict()
                chosen.update(partner)
                break
        if chosen is None:
            raise ValueError(f"No simple true cliff pair found for the {coverage} stratum")
        chosen.update({
            "case_role": role,
            "selection_rule": f"{ranking}; at most two unmatched MCS atoms",
            "transformation": TRANSFORMATION_LABELS.get(str(chosen["dataset"]), "local edit"),
        })
        rows.append(chosen)
    return pd.DataFrame(rows)


def add_case_panel(fig: plt.Figure, spec, row: pd.Series, label: str) -> None:
    nested = spec.subgridspec(5, 1, height_ratios=[.13, .08, .43, .17, .19], hspace=.01)
    header = fig.add_subplot(nested[0])
    header.axis("off")
    header.text(0, .65, label, transform=header.transAxes, ha="left", va="center",
                fontsize=9.6, fontweight="bold")
    header.text(.07, .65, str(row.dataset).replace("_", "  ·  "), transform=header.transAxes,
                ha="left", va="center",
                fontsize=8.1, fontweight="bold")
    gain = float(row.absolute_error_reduction)
    header.text(1, .65, f"error reduction {gain:+.2f}", transform=header.transAxes,
                ha="right", va="center", fontsize=7.3,
                color=COLORS["blue"] if gain > 0 else COLORS["red"])
    header.plot([0, 1], [.05, .05], transform=header.transAxes, color="#777777", lw=.65)

    subhead = fig.add_subplot(nested[1]); subhead.axis("off")
    subhead.text(.02, .45, f"Cliff partner ({row.partner_split})", transform=subhead.transAxes,
                 ha="left", va="center", fontsize=7.2, color=COLORS["muted"])
    subhead.text(.52, .45, "Query (test)", transform=subhead.transAxes,
                 ha="left", va="center", fontsize=7.2, color=COLORS["muted"])

    molax = fig.add_subplot(nested[2])
    image = render_aligned_pair_grid(row.partner_smiles, row.smiles, panel_size=(620, 355))
    molax.imshow(image)
    molax.axis("off")
    molax.text(.25, .04, f"{row.partner_target:.2f}", transform=molax.transAxes,
               ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    molax.text(.75, .04, f"{row.target:.2f}", transform=molax.transAxes,
               ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    molax.text(.50, .52, "→", transform=molax.transAxes, ha="center", va="center",
               fontsize=10, color=COLORS["ink"])

    meta = fig.add_subplot(nested[3]); meta.axis("off")
    meta.text(.5, .78, str(row.transformation), transform=meta.transAxes,
              ha="center", va="center", fontsize=7.4, color=COLORS["orange"],
              fontweight="bold")
    meta.text(.5, .18,
              f"Δactivity {row.pair_delta_activity:.2f}    ·    SMILES {row.pair_smiles_similarity:.3f} (trigger)"
              f"    ·    responses {int(row.observed_response_features)}",
              transform=meta.transAxes, ha="center", va="center", fontsize=7.2,
              color=COLORS["muted"])

    err = fig.add_subplot(nested[4])
    baseline_error = float(row.prediction_svm-row.target)
    ours_error = float(row.prediction_ours-row.target)
    err.axvline(0, color="#555555", lw=.7, linestyle=(0, (3, 2)))
    err.hlines([1, 0], -4, 4, color="#E0E0E0", lw=.55)
    err.scatter(baseline_error, 1, s=22, facecolor="white", edgecolor="#777777",
                linewidth=.9, zorder=3)
    err.scatter(ours_error, 0, s=24, facecolor=COLORS["blue"], edgecolor="white",
                linewidth=.4, zorder=4)
    err.set_xlim(-4, 4); err.set_ylim(-.65, 1.55)
    err.set_yticks([1, 0], ["SVM", "Ours"])
    err.set_xticks([-4, -2, 0, 2, 4])
    err.set_xlabel("signed error  prediction − observation", labelpad=1)
    clean_axis(err)
    err.spines[["left", "top", "right"]].set_visible(False)
    err.tick_params(axis="y", length=0, pad=2)


def add_context_panel(fig: plt.Figure, spec, merged: pd.DataFrame,
                      selected: pd.DataFrame, label: str) -> None:
    ax = fig.add_subplot(spec)
    cliff = merged[merged.cliff_mol.astype(bool)].copy()
    groups = [
        ("with response", cliff[cliff.observed_response_features.gt(0)], COLORS["blue"], "-"),
        ("no response", cliff[cliff.observed_response_features.eq(0)], "#777777", (0, (3, 2))),
    ]
    for name, part, color, linestyle in groups:
        values = np.sort(part.absolute_error_reduction.to_numpy(float))
        y = np.arange(1, len(values)+1) / len(values)
        ax.step(values, y, where="post", color=color, lw=1.15, linestyle=linestyle,
                label=f"{name} ($n$={len(values):,})")
    ax.axvline(0, color="#555555", lw=.7, linestyle=(0, (3, 2)))
    for letter, (_, row) in zip("abc", selected.iterrows()):
        part = cliff[cliff.observed_response_features.gt(0)] if row.observed_response_features > 0 \
            else cliff[cliff.observed_response_features.eq(0)]
        values = np.sort(part.absolute_error_reduction.to_numpy(float))
        yy = np.searchsorted(values, row.absolute_error_reduction, side="right") / len(values)
        color = COLORS["blue"] if row.absolute_error_reduction > 0 else COLORS["red"]
        ax.scatter(row.absolute_error_reduction, yy, marker="v", s=27, color=color,
                   edgecolor="white", linewidth=.35, zorder=4)
        ax.text(row.absolute_error_reduction, yy+.055, letter, ha="center", va="bottom",
                fontsize=7.3, fontweight="bold", color=color)
    ax.set_xlim(-1.45, 2.15); ax.set_ylim(0, 1.02)
    ax.set_xlabel("error reduction  |e(SVM)| − |e(ours)|")
    ax.set_ylabel("Fraction of cliff queries")
    ax.legend(loc="upper left", frameon=False, fontsize=7.2, handlelength=2.0)
    panel_label(ax, label, x=-.13, y=1.02)
    clean_axis(ax)
    ax.text(.98, .05, "positive values favour ResponseKernel", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.2, color=COLORS["muted"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/results/case_studies"))
    parser.add_argument("--figure-dir", type=Path, default=Path("paper/figures"))
    parser.add_argument("--cases", type=int, default=3)
    parser.add_argument("--reuse-selection", action="store_true",
                        help="Redraw from selected_main_cases.csv without repeating pair mining")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    set_publication_style()

    ours = pd.concat([pd.read_csv(path) for path in sorted(args.ours.glob("*.csv"))], ignore_index=True)
    base = pd.concat([pd.read_csv(path) for path in sorted(
        args.baselines.glob("*.svm_official.predictions.csv"))], ignore_index=True)
    keys = ["dataset", "smiles", "target", "cliff_mol", "split"]
    merged = ours.merge(base[keys + ["prediction"]], on=keys, suffixes=("_ours", "_svm"))
    merged["error_ours"] = (merged.target-merged.prediction_ours).abs()
    merged["error_svm"] = (merged.target-merged.prediction_svm).abs()
    merged["absolute_error_reduction"] = merged.error_svm-merged.error_ours
    merged.to_csv(args.output_dir / "all_test_predictions.csv", index=False)

    selection_path = args.output_dir / "selected_main_cases.csv"
    if args.reuse_selection and selection_path.exists():
        selected = pd.read_csv(selection_path)
    else:
        selected = select_main_cases(merged)
        selected.to_csv(selection_path, index=False)
    selected[selected.observed_response_features.gt(0)].to_csv(
        args.output_dir / "selected_cases.csv", index=False)
    selected[selected.observed_response_features.eq(0)].to_csv(
        args.output_dir / "selected_boundary_cases.csv", index=False)

    fig = plt.figure(figsize=(7.05, 4.25), facecolor="white")
    outer = fig.add_gridspec(2, 2, left=.055, right=.985, top=.985, bottom=.075,
                             wspace=.18, hspace=.25)
    for spec, label, (_, row) in zip(list(outer)[:3], "abc", selected.iterrows()):
        add_case_panel(fig, spec, row, label)
    add_context_panel(fig, list(outer)[3], merged, selected, "d")
    save_figure(fig, args.figure_dir / "fig_case_studies.pdf", dpi=360)

    merged["coverage_group"] = pd.cut(
        merged.observed_response_features, bins=[-1, 0, 2, 8, np.inf],
        labels=["none", "low", "medium", "high"]
    )
    rows = []
    for subset_name, subset in (("All molecules", merged),
                                ("Cliff molecules", merged[merged.cliff_mol.astype(bool)])):
        for coverage, group in subset.groupby("coverage_group", observed=True):
            rows.append({"subset": subset_name, "coverage": str(coverage), "n": len(group),
                         "svm_rmse": float(np.sqrt(np.mean((group.target-group.prediction_svm)**2))),
                         "ours_rmse": float(np.sqrt(np.mean((group.target-group.prediction_ours)**2)))})
    pd.DataFrame(rows).to_csv(args.output_dir / "coverage_cohorts.csv", index=False)

    payload = {
        "matched_test_rows": int(len(merged)),
        "cliff_rows": int(merged.cliff_mol.astype(bool).sum()),
        "selected_datasets": selected.dataset.tolist(),
        "selected_roles": selected.case_role.tolist(),
        "median_absolute_error_reduction": float(merged.absolute_error_reduction.median()),
        "cliff_median_absolute_error_reduction": float(
            merged.loc[merged.cliff_mol.astype(bool), "absolute_error_reduction"].median()),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
