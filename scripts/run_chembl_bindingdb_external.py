from __future__ import annotations

"""Strict ChEMBL-to-BindingDB external transfer for the response-kernel idea.

The affinity model is fitted only on each released MoleculeACE/ChEMBL training
partition.  BindingDB labels are never used for fitting, calibration, feature
selection, or kernel weighting.  Test molecules are BindingDB molecules absent
from that ChEMBL training partition.  Before constructing a response profile,
all BindingDB systems biologically equivalent to the benchmark protein are
removed, including the systems that supply the external labels.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import SVR

from molcliff.data import MOLECULEACE_DATASETS, load_moleculeace
from molcliff.metrics import regression_metrics
from validate_moleculeace_operator_action_kernel import exact_kernel
from validate_moleculeace_profile_alignment_kernel import (
    filter_bindingdb_equivalent_targets,
    load_bindingdb_profiles,
    load_sequences,
    load_target_metadata,
    matrices,
    profile_design,
)


def external_cliffs(frame: pd.DataFrame, kernel: np.ndarray) -> np.ndarray:
    """Define cliffs within the held-out BindingDB set, never against train."""
    y = frame.target.to_numpy(float)
    similar = kernel >= 0.90
    np.fill_diagonal(similar, False)
    return np.any(similar & (np.abs(y[:, None] - y[None, :]) >= 1.0), axis=1)


def external_metrics(y: np.ndarray, pred: np.ndarray, cliff: np.ndarray) -> dict:
    """Retain an external target when its cliff subset is degenerate."""
    if cliff.any() and (~cliff).any():
        return regression_metrics(y, pred, cliff)
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    result = {"n_test": int(len(y)), "n_cliff": int(cliff.sum()), "rmse": rmse,
              "mae": float(np.mean(np.abs(y - pred))), "cliff_rmse": float("nan"),
              "noncliff_rmse": float("nan"), "cliff_mae": float("nan")}
    if cliff.any():
        result["cliff_rmse"] = rmse
        result["cliff_mae"] = result["mae"]
    else:
        result["noncliff_rmse"] = rmse
    return result


def run_one(
    dataset: str, external: pd.DataFrame, binding_profile: pd.DataFrame,
    binding_metadata: dict[str, dict], chembl_metadata: dict[str, dict],
    sequences: dict[str, str],
) -> list[dict]:
    fit = load_moleculeace(dataset).query("split == 'train'").copy().reset_index(drop=True)
    seen = set(fit.smiles.astype(str))
    valid = external[external.dataset.eq(dataset) & ~external.smiles.astype(str).isin(seen)].copy()
    # A molecule can occur under several external experimental constructs;
    # its median is the predetermined assay-level aggregation.
    valid = valid.groupby("smiles", as_index=False).target.median()
    if len(valid) < 20:
        return []
    valid["cliff_mol"] = False
    chemistry_fit, chemistry_valid = exact_kernel(fit, valid)
    # Reuse the exact chemistry kernel on the held-out table only to define
    # its descriptive cliff subset; no training molecule or label enters it.
    query_kernel, _ = exact_kernel(valid, valid)
    valid["cliff_mol"] = external_cliffs(valid, query_kernel)

    profile, excluded = filter_bindingdb_equivalent_targets(
        dataset, binding_profile, binding_metadata, chembl_metadata, sequences
    )
    fit_ids, valid_ids = fit.smiles.astype(str).tolist(), valid.smiles.astype(str).tolist()
    _, _, nearest, audit = profile_design(fit_ids, valid_ids, profile, fit.target.to_numpy(float))
    coverage_fit, coverage_valid, residual_fit, residual_valid, _ = matrices(fit_ids, valid_ids, profile)
    profile_fit = (coverage_fit + residual_fit) / 2
    profile_valid = (coverage_valid + residual_valid) / 2
    # Fixed 10:8 mixing is the frozen response-kernel recipe; it is not
    # optimized using BindingDB labels.
    kernels = [
        ("ECFP-SVM", chemistry_fit, chemistry_valid),
        ("Response memory (frozen 10:8)",
         (10 * chemistry_fit + 8 * profile_fit) / 18,
         (10 * chemistry_valid + 8 * profile_valid) / 18),
    ]
    rows = []
    for model_name, train_kernel, test_kernel in kernels:
        model = SVR(C=10.0, epsilon=0.1, kernel="precomputed")
        model.fit(train_kernel, fit.target.to_numpy(float))
        pred = model.predict(test_kernel)
        rows.append({
            "dataset": dataset, "model": model_name, "n_train": len(fit),
            "n_external": len(valid), "n_cliff": int(valid.cliff_mol.sum()),
            "external_systems_before_pooling": int(external[external.dataset.eq(dataset)].system.nunique()),
            "equivalent_systems_excluded": len(excluded),
            "profile_systems": audit["profile_targets"],
            "profile_positive_systems": audit["positive_weight_targets"],
            "response_coverage": float((coverage_valid.max(axis=1) > 0).mean()),
            **external_metrics(valid.target.to_numpy(float), pred, valid.cliff_mol.to_numpy(bool)),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external", type=Path, default=Path("data/bindingdb_external_exact_protein.csv"))
    parser.add_argument("--profile", type=Path, default=Path("data/bindingdb_external_response_profiles.jsonl.gz"))
    parser.add_argument("--metadata", type=Path, default=Path("data/bindingdb_system_metadata.jsonl.gz"))
    parser.add_argument("--chembl-metadata", type=Path, default=Path("data/chembl_target_metadata.jsonl.gz"))
    parser.add_argument("--sequences", type=Path, default=Path("data/uniprot_target_sequences.json.gz"))
    parser.add_argument("--output", type=Path, default=Path("reports/results/extended_evidence/chembl_bindingdb_external.csv"))
    args = parser.parse_args()
    external = pd.read_csv(args.external)
    profile, binding_metadata = load_bindingdb_profiles(args.profile, args.metadata)
    chembl_metadata = load_target_metadata(args.chembl_metadata)
    sequences = load_sequences(args.sequences)
    rows = []
    for dataset in MOLECULEACE_DATASETS:
        if dataset not in set(external.dataset):
            continue
        print(f"[external] {dataset}", flush=True)
        rows.extend(run_one(dataset, external, profile, binding_metadata, chembl_metadata, sequences))
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    macro = result.groupby("model")[["rmse", "cliff_rmse", "noncliff_rmse", "mae", "response_coverage"]].mean()
    payload = {"datasets": int(result.dataset.nunique()), "models": {
        name: {key: float(value) for key, value in row.items()} for name, row in macro.iterrows()
    }, "protocol": "ChEMBL training labels only; unseen BindingDB molecules with exact-protein matching; target-equivalent BindingDB systems excluded before response-memory construction; frozen 10:8 chemistry/profile kernel mixture."}
    args.output.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
