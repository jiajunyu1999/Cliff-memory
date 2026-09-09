from __future__ import annotations

"""Explicit retrieval controls for target-excluded response memory.

These models intentionally use no molecular graph or fingerprint in their
profile-only arms.  All response coordinates belonging to the prediction
target, or a biologically equivalent target, are removed before any feature is
constructed.  The target-specific coordinate relevance is estimated from the
fit partition only.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR

from molcliff.data import MOLECULEACE_DATASETS, load_moleculeace, split_official_train
from molcliff.metrics import regression_metrics
from validate_moleculeace_offtarget_kernel import molecule_mapping
from validate_moleculeace_operator_action_kernel import exact_kernel
from validate_moleculeace_profile_alignment_kernel import (
    load_profiles, load_target_metadata, profile_design, remove_equivalent_targets,
)


def profile_knn(train_kernel: np.ndarray, query_kernel: np.ndarray, y: np.ndarray, k: int = 5) -> np.ndarray:
    """Fixed-k response-profile retrieval with a training-mean fallback."""
    pred = np.full(len(query_kernel), float(np.mean(y)), dtype=float)
    for row, similarities in enumerate(query_kernel):
        candidates = np.flatnonzero(similarities > 0)
        if not len(candidates):
            continue
        order = candidates[np.argsort(similarities[candidates])[::-1][:k]]
        weights = similarities[order].astype(float)
        pred[row] = float(np.average(y[order], weights=weights))
    return pred


def run_one(dataset: str, root: Path, metadata: dict[str, dict]) -> list[dict]:
    fit, valid = split_official_train(load_moleculeace(dataset))
    y = fit.target.to_numpy(float)
    mapping = molecule_mapping(dataset)
    fit_ids = [mapping[smiles] for smiles in fit.smiles]
    valid_ids = [mapping[smiles] for smiles in valid.smiles]
    profile, excluded = remove_equivalent_targets(load_profiles(dataset, root), dataset, metadata)
    profile_fit_design, profile_valid_design, _, audit = profile_design(fit_ids, valid_ids, profile, y)
    profile_fit = (profile_fit_design @ profile_fit_design.T).toarray().astype(np.float64)
    profile_valid = (profile_valid_design @ profile_fit_design.T).toarray().astype(np.float64)
    chemistry_fit, chemistry_valid = exact_kernel(fit, valid)

    ecfp = SVR(C=10.0, epsilon=.1, kernel="precomputed").fit(chemistry_fit, y)
    # Precomputed kernels have no fitted intercept in sklearn's KRR. Center
    # the fit labels explicitly so a query without any profile receives the
    # training endpoint mean, rather than an artificial potency of zero.
    y_mean = float(y.mean())
    profile_ridge = KernelRidge(alpha=.05, kernel="precomputed").fit(profile_fit, y - y_mean)
    pred_ecfp = ecfp.predict(chemistry_valid)
    pred_knn = profile_knn(profile_fit, profile_valid, y, k=5)
    pred_krr = y_mean + profile_ridge.predict(profile_valid)
    # The 1:1 late fusion coefficient is fixed before evaluation; this arm is
    # diagnostic rather than a response-kernel replacement.
    pred_late = .5 * pred_ecfp + .5 * pred_krr
    coverage = np.asarray(profile_valid.sum(axis=1)).reshape(-1) > 0
    predictions = [
        ("ECFP-SVM", pred_ecfp),
        ("Profile-kNN (k=5)", pred_knn),
        ("Profile-KRR", pred_krr),
        ("ECFP + Profile-KRR late fusion", pred_late),
    ]
    rows = []
    for name, pred in predictions:
        rows.append({"dataset": dataset, "model": name, "profile_dimensions": audit["profile_targets"],
                     "positive_dimensions": audit["positive_weight_targets"],
                     "biologically_equivalent_targets_excluded": len(excluded),
                     "test_profile_coverage": float(coverage.mean()),
                     **regression_metrics(valid.target.to_numpy(float), pred, valid.cliff_mol.to_numpy(bool))})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(MOLECULEACE_DATASETS))
    parser.add_argument("--profile-root", type=Path, default=Path("data/chembl_offtarget_profiles"))
    parser.add_argument("--metadata", type=Path, default=Path("data/chembl_target_metadata.jsonl.gz"))
    parser.add_argument("--output", type=Path, default=Path("reports/results/extended_evidence/profile_retrieval_controls.csv"))
    args = parser.parse_args()
    metadata = load_target_metadata(args.metadata)
    rows = []
    for dataset in args.datasets:
        print(f"[profile-control] {dataset}", flush=True)
        rows.extend(run_one(dataset, args.profile_root, metadata))
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    macro = result.groupby("model")[["rmse", "cliff_rmse", "noncliff_rmse", "mae"]].mean()
    payload = {"datasets": int(result.dataset.nunique()), "models": {
        name: {key: float(value) for key, value in row.items()} for name, row in macro.iterrows()
    }, "protocol": "Official MoleculeACE split; target and biological equivalents excluded; profile kNN uses k=5 response cosine retrieval; Profile-KRR alpha=0.05; late fusion is a fixed equal average."}
    args.output.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
