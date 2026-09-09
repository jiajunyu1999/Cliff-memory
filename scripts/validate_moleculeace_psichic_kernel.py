from __future__ import annotations

"""Internal-only validation of a frozen PSICHIC interaction kernel."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import SVR

from molcliff.data import load_moleculeace, split_official_train
from molcliff.metrics import regression_metrics
from run_moleculeace_local_chemistry_kernel import _generators
from validate_moleculeace_collision_free_kernel import sparse_counts, sparse_tanimoto


DEFAULT_DATASETS = ("CHEMBL4203_Ki", "CHEMBL2971_Ki", "CHEMBL237_EC50", "CHEMBL228_Ki")


def rbf_kernel(fit: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    fit = fit.astype(np.float64)
    query = query.astype(np.float64)
    fit_sq = np.square(fit).sum(1)
    distances = np.maximum(fit_sq[:, None] + fit_sq[None, :] - 2 * fit @ fit.T, 0.0)
    scale = float(np.median(distances[np.triu_indices(len(fit), 1)]))
    scale = max(scale, 1e-12)
    query_sq = np.square(query).sum(1)
    cross = np.maximum(query_sq[:, None] + fit_sq[None, :] - 2 * query @ fit.T, 0.0)
    return np.exp(-distances / scale), np.exp(-cross / scale), scale


def evaluate(dataset: str, feature_root: Path) -> list[dict]:
    full_train = load_moleculeace(dataset).query("split == 'train'").reset_index(drop=True)
    fit, valid = split_official_train(load_moleculeace(dataset))
    saved = np.load(feature_root / f"{dataset}.train.npz")
    if not np.array_equal(saved["smiles"], full_train.smiles.to_numpy(dtype=str)):
        raise ValueError(f"Feature order mismatch for {dataset}")
    lookup = {s: i for i, s in enumerate(full_train.smiles)}
    fit_idx = np.array([lookup[s] for s in fit.smiles])
    valid_idx = np.array([lookup[s] for s in valid.smiles])

    chemistry_fit = np.zeros((len(fit), len(fit)), dtype=np.float32)
    chemistry_valid = np.zeros((len(valid), len(fit)), dtype=np.float32)
    generators = _generators()
    for _, generator in generators:
        all_count, vocabulary = sparse_counts(full_train.smiles.to_numpy(), generator)
        fit_count = all_count[fit_idx]
        valid_count = all_count[valid_idx]
        fit_binary, valid_binary = fit_count.copy(), valid_count.copy()
        fit_binary.data[:] = 1.0
        valid_binary.data[:] = 1.0
        chemistry_fit += (sparse_tanimoto(fit_binary, fit_binary)
                          + sparse_tanimoto(fit_count, fit_count)) / (2 * len(generators))
        chemistry_valid += (sparse_tanimoto(valid_binary, fit_binary)
                            + sparse_tanimoto(valid_count, fit_count)) / (2 * len(generators))

    interaction_fit, interaction_valid, scale = rbf_kernel(
        saved["feature"][fit_idx], saved["feature"][valid_idx])
    rows = []
    candidates = (
        ("collision_free_binary_count_kernel", chemistry_fit, chemistry_valid),
        ("psichic_interaction_kernel", interaction_fit, interaction_valid),
        ("collision_free_plus_psichic_10to1",
         (10 * chemistry_fit + interaction_fit) / 11,
         (10 * chemistry_valid + interaction_valid) / 11),
    )
    for name, train_kernel, valid_kernel in candidates:
        model = SVR(C=10.0, epsilon=0.1, kernel="precomputed")
        model.fit(train_kernel, fit.target.to_numpy(dtype=float))
        prediction = model.predict(valid_kernel)
        rows.append({"dataset": dataset, "model": name, "rbf_median_sqdist": scale,
                     **regression_metrics(valid.target.to_numpy(dtype=float), prediction,
                                          valid.cliff_mol.to_numpy(dtype=bool))})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--feature-root", type=Path,
                        default=Path("data/moleculeace_psichic_pdb2020"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/moleculeace_psichic_validation_v1"))
    args = parser.parse_args()
    rows = [row for dataset in args.datasets for row in evaluate(dataset, args.feature_root)]
    summary = pd.DataFrame(rows).sort_values(["model", "dataset"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "summary.validation.csv", index=False)
    macro = summary.groupby("model")[["rmse", "cliff_rmse", "noncliff_rmse"]].mean()
    result = {name: dict(map(lambda item: (item[0], float(item[1])), row.items()))
              for name, row in macro.iterrows()}
    result["protocol"] = "four preregistered hard targets; official-train split only; zero test evaluations"
    (args.output_dir / "aggregate.validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
