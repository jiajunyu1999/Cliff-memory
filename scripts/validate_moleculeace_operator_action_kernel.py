from __future__ import annotations

"""Validate a single RKHS potential from value and local action observations.

For every train-only close analogue edge, the same potential is observed through
the linear functional f(x_j)-f(x_i).  The operator kernel A K A^T is PSD by
construction.  This is one SVR, not a prediction ensemble.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.svm import SVR

from molcliff.data import MOLECULEACE_DATASETS, load_moleculeace, split_official_train
from molcliff.metrics import regression_metrics
from run_moleculeace_local_chemistry_kernel import _generators
from validate_moleculeace_collision_free_kernel import sparse_counts, sparse_tanimoto


def exact_kernel(fit: pd.DataFrame, valid: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    train_kernel = np.zeros((len(fit), len(fit)), dtype=np.float32)
    valid_kernel = np.zeros((len(valid), len(fit)), dtype=np.float32)
    generators = _generators()
    for _, generator in generators:
        fit_count, vocab = sparse_counts(fit.smiles.to_numpy(), generator)
        valid_count, _ = sparse_counts(valid.smiles.to_numpy(), generator, vocab)
        fit_binary, valid_binary = fit_count.copy(), valid_count.copy()
        fit_binary.data[:] = 1.0
        valid_binary.data[:] = 1.0
        train_kernel += (sparse_tanimoto(fit_binary, fit_binary)
                         + sparse_tanimoto(fit_count, fit_count)) / (2 * len(generators))
        valid_kernel += (sparse_tanimoto(valid_binary, fit_binary)
                         + sparse_tanimoto(valid_count, fit_count)) / (2 * len(generators))
    return train_kernel, valid_kernel


def run_one(dataset: str) -> list[dict]:
    fit, valid = split_official_train(load_moleculeace(dataset))
    kernel, cross = exact_kernel(fit, valid)
    y = fit.target.to_numpy(dtype=float)
    upper_i, upper_j = np.where(np.triu(kernel, 1) >= 0.90)
    n, m = len(fit), len(upper_i)
    # A contains point evaluations followed by directed finite differences j-i.
    if m:
        rows = np.concatenate((np.arange(n), n + np.arange(m), n + np.arange(m)))
        cols = np.concatenate((np.arange(n), upper_i, upper_j))
        values = np.concatenate((np.ones(n), -np.ones(m), np.ones(m)))
        operator = sparse.csr_matrix((values, (rows, cols)), shape=(n + m, n))
        obs_kernel = np.asarray(operator @ kernel @ operator.T)
        query_kernel = np.asarray(cross @ operator.T)
        obs_y = np.asarray(operator @ y)
        edge_similarity = kernel[upper_i, upper_j].astype(float)
        edge_weight = edge_similarity * n / edge_similarity.sum()
        sample_weight = np.concatenate((np.ones(n), edge_weight))
    else:
        obs_kernel, query_kernel, obs_y = kernel, cross, y
        sample_weight = np.ones(n)

    rows_out = []
    for name, train_k, valid_k, target, weight in (
        ("collision_free_binary_count_kernel", kernel, cross, y, None),
        ("operator_action_kernel", obs_kernel, query_kernel, obs_y, sample_weight),
    ):
        model = SVR(C=10.0, epsilon=0.1, kernel="precomputed")
        model.fit(train_k, target, sample_weight=weight)
        pred = model.predict(valid_k)
        rows_out.append({"dataset": dataset, "model": name, "action_edges": m,
                         **regression_metrics(valid.target.to_numpy(dtype=float), pred,
                                              valid.cliff_mol.to_numpy(dtype=bool))})
    return rows_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--datasets", nargs="+", default=list(MOLECULEACE_DATASETS))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/moleculeace_operator_action_validation_v1"))
    args = parser.parse_args()
    nested = Parallel(n_jobs=args.jobs)(delayed(run_one)(name) for name in args.datasets)
    summary = pd.DataFrame([row for group in nested for row in group]).sort_values(["model", "dataset"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "summary.validation.csv", index=False)
    macro = summary.groupby("model")[["rmse", "cliff_rmse", "noncliff_rmse"]].mean()
    result = {name: {key: float(value) for key, value in row.items()}
              for name, row in macro.iterrows()}
    result["protocol"] = {
        "split": "fixed official-train 80/20 seed42 stratified; zero test evaluations",
        "edge_rule": "exact-kernel similarity >=0.90; edge mass equals point mass",
        "estimator": "one operator-kernel SVR C=10 epsilon=0.1",
    }
    (args.output_dir / "aggregate.validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
