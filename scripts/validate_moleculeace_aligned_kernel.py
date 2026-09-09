from __future__ import annotations

"""Train-only closed-form kernel alignment for local chemistry views."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.svm import SVR

from molcliff.data import MOLECULEACE_DATASETS, load_moleculeace, split_official_train
from molcliff.metrics import regression_metrics
from run_moleculeace_local_chemistry_kernel import _generators
from validate_moleculeace_collision_free_kernel import sparse_counts, sparse_tanimoto


def _alignment(kernel: np.ndarray, y: np.ndarray) -> float:
    row_mean = kernel.mean(axis=1, keepdims=True)
    centered = kernel - row_mean - row_mean.T + kernel.mean()
    yc = y - y.mean()
    numerator = float(yc @ centered @ yc)
    denominator = float(np.linalg.norm(centered) * np.dot(yc, yc))
    return max(0.0, numerator / max(denominator, 1e-12))


def _action_alignment_weights(kernels: list[np.ndarray], y: np.ndarray) -> np.ndarray:
    reference = sum(kernels) / len(kernels)
    left, right = [], []
    neighbor_budget = max(4, min(24, int(np.sqrt(len(y)))))
    for i in range(len(y)):
        order = np.argsort(reference[i])[::-1]
        order = order[order != i][:neighbor_budget]
        order = order[reference[i, order] >= 0.35]
        left.extend([i] * len(order)); right.extend(order.tolist())
    left_array, right_array = np.asarray(left), np.asarray(right)
    target = np.abs(y[left_array] - y[right_array])
    target = target - target.mean()
    scores = []
    for kernel in kernels:
        distance = 1.0 - kernel[left_array, right_array]
        distance = distance - distance.mean()
        score = float(np.dot(distance, target) /
                      max(np.linalg.norm(distance) * np.linalg.norm(target), 1e-12))
        scores.append(max(0.0, score))
    weights = np.asarray(scores)
    if weights.sum() <= 0:
        weights[:] = 1.0
    return weights / weights.sum()


def run_one(dataset: str) -> list[dict]:
    fit, valid = split_official_train(load_moleculeace(dataset))
    train_views, valid_views, names = [], [], []
    for generator_name, generator in _generators():
        fit_count, vocabulary = sparse_counts(fit.smiles.to_numpy(), generator)
        valid_count, _ = sparse_counts(valid.smiles.to_numpy(), generator, vocabulary)
        fit_binary, valid_binary = fit_count.copy(), valid_count.copy()
        fit_binary.data[:] = 1.0; valid_binary.data[:] = 1.0
        for kind, left, right in (("binary", fit_binary, valid_binary),
                                  ("count", fit_count, valid_count)):
            train_views.append(sparse_tanimoto(left, left))
            valid_views.append(sparse_tanimoto(right, left))
            names.append(f"{kind}:{generator_name}")
    y = fit.target.to_numpy(dtype=float)
    raw_weight = np.asarray([_alignment(kernel, y) for kernel in train_views])
    if raw_weight.sum() <= 0:
        raw_weight[:] = 1.0
    weight = raw_weight / raw_weight.sum()
    equal_train = sum(train_views) / len(train_views)
    equal_valid = sum(valid_views) / len(valid_views)
    aligned_train = sum(w * kernel for w, kernel in zip(weight, train_views, strict=True))
    aligned_valid = sum(w * kernel for w, kernel in zip(weight, valid_views, strict=True))
    action_weight = _action_alignment_weights(train_views, y)
    action_train = sum(w * kernel for w, kernel in zip(action_weight, train_views, strict=True))
    action_valid = sum(w * kernel for w, kernel in zip(action_weight, valid_views, strict=True))
    rows = []
    for model_name, train_kernel, valid_kernel in (
        ("collision_free_equal_kernel", equal_train, equal_valid),
        ("collision_free_aligned_kernel", aligned_train, aligned_valid),
        ("collision_free_action_aligned_kernel", action_train, action_valid),
    ):
        model = SVR(C=10.0, epsilon=0.1, kernel="precomputed")
        model.fit(train_kernel, y)
        prediction = model.predict(valid_kernel)
        metric = regression_metrics(valid.target.to_numpy(dtype=float), prediction,
                                    valid.cliff_mol.to_numpy(dtype=bool))
        selected_weight = action_weight if "action_aligned" in model_name else weight
        rows.append({"dataset": dataset, "model": model_name, **metric,
                     "weights": json.dumps(dict(zip(names, selected_weight.tolist(), strict=True)),
                                           sort_keys=True)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/moleculeace_aligned_kernel_validation_v1"))
    args = parser.parse_args()
    nested = Parallel(n_jobs=args.jobs)(delayed(run_one)(name) for name in MOLECULEACE_DATASETS)
    summary = pd.DataFrame([row for rows in nested for row in rows]).sort_values(["model", "dataset"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "summary.validation.csv", index=False)
    macro = summary.groupby("model")[["rmse", "cliff_rmse", "noncliff_rmse"]].mean()
    aggregate = {name: {key: float(value) for key, value in row.items()}
                 for name, row in macro.iterrows()}
    aggregate["protocol"] = {
        "split": "fixed official-train 80/20 seed42 stratified", "test_evaluations": 0,
        "weighting": "closed-form nonnegative centered kernel alignment; no search",
    }
    (args.output_dir / "aggregate.validation.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(aggregate, sort_keys=True))


if __name__ == "__main__":
    main()
