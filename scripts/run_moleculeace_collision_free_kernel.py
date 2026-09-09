from __future__ import annotations

"""Official evaluation of the validation-frozen collision-free local kernel."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.svm import SVR

from molcliff.data import MOLECULEACE_DATASETS, load_moleculeace
from molcliff.metrics import regression_metrics
from run_moleculeace_local_chemistry_kernel import _generators
from validate_moleculeace_collision_free_kernel import sparse_counts, sparse_tanimoto


def run_one(dataset: str, output_dir: Path) -> dict:
    frame = load_moleculeace(dataset)
    train = frame[frame.split.eq("train")].reset_index(drop=True)
    test = frame[frame.split.eq("test")].reset_index(drop=True)
    generators = _generators()
    train_kernel = np.zeros((len(train), len(train)), dtype=np.float32)
    test_kernel = np.zeros((len(test), len(train)), dtype=np.float32)
    dimensions = []
    for _name, generator in generators:
        train_count, vocabulary = sparse_counts(train.smiles.to_numpy(), generator)
        test_count, _ = sparse_counts(test.smiles.to_numpy(), generator, vocabulary)
        train_binary, test_binary = train_count.copy(), test_count.copy()
        train_binary.data[:] = 1.0; test_binary.data[:] = 1.0
        train_kernel += (sparse_tanimoto(train_binary, train_binary)
                         + sparse_tanimoto(train_count, train_count)) / (2 * len(generators))
        test_kernel += (sparse_tanimoto(test_binary, train_binary)
                        + sparse_tanimoto(test_count, train_count)) / (2 * len(generators))
        dimensions.append(len(vocabulary))
    model = SVR(C=10.0, epsilon=0.1, kernel="precomputed")
    model.fit(train_kernel, train.target.to_numpy(dtype=float))
    prediction = model.predict(test_kernel)
    metric = regression_metrics(test.target.to_numpy(dtype=float), prediction,
                                test.cliff_mol.to_numpy(dtype=bool))
    metric.update({"dataset": dataset, "model": "collision_free_binary_count_kernel",
                   "seed": 42, "dimensions": json.dumps(dimensions)})
    output_dir.mkdir(parents=True, exist_ok=True)
    out = test[["dataset", "smiles", "target", "cliff_mol", "split"]].copy()
    out["prediction"] = prediction
    out.to_csv(output_dir / f"{dataset}.collision_free_kernel.predictions.csv", index=False)
    return metric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/moleculeace_collision_free_kernel_v1"))
    args = parser.parse_args()
    rows = Parallel(n_jobs=args.jobs)(
        delayed(run_one)(dataset, args.output_dir) for dataset in MOLECULEACE_DATASETS)
    summary = pd.DataFrame(rows).sort_values("dataset")
    summary.to_csv(args.output_dir / "summary.collision_free_kernel.csv", index=False)
    aggregate = {
        "model": "collision_free_binary_count_kernel", "datasets": len(summary),
        "macro_rmse": float(summary.rmse.mean()),
        "macro_cliff_rmse": float(summary.cliff_rmse.mean()),
        "macro_noncliff_rmse": float(summary.noncliff_rmse.mean()),
        "views": {"binary": [name for name, _ in _generators()],
                  "count": [name for name, _ in _generators()]},
        "feature_space": "exact sparse train vocabulary", "kernel_weights": "equal",
        "seed": 42, "selection": "fixed_internal_validation_then_one_test_evaluation",
        "ensemble": False, "uses_test_labels_as_input": False,
    }
    (args.output_dir / "aggregate.collision_free_kernel.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(aggregate, sort_keys=True))


if __name__ == "__main__":
    main()
