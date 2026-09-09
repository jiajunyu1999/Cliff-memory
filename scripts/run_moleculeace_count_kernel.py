from __future__ import annotations

"""Official evaluation of the validation-frozen binary+count chemistry kernel."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.svm import SVR

from molcliff.data import MOLECULEACE_DATASETS, load_moleculeace, tanimoto_matrix
from molcliff.metrics import regression_metrics
from run_moleculeace_local_chemistry_kernel import _generators
from validate_moleculeace_count_kernel import _generalized_tanimoto, _representations


def run_one(dataset: str, output_dir: Path) -> dict:
    frame = load_moleculeace(dataset)
    train = frame[frame.split.eq("train")].reset_index(drop=True)
    test = frame[frame.split.eq("test")].reset_index(drop=True)
    generators = _generators()
    train_kernel = np.zeros((len(train), len(train)), dtype=np.float32)
    test_kernel = np.zeros((len(test), len(train)), dtype=np.float32)
    for _name, generator in generators:
        train_bits, train_counts = _representations(train.smiles.to_numpy(), generator)
        test_bits, test_counts = _representations(test.smiles.to_numpy(), generator)
        train_kernel += (tanimoto_matrix(train_bits, train_bits)
                         + _generalized_tanimoto(train_counts, train_counts)) / (2 * len(generators))
        test_kernel += (tanimoto_matrix(test_bits, train_bits)
                        + _generalized_tanimoto(test_counts, train_counts)) / (2 * len(generators))
    model = SVR(C=10.0, epsilon=0.1, kernel="precomputed")
    model.fit(train_kernel, train.target.to_numpy(dtype=float))
    prediction = model.predict(test_kernel)
    metric = regression_metrics(test.target.to_numpy(dtype=float), prediction,
                                test.cliff_mol.to_numpy(dtype=bool))
    metric.update({"dataset": dataset, "model": "binary_count_local_kernel", "seed": 42})
    output_dir.mkdir(parents=True, exist_ok=True)
    out = test[["dataset", "smiles", "target", "cliff_mol", "split"]].copy()
    out["prediction"] = prediction
    out.to_csv(output_dir / f"{dataset}.binary_count_local_kernel.predictions.csv", index=False)
    return metric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/moleculeace_count_kernel_v1"))
    args = parser.parse_args()
    rows = Parallel(n_jobs=args.jobs)(
        delayed(run_one)(dataset, args.output_dir) for dataset in MOLECULEACE_DATASETS)
    summary = pd.DataFrame(rows).sort_values("dataset")
    summary.to_csv(args.output_dir / "summary.binary_count_local_kernel.csv", index=False)
    aggregate = {
        "model": "binary_count_local_kernel", "datasets": len(summary),
        "views": {"binary": [name for name, _ in _generators()],
                  "count": [name for name, _ in _generators()]},
        "kernel_weights": "equal", "macro_rmse": float(summary.rmse.mean()),
        "macro_cliff_rmse": float(summary.cliff_rmse.mean()),
        "macro_noncliff_rmse": float(summary.noncliff_rmse.mean()),
        "seed": 42, "selection": "fixed_internal_validation_then_one_test_evaluation",
        "ensemble": False, "uses_test_labels_as_input": False,
    }
    (args.output_dir / "aggregate.binary_count_local_kernel.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(aggregate, sort_keys=True))


if __name__ == "__main__":
    main()
