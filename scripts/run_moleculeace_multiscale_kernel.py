from __future__ import annotations

"""Run the fixed, equal-weight multiscale Tanimoto potential baseline."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.svm import SVR

from molcliff.data import Fingerprints, MOLECULEACE_DATASETS, load_moleculeace, tanimoto_matrix
from molcliff.metrics import regression_metrics


RADII = (1, 2, 3)


def run_one(dataset: str, output_dir: Path) -> dict:
    frame = load_moleculeace(dataset)
    train = frame[frame.split.eq("train")].reset_index(drop=True)
    test = frame[frame.split.eq("test")].reset_index(drop=True)
    train_kernel: np.ndarray | float = 0.0
    test_kernel: np.ndarray | float = 0.0
    for radius in RADII:
        fp = Fingerprints(radius=radius, n_bits=1024)
        x_train = fp.bits(train.smiles.to_numpy())
        x_test = fp.bits(test.smiles.to_numpy())
        train_kernel = train_kernel + tanimoto_matrix(x_train, x_train) / len(RADII)
        test_kernel = test_kernel + tanimoto_matrix(x_test, x_train) / len(RADII)
    # These are the uniform MoleculeACE SVM constants, not selected per target.
    model = SVR(C=10.0, epsilon=0.1, kernel="precomputed")
    model.fit(train_kernel, train.target.to_numpy(dtype=float))
    prediction = model.predict(test_kernel)
    metric = regression_metrics(
        test.target.to_numpy(dtype=float), prediction, test.cliff_mol.to_numpy(dtype=bool)
    )
    metric.update({"dataset": dataset, "model": "multiscale_tanimoto", "seed": 42})
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = test[["dataset", "smiles", "target", "cliff_mol", "split"]].copy()
    predictions["prediction"] = prediction
    predictions.to_csv(output_dir / f"{dataset}.multiscale_tanimoto.predictions.csv", index=False)
    return metric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/moleculeace_multiscale_kernel_v1"))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    rows = Parallel(n_jobs=args.jobs)(
        delayed(run_one)(dataset, args.output_dir) for dataset in MOLECULEACE_DATASETS
    )
    summary = pd.DataFrame(rows).sort_values("dataset")
    summary.to_csv(args.output_dir / "summary.multiscale_tanimoto.csv", index=False)
    aggregate = {
        "model": "multiscale_tanimoto",
        "datasets": len(summary),
        "radii": list(RADII),
        "kernel_weights": "equal",
        "macro_rmse": float(summary.rmse.mean()),
        "macro_cliff_rmse": float(summary.cliff_rmse.mean()),
        "macro_noncliff_rmse": float(summary.noncliff_rmse.mean()),
        "ensemble": False,
        "selection": "none",
    }
    (args.output_dir / "aggregate.multiscale_tanimoto.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(aggregate, sort_keys=True))


if __name__ == "__main__":
    main()
