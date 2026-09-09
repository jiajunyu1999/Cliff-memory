from __future__ import annotations

"""Fixed local-chemistry kernel with train-only analytic OOF de-shrinkage."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from sklearn.svm import SVR

from molcliff.data import MOLECULEACE_DATASETS, load_moleculeace, tanimoto_matrix
from molcliff.metrics import regression_metrics
from run_moleculeace_local_chemistry_kernel import _bits, _generators


def _kernels(train_smiles: np.ndarray, test_smiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    generators = _generators()
    train_kernel = np.zeros((len(train_smiles), len(train_smiles)), dtype=np.float32)
    test_kernel = np.zeros((len(test_smiles), len(train_smiles)), dtype=np.float32)
    for _name, generator in generators:
        x_train = _bits(train_smiles, generator)
        x_test = _bits(test_smiles, generator)
        train_kernel += tanimoto_matrix(x_train, x_train) / len(generators)
        test_kernel += tanimoto_matrix(x_test, x_train) / len(generators)
    return train_kernel, test_kernel


def run_one(dataset: str, output_dir: Path) -> dict:
    frame = load_moleculeace(dataset)
    train = frame[frame.split.eq("train")].reset_index(drop=True)
    test = frame[frame.split.eq("test")].reset_index(drop=True)
    y = train.target.to_numpy(dtype=float)
    k_train, k_test = _kernels(train.smiles.to_numpy(), test.smiles.to_numpy())
    oof = np.empty(len(train), dtype=float)
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    for fit_idx, valid_idx in folds.split(y):
        fold_model = SVR(C=10.0, epsilon=0.1, kernel="precomputed")
        fold_model.fit(k_train[np.ix_(fit_idx, fit_idx)], y[fit_idx])
        oof[valid_idx] = fold_model.predict(k_train[np.ix_(valid_idx, fit_idx)])
    design = np.column_stack([np.ones(len(oof)), oof])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    model = SVR(C=10.0, epsilon=0.1, kernel="precomputed")
    model.fit(k_train, y)
    raw_prediction = model.predict(k_test)
    prediction = intercept + slope * raw_prediction
    metric = regression_metrics(
        test.target.to_numpy(dtype=float), prediction, test.cliff_mol.to_numpy(dtype=bool))
    metric.update({"dataset": dataset, "model": "deshrunk_local_chemistry_kernel", "seed": 42,
                   "calibration_intercept": float(intercept), "calibration_slope": float(slope),
                   "oof_rmse": float(np.sqrt(np.mean(np.square(oof - y))))})
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = test[["dataset", "smiles", "target", "cliff_mol", "split"]].copy()
    predictions["raw_prediction"] = raw_prediction
    predictions["prediction"] = prediction
    predictions.to_csv(output_dir / f"{dataset}.deshrunk_kernel.predictions.csv", index=False)
    return metric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/moleculeace_deshrunk_kernel_v1"))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    rows = Parallel(n_jobs=args.jobs)(
        delayed(run_one)(dataset, args.output_dir) for dataset in MOLECULEACE_DATASETS)
    summary = pd.DataFrame(rows).sort_values("dataset")
    summary.to_csv(args.output_dir / "summary.deshrunk_kernel.csv", index=False)
    aggregate = {"model": "deshrunk_local_chemistry_kernel", "datasets": len(summary),
                 "views": [name for name, _ in _generators()], "kernel_weights": "equal",
                 "calibration": "train_only_5fold_oof_ols", "fold_seed": 42,
                 "mean_calibration_slope": float(summary.calibration_slope.mean()),
                 "macro_rmse": float(summary.rmse.mean()),
                 "macro_cliff_rmse": float(summary.cliff_rmse.mean()),
                 "macro_noncliff_rmse": float(summary.noncliff_rmse.mean()),
                 "ensemble": False, "selection": "none"}
    (args.output_dir / "aggregate.deshrunk_kernel.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(aggregate, sort_keys=True))


if __name__ == "__main__":
    main()
