from __future__ import annotations

"""Train-only cross-fitted affine calibration of the exact-kernel SVR."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from sklearn.svm import SVR

from molcliff.data import MOLECULEACE_DATASETS, load_moleculeace, split_official_train
from molcliff.metrics import regression_metrics
from validate_moleculeace_operator_action_kernel import exact_kernel


def run_one(dataset: str) -> list[dict]:
    fit, valid = split_official_train(load_moleculeace(dataset))
    kernel, cross = exact_kernel(fit, valid)
    y = fit.target.to_numpy(dtype=float)
    oof = np.empty(len(y), dtype=float)
    # The folds estimate calibration only. They are never averaged at inference.
    for train_idx, held_idx in KFold(n_splits=5, shuffle=True, random_state=42).split(y):
        model = SVR(C=10.0, epsilon=0.1, kernel="precomputed")
        model.fit(kernel[np.ix_(train_idx, train_idx)], y[train_idx])
        oof[held_idx] = model.predict(kernel[np.ix_(held_idx, train_idx)])
    design = np.column_stack((np.ones(len(oof)), oof))
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]

    final_model = SVR(C=10.0, epsilon=0.1, kernel="precomputed")
    final_model.fit(kernel, y)
    raw = final_model.predict(cross)
    residual = np.abs(y - oof)
    # Continuous hard-example curriculum with exactly unit mean.  Ranks remove
    # target-scale dependence and avoid a threshold or temperature parameter.
    ranks = pd.Series(residual).rank(method="average").to_numpy(dtype=float)
    hard_weight = 2.0 * ranks / (len(ranks) + 1.0)
    hard_model = SVR(C=10.0, epsilon=0.1, kernel="precomputed")
    hard_model.fit(kernel, y, sample_weight=hard_weight)
    hard_prediction = hard_model.predict(cross)
    candidates = (
        ("collision_free_binary_count_kernel", raw),
        ("crossfit_affine_collision_free_kernel", intercept + slope * raw),
        ("crossfit_residual_rank_weighted_kernel", hard_prediction),
    )
    return [
        {"dataset": dataset, "model": name, "calibration_intercept": intercept,
         "calibration_slope": slope,
         **regression_metrics(valid.target.to_numpy(dtype=float), pred,
                              valid.cliff_mol.to_numpy(dtype=bool))}
        for name, pred in candidates
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(MOLECULEACE_DATASETS))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/moleculeace_crossfit_calibration_validation_v1"))
    args = parser.parse_args()
    nested = Parallel(n_jobs=args.jobs)(delayed(run_one)(d) for d in args.datasets)
    summary = pd.DataFrame([row for group in nested for row in group]).sort_values(["model", "dataset"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "summary.validation.csv", index=False)
    macro = summary.groupby("model")[["rmse", "cliff_rmse", "noncliff_rmse"]].mean()
    result = {name: {k: float(v) for k, v in row.items()} for name, row in macro.iterrows()}
    result["protocol"] = "fixed 5-fold seed42 OOF analytic affine calibration; one final SVR; zero test evaluations"
    (args.output_dir / "aggregate.validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
