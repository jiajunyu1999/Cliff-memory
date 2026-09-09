from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from sklearn.svm import SVR

from molcliff.baselines import make_regressor
from molcliff.data import (
    Fingerprints,
    MOLECULEACE_DATASETS,
    load_moleculeace,
    split_official_train,
    tanimoto_matrix,
)
from molcliff.metrics import regression_metrics


def validation_tanimoto(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    fit, valid = split_official_train(load_moleculeace(dataset))
    fp = Fingerprints()
    x_fit = fp.bits(fit.smiles.to_numpy())
    x_valid = fp.bits(valid.smiles.to_numpy())
    model = SVR(C=10.0, epsilon=0.1, kernel="precomputed")
    model.fit(tanimoto_matrix(x_fit, x_fit), fit.target.to_numpy(dtype=float))
    return model.predict(tanimoto_matrix(x_valid, x_fit)), valid.target.to_numpy(dtype=float)


def validation_ecfp_svr(dataset: str, model_name: str) -> tuple[np.ndarray, np.ndarray]:
    fit, valid = split_official_train(load_moleculeace(dataset))
    fp = Fingerprints()
    model = make_regressor(model_name, dataset=dataset)
    model.fit(fp.bits(fit.smiles.to_numpy()), fit.target.to_numpy(dtype=float))
    return (
        np.asarray(model.predict(fp.bits(valid.smiles.to_numpy())), dtype=float),
        valid.target.to_numpy(dtype=float),
    )


def read_prediction(directory: Path, dataset: str, model: str) -> pd.DataFrame:
    return pd.read_csv(directory / f"{dataset}.{model}.predictions.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation-dir", type=Path, default=Path("outputs/mmp_memory_diagnostics_all_v1")
    )
    parser.add_argument("--rbf-dir", type=Path, default=Path("outputs/moleculeace"))
    parser.add_argument("--base-model", choices=["svm", "svm_official"], default="svm")
    parser.add_argument("--tanimoto-dir", type=Path, default=Path("outputs/moleculeace_tanimoto_svm_v1"))
    parser.add_argument("--mmp-dir", type=Path, default=Path("outputs/moleculeace_mmp_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/action_ensemble_v1"))
    parser.add_argument("--validation-gain-threshold", type=float, default=0.0)
    parser.add_argument("--shrinkage", type=float, default=1.0)
    args = parser.parse_args()
    if not 0.0 <= args.shrinkage <= 1.0:
        raise ValueError("--shrinkage must be in [0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset in MOLECULEACE_DATASETS:
        validation = pd.read_csv(args.validation_dir / f"{dataset}.csv")
        tan_valid, truth_valid = validation_tanimoto(dataset)
        np.testing.assert_allclose(validation.target.to_numpy(dtype=float), truth_valid)
        if args.base_model == "svm":
            rbf_valid = validation.base.to_numpy(dtype=float)
        else:
            rbf_valid, truth_check = validation_ecfp_svr(dataset, args.base_model)
            np.testing.assert_allclose(truth_valid, truth_check)
        design = np.c_[
            tan_valid - rbf_valid,
            validation.blended.to_numpy(dtype=float) - rbf_valid,
        ]
        coefficients = lsq_linear(
            design,
            truth_valid - rbf_valid,
            bounds=(np.zeros(2), np.ones(2)),
            method="bvls",
        ).x
        validation_prediction = rbf_valid + design @ coefficients
        validation_gain = float(
            np.sqrt(np.mean(np.square(truth_valid - rbf_valid)))
            - np.sqrt(np.mean(np.square(truth_valid - validation_prediction)))
        )
        if validation_gain < args.validation_gain_threshold:
            coefficients = np.zeros_like(coefficients)
        else:
            coefficients = args.shrinkage * coefficients

        rbf = read_prediction(args.rbf_dir, dataset, args.base_model)
        tan = read_prediction(args.tanimoto_dir, dataset, "tanimoto_svm")
        mmp = read_prediction(args.mmp_dir, dataset, "mmp_memory")
        np.testing.assert_array_equal(rbf.smiles.to_numpy(), tan.smiles.to_numpy())
        np.testing.assert_array_equal(rbf.smiles.to_numpy(), mmp.smiles.to_numpy())
        prediction = (
            rbf.prediction.to_numpy(dtype=float)
            + coefficients[0]
            * (tan.prediction.to_numpy(dtype=float) - rbf.prediction.to_numpy(dtype=float))
            + coefficients[1]
            * (mmp.prediction.to_numpy(dtype=float) - rbf.prediction.to_numpy(dtype=float))
        )
        metric = regression_metrics(
            rbf.target.to_numpy(dtype=float), prediction, rbf.cliff_mol.to_numpy(dtype=bool)
        )
        metric.update(
            {
                "dataset": dataset,
                "model": "action_ensemble",
                "base_model": args.base_model,
                "tanimoto_weight": float(coefficients[0]),
                "mmp_action_weight": float(coefficients[1]),
                "validation_gain": validation_gain,
                "validation_gain_threshold": args.validation_gain_threshold,
                "shrinkage": args.shrinkage,
            }
        )
        rows.append(metric)
        output = rbf.copy()
        output["prediction"] = prediction
        output["base_model"] = args.base_model
        output["tanimoto_weight"] = coefficients[0]
        output["mmp_action_weight"] = coefficients[1]
        output.to_csv(args.output_dir / f"{dataset}.action_ensemble.predictions.csv", index=False)
        print(json.dumps(metric, sort_keys=True), flush=True)

    summary = pd.DataFrame(rows).sort_values("dataset")
    summary.to_csv(args.output_dir / "summary.action_ensemble.csv", index=False)
    aggregate = {
        "model": "action_ensemble",
        "base_model": args.base_model,
        "datasets": len(summary),
        "macro_rmse": float(summary.rmse.mean()),
        "macro_cliff_rmse": float(summary.cliff_rmse.mean()),
        "macro_noncliff_rmse": float(summary.noncliff_rmse.mean()),
    }
    (args.output_dir / "aggregate.action_ensemble.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(aggregate, sort_keys=True))


if __name__ == "__main__":
    main()
