from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from molcliff.baselines import fit_predict
from molcliff.data import (
    Fingerprints,
    MOLECULEACE_DATASETS,
    load_moleculeace,
    tanimoto_matrix,
)
from molcliff.metrics import regression_metrics


def run_one(dataset: str, model: str, output_dir: Path) -> dict:
    frame = load_moleculeace(dataset)
    train = frame[frame.split.eq("train")].reset_index(drop=True)
    test = frame[frame.split.eq("test")].reset_index(drop=True)
    fps = Fingerprints(radius=2, n_bits=1024)
    x_train = fps.bits(train.smiles.to_numpy())
    x_test = fps.bits(test.smiles.to_numpy())

    if model == "tanimoto_svm":
        from sklearn.svm import SVR

        estimator = SVR(C=10.0, epsilon=0.1, kernel="precomputed")
        estimator.fit(tanimoto_matrix(x_train, x_train), train.target.to_numpy(dtype=float))
        pred = estimator.predict(tanimoto_matrix(x_test, x_train))
    elif model in {"svm", "svm_official", "rf", "rf_official", "gbm_official"}:
        pred = fit_predict(
            model,
            x_train,
            train.target.to_numpy(dtype=float),
            x_test,
            dataset=dataset,
        )
    elif model == "action_anchor":
        from molcliff.action_anchor import ActionAnchorRegressor

        estimator = ActionAnchorRegressor(seed=42)
        estimator.fit(train.smiles.tolist(), train.target.to_numpy(dtype=float))
        pred = estimator.predict(test.smiles.tolist())
    elif model == "mmp_memory":
        from molcliff.mmp_action import MMPActionMemoryRegressor

        estimator = MMPActionMemoryRegressor()
        estimator.fit(train.smiles.tolist(), train.target.to_numpy(dtype=float))
        pred = estimator.predict(test.smiles.tolist())
    elif model == "core_anchored_mmp":
        from molcliff.mmp_action import CoreAnchoredMMPRegressor

        estimator = CoreAnchoredMMPRegressor()
        estimator.fit(train.smiles.tolist(), train.target.to_numpy(dtype=float))
        pred = estimator.predict(test.smiles.tolist())
    elif model == "calibrated_mmp":
        from molcliff.calibrated_action import CalibratedMMPActionRegressor

        estimator = CalibratedMMPActionRegressor()
        estimator.fit(train.smiles.tolist(), train.target.to_numpy(dtype=float))
        pred = estimator.predict(test.smiles.tolist())
    elif model == "integrable_action":
        from molcliff.integrable_action import IntegrableActionRegressor

        estimator = IntegrableActionRegressor()
        estimator.fit(train.smiles.tolist(), train.target.to_numpy(dtype=float))
        pred = estimator.predict(test.smiles.tolist())
    elif model == "action_kernel":
        from molcliff.action_kernel import ActionKernelDeltaRegressor

        estimator = ActionKernelDeltaRegressor()
        estimator.fit(train.smiles.tolist(), train.target.to_numpy(dtype=float))
        pred = estimator.predict(test.smiles.tolist())
    elif model == "action_retrieval":
        from molcliff.action_kernel import ActionRetrievalDeltaRegressor

        estimator = ActionRetrievalDeltaRegressor()
        estimator.fit(train.smiles.tolist(), train.target.to_numpy(dtype=float))
        pred = estimator.predict(test.smiles.tolist())
    elif model == "action_potential":
        from molcliff.action_potential import ActionPotentialRegressor

        estimator = ActionPotentialRegressor()
        estimator.fit(train.smiles.tolist(), train.target.to_numpy(dtype=float))
        pred = estimator.predict(test.smiles.tolist())
    elif model == "neural_action":
        from molcliff.neural_action import NeuralActionDeltaRegressor

        estimator = NeuralActionDeltaRegressor()
        estimator.fit(train.smiles.tolist(), train.target.to_numpy(dtype=float))
        pred = estimator.predict(test.smiles.tolist())
    elif model == "neural_action_potential":
        from molcliff.neural_action import NeuralActionPotentialRegressor

        estimator = NeuralActionPotentialRegressor()
        estimator.fit(train.smiles.tolist(), train.target.to_numpy(dtype=float))
        pred = estimator.predict(test.smiles.tolist())
    elif model == "external_action":
        from molcliff.external_action import ExternalSingleAnchorActionRegressor

        estimator = ExternalSingleAnchorActionRegressor()
        estimator.fit(train.smiles.tolist(), train.target.to_numpy(dtype=float))
        pred = estimator.predict(test.smiles.tolist())
    elif model == "prototype_action":
        from molcliff.prototype_action import PrototypeActionTransportRegressor

        estimator = PrototypeActionTransportRegressor()
        estimator.fit(train.smiles.tolist(), train.target.to_numpy(dtype=float))
        pred = estimator.predict(test.smiles.tolist())
    elif model == "action_laplacian":
        from molcliff.action_laplacian import ActionGatedLaplacianRegressor

        estimator = ActionGatedLaplacianRegressor()
        estimator.fit(train.smiles.tolist(), train.target.to_numpy(dtype=float))
        pred = estimator.predict(test.smiles.tolist())
    else:
        raise ValueError(model)

    metrics = regression_metrics(
        test.target.to_numpy(dtype=float), pred, test.cliff_mol.to_numpy(dtype=bool)
    )
    metrics.update({"dataset": dataset, "model": model, "seed": 42})
    if model == "calibrated_mmp":
        metrics.update(
            {
                "action_shrinkage": estimator.shrinkage_,
                "raw_action_shrinkage": estimator.raw_shrinkage_,
                "calibration_size": estimator.calibration_size_,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = test[["dataset", "smiles", "target", "cliff_mol", "split"]].copy()
    predictions["prediction"] = pred
    predictions.to_csv(output_dir / f"{dataset}.{model}.predictions.csv", index=False)
    (output_dir / f"{dataset}.{model}.metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metrics, sort_keys=True), flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all")
    parser.add_argument(
        "--model",
        choices=(
            "svm", "svm_official", "tanimoto_svm", "rf", "rf_official", "gbm_official",
            "action_anchor", "mmp_memory",
            "core_anchored_mmp", "calibrated_mmp", "integrable_action",
            "action_kernel", "action_retrieval", "action_potential",
            "neural_action", "neural_action_potential", "external_action",
            "prototype_action", "action_laplacian",
        ),
        default="svm",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/moleculeace"))
    args = parser.parse_args()
    datasets = MOLECULEACE_DATASETS if args.dataset == "all" else (args.dataset,)
    rows = [run_one(name, args.model, args.output_dir) for name in datasets]
    summary = pd.DataFrame(rows).sort_values("dataset")
    summary.to_csv(args.output_dir / f"summary.{args.model}.csv", index=False)
    aggregate = {
        "model": args.model,
        "datasets": len(summary),
        "macro_rmse": float(summary.rmse.mean()),
        "macro_cliff_rmse": float(summary.cliff_rmse.mean()),
        "macro_noncliff_rmse": float(summary.noncliff_rmse.mean()),
    }
    (args.output_dir / f"aggregate.{args.model}.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(aggregate, sort_keys=True))


if __name__ == "__main__":
    main()
