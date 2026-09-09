from __future__ import annotations

"""Run fixed ECFP baselines under the manuscript's regression CV protocols."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVR

from molcliff.data import (
    EXPANDED_CLIFF_ROOT,
    Fingerprints,
    MOLECULEACE_DATASETS,
    load_moleculeace,
)
from molcliff.metrics import regression_metrics


EXTERNAL_DATASETS = (
    "CHEMBL2311243_IC50",
    "CHEMBL2598_IC50",
    "CHEMBL3024_IC50",
    "CHEMBL4685_IC50",
    "CHEMBL5014_IC50",
)
MODELS = (
    "linear_ecfp", "svm_ecfp", "hist_gradient_boosting_ecfp", "mlp_ecfp",
    "extratrees_ecfp", "random_forest_ecfp",
)


def make_model(name: str):
    if name == "linear_ecfp":
        return Ridge(alpha=1.0)
    if name == "svm_ecfp":
        return SVR(C=10.0, epsilon=0.1, kernel="rbf", gamma="scale")
    if name == "hist_gradient_boosting_ecfp":
        return HistGradientBoostingRegressor(
            max_iter=100, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1e-3, random_state=42,
        )
    if name == "mlp_ecfp":
        return MLPRegressor(
            hidden_layer_sizes=(256, 64), activation="relu", alpha=1e-4,
            learning_rate_init=1e-3, max_iter=120, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=12, random_state=42,
        )
    common = dict(
        n_estimators=192,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1,
    )
    if name == "extratrees_ecfp":
        return ExtraTreesRegressor(**common)
    if name == "random_forest_ecfp":
        return RandomForestRegressor(**common)
    raise ValueError(name)


def run_target(
    dataset: str, seeds: tuple[int, ...], external: bool, models: tuple[str, ...]
) -> list[dict]:
    root = EXPANDED_CLIFF_ROOT if external else None
    frame = load_moleculeace(dataset, root=root)
    pool = frame[frame.split.eq("train")].reset_index(drop=True)
    x = Fingerprints(radius=2, n_bits=1024).bits(pool.smiles.to_numpy())
    y = pool.target.to_numpy(float)
    cliff = pool.cliff_mol.to_numpy(bool)
    rows: list[dict] = []
    for seed in seeds:
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (train_index, test_index) in enumerate(folds.split(x, cliff)):
            for name in models:
                model = make_model(name)
                model.fit(x[train_index], y[train_index])
                prediction = model.predict(x[test_index])
                rows.append({
                    "collection": "GraphCliff external" if external else "MoleculeACE",
                    "dataset": dataset,
                    "model": name,
                    "seed": seed,
                    "fold": fold,
                    **regression_metrics(y[test_index], prediction, cliff[test_index]),
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument(
        "--collections", nargs="+", choices=("moleculeace", "external"),
        default=["moleculeace", "external"],
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/cross_dataset_regression_baselines/summary.csv"),
    )
    args = parser.parse_args()
    tasks = ([
        (dataset, (7, 42, 73), False, tuple(args.models)) for dataset in MOLECULEACE_DATASETS
    ] if "moleculeace" in args.collections else []) + ([
        (dataset, (42,), True, tuple(args.models)) for dataset in EXTERNAL_DATASETS
    ] if "external" in args.collections else [])
    nested = Parallel(n_jobs=args.jobs, verbose=10)(
        delayed(run_target)(dataset, seeds, external, models)
        for dataset, seeds, external, models in tasks
    )
    result = pd.DataFrame([row for part in nested for row in part])
    result = result.sort_values(["collection", "model", "dataset", "seed", "fold"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    target = (
        result.groupby(["collection", "model", "dataset"])[
            ["rmse", "cliff_rmse", "noncliff_rmse"]
        ]
        .mean()
        .reset_index()
    )
    protocols = {
        "features": "1024-bit radius-2 ECFP",
        "selection": "fixed global hyperparameters; no target-specific tuning",
    }
    if "moleculeace" in args.collections:
        protocols["MoleculeACE"] = "official-train-only stratified five-fold CV; seeds 7, 42, 73"
    if "external" in args.collections:
        protocols["GraphCliff external"] = "official-train-only stratified five-fold CV; seed 42"
    payload = {
        "protocol": protocols,
        "macro": (
            target.groupby(["collection", "model"])[
                ["rmse", "cliff_rmse", "noncliff_rmse"]
            ]
            .mean()
            .reset_index()
            .to_dict(orient="records")
        ),
        "targets": target.groupby("collection").dataset.nunique().to_dict(),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
