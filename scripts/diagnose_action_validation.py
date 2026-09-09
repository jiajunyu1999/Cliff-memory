from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from molcliff.action_anchor import ActionAnchorRegressor
from molcliff.data import MOLECULEACE_DATASETS, load_moleculeace, split_official_train
from molcliff.metrics import regression_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="CHEMBL204_Ki")
    parser.add_argument(
        "--method",
        choices=(
            "neural", "mmp_memory", "core_anchored_mmp", "calibrated_mmp",
            "integrable_action",
        ),
        default="neural",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/action_diagnostics"))
    args = parser.parse_args()
    datasets = MOLECULEACE_DATASETS if args.dataset == "all" else (args.dataset,)
    rows = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        fit, valid = split_official_train(load_moleculeace(dataset))
        if args.method == "neural":
            model = ActionAnchorRegressor(seed=42).fit(
                fit.smiles.tolist(), fit.target.to_numpy(dtype=float)
            )
        elif args.method == "mmp_memory":
            from molcliff.mmp_action import MMPActionMemoryRegressor

            model = MMPActionMemoryRegressor().fit(
                fit.smiles.tolist(), fit.target.to_numpy(dtype=float)
            )
        elif args.method == "core_anchored_mmp":
            from molcliff.mmp_action import CoreAnchoredMMPRegressor

            model = CoreAnchoredMMPRegressor().fit(
                fit.smiles.tolist(), fit.target.to_numpy(dtype=float)
            )
        elif args.method == "calibrated_mmp":
            from molcliff.calibrated_action import CalibratedMMPActionRegressor

            model = CalibratedMMPActionRegressor().fit(
                fit.smiles.tolist(), fit.target.to_numpy(dtype=float)
            )
        else:
            from molcliff.integrable_action import IntegrableActionRegressor

            model = IntegrableActionRegressor().fit(
                fit.smiles.tolist(), fit.target.to_numpy(dtype=float)
            )
        components = model.predict_components(valid.smiles.tolist())
        truth = valid.target.to_numpy(dtype=float)
        cliff = valid.cliff_mol.to_numpy(dtype=bool)
        names = ("base", "neighbor", "action", "blended") if args.method == "neural" else ("base", "action", "blended")
        for name in names:
            metric = regression_metrics(truth, components[name], cliff)
            metric.update({"dataset": dataset, "component": name})
            rows.append(metric)
        detail = valid[["dataset", "smiles", "target", "cliff_mol"]].copy()
        for name, values in components.items():
            detail[name] = values
        detail.to_csv(args.output_dir / f"{dataset}.csv", index=False)
        print(json.dumps(rows[-len(names):], sort_keys=True), flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(args.output_dir / "summary.csv", index=False)
    aggregate = result.groupby("component")[["rmse", "cliff_rmse", "noncliff_rmse"]].mean()
    print(aggregate.to_string())


if __name__ == "__main__":
    main()
