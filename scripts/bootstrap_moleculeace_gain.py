from __future__ import annotations

"""Paired target-cluster bootstrap for the frozen MoleculeACE comparison."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from molcliff.data import MOLECULEACE_DATASETS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260902,
                        help="Reporting-only bootstrap RNG; never used for model fitting")
    parser.add_argument("--output", type=Path,
                        default=Path("reports/results/collision_free_bootstrap.json"))
    args = parser.parse_args()
    per_target = []
    for dataset in MOLECULEACE_DATASETS:
        base = pd.read_csv(Path("outputs/moleculeace_official_baselines")
                           / f"{dataset}.svm_official.predictions.csv")
        model = pd.read_csv(Path("outputs/moleculeace_collision_free_kernel_v1")
                            / f"{dataset}.collision_free_kernel.predictions.csv")
        if base.smiles.tolist() != model.smiles.tolist():
            raise RuntimeError(f"Prediction row mismatch: {dataset}")
        target = base.target.to_numpy(dtype=float)
        per_target.append({
            "overall": (np.square(target - base.prediction),
                        np.square(target - model.prediction)),
            "cliff": (np.square(target[base.cliff_mol.astype(bool)]
                                - base.prediction.to_numpy()[base.cliff_mol.astype(bool)]),
                      np.square(target[base.cliff_mol.astype(bool)]
                                - model.prediction.to_numpy()[base.cliff_mol.astype(bool)])),
        })
    rng = np.random.default_rng(args.seed)
    result = {}
    for subset, threshold in (("overall", 0.10), ("cliff", 0.20)):
        draws = np.empty(args.samples, dtype=np.float64)
        for draw in range(args.samples):
            targets = rng.integers(0, len(per_target), size=len(per_target))
            base_macro, model_macro = [], []
            for target_index in targets:
                base_error, model_error = per_target[int(target_index)][subset]
                rows = rng.integers(0, len(base_error), size=len(base_error))
                base_macro.append(float(np.sqrt(base_error[rows].mean())))
                model_macro.append(float(np.sqrt(model_error[rows].mean())))
            draws[draw] = 1.0 - np.mean(model_macro) / np.mean(base_macro)
        result[subset] = {
            "relative_gain_mean": float(draws.mean()),
            "relative_gain_ci95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
            "probability_gain_positive": float(np.mean(draws > 0)),
            "required_gain": threshold,
            "probability_meets_required_gain": float(np.mean(draws >= threshold)),
        }
    result["protocol"] = {
        "samples": args.samples, "seed": args.seed,
        "resampling": "paired targets and paired molecules within target",
        "reporting_only": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
