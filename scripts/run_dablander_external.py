from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from molcliff.baselines import fit_predict
from molcliff.calibrated_action import CalibratedMMPActionRegressor
from molcliff.data import Fingerprints, PROJECT_ROOT
from molcliff.metrics import regression_metrics


DATASETS = {
    "chembl_dopamine_d2": ("Ki [nM]", 9.0),
    "chembl_factor_xa": ("Ki [nM]", 9.0),
    "postera_sars_cov_2_mpro": ("f_avg_IC50 [uM]", 6.0),
}


def load_dataset(name: str) -> tuple[pd.DataFrame, set[str]]:
    value_column, offset = DATASETS[name]
    root = PROJECT_ROOT / "third_party" / "QSAR-activity-cliff-experiments" / "data" / name
    molecules = pd.read_csv(root / "molecule_data_clean.csv").rename(columns={"SMILES": "smiles"})
    molecules["target"] = offset - np.log10(molecules[value_column].to_numpy(dtype=float))
    if molecules.smiles.duplicated().any():
        raise ValueError(f"{name}: duplicate SMILES")
    target = dict(zip(molecules.smiles.astype(str), molecules.target.astype(float), strict=True))
    mmps = pd.read_csv(root / "MMP_data_clean.csv")
    cliff_smiles: set[str] = set()
    for row in mmps.itertuples(index=False):
        left, right = str(row.smiles_1), str(row.smiles_2)
        if left in target and right in target and abs(target[left] - target[right]) >= 2.0:
            cliff_smiles.update((left, right))
    return molecules[["smiles", "target"]].copy(), cliff_smiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dablander_external_v1"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fp = Fingerprints()
    rows = []
    for dataset in DATASETS:
        frame, cliff_smiles = load_dataset(dataset)
        folds = KFold(n_splits=2, shuffle=True, random_state=42)
        for fold, (train_idx, test_idx) in enumerate(folds.split(frame)):
            train = frame.iloc[train_idx].reset_index(drop=True)
            test = frame.iloc[test_idx].reset_index(drop=True)
            cliff = test.smiles.isin(cliff_smiles).to_numpy(dtype=bool)
            x_train = fp.bits(train.smiles.to_numpy())
            x_test = fp.bits(test.smiles.to_numpy())
            predictions = {
                "svm": fit_predict(
                    "svm", x_train, train.target.to_numpy(dtype=float), x_test
                ),
                "calibrated_mmp": CalibratedMMPActionRegressor()
                .fit(train.smiles.tolist(), train.target.to_numpy(dtype=float))
                .predict(test.smiles.tolist()),
            }
            output = test.copy()
            output["cliff_mol"] = cliff
            output["fold"] = fold
            for model, prediction in predictions.items():
                metric = regression_metrics(test.target.to_numpy(dtype=float), prediction, cliff)
                metric.update({"dataset": dataset, "fold": fold, "model": model})
                rows.append(metric)
                output[f"prediction_{model}"] = prediction
                print(json.dumps(metric, sort_keys=True), flush=True)
            output.to_csv(args.output_dir / f"{dataset}.fold{fold}.predictions.csv", index=False)
    result = pd.DataFrame(rows)
    result.to_csv(args.output_dir / "summary.csv", index=False)
    aggregate = result.groupby("model")[["rmse", "cliff_rmse", "noncliff_rmse"]].mean()
    aggregate.to_csv(args.output_dir / "aggregate.csv")
    print(aggregate.to_string())


if __name__ == "__main__":
    main()
