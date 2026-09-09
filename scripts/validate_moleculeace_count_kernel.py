from __future__ import annotations

"""Fixed internal validation of binary versus binary+count chemistry kernels."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem, DataStructs
from sklearn.svm import SVR

from molcliff.data import MOLECULEACE_DATASETS, load_moleculeace, split_official_train, tanimoto_matrix
from molcliff.metrics import regression_metrics
from run_moleculeace_local_chemistry_kernel import FP_SIZE, _generators


def _representations(smiles: np.ndarray, generator: object) -> tuple[np.ndarray, np.ndarray]:
    mols = [Chem.MolFromSmiles(str(value)) for value in smiles]
    if any(mol is None for mol in mols):
        raise ValueError("RDKit parse failure")
    bits = np.zeros((len(mols), FP_SIZE), dtype=np.float32)
    counts = np.zeros((len(mols), FP_SIZE), dtype=np.float32)
    for i, mol in enumerate(mols):
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), bits[i])
        DataStructs.ConvertToNumpyArray(generator.GetCountFingerprint(mol), counts[i])
    return bits, counts


def _generalized_tanimoto(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    product = left @ right.T
    denominator = ((left * left).sum(1, keepdims=True)
                   + (right * right).sum(1)[None, :] - product)
    return product / np.maximum(denominator, 1e-8)


def run_one(dataset: str) -> list[dict]:
    fit, valid = split_official_train(load_moleculeace(dataset))
    generators = _generators()
    binary_train = np.zeros((len(fit), len(fit)), dtype=np.float32)
    binary_valid = np.zeros((len(valid), len(fit)), dtype=np.float32)
    count_train = np.zeros_like(binary_train)
    count_valid = np.zeros_like(binary_valid)
    for _name, generator in generators:
        fit_bits, fit_counts = _representations(fit.smiles.to_numpy(), generator)
        valid_bits, valid_counts = _representations(valid.smiles.to_numpy(), generator)
        binary_train += tanimoto_matrix(fit_bits, fit_bits) / len(generators)
        binary_valid += tanimoto_matrix(valid_bits, fit_bits) / len(generators)
        count_train += _generalized_tanimoto(fit_counts, fit_counts) / len(generators)
        count_valid += _generalized_tanimoto(valid_counts, fit_counts) / len(generators)
    rows = []
    for name, train_kernel, valid_kernel in (
        ("binary_local_kernel", binary_train, binary_valid),
        ("binary_count_local_kernel", (binary_train + count_train) / 2.0,
         (binary_valid + count_valid) / 2.0),
    ):
        model = SVR(C=10.0, epsilon=0.1, kernel="precomputed")
        model.fit(train_kernel, fit.target.to_numpy(dtype=float))
        prediction = model.predict(valid_kernel)
        metric = regression_metrics(valid.target.to_numpy(dtype=float), prediction,
                                    valid.cliff_mol.to_numpy(dtype=bool))
        rows.append({"dataset": dataset, "model": name, **metric})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/moleculeace_count_kernel_validation_v1"))
    args = parser.parse_args()
    nested = Parallel(n_jobs=args.jobs)(delayed(run_one)(name) for name in MOLECULEACE_DATASETS)
    summary = pd.DataFrame([row for rows in nested for row in rows]).sort_values(["model", "dataset"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "summary.validation.csv", index=False)
    aggregate = summary.groupby("model")[["rmse", "cliff_rmse", "noncliff_rmse"]].mean()
    payload = {name: {key: float(value) for key, value in row.items()}
               for name, row in aggregate.iterrows()}
    payload["protocol"] = {"split": "fixed official-train 80/20 seed42 stratified",
                           "test_evaluations": 0, "kernel_selection": "validation only"}
    (args.output_dir / "aggregate.validation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
