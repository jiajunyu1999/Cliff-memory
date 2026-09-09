from __future__ import annotations

"""Evaluate one fixed kernel that resolves local chemical edits explicitly.

The five equal-weight views are preregistered from chemical invariances rather
than selected on MoleculeACE: three chiral Morgan scales, a pharmacophore
Morgan view, and a long-range atom-pair view.  Their average is one positive
semidefinite kernel and therefore one SVR, not an ensemble.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.svm import SVR

from molcliff.data import MOLECULEACE_DATASETS, load_moleculeace, tanimoto_matrix
from molcliff.metrics import regression_metrics


FP_SIZE = 1024


def _generators() -> tuple[tuple[str, object], ...]:
    feature_invariants = rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
    return (
        ("chiral_morgan_r1", rdFingerprintGenerator.GetMorganGenerator(
            radius=1, fpSize=FP_SIZE, includeChirality=True)),
        ("chiral_morgan_r2", rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=FP_SIZE, includeChirality=True)),
        ("chiral_morgan_r3", rdFingerprintGenerator.GetMorganGenerator(
            radius=3, fpSize=FP_SIZE, includeChirality=True)),
        ("feature_morgan_r2", rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=FP_SIZE, includeChirality=True,
            atomInvariantsGenerator=feature_invariants)),
        ("chiral_atom_pair", rdFingerprintGenerator.GetAtomPairGenerator(
            fpSize=FP_SIZE, includeChirality=True)),
    )


def _bits(smiles: np.ndarray, generator: object) -> np.ndarray:
    mols = [Chem.MolFromSmiles(str(s)) for s in smiles]
    if any(mol is None for mol in mols):
        raise ValueError("RDKit failed to parse a molecule")
    out = np.zeros((len(mols), FP_SIZE), dtype=np.float32)
    for row, mol in zip(out, mols, strict=True):
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), row)
    return out


def run_one(dataset: str, output_dir: Path) -> dict:
    frame = load_moleculeace(dataset)
    train = frame[frame.split.eq("train")].reset_index(drop=True)
    test = frame[frame.split.eq("test")].reset_index(drop=True)
    generators = _generators()
    train_kernel = np.zeros((len(train), len(train)), dtype=np.float32)
    test_kernel = np.zeros((len(test), len(train)), dtype=np.float32)
    for _name, generator in generators:
        x_train = _bits(train.smiles.to_numpy(), generator)
        x_test = _bits(test.smiles.to_numpy(), generator)
        train_kernel += tanimoto_matrix(x_train, x_train) / len(generators)
        test_kernel += tanimoto_matrix(x_test, x_train) / len(generators)

    model = SVR(C=10.0, epsilon=0.1, kernel="precomputed")
    model.fit(train_kernel, train.target.to_numpy(dtype=float))
    prediction = model.predict(test_kernel)
    metric = regression_metrics(
        test.target.to_numpy(dtype=float), prediction, test.cliff_mol.to_numpy(dtype=bool)
    )
    metric.update({"dataset": dataset, "model": "local_chemistry_kernel", "seed": 42})
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = test[["dataset", "smiles", "target", "cliff_mol", "split"]].copy()
    predictions["prediction"] = prediction
    predictions.to_csv(output_dir / f"{dataset}.local_chemistry_kernel.predictions.csv", index=False)
    return metric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/moleculeace_local_chemistry_kernel_v1"))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    rows = Parallel(n_jobs=args.jobs)(
        delayed(run_one)(dataset, args.output_dir) for dataset in MOLECULEACE_DATASETS
    )
    summary = pd.DataFrame(rows).sort_values("dataset")
    summary.to_csv(args.output_dir / "summary.local_chemistry_kernel.csv", index=False)
    aggregate = {
        "model": "local_chemistry_kernel",
        "datasets": len(summary),
        "views": [name for name, _ in _generators()],
        "kernel_weights": "equal",
        "macro_rmse": float(summary.rmse.mean()),
        "macro_cliff_rmse": float(summary.cliff_rmse.mean()),
        "macro_noncliff_rmse": float(summary.noncliff_rmse.mean()),
        "ensemble": False,
        "selection": "none",
    }
    (args.output_dir / "aggregate.local_chemistry_kernel.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(aggregate, sort_keys=True))


if __name__ == "__main__":
    main()
