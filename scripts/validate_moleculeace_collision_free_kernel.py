from __future__ import annotations

"""Internal validation for a collision-free sparse binary+count kernel."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy import sparse
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR

from molcliff.data import MOLECULEACE_DATASETS, load_moleculeace, split_official_train
from molcliff.metrics import regression_metrics
from run_moleculeace_local_chemistry_kernel import _generators


def sparse_counts(smiles: np.ndarray, generator: object,
                  vocabulary: dict[int, int] | None = None) -> tuple[sparse.csr_matrix, dict[int, int]]:
    records = []
    keys: set[int] = set()
    for value in smiles:
        mol = Chem.MolFromSmiles(str(value))
        if mol is None:
            raise ValueError(f"RDKit parse failure: {value}")
        record = {int(k): float(v) for k, v in
                  generator.GetSparseCountFingerprint(mol).GetNonzeroElements().items()}
        records.append(record)
        keys.update(record)
    if vocabulary is None:
        vocabulary = {key: i for i, key in enumerate(sorted(keys))}
    rows, cols, values = [], [], []
    for row, record in enumerate(records):
        for key, value in record.items():
            col = vocabulary.get(key)
            if col is not None:
                rows.append(row); cols.append(col); values.append(value)
    matrix = sparse.csr_matrix((values, (rows, cols)),
                               shape=(len(records), len(vocabulary)), dtype=np.float32)
    return matrix, vocabulary


def sparse_tanimoto(left: sparse.csr_matrix, right: sparse.csr_matrix) -> np.ndarray:
    product = (left @ right.T).toarray().astype(np.float32, copy=False)
    left_norm = np.asarray(left.multiply(left).sum(1)).reshape(-1, 1)
    right_norm = np.asarray(right.multiply(right).sum(1)).reshape(1, -1)
    return product / np.maximum(left_norm + right_norm - product, 1e-8)


def run_one(dataset: str) -> list[dict]:
    fit, valid = split_official_train(load_moleculeace(dataset))
    generators = _generators()
    binary_fit = np.zeros((len(fit), len(fit)), dtype=np.float32)
    binary_valid = np.zeros((len(valid), len(fit)), dtype=np.float32)
    count_fit = np.zeros_like(binary_fit)
    count_valid = np.zeros_like(binary_valid)
    dimensions = []
    for _name, generator in generators:
        fit_count, vocabulary = sparse_counts(fit.smiles.to_numpy(), generator)
        valid_count, _ = sparse_counts(valid.smiles.to_numpy(), generator, vocabulary)
        fit_binary, valid_binary = fit_count.copy(), valid_count.copy()
        fit_binary.data[:] = 1.0; valid_binary.data[:] = 1.0
        binary_fit += sparse_tanimoto(fit_binary, fit_binary) / len(generators)
        binary_valid += sparse_tanimoto(valid_binary, fit_binary) / len(generators)
        count_fit += sparse_tanimoto(fit_count, fit_count) / len(generators)
        count_valid += sparse_tanimoto(valid_count, fit_count) / len(generators)
        dimensions.append(len(vocabulary))
    # Two complementary exact path semantics absent from circular Morgan and
    # unordered atom-pair views: contiguous paths and length-four torsions.
    semantic_fit = (binary_fit + count_fit) * len(generators)
    semantic_valid = (binary_valid + count_valid) * len(generators)
    semantic_generators = (
        rdFingerprintGenerator.GetTopologicalTorsionGenerator(includeChirality=True),
        rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=7),
    )
    for generator in semantic_generators:
        fit_count, vocabulary = sparse_counts(fit.smiles.to_numpy(), generator)
        valid_count, _ = sparse_counts(valid.smiles.to_numpy(), generator, vocabulary)
        fit_binary, valid_binary = fit_count.copy(), valid_count.copy()
        fit_binary.data[:] = 1.0; valid_binary.data[:] = 1.0
        semantic_fit += sparse_tanimoto(fit_binary, fit_binary) + sparse_tanimoto(fit_count, fit_count)
        semantic_valid += sparse_tanimoto(valid_binary, fit_binary) + sparse_tanimoto(valid_count, fit_count)
    semantic_fit /= 2 * (len(generators) + len(semantic_generators))
    semantic_valid /= 2 * (len(generators) + len(semantic_generators))
    rows = []
    for name, fit_kernel, valid_kernel in (
        ("collision_free_binary_kernel", binary_fit, binary_valid),
        ("collision_free_count_kernel", count_fit, count_valid),
        ("collision_free_binary_count_kernel", (binary_fit + count_fit) / 2,
         (binary_valid + count_valid) / 2),
        ("collision_free_semantic_kernel", semantic_fit, semantic_valid),
    ):
        model = SVR(C=10.0, epsilon=0.1, kernel="precomputed")
        model.fit(fit_kernel, fit.target.to_numpy(dtype=float))
        prediction = model.predict(valid_kernel)
        metric = regression_metrics(valid.target.to_numpy(dtype=float), prediction,
                                    valid.cliff_mol.to_numpy(dtype=bool))
        rows.append({"dataset": dataset, "model": name,
                     "dimensions": json.dumps(dimensions), **metric})
    # Least-squares kernel potential with regularization analytically matched
    # to the frozen SVR penalty C=10: alpha=1/(2C).  This is one preregistered
    # estimator comparison, not an alpha sweep.
    ridge = KernelRidge(alpha=0.05, kernel="precomputed")
    ridge.fit((binary_fit + count_fit) / 2, fit.target.to_numpy(dtype=float))
    prediction = ridge.predict((binary_valid + count_valid) / 2)
    metric = regression_metrics(valid.target.to_numpy(dtype=float), prediction,
                                valid.cliff_mol.to_numpy(dtype=bool))
    rows.append({"dataset": dataset, "model": "collision_free_binary_count_krr",
                 "dimensions": json.dumps(dimensions), **metric})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/moleculeace_collision_free_validation_v1"))
    args = parser.parse_args()
    nested = Parallel(n_jobs=args.jobs)(delayed(run_one)(name) for name in MOLECULEACE_DATASETS)
    summary = pd.DataFrame([row for rows in nested for row in rows]).sort_values(["model", "dataset"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "summary.validation.csv", index=False)
    macro = summary.groupby("model")[["rmse", "cliff_rmse", "noncliff_rmse"]].mean()
    aggregate = {name: {key: float(value) for key, value in row.items()}
                 for name, row in macro.iterrows()}
    aggregate["protocol"] = "fixed official-train 80/20 seed42 stratified; zero test evaluations"
    (args.output_dir / "aggregate.validation.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(aggregate, sort_keys=True))


if __name__ == "__main__":
    main()
