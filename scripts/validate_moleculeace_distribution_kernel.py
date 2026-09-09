from __future__ import annotations

"""Internal validation of a KerRead-inspired environment-distribution view."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.svm import SVR

from molcliff.data import MOLECULEACE_DATASETS, load_moleculeace, split_official_train
from molcliff.metrics import regression_metrics
from run_moleculeace_local_chemistry_kernel import _generators
from validate_moleculeace_collision_free_kernel import sparse_counts, sparse_tanimoto


def run_one(dataset: str) -> list[dict]:
    fit, valid = split_official_train(load_moleculeace(dataset))
    chemistry_fit = np.zeros((len(fit), len(fit)), dtype=np.float32)
    chemistry_valid = np.zeros((len(valid), len(fit)), dtype=np.float32)
    distribution_fit = np.zeros_like(chemistry_fit)
    distribution_valid = np.zeros_like(chemistry_valid)
    scales = []
    generators = _generators()
    for _, generator in generators:
        fit_count, vocab = sparse_counts(fit.smiles.to_numpy(), generator)
        valid_count, _ = sparse_counts(valid.smiles.to_numpy(), generator, vocab)
        fit_binary, valid_binary = fit_count.copy(), valid_count.copy()
        fit_binary.data[:] = 1.0
        valid_binary.data[:] = 1.0
        chemistry_fit += (sparse_tanimoto(fit_binary, fit_binary)
                          + sparse_tanimoto(fit_count, fit_count)) / (2 * len(generators))
        chemistry_valid += (sparse_tanimoto(valid_binary, fit_binary)
                            + sparse_tanimoto(valid_count, fit_count)) / (2 * len(generators))

        # L1-normalized exact environment measures form empirical kernel mean
        # embeddings.  Their Laplacian kernel is PSD and responds directly to
        # probability mass moved by a local chemical edit.
        fit_mass = np.asarray(fit_count.sum(1)).reshape(-1)
        valid_mass = np.asarray(valid_count.sum(1)).reshape(-1)
        fit_measure = fit_count.multiply(1.0 / np.maximum(fit_mass, 1.0)[:, None])
        valid_measure = valid_count.multiply(1.0 / np.maximum(valid_mass, 1.0)[:, None])
        distance = manhattan_distances(fit_measure, fit_measure)
        scale = max(float(np.median(distance[np.triu_indices(len(fit), 1)])), 1e-8)
        scales.append(scale)
        distribution_fit += np.exp(-distance / scale).astype(np.float32) / len(generators)
        distribution_valid += np.exp(
            -manhattan_distances(valid_measure, fit_measure) / scale
        ).astype(np.float32) / len(generators)

    rows = []
    for name, train_kernel, valid_kernel in (
        ("collision_free_binary_count_kernel", chemistry_fit, chemistry_valid),
        ("environment_distribution_kernel", distribution_fit, distribution_valid),
        ("collision_free_plus_distribution_10to1",
         (10 * chemistry_fit + distribution_fit) / 11,
         (10 * chemistry_valid + distribution_valid) / 11),
    ):
        model = SVR(C=10.0, epsilon=0.1, kernel="precomputed")
        model.fit(train_kernel, fit.target.to_numpy(dtype=float))
        pred = model.predict(valid_kernel)
        rows.append({"dataset": dataset, "model": name, "l1_scales": json.dumps(scales),
                     **regression_metrics(valid.target.to_numpy(dtype=float), pred,
                                          valid.cliff_mol.to_numpy(dtype=bool))})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(MOLECULEACE_DATASETS))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/moleculeace_distribution_validation_v1"))
    args = parser.parse_args()
    nested = Parallel(n_jobs=args.jobs)(delayed(run_one)(d) for d in args.datasets)
    summary = pd.DataFrame([row for group in nested for row in group]).sort_values(["model", "dataset"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "summary.validation.csv", index=False)
    macro = summary.groupby("model")[["rmse", "cliff_rmse", "noncliff_rmse"]].mean()
    result = {name: {k: float(v) for k, v in row.items()} for name, row in macro.iterrows()}
    result["protocol"] = "train-only median L1 scales; 10 chemical views + 1 distribution view; no search"
    (args.output_dir / "aggregate.validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
