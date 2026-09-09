from __future__ import annotations

"""Train-only fixed CliffLoss reweighting of the single local chemistry SVR.

The severity construction follows Hu et al. (2026): top similar training
neighbors are ranked by similarity times activity difference and normalized by
the 95th-percentile activity difference.  The published fixed lambda=0.1 is
used; the validation-feedback controller is deliberately omitted.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.svm import SVR

from molcliff.data import MOLECULEACE_DATASETS, load_moleculeace, tanimoto_matrix
from molcliff.metrics import regression_metrics
from run_moleculeace_local_chemistry_kernel import _bits, _generators


LAMBDA = 0.1


def _severity(bits: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict]:
    similarity = tanimoto_matrix(bits, bits)
    np.fill_diagonal(similarity, -1.0)
    k = max(4, min(24, int(np.sqrt(len(bits)))))
    m = max(1, int(np.sqrt(k)))
    collected_delta: list[float] = []
    neighborhoods: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for i in range(len(bits)):
        order = np.argsort(similarity[i])[::-1][:k]
        order = order[similarity[i, order] >= 0.35]
        sim = similarity[i, order]
        delta = np.abs(y[i] - y[order])
        neighborhoods.append((order, sim, delta))
        collected_delta.extend(delta.tolist())
    q95 = float(np.quantile(collected_delta, 0.95)) if collected_delta else 1.0
    q95 = max(q95, 1e-8)
    severity = np.zeros(len(bits), dtype=np.float64)
    for i, (_order, sim, delta) in enumerate(neighborhoods):
        if len(sim):
            strength = sim * delta
            selected = np.argsort(strength)[::-1][:m]
            # Divide by the actual support to avoid penalizing sparse regions.
            severity[i] = float(np.mean(strength[selected] / q95))
    sample_weight = 1.0 + LAMBDA * severity
    return sample_weight, {
        "neighbor_budget": k, "aggregation_budget": m, "similarity_threshold": 0.35,
        "q95": q95, "mean_severity": float(severity.mean()),
        "max_severity": float(severity.max()),
    }


def run_one(dataset: str, output_dir: Path) -> dict:
    frame = load_moleculeace(dataset)
    train = frame[frame.split.eq("train")].reset_index(drop=True)
    test = frame[frame.split.eq("test")].reset_index(drop=True)
    y = train.target.to_numpy(dtype=float)
    train_kernel = np.zeros((len(train), len(train)), dtype=np.float32)
    test_kernel = np.zeros((len(test), len(train)), dtype=np.float32)
    severity_bits = None
    generators = _generators()
    for name, generator in generators:
        x_train = _bits(train.smiles.to_numpy(), generator)
        x_test = _bits(test.smiles.to_numpy(), generator)
        if name == "chiral_morgan_r2":
            severity_bits = x_train
        train_kernel += tanimoto_matrix(x_train, x_train) / len(generators)
        test_kernel += tanimoto_matrix(x_test, x_train) / len(generators)
    if severity_bits is None:
        raise AssertionError("severity fingerprint was not built")
    sample_weight, diagnostics = _severity(severity_bits, y)
    model = SVR(C=10.0, epsilon=0.1, kernel="precomputed")
    model.fit(train_kernel, y, sample_weight=sample_weight)
    prediction = model.predict(test_kernel)
    metric = regression_metrics(
        test.target.to_numpy(dtype=float), prediction, test.cliff_mol.to_numpy(dtype=bool)
    )
    metric.update({"dataset": dataset, "model": "fixed_cliffloss_kernel", "seed": 42,
                   **diagnostics})
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = test[["dataset", "smiles", "target", "cliff_mol", "split"]].copy()
    predictions["prediction"] = prediction
    predictions.to_csv(output_dir / f"{dataset}.fixed_cliffloss_kernel.predictions.csv", index=False)
    return metric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/moleculeace_fixed_cliffloss_kernel_v1"))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    rows = Parallel(n_jobs=args.jobs)(
        delayed(run_one)(dataset, args.output_dir) for dataset in MOLECULEACE_DATASETS
    )
    summary = pd.DataFrame(rows).sort_values("dataset")
    summary.to_csv(args.output_dir / "summary.fixed_cliffloss_kernel.csv", index=False)
    aggregate = {
        "model": "fixed_cliffloss_kernel", "datasets": len(summary), "lambda": LAMBDA,
        "macro_rmse": float(summary.rmse.mean()),
        "macro_cliff_rmse": float(summary.cliff_rmse.mean()),
        "macro_noncliff_rmse": float(summary.noncliff_rmse.mean()),
        "seed": 42, "selection": "none", "ensemble": False,
        "validation_feedback": False, "uses_test_labels_as_input": False,
    }
    (args.output_dir / "aggregate.fixed_cliffloss_kernel.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(aggregate, sort_keys=True))


if __name__ == "__main__":
    main()
