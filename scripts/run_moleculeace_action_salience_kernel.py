from __future__ import annotations

"""Train-only MMP salience metric followed by a single fixed-kernel SVR."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.svm import SVR

from molcliff.data import MOLECULEACE_DATASETS, load_moleculeace
from molcliff.metrics import regression_metrics
from molcliff.mmp_action import single_cuts
from run_moleculeace_local_chemistry_kernel import _bits, _generators


def _mmp_pairs(smiles: list[str]) -> list[tuple[int, int]]:
    groups: dict[str, set[int]] = defaultdict(set)
    for index, smi in enumerate(smiles):
        for cut in single_cuts(smi):
            groups[cut.core].add(index)
    pairs: set[tuple[int, int]] = set()
    for members in groups.values():
        ordered = sorted(members)
        for left_pos, left in enumerate(ordered):
            for right in ordered[left_pos + 1:]:
                pairs.add((left, right))
    return sorted(pairs)


def _action_weights(bits: np.ndarray, y: np.ndarray,
                    pairs: list[tuple[int, int]]) -> np.ndarray:
    """Closed-form empirical-Bayes weight for bits changed by exact MMPs."""
    document_frequency = bits.sum(axis=0, dtype=np.float64)
    idf = np.log((len(bits) + 1.0) / (document_frequency + 1.0)) + 1.0
    effect_sum = np.zeros(bits.shape[1], dtype=np.float64)
    support = np.zeros(bits.shape[1], dtype=np.float64)
    for left, right in pairs:
        changed = bits[left] != bits[right]
        if not np.any(changed):
            continue
        effect_sum[changed] += float((y[right] - y[left]) ** 2)
        support[changed] += 1.0
    observed = support > 0
    if not np.any(observed):
        return idf.astype(np.float32)
    mean_effect = np.zeros_like(effect_sum)
    mean_effect[observed] = effect_sum[observed] / support[observed]
    global_effect = float(effect_sum.sum() / max(support.sum(), 1.0))
    median_support = float(np.median(support[observed]))
    reliability = support / (support + median_support)
    salience = mean_effect / max(global_effect, 1e-12)
    # Unit background plus a reliability-shrunk action contribution.  The
    # final normalization changes only kernel geometry, not SVR regularization.
    weight = idf * (1.0 + reliability * salience)
    weight /= max(float(weight.mean()), 1e-12)
    return weight.astype(np.float32)


def _weighted_tanimoto(left: np.ndarray, right: np.ndarray,
                       weight: np.ndarray) -> np.ndarray:
    left_weighted = left * weight[None, :]
    intersection = left_weighted @ right.T
    left_mass = left_weighted.sum(axis=1, keepdims=True)
    right_mass = (right * weight[None, :]).sum(axis=1)[None, :]
    return intersection / np.maximum(left_mass + right_mass - intersection, 1e-12)


def run_one(dataset: str, output_dir: Path) -> dict:
    frame = load_moleculeace(dataset)
    train = frame[frame.split.eq("train")].reset_index(drop=True)
    test = frame[frame.split.eq("test")].reset_index(drop=True)
    y = train.target.to_numpy(dtype=float)
    pairs = _mmp_pairs(train.smiles.tolist())
    generators = _generators()
    train_kernel = np.zeros((len(train), len(train)), dtype=np.float32)
    test_kernel = np.zeros((len(test), len(train)), dtype=np.float32)
    for _name, generator in generators:
        x_train = _bits(train.smiles.to_numpy(), generator)
        x_test = _bits(test.smiles.to_numpy(), generator)
        weight = _action_weights(x_train, y, pairs)
        train_kernel += _weighted_tanimoto(x_train, x_train, weight) / len(generators)
        test_kernel += _weighted_tanimoto(x_test, x_train, weight) / len(generators)
    model = SVR(C=10.0, epsilon=0.1, kernel="precomputed")
    model.fit(train_kernel, y)
    prediction = model.predict(test_kernel)
    metric = regression_metrics(
        test.target.to_numpy(dtype=float), prediction, test.cliff_mol.to_numpy(dtype=bool)
    )
    metric.update({"dataset": dataset, "model": "action_salience_kernel", "seed": 42,
                   "train_mmp_pairs": len(pairs)})
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = test[["dataset", "smiles", "target", "cliff_mol", "split"]].copy()
    predictions["prediction"] = prediction
    predictions.to_csv(output_dir / f"{dataset}.action_salience_kernel.predictions.csv", index=False)
    return metric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/moleculeace_action_salience_kernel_v1"))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    rows = Parallel(n_jobs=args.jobs)(
        delayed(run_one)(dataset, args.output_dir) for dataset in MOLECULEACE_DATASETS
    )
    summary = pd.DataFrame(rows).sort_values("dataset")
    summary.to_csv(args.output_dir / "summary.action_salience_kernel.csv", index=False)
    aggregate = {
        "model": "action_salience_kernel", "datasets": len(summary),
        "views": [name for name, _ in _generators()], "kernel_weights": "equal",
        "salience": "train_only_exact_mmp_closed_form", "train_mmp_pairs": int(summary.train_mmp_pairs.sum()),
        "macro_rmse": float(summary.rmse.mean()),
        "macro_cliff_rmse": float(summary.cliff_rmse.mean()),
        "macro_noncliff_rmse": float(summary.noncliff_rmse.mean()),
        "ensemble": False, "selection": "none",
    }
    (args.output_dir / "aggregate.action_salience_kernel.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(aggregate, sort_keys=True))


if __name__ == "__main__":
    main()
