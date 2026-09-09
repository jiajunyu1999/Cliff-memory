from __future__ import annotations

"""Run cross-collection ECFP baseline families on all ACNet target tasks."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from run_acnet_action import (
    ACNET_ROOT,
    SparsePairFeaturizer,
    _labels,
    _metrics,
    official_random_split,
)


MODELS = (
    "linear_ecfp", "svm_ecfp", "hist_gradient_boosting_ecfp", "mlp_ecfp",
    "extratrees_ecfp", "random_forest_ecfp",
)


def run_target(
    target: str, raw_items: list[dict], models: tuple[str, ...],
    feature_modes: tuple[str, ...],
) -> list[dict]:
    featurizer = SparsePairFeaturizer(n_bits=1024)
    items = []
    for item in raw_items:
        try:
            featurizer._mol(item["SMILES1"])
            featurizer._mol(item["SMILES2"])
            items.append(item)
        except ValueError:
            pass
    split = official_random_split(items, seed=8)
    y = _labels(items)
    y_train, y_valid, y_test = y[split.train], y[split.valid], y[split.test]
    rows = []
    for feature_mode in feature_modes:
        feature = featurizer.transform(items, feature_mode)
        x_train = feature[split.train]
        x_valid = feature[split.valid]
        x_test = feature[split.test]
        for name in models:
            if name == "linear_ecfp":
                scaler = StandardScaler(with_mean=False)
                train_design = scaler.fit_transform(x_train)
                valid_design = scaler.transform(x_valid)
                test_design = scaler.transform(x_test)
                model = SGDClassifier(
                    loss="log_loss",
                    alpha=1e-4,
                    max_iter=50,
                    tol=1e-4,
                    class_weight="balanced",
                    average=True,
                    random_state=42,
                )
                model.fit(train_design, y_train)
                valid_score = model.predict_proba(valid_design)[:, 1]
                score = model.predict_proba(test_design)[:, 1]
            elif name in {"extratrees_ecfp", "random_forest_ecfp"}:
                cls = ExtraTreesClassifier if name == "extratrees_ecfp" else RandomForestClassifier
                model = cls(
                    n_estimators=192,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=1,
                )
                model.fit(x_train, y_train)
                valid_score = model.predict_proba(x_valid)[:, 1]
                score = model.predict_proba(x_test)[:, 1]
            else:
                train_dense = x_train.toarray()
                valid_dense = x_valid.toarray()
                test_dense = x_test.toarray()
                if name == "svm_ecfp":
                    model = SVC(C=10.0, kernel="rbf", gamma="scale", class_weight="balanced")
                    model.fit(train_dense, y_train)
                    valid_score = model.decision_function(valid_dense)
                    score = model.decision_function(test_dense)
                elif name == "hist_gradient_boosting_ecfp":
                    model = HistGradientBoostingClassifier(
                        max_iter=100, learning_rate=0.06, max_leaf_nodes=31,
                        l2_regularization=1e-3, random_state=42,
                    )
                    weights = np.where(
                        y_train == 1, len(y_train) / max(2 * y_train.sum(), 1),
                        len(y_train) / max(2 * (len(y_train) - y_train.sum()), 1),
                    )
                    model.fit(train_dense, y_train, sample_weight=weights)
                    valid_score = model.predict_proba(valid_dense)[:, 1]
                    score = model.predict_proba(test_dense)[:, 1]
                elif name == "mlp_ecfp":
                    model = MLPClassifier(
                        hidden_layer_sizes=(256, 64), activation="relu", alpha=1e-4,
                        learning_rate_init=1e-3, max_iter=120, early_stopping=True,
                        validation_fraction=0.1, n_iter_no_change=12, random_state=42,
                    )
                    model.fit(train_dense, y_train)
                    valid_score = model.predict_proba(valid_dense)[:, 1]
                    score = model.predict_proba(test_dense)[:, 1]
                else:
                    raise ValueError(name)
            valid_metrics = {
                f"valid_{key}": value
                for key, value in _metrics(y_valid, valid_score).items()
            }
            rows.append({
                "collection": "ACNet",
                "target": target,
                "model": name,
                "feature_mode": feature_mode,
                "n": len(items),
                "n_train": len(split.train),
                "n_valid": len(split.valid),
                "n_test": len(split.test),
                "test_pos": int(y_test.sum()),
                **valid_metrics,
                **_metrics(y_test, score),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument(
        "--feature-modes", nargs="+", choices=("endpoint", "endpoint_action"),
        default=["endpoint", "endpoint_action"],
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/acnet_cross_baselines/summary.csv"),
    )
    parser.add_argument(
        "--targets", nargs="+",
        help="Optional ACNet target subset. Uses the identical official-split code path for smoke tests and full runs.",
    )
    args = parser.parse_args()
    data = json.loads((ACNET_ROOT / "MMP_AC.json").read_text())
    if args.targets:
        missing = sorted(set(args.targets).difference(data))
        if missing:
            raise ValueError(f"unknown ACNet target(s): {missing}")
        data = {target: data[target] for target in args.targets}
    nested = Parallel(n_jobs=args.jobs, verbose=10)(
        delayed(run_target)(target, items, tuple(args.models), tuple(args.feature_modes))
        for target, items in data.items()
    )
    result = pd.DataFrame([row for part in nested for row in part])
    result = result.sort_values(["feature_mode", "model", "target"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    payload = {
        "protocol": (
            "ACNet official random split; endpoint pair ECFP; fixed global "
            "hyperparameters; no target-specific tuning"
        ),
        "targets": int(result.target.nunique()),
        "macro": (
            result.groupby(["feature_mode", "model"])[
                ["valid_auc", "valid_ap", "valid_acc", "auc", "ap", "acc"]
            ]
            .mean()
            .reset_index()
            .to_dict(orient="records")
        ),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
