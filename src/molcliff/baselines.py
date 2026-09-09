from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR


def _official_config(dataset: str, stem: str) -> dict:
    config = (
        Path(__file__).resolve().parents[2]
        / "third_party"
        / "MoleculeACE"
        / "MoleculeACE"
        / "Data"
        / "configures"
        / "benchmark"
        / dataset
        / f"{stem}.yml"
    )
    return yaml.safe_load(config.read_text())


def make_regressor(name: str, seed: int = 42, dataset: str | None = None):
    """Fixed lightweight baselines; no dataset-specific search is performed."""

    name = name.lower()
    if name == "svm":
        # A deliberately uniform baseline.  MoleculeACE's published SVM uses
        # C=100 on two targets and C=10000 on one target; all other parameters
        # and the remaining 27 targets use these values.
        return SVR(C=10.0, epsilon=0.1, gamma=0.01, kernel="rbf")
    if name == "svm_official":
        if dataset is None:
            raise ValueError("svm_official requires a dataset name")
        return SVR(**_official_config(dataset, "SVM_ECFP"))
    if name == "rf_official":
        if dataset is None:
            raise ValueError("rf_official requires a dataset name")
        return RandomForestRegressor(
            **_official_config(dataset, "RF_ECFP"), random_state=seed, n_jobs=-1
        )
    if name == "gbm_official":
        if dataset is None:
            raise ValueError("gbm_official requires a dataset name")
        return GradientBoostingRegressor(
            **_official_config(dataset, "GBM_ECFP"), random_state=seed
        )
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=500,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(f"Unknown baseline: {name}")


def fit_predict(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    dataset: str | None = None,
) -> np.ndarray:
    model = make_regressor(name, dataset=dataset)
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_test), dtype=float)
