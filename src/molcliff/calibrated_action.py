from __future__ import annotations

import hashlib

import numpy as np

from .mmp_action import MMPActionMemoryRegressor


def _stable_calibration_mask(smiles: list[str], modulus: int = 5) -> np.ndarray:
    """Select a reproducible calibration fold without a random seed."""

    fold = [
        int(hashlib.sha1(value.encode("utf-8")).hexdigest()[:8], 16) % modulus
        for value in smiles
    ]
    mask = np.asarray(fold, dtype=int) == 0
    # Extremely small or adversarial collections should still fail clearly,
    # rather than silently calibrating on an empty fold.
    if mask.sum() < 8 or (~mask).sum() < 16:
        order = np.argsort(np.asarray(smiles, dtype=object), kind="stable")
        mask = np.zeros(len(smiles), dtype=bool)
        mask[order[::modulus]] = True
    return mask


class CalibratedMMPActionRegressor:
    """MMP action transport with a train-only, closed-form safety shrinkage.

    A stable 20% calibration fold estimates one non-negative scalar for the
    confidence-weighted action correction.  The scalar is the analytic least-
    squares solution, not a hyperparameter grid or a test-set choice.  The MMP
    model is then refit on every training molecule and the frozen scalar is
    applied to its correction.
    """

    def __init__(self, min_other_cores: int = 2):
        self.min_other_cores = int(min_other_cores)

    def fit(self, smiles: list[str], y: np.ndarray) -> "CalibratedMMPActionRegressor":
        smiles = list(smiles)
        values = np.asarray(y, dtype=float).reshape(-1)
        calibration = _stable_calibration_mask(smiles)
        fit = ~calibration

        pilot = MMPActionMemoryRegressor(self.min_other_cores).fit(
            [smiles[i] for i in np.flatnonzero(fit)], values[fit]
        )
        components = pilot.predict_components(
            [smiles[i] for i in np.flatnonzero(calibration)]
        )
        correction = components["blended"] - components["base"]
        residual = values[calibration] - components["base"]
        denominator = float(np.dot(correction, correction))
        raw = float(np.dot(correction, residual) / denominator) if denominator > 1e-12 else 0.0
        self.raw_shrinkage_ = raw
        self.shrinkage_ = float(np.clip(raw, 0.0, 1.0))
        self.calibration_size_ = int(calibration.sum())
        self.model_ = MMPActionMemoryRegressor(self.min_other_cores).fit(smiles, values)
        return self

    def predict_components(self, smiles: list[str]) -> dict[str, np.ndarray]:
        result = self.model_.predict_components(smiles)
        result["uncalibrated"] = result["blended"].copy()
        result["blended"] = result["base"] + self.shrinkage_ * (
            result["uncalibrated"] - result["base"]
        )
        result["shrinkage"] = np.full(len(smiles), self.shrinkage_, dtype=float)
        return result

    def predict(self, smiles: list[str]) -> np.ndarray:
        return self.predict_components(smiles)["blended"]
