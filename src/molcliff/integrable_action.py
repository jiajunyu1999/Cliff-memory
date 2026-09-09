from __future__ import annotations

from collections import Counter

import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

from .data import Fingerprints
from .mmp_action import single_cuts


class IntegrableActionRegressor:
    """A regularized core + substituent potential model over one-cut MMPs.

    ``y(core, substituent) = intercept + U(core) + V(substituent)``

    The action effect for replacing ``s1`` by ``s2`` on a retained core is
    therefore ``V(s2) - V(s1)``.  It is antisymmetric and every action cycle
    sums to zero by construction, rather than by an auxiliary penalty.
    """

    def __init__(self, ridge_alpha: float = 10.0):
        self.ridge_alpha = float(ridge_alpha)
        self.fp = Fingerprints(radius=2, n_bits=1024)

    def fit(self, smiles: list[str], y: np.ndarray) -> "IntegrableActionRegressor":
        self.y_ = np.asarray(y, dtype=float).reshape(-1)
        bits = self.fp.bits(np.asarray(smiles))
        self.base_ = SVR(C=10.0, epsilon=0.1, gamma=0.01, kernel="rbf")
        self.base_.fit(bits, self.y_)

        cuts_per_molecule = [single_cuts(s) for s in smiles]
        cores = sorted({cut.core for cuts in cuts_per_molecule for cut in cuts})
        substituents = sorted({cut.substituent for cuts in cuts_per_molecule for cut in cuts})
        self.core_to_id_ = {value: i for i, value in enumerate(cores)}
        self.sub_to_id_ = {value: i for i, value in enumerate(substituents)}
        self.core_count_ = Counter(cut.core for cuts in cuts_per_molecule for cut in cuts)
        self.sub_count_ = Counter(cut.substituent for cuts in cuts_per_molecule for cut in cuts)

        rows: list[int] = []
        cols: list[int] = []
        values: list[float] = []
        targets: list[float] = []
        weights: list[float] = []
        offset = len(cores)
        row = 0
        for molecule_index, cuts in enumerate(cuts_per_molecule):
            if not cuts:
                continue
            molecule_weight = 1.0 / len(cuts)
            for cut in cuts:
                rows.extend((row, row))
                cols.extend((self.core_to_id_[cut.core], offset + self.sub_to_id_[cut.substituent]))
                values.extend((1.0, 1.0))
                targets.append(float(self.y_[molecule_index]))
                weights.append(molecule_weight)
                row += 1
        design = sparse.csr_matrix(
            (values, (rows, cols)), shape=(row, len(cores) + len(substituents)), dtype=np.float64
        )
        self.potential_ = Ridge(
            alpha=self.ridge_alpha, fit_intercept=True, solver="lsqr", tol=1e-7
        )
        self.potential_.fit(design, np.asarray(targets), sample_weight=np.asarray(weights))
        self.n_cores_ = len(cores)
        return self

    def predict_components(self, smiles: list[str]) -> dict[str, np.ndarray]:
        bits = self.fp.bits(np.asarray(smiles))
        base = np.asarray(self.base_.predict(bits), dtype=float)
        action = base.copy()
        confidence = np.zeros(len(smiles), dtype=float)
        path_count = np.zeros(len(smiles), dtype=float)
        for index, value in enumerate(smiles):
            candidates = []
            reliabilities = []
            for cut in single_cuts(value):
                core_id = self.core_to_id_.get(cut.core)
                sub_id = self.sub_to_id_.get(cut.substituent)
                if core_id is None or sub_id is None:
                    continue
                estimate = (
                    float(self.potential_.intercept_)
                    + float(self.potential_.coef_[core_id])
                    + float(self.potential_.coef_[self.n_cores_ + sub_id])
                )
                core_n = self.core_count_[cut.core]
                sub_n = self.sub_count_[cut.substituent]
                reliability = min(core_n, sub_n) / (min(core_n, sub_n) + 4.0)
                candidates.append(estimate)
                reliabilities.append(reliability)
            if candidates:
                values = np.asarray(candidates, dtype=float)
                weights = np.asarray(reliabilities, dtype=float)
                action[index] = float(np.sum(values * weights) / np.sum(weights))
                disagreement = float(np.sum(weights * np.abs(values - action[index])) / np.sum(weights))
                confidence[index] = min(0.5, float(weights.mean() / (1.0 + disagreement)))
                path_count[index] = len(candidates)
        blended = base + confidence * (action - base)
        return {
            "base": base,
            "action": action,
            "blended": blended,
            "confidence": confidence,
            "path_count": path_count,
        }

    def predict(self, smiles: list[str]) -> np.ndarray:
        return self.predict_components(smiles)["blended"]

