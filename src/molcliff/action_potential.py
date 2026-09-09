from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from .data import Fingerprints, tanimoto_matrix


class ActionPotentialRegressor:
    """Single potential model trained with absolute and action-delta equations."""

    def __init__(
        self,
        n_bits: int = 1024,
        train_neighbors: int = 16,
        min_similarity: float = 0.35,
        alpha: float = 1.0,
    ):
        self.n_bits = int(n_bits)
        self.train_neighbors = int(train_neighbors)
        self.min_similarity = float(min_similarity)
        self.alpha = float(alpha)
        self.fp = Fingerprints(radius=2, n_bits=self.n_bits)

    def _features(self, smiles: list[str] | np.ndarray) -> np.ndarray:
        bits = self.fp.bits(np.asarray(smiles))
        counts = self.fp.counts(np.asarray(smiles))
        return np.concatenate([bits, counts], axis=1).astype(np.float32, copy=False)

    def fit(self, smiles: list[str], y: np.ndarray) -> "ActionPotentialRegressor":
        self.smiles_ = list(smiles)
        y = np.asarray(y, dtype=float)
        x = self._features(self.smiles_)
        bits = x[:, : self.n_bits]
        sim = tanimoto_matrix(bits, bits)
        np.fill_diagonal(sim, -1.0)
        rows = [x]
        targets = [y]
        weights = [np.ones(len(y), dtype=np.float32)]
        pair_rows = []
        pair_targets = []
        pair_weights = []
        for src in range(len(y)):
            order = np.argsort(sim[src])[::-1][: self.train_neighbors]
            for dst in order:
                s = float(sim[src, dst])
                if s < self.min_similarity:
                    continue
                pair_rows.append(x[dst] - x[src])
                pair_targets.append(float(y[dst] - y[src]))
                pair_weights.append(s)
        if pair_rows:
            pair_x = np.vstack(pair_rows).astype(np.float32, copy=False)
            pair_y = np.asarray(pair_targets, dtype=float)
            pair_w = np.asarray(pair_weights, dtype=np.float32)
            # Equalize absolute and action equation mass without a tuned lambda.
            pair_w = pair_w * (len(y) / max(float(pair_w.sum()), 1.0))
            rows.append(pair_x)
            targets.append(pair_y)
            weights.append(pair_w)
        train_x = np.vstack(rows)
        train_y = np.concatenate(targets)
        train_w = np.concatenate(weights)
        self.model_ = Ridge(alpha=self.alpha, random_state=42)
        self.model_.fit(train_x, train_y, sample_weight=train_w)
        return self

    def predict(self, smiles: list[str]) -> np.ndarray:
        return np.asarray(self.model_.predict(self._features(smiles)), dtype=float)
