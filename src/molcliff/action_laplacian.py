from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.neighbors import NearestNeighbors

from .data import Fingerprints


class ActionGatedLaplacianRegressor:
    """Single transductive graph potential with action-gated smoothing."""

    def __init__(self, n_bits: int = 1024):
        self.n_bits = int(n_bits)
        self.fp = Fingerprints(radius=2, n_bits=self.n_bits)

    def fit(self, smiles: list[str], y: np.ndarray) -> "ActionGatedLaplacianRegressor":
        self.train_smiles_ = list(smiles)
        self.y_ = np.asarray(y, dtype=float)
        return self

    @staticmethod
    def _cosine_knn(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(x)
        k = max(4, min(32, int(np.sqrt(n))))
        nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric="cosine", algorithm="brute")
        nn.fit(x)
        dist, ind = nn.kneighbors(x)
        return dist[:, 1:], ind[:, 1:]

    def predict(self, smiles: list[str]) -> np.ndarray:
        all_smiles = self.train_smiles_ + list(smiles)
        bits = self.fp.bits(np.asarray(all_smiles))
        counts = self.fp.counts(np.asarray(all_smiles))
        dist, ind = self._cosine_knn(bits)
        rows = []
        cols = []
        data = []
        for i in range(len(all_smiles)):
            source_count = counts[i]
            for d, j in zip(dist[i], ind[i], strict=True):
                sim = max(0.0, 1.0 - float(d))
                if sim <= 0:
                    continue
                action_mass = float(np.abs(counts[int(j)] - source_count).sum())
                signed_mass = abs(float((counts[int(j)] - source_count).sum()))
                # Big local edits are likely cliff boundaries; do not over-smooth them.
                gate = 1.0 / (1.0 + action_mass + 0.5 * signed_mass)
                weight = sim * gate
                rows.extend([i, int(j)])
                cols.extend([int(j), i])
                data.extend([weight, weight])
        n = len(all_smiles)
        w = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        w = w.maximum(w.T)
        degree = np.asarray(w.sum(axis=1)).reshape(-1)
        mean_degree = float(np.mean(degree[degree > 0])) if np.any(degree > 0) else 1.0
        laplacian = sparse.diags(degree) - w
        laplacian = laplacian / max(mean_degree, 1e-6)
        train_mask = np.zeros(n, dtype=float)
        train_mask[: len(self.y_)] = 1.0
        anchor = sparse.diags(train_mask)
        # One fixed equation: train labels are hard anchors; graph supplies unlabeled potential.
        system = anchor + laplacian + sparse.eye(n, format="csr") * 1e-6
        rhs = np.zeros(n, dtype=float)
        rhs[: len(self.y_)] = self.y_
        solution = spsolve(system.tocsr(), rhs)
        return np.asarray(solution[len(self.y_) :], dtype=float)
