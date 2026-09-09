from __future__ import annotations

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import Ridge

from .data import Fingerprints, tanimoto_matrix


class PrototypeActionTransportRegressor:
    """Single KerRead-inspired action-prototype transport model."""

    def __init__(
        self,
        n_bits: int = 1024,
        n_prototypes: int = 32,
        train_neighbors: int = 24,
        min_similarity: float = 0.35,
        alpha: float = 1.0,
    ):
        self.n_bits = int(n_bits)
        self.n_prototypes = int(n_prototypes)
        self.train_neighbors = int(train_neighbors)
        self.min_similarity = float(min_similarity)
        self.alpha = float(alpha)
        self.fp = Fingerprints(radius=2, n_bits=self.n_bits)

    def _action_vector(self, src: int, dst_counts: np.ndarray) -> np.ndarray:
        signed = dst_counts - self.counts_[src]
        absolute = np.abs(signed)
        return np.concatenate([signed, absolute]).astype(np.float32, copy=False)

    @staticmethod
    def _normalize_rows(x: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(norm, 1e-6)

    def fit(self, smiles: list[str], y: np.ndarray) -> "PrototypeActionTransportRegressor":
        self.smiles_ = list(smiles)
        self.y_ = np.asarray(y, dtype=float)
        self.bits_ = self.fp.bits(np.asarray(smiles))
        self.counts_ = self.fp.counts(np.asarray(smiles))
        sim = tanimoto_matrix(self.bits_, self.bits_)
        np.fill_diagonal(sim, -1.0)
        actions = []
        srcs = []
        dsts = []
        sims = []
        for src in range(len(self.y_)):
            order = np.argsort(sim[src])[::-1][: self.train_neighbors]
            for dst in order:
                s = float(sim[src, dst])
                if s < self.min_similarity:
                    continue
                action = self._action_vector(src, self.counts_[dst])
                if np.linalg.norm(action) <= 0:
                    continue
                actions.append(action)
                srcs.append(src)
                dsts.append(dst)
                sims.append(s)
        if len(actions) < 4:
            self.fallback_ = float(np.mean(self.y_))
            self.model_ = None
            return self
        action_matrix = self._normalize_rows(np.vstack(actions))
        k = min(self.n_prototypes, len(action_matrix))
        self.kmeans_ = MiniBatchKMeans(
            n_clusters=k,
            random_state=42,
            batch_size=2048,
            n_init=1,
            max_iter=100,
        )
        self.kmeans_.fit(action_matrix)
        self.centers_ = self._normalize_rows(self.kmeans_.cluster_centers_.astype(np.float32))
        x = self._transport_features(
            np.asarray(srcs, dtype=int),
            np.asarray(dsts, dtype=int),
            np.asarray(sims, dtype=np.float32),
            self.counts_[np.asarray(dsts, dtype=int)],
        )
        target = self.y_[np.asarray(dsts, dtype=int)]
        self.model_ = Ridge(alpha=self.alpha, random_state=42)
        self.model_.fit(x, target)
        self.fallback_ = float(np.mean(self.y_))
        return self

    def _kernel_response(self, action_matrix: np.ndarray) -> np.ndarray:
        action_matrix = self._normalize_rows(action_matrix)
        cosine = action_matrix @ self.centers_.T
        return np.exp(8.0 * (cosine - 1.0)).astype(np.float32, copy=False)

    def _transport_features(
        self,
        srcs: np.ndarray,
        dsts: np.ndarray,
        sims: np.ndarray,
        dst_counts: np.ndarray,
    ) -> np.ndarray:
        actions = np.vstack([self._action_vector(int(src), count) for src, count in zip(srcs, dst_counts, strict=True)])
        response = self._kernel_response(actions)
        signed_mass = actions[:, : self.n_bits].sum(axis=1, keepdims=True)
        action_mass = actions[:, self.n_bits :].sum(axis=1, keepdims=True)
        scalar = np.c_[
            self.y_[srcs],
            sims,
            signed_mass,
            action_mass,
            np.abs(self.y_[srcs] - self.fallback_) if hasattr(self, "fallback_") else np.zeros(len(srcs)),
        ]
        return np.concatenate([scalar.astype(np.float32), response], axis=1)

    def predict(self, smiles: list[str]) -> np.ndarray:
        if self.model_ is None:
            return np.full(len(smiles), self.fallback_, dtype=float)
        query_bits = self.fp.bits(np.asarray(smiles))
        query_counts = self.fp.counts(np.asarray(smiles))
        sim = tanimoto_matrix(query_bits, self.bits_)
        anchor = np.argmax(sim, axis=1)
        features = self._transport_features(
            anchor.astype(int),
            anchor.astype(int),
            sim[np.arange(len(smiles)), anchor].astype(np.float32),
            query_counts,
        )
        return np.asarray(self.model_.predict(features), dtype=float)
