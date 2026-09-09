from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVR

from .data import Fingerprints, tanimoto_matrix


class ActionKernelDeltaRegressor:
    """Predict potency by transporting from anchors through similar actions.

    The model builds a directed local SAR graph in the training set.  Each edge
    carries a signed count-ECFP edit and the observed potency delta.  At test
    time, the query is reconstructed from its nearest measured anchors through
    the same signed edit representation.
    """

    def __init__(
        self,
        n_bits: int = 1024,
        train_neighbors: int = 16,
        query_neighbors: int = 16,
        min_similarity: float = 0.35,
        alpha: float = 1.0,
    ):
        self.n_bits = int(n_bits)
        self.train_neighbors = int(train_neighbors)
        self.query_neighbors = int(query_neighbors)
        self.min_similarity = float(min_similarity)
        self.alpha = float(alpha)
        self.fp = Fingerprints(radius=2, n_bits=self.n_bits)

    def fit(self, smiles: list[str], y: np.ndarray) -> "ActionKernelDeltaRegressor":
        self.smiles_ = list(smiles)
        self.y_ = np.asarray(y, dtype=float)
        self.bits_ = self.fp.bits(np.asarray(smiles))
        self.counts_ = self.fp.counts(np.asarray(smiles))
        self.base_ = SVR(C=10.0, epsilon=0.1, gamma=0.01, kernel="rbf")
        self.base_.fit(self.bits_, self.y_)
        sim = tanimoto_matrix(self.bits_, self.bits_)
        np.fill_diagonal(sim, -1.0)
        features = []
        deltas = []
        weights = []
        for src in range(len(self.y_)):
            order = np.argsort(sim[src])[::-1][: self.train_neighbors]
            for dst in order:
                s = float(sim[src, dst])
                if s < self.min_similarity:
                    continue
                features.append(self._edge_features(src, dst))
                deltas.append(float(self.y_[dst] - self.y_[src]))
                weights.append(max(0.05, s))
        if len(features) < 8:
            self.delta_model_ = None
            self.delta_residual_scale_ = np.inf
            return self
        x = np.vstack(features).astype(np.float32, copy=False)
        z = np.asarray(deltas, dtype=float)
        w = np.asarray(weights, dtype=float)
        self.delta_model_ = Ridge(alpha=self.alpha, random_state=42)
        self.delta_model_.fit(x, z, sample_weight=w)
        residual = z - self.delta_model_.predict(x)
        self.delta_residual_scale_ = float(np.sqrt(np.average(np.square(residual), weights=w)))
        return self

    def _edge_features_from_counts(
        self,
        source_counts: np.ndarray,
        destination_counts: np.ndarray,
        source_y: float,
        similarity: float,
    ) -> np.ndarray:
        signed = destination_counts - source_counts
        absolute = np.abs(signed)
        scalar = np.asarray(
            [
                source_y,
                similarity,
                float(signed.sum()),
                float(absolute.sum()),
                float(np.count_nonzero(absolute)),
            ],
            dtype=np.float32,
        )
        return np.concatenate([signed, absolute, scalar]).astype(np.float32, copy=False)

    def _edge_features(self, source: int, destination: int) -> np.ndarray:
        return self._edge_features_from_counts(
            self.counts_[source],
            self.counts_[destination],
            float(self.y_[source]),
            1.0,
        )

    def predict_components(self, smiles: list[str]) -> dict[str, np.ndarray]:
        query_bits = self.fp.bits(np.asarray(smiles))
        query_counts = self.fp.counts(np.asarray(smiles))
        base = np.asarray(self.base_.predict(query_bits), dtype=float)
        action = base.copy()
        confidence = np.zeros(len(smiles), dtype=float)
        if self.delta_model_ is None:
            return {"base": base, "action": action, "confidence": confidence, "blended": base}
        sim = tanimoto_matrix(query_bits, self.bits_)
        for qi in range(len(smiles)):
            order = np.argsort(sim[qi])[::-1][: self.query_neighbors]
            votes = []
            for anchor in order:
                s = float(sim[qi, anchor])
                if s < self.min_similarity:
                    continue
                feature = self._edge_features_from_counts(
                    self.counts_[anchor],
                    query_counts[qi],
                    float(self.y_[anchor]),
                    s,
                )
                delta = float(self.delta_model_.predict(feature.reshape(1, -1))[0])
                votes.append((float(self.y_[anchor] + delta), s))
            if not votes:
                continue
            values = np.asarray([v[0] for v in votes], dtype=float)
            raw_weights = np.asarray([v[1] for v in votes], dtype=float)
            weights = np.exp(12.0 * (raw_weights - raw_weights.max()))
            estimate = float(np.sum(weights * values) / np.sum(weights))
            disagreement = float(np.sum(weights * np.abs(values - estimate)) / np.sum(weights))
            reliability = raw_weights.max() / (1.0 + disagreement + self.delta_residual_scale_)
            confidence[qi] = min(0.8, max(0.0, reliability))
            action[qi] = estimate
        blended = (1.0 - confidence) * base + confidence * action
        return {"base": base, "action": action, "confidence": confidence, "blended": blended}

    def predict(self, smiles: list[str]) -> np.ndarray:
        return self.predict_components(smiles)["blended"]


class ActionRetrievalDeltaRegressor(ActionKernelDeltaRegressor):
    """RAG-style action transport using retrieved measured deltas."""

    def __init__(
        self,
        n_bits: int = 1024,
        train_neighbors: int = 16,
        query_neighbors: int = 12,
        retrieved_actions: int = 24,
        min_similarity: float = 0.35,
    ):
        super().__init__(
            n_bits=n_bits,
            train_neighbors=train_neighbors,
            query_neighbors=query_neighbors,
            min_similarity=min_similarity,
            alpha=1.0,
        )
        self.retrieved_actions = int(retrieved_actions)

    def fit(self, smiles: list[str], y: np.ndarray) -> "ActionRetrievalDeltaRegressor":
        self.smiles_ = list(smiles)
        self.y_ = np.asarray(y, dtype=float)
        self.bits_ = self.fp.bits(np.asarray(smiles))
        self.counts_ = self.fp.counts(np.asarray(smiles))
        self.base_ = SVR(C=10.0, epsilon=0.1, gamma=0.01, kernel="rbf")
        self.base_.fit(self.bits_, self.y_)
        sim = tanimoto_matrix(self.bits_, self.bits_)
        np.fill_diagonal(sim, -1.0)
        actions = []
        deltas = []
        similarities = []
        for src in range(len(self.y_)):
            order = np.argsort(sim[src])[::-1][: self.train_neighbors]
            for dst in order:
                s = float(sim[src, dst])
                if s < self.min_similarity:
                    continue
                signed = self.counts_[dst] - self.counts_[src]
                absolute = np.abs(signed)
                vector = np.concatenate([signed, absolute]).astype(np.float32, copy=False)
                norm = float(np.linalg.norm(vector))
                if norm <= 0:
                    continue
                actions.append(vector / norm)
                deltas.append(float(self.y_[dst] - self.y_[src]))
                similarities.append(s)
        if len(actions) < self.retrieved_actions:
            self.retriever_ = None
            self.action_bank_ = np.empty((0, 2 * self.n_bits), dtype=np.float32)
            self.delta_bank_ = np.empty(0, dtype=float)
            self.similarity_bank_ = np.empty(0, dtype=float)
            return self
        self.action_bank_ = np.vstack(actions).astype(np.float32, copy=False)
        self.delta_bank_ = np.asarray(deltas, dtype=float)
        self.similarity_bank_ = np.asarray(similarities, dtype=float)
        self.retriever_ = NearestNeighbors(
            n_neighbors=min(self.retrieved_actions, len(self.action_bank_)),
            metric="cosine",
            algorithm="brute",
        )
        self.retriever_.fit(self.action_bank_)
        return self

    def _normalized_action(self, source_counts: np.ndarray, destination_counts: np.ndarray) -> np.ndarray | None:
        signed = destination_counts - source_counts
        absolute = np.abs(signed)
        vector = np.concatenate([signed, absolute]).astype(np.float32, copy=False)
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            return None
        return vector / norm

    def predict_components(self, smiles: list[str]) -> dict[str, np.ndarray]:
        query_bits = self.fp.bits(np.asarray(smiles))
        query_counts = self.fp.counts(np.asarray(smiles))
        base = np.asarray(self.base_.predict(query_bits), dtype=float)
        action = base.copy()
        confidence = np.zeros(len(smiles), dtype=float)
        if self.retriever_ is None:
            return {"base": base, "action": action, "confidence": confidence, "blended": base}
        sim = tanimoto_matrix(query_bits, self.bits_)
        for qi in range(len(smiles)):
            order = np.argsort(sim[qi])[::-1][: self.query_neighbors]
            anchor_votes = []
            for anchor in order:
                anchor_similarity = float(sim[qi, anchor])
                if anchor_similarity < self.min_similarity:
                    continue
                vector = self._normalized_action(self.counts_[anchor], query_counts[qi])
                if vector is None:
                    continue
                distances, indices = self.retriever_.kneighbors(vector.reshape(1, -1))
                action_similarity = 1.0 - distances[0]
                deltas = self.delta_bank_[indices[0]]
                weights = np.exp(10.0 * (action_similarity - action_similarity.max()))
                delta = float(np.sum(weights * deltas) / np.sum(weights))
                spread = float(np.sum(weights * np.abs(deltas - delta)) / np.sum(weights))
                evidence = float(np.mean(np.maximum(action_similarity, 0.0)))
                reliability = anchor_similarity * evidence / (1.0 + spread)
                anchor_votes.append((float(self.y_[anchor] + delta), reliability))
            if not anchor_votes:
                continue
            values = np.asarray([v[0] for v in anchor_votes], dtype=float)
            raw_weights = np.asarray([max(v[1], 1e-6) for v in anchor_votes], dtype=float)
            estimate = float(np.sum(raw_weights * values) / np.sum(raw_weights))
            disagreement = float(np.sum(raw_weights * np.abs(values - estimate)) / np.sum(raw_weights))
            confidence[qi] = min(0.85, float(raw_weights.mean() / (1.0 + disagreement)))
            action[qi] = estimate
        blended = (1.0 - confidence) * base + confidence * action
        return {"base": base, "action": action, "confidence": confidence, "blended": blended}
