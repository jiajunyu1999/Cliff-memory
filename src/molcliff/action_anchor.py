from __future__ import annotations

import os
import random

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from sklearn.svm import SVR
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import Fingerprints, moleculeace_similarity_matrix


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _hash_projection(n_bits: int, width: int) -> np.ndarray:
    """Deterministic signed feature hashing, with no tunable random draw."""

    projection = np.zeros((n_bits, width), dtype=np.float32)
    indices = np.arange(n_bits, dtype=np.uint64)
    buckets = ((indices * np.uint64(2654435761)) % np.uint64(width)).astype(int)
    signs = np.where(
        ((indices * np.uint64(2246822519) + np.uint64(3266489917)) & np.uint64(1)) == 0,
        1.0,
        -1.0,
    ).astype(np.float32)
    projection[np.arange(n_bits), buckets] = signs / np.sqrt(float(width))
    return projection


def _nearest_pairs(similarity: np.ndarray, neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    similarity = similarity.copy()
    np.fill_diagonal(similarity, -1.0)
    k = min(int(neighbors), max(len(similarity) - 1, 1))
    nearest = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
    src = np.repeat(np.arange(len(similarity), dtype=np.int64), k)
    dst = nearest.reshape(-1).astype(np.int64)
    # Preserve one copy of each directed edge.  Direction matters for the
    # finite difference; the reverse is supplied explicitly to the network.
    return src, dst


def _action_features(
    src_bits: np.ndarray,
    dst_bits: np.ndarray,
    src_counts: np.ndarray,
    dst_counts: np.ndarray,
    projection: np.ndarray,
) -> np.ndarray:
    common = src_bits * dst_bits
    removed = src_bits * (1.0 - dst_bits)
    added = dst_bits * (1.0 - src_bits)
    count_delta = np.clip(dst_counts - src_counts, -4.0, 4.0) / 4.0
    inter = common.sum(1)
    union = src_bits.sum(1) + dst_bits.sum(1) - inter
    sim = inter / np.maximum(union, 1.0)
    scalars = np.column_stack(
        [
            sim,
            1.0 - sim,
            removed.sum(1) / src_bits.shape[1],
            added.sum(1) / src_bits.shape[1],
            count_delta.sum(1) / src_bits.shape[1],
        ]
    ).astype(np.float32)
    return np.concatenate(
        [
            common @ projection,
            removed @ projection,
            added @ projection,
            count_delta @ projection,
            scalars,
        ],
        axis=1,
    ).astype(np.float32)


class _DirectedActionNet(nn.Module):
    """A small scorer made antisymmetric by evaluating both edit directions."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, forward: torch.Tensor, reverse: torch.Tensor) -> torch.Tensor:
        return 0.5 * (self.score(forward) - self.score(reverse)).squeeze(-1)


class ActionAnchorRegressor:
    """PC-GAM-style finite-difference transport from measured analog anchors.

    All values are fixed globally rather than searched per dataset.  The model
    is intentionally small (roughly 50k trainable parameters).
    """

    def __init__(
        self,
        *,
        seed: int = 42,
        n_bits: int = 1024,
        projection_dim: int = 64,
        train_neighbors: int = 8,
        predict_neighbors: int = 8,
        epochs: int = 40,
        batch_size: int = 512,
        learning_rate: float = 2e-3,
    ):
        self.seed = seed
        self.n_bits = n_bits
        self.projection_dim = projection_dim
        self.train_neighbors = train_neighbors
        self.predict_neighbors = predict_neighbors
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.fp = Fingerprints(radius=2, n_bits=n_bits)
        self.projection = _hash_projection(n_bits, projection_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, smiles: list[str], y: np.ndarray) -> "ActionAnchorRegressor":
        _set_seed(self.seed)
        self.smiles_ = list(smiles)
        self.y_ = np.asarray(y, dtype=np.float32).reshape(-1)
        self.bits_ = self.fp.bits(np.asarray(smiles))
        self.counts_ = self.fp.counts(np.asarray(smiles))
        self.y_scale_ = float(max(self.y_.std(), 1e-6))

        # The exact MoleculeACE ECFP-SVM is the global fallback.
        self.base_ = SVR(C=10.0, epsilon=0.1, gamma=0.01, kernel="rbf")
        self.base_.fit(self.bits_, self.y_)

        self.similarity_, self.similarity_views_ = moleculeace_similarity_matrix(
            self.smiles_, self.smiles_, self.fp
        )
        src, dst = _nearest_pairs(self.similarity_, self.train_neighbors)
        forward = _action_features(
            self.bits_[src], self.bits_[dst], self.counts_[src], self.counts_[dst], self.projection
        )
        reverse = _action_features(
            self.bits_[dst], self.bits_[src], self.counts_[dst], self.counts_[src], self.projection
        )
        delta = (self.y_[dst] - self.y_[src]) / self.y_scale_
        # Large finite differences are rare and define the task.  A smooth,
        # bounded weight prevents the abundant flat pairs from erasing them.
        weight = 1.0 + np.minimum(np.abs(delta), 2.0)

        tensors = TensorDataset(
            torch.from_numpy(forward),
            torch.from_numpy(reverse),
            torch.from_numpy(delta.astype(np.float32)),
            torch.from_numpy(weight.astype(np.float32)),
        )
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            tensors,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        self.net_ = _DirectedActionNet(forward.shape[1]).to(self.device)
        optimizer = torch.optim.AdamW(
            self.net_.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        self.net_.train()
        for _epoch in range(self.epochs):
            for fwd, rev, target, sample_weight in loader:
                fwd = fwd.to(self.device)
                rev = rev.to(self.device)
                target = target.to(self.device)
                sample_weight = sample_weight.to(self.device)
                prediction = self.net_(fwd, rev)
                error = nn.functional.smooth_l1_loss(
                    prediction, target, beta=0.5, reduction="none"
                )
                # Direct sign supervision matters for crossing a cliff and is
                # zero-cost for nearly flat pairs.
                sign_loss = nn.functional.softplus(-prediction * target.sign())
                sign_mask = target.abs().ge(0.5).float()
                loss = (sample_weight * error).mean() + 0.1 * (
                    sign_loss * sign_mask
                ).sum() / sign_mask.sum().clamp_min(1.0)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net_.parameters(), 5.0)
                optimizer.step()
        self.net_.eval()
        return self

    @torch.inference_mode()
    def predict_components(self, smiles: list[str]) -> dict[str, np.ndarray]:
        query_bits = self.fp.bits(np.asarray(smiles))
        query_counts = self.fp.counts(np.asarray(smiles))
        base = np.asarray(self.base_.predict(query_bits), dtype=np.float32)
        similarities, similarity_views = moleculeace_similarity_matrix(
            smiles, self.smiles_, self.fp
        )
        k = min(self.predict_neighbors, len(self.bits_))
        anchors = np.argpartition(-similarities, kth=k - 1, axis=1)[:, :k]

        query_index = np.repeat(np.arange(len(query_bits), dtype=np.int64), k)
        anchor_index = anchors.reshape(-1)
        forward = _action_features(
            self.bits_[anchor_index],
            query_bits[query_index],
            self.counts_[anchor_index],
            query_counts[query_index],
            self.projection,
        )
        reverse = _action_features(
            query_bits[query_index],
            self.bits_[anchor_index],
            query_counts[query_index],
            self.counts_[anchor_index],
            self.projection,
        )
        delta = self.net_(
            torch.from_numpy(forward).to(self.device),
            torch.from_numpy(reverse).to(self.device),
        ).cpu().numpy()
        transported = self.y_[anchor_index] + delta * self.y_scale_
        transported = transported.reshape(len(query_bits), k)
        anchor_sim = np.take_along_axis(similarities, anchors, axis=1)
        # Softmax relative to the best anchor; stable and label-free.
        vote_weight = np.exp(8.0 * (anchor_sim - anchor_sim.max(1, keepdims=True)))
        action_prediction = (transported * vote_weight).sum(1) / vote_weight.sum(1)
        neighbor_prediction = (
            self.y_[anchor_index].reshape(len(query_bits), k) * vote_weight
        ).sum(1) / vote_weight.sum(1)

        # MoleculeACE itself defines a structurally related pair at consensus
        # similarity >= 0.9.  Restricting action transport to that same domain
        # is therefore a protocol-derived gate, not a searched threshold.
        max_similarity = anchor_sim.max(1)
        action_weight = 0.75 * (max_similarity >= 0.90).astype(np.float32)
        blended = (1.0 - action_weight) * base + action_weight * action_prediction
        return {
            "base": base.astype(float),
            "neighbor": neighbor_prediction.astype(float),
            "action": action_prediction.astype(float),
            "blended": blended.astype(float),
            "max_similarity": max_similarity.astype(float),
            "action_weight": action_weight.astype(float),
            "max_ecfp_similarity": similarity_views["ecfp"].max(1).astype(float),
            "max_scaffold_similarity": similarity_views["scaffold"].max(1).astype(float),
            "max_smiles_similarity": similarity_views["smiles"].max(1).astype(float),
        }

    def predict(self, smiles: list[str]) -> np.ndarray:
        return self.predict_components(smiles)["blended"]
