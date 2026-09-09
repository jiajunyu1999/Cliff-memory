from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .data import Fingerprints, tanimoto_matrix


class _ActionNet(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
        )
        self.potential = nn.Linear(256, 1)
        self.delta = nn.Sequential(
            nn.Linear(256 * 4 + 3, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.potential(self.encode(x)).squeeze(-1)

    def pair_delta(self, z_src: torch.Tensor, z_dst: torch.Tensor, sim: torch.Tensor) -> torch.Tensor:
        pair = torch.cat(
            [
                z_src,
                z_dst,
                z_dst - z_src,
                torch.abs(z_dst - z_src),
                sim.view(-1, 1),
                torch.norm(z_dst - z_src, dim=1, keepdim=True),
                (z_src * z_dst).sum(dim=1, keepdim=True) / z_src.shape[1],
            ],
            dim=1,
        )
        return self.delta(pair).squeeze(-1)


class NeuralActionDeltaRegressor:
    """One neural potential with an auxiliary local action-delta objective."""

    def __init__(
        self,
        n_bits: int = 1024,
        train_neighbors: int = 16,
        min_similarity: float = 0.35,
        epochs: int = 160,
        batch_size: int = 256,
        pair_batch_size: int = 512,
        device: str | None = None,
    ):
        self.n_bits = int(n_bits)
        self.train_neighbors = int(train_neighbors)
        self.min_similarity = float(min_similarity)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.pair_batch_size = int(pair_batch_size)
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fp = Fingerprints(radius=2, n_bits=self.n_bits)

    def _features(self, smiles: list[str] | np.ndarray) -> np.ndarray:
        bits = self.fp.bits(np.asarray(smiles))
        counts = self.fp.counts(np.asarray(smiles)) / 4.0
        return np.concatenate([bits, counts], axis=1).astype(np.float32, copy=False)

    def _pair_index(self, bits: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sim = tanimoto_matrix(bits, bits)
        np.fill_diagonal(sim, -1.0)
        srcs = []
        dsts = []
        sims = []
        deltas = []
        for src in range(len(y)):
            order = np.argsort(sim[src])[::-1][: self.train_neighbors]
            for dst in order:
                s = float(sim[src, dst])
                if s < self.min_similarity:
                    continue
                srcs.append(src)
                dsts.append(dst)
                sims.append(s)
                deltas.append(float(y[dst] - y[src]))
        return (
            np.asarray(srcs, dtype=np.int64),
            np.asarray(dsts, dtype=np.int64),
            np.asarray(sims, dtype=np.float32),
            np.asarray(deltas, dtype=np.float32),
        )

    def fit(self, smiles: list[str], y: np.ndarray) -> "NeuralActionDeltaRegressor":
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        x_np = self._features(smiles)
        bits = x_np[:, : self.n_bits]
        y_np = np.asarray(y, dtype=np.float32)
        self.y_mean_ = float(y_np.mean())
        self.y_std_ = float(y_np.std() + 1e-6)
        y_scaled = (y_np - self.y_mean_) / self.y_std_
        srcs, dsts, sims, deltas = self._pair_index(bits, y_scaled)
        device = torch.device(self.device)
        self.model_ = _ActionNet(x_np.shape[1]).to(device)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=1e-3, weight_decay=1e-4)
        x = torch.tensor(x_np, dtype=torch.float32, device=device)
        target = torch.tensor(y_scaled, dtype=torch.float32, device=device)
        if len(srcs):
            src_t = torch.tensor(srcs, dtype=torch.long, device=device)
            dst_t = torch.tensor(dsts, dtype=torch.long, device=device)
            sim_t = torch.tensor(sims, dtype=torch.float32, device=device)
            delta_t = torch.tensor(deltas, dtype=torch.float32, device=device)
        else:
            src_t = dst_t = torch.empty(0, dtype=torch.long, device=device)
            sim_t = delta_t = torch.empty(0, dtype=torch.float32, device=device)

        n = len(x)
        p = len(src_t)
        for _epoch in range(self.epochs):
            perm = torch.randperm(n, device=device)
            pair_perm = torch.randperm(p, device=device) if p else torch.empty(0, dtype=torch.long, device=device)
            pair_cursor = 0
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                pred = self.model_(x[idx])
                loss = F.smooth_l1_loss(pred, target[idx])
                if p:
                    if pair_cursor >= p:
                        pair_cursor = 0
                    pidx = pair_perm[pair_cursor : pair_cursor + self.pair_batch_size]
                    pair_cursor += self.pair_batch_size
                    z_src = self.model_.encode(x[src_t[pidx]])
                    z_dst = self.model_.encode(x[dst_t[pidx]])
                    delta_pred = self.model_.pair_delta(z_src, z_dst, sim_t[pidx])
                    potential_delta = (
                        self.model_.potential(z_dst).squeeze(-1)
                        - self.model_.potential(z_src).squeeze(-1)
                    )
                    action_loss = F.smooth_l1_loss(delta_pred, delta_t[pidx])
                    consistency_loss = F.smooth_l1_loss(potential_delta, delta_t[pidx])
                    loss = loss + action_loss + consistency_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 5.0)
                opt.step()
        self.model_.eval()
        return self

    def predict(self, smiles: list[str]) -> np.ndarray:
        device = torch.device(self.device)
        x = torch.tensor(self._features(smiles), dtype=torch.float32, device=device)
        out = []
        with torch.no_grad():
            for start in range(0, len(x), self.batch_size):
                pred = self.model_(x[start : start + self.batch_size])
                out.append(pred.detach().cpu().numpy())
        scaled = np.concatenate(out)
        return scaled * self.y_std_ + self.y_mean_


class NeuralActionPotentialRegressor(NeuralActionDeltaRegressor):
    """Single neural potential trained only by absolute and potential-delta loss."""

    def fit(self, smiles: list[str], y: np.ndarray) -> "NeuralActionPotentialRegressor":
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        x_np = self._features(smiles)
        bits = x_np[:, : self.n_bits]
        y_np = np.asarray(y, dtype=np.float32)
        self.y_mean_ = float(y_np.mean())
        self.y_std_ = float(y_np.std() + 1e-6)
        y_scaled = (y_np - self.y_mean_) / self.y_std_
        srcs, dsts, sims, deltas = self._pair_index(bits, y_scaled)
        device = torch.device(self.device)
        self.model_ = _ActionNet(x_np.shape[1]).to(device)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=8e-4, weight_decay=1e-4)
        x = torch.tensor(x_np, dtype=torch.float32, device=device)
        target = torch.tensor(y_scaled, dtype=torch.float32, device=device)
        if len(srcs):
            src_t = torch.tensor(srcs, dtype=torch.long, device=device)
            dst_t = torch.tensor(dsts, dtype=torch.long, device=device)
            delta_t = torch.tensor(deltas, dtype=torch.float32, device=device)
        else:
            src_t = dst_t = torch.empty(0, dtype=torch.long, device=device)
            delta_t = torch.empty(0, dtype=torch.float32, device=device)

        n = len(x)
        p = len(src_t)
        for _epoch in range(self.epochs):
            perm = torch.randperm(n, device=device)
            pair_perm = torch.randperm(p, device=device) if p else torch.empty(0, dtype=torch.long, device=device)
            pair_cursor = 0
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                pred = self.model_(x[idx])
                loss = F.smooth_l1_loss(pred, target[idx])
                if p:
                    if pair_cursor >= p:
                        pair_cursor = 0
                    pidx = pair_perm[pair_cursor : pair_cursor + self.pair_batch_size]
                    pair_cursor += self.pair_batch_size
                    f_src = self.model_(x[src_t[pidx]])
                    f_dst = self.model_(x[dst_t[pidx]])
                    loss = loss + F.smooth_l1_loss(f_dst - f_src, delta_t[pidx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 5.0)
                opt.step()
        self.model_.eval()
        return self
