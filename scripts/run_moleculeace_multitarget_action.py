from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from molcliff.data import Fingerprints, MOLECULEACE_DATASETS, load_moleculeace, tanimoto_matrix
from molcliff.metrics import regression_metrics


class TargetActionNet(nn.Module):
    def __init__(self, ligand_dim: int, n_targets: int, target_dim: int = 64):
        super().__init__()
        self.target_embedding = nn.Embedding(n_targets, target_dim)
        self.ligand_encoder = nn.Sequential(
            nn.Linear(ligand_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
        )
        self.potential = nn.Sequential(
            nn.Linear(256 + target_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

    def forward(self, ligand: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        z = self.ligand_encoder(ligand)
        t = self.target_embedding(target)
        return self.potential(torch.cat([z, t], dim=1)).squeeze(-1)


def _features(fp: Fingerprints, smiles: np.ndarray) -> np.ndarray:
    bits = fp.bits(smiles)
    counts = fp.counts(smiles) / 4.0
    return np.concatenate([bits, counts], axis=1).astype(np.float32, copy=False)


def build_pair_edges(
    frame: pd.DataFrame,
    features: np.ndarray,
    target_ids: np.ndarray,
    y_scaled: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    srcs: list[int] = []
    dsts: list[int] = []
    deltas: list[float] = []
    n_bits = features.shape[1] // 2
    bits = features[:, :n_bits]
    for _dataset, group in frame.groupby("dataset", sort=False):
        idx = group.index.to_numpy()
        sim = tanimoto_matrix(bits[idx], bits[idx])
        np.fill_diagonal(sim, -1.0)
        k = max(4, min(24, int(np.sqrt(len(idx)))))
        for local_src, global_src in enumerate(idx):
            order = np.argsort(sim[local_src])[::-1][:k]
            for local_dst in order:
                if float(sim[local_src, local_dst]) < 0.35:
                    continue
                global_dst = int(idx[local_dst])
                srcs.append(int(global_src))
                dsts.append(global_dst)
                deltas.append(float(y_scaled[global_dst] - y_scaled[int(global_src)]))
    src = np.asarray(srcs, dtype=np.int64)
    dst = np.asarray(dsts, dtype=np.int64)
    delta = np.asarray(deltas, dtype=np.float32)
    node_action_scale = np.ones(len(frame), dtype=np.float32)
    if len(delta):
        node_max_delta = np.zeros(len(frame), dtype=np.float32)
        abs_delta = np.abs(delta)
        np.maximum.at(node_max_delta, src, abs_delta)
        np.maximum.at(node_max_delta, dst, abs_delta)
        positive = node_max_delta[node_max_delta > 0]
        if len(positive):
            mean_positive = float(positive.mean() + 1e-6)
            node_action_scale = np.maximum(node_max_delta, mean_positive) / mean_positive
    return src, dst, delta, node_action_scale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/moleculeace_multitarget_action_v1"))
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--pair-batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--loss-mode",
        choices=["uniform", "action_balanced"],
        default="uniform",
        help="Fixed objective. action_balanced upweights train edges/molecules with large observed activity jumps.",
    )
    parser.add_argument(
        "--scale-mode",
        choices=["global", "target"],
        default="global",
        help="Scale activities globally or separately per MoleculeACE target before fitting the single shared model.",
    )
    args = parser.parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    fp = Fingerprints(radius=2, n_bits=1024)
    frames = []
    for target_id, dataset in enumerate(MOLECULEACE_DATASETS):
        frame = load_moleculeace(dataset).copy()
        frame["target_id"] = target_id
        frames.append(frame)
    all_frame = pd.concat(frames, ignore_index=True)
    train = all_frame[all_frame.split.eq("train")].reset_index(drop=True)
    test = all_frame[all_frame.split.eq("test")].reset_index(drop=True)
    x_train = _features(fp, train.smiles.to_numpy())
    x_test = _features(fp, test.smiles.to_numpy())
    y = train.target.to_numpy(dtype=np.float32)
    target_id_train = train.target_id.to_numpy(dtype=np.int64)
    target_means = np.zeros(len(MOLECULEACE_DATASETS), dtype=np.float32)
    target_stds = np.ones(len(MOLECULEACE_DATASETS), dtype=np.float32)
    if args.scale_mode == "target":
        y_scaled = np.empty_like(y)
        for target_id in range(len(MOLECULEACE_DATASETS)):
            mask = target_id_train == target_id
            target_means[target_id] = float(y[mask].mean())
            target_stds[target_id] = float(y[mask].std() + 1e-6)
            y_scaled[mask] = (y[mask] - target_means[target_id]) / target_stds[target_id]
    else:
        target_means[:] = float(y.mean())
        target_stds[:] = float(y.std() + 1e-6)
        y_scaled = (y - target_means[0]) / target_stds[0]
    srcs, dsts, deltas, node_action_scale = build_pair_edges(train, x_train, train.target_id.to_numpy(), y_scaled)
    if args.loss_mode == "action_balanced" and len(deltas):
        mean_abs_delta = float(np.abs(deltas).mean() + 1e-6)
        pair_action_scale = np.maximum(np.abs(deltas), mean_abs_delta) / mean_abs_delta
    else:
        pair_action_scale = np.ones(len(deltas), dtype=np.float32)
        node_action_scale = np.ones(len(train), dtype=np.float32)
    print(
        json.dumps(
            {
                "train": len(train),
                "test": len(test),
                "pair_edges": len(srcs),
                "loss_mode": args.loss_mode,
                "scale_mode": args.scale_mode,
                "mean_pair_weight": float(pair_action_scale.mean()) if len(pair_action_scale) else 1.0,
                "mean_node_weight": float(node_action_scale.mean()),
            }
        ),
        flush=True,
    )

    device = torch.device(args.device)
    model = TargetActionNet(x_train.shape[1], len(MOLECULEACE_DATASETS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    x_t = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_scaled, dtype=torch.float32, device=device)
    target_t = torch.tensor(target_id_train, dtype=torch.long, device=device)
    src_t = torch.tensor(srcs, dtype=torch.long, device=device)
    dst_t = torch.tensor(dsts, dtype=torch.long, device=device)
    delta_t = torch.tensor(deltas, dtype=torch.float32, device=device)
    pair_weight_t = torch.tensor(pair_action_scale, dtype=torch.float32, device=device)
    node_weight_t = torch.tensor(node_action_scale, dtype=torch.float32, device=device)
    n = len(train)
    p = len(srcs)
    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=device)
        pair_perm = torch.randperm(p, device=device)
        cursor = 0
        losses = []
        for start in range(0, n, args.batch_size):
            idx = perm[start : start + args.batch_size]
            pred = model(x_t[idx], target_t[idx])
            abs_loss = F.smooth_l1_loss(pred, y_t[idx], reduction="none")
            loss = (abs_loss * node_weight_t[idx]).mean()
            pidx = pair_perm[cursor : cursor + args.pair_batch_size]
            cursor += args.pair_batch_size
            if len(pidx):
                f_src = model(x_t[src_t[pidx]], target_t[src_t[pidx]])
                f_dst = model(x_t[dst_t[pidx]], target_t[dst_t[pidx]])
                pair_loss = F.smooth_l1_loss(f_dst - f_src, delta_t[pidx], reduction="none")
                loss = loss + (pair_loss * pair_weight_t[pidx]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(json.dumps({"epoch": epoch + 1, "loss": float(np.mean(losses))}), flush=True)

    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preds = []
    with torch.no_grad():
        x_eval = torch.tensor(x_test, dtype=torch.float32, device=device)
        target_eval = torch.tensor(test.target_id.to_numpy(dtype=np.int64), dtype=torch.long, device=device)
        for start in range(0, len(test), args.batch_size):
            out = model(x_eval[start : start + args.batch_size], target_eval[start : start + args.batch_size])
            preds.append(out.detach().cpu().numpy())
    pred_scaled = np.concatenate(preds)
    test_target_id = test.target_id.to_numpy(dtype=np.int64)
    prediction = pred_scaled * target_stds[test_target_id] + target_means[test_target_id]
    test = test.copy()
    test["prediction"] = prediction
    rows = []
    for dataset, group in test.groupby("dataset", sort=False):
        metric = regression_metrics(
            group.target.to_numpy(dtype=float),
            group.prediction.to_numpy(dtype=float),
            group.cliff_mol.to_numpy(dtype=bool),
        )
        metric.update(
            {
                "dataset": dataset,
                "model": f"multitarget_action_{args.scale_mode}_{args.loss_mode}",
                "seed": 42,
            }
        )
        rows.append(metric)
        group[["dataset", "smiles", "target", "cliff_mol", "split", "prediction"]].to_csv(
            args.output_dir / f"{dataset}.multitarget_action.predictions.csv",
            index=False,
        )
        print(json.dumps(metric, sort_keys=True), flush=True)
    summary = pd.DataFrame(rows).sort_values("dataset")
    summary.to_csv(args.output_dir / "summary.multitarget_action.csv", index=False)
    aggregate = {
        "model": "multitarget_action",
        "scale_mode": args.scale_mode,
        "loss_mode": args.loss_mode,
        "datasets": len(summary),
        "macro_rmse": float(summary.rmse.mean()),
        "macro_cliff_rmse": float(summary.cliff_rmse.mean()),
        "macro_noncliff_rmse": float(summary.noncliff_rmse.mean()),
    }
    (args.output_dir / "aggregate.multitarget_action.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(aggregate, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
