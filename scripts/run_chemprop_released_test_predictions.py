from __future__ import annotations

"""Locally train the Chemprop/D-MPNN baseline on released train and test pool.

This produces molecule-level predictions for the same held-out MoleculeACE
test rows used by the retrospective virtual-screening analysis.  A fixed
held-in subset chooses early stopping; no test label controls optimization.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from molcliff.data import MOLECULEACE_DATASETS, load_moleculeace
from molcliff.metrics import regression_metrics
from run_graph_cross_dataset_baselines import (
    GraphList, GraphRegressor, collate_graphs, predict_regression,
    seed_everything, to_device,
)


def run_one(dataset: str, seed: int, epochs: int, batch_size: int) -> tuple[dict, pd.DataFrame]:
    seed_everything(seed)
    frame = load_moleculeace(dataset)
    fit = frame[frame.split.eq("train")].copy().reset_index(drop=True)
    test = frame[frame.split.eq("test")].copy().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(fit))
    n_valid = max(1, int(round(.12 * len(fit))))
    valid_index, train_index = permutation[:n_valid], permutation[n_valid:]
    mean = float(fit.iloc[train_index].target.mean())
    std = max(float(fit.iloc[train_index].target.std()), 1e-6)
    labels = np.concatenate([(fit.target.to_numpy(float) - mean) / std,
                             (test.target.to_numpy(float) - mean) / std])
    graphs = GraphList(fit.smiles.tolist() + test.smiles.tolist(), labels)
    nfit = len(fit)

    def loader(indices, shuffle: bool):
        return DataLoader(torch.utils.data.Subset(graphs, list(map(int, indices))),
                          batch_size=batch_size, shuffle=shuffle, collate_fn=collate_graphs,
                          num_workers=0, pin_memory=True)
    train_loader = loader(train_index, True)
    valid_loader = loader(valid_index, False)
    test_loader = loader(np.arange(nfit, nfit + len(test)), False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphRegressor("chemprop").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    best, state, stale = np.inf, None, 0
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(batch), batch["y"].view(-1))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
        y_valid, p_valid = predict_regression(model, valid_loader, device)
        score = float(np.mean((y_valid - p_valid) ** 2))
        if score < best - 1e-5:
            best, stale = score, 0
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 8:
                break
    if state is None:
        raise RuntimeError("No early-stopping state")
    model.load_state_dict(state)
    y_test, p_test = predict_regression(model, test_loader, device)
    pred = p_test * std + mean
    truth = y_test * std + mean
    row = {"dataset": dataset, "model": "Chemprop", "seed": seed, "epochs_trained": epoch + 1,
           "n_train": len(train_index), "n_test": len(test),
           **regression_metrics(truth, pred, test.cliff_mol.to_numpy(bool))}
    out = test[["smiles", "target", "cliff_mol"]].copy()
    out.insert(0, "dataset", dataset)
    out["prediction_chemprop"] = pred
    return row, out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(MOLECULEACE_DATASETS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/chemprop_released_test"))
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, predictions = [], []
    for dataset in args.datasets:
        row, pred = run_one(dataset, args.seed, args.epochs, args.batch_size)
        rows.append(row); predictions.append(pred); print(json.dumps(row), flush=True)
    summary = pd.DataFrame(rows); pred_table = pd.concat(predictions, ignore_index=True)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    pred_table.to_csv(args.output_dir / "predictions.csv", index=False)
    macro = summary[["rmse", "cliff_rmse", "noncliff_rmse", "mae"]].mean().to_dict()
    (args.output_dir / "aggregate.json").write_text(json.dumps({"datasets": len(summary), **macro}, indent=2) + "\n")


if __name__ == "__main__":
    main()
