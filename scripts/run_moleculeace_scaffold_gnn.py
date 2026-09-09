from __future__ import annotations

"""GINE-NodeNorm baseline under the exact scaffold protocol used by TAPER."""

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
from run_graph_cross_dataset_baselines import GraphList, GraphRegressor, collate_graphs, predict_regression, seed_everything, to_device
from run_moleculeace_scaffold_taper import scaffold_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(MOLECULEACE_DATASETS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gine_nodenorm_scaffold_seed42"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for dataset in args.datasets:
        seed_everything(args.seed)
        fit, test = scaffold_split(load_moleculeace(dataset), args.seed)
        # A fixed held-in validation subset is used only for early stopping.
        rng = np.random.default_rng(args.seed)
        indices = rng.permutation(len(fit))
        n_valid = max(1, int(round(.12 * len(fit))))
        valid_index, train_index = indices[:n_valid], indices[n_valid:]
        mean = float(fit.iloc[train_index].target.mean()); std = max(float(fit.iloc[train_index].target.std()), 1e-6)
        all_smiles = fit.smiles.tolist() + test.smiles.tolist()
        all_y = np.concatenate([(fit.target.to_numpy(float) - mean) / std,
                                (test.target.to_numpy(float) - mean) / std])
        graphs = GraphList(all_smiles, all_y)
        nfit = len(fit)
        def loader(idx, shuffle):
            return DataLoader(torch.utils.data.Subset(graphs, list(map(int, idx))),
                              batch_size=args.batch_size, shuffle=shuffle,
                              collate_fn=collate_graphs, num_workers=0, pin_memory=True)
        train_loader, valid_loader = loader(train_index, True), loader(valid_index, False)
        test_loader = loader(np.arange(nfit, nfit + len(test)), False)
        model = GraphRegressor("gine_nodenorm").to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
        best, state, stale = np.inf, None, 0
        for epoch in range(args.epochs):
            model.train()
            for batch in train_loader:
                batch = to_device(batch, device); optimizer.zero_grad(set_to_none=True)
                loss = F.mse_loss(model(batch), batch["y"].view(-1)); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
            yv, pv = predict_regression(model, valid_loader, device)
            score = float(np.mean((yv - pv) ** 2))
            if score < best - 1e-5:
                best, stale = score, 0
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= 8: break
        if state is None: raise RuntimeError("no checkpoint")
        model.load_state_dict(state)
        yt, pt = predict_regression(model, test_loader, device)
        metrics = regression_metrics(yt * std + mean, pt * std + mean, test.cliff_mol.to_numpy(bool))
        row = {"dataset": dataset, "model": "GINE + NodeNorm", "seed": args.seed,
               "epochs_trained": epoch + 1, "n_train": len(train_index), "n_test": len(test),
               "n_cliff": int(test.cliff_mol.sum()), **metrics}
        rows.append(row); print(json.dumps(row), flush=True)
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows); table.to_csv(out / "summary.scaffold.csv", index=False)
    aggregate = table[["rmse", "cliff_rmse", "noncliff_rmse"]].mean().to_dict()
    (out / "aggregate.scaffold.json").write_text(json.dumps({"datasets": len(table), "model": "GINE + NodeNorm", **aggregate}, indent=2)+"\n")


if __name__ == "__main__":
    main()
