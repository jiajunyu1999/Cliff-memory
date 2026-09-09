from __future__ import annotations

"""Protocol-matched graph baselines for GraphCliff regression and ACNet pairs."""

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import GINEConv, SAGPooling, global_max_pool, global_mean_pool
from torch_scatter import scatter_add


ROOT = Path(__file__).resolve().parents[1]
GRAPHCLIFF_ROOT = ROOT / "third_party" / "GraphCliff"
sys.path.insert(0, str(GRAPHCLIFF_ROOT))
from dataset import collate_graphs  # noqa: E402
from dataset_utils import smiles_to_graph  # noqa: E402
from model import AtomEncoder, GraphCliffEncoder  # noqa: E402

from molcliff.data import EXPANDED_CLIFF_ROOT, load_moleculeace  # noqa: E402
from molcliff.metrics import regression_metrics  # noqa: E402
from run_acnet_action import _labels, official_random_split  # noqa: E402


EXTERNAL_DATASETS = (
    "CHEMBL2311243_IC50", "CHEMBL2598_IC50", "CHEMBL3024_IC50",
    "CHEMBL4685_IC50", "CHEMBL5014_IC50",
)
MODELS = ("chemprop", "gine_residual", "gine_nodenorm", "gine_pairnorm", "graphcliff")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NodeNorm(nn.Module):
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        del batch
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return (x - mean) / std


class PairNorm(nn.Module):
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        mean = global_mean_pool(x, batch)
        centered = x - mean[batch]
        squared_norm = centered.square().sum(dim=-1, keepdim=True)
        scale = global_mean_pool(squared_norm, batch).sqrt().clamp_min(1e-6)
        return centered / scale[batch]


class GraphBackbone(nn.Module):
    def __init__(self, name: str, atom_dim: int = 38, edge_dim: int = 13,
                 hidden: int = 128, layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.name = name
        self.atom = AtomEncoder(atom_dim, hidden, dropout)
        self.dropout = nn.Dropout(dropout)
        if name == "graphcliff":
            self.encoder = GraphCliffEncoder(
                hidden, edge_dim, num_layers=layers, groups=4, mid_K=3,
                dropout=dropout,
            )
            self.pool = SAGPooling(hidden, ratio=0.8)
        elif name == "chemprop":
            self.dmpnn_init = nn.Linear(hidden + edge_dim, hidden)
            self.dmpnn_msg = nn.Linear(hidden, hidden, bias=False)
            self.dmpnn_out = nn.Linear(hidden + hidden, hidden)
            self.dmpnn_depth = layers
        else:
            self.layers = nn.ModuleList()
            for _ in range(layers):
                mlp = nn.Sequential(
                    nn.Linear(hidden, hidden), nn.ReLU(),
                    nn.Linear(hidden, hidden),
                )
                self.layers.append(GINEConv(mlp, edge_dim=edge_dim, train_eps=True))
            self.norm = (
                NodeNorm() if name == "gine_nodenorm" else
                PairNorm() if name == "gine_pairnorm" else None
            )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.atom(batch["x"])
        edge_index, edge_attr, graph = (
            batch["edge_index"], batch["edge_attr"], batch["batch"]
        )
        if self.name == "graphcliff":
            x = self.encoder(x, edge_index, edge_attr)
            x, edge_index, _, graph, _, _ = self.pool(
                x, edge_index, edge_attr, graph
            )
        elif self.name == "chemprop":
            if edge_index.numel() == 0:
                x = F.relu(self.dmpnn_out(torch.cat([x, torch.zeros_like(x)], dim=-1)))
            else:
                src, dst = edge_index
                init = F.relu(self.dmpnn_init(torch.cat([x[src], edge_attr], dim=-1)))
                hidden = init
                reverse = torch.arange(edge_index.size(1), device=edge_index.device)
                reverse = reverse.view(-1, 2)[:, [1, 0]].reshape(-1)
                for _ in range(self.dmpnn_depth - 1):
                    aggregate = scatter_add(hidden, dst, dim=0, dim_size=x.size(0))
                    directed = aggregate[src] - hidden[reverse]
                    hidden = F.relu(init + self.dmpnn_msg(directed))
                aggregate = scatter_add(hidden, dst, dim=0, dim_size=x.size(0))
                x = F.relu(self.dmpnn_out(torch.cat([x, aggregate], dim=-1)))
        else:
            for layer in self.layers:
                residual = x
                x = layer(x, edge_index, edge_attr)
                x = self.dropout(F.relu(x)) + residual
                if self.norm is not None:
                    x = self.norm(x, graph)
        return torch.cat([global_mean_pool(x, graph), global_max_pool(x, graph)], dim=-1)


class GraphRegressor(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.backbone = GraphBackbone(name)
        self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(.1), nn.Linear(128, 1))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.head(self.backbone(batch)).view(-1)


class PairClassifier(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.backbone = GraphBackbone(name)
        self.head = nn.Sequential(nn.Linear(768, 192), nn.ReLU(), nn.Dropout(.15), nn.Linear(192, 1))

    def score(self, embedding: torch.Tensor, first_index: torch.Tensor,
              second_index: torch.Tensor) -> torch.Tensor:
        first, second = embedding[first_index], embedding[second_index]
        symmetric = torch.cat([first + second, (first - second).abs(), first * second], dim=-1)
        return self.head(symmetric).view(-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        embedding = self.backbone(batch)
        half = embedding.shape[0] // 2
        index = torch.arange(half, device=embedding.device)
        return self.score(embedding, index, index + half)


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


class GraphList(Dataset):
    def __init__(self, smiles: list[str], targets: np.ndarray):
        self.graphs = [smiles_to_graph(s, float(y)) for s, y in zip(smiles, targets)]
    def __len__(self): return len(self.graphs)
    def __getitem__(self, index): return self.graphs[index]


@dataclass
class PairRecord:
    first: object
    second: object
    label: int


class PairList(Dataset):
    def __init__(self, items: list[dict]):
        cache = {}
        self.records = []
        for item in items:
            try:
                for smiles in (item["SMILES1"], item["SMILES2"]):
                    if smiles not in cache:
                        cache[smiles] = smiles_to_graph(smiles, 0.0)
                one, two = cache[item["SMILES1"]], cache[item["SMILES2"]]
            except ValueError:
                continue
            self.records.append(PairRecord(one, two, int(item["Value"])))
    def __len__(self): return len(self.records)
    def __getitem__(self, index): return self.records[index]


def collate_pairs(records: list[PairRecord]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    graphs = [record.first for record in records] + [record.second for record in records]
    batch = collate_graphs(graphs)
    labels = torch.tensor([record.label for record in records], dtype=torch.float32)
    return batch, labels


def predict_regression(model, loader, device):
    model.eval(); truth, scores = [], []
    with torch.no_grad():
        for batch in loader:
            truth.extend(batch["y"].view(-1).numpy().tolist())
            scores.extend(model(to_device(batch, device)).cpu().numpy().tolist())
    return np.asarray(truth), np.asarray(scores)


def predict_pairs(model, loader, device):
    model.eval(); truth, scores = [], []
    with torch.no_grad():
        for batch, labels in loader:
            truth.extend(labels.numpy().tolist())
            scores.extend(model(to_device(batch, device)).cpu().numpy().tolist())
    return np.asarray(truth), np.asarray(scores)


def run_regression(name: str, dataset_names: tuple[str, ...], root, collection: str,
                   device: torch.device, epochs: int, batch_size: int,
                   checkpoint: Path | None = None) -> list[dict]:
    rows = []
    completed: set[tuple[str, str, int]] = set()
    if checkpoint is not None and checkpoint.exists() and checkpoint.stat().st_size > 0:
        try:
            previous = pd.read_csv(checkpoint)
            if {"dataset", "model", "fold"}.issubset(previous.columns):
                for row in previous[["dataset", "model", "fold"]].itertuples(index=False):
                    completed.add((str(row.dataset), str(row.model), int(row.fold)))
        except Exception as exc:
            print(json.dumps({"warning": "could not read checkpoint", "path": str(checkpoint),
                              "error": str(exc)}), flush=True)
    for dataset_name in dataset_names:
        frame = load_moleculeace(dataset_name, root=root)
        pool = frame[frame.split.eq("train")].reset_index(drop=True)
        strata = pool.cliff_mol.astype(int).to_numpy()
        folds = StratifiedKFold(5, shuffle=True, random_state=42)
        for fold, (fit_index, test_index) in enumerate(folds.split(pool, strata)):
            if (dataset_name, name, fold) in completed:
                continue
            seed_everything(4200 + fold)
            fit_index = np.asarray(fit_index); test_index = np.asarray(test_index)
            rng = np.random.RandomState(4200 + fold)
            shuffled = fit_index[rng.permutation(len(fit_index))]
            n_valid = max(1, int(round(.12 * len(shuffled))))
            valid_index, train_index = shuffled[:n_valid], shuffled[n_valid:]
            mean = float(pool.iloc[train_index].target.mean())
            std = max(float(pool.iloc[train_index].target.std()), 1e-6)
            target_z = (pool.target.to_numpy(float) - mean) / std
            graph_data = GraphList(pool.smiles.tolist(), target_z)
            make_loader = lambda indices, shuffle: DataLoader(
                torch.utils.data.Subset(graph_data, indices.tolist()), batch_size=batch_size,
                shuffle=shuffle, collate_fn=collate_graphs, num_workers=0, pin_memory=True,
            )
            train_loader = make_loader(train_index, True)
            valid_loader = make_loader(valid_index, False)
            test_loader = make_loader(test_index, False)
            model = GraphRegressor(name).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
            best, best_state, stale = np.inf, None, 0
            for _ in range(epochs):
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
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                else:
                    stale += 1
                    if stale >= 8: break
            model.load_state_dict(best_state)
            y_test_z, pred_z = predict_regression(model, test_loader, device)
            y_test = y_test_z * std + mean; prediction = pred_z * std + mean
            metrics = regression_metrics(y_test, prediction, pool.iloc[test_index].cliff_mol.to_numpy(bool))
            rows.append({"collection": collection, "dataset": dataset_name,
                         "model": name, "seed": 42, "fold": fold,
                         "n_test": len(test_index), "n_cliff": int(strata[test_index].sum()), **metrics})
            print(json.dumps(rows[-1]), flush=True)
            if checkpoint is not None:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                fields = list(rows[-1].keys())
                needs_header = (not checkpoint.exists()) or checkpoint.stat().st_size == 0
                with checkpoint.open("a", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    if needs_header:
                        writer.writeheader()
                    writer.writerow(rows[-1])
                completed.add((dataset_name, name, fold))
    return rows


def run_acnet(name: str, device: torch.device, epochs: int, batch_size: int,
              selected_targets: set[str] | None = None) -> list[dict]:
    data = json.loads((ROOT / "data" / "acnet" / "generated" / "MMP_AC.json").read_text())
    rows = []
    for target, raw_items in data.items():
        if selected_targets is not None and target not in selected_targets:
            continue
        seed_everything(42)
        pair_data = PairList(raw_items)
        items = [{"Value": record.label} for record in pair_data.records]
        split = official_random_split(items, seed=8)
        labels = np.asarray([record.label for record in pair_data.records])
        graph_index: dict[int, int] = {}
        unique_graphs = []
        first_index, second_index = [], []
        for record in pair_data.records:
            for graph in (record.first, record.second):
                if id(graph) not in graph_index:
                    graph_index[id(graph)] = len(unique_graphs)
                    unique_graphs.append(graph)
            first_index.append(graph_index[id(record.first)])
            second_index.append(graph_index[id(record.second)])
        graph_batch = to_device(collate_graphs(unique_graphs), device)
        first_index = torch.tensor(first_index, dtype=torch.long, device=device)
        second_index = torch.tensor(second_index, dtype=torch.long, device=device)
        label_tensor = torch.tensor(labels, dtype=torch.float32, device=device)
        train_index = torch.tensor(split.train, dtype=torch.long, device=device)
        valid_index = torch.tensor(split.valid, dtype=torch.long, device=device)
        test_index = torch.tensor(split.test, dtype=torch.long, device=device)
        model = PairClassifier(name).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-5)
        positives = max(int(labels[split.train].sum()), 1)
        negatives = max(len(split.train) - positives, 1)
        pos_weight = torch.tensor([negatives / positives], device=device)
        best, best_state, stale = -np.inf, None, 0
        for _ in range(epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            embedding = model.backbone(graph_batch)
            logits = model.score(embedding, first_index[train_index], second_index[train_index])
            loss = F.binary_cross_entropy_with_logits(
                logits, label_tensor[train_index], pos_weight=pos_weight
            )
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
            model.eval()
            with torch.no_grad():
                embedding = model.backbone(graph_batch)
                p_valid = model.score(
                    embedding, first_index[valid_index], second_index[valid_index]
                ).cpu().numpy()
            y_valid = labels[split.valid]
            score = average_precision_score(y_valid, p_valid)
            if score > best + 1e-5:
                best, stale = score, 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
                if stale >= 5: break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            embedding = model.backbone(graph_batch)
            scores = model.score(
                embedding, first_index[test_index], second_index[test_index]
            ).cpu().numpy()
        y_test = labels[split.test]
        row = {"collection": "ACNet", "target": target, "model": name,
               "feature_mode": "graph_pair", "n": len(pair_data),
               "n_train": len(split.train), "n_valid": len(split.valid), "n_test": len(split.test),
               "test_pos": int(y_test.sum()), "valid_ap": float(best),
               "auc": float(roc_auc_score(y_test, scores)),
               "ap": float(average_precision_score(y_test, scores))}
        rows.append(row); print(json.dumps(row), flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("moleculeace", "external", "acnet"), required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--targets", nargs="+",
        help=("Optional dataset subset for regression, or ACNet target subset for "
              "classification. This keeps smoke tests and full runs on the same code path."),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device requests CUDA, but CUDA is unavailable; pass --device cpu")
    device = torch.device(args.device)
    all_rows = []
    for name in args.models:
        if args.task == "external":
            datasets = tuple(args.targets) if args.targets else EXTERNAL_DATASETS
            invalid = sorted(set(datasets).difference(EXTERNAL_DATASETS))
            if invalid:
                raise ValueError(f"unknown external dataset(s): {invalid}")
            all_rows.extend(run_regression(name, datasets, EXPANDED_CLIFF_ROOT,
                                           "GraphCliff external", device, args.epochs or 80, args.batch_size,
                                           args.output))
        elif args.task == "moleculeace":
            from molcliff.data import MOLECULEACE_DATASETS
            datasets = tuple(args.targets) if args.targets else tuple(MOLECULEACE_DATASETS)
            invalid = sorted(set(datasets).difference(MOLECULEACE_DATASETS))
            if invalid:
                raise ValueError(f"unknown MoleculeACE dataset(s): {invalid}")
            all_rows.extend(run_regression(name, datasets, None,
                                           "MoleculeACE", device, args.epochs or 80, args.batch_size,
                                           args.output))
        else:
            selected = set(args.targets) if args.targets else None
            all_rows.extend(run_acnet(name, device, args.epochs or 30, args.batch_size, selected))
        if args.task == "acnet":
            # A full suite can take hours.  Persist every completed model so a
            # pre-empted run remains auditable and can be resumed model-by-model.
            result = pd.DataFrame(all_rows)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(args.output, index=False)
    result = pd.DataFrame(all_rows)
    print(json.dumps({"task": args.task, "models": args.models, "rows": len(result),
                      "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
