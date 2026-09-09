from __future__ import annotations

"""Common-scaffold-cancelled atom potential for MoleculeACE.

One GINE maps every atom to a scalar contribution; their normalized sum is the
only prediction.  For train-only aligned cliff pairs, matched common atoms are
constrained to retain their contributions and the uncommon contribution
difference is constrained to equal the measured activity action.  No ensemble,
checkpoint selection, test-time retrieval, or hyperparameter/seed search.
"""

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINEConv, global_add_pool

from molcliff.data import MOLECULEACE_DATASETS, load_moleculeace
from molcliff.metrics import regression_metrics


SEED = 42
ATOM_NUMBERS = (6, 7, 8, 9, 15, 16, 17, 35, 53)
BOND_TYPES = (Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
              Chem.BondType.TRIPLE, Chem.BondType.AROMATIC)
HYBRIDIZATIONS = (Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
                  Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D,
                  Chem.HybridizationType.SP3D2)


def _one_hot(value, choices) -> list[float]:
    return [float(value == item) for item in choices] + [float(value not in choices)]


def _graph(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed: {smiles}")
    atoms = []
    for atom in mol.GetAtoms():
        atoms.append(
            _one_hot(atom.GetAtomicNum(), ATOM_NUMBERS)
            + _one_hot(atom.GetDegree(), (0, 1, 2, 3, 4, 5))
            + _one_hot(atom.GetFormalCharge(), (-2, -1, 0, 1, 2))
            + _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
            + _one_hot(atom.GetTotalNumHs(), (0, 1, 2, 3, 4))
            + [float(atom.GetIsAromatic()), float(atom.IsInRing())]
        )
    edges, attrs = [], []
    for bond in mol.GetBonds():
        attr = (_one_hot(bond.GetBondType(), BOND_TYPES)
                + [float(bond.GetIsConjugated()), float(bond.IsInRing())]
                + _one_hot(int(bond.GetStereo()), (0, 1, 2, 3, 4, 5)))
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend(((left, right), (right, left)))
        attrs.extend((attr, attr))
    if not edges:
        edges = [(0, 0)]
        attrs = [[0.0] * 14]
    return Data(
        x=torch.tensor(atoms, dtype=torch.float32),
        edge_index=torch.tensor(edges, dtype=torch.long).T.contiguous(),
        edge_attr=torch.tensor(attrs, dtype=torch.float32),
    )


class AtomPotential(nn.Module):
    def __init__(self, atom_dim: int, bond_dim: int, hidden: int = 128, layers: int = 4):
        super().__init__()
        self.atom = nn.Linear(atom_dim, hidden)
        self.bond = nn.Linear(bond_dim, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.SiLU(),
                                nn.Linear(hidden * 2, hidden))
            self.convs.append(GINEConv(mlp, edge_dim=hidden, train_eps=True))
            self.norms.append(nn.LayerNorm(hidden))
        self.energy = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.SiLU(),
                                    nn.Linear(hidden // 2, 1))

    def forward(self, data: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.atom(data.x)
        edge = self.bond(data.edge_attr)
        for conv, norm in zip(self.convs, self.norms, strict=True):
            h = norm(h + conv(h, data.edge_index, edge))
        atom_energy = self.energy(h).squeeze(-1)
        total = global_add_pool(atom_energy, data.batch)
        count = torch.bincount(data.batch, minlength=int(data.num_graphs)).float()
        return total / count.sqrt().clamp_min(1.0), atom_energy


def _split(frame: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    official_train = frame[frame.split.eq("train")].reset_index(drop=True)
    if mode == "test":
        return official_train, frame[frame.split.eq("test")].reset_index(drop=True)
    fit, valid = train_test_split(
        np.arange(len(official_train)), test_size=0.2, random_state=SEED,
        stratify=official_train.cliff_mol.to_numpy(), shuffle=True,
    )
    return (official_train.iloc[np.sort(fit)].reset_index(drop=True),
            official_train.iloc[np.sort(valid)].reset_index(drop=True))


def _pairs(dataset: str, train: pd.DataFrame) -> list[dict]:
    path = Path("third_party/XACs/Data") / dataset / "mcs_dict_0.9.pkl"
    mapping = pickle.loads(path.read_bytes())
    index = {smiles: i for i, smiles in enumerate(train.smiles.astype(str))}
    y = train.target.to_numpy(dtype=np.float32)
    pairs, seen = [], set()
    for smiles_i, records in mapping.items():
        if smiles_i not in index:
            continue
        for record in records[1:]:
            smiles_j = record["smiles"]
            if smiles_j not in index:
                continue
            i, j = index[smiles_i], index[smiles_j]
            key = tuple(sorted((i, j)))
            if i == j or key in seen:
                continue
            seen.add(key)
            pairs.append({
                "i": i, "j": j, "delta": float(y[i] - y[j]),
                "common_i": list(map(int, record["common_atom_idx_i"])),
                "common_j": list(map(int, record["common_atom_idx_j"])),
                "uncommon_i": list(map(int, record["uncommon_atom_idx_i"])),
                "uncommon_j": list(map(int, record["uncommon_atom_idx_j"])),
            })
    return pairs


def _batch(graphs: list[Data], indices: np.ndarray, device: torch.device) -> Batch:
    return Batch.from_data_list([graphs[int(i)] for i in indices]).to(device)


def run_one(dataset: str, mode: str, device: torch.device, epochs: int,
            output_dir: Path) -> dict:
    frame = load_moleculeace(dataset)
    train, evaluation = _split(frame, mode)
    train_graphs = [_graph(s) for s in train.smiles.astype(str)]
    eval_graphs = [_graph(s) for s in evaluation.smiles.astype(str)]
    pairs = _pairs(dataset, train)
    y_raw = train.target.to_numpy(dtype=np.float32)
    y_mean, y_std = float(y_raw.mean()), float(y_raw.std() + 1e-6)
    y = torch.tensor((y_raw - y_mean) / y_std, device=device)
    for pair in pairs:
        pair["delta"] /= y_std
    model = AtomPotential(train_graphs[0].x.shape[1], train_graphs[0].edge_attr.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = np.random.default_rng(SEED)
    node_batch, pair_batch = 256, 64
    model.train()
    for epoch in range(epochs):
        node_order = rng.permutation(len(train_graphs))
        pair_order = rng.permutation(len(pairs)) if pairs else np.empty(0, dtype=int)
        pair_cursor = 0
        losses = []
        for start in range(0, len(node_order), node_batch):
            ids = node_order[start:start + node_batch]
            graph_batch = _batch(train_graphs, ids, device)
            pred, _ = model(graph_batch)
            loss = F.smooth_l1_loss(pred, y[torch.tensor(ids, device=device)])
            if len(pair_order):
                selected = pair_order[pair_cursor:pair_cursor + pair_batch]
                if len(selected) < pair_batch:
                    selected = np.concatenate((selected, pair_order[:pair_batch - len(selected)]))
                pair_cursor = (pair_cursor + pair_batch) % len(pair_order)
                records = [pairs[int(k)] for k in selected]
                flat_ids = np.asarray([[r["i"], r["j"]] for r in records]).reshape(-1)
                pair_graphs = _batch(train_graphs, flat_ids, device)
                pair_pred, energy = model(pair_graphs)
                action_pred = pair_pred[0::2] - pair_pred[1::2]
                action_true = torch.tensor([r["delta"] for r in records], device=device)
                action_loss = F.smooth_l1_loss(action_pred, action_true)
                common_losses, uncommon_pred = [], []
                for row, record in enumerate(records):
                    offset_i = int(pair_graphs.ptr[2 * row])
                    offset_j = int(pair_graphs.ptr[2 * row + 1])
                    ci, cj = record["common_i"], record["common_j"]
                    if ci and cj and len(ci) == len(cj):
                        ii = torch.tensor(ci, device=device) + offset_i
                        jj = torch.tensor(cj, device=device) + offset_j
                        common_losses.append(F.smooth_l1_loss(energy[ii], energy[jj]))
                    ui, uj = record["uncommon_i"], record["uncommon_j"]
                    ei = energy[torch.tensor(ui, device=device) + offset_i].sum() if ui else energy.new_zeros(())
                    ej = energy[torch.tensor(uj, device=device) + offset_j].sum() if uj else energy.new_zeros(())
                    uncommon_pred.append(ei - ej)
                common_loss = torch.stack(common_losses).mean() if common_losses else loss.new_zeros(())
                uncommon_loss = F.smooth_l1_loss(torch.stack(uncommon_pred), action_true)
                loss = loss + action_loss + common_loss + uncommon_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch == 0 or (epoch + 1) % 20 == 0:
            print(json.dumps({"dataset": dataset, "mode": mode, "epoch": epoch + 1,
                              "loss": float(np.mean(losses)), "pairs": len(pairs)}), flush=True)
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(eval_graphs), 512):
            ids = np.arange(start, min(start + 512, len(eval_graphs)))
            out, _ = model(_batch(eval_graphs, ids, device))
            predictions.extend((out.cpu().numpy() * y_std + y_mean).tolist())
    metric = regression_metrics(
        evaluation.target.to_numpy(dtype=float), np.asarray(predictions),
        evaluation.cliff_mol.to_numpy(dtype=bool),
    )
    metric.update({"dataset": dataset, "model": "uncommon_atom_potential", "mode": mode,
                   "seed": SEED, "epochs": epochs, "train_pairs": len(pairs)})
    output_dir.mkdir(parents=True, exist_ok=True)
    out = evaluation[["dataset", "smiles", "target", "cliff_mol", "split"]].copy()
    out["prediction"] = predictions
    out.to_csv(output_dir / f"{dataset}.{mode}.predictions.csv", index=False)
    print(json.dumps(metric, sort_keys=True), flush=True)
    return metric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(MOLECULEACE_DATASETS))
    parser.add_argument("--mode", choices=("validation", "test"), default="validation")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/moleculeace_uncommon_potential_v1"))
    args = parser.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    rows = [run_one(name, args.mode, torch.device(args.device), args.epochs, args.output_dir)
            for name in args.datasets]
    summary = pd.DataFrame(rows).sort_values("dataset")
    summary.to_csv(args.output_dir / f"summary.{args.mode}.csv", index=False)
    aggregate = {
        "model": "uncommon_atom_potential", "mode": args.mode, "datasets": len(summary),
        "macro_rmse": float(summary.rmse.mean()),
        "macro_cliff_rmse": float(summary.cliff_rmse.mean()),
        "macro_noncliff_rmse": float(summary.noncliff_rmse.mean()),
        "seed": SEED, "epochs": args.epochs, "selection": "final_epoch",
        "ensemble": False, "uses_test_labels_as_input": False,
    }
    (args.output_dir / f"aggregate.{args.mode}.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(aggregate, sort_keys=True))


if __name__ == "__main__":
    main()
