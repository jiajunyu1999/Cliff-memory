from __future__ import annotations

"""One frozen-representation, target-conditioned PC-GAM potential.

This is a deliberately single, pre-registered recipe: one potential produces
the prediction and action supervision is its finite difference.  There is no
fallback predictor, model averaging, validation selection, checkpoint choice,
hyperparameter sweep, or seed sweep.
"""

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


SEED = 42
EPOCHS = 160
BATCH_SIZE = 512
PAIR_BATCH_SIZE = 1024


class TargetConditionedPotential(nn.Module):
    """A single conservative potential with protein-conditioned FiLM."""

    def __init__(self, fp_dim: int, chemical_dim: int, protein_dim: int, context_dim: int):
        super().__init__()
        self.fp_encoder = nn.Sequential(
            nn.Linear(fp_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 256),
        )
        self.chemical_encoder = nn.Sequential(
            nn.Linear(chemical_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
        )
        self.ligand_fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
        )
        self.target_encoder = nn.Sequential(
            nn.Linear(protein_dim + context_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
        )
        self.film = nn.Linear(128, 512)
        self.potential = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        fingerprint: torch.Tensor,
        chemical: torch.Tensor,
        protein: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        fp = self.fp_encoder(fingerprint)
        pretrained = self.chemical_encoder(chemical)
        ligand = self.ligand_fusion(torch.cat([fp, pretrained], dim=1))
        target = self.target_encoder(torch.cat([protein, context], dim=1))
        gamma, beta = self.film(target).chunk(2, dim=1)
        conditioned = ligand * (1.0 + 0.25 * torch.tanh(gamma)) + 0.25 * beta
        return self.potential(torch.cat([conditioned, target], dim=1)).squeeze(-1)


def _load_smiles_embeddings(path: Path, smiles: pd.Series) -> np.ndarray:
    payload = np.load(path, allow_pickle=False)
    lookup = {str(value): i for i, value in enumerate(payload["smiles"])}
    missing = [value for value in smiles.astype(str) if value not in lookup]
    if missing:
        raise RuntimeError(f"ChemBERTa cache misses {len(missing)} molecules; first={missing[0]}")
    return payload["embedding"][[lookup[value] for value in smiles.astype(str)]].astype(np.float32)


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(0, keepdims=True)
    std = train.std(0, keepdims=True)
    std[std < 1e-5] = 1.0
    return ((train - mean) / std).astype(np.float32), ((test - mean) / std).astype(np.float32)


def _target_arrays(
    frame: pd.DataFrame,
    target_context: pd.DataFrame,
    protein_cache: Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    # The checked-in cache builder writes Unicode arrays.  ``allow_pickle`` is
    # retained for the first locally generated cache made before that invariant
    # was enforced; both arrays are immediately cast/validated below.
    protein_payload = np.load(protein_cache, allow_pickle=True)
    protein_lookup = {
        str(value): protein_payload["embedding"][i]
        for i, value in enumerate(protein_payload["accession"])
    }
    context_by_dataset = target_context.set_index("dataset")
    classes = sorted(target_context.receptor_class.astype(str).unique())
    measurements = sorted(target_context.measurement_type.astype(str).unique())
    context_names = [f"class={value}" for value in classes] + [
        f"measurement={value}" for value in measurements
    ]
    class_index = {value: i for i, value in enumerate(classes)}
    measurement_index = {value: i + len(classes) for i, value in enumerate(measurements)}
    proteins = []
    contexts = np.zeros((len(frame), len(context_names)), dtype=np.float32)
    for i, dataset in enumerate(frame.dataset.astype(str)):
        row = context_by_dataset.loc[dataset]
        accession = str(row.uniprot_accession)
        if accession not in protein_lookup:
            raise RuntimeError(f"ESM cache misses target {accession}")
        proteins.append(protein_lookup[accession])
        contexts[i, class_index[str(row.receptor_class)]] = 1.0
        contexts[i, measurement_index[str(row.measurement_type)]] = 1.0
    return np.asarray(proteins, dtype=np.float32), contexts, context_names


def _pair_edges(frame: pd.DataFrame, bits: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, ...]:
    srcs: list[int] = []
    dsts: list[int] = []
    for _dataset, group in frame.groupby("dataset", sort=False):
        index = group.index.to_numpy()
        similarity = tanimoto_matrix(bits[index], bits[index])
        np.fill_diagonal(similarity, -1.0)
        neighbors = max(4, min(24, int(np.sqrt(len(index)))))
        for local_src, global_src in enumerate(index):
            order = np.argsort(similarity[local_src])[::-1][:neighbors]
            accepted = order[similarity[local_src, order] >= 0.35]
            srcs.extend([int(global_src)] * len(accepted))
            dsts.extend(index[accepted].astype(int).tolist())
    src = np.asarray(srcs, dtype=np.int64)
    dst = np.asarray(dsts, dtype=np.int64)
    delta = (y[dst] - y[src]).astype(np.float32)
    mean_delta = float(np.abs(delta).mean() + 1e-6)
    pair_weight = np.maximum(np.abs(delta), mean_delta) / mean_delta
    node_max = np.zeros(len(frame), dtype=np.float32)
    np.maximum.at(node_max, src, np.abs(delta))
    np.maximum.at(node_max, dst, np.abs(delta))
    positive_mean = float(node_max[node_max > 0].mean() + 1e-6)
    node_weight = np.maximum(node_max, positive_mean) / positive_mean
    return src, dst, delta, pair_weight.astype(np.float32), node_weight.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/moleculeace_target_action_v1"))
    parser.add_argument("--chemical-cache", type=Path, default=Path("data/moleculeace_chemberta_mtr.npz"))
    parser.add_argument("--target-context", type=Path, default=Path("data/moleculeace_target_context.csv"))
    parser.add_argument("--protein-cache", type=Path, default=Path("data/moleculeace_esm2_t30.npz"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    frames = []
    for dataset in MOLECULEACE_DATASETS:
        frame = load_moleculeace(dataset).copy()
        frames.append(frame)
    all_frame = pd.concat(frames, ignore_index=True)
    train = all_frame[all_frame.split.eq("train")].reset_index(drop=True)
    test = all_frame[all_frame.split.eq("test")].reset_index(drop=True)

    fingerprint = Fingerprints(radius=2, n_bits=1024)
    train_bits = fingerprint.bits(train.smiles.to_numpy()).astype(np.float32)
    test_bits = fingerprint.bits(test.smiles.to_numpy()).astype(np.float32)
    train_counts = fingerprint.counts(train.smiles.to_numpy()).astype(np.float32) / 4.0
    test_counts = fingerprint.counts(test.smiles.to_numpy()).astype(np.float32) / 4.0
    train_fp = np.concatenate([train_bits, train_counts], axis=1)
    test_fp = np.concatenate([test_bits, test_counts], axis=1)

    chemical_train = _load_smiles_embeddings(args.chemical_cache, train.smiles)
    chemical_test = _load_smiles_embeddings(args.chemical_cache, test.smiles)
    chemical_train, chemical_test = _standardize(chemical_train, chemical_test)
    target_context = pd.read_csv(args.target_context)
    protein_train, context_train, context_names = _target_arrays(train, target_context, args.protein_cache)
    protein_test, context_test, _ = _target_arrays(test, target_context, args.protein_cache)
    protein_train, protein_test = _standardize(protein_train, protein_test)

    raw_y = train.target.to_numpy(dtype=np.float32)
    y_mean = float(raw_y.mean())
    y_std = float(raw_y.std() + 1e-6)
    y = (raw_y - y_mean) / y_std
    src, dst, delta, pair_weight, node_weight = _pair_edges(train, train_bits, y)
    print(
        json.dumps(
            {
                "train": len(train),
                "test": len(test),
                "pair_edges": len(src),
                "chemical_dim": chemical_train.shape[1],
                "protein_dim": protein_train.shape[1],
                "context": context_names,
                "seed": SEED,
                "epochs": EPOCHS,
                "selection": "none",
            }
        ),
        flush=True,
    )

    device = torch.device(args.device)
    arrays = [train_fp, chemical_train, protein_train, context_train, y, node_weight]
    fp_t, chemical_t, protein_t, context_t, y_t, node_weight_t = [
        torch.tensor(value, dtype=torch.float32, device=device) for value in arrays
    ]
    src_t = torch.tensor(src, dtype=torch.long, device=device)
    dst_t = torch.tensor(dst, dtype=torch.long, device=device)
    delta_t = torch.tensor(delta, dtype=torch.float32, device=device)
    pair_weight_t = torch.tensor(pair_weight, dtype=torch.float32, device=device)
    model = TargetConditionedPotential(
        train_fp.shape[1], chemical_train.shape[1], protein_train.shape[1], context_train.shape[1]
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    n = len(train)
    p = len(src)
    model.train()
    for epoch in range(EPOCHS):
        node_order = torch.randperm(n, device=device)
        pair_order = torch.randperm(p, device=device)
        cursor = 0
        losses: list[float] = []
        for start in range(0, n, BATCH_SIZE):
            index = node_order[start : start + BATCH_SIZE]
            prediction = model(fp_t[index], chemical_t[index], protein_t[index], context_t[index])
            absolute = F.smooth_l1_loss(prediction, y_t[index], reduction="none")
            loss = (absolute * node_weight_t[index]).mean()
            pair_index = pair_order[cursor : cursor + PAIR_BATCH_SIZE]
            cursor += PAIR_BATCH_SIZE
            source_prediction = model(
                fp_t[src_t[pair_index]], chemical_t[src_t[pair_index]],
                protein_t[src_t[pair_index]], context_t[src_t[pair_index]],
            )
            destination_prediction = model(
                fp_t[dst_t[pair_index]], chemical_t[dst_t[pair_index]],
                protein_t[dst_t[pair_index]], context_t[dst_t[pair_index]],
            )
            action = F.smooth_l1_loss(
                destination_prediction - source_prediction, delta_t[pair_index], reduction="none"
            )
            loss = loss + (action * pair_weight_t[pair_index]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 0 or (epoch + 1) % 20 == 0:
            print(json.dumps({"epoch": epoch + 1, "loss": float(np.mean(losses))}), flush=True)

    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(test), BATCH_SIZE):
            stop = start + BATCH_SIZE
            output = model(
                torch.tensor(test_fp[start:stop], dtype=torch.float32, device=device),
                torch.tensor(chemical_test[start:stop], dtype=torch.float32, device=device),
                torch.tensor(protein_test[start:stop], dtype=torch.float32, device=device),
                torch.tensor(context_test[start:stop], dtype=torch.float32, device=device),
            )
            predictions.append(output.cpu().numpy())
    test = test.copy()
    test["prediction"] = np.concatenate(predictions) * y_std + y_mean
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset, group in test.groupby("dataset", sort=False):
        metric = regression_metrics(
            group.target.to_numpy(dtype=float),
            group.prediction.to_numpy(dtype=float),
            group.cliff_mol.to_numpy(dtype=bool),
        )
        metric.update({"dataset": dataset, "model": "target_action_v1", "seed": SEED})
        rows.append(metric)
        group[["dataset", "smiles", "target", "cliff_mol", "split", "prediction"]].to_csv(
            args.output_dir / f"{dataset}.target_action.predictions.csv", index=False
        )
        print(json.dumps(metric, sort_keys=True), flush=True)
    summary = pd.DataFrame(rows).sort_values("dataset")
    summary.to_csv(args.output_dir / "summary.target_action.csv", index=False)
    aggregate = {
        "model": "target_action_v1",
        "chemical_cache": str(args.chemical_cache),
        "protein_cache": str(args.protein_cache),
        "datasets": len(summary),
        "macro_rmse": float(summary.rmse.mean()),
        "macro_cliff_rmse": float(summary.cliff_rmse.mean()),
        "macro_noncliff_rmse": float(summary.noncliff_rmse.mean()),
        "seed": SEED,
        "epochs": EPOCHS,
        "selection": "none",
        "ensemble": False,
    }
    (args.output_dir / "aggregate.target_action.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    )
    torch.save({"model": model.state_dict(), "aggregate": aggregate}, args.output_dir / "model.pt")
    print(json.dumps(aggregate, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
