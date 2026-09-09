from __future__ import annotations

"""Train-only validation of one shared missing-aware bioactivity potential.

This is a single multitask predictor, not a prediction ensemble.  Sparse
off-target profiles are compressed once with train-only SVD and fused with a
small ligand encoder.  The same scalar potential is supervised by endpoint
activities and finite differences between same-target analogues.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import torch
from torch import nn
import torch.nn.functional as F

from molcliff.data import Fingerprints, MOLECULEACE_DATASETS, load_moleculeace, split_official_train
from molcliff.metrics import regression_metrics
from run_moleculeace_multitarget_action import build_pair_edges
from validate_moleculeace_offtarget_kernel import load_profiles, molecule_mapping
from validate_moleculeace_profile_alignment_kernel import (
    load_target_metadata,
    remove_equivalent_targets,
)


SEED = 42


class ProfilePotential(nn.Module):
    def __init__(self, chemical_dim: int, profile_dim: int, n_targets: int) -> None:
        super().__init__()
        self.chemical = nn.Sequential(
            nn.Linear(chemical_dim, 512), nn.LayerNorm(512), nn.SiLU(),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.SiLU(),
        )
        self.profile = nn.Sequential(
            nn.Linear(profile_dim, 256), nn.LayerNorm(256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(),
        )
        self.target = nn.Embedding(n_targets, 64)
        self.output = nn.Sequential(
            nn.Linear(256 + 128 + 64, 256), nn.LayerNorm(256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1),
        )

    def forward(
        self, chemical: torch.Tensor, profile: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return self.output(torch.cat([
            self.chemical(chemical), self.profile(profile), self.target(target)
        ], dim=1)).squeeze(1)


def profile_representation(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    profile_root: Path,
    metadata_path: Path,
    components: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    metadata = load_target_metadata(metadata_path)
    combined = pd.concat([train.assign(_partition=0), valid.assign(_partition=1)], ignore_index=True)
    row_lookup = {
        (str(row.dataset), str(row.molecule_id)): index
        for index, row in enumerate(combined.itertuples(index=False))
    }
    records = []
    train_targets: set[str] = set()
    train_molecules = set(zip(train.dataset.astype(str), train.molecule_id.astype(str)))
    for dataset in MOLECULEACE_DATASETS:
        profile, _ = remove_equivalent_targets(
            load_profiles(dataset, profile_root), dataset, metadata
        )
        for record in profile.itertuples(index=False):
            key = (dataset, str(record.molecule))
            row = row_lookup.get(key)
            if row is None:
                continue
            target = str(record.target)
            records.append((row, target, float(record.value)))
            if key in train_molecules:
                train_targets.add(target)
    vocabulary = {target: col for col, target in enumerate(sorted(train_targets))}

    fit_values: dict[int, list[float]] = {col: [] for col in vocabulary.values()}
    for row, target, value in records:
        col = vocabulary.get(target)
        if col is not None and row < len(train):
            fit_values[col].append(value)
    location = np.zeros(len(vocabulary), dtype=np.float32)
    scale = np.ones(len(vocabulary), dtype=np.float32)
    for col, values in fit_values.items():
        if values:
            location[col] = float(np.median(values))
            std = float(np.std(values))
            if std >= 1e-6:
                scale[col] = std

    rows: list[int] = []
    cols: list[int] = []
    coverage_values: list[float] = []
    residual_values: list[float] = []
    for row, target, value in records:
        col = vocabulary.get(target)
        if col is None:
            continue
        rows.append(row); cols.append(col); coverage_values.append(1.0)
        residual_values.append((value - float(location[col])) / float(scale[col]))
    shape = (len(combined), len(vocabulary))
    coverage = sparse.csr_matrix((coverage_values, (rows, cols)), shape=shape, dtype=np.float32)
    residual = sparse.csr_matrix((residual_values, (rows, cols)), shape=shape, dtype=np.float32)
    design = sparse.hstack([
        normalize(coverage, norm="l2"), normalize(residual, norm="l2")
    ], format="csr")
    dimension = min(components, design.shape[0] - 1, design.shape[1] - 1)
    reducer = TruncatedSVD(n_components=dimension, n_iter=7, random_state=SEED)
    train_embedding = reducer.fit_transform(design[:len(train)]).astype(np.float32)
    valid_embedding = reducer.transform(design[len(train):]).astype(np.float32)
    mean = train_embedding.mean(0, keepdims=True)
    std = train_embedding.std(0, keepdims=True)
    std[std < 1e-5] = 1.0
    train_embedding = ((train_embedding - mean) / std).astype(np.float32)
    valid_embedding = ((valid_embedding - mean) / std).astype(np.float32)
    audit = {
        "profile_targets": len(vocabulary),
        "profile_svd_components": dimension,
        "profile_explained_variance": float(reducer.explained_variance_ratio_.sum()),
        "train_profile_coverage": float(np.asarray(coverage[:len(train)].sum(1)).ravel().astype(bool).mean()),
        "valid_profile_coverage": float(np.asarray(coverage[len(train):].sum(1)).ravel().astype(bool).mean()),
    }
    return train_embedding, valid_embedding, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--pair-batch-size", type=int, default=1024)
    parser.add_argument("--profile-components", type=int, default=256)
    parser.add_argument("--profile-root", type=Path, default=Path("data/chembl_offtarget_profiles"))
    parser.add_argument("--target-metadata", type=Path,
                        default=Path("data/chembl_target_metadata.jsonl.gz"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/moleculeace_multitask_profile_potential_validation_v1"))
    args = parser.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    train_frames, valid_frames = [], []
    for target_id, dataset in enumerate(MOLECULEACE_DATASETS):
        fit, valid = split_official_train(load_moleculeace(dataset))
        mapping = molecule_mapping(dataset)
        for frame, destination in ((fit, train_frames), (valid, valid_frames)):
            frame = frame.copy()
            frame["target_id"] = target_id
            frame["molecule_id"] = frame.smiles.map(mapping).astype(str)
            destination.append(frame)
    train = pd.concat(train_frames, ignore_index=True)
    valid = pd.concat(valid_frames, ignore_index=True)

    fp = Fingerprints(radius=2, n_bits=1024)
    train_bits = fp.bits(train.smiles.to_numpy()).astype(np.float32)
    valid_bits = fp.bits(valid.smiles.to_numpy()).astype(np.float32)
    train_chemical = np.concatenate([
        train_bits, fp.counts(train.smiles.to_numpy()).astype(np.float32) / 4.0
    ], axis=1)
    valid_chemical = np.concatenate([
        valid_bits, fp.counts(valid.smiles.to_numpy()).astype(np.float32) / 4.0
    ], axis=1)
    train_profile, valid_profile, audit = profile_representation(
        train, valid, args.profile_root, args.target_metadata, args.profile_components
    )
    raw_y = train.target.to_numpy(np.float32)
    y_mean, y_std = float(raw_y.mean()), float(raw_y.std() + 1e-6)
    y = (raw_y - y_mean) / y_std
    pair_features = np.concatenate([train_bits, train_chemical[:, 1024:]], axis=1)
    src, dst, delta, _ = build_pair_edges(
        train, pair_features, train.target_id.to_numpy(np.int64), y
    )
    print(json.dumps({
        "train": len(train), "valid": len(valid), "pair_edges": len(src),
        "epochs": args.epochs, "seed": SEED, **audit,
    }, sort_keys=True), flush=True)

    device = torch.device(args.device)
    model = ProfilePotential(
        train_chemical.shape[1], train_profile.shape[1], len(MOLECULEACE_DATASETS)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    chemical_t = torch.tensor(train_chemical, device=device)
    profile_t = torch.tensor(train_profile, device=device)
    target_t = torch.tensor(train.target_id.to_numpy(np.int64), device=device)
    y_t = torch.tensor(y, device=device)
    src_t = torch.tensor(src, dtype=torch.long, device=device)
    dst_t = torch.tensor(dst, dtype=torch.long, device=device)
    delta_t = torch.tensor(delta, device=device)
    for epoch in range(args.epochs):
        node_order = torch.randperm(len(train), device=device)
        pair_order = torch.randperm(len(src), device=device)
        cursor = 0
        losses = []
        for start in range(0, len(train), args.batch_size):
            index = node_order[start:start + args.batch_size]
            prediction = model(chemical_t[index], profile_t[index], target_t[index])
            loss = F.smooth_l1_loss(prediction, y_t[index])
            pair_index = pair_order[cursor:cursor + args.pair_batch_size]
            cursor += args.pair_batch_size
            if len(pair_index):
                left, right = src_t[pair_index], dst_t[pair_index]
                left_prediction = model(chemical_t[left], profile_t[left], target_t[left])
                right_prediction = model(chemical_t[right], profile_t[right], target_t[right])
                loss = loss + F.smooth_l1_loss(
                    right_prediction - left_prediction, delta_t[pair_index]
                )
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
        for start in range(0, len(valid), args.batch_size):
            stop = start + args.batch_size
            output = model(
                torch.tensor(valid_chemical[start:stop], device=device),
                torch.tensor(valid_profile[start:stop], device=device),
                torch.tensor(valid.target_id.to_numpy(np.int64)[start:stop], device=device),
            )
            predictions.append(output.cpu().numpy())
    valid = valid.copy()
    valid["prediction"] = np.concatenate(predictions) * y_std + y_mean
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset, group in valid.groupby("dataset", sort=False):
        metric = regression_metrics(
            group.target.to_numpy(float), group.prediction.to_numpy(float),
            group.cliff_mol.to_numpy(bool),
        )
        rows.append({"dataset": dataset, "model": "multitask_profile_potential", **metric})
        group[["dataset", "smiles", "target", "cliff_mol", "prediction"]].to_csv(
            args.output_dir / f"{dataset}.validation.predictions.csv", index=False
        )
    summary = pd.DataFrame(rows).sort_values("dataset")
    summary.to_csv(args.output_dir / "summary.validation.csv", index=False)
    aggregate = {
        "model": "multitask_profile_potential", "datasets": len(summary),
        "macro_rmse": float(summary.rmse.mean()),
        "macro_cliff_rmse": float(summary.cliff_rmse.mean()),
        "macro_noncliff_rmse": float(summary.noncliff_rmse.mean()),
        "selection": "train-only fixed split", "ensemble": False,
        "seed": SEED, **audit,
    }
    (args.output_dir / "aggregate.validation.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    )
    torch.save({"model": model.state_dict(), "aggregate": aggregate},
               args.output_dir / "model.pt")
    print(json.dumps(aggregate, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
