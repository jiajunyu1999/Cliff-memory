from __future__ import annotations

"""A single antisymmetric, target-conditioned MMP retrieval model."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from molcliff.data import Fingerprints, MOLECULEACE_DATASETS, load_moleculeace, tanimoto_matrix
from molcliff.metrics import regression_metrics
from molcliff.mmp_action import single_cuts


class ConditionalMMP(nn.Module):
    def __init__(self, n_bits: int, n_targets: int) -> None:
        super().__init__()
        self.core = nn.Sequential(nn.Linear(n_bits, 256), nn.LayerNorm(256), nn.SiLU())
        self.action = nn.Linear(n_bits, 256, bias=False)
        self.target = nn.Embedding(n_targets, 64)
        self.effect = nn.Sequential(
            nn.Linear(256 + 256 + 64, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1))

    def _directed(self, common: torch.Tensor, action: torch.Tensor,
                  target: torch.Tensor) -> torch.Tensor:
        z = torch.cat([self.core(common), self.action(action), self.target(target)], dim=1)
        return self.effect(z).squeeze(1)

    def forward(self, common: torch.Tensor, action: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        # Antisymmetry is architectural, not a penalty.
        return 0.5 * (self._directed(common, action, target)
                      - self._directed(common, -action, target))


def _target_pairs(smiles: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, list[int]]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, smi in enumerate(smiles):
        for cut in single_cuts(smi):
            groups[cut.core].append(index)
    pairs: set[tuple[int, int]] = set()
    for core, members in groups.items():
        unique = sorted(set(members))
        groups[core] = unique
        for pos, left in enumerate(unique):
            for right in unique[pos + 1:]:
                pairs.add((left, right))
    ordered = sorted(pairs)
    return (np.asarray([p[0] for p in ordered], dtype=np.int64),
            np.asarray([p[1] for p in ordered], dtype=np.int64), groups)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/moleculeace_conditional_mmp_v1"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)

    fp = Fingerprints(radius=2, n_bits=1024)
    train_frames, test_frames = [], []
    pair_src, pair_dst, pair_target = [], [], []
    train_groups: list[dict[str, list[int]]] = []
    offset = 0
    for target_id, dataset in enumerate(MOLECULEACE_DATASETS):
        frame = load_moleculeace(dataset)
        train = frame[frame.split.eq("train")].reset_index(drop=True).copy()
        test = frame[frame.split.eq("test")].reset_index(drop=True).copy()
        train["target_id"] = target_id; test["target_id"] = target_id
        left, right, groups = _target_pairs(train.smiles.tolist())
        pair_src.append(left + offset); pair_dst.append(right + offset)
        pair_target.append(np.full(len(left), target_id, dtype=np.int64))
        train_groups.append(groups)
        train_frames.append(train); test_frames.append(test)
        offset += len(train)
    train = pd.concat(train_frames, ignore_index=True)
    test = pd.concat(test_frames, ignore_index=True)
    counts = fp.counts(train.smiles.to_numpy()).astype(np.float32)
    bits = fp.bits(train.smiles.to_numpy()).astype(np.float32)
    test_counts = fp.counts(test.smiles.to_numpy()).astype(np.float32)
    test_bits = fp.bits(test.smiles.to_numpy()).astype(np.float32)
    y = train.target.to_numpy(dtype=np.float32)
    y_scale = float(y.std() + 1e-6)
    src = np.concatenate(pair_src); dst = np.concatenate(pair_dst)
    target_ids = np.concatenate(pair_target)
    delta = (y[dst] - y[src]) / y_scale
    print(json.dumps({"train": len(train), "test": len(test), "exact_mmp_pairs": len(src)}), flush=True)

    device = torch.device(args.device)
    model = ConditionalMMP(1024, len(MOLECULEACE_DATASETS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    counts_t = torch.tensor(counts, device=device)
    y_delta_t = torch.tensor(delta, device=device)
    src_t = torch.tensor(src, device=device); dst_t = torch.tensor(dst, device=device)
    targets_t = torch.tensor(target_ids, device=device)
    for epoch in range(args.epochs):
        order = torch.randperm(len(src_t), device=device)
        losses = []
        for start in range(0, len(order), args.batch_size):
            p = order[start:start + args.batch_size]
            a, b = src_t[p], dst_t[p]
            common = torch.minimum(counts_t[a], counts_t[b]) / 4.0
            action = (counts_t[b] - counts_t[a]) / 4.0
            pred = model(common, action, targets_t[p])
            loss = F.smooth_l1_loss(pred, y_delta_t[p])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(json.dumps({"epoch": epoch + 1, "loss": float(np.mean(losses))}), flush=True)

    # One nearest anchor supplies one prediction.  Exact retained-core matches
    # take precedence; if absent, the nearest training molecule is still passed
    # through the same action model (there is no separately trained fallback).
    prediction = np.empty(len(test), dtype=np.float32)
    exact = np.zeros(len(test), dtype=bool)
    model.eval()
    train_offset = test_offset = 0
    with torch.no_grad():
        for target_id, (train_frame, test_frame, groups) in enumerate(
                zip(train_frames, test_frames, train_groups, strict=True)):
            n_train, n_test = len(train_frame), len(test_frame)
            local_train_bits = bits[train_offset:train_offset + n_train]
            local_test_bits = test_bits[test_offset:test_offset + n_test]
            similarity = tanimoto_matrix(local_test_bits, local_train_bits)
            anchors = np.empty(n_test, dtype=np.int64)
            for qi, smi in enumerate(test_frame.smiles.tolist()):
                candidates: set[int] = set()
                for cut in single_cuts(smi):
                    candidates.update(groups.get(cut.core, ()))
                if candidates:
                    cand = np.asarray(sorted(candidates), dtype=np.int64)
                    anchors[qi] = int(cand[np.argmax(similarity[qi, cand])])
                    exact[test_offset + qi] = True
                else:
                    anchors[qi] = int(np.argmax(similarity[qi]))
            global_anchor = anchors + train_offset
            qidx = np.arange(test_offset, test_offset + n_test)
            common = np.minimum(counts[global_anchor], test_counts[qidx]) / 4.0
            action = (test_counts[qidx] - counts[global_anchor]) / 4.0
            out = model(torch.tensor(common, device=device), torch.tensor(action, device=device),
                        torch.full((n_test,), target_id, dtype=torch.long, device=device))
            prediction[qidx] = y[global_anchor] + out.cpu().numpy() * y_scale
            train_offset += n_train; test_offset += n_test

    args.output_dir.mkdir(parents=True, exist_ok=True)
    test = test.copy(); test["prediction"] = prediction; test["exact_core_anchor"] = exact
    rows = []
    for dataset, group in test.groupby("dataset", sort=False):
        metric = regression_metrics(group.target.to_numpy(float), group.prediction.to_numpy(float),
                                    group.cliff_mol.to_numpy(bool))
        metric.update({"dataset": dataset, "model": "conditional_mmp", "seed": 42,
                       "exact_core_coverage": float(group.exact_core_anchor.mean())})
        rows.append(metric)
        group[["dataset", "smiles", "target", "cliff_mol", "split", "prediction",
               "exact_core_anchor"]].to_csv(
            args.output_dir / f"{dataset}.conditional_mmp.predictions.csv", index=False)
    summary = pd.DataFrame(rows).sort_values("dataset")
    summary.to_csv(args.output_dir / "summary.conditional_mmp.csv", index=False)
    aggregate = {"model": "conditional_mmp", "datasets": len(summary), "seed": 42,
                 "epochs": args.epochs, "selection": "none", "ensemble": False,
                 "exact_mmp_pairs": len(src), "exact_core_coverage": float(exact.mean()),
                 "macro_rmse": float(summary.rmse.mean()),
                 "macro_cliff_rmse": float(summary.cliff_rmse.mean()),
                 "macro_noncliff_rmse": float(summary.noncliff_rmse.mean())}
    (args.output_dir / "aggregate.conditional_mmp.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(aggregate, sort_keys=True))


if __name__ == "__main__":
    main()
