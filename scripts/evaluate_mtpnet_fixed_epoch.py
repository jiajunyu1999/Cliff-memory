from __future__ import annotations

"""Re-score MTPNet's unselected final epoch on MoleculeACE macro metrics."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from molcliff.metrics import regression_metrics


ROOT = Path(__file__).resolve().parents[1]
MTP = ROOT / "third_party" / "MTPNet"
sys.path.insert(0, str(MTP))

from configs import get_cfg_defaults  # noqa: E402
from dataloader import ACDataset  # noqa: E402
from models import ESI_Pred  # noqa: E402
from utils import graph_collate_func  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/moleculeace_mtpnet_epoch50_audit"))
    args = parser.parse_args()
    torch.manual_seed(42)
    cfg = get_cfg_defaults()
    cfg.merge_from_file(str(MTP / "configs" / "model" / "MTPNet.yaml"))
    cfg.merge_from_file(str(MTP / "configs" / "data" / "MolACE.yaml"))
    frame = pd.read_csv(MTP / "data" / "MoleculeACE_saprot.csv")
    frame = frame[frame.split.eq("test")].reset_index(drop=True)
    frame["SA_Protein_Path"] = frame.Dataset.map(
        lambda name: str(MTP / "data" / "saprot_feat" / f"{name}.pt"))
    dataset = ACDataset(frame.index.to_numpy(), frame, task="regression")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=graph_collate_func)
    model = ESI_Pred(**cfg)
    weight = MTP / "result" / "ALL" / "MTPNet" / "model_epoch_50.pth"
    model.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True))
    device = torch.device(args.device); model.to(device).eval()
    predictions = []
    with torch.no_grad():
        for step, (drug, protein, _label, protein_mask) in enumerate(loader):
            _, _, score = model(drug.to(device), protein.to(device), protein_mask.to(device))
            predictions.append(score.squeeze(1).cpu().numpy())
            if (step + 1) % 25 == 0:
                print(json.dumps({"batches": step + 1, "rows": min((step + 1) * args.batch_size, len(frame))}), flush=True)
    frame["prediction"] = np.concatenate(predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, group in frame.groupby("Dataset", sort=False):
        metric = regression_metrics(group.Score.to_numpy(float), group.prediction.to_numpy(float),
                                    group.cliff_mol.to_numpy(bool))
        metric.update({"dataset": name, "model": "mtpnet_released_epoch50", "seed": 42})
        rows.append(metric)
        group[["Dataset", "SMILES", "Score", "cliff_mol", "split", "prediction"]].to_csv(
            args.output_dir / f"{name}.mtpnet_epoch50.predictions.csv", index=False)
    summary = pd.DataFrame(rows).sort_values("dataset")
    summary.to_csv(args.output_dir / "summary.mtpnet_epoch50.csv", index=False)
    aggregate = {"model": "mtpnet_released_epoch50", "checkpoint_selection": "fixed_final_epoch",
                 "datasets": len(summary), "macro_rmse": float(summary.rmse.mean()),
                 "macro_cliff_rmse": float(summary.cliff_rmse.mean()),
                 "macro_noncliff_rmse": float(summary.noncliff_rmse.mean())}
    (args.output_dir / "aggregate.mtpnet_epoch50.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(aggregate, sort_keys=True))


if __name__ == "__main__":
    main()
