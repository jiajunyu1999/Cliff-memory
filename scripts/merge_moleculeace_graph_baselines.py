from __future__ import annotations

"""Merge independently checkpointed MoleculeACE graph baseline runs."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs/baseline_runs"
MODELS = ("chemprop", "gine_residual", "gine_nodenorm", "gine_pairnorm", "graphcliff")

def main() -> None:
    pieces = []
    for model in MODELS:
        path = RUNS / f"moleculeace_{model}_full.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if set(frame.model) != {model} or len(frame) != 150 or frame.dataset.nunique() != 30:
            raise RuntimeError(f"{model}: expected 150 rows and 30 targets, got {len(frame)} rows/{frame.dataset.nunique()} targets")
        if frame.groupby("dataset").fold.nunique().ne(5).any():
            raise RuntimeError(f"{model}: incomplete five-fold coverage")
        if frame[["rmse", "cliff_rmse", "noncliff_rmse"]].isna().any().any():
            raise RuntimeError(f"{model}: missing metric")
        pieces.append(frame)
    result = pd.concat(pieces, ignore_index=True)
    result.to_csv(RUNS / "moleculeace_graph_full.csv", index=False)
    print(result.groupby("model").size().to_string())

if __name__ == "__main__":
    main()
