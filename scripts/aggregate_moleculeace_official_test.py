from __future__ import annotations

"""Aggregate independently executed per-target official-test evaluations."""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-targets", type=int, default=30)
    args = parser.parse_args()
    summaries = sorted(args.parts_dir.glob("*/summary.test.csv"))
    if len(summaries) != args.expected_targets:
        raise RuntimeError(f"expected {args.expected_targets} summaries, found {len(summaries)}")
    frame = pd.concat([pd.read_csv(path) for path in summaries], ignore_index=True)
    if frame.dataset.nunique() != args.expected_targets:
        raise RuntimeError("duplicate or missing datasets in official-test parts")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = args.output_dir / "predictions"
    predictions.mkdir(exist_ok=True)
    for source in args.parts_dir.glob("*/predictions/*.csv"):
        shutil.copy2(source, predictions / source.name)
    frame.sort_values(["dataset", "model"]).to_csv(
        args.output_dir / "summary.test.csv", index=False
    )
    payload = {
        "protocol": "single frozen official-test evaluation; fit uses official train; external profiles are leakage-filtered; no test labels used for model selection",
        "datasets": int(frame.dataset.nunique()),
        "model": str(frame.model.unique()[0]),
        "macro_rmse": float(frame.rmse.mean()),
        "macro_cliff_rmse": float(frame.cliff_rmse.mean()),
        "macro_noncliff_rmse": float(frame.noncliff_rmse.mean()),
        "test_rows": int(frame.n_test.sum()),
        "test_cliff_rows": int(frame.n_cliff.sum()),
        "candidate_frozen_before_test": True,
    }
    (args.output_dir / "aggregate.test.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
