from __future__ import annotations

"""Aggregate independently executed MoleculeACE profile-kernel CV parts."""

import argparse
import json
from pathlib import Path

import pandas as pd

from crossvalidate_moleculeace_profile_kernel import BASELINE, CANDIDATE


METRICS = ("rmse", "cliff_rmse", "noncliff_rmse")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parts-dir", type=Path,
        default=Path("outputs/moleculeace_profile_kernel_cv5_v1_parts"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/moleculeace_profile_kernel_cv5_v1"),
    )
    parser.add_argument("--models", nargs="+", default=[BASELINE, CANDIDATE])
    parser.add_argument("--primary-model", default=CANDIDATE)
    parser.add_argument("--expected-targets", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    files = sorted(args.parts_dir.glob("*/summary.cv5.csv"))
    if len(files) != args.expected_targets:
        raise RuntimeError(
            f"expected {args.expected_targets} completed target files, found {len(files)}"
        )
    summary = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    counts = summary.groupby(["dataset", "model"]).size()
    if set(summary.model) != set(args.models) or not (counts == 5).all():
        raise RuntimeError("each target must contain five folds for every requested model")

    target_macro = (
        summary.groupby(["dataset", "model"])[list(METRICS)].mean()
        .unstack("model")
    )
    per_target = pd.DataFrame(index=target_macro.index)
    candidate_name = args.primary_model
    for metric in METRICS:
        per_target[f"{BASELINE}_{metric}"] = target_macro[metric][BASELINE]
        per_target[f"{candidate_name}_{metric}"] = target_macro[metric][candidate_name]
        per_target[f"relative_gain_{metric}"] = (
            target_macro[metric][BASELINE] - target_macro[metric][candidate_name]
        ) / target_macro[metric][BASELINE]
    per_target = per_target.reset_index()

    macro = summary.groupby("model")[list(METRICS)].mean()
    base, candidate = macro.loc[BASELINE], macro.loc[candidate_name]
    paired = summary.pivot(
        index=["dataset", "fold"], columns="model", values=list(METRICS)
    )
    payload = {
        "protocol": (
            f"official-train-only stratified five-fold CV; seed {args.seed}; "
            "frozen model set; official test untouched"
        ),
        "datasets": int(summary.dataset.nunique()),
        "folds_per_dataset": 5,
        "seed": args.seed,
        "primary_model": candidate_name,
        "paired_comparisons": int(len(paired)),
        "models": {
            name: {key: float(value) for key, value in row.items()}
            for name, row in macro.iterrows()
        },
        "relative_gain_vs_collision_free": {
            metric: float((base[metric] - candidate[metric]) / base[metric])
            for metric in METRICS
        },
        "paired_fold_wins": {
            metric: int((paired[metric][BASELINE] > paired[metric][candidate_name]).sum())
            for metric in METRICS
        },
        "target_macro_wins": {
            metric: int((per_target[f"relative_gain_{metric}"] > 0).sum())
            for metric in METRICS
        },
        "passes_30_percent_gate": {
            metric: bool((base[metric] - candidate[metric]) / base[metric] >= 0.30)
            for metric in METRICS
        },
        "official_test_evaluations": 0,
        "candidate_selection": "no CV-fold, hyperparameter, or seed selection",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.sort_values(["dataset", "fold", "model"]).to_csv(
        args.output_dir / "summary.cv5.csv", index=False
    )
    per_target.sort_values("dataset").to_csv(
        args.output_dir / "per_target.cv5.csv", index=False
    )
    (args.output_dir / "aggregate.cv5.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
