from __future__ import annotations

"""Five-fold train-only audit of one frozen bioactivity-profile kernel.

This is deliberately an audit rather than another candidate search: it compares
the collision-free chemistry baseline with the single profile candidate frozen
after the original seed-42 holdout study.  Every fold is constructed only from
the official training pool; official test rows and labels are never loaded into
either fitting or model selection.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from molcliff.data import MOLECULEACE_DATASETS
from validate_moleculeace_profile_alignment_kernel import (
    load_sequence_kmers,
    load_sequences,
    load_target_metadata,
    run_one,
)


BASELINE = "collision_free_binary_count_kernel"
CANDIDATE = "responsekernel_final"


def fold_splitter(fold: int, seed: int = 42):
    def split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        pool = frame[frame.split.eq("train")].reset_index(drop=True)
        indices = np.arange(len(pool))
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        train_index, valid_index = list(
            folds.split(indices, pool.cliff_mol.to_numpy())
        )[fold]
        return (
            pool.iloc[np.sort(train_index)].reset_index(drop=True),
            pool.iloc[np.sort(valid_index)].reset_index(drop=True),
        )

    return split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(MOLECULEACE_DATASETS))
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models", nargs="+", default=[BASELINE, CANDIDATE],
        help="Frozen candidate names evaluated on every requested fold.",
    )
    parser.add_argument("--primary-model", default=CANDIDATE)
    parser.add_argument("--profile-root", type=Path, default=Path("data/chembl_offtarget_profiles"))
    parser.add_argument("--target-metadata", type=Path, default=Path("data/chembl_target_metadata.jsonl.gz"))
    parser.add_argument("--sequence-cache", type=Path, default=Path("data/uniprot_target_sequences.json.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/moleculeace_profile_kernel_cv5_combined_v1"))
    args = parser.parse_args()

    if any(fold < 0 or fold >= 5 for fold in args.folds):
        raise ValueError("fold indices must be in [0, 4]")
    metadata = load_target_metadata(args.target_metadata)
    kmers = load_sequence_kmers(args.sequence_cache)
    sequences = load_sequences(args.sequence_cache)
    rows = []
    for dataset in args.datasets:
        for fold in args.folds:
            fold_rows = run_one(
                dataset,
                args.profile_root,
                metadata,
                kmers,
                sequences,
                None,
                None,
                None,
                splitter=fold_splitter(fold, args.seed),
                candidate_names=set(args.models),
            )
            for row in fold_rows:
                row["fold"] = fold
                row["seed"] = args.seed
            rows.extend(fold_rows)
            print(json.dumps({
                "dataset": dataset,
                "fold": fold,
                "metrics": {
                    row["model"]: {key: row[key] for key in ("rmse", "cliff_rmse", "noncliff_rmse")}
                    for row in fold_rows
                },
            }, sort_keys=True), flush=True)

    summary = pd.DataFrame(rows).sort_values(["model", "dataset", "fold"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "summary.cv5.csv", index=False)
    macro = summary.groupby("model")[["rmse", "cliff_rmse", "noncliff_rmse"]].mean()
    if BASELINE not in macro.index or args.primary_model not in macro.index:
        raise ValueError("models must include the baseline and primary model")
    base, candidate = macro.loc[BASELINE], macro.loc[args.primary_model]
    paired = summary.pivot(index=["dataset", "fold"], columns="model", values=["rmse", "cliff_rmse", "noncliff_rmse"])
    payload = {
        "protocol": f"official-train-only stratified five-fold CV; seed {args.seed}; official test untouched",
        "seed": args.seed,
        "primary_model": args.primary_model,
        "datasets": int(summary.dataset.nunique()),
        "folds": sorted(map(int, summary.fold.unique())),
        "models": {
            name: {key: float(value) for key, value in row.items()}
            for name, row in macro.iterrows()
        },
        "relative_gain_vs_collision_free": {
            key: float((base[key] - candidate[key]) / base[key])
            for key in ("rmse", "cliff_rmse", "noncliff_rmse")
        },
        "paired_wins": {
            key: int((paired[key][BASELINE] > paired[key][args.primary_model]).sum())
            for key in ("rmse", "cliff_rmse", "noncliff_rmse")
        },
        "paired_comparisons": int(len(paired)),
        "official_test_evaluations": 0,
        "candidate_selection": (
            "frozen candidate set evaluated uniformly; no fold-specific or target-specific selection"
        ),
    }
    (args.output_dir / "aggregate.cv5.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
