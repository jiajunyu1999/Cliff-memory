from __future__ import annotations

"""Fit the frozen combined positive-signal model on each full MoleculeACE train pool."""

import argparse
import json
from pathlib import Path

import pandas as pd

from molcliff.data import MOLECULEACE_DATASETS, load_moleculeace
from validate_moleculeace_profile_alignment_kernel import (
    load_sequence_kmers,
    load_sequences,
    load_target_metadata,
    run_one,
)


CANDIDATE = "responsekernel_final"


def full_train(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool = frame[frame.split.eq("train")].reset_index(drop=True)
    return pool, pool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(MOLECULEACE_DATASETS))
    parser.add_argument("--profile-root", type=Path, default=Path("data/chembl_offtarget_profiles"))
    parser.add_argument("--target-metadata", type=Path, default=Path("data/chembl_target_metadata.jsonl.gz"))
    parser.add_argument("--sequence-cache", type=Path, default=Path("data/uniprot_target_sequences.json.gz"))
    parser.add_argument("--model", default=CANDIDATE)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/moleculeace_combined_model_v1"))
    args = parser.parse_args()
    metadata = load_target_metadata(args.target_metadata)
    kmers = load_sequence_kmers(args.sequence_cache) if args.sequence_cache.exists() else {}
    sequences = load_sequences(args.sequence_cache) if args.sequence_cache.exists() else {}
    rows = []
    for dataset in args.datasets:
        result = run_one(
            dataset, args.profile_root, metadata, kmers, sequences,
            None, None, None, splitter=full_train,
            candidate_names={args.model}, artifacts_dir=args.output_dir,
        )
        rows.extend(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    summary = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "train_metrics.csv", index=False)
    (args.output_dir / "recipe.json").write_text(json.dumps({
        "model": args.model,
        "fit_split": "all official train rows per target",
        "test_labels_used": False,
        "svr": {"C": 10.0, "epsilon": 0.1, "kernel": "precomputed"},
        "kernel_specification": (
            {"chemistry": 10, "response_without_biological_neighborhood": 8}
            if args.model == "responsekernel_final" else
            {"chemistry": 10, "profile_block": 8, "physical_local": 4,
             "chemistry_x_susceptibility": 1}
        ),
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"datasets": int(summary.dataset.nunique()), "output_dir": str(args.output_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
