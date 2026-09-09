from __future__ import annotations

"""One frozen official-test evaluation of the profile-interaction kernel."""

import argparse
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from molcliff.data import MOLECULEACE_DATASETS
from validate_moleculeace_profile_alignment_kernel import (
    load_sequence_kmers,
    load_sequences,
    load_target_metadata,
    run_one,
)


CANDIDATE = "responsekernel_final"


def official_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame[frame.split.eq("train")].reset_index(drop=True)
    test = frame[frame.split.eq("test")].reset_index(drop=True)
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(MOLECULEACE_DATASETS))
    parser.add_argument("--train-profile-root", type=Path, default=Path("data/chembl_offtarget_profiles"))
    parser.add_argument("--test-profile-root", type=Path, default=Path("data/chembl_offtarget_profiles_test"))
    parser.add_argument("--target-metadata", type=Path, default=Path("data/chembl_target_metadata.jsonl.gz"))
    parser.add_argument("--sequence-cache", type=Path, default=Path("data/uniprot_target_sequences.json.gz"))
    parser.add_argument("--model", default=CANDIDATE,
                        help="Single train-only-frozen candidate to evaluate.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/moleculeace_profile_kernel_test_v1"))
    args = parser.parse_args()
    metadata = load_target_metadata(args.target_metadata)
    sequence_kmers = load_sequence_kmers(args.sequence_cache) if args.sequence_cache.exists() else {}
    sequences = load_sequences(args.sequence_cache) if args.sequence_cache.exists() else {}
    rows = []
    with tempfile.TemporaryDirectory(prefix="moleculeace_profile_lockbox_") as temp:
        root = Path(temp)
        for dataset in args.datasets:
            for part, source in (("train", args.train_profile_root), ("test", args.test_profile_root)):
                src = source / f"{dataset}.{part}_offtarget.jsonl.gz"
                if not src.exists():
                    raise FileNotFoundError(src)
                os.symlink(src.resolve(), root / src.name)
            result = run_one(
                dataset, root, metadata,
                sequence_kmers, sequences, None, None, None,
                splitter=official_split,
                candidate_names={args.model},
                profile_split="both",
                predictions_dir=args.output_dir / "predictions",
            )
            rows.extend(result)
            print(json.dumps(result, sort_keys=True), flush=True)
    summary = pd.DataFrame(rows).sort_values("dataset")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "summary.test.csv", index=False)
    payload = {
        "protocol": "single official-test lockbox evaluation; fit uses official train; test off-target profiles only; exact and biologically equivalent benchmark targets removed; no test labels used for selection",
        "datasets": int(summary.dataset.nunique()),
        "model": args.model,
        "macro_rmse": float(summary.rmse.mean()),
        "macro_cliff_rmse": float(summary.cliff_rmse.mean()),
        "macro_noncliff_rmse": float(summary.noncliff_rmse.mean()),
        "test_rows": int(summary.n_test.sum()),
        "test_cliff_rows": int(summary.n_cliff.sum()),
        "candidate_frozen_before_test": True,
    }
    (args.output_dir / "aggregate.test.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
