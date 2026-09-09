from __future__ import annotations

"""Scaffold-disjoint cold-start evaluation of the frozen TAPER recipe.

The split is performed over the complete released MoleculeACE table.  Molecules
sharing a Bemis--Murcko scaffold never occur in both fitting and held-out sets.
All profile construction remains target-excluded and fit-only as implemented by
``validate_moleculeace_profile_alignment_kernel.run_one``.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from sklearn.model_selection import GroupShuffleSplit

from molcliff.data import MOLECULEACE_DATASETS
from validate_moleculeace_profile_alignment_kernel import (
    load_bindingdb_profiles, load_sequences, load_sequence_kmers,
    load_target_metadata, run_one,
)


def scaffold_split(frame: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    scaffolds = []
    for smiles in frame.smiles.astype(str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        # Acyclic molecules have an empty Murcko scaffold; retaining their
        # canonical SMILES as the group prevents accidental identity overlap.
        scaffolds.append(scaffold or smiles)
    indices = np.arange(len(frame))
    split = GroupShuffleSplit(n_splits=1, test_size=.20, random_state=seed)
    fit_idx, test_idx = next(split.split(indices, groups=np.asarray(scaffolds)))
    fit = frame.iloc[np.sort(fit_idx)].copy().reset_index(drop=True)
    test = frame.iloc[np.sort(test_idx)].copy().reset_index(drop=True)
    overlap = set(np.asarray(scaffolds)[fit_idx]).intersection(np.asarray(scaffolds)[test_idx])
    if overlap:
        raise RuntimeError("Scaffold overlap detected")
    return fit, test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(MOLECULEACE_DATASETS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/taper_scaffold_split_seed42"))
    parser.add_argument("--profile-root", type=Path, default=Path("data/chembl_offtarget_profiles"))
    parser.add_argument("--target-metadata", type=Path, default=Path("data/chembl_target_metadata.jsonl.gz"))
    parser.add_argument("--sequence-cache", type=Path, default=Path("data/uniprot_target_sequences.json.gz"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_target_metadata(args.target_metadata)
    sequence_kmers = load_sequence_kmers(args.sequence_cache) if args.sequence_cache.exists() else {}
    sequences = load_sequences(args.sequence_cache) if args.sequence_cache.exists() else {}
    splitter = lambda frame: scaffold_split(frame, args.seed)
    rows: list[dict] = []
    pred_dir = args.output_dir / "predictions"
    for dataset in args.datasets:
        print(f"[scaffold] {dataset}", flush=True)
        rows.extend(run_one(
            dataset, args.profile_root, metadata, sequence_kmers, sequences,
            qualitative_root=Path("data/chembl_qualitative_profiles"),
            splitter=splitter,
            candidate_names={"collision_free_binary_count_kernel", "responsekernel_final"},
            # Only archived pre-existing training-pool profiles are used. The
            # absence of a test-profile cache is itself the strict cold-start
            # condition: held-out scaffolds may receive zero response memory.
            profile_split="train", predictions_dir=pred_dir,
        ))
    summary = pd.DataFrame(rows).sort_values(["model", "dataset"])
    summary.to_csv(args.output_dir / "summary.scaffold.csv", index=False)
    macro = summary.groupby("model")[["rmse", "cliff_rmse", "noncliff_rmse"]].mean()
    payload = {"seed": args.seed, "datasets": int(summary.dataset.nunique()),
               "models": {name: {k: float(v) for k, v in row.items()} for name, row in macro.iterrows()},
               "protocol": "single 80/20 Bemis--Murcko GroupShuffleSplit over each complete released target table; target-excluded archived training-pool profiles; profile relevance fit on held-in molecules only"}
    (args.output_dir / "aggregate.scaffold.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
