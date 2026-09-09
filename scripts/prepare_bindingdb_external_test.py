from __future__ import annotations

"""Build exact-protein ChEMBL-to-BindingDB external assay tables locally."""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from molcliff.data import MOLECULEACE_DATASETS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=Path("../training_data_860.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("data/bindingdb_system_metadata.jsonl.gz"))
    parser.add_argument("--chembl-meta", type=Path, default=Path("data/chembl_target_metadata.jsonl.gz"))
    parser.add_argument("--profile-root", type=Path, default=Path("data/chembl_offtarget_profiles"))
    parser.add_argument("--output", type=Path, default=Path("data/bindingdb_external_exact_protein.csv"))
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()
    dataset_target = {}
    for dataset in MOLECULEACE_DATASETS:
        metadata_path = args.profile_root / f"{dataset}.metadata.json"
        if metadata_path.exists():
            payload = json.loads(metadata_path.read_text())
            dataset_target[dataset] = str(payload["benchmark_target_excluded"])
        else:
            # MoleculeACE dataset names encode their ChEMBL target identity.
            dataset_target[dataset] = dataset.rsplit("_", 1)[0]
    chembl = {}
    with gzip.open(args.chembl_meta, "rt") as handle:
        for line in handle:
            x = json.loads(line); chembl[str(x["target_chembl_id"])] = x
    dataset_accessions = {}
    for dataset, target in dataset_target.items():
        accessions = set()
        for component in chembl.get(target, {}).get("target_components", []):
            if component.get("accession"): accessions.add(str(component["accession"]).upper())
        dataset_accessions[dataset] = accessions
    system_to_datasets: dict[str, list[str]] = defaultdict(list)
    with gzip.open(args.metadata, "rt") as handle:
        for line in handle:
            x = json.loads(line); system_acc = set(map(str.upper, x.get("accessions", [])))
            for dataset, accessions in dataset_accessions.items():
                if accessions & system_acc: system_to_datasets[str(x["system"])].append(dataset)
    records = []
    usecols = ["lig1", "lig2", "smile1", "smile2", "Label1", "Label2"]
    for chunk_i, chunk in enumerate(pd.read_csv(args.pairs, usecols=usecols, chunksize=args.chunksize), 1):
        systems1 = chunk.lig1.astype(str).str.rsplit("_", n=1).str[0]
        systems2 = chunk.lig2.astype(str).str.rsplit("_", n=1).str[0]
        for systems, smiles, labels in [(systems1, chunk.smile1, chunk.Label1),
                                        (systems2, chunk.smile2, chunk.Label2)]:
            mask = systems.isin(system_to_datasets)
            if not mask.any(): continue
            for system, smi, value in zip(systems[mask], smiles[mask], labels[mask], strict=True):
                if pd.isna(value): continue
                for dataset in system_to_datasets[str(system)]:
                    records.append((dataset, str(system), str(smi), float(value)))
        if chunk_i % 10 == 0:
            print(json.dumps({"chunks": chunk_i, "raw_records": len(records)}), flush=True)
    frame = pd.DataFrame(records, columns=["dataset", "system", "smiles", "target"])
    # Collapse repeat pair appearances and retain only multi-ligand systems.
    frame = frame.groupby(["dataset", "system", "smiles"], as_index=False).target.median()
    counts = frame.groupby(["dataset", "system"]).smiles.nunique().rename("n_molecules").reset_index()
    frame = frame.merge(counts, on=["dataset", "system"])
    frame.to_csv(args.output, index=False)
    summary = frame.groupby("dataset").agg(systems=("system", "nunique"), molecules=("smiles", "nunique"), records=("target", "size")).reset_index()
    summary.to_csv(args.output.with_suffix(".summary.csv"), index=False)
    print(json.dumps({"datasets": int(summary.shape[0]), "rows": int(frame.shape[0]), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
