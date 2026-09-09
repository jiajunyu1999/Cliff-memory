from __future__ import annotations

"""Standardize the eight GraphCliff external ChEMBL assays with molecule IDs."""

import json
from pathlib import Path

import pandas as pd


SOURCE = Path("third_party/GraphCliff/benchmark_data")
OUTPUT = Path("data/expanded_cliff_regression")
DATASETS = {
    "IDO1": "CHEMBL4685_IC50",
    "PHGDH": "CHEMBL2311243_IC50",
    "PKCi": "CHEMBL2598_IC50",
    "PLK1": "CHEMBL3024_IC50",
    "RIP2": "CHEMBL5014_IC50",
    "RXFP1": "CHEMBL1293316_AC50",
    "USP7": "CHEMBL2157850_IC50",
    "mGluR2": "CHEMBL5137_IC50",
}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "raw").mkdir(parents=True, exist_ok=True)
    summary = []
    for source_name, dataset in DATASETS.items():
        frame = pd.read_csv(SOURCE / f"{source_name}.csv")
        raw = pd.read_csv(SOURCE / "raw" / f"{source_name}.csv")
        mapping = raw[["Smiles", "Molecule ChEMBL ID"]].drop_duplicates("Smiles")
        mapping = mapping.rename(columns={"Smiles": "smiles", "Molecule ChEMBL ID": "chembl_id"})
        if set(frame.smiles) != set(mapping.smiles):
            raise RuntimeError(f"{source_name}: processed/raw SMILES mismatch")
        standardized = frame[["smiles", "y", "cliff_mol", "split"]].copy()
        standardized["dataset"] = dataset
        standardized.to_csv(OUTPUT / f"{dataset}.csv", index=False)
        mapping.to_csv(OUTPUT / "raw" / f"{dataset}.csv", index=False)
        summary.append({
            "source": source_name,
            "dataset": dataset,
            "molecules": int(len(standardized)),
            "train": int(standardized.split.eq("train").sum()),
            "test": int(standardized.split.eq("test").sum()),
            "cliff_molecules": int(standardized.cliff_mol.astype(bool).sum()),
        })
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
