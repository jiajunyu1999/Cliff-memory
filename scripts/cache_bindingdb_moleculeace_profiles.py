from __future__ import annotations

"""Extract exact MoleculeACE molecule activities from the local BindingDB cache."""

import argparse
import gzip
import json
from pathlib import Path

import pandas as pd

from molcliff.data import MOLECULEACE_DATASETS, load_moleculeace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("../training_data_860.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/bindingdb_moleculeace_profiles.jsonl.gz"))
    parser.add_argument("--chunksize", type=int, default=250_000)
    args = parser.parse_args()
    wanted = {
        str(smiles) for dataset in MOLECULEACE_DATASETS
        for smiles in load_moleculeace(dataset).smiles
    }
    records: dict[tuple[str, str], list[float]] = {}
    usecols = ["lig1", "lig2", "smile1", "smile2", "Label1", "Label2"]
    rows = 0
    for chunk in pd.read_csv(args.input, usecols=usecols, chunksize=args.chunksize):
        rows += len(chunk)
        for ligand_col, smiles_col, value_col in (
            ("lig1", "smile1", "Label1"), ("lig2", "smile2", "Label2")
        ):
            subset = chunk[chunk[smiles_col].isin(wanted)]
            for ligand, smiles, value in subset[
                [ligand_col, smiles_col, value_col]
            ].itertuples(index=False, name=None):
                system = str(ligand).rsplit("_", 1)[0]
                records.setdefault((str(smiles), system), []).append(float(value))
        print(json.dumps({"scanned_rows": rows, "profile_records": len(records)}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt") as handle:
        for (smiles, system), values in sorted(records.items()):
            handle.write(json.dumps({
                "smiles": smiles, "system": system,
                "value": float(pd.Series(values).median()), "observations": len(values),
            }) + "\n")
    print(json.dumps({
        "output": str(args.output), "records": len(records),
        "molecules": len({key[0] for key in records}),
        "systems": len({key[1] for key in records}),
    }), flush=True)


if __name__ == "__main__":
    main()
