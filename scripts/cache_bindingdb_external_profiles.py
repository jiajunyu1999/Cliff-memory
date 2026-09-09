from __future__ import annotations

"""Cache non-target BindingDB response records for external-transfer molecules."""

import argparse
import gzip
import json
from pathlib import Path

import pandas as pd

from molcliff.data import MOLECULEACE_DATASETS, load_moleculeace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=Path("../training_data_860.csv"))
    parser.add_argument("--external", type=Path, default=Path("data/bindingdb_external_exact_protein.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/bindingdb_external_response_profiles.jsonl.gz"))
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()
    external = pd.read_csv(args.external)
    wanted = set(external.smiles.astype(str))
    for dataset in external.dataset.unique():
        wanted.update(load_moleculeace(str(dataset)).query("split == 'train'").smiles.astype(str))
    records = []
    usecols = ["lig1", "lig2", "smile1", "smile2", "Label1", "Label2"]
    for chunk_i, chunk in enumerate(pd.read_csv(args.pairs, usecols=usecols, chunksize=args.chunksize), 1):
        for lig, smile, value in [(chunk.lig1, chunk.smile1, chunk.Label1),
                                  (chunk.lig2, chunk.smile2, chunk.Label2)]:
            mask = smile.astype(str).isin(wanted) & value.notna()
            if not mask.any():
                continue
            systems = lig[mask].astype(str).str.rsplit("_", n=1).str[0]
            records.extend(zip(smile[mask].astype(str), systems, value[mask].astype(float), strict=True))
        if chunk_i % 10 == 0:
            print(json.dumps({"chunks": chunk_i, "records": len(records)}), flush=True)
    profile = pd.DataFrame(records, columns=["smiles", "system", "value"])
    profile = profile.groupby(["smiles", "system"], as_index=False).value.median()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt") as handle:
        for row in profile.itertuples(index=False):
            handle.write(json.dumps({"smiles": row.smiles, "system": row.system, "value": float(row.value)}) + "\n")
    print(json.dumps({"rows": len(profile), "molecules": int(profile.smiles.nunique()), "systems": int(profile.system.nunique())}, indent=2))


if __name__ == "__main__":
    main()
