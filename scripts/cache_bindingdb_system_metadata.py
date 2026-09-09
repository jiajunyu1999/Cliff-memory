from __future__ import annotations

"""Cache target identity fields for BindingDB systems used by MoleculeACE."""

import argparse
import csv
import gzip
import io
import json
from pathlib import Path
import zipfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=Path,
                        default=Path("data/bindingdb_moleculeace_profiles.jsonl.gz"))
    parser.add_argument("--archive", type=Path, default=Path(
        "../outputs/pbcnet_exact_v1/PBCNet2.0_training_BindingDB_sparse_part7zero.zip"
    ))
    parser.add_argument("--output", type=Path,
                        default=Path("data/bindingdb_system_metadata.jsonl.gz"))
    args = parser.parse_args()
    systems = set()
    with gzip.open(args.profiles, "rt") as handle:
        for line in handle:
            systems.add(str(json.loads(line)["system"]))
    records = []
    missing = []
    prefix = "PBCNet2.0_training_BindingDB"
    with zipfile.ZipFile(args.archive) as archive:
        for index, system in enumerate(sorted(systems), 1):
            path = f"{prefix}/{system}/{system}.csv"
            try:
                with archive.open(path) as raw:
                    reader = csv.DictReader(io.TextIOWrapper(
                        raw, encoding="utf-8", errors="replace"
                    ))
                    row = next(reader)
            except (KeyError, zipfile.BadZipFile, EOFError, StopIteration):
                missing.append(system)
                continue
            sequence_columns = [key for key in row if "Target Chain Sequence" in key]
            accession_columns = [key for key in row if "Primary ID of Target Chain" in key]
            name_columns = [key for key in row if "Recommended Name of Target Chain" in key]
            sequences = sorted({
                str(row[key]).strip() for key in sequence_columns
                if str(row.get(key, "")).strip()
            })
            accessions = sorted({
                value.strip().upper()
                for key in accession_columns
                for value in str(row.get(key, "")).replace(";", ",").split(",")
                if value.strip()
            })
            names = sorted({
                " ".join(str(row[key]).lower().split()) for key in name_columns
                if str(row.get(key, "")).strip()
            })
            records.append({
                "system": system, "target_name": str(row.get("Target Name", "")),
                "accessions": accessions, "sequences": sequences, "names": names,
            })
            if index % 250 == 0:
                print(json.dumps({"cached": index, "total": len(systems)}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(json.dumps({
        "output": str(args.output), "systems": len(records),
        "unreadable_systems": len(missing), "unreadable_examples": missing[:10],
    }), flush=True)


if __name__ == "__main__":
    main()
