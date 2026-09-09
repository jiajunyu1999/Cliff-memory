from __future__ import annotations

"""Cache UniProt sequences for ChEMBL target components."""

import argparse
import gzip
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests


API = "https://rest.uniprot.org/uniprotkb/search"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-metadata", type=Path,
                        default=Path("data/chembl_target_metadata.jsonl.gz"))
    parser.add_argument("--target-context", type=Path,
                        default=Path("data/moleculeace_target_context.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/uniprot_target_sequences.json.gz"))
    parser.add_argument("--batch-size", type=int, default=25)
    args = parser.parse_args()

    accessions: set[str] = set()
    with gzip.open(args.target_metadata, "rt") as handle:
        for line in handle:
            record = json.loads(line)
            for component in record.get("target_components", []):
                accession = component.get("accession")
                if accession:
                    value = str(accession).split("-", 1)[0]
                    if re.fullmatch(r"[A-Z0-9]{6}|[A-Z0-9]{10}", value):
                        accessions.add(value)
    sequences: dict[str, str] = {}
    context = pd.read_csv(args.target_context)
    for record in context.itertuples(index=False):
        sequences[str(record.uniprot_accession)] = str(record.sequence)
    if args.output.exists():
        with gzip.open(args.output, "rt") as handle:
            sequences.update(json.load(handle))

    missing = sorted(accessions.difference(sequences))
    session = requests.Session()
    for start in range(0, len(missing), args.batch_size):
        batch = missing[start:start + args.batch_size]
        query = " OR ".join(f"accession:{accession}" for accession in batch)
        params = {"query": f"({query})", "format": "tsv", "fields": "accession,sequence", "size": 500}
        for attempt in range(5):
            response = session.get(API, params=params, timeout=90)
            if response.ok:
                break
            if response.status_code not in {429, 500, 502, 503, 504}:
                response.raise_for_status()
            time.sleep(2**attempt)
        response.raise_for_status()
        lines = response.text.splitlines()
        for line in lines[1:]:
            fields = line.split("\t")
            if len(fields) == 2:
                sequences[fields[0]] = fields[1]
        print(f"accessions: {min(start + args.batch_size, len(missing))}/{len(missing)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt") as handle:
        json.dump(sequences, handle, sort_keys=True)
    print(json.dumps({
        "requested": len(accessions), "cached": len(sequences),
        "missing": len(accessions.difference(sequences)), "output": str(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
