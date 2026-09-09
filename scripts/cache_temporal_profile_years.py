from __future__ import annotations

"""Fetch immutable ChEMBL document years for a temporal-profile audit.

The cache is limited to one predeclared MoleculeACE target and its own
benchmark molecules.  It records publication year, never response labels, and
is written as a reusable local audit artifact.
"""

import argparse
import gzip
import json
import time
from pathlib import Path

import pandas as pd
import requests

from molcliff.data import load_moleculeace
from validate_moleculeace_offtarget_kernel import molecule_mapping

API = "https://www.ebi.ac.uk/chembl/api/data"


def get(session: requests.Session, endpoint: str, params: dict) -> dict:
    error = None
    for attempt in range(6):
        try:
            response = session.get(f"{API}/{endpoint}.json", params=params, timeout=90)
            if response.ok:
                return response.json()
            response.raise_for_status()
        except requests.RequestException as exc:
            error = exc; time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"ChEMBL request failed: {endpoint}: {error}")


def batches(values: list[str], size: int = 100):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train-root", type=Path, default=Path("data/chembl_offtarget_profiles"))
    parser.add_argument("--test-root", type=Path, default=Path("data/chembl_offtarget_profiles_test"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or Path("data/temporal_profiles") / f"{args.dataset}.years.json.gz"
    frame = load_moleculeace(args.dataset)
    mapping = molecule_mapping(args.dataset)
    molecule_ids = sorted({mapping[str(s)] for s in frame.smiles})
    benchmark_target = args.dataset.rsplit("_", 1)[0]
    session = requests.Session()
    benchmark = []
    for group in batches(molecule_ids, 50):
        payload = get(session, "activity", {"molecule_chembl_id__in": ",".join(group),
                                             "target_chembl_id": benchmark_target,
                                             "pchembl_value__isnull": "False", "limit": 1000,
                                             "only": "molecule_chembl_id,document_chembl_id,standard_type,pchembl_value"})
        benchmark.extend(payload.get("activities", []))
        print(json.dumps({"benchmark_molecules_queried": len(group), "records": len(benchmark)}), flush=True)
    document_ids = {str(x["document_chembl_id"]) for x in benchmark if x.get("document_chembl_id")}
    profile_records = []
    for root, part in [(args.train_root, "train"), (args.test_root, "test")]:
        path = root / f"{args.dataset}.{part}_offtarget.jsonl.gz"
        with gzip.open(path, "rt") as handle:
            for line in handle:
                x = json.loads(line); profile_records.append(x)
                if x.get("document_chembl_id"):
                    document_ids.add(str(x["document_chembl_id"]))
    years = {}
    ids = sorted(document_ids)
    for group in batches(ids, 100):
        payload = get(session, "document", {"document_chembl_id__in": ",".join(group),
                                             "limit": 100, "only": "document_chembl_id,year"})
        for record in payload.get("documents", []):
            if record.get("year") is not None:
                years[str(record["document_chembl_id"])] = int(record["year"])
        print(json.dumps({"documents": len(years), "requested": len(ids)}), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"dataset": args.dataset, "benchmark_target": benchmark_target,
               "benchmark_activities": benchmark, "document_year": years,
               "profile_records": profile_records,
               "protocol": "ChEMBL document publication year; fixed target and molecule set; no endpoint labels beyond record identity."}
    with gzip.open(output, "wt") as handle: json.dump(payload, handle)
    print(json.dumps({"output": str(output), "benchmark_records": len(benchmark),
                      "profile_records": len(profile_records), "document_years": len(years)}))


if __name__ == "__main__":
    main()
