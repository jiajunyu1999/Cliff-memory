from __future__ import annotations

"""Cache ChEMBL target identities needed for off-target leakage filtering."""

import argparse
import gzip
import json
import time
from pathlib import Path

import requests

from molcliff.data import MOLECULEACE_DATASETS


API = "https://www.ebi.ac.uk/chembl/api/data/target.json"


def profile_target_ids(profile_root: Path, qualitative_root: Path | None = None) -> list[str]:
    target_ids: set[str] = {dataset.rsplit("_", 1)[0] for dataset in MOLECULEACE_DATASETS}
    for path in sorted(profile_root.glob("*.*_offtarget.jsonl.gz")):
        dataset = path.name.split(".", 1)[0]
        benchmark_id = dataset.rsplit("_", 1)[0]
        if benchmark_id.startswith("CHEMBL"):
            target_ids.add(benchmark_id)
        with gzip.open(path, "rt") as handle:
            for line in handle:
                target_id = json.loads(line).get("target_chembl_id")
                if target_id:
                    target_ids.add(str(target_id))
    if qualitative_root is not None and qualitative_root.exists():
        for path in sorted(qualitative_root.glob("*.train_qualitative.jsonl.gz")):
            with gzip.open(path, "rt") as handle:
                for line in handle:
                    target_id = json.loads(line).get("target_chembl_id")
                    if target_id:
                        target_ids.add(str(target_id))
    return sorted(target_ids)


def fetch_batch(session: requests.Session, target_ids: list[str]) -> list[dict]:
    params = {
        "target_chembl_id__in": ",".join(target_ids),
        "limit": len(target_ids),
    }
    for attempt in range(5):
        response = session.get(API, params=params, timeout=90)
        if response.ok:
            return response.json().get("targets", [])
        if response.status_code not in {429, 500, 502, 503, 504}:
            response.raise_for_status()
        time.sleep(2**attempt)
    response.raise_for_status()
    raise RuntimeError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-root", type=Path,
                        default=Path("data/chembl_offtarget_profiles"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/chembl_target_metadata.jsonl.gz"))
    parser.add_argument("--qualitative-root", type=Path,
                        default=Path("data/chembl_qualitative_profiles"))
    parser.add_argument("--batch-size", type=int, default=25)
    args = parser.parse_args()

    target_ids = profile_target_ids(args.profile_root, args.qualitative_root)
    records: dict[str, dict] = {}
    if args.output.exists():
        with gzip.open(args.output, "rt") as handle:
            for line in handle:
                record = json.loads(line)
                records[str(record["target_chembl_id"])] = record
    missing = [target_id for target_id in target_ids if target_id not in records]
    session = requests.Session()
    for start in range(0, len(missing), args.batch_size):
        batch = missing[start : start + args.batch_size]
        for record in fetch_batch(session, batch):
            records[str(record["target_chembl_id"])] = record
        print(f"targets: {min(start + args.batch_size, len(missing))}/{len(missing)}", flush=True)

    absent = sorted(set(target_ids).difference(records))
    if absent:
        raise RuntimeError(f"ChEMBL did not return {len(absent)} targets; first={absent[0]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt") as handle:
        for target_id in sorted(records):
            handle.write(json.dumps(records[target_id], sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "targets": len(records)}, sort_keys=True))


if __name__ == "__main__":
    main()
