from __future__ import annotations

"""Fetch explicit ChEMBL active/inactive annotations for train molecules."""

import argparse
import gzip
import json
import time
from pathlib import Path

import requests

from fetch_moleculeace_offtarget_profiles import API, dataset_ids


def fetch(dataset: str, output_root: Path, batch_size: int) -> None:
    target_id = dataset.rsplit("_", 1)[0]
    molecule_ids = dataset_ids(dataset, "train")
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / f"{dataset}.train_qualitative.jsonl.gz"
    session = requests.Session()
    seen = kept = 0
    labels: dict[str, int] = {}
    with gzip.open(output, "wt") as handle:
        for start in range(0, len(molecule_ids), batch_size):
            batch = molecule_ids[start:start + batch_size]
            offset = 0
            while True:
                params = {
                    "molecule_chembl_id__in": ",".join(batch),
                    "activity_comment__isnull": False,
                    "limit": 1000,
                    "offset": offset,
                    "only": ("molecule_chembl_id,target_chembl_id,assay_chembl_id,"
                             "activity_comment,standard_type,standard_relation,"
                             "standard_value,standard_units,document_chembl_id"),
                }
                for attempt in range(5):
                    try:
                        response = session.get(API, params=params, timeout=90)
                    except requests.RequestException:
                        if attempt == 4:
                            raise
                        time.sleep(2 ** attempt)
                        continue
                    if response.ok:
                        break
                    if response.status_code not in {429, 500, 502, 503, 504}:
                        response.raise_for_status()
                    time.sleep(2 ** attempt)
                response.raise_for_status()
                payload = response.json()
                activities = payload.get("activities", [])
                seen += len(activities)
                for activity in activities:
                    label = str(activity.get("activity_comment", "")).strip().lower()
                    if label not in {"active", "inactive"}:
                        continue
                    if activity.get("target_chembl_id") == target_id:
                        continue
                    activity["binary_activity"] = int(label == "active")
                    handle.write(json.dumps(activity, sort_keys=True) + "\n")
                    labels[label] = labels.get(label, 0) + 1
                    kept += 1
                if payload.get("page_meta", {}).get("next") is None:
                    break
                offset += len(activities)
            print(json.dumps({
                "dataset": dataset, "molecules": min(start + batch_size, len(molecule_ids)),
                "total_molecules": len(molecule_ids), "kept": kept,
            }), flush=True)
    metadata = {
        "dataset": dataset, "split": "official train only", "records_seen": seen,
        "records_kept": kept, "labels": labels,
        "benchmark_target_excluded": target_id,
        "selection": "activity_comment exactly active or inactive",
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output-root", type=Path,
                        default=Path("data/chembl_qualitative_profiles"))
    parser.add_argument("--batch-size", type=int, default=25)
    args = parser.parse_args()
    for dataset in args.datasets:
        fetch(dataset, args.output_root, args.batch_size)


if __name__ == "__main__":
    main()
