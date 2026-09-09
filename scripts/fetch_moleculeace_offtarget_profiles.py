from __future__ import annotations

"""Fetch train-only ChEMBL off-target bioactivity profiles with an audit trail."""

import argparse
import gzip
import json
import time
from pathlib import Path

import pandas as pd
import requests

from molcliff.data import (
    EXPANDED_CLIFF_ROOT,
    MOLECULEACE_DATASETS,
    MOLECULEACE_ROOT,
    load_moleculeace,
)


API = "https://www.ebi.ac.uk/chembl/api/data/activity.json"


def fetch_page(session: requests.Session, params: dict) -> dict:
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = session.get(API, params=params, timeout=90)
        except requests.RequestException as error:
            last_error = error
            time.sleep(2 ** attempt)
            continue
        if response.ok:
            return response.json()
        if response.status_code not in {429, 500, 502, 503, 504}:
            response.raise_for_status()
        time.sleep(2 ** attempt)
    if last_error is not None:
        raise last_error
    response.raise_for_status()
    raise RuntimeError("unreachable")


def fetch_batch_activities(session: requests.Session, batch: list[str]) -> list[dict]:
    """Fetch one molecule batch, bisecting batches that trigger persistent API errors."""
    activities: list[dict] = []
    offset = 0
    try:
        while True:
            params = {
                "molecule_chembl_id__in": ",".join(batch),
                "pchembl_value__isnull": False,
                "limit": 1000,
                "offset": offset,
                "only": ("molecule_chembl_id,target_chembl_id,pchembl_value,"
                         "standard_type,assay_chembl_id,document_chembl_id"),
            }
            payload = fetch_page(session, params)
            page = payload.get("activities", [])
            activities.extend(page)
            if payload.get("page_meta", {}).get("next") is None:
                return activities
            offset += len(page)
    except requests.RequestException:
        if len(batch) == 1:
            raise
        middle = len(batch) // 2
        return (
            fetch_batch_activities(session, batch[:middle])
            + fetch_batch_activities(session, batch[middle:])
        )


def dataset_ids(dataset: str, split: str) -> list[str]:
    benchmark = load_moleculeace(dataset)
    split_smiles = set(benchmark.loc[benchmark.split.eq(split), "smiles"])
    path = MOLECULEACE_ROOT / "raw" / f"{dataset}.csv"
    if not path.exists():
        path = EXPANDED_CLIFF_ROOT / "raw" / f"{dataset}.csv"
    raw = pd.read_csv(path)
    mapping = raw.drop_duplicates("smiles").set_index("smiles").chembl_id
    missing = sorted(split_smiles.difference(mapping.index))
    if missing:
        raise ValueError(f"{dataset}: {len(missing)} {split} SMILES lack ChEMBL IDs")
    return sorted(set(mapping.loc[list(split_smiles)].astype(str)))


def fetch_dataset(dataset: str, output_root: Path, batch_size: int, split: str) -> None:
    target_id = dataset.rsplit("_", 1)[0]
    ids = dataset_ids(dataset, split)
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / f"{dataset}.{split}_offtarget.jsonl.gz"
    suffix = "" if split == "train" else f".{split}"
    meta_path = output_root / f"{dataset}{suffix}.metadata.json"
    session = requests.Session()
    kept = 0
    seen = 0
    with gzip.open(path, "wt") as handle:
        for start in range(0, len(ids), batch_size):
            batch = ids[start:start + batch_size]
            activities = fetch_batch_activities(session, batch)
            seen += len(activities)
            for activity in activities:
                # Removing every record for the benchmark target is the
                # central anti-leakage invariant, independent of assay type.
                if activity.get("target_chembl_id") == target_id:
                    continue
                handle.write(json.dumps(activity, sort_keys=True) + "\n")
                kept += 1
            print(f"{dataset}: {min(start + batch_size, len(ids))}/{len(ids)} molecules", flush=True)
    meta = {
        "dataset": dataset,
        "benchmark_target_excluded": target_id,
        "molecules": len(ids),
        "records_seen": seen,
        "offtarget_records_kept": kept,
        "endpoint": API,
        "query": "pchembl_value non-null; exact molecule ChEMBL IDs; batches of 25",
        "split": f"official {split} pool only; benchmark target labels excluded",
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(MOLECULEACE_DATASETS))
    parser.add_argument("--output-root", type=Path,
                        default=Path("data/chembl_offtarget_profiles"))
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    args = parser.parse_args()
    for dataset in args.datasets:
        fetch_dataset(dataset, args.output_root, args.batch_size, args.split)


if __name__ == "__main__":
    main()
