from __future__ import annotations

"""Document-year temporal response-memory audit on a predeclared target."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import SVR

from molcliff.data import load_moleculeace
from molcliff.metrics import regression_metrics
from run_response_memory_dose_response import entity_from_assay
from run_taper_profile_controls import assemble_kernel, filter_profiles, profile_components
from validate_moleculeace_offtarget_kernel import molecule_mapping
from validate_moleculeace_operator_action_kernel import exact_kernel
from validate_moleculeace_profile_alignment_kernel import load_target_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="CHEMBL4203_Ki")
    parser.add_argument("--years", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=Path("data/chembl_target_metadata.jsonl.gz"))
    parser.add_argument("--output", type=Path, default=Path("reports/results/temporal_profile_cutoff.csv"))
    args = parser.parse_args()
    years_path = args.years or Path("data/temporal_profiles") / f"{args.dataset}.years.json.gz"
    with gzip.open(years_path, "rt") as handle: cache = json.load(handle)
    document_year = {str(k): int(v) for k, v in cache["document_year"].items()}
    endpoint = args.dataset.rsplit("_", 1)[1].lower()
    benchmark_year: dict[str, int] = {}
    for record in cache["benchmark_activities"]:
        if str(record.get("standard_type", "")).lower() != endpoint:
            continue
        year = document_year.get(str(record.get("document_chembl_id")))
        molecule = str(record.get("molecule_chembl_id"))
        if year is not None:
            benchmark_year[molecule] = min(benchmark_year.get(molecule, year), year)
    frame = load_moleculeace(args.dataset)
    mapping = molecule_mapping(args.dataset)
    frame["molecule"] = [mapping[str(x)] for x in frame.smiles]
    fit = frame[frame.split.eq("train") & frame.molecule.isin(benchmark_year)].copy().reset_index(drop=True)
    valid = frame[frame.split.eq("test") & frame.molecule.isin(benchmark_year)].copy().reset_index(drop=True)
    if valid.cliff_mol.nunique() < 2: raise RuntimeError("temporal subset lacks both cliff classes")
    known = load_target_metadata(args.metadata)
    records = pd.DataFrame(cache["profile_records"])
    records["molecule"] = records.molecule_chembl_id.astype(str)
    records["target_entity"] = records.target_chembl_id.astype(str)
    records["target"] = records.assay_chembl_id.astype(str)
    records["value"] = records.pchembl_value.astype(float)
    records["document_year"] = records.document_chembl_id.astype(str).map(document_year)
    records = records[records.target_entity.isin(known) & records.molecule.isin(benchmark_year)].copy()
    records["benchmark_year"] = records.molecule.map(benchmark_year)
    temporal = records[records.document_year.notna() & (records.document_year <= records.benchmark_year)].copy()
    # entity/assay profiles are constructed per molecule; target-equivalent
    # biological records are removed in both archive and temporal conditions.
    fit_ids, valid_ids = fit.molecule.tolist(), valid.molecule.tolist()
    chemistry_fit, chemistry_valid = exact_kernel(fit, valid)
    y = fit.target.to_numpy(float)
    rows = []
    for condition, source in [("archive", records), ("temporal_cutoff", temporal)]:
        assay = source[["molecule", "target_entity", "target", "value"]].groupby(
            ["molecule", "target_entity", "target"], as_index=False
        ).value.median()
        entity = entity_from_assay(assay)
        entity, assay, excluded = filter_profiles(entity, assay, args.dataset, known, "strict")
        components, audit = profile_components(fit_ids, valid_ids, y, entity, assay)
        train_kernel, test_kernel = assemble_kernel(chemistry_fit, chemistry_valid, components)
        pred = SVR(C=10.0, epsilon=.1, kernel="precomputed").fit(train_kernel, y).predict(test_kernel)
        metrics = regression_metrics(valid.target.to_numpy(float), pred, valid.cliff_mol.to_numpy(bool))
        coverage = int(assay[assay.molecule.isin(valid_ids)].molecule.nunique())
        rows.append({"dataset": args.dataset, "condition": condition, "n_train": len(fit), "n_test": len(valid),
                     "n_cliff": int(valid.cliff_mol.sum()), "n_benchmark_year": len(benchmark_year),
                     "n_profile_records": len(source), "n_valid_with_profile": coverage,
                     "excluded_equivalent_targets": len(excluded), **audit, **metrics})
    base = SVR(C=10.0, epsilon=.1, kernel="precomputed").fit(chemistry_fit, y).predict(chemistry_valid)
    rows.append({"dataset": args.dataset, "condition": "chemistry_only", "n_train": len(fit), "n_test": len(valid),
                 "n_cliff": int(valid.cliff_mol.sum()), "n_benchmark_year": len(benchmark_year),
                 **regression_metrics(valid.target.to_numpy(float), base, valid.cliff_mol.to_numpy(bool))})
    out = pd.DataFrame(rows); args.output.parent.mkdir(parents=True, exist_ok=True); out.to_csv(args.output, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__": main()
