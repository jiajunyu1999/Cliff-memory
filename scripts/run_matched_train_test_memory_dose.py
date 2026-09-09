from __future__ import annotations

"""Matched response-memory dose experiment with train and test masked together.

For every eligible official-split target, the same rich train molecules and the
same rich test molecules are retained at every dose.  A dose mask is applied
to both partitions before the response kernel is rebuilt and the SVR is
refitted.  Affinity labels, molecular structures, and the split never change.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import SVR

from molcliff.data import MOLECULEACE_DATASETS, load_moleculeace
from molcliff.metrics import regression_metrics
from run_response_memory_dose_response import entity_from_assay, load_assay_records
from run_taper_profile_controls import assemble_kernel, filter_profiles, profile_components
from validate_moleculeace_offtarget_kernel import molecule_mapping
from validate_moleculeace_operator_action_kernel import exact_kernel
from validate_moleculeace_profile_alignment_kernel import load_target_metadata


def rng_for(dataset: str, partition: str, dose: int, repeat: int) -> np.random.Generator:
    token = f"matched-dose|{dataset}|{partition}|{dose}|{repeat}".encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(token).digest()[:8], "little"))


def mask_records(records: pd.DataFrame, dose: int, rng: np.random.Generator) -> pd.DataFrame:
    if not dose:
        return records.iloc[0:0].copy()
    keep = []
    for _, group in records.groupby("molecule", sort=False):
        # Eligibility is checked before calling this function.
        keep.append(group.iloc[np.sort(rng.choice(len(group), dose, replace=False))])
    return pd.concat(keep, ignore_index=True)


def run_one(dataset: str, train_root: Path, test_root: Path, metadata: dict[str, dict],
            doses: list[int], repeats: int) -> list[dict]:
    frame = load_moleculeace(dataset)
    fit_all = frame[frame.split.eq("train")].copy().reset_index(drop=True)
    test_all = frame[frame.split.eq("test")].copy().reset_index(drop=True)
    mapping = molecule_mapping(dataset)
    fit_all_ids = [mapping[str(x)] for x in fit_all.smiles]
    test_all_ids = [mapping[str(x)] for x in test_all.smiles]
    assay = pd.concat([
        load_assay_records(dataset, train_root, "train"),
        load_assay_records(dataset, test_root, "test"),
    ], ignore_index=True)
    assay = assay[assay.target_entity.astype(str).isin(metadata)].copy()
    entity = entity_from_assay(assay)
    entity, assay, excluded = filter_profiles(entity, assay, dataset, metadata, "strict")
    max_dose = max(doses)
    counts = assay.groupby("molecule").size()
    rich = set(counts[counts >= max_dose].index)
    fit_take = np.asarray([i for i, x in enumerate(fit_all_ids) if x in rich], dtype=int)
    test_take = np.asarray([i for i, x in enumerate(test_all_ids) if x in rich], dtype=int)
    # Both partitions must be rich at the largest dose.  A small released
    # test subset is still informative if it contains both cliff labels;
    # target-level macro aggregation prevents large targets dominating.
    n_rich_cliff = int(test_all.iloc[test_take].cliff_mol.sum()) if len(test_take) else 0
    if len(fit_take) < 20 or len(test_take) < 4 or n_rich_cliff < 2 or (len(test_take) - n_rich_cliff) < 2:
        return []
    fit = fit_all.iloc[fit_take].copy().reset_index(drop=True)
    test = test_all.iloc[test_take].copy().reset_index(drop=True)
    fit_ids = [fit_all_ids[i] for i in fit_take]
    test_ids = [test_all_ids[i] for i in test_take]
    fit_records = assay[assay.molecule.isin(set(fit_ids))].copy()
    test_records = assay[assay.molecule.isin(set(test_ids))].copy()
    chemistry_fit, chemistry_test = exact_kernel(fit, test)
    y = fit.target.to_numpy(float)
    chem = SVR(C=10., epsilon=.1, kernel="precomputed").fit(chemistry_fit, y)
    chem_pred = chem.predict(chemistry_test)
    chem_metrics = regression_metrics(test.target.to_numpy(float), chem_pred, test.cliff_mol.to_numpy(bool))
    rows = []
    for dose in doses:
        for repeat in range(repeats):
            masked_fit = mask_records(fit_records, dose, rng_for(dataset, "train", dose, repeat))
            masked_test = mask_records(test_records, dose, rng_for(dataset, "test", dose, repeat))
            masked_assay = pd.concat([masked_fit, masked_test], ignore_index=True)
            masked_entity = entity_from_assay(masked_assay)
            components, audit = profile_components(fit_ids, test_ids, y, masked_entity, masked_assay)
            kfit, ktest = assemble_kernel(chemistry_fit, chemistry_test, components)
            pred = SVR(C=10., epsilon=.1, kernel="precomputed").fit(kfit, y).predict(ktest)
            metrics = regression_metrics(test.target.to_numpy(float), pred, test.cliff_mol.to_numpy(bool))
            rows.append({"dataset": dataset, "dose": dose, "repeat": repeat,
                         "n_rich_train": len(fit), "n_rich_test": len(test),
                         "n_rich_cliff": int(test.cliff_mol.sum()),
                         "excluded_equivalent_targets": len(excluded),
                         "masked_train_records": len(masked_fit), "masked_test_records": len(masked_test),
                         "chemistry_rmse": chem_metrics["rmse"], "chemistry_cliff_rmse": chem_metrics["cliff_rmse"],
                         "integrated_rmse": metrics["rmse"], "integrated_cliff_rmse": metrics["cliff_rmse"],
                         "overall_rmse_reduction": chem_metrics["rmse"] - metrics["rmse"],
                         "cliff_rmse_reduction": chem_metrics["cliff_rmse"] - metrics["cliff_rmse"], **audit})
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(MOLECULEACE_DATASETS))
    p.add_argument("--doses", nargs="+", type=int, default=[0, 1, 2, 4, 8, 16])
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--train-root", type=Path, default=Path("data/chembl_offtarget_profiles"))
    p.add_argument("--test-root", type=Path, default=Path("data/chembl_offtarget_profiles_test"))
    p.add_argument("--metadata", type=Path, default=Path("data/chembl_target_metadata.jsonl.gz"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/matched_train_test_memory_dose"))
    a = p.parse_args()
    if min(a.doses) != 0 or a.doses != sorted(a.doses): raise ValueError("doses must start at 0 and be sorted")
    rows = []
    metadata = load_target_metadata(a.metadata)
    for dataset in a.datasets:
        print(f"[matched-dose] {dataset}", flush=True)
        rows.extend(run_one(dataset, a.train_root, a.test_root, metadata, a.doses, a.repeats))
    a.output_dir.mkdir(parents=True, exist_ok=True)
    d = pd.DataFrame(rows); d.to_csv(a.output_dir / "summary.csv", index=False)
    (a.output_dir / "protocol.json").write_text(json.dumps({"doses": a.doses, "repeats": a.repeats, "protocol": "same rich train and test molecules at every dose; masks applied to both partitions; labels, structures and split fixed"}, indent=2)+"\n")
    print(json.dumps({"datasets": int(d.dataset.nunique()) if len(d) else 0, "rows": len(d)}, indent=2))


if __name__ == "__main__": main()
