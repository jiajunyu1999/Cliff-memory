from __future__ import annotations

"""Frozen-model response-memory dose--response experiment.

For each released MoleculeACE test target, a TAPER kernel is fitted once on
the released training partition.  Evaluation is then restricted to the same
test molecules that have at least ``max(k)`` target-excluded assay records.
For every repeat and every dose, only those molecules' response records are
randomly subsampled; chemical structures, labels, target, training kernel and
SVR are unchanged.  Thus dose is the only manipulated variable.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import SVR

from molcliff.data import MOLECULEACE_DATASETS, load_moleculeace
from molcliff.metrics import regression_metrics
from run_taper_profile_controls import (
    assemble_kernel, filter_profiles, profile_components,
)
from validate_moleculeace_offtarget_kernel import molecule_mapping
from validate_moleculeace_operator_action_kernel import exact_kernel
from validate_moleculeace_profile_alignment_kernel import (
    load_target_metadata,
)


def load_assay_records(dataset: str, root: Path, part: str) -> pd.DataFrame:
    path = root / f"{dataset}.{part}_offtarget.jsonl.gz"
    rows = []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            record = json.loads(line)
            rows.append((str(record["molecule_chembl_id"]), str(record["target_chembl_id"]),
                         str(record["assay_chembl_id"]), float(record["pchembl_value"])))
    frame = pd.DataFrame(rows, columns=["molecule", "target_entity", "target", "value"])
    return frame.groupby(["molecule", "target_entity", "target"], as_index=False).value.median()


def entity_from_assay(assay: pd.DataFrame) -> pd.DataFrame:
    return assay[["molecule", "target_entity", "value"]].rename(
        columns={"target_entity": "target"}
    ).groupby(
        ["molecule", "target"], as_index=False
    ).value.median()


def rng_for(dataset: str, dose: int, repeat: int) -> np.random.Generator:
    digest = hashlib.sha256(f"dose|{dataset}|{dose}|{repeat}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def run_one(dataset: str, train_root: Path, test_root: Path, metadata: dict[str, dict],
            doses: list[int], repeats: int) -> list[dict]:
    frame = load_moleculeace(dataset)
    fit = frame[frame.split.eq("train")].copy().reset_index(drop=True)
    valid_all = frame[frame.split.eq("test")].copy().reset_index(drop=True)
    mapping = molecule_mapping(dataset)
    fit_ids = [mapping[str(s)] for s in fit.smiles]
    valid_all_ids = [mapping[str(s)] for s in valid_all.smiles]

    train_assay = load_assay_records(dataset, train_root, "train")
    test_assay = load_assay_records(dataset, test_root, "test")
    # Strict target/equivalent-target removal is applied before determining
    # which queries are rich enough to participate.
    train_entity = entity_from_assay(train_assay)
    test_entity = entity_from_assay(test_assay)
    all_entity = pd.concat([train_entity, test_entity], ignore_index=True)
    all_assay = pd.concat([train_assay, test_assay], ignore_index=True)
    # The temporal test cache includes some ChEMBL target identifiers without
    # a local identity record. They cannot be assessed for equivalence, so
    # remove them rather than silently retaining a possible target proxy.
    all_entity = all_entity[all_entity.target.astype(str).isin(metadata)].copy()
    all_assay = all_assay[all_assay.target_entity.astype(str).isin(metadata)].copy()
    entity, assay, excluded = filter_profiles(all_entity, all_assay, dataset, metadata, "strict")
    train_set, test_set = set(fit_ids), set(valid_all_ids)
    entity_train = entity[entity.molecule.isin(train_set)].copy()
    assay_train = assay[assay.molecule.isin(train_set)].copy()
    assay_test = assay[assay.molecule.isin(test_set)].copy()
    count = assay_test.groupby("molecule").size()
    rich_ids = set(count[count >= max(doses)].index)
    take = np.asarray([idx for idx, mid in enumerate(valid_all_ids) if mid in rich_ids], dtype=int)
    if len(take) < 8 or valid_all.iloc[take].cliff_mol.nunique() < 2:
        return []
    valid = valid_all.iloc[take].copy().reset_index(drop=True)
    valid_ids = [valid_all_ids[idx] for idx in take]
    chemistry_fit, chemistry_valid = exact_kernel(fit, valid)
    # Fit the response-kernel SVR once. The selected coordinate vocabulary and
    # every training-side profile component are frozen for all dose conditions.
    full_entity = pd.concat([entity_train, entity[entity.molecule.isin(rich_ids)]], ignore_index=True)
    full_assay = pd.concat([assay_train, assay_test[assay_test.molecule.isin(rich_ids)]], ignore_index=True)
    frozen_components, audit = profile_components(
        fit_ids, valid_ids, fit.target.to_numpy(float), full_entity, full_assay
    )
    train_kernel, _ = assemble_kernel(chemistry_fit, chemistry_valid, frozen_components)
    taper = SVR(C=10.0, epsilon=.1, kernel="precomputed").fit(train_kernel, fit.target.to_numpy(float))
    chemistry = SVR(C=10.0, epsilon=.1, kernel="precomputed").fit(chemistry_fit, fit.target.to_numpy(float))
    chemistry_prediction = chemistry.predict(chemistry_valid)
    rows = []
    rich_assay = assay_test[assay_test.molecule.isin(rich_ids)].copy()
    for dose in doses:
        for repeat in range(repeats):
            rng = rng_for(dataset, dose, repeat)
            selected = []
            if dose:
                for molecule, group in rich_assay.groupby("molecule", sort=False):
                    # At this point every molecule has >= max(doses) records.
                    index = rng.choice(len(group), size=dose, replace=False)
                    selected.append(group.iloc[np.sort(index)])
            masked_assay = (pd.concat(selected, ignore_index=True) if selected else
                            rich_assay.iloc[0:0].copy())
            masked_entity = entity_from_assay(masked_assay)
            entity_condition = pd.concat([entity_train, masked_entity], ignore_index=True)
            assay_condition = pd.concat([assay_train, masked_assay], ignore_index=True)
            components, condition_audit = profile_components(
                fit_ids, valid_ids, fit.target.to_numpy(float), entity_condition, assay_condition
            )
            # Assert the fit-side kernel is unchanged to numerical precision;
            # otherwise this would not be a frozen-model dose manipulation.
            condition_train, condition_valid = assemble_kernel(chemistry_fit, chemistry_valid, components)
            if not np.allclose(condition_train, train_kernel, rtol=1e-6, atol=1e-6):
                raise AssertionError("masking altered the frozen training kernel")
            prediction = taper.predict(condition_valid)
            taper_metrics = regression_metrics(valid.target.to_numpy(float), prediction,
                                               valid.cliff_mol.to_numpy(bool))
            chem_metrics = regression_metrics(valid.target.to_numpy(float), chemistry_prediction,
                                              valid.cliff_mol.to_numpy(bool))
            rows.append({"dataset": dataset, "dose": dose, "repeat": repeat,
                         "n_rich_test": len(valid), "n_rich_cliff": int(valid.cliff_mol.sum()),
                         "equivalent_targets_excluded": len(excluded),
                         "fit_selected_entity_dimensions": audit["selected_entity_dimensions"],
                         "fit_selected_assay_dimensions": audit["selected_assay_dimensions"],
                         "taper_rmse": taper_metrics["rmse"],
                         "taper_cliff_rmse": taper_metrics["cliff_rmse"],
                         "chemistry_rmse": chem_metrics["rmse"],
                         "chemistry_cliff_rmse": chem_metrics["cliff_rmse"],
                         "overall_rmse_reduction": chem_metrics["rmse"] - taper_metrics["rmse"],
                         "cliff_rmse_reduction": chem_metrics["cliff_rmse"] - taper_metrics["cliff_rmse"],
                         "valid_selected_entity_dimensions": condition_audit["selected_entity_dimensions"],
                         "valid_selected_assay_dimensions": condition_audit["selected_assay_dimensions"]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(MOLECULEACE_DATASETS))
    parser.add_argument("--doses", nargs="+", type=int, default=[0, 1, 2, 4, 8, 16])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--train-root", type=Path, default=Path("data/chembl_offtarget_profiles"))
    parser.add_argument("--test-root", type=Path, default=Path("data/chembl_offtarget_profiles_test"))
    parser.add_argument("--metadata", type=Path, default=Path("data/chembl_target_metadata.jsonl.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/response_memory_dose_response"))
    args = parser.parse_args()
    if min(args.doses) != 0 or sorted(args.doses) != args.doses:
        raise ValueError("doses must be sorted and begin at zero")
    metadata = load_target_metadata(args.metadata)
    rows = []
    for dataset in args.datasets:
        print(f"[dose-response] {dataset}", flush=True)
        rows.extend(run_one(dataset, args.train_root, args.test_root, metadata, args.doses, args.repeats))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(args.output_dir / "dose_response.csv", index=False)
    summary = result.groupby("dose")[["overall_rmse_reduction", "cliff_rmse_reduction"]].agg(["mean", "std", "count"])
    summary.to_csv(args.output_dir / "dose_response_summary.csv")
    (args.output_dir / "summary.json").write_text(json.dumps({
        "datasets": int(result.dataset.nunique()), "rows": int(len(result)),
        "doses": args.doses, "repeats": args.repeats,
        "protocol": "same rich official-test molecules per target; frozen training kernel and SVR; strict biological exclusion; random assay-record masking only",
    }, indent=2) + "\n")
    print(json.dumps({"datasets": int(result.dataset.nunique()), "rows": int(len(result))}, indent=2))


if __name__ == "__main__":
    main()
