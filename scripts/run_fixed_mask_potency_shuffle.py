from __future__ import annotations

"""Fixed-observation-mask potency-shuffle control for response memory.

The molecule-by-coordinate observation pattern is untouched.  Only observed
potencies are permuted within each response coordinate, destroying
molecule-specific biological response while preserving coverage, dimensionality,
coordinate-wise potency distributions, and kernel construction.
"""

import argparse
import hashlib
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


def shuffle_potencies(assay: pd.DataFrame, dataset: str) -> pd.DataFrame:
    out = assay.copy()
    # Both entity and assay target IDs identify response coordinates. Values
    # are shuffled only within a coordinate, with a fixed deterministic seed.
    for key, index in out.groupby(["target_entity", "target"], sort=False).groups.items():
        seed = int.from_bytes(hashlib.sha256(f"potency-shuffle|{dataset}|{key}".encode()).digest()[:8], "little")
        values = out.loc[index, "value"].to_numpy(copy=True)
        out.loc[index, "value"] = np.random.default_rng(seed).permutation(values)
    return out


def run_one(dataset: str, train_root: Path, test_root: Path, metadata: dict[str, dict]) -> list[dict]:
    frame = load_moleculeace(dataset); fit = frame[frame.split.eq("train")].reset_index(drop=True); test = frame[frame.split.eq("test")].reset_index(drop=True)
    mapping = molecule_mapping(dataset); fit_ids=[mapping[str(x)] for x in fit.smiles]; test_ids=[mapping[str(x)] for x in test.smiles]
    assay = pd.concat([load_assay_records(dataset,train_root,"train"), load_assay_records(dataset,test_root,"test")], ignore_index=True)
    assay=assay[assay.target_entity.astype(str).isin(metadata)].copy(); entity=entity_from_assay(assay)
    entity, assay, excluded = filter_profiles(entity, assay, dataset, metadata, "strict")
    shuffled_assay = shuffle_potencies(assay, dataset); shuffled_entity = entity_from_assay(shuffled_assay)
    # Exact observation masks and coordinate dimensions are invariants.
    original_mask = entity[["molecule", "target"]].sort_values(["molecule", "target"]).reset_index(drop=True)
    shuffled_mask = shuffled_entity[["molecule", "target"]].sort_values(["molecule", "target"]).reset_index(drop=True)
    assert original_mask.equals(shuffled_mask)
    chemistry_fit, chemistry_test = exact_kernel(fit, test); y=fit.target.to_numpy(float)
    rows=[]
    for condition, ent, ass in [("true_molecule_specific_potency",entity,assay), ("fixed_mask_potency_shuffle",shuffled_entity,shuffled_assay)]:
        comp,audit=profile_components(fit_ids,test_ids,y,ent,ass); kfit,ktest=assemble_kernel(chemistry_fit,chemistry_test,comp)
        pred=SVR(C=10.,epsilon=.1,kernel="precomputed").fit(kfit,y).predict(ktest)
        rows.append({"dataset":dataset,"condition":condition,"excluded_equivalent_targets":len(excluded),"n_response_records":len(ass), **audit, **regression_metrics(test.target.to_numpy(float),pred,test.cliff_mol.to_numpy(bool))})
    return rows


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument("--datasets",nargs="+",default=list(MOLECULEACE_DATASETS));p.add_argument("--train-root",type=Path,default=Path("data/chembl_offtarget_profiles"));p.add_argument("--test-root",type=Path,default=Path("data/chembl_offtarget_profiles_test"));p.add_argument("--metadata",type=Path,default=Path("data/chembl_target_metadata.jsonl.gz"));p.add_argument("--output-dir",type=Path,default=Path("outputs/fixed_mask_potency_shuffle"));a=p.parse_args()
    metadata=load_target_metadata(a.metadata);rows=[]
    for dataset in a.datasets: print(f"[potency-shuffle] {dataset}",flush=True);rows.extend(run_one(dataset,a.train_root,a.test_root,metadata))
    a.output_dir.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(a.output_dir/"summary.csv",index=False)

if __name__ == "__main__": main()
