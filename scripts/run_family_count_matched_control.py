from __future__ import annotations

"""Family exclusion versus record-count-matched random target exclusion."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import SVR

from molcliff.data import MOLECULEACE_DATASETS, load_moleculeace
from molcliff.metrics import regression_metrics
from run_response_memory_dose_response import entity_from_assay, load_assay_records
from run_taper_profile_controls import assemble_kernel, profile_components
from run_biological_exclusion_radius import radius_exclusions
from validate_moleculeace_offtarget_kernel import molecule_mapping
from validate_moleculeace_operator_action_kernel import exact_kernel
from validate_moleculeace_profile_alignment_kernel import load_sequences, load_target_metadata


def response_count(entity: pd.DataFrame, assay: pd.DataFrame, targets: set[str]) -> int:
    return int(entity.target.astype(str).isin(targets).sum() + assay.target_entity.astype(str).isin(targets).sum())


def count_matched_random_targets(available: set[str], mandatory: set[str], target_counts: dict[str, int],
                                 target_count: int, dataset: str, repeat: int) -> set[str]:
    """Select target IDs with total response rows as close as possible to family removal.

    Candidate subsets are generated from independent seeded random rankings and
    greedily accumulated; the best of many candidates is retained.  This keeps
    removal at the target level while closely matching the actual number of
    entity-plus-assay records removed by the family condition.
    """
    candidates = sorted(available - mandatory)
    if not candidates or target_count <= 0: return set()
    seed = int.from_bytes(hashlib.sha256(f"count-match|{dataset}|{repeat}".encode()).digest()[:8], "little")
    root = np.random.default_rng(seed); best: set[str] = set(); best_error = float("inf")
    for _ in range(256):
        order = list(np.asarray(candidates)[root.permutation(len(candidates))])
        chosen: set[str] = set(); total = 0
        for target in order:
            count = target_counts[str(target)]
            before = abs(total - target_count)
            after = abs(total + count - target_count)
            # Continue until the target is crossed, then retain only if closer.
            if total < target_count or after < before:
                chosen.add(str(target)); total += count
            if total >= target_count and after >= before: break
        error = abs(total - target_count)
        if error < best_error: best, best_error = chosen, error
        if best_error == 0: break
    return best


def fit_condition(dataset: str, condition: str, excluded: set[str], entity: pd.DataFrame, assay: pd.DataFrame,
                  fit, test, fit_ids: list[str], test_ids: list[str], chemistry_fit: np.ndarray, chemistry_test: np.ndarray,
                  baseline_removed: int, repeat: int) -> dict:
    ent = entity[~entity.target.astype(str).isin(excluded)].copy()
    ass = assay[~assay.target_entity.astype(str).isin(excluded)].copy()
    comp,audit=profile_components(fit_ids,test_ids,fit.target.to_numpy(float),ent,ass)
    kfit,ktest=assemble_kernel(chemistry_fit,chemistry_test,comp)
    pred=SVR(C=10.,epsilon=.1,kernel="precomputed").fit(kfit,fit.target.to_numpy(float)).predict(ktest)
    return {"dataset":dataset,"condition":condition,"repeat":repeat,"excluded_targets":len(excluded),
            "removed_response_records": baseline_removed - (len(ent)+len(ass)),
            "remaining_response_records":len(ent)+len(ass), **audit,
            **regression_metrics(test.target.to_numpy(float),pred,test.cliff_mol.to_numpy(bool))}


def run_one(dataset: str, train_root: Path, test_root: Path, metadata: dict[str,dict], sequences: dict[str,str], repeats: int) -> list[dict]:
    frame=load_moleculeace(dataset);fit=frame[frame.split.eq("train")].reset_index(drop=True);test=frame[frame.split.eq("test")].reset_index(drop=True)
    mapping=molecule_mapping(dataset);fit_ids=[mapping[str(x)] for x in fit.smiles];test_ids=[mapping[str(x)] for x in test.smiles]
    assay=pd.concat([load_assay_records(dataset,train_root,"train"),load_assay_records(dataset,test_root,"test")],ignore_index=True)
    assay=assay[assay.target_entity.astype(str).isin(metadata)].copy();entity=entity_from_assay(assay)
    available=set(entity.target.astype(str)).union(assay.target_entity.astype(str));sets=radius_exclusions(dataset,available,metadata,sequences)
    strict=sets["strict_equivalent"];family=sets["same_protein_family"]
    all_rows=len(entity)+len(assay); family_extra=family-strict; target_counts={target:response_count(entity,assay,{target}) for target in available}
    family_extra_records=sum(target_counts[x] for x in family_extra)
    chemistry_fit,chemistry_test=exact_kernel(fit,test);rows=[]
    rows.append(fit_condition(dataset,"strict_equivalent",strict,entity,assay,fit,test,fit_ids,test_ids,chemistry_fit,chemistry_test,all_rows,0))
    rows.append(fit_condition(dataset,"same_protein_family",family,entity,assay,fit,test,fit_ids,test_ids,chemistry_fit,chemistry_test,all_rows,0))
    for repeat in range(repeats):
        random_extra=count_matched_random_targets(available,strict,target_counts,family_extra_records,dataset,repeat)
        row=fit_condition(dataset,"count_matched_random_exclusion",strict|random_extra,entity,assay,fit,test,fit_ids,test_ids,chemistry_fit,chemistry_test,all_rows,repeat)
        row["family_extra_removed_records"]=family_extra_records
        row["count_match_absolute_error"]=abs((row["removed_response_records"]-response_count(entity,assay,strict))-family_extra_records)
        rows.append(row)
    return rows


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument("--datasets",nargs="+",default=list(MOLECULEACE_DATASETS));p.add_argument("--repeats",type=int,default=20);p.add_argument("--train-root",type=Path,default=Path("data/chembl_offtarget_profiles"));p.add_argument("--test-root",type=Path,default=Path("data/chembl_offtarget_profiles_test"));p.add_argument("--metadata",type=Path,default=Path("data/chembl_target_metadata.jsonl.gz"));p.add_argument("--sequences",type=Path,default=Path("data/uniprot_target_sequences.json.gz"));p.add_argument("--output-dir",type=Path,default=Path("outputs/family_count_matched_control"));a=p.parse_args()
    metadata=load_target_metadata(a.metadata); sequences=load_sequences(a.sequences);rows=[]
    for dataset in a.datasets: print(f"[family-count] {dataset}",flush=True);rows.extend(run_one(dataset,a.train_root,a.test_root,metadata,sequences,a.repeats))
    a.output_dir.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(a.output_dir/"summary.csv",index=False)

if __name__ == "__main__": main()
