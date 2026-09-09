from __future__ import annotations

"""Sequence-identity and protein-family exclusion-radius stress test."""

import argparse
import gzip
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import Align
from sklearn.svm import SVR

from molcliff.data import MOLECULEACE_DATASETS, load_moleculeace
from molcliff.metrics import regression_metrics
from run_response_memory_dose_response import entity_from_assay, load_assay_records
from run_taper_profile_controls import assemble_kernel, filter_profiles, profile_components
from validate_moleculeace_offtarget_kernel import molecule_mapping
from validate_moleculeace_operator_action_kernel import exact_kernel
from validate_moleculeace_profile_alignment_kernel import biological_identity, load_sequences, load_target_metadata


def family_ids(record: dict) -> set[str]:
    values: set[str] = set()
    for component in record.get("target_components", []):
        for xref in component.get("target_component_xrefs", []):
            xref_id = str(xref.get("xref_id", ""))
            if xref_id.startswith(("IPR", "PF", "PTHR")):
                values.add(xref_id.split(":")[0])
    return values


def kmer_set(sequence: str, k: int = 3) -> set[str]:
    return {sequence[i:i+k] for i in range(max(len(sequence)-k+1, 0))}


def identity_score(reference: str, candidate: str, aligner: Align.PairwiseAligner) -> float:
    # Global exact-match fraction relative to the longer chain. Candidate
    # prefiltering is only computational; the reported threshold uses this
    # aligned sequence-identity score.
    return float(aligner.score(reference, candidate) / max(len(reference), len(candidate), 1))


def radius_exclusions(dataset: str, available: set[str], metadata: dict[str, dict],
                      sequences: dict[str, str]) -> dict[str, set[str]]:
    benchmark = metadata[dataset.rsplit("_", 1)[0]]
    benchmark_accessions, _, _ = biological_identity(benchmark)
    refs = [sequences[x] for x in benchmark_accessions if x in sequences]
    ref_kmers = [kmer_set(x) for x in refs]
    aligner = Align.PairwiseAligner(); aligner.mode = "global"
    aligner.match_score = 1.0; aligner.mismatch_score = 0.0
    aligner.open_gap_score = -1.0; aligner.extend_gap_score = -0.1
    strict = set()
    for target in available:
        rec = metadata.get(str(target))
        if rec is None: continue
        acc, genes, name = biological_identity(rec)
        _, bench_genes, bench_name = biological_identity(benchmark)
        if (str(target) == dataset.rsplit("_", 1)[0] or acc.intersection(benchmark_accessions)
                or genes.intersection(bench_genes) or (bench_name and name == bench_name)):
            strict.add(str(target))
    identity80, identity50, family = set(strict), set(strict), set(strict)
    family_ref = family_ids(benchmark)
    for target in available:
        rec = metadata.get(str(target))
        if rec is None: continue
        if family_ref.intersection(family_ids(rec)):
            family.add(str(target))
        acc, _, _ = biological_identity(rec)
        candidates = [sequences[x] for x in acc if x in sequences]
        max_identity = 0.0
        for candidate in candidates:
            ck = kmer_set(candidate)
            for reference, rk in zip(refs, ref_kmers, strict=True):
                # Conservatively skip impossible distant sequences; retained
                # candidates are evaluated by a global aligned identity.
                union = len(ck | rk)
                if union and len(ck & rk) / union < .04:
                    continue
                max_identity = max(max_identity, identity_score(reference, candidate, aligner))
        if max_identity >= .80: identity80.add(str(target))
        if max_identity >= .50: identity50.add(str(target))
    return {"strict_equivalent": strict, "identity_ge_80": identity80,
            "identity_ge_50": identity50, "same_protein_family": family}


def run_one(dataset: str, train_root: Path, test_root: Path, metadata: dict[str, dict],
            sequences: dict[str, str]) -> list[dict]:
    frame = load_moleculeace(dataset); fit = frame[frame.split.eq("train")].copy().reset_index(drop=True)
    valid = frame[frame.split.eq("test")].copy().reset_index(drop=True)
    mapping = molecule_mapping(dataset); fit_ids = [mapping[str(x)] for x in fit.smiles]; valid_ids = [mapping[str(x)] for x in valid.smiles]
    train_assay = load_assay_records(dataset, train_root, "train"); test_assay = load_assay_records(dataset, test_root, "test")
    assay = pd.concat([train_assay, test_assay], ignore_index=True)
    assay = assay[assay.target_entity.astype(str).isin(metadata)].copy()
    entity = entity_from_assay(assay)
    available = set(entity.target.astype(str)).union(assay.target_entity.astype(str))
    sets = radius_exclusions(dataset, available, metadata, sequences)
    chemistry_fit, chemistry_valid = exact_kernel(fit, valid); y = fit.target.to_numpy(float)
    rows = []
    for condition, excluded in sets.items():
        ent = entity[~entity.target.astype(str).isin(excluded)].copy()
        ass = assay[~assay.target_entity.astype(str).isin(excluded)].copy()
        components, audit = profile_components(fit_ids, valid_ids, y, ent, ass)
        train_kernel, test_kernel = assemble_kernel(chemistry_fit, chemistry_valid, components)
        pred = SVR(C=10.0, epsilon=.1, kernel="precomputed").fit(train_kernel, y).predict(test_kernel)
        rows.append({"dataset": dataset, "condition": condition, "excluded_targets": len(excluded),
                     "remaining_entity_records": len(ent), "remaining_assay_records": len(ass),
                     **audit, **regression_metrics(valid.target.to_numpy(float), pred, valid.cliff_mol.to_numpy(bool))})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--datasets", nargs="+", default=list(MOLECULEACE_DATASETS))
    parser.add_argument("--train-root", type=Path, default=Path("data/chembl_offtarget_profiles")); parser.add_argument("--test-root", type=Path, default=Path("data/chembl_offtarget_profiles_test"))
    parser.add_argument("--metadata", type=Path, default=Path("data/chembl_target_metadata.jsonl.gz")); parser.add_argument("--sequences", type=Path, default=Path("data/uniprot_target_sequences.json.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/biological_exclusion_radius")); args = parser.parse_args()
    metadata = load_target_metadata(args.metadata); sequences = load_sequences(args.sequences); rows = []
    for dataset in args.datasets:
        print(f"[radius] {dataset}", flush=True); rows.extend(run_one(dataset,args.train_root,args.test_root,metadata,sequences))
    args.output_dir.mkdir(parents=True,exist_ok=True); d=pd.DataFrame(rows); d.to_csv(args.output_dir/'summary.csv',index=False)
    print(json.dumps({"datasets":int(d.dataset.nunique()),"rows":len(d)},indent=2))

if __name__ == "__main__": main()
