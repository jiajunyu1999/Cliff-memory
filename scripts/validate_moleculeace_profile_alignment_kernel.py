from __future__ import annotations

"""Validate a target-aligned, leakage-filterable bioactivity profile kernel.

The profile view is one PSD kernel.  Feature weights are not searched: each is
the squared fit-only activity correlation after subtracting its finite-sample
null expectation.  This gives a simple supervised metric that suppresses the
many irrelevant off-target assays without fitting a second predictor.
"""

import argparse
import gzip
import json
import pickle
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from Bio import Align
from Bio.Align import substitution_matrices
from scipy import sparse
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors, rdPartialCharges
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold
from sklearn.svm import SVR

from molcliff.data import (
    MOLECULEACE_DATASETS, load_moleculeace, moleculeace_similarity_matrix,
    split_official_train,
)
from molcliff.metrics import regression_metrics
from validate_moleculeace_offtarget_kernel import load_profiles, matrices, molecule_mapping
from validate_moleculeace_operator_action_kernel import exact_kernel


def _rbf_from_train_scale(
    train: np.ndarray, query: np.ndarray, bandwidth_multiplier: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """RBF view with its sole length scale fixed by train geometry."""
    location = np.median(train, axis=0)
    scale = np.median(np.abs(train - location), axis=0) * 1.4826
    fallback = np.std(train, axis=0)
    scale = np.where(scale > 1e-8, scale, np.where(fallback > 1e-8, fallback, 1.0))
    train = (train - location) / scale
    query = (query - location) / scale
    train_norm = np.sum(train * train, axis=1)
    query_norm = np.sum(query * query, axis=1)
    train_distance = np.maximum(
        train_norm[:, None] + train_norm[None, :] - 2 * train @ train.T, 0.0
    )
    query_distance = np.maximum(
        query_norm[:, None] + train_norm[None, :] - 2 * query @ train.T, 0.0
    )
    upper = train_distance[np.triu_indices(len(train), 1)]
    positive = upper[upper > 1e-12]
    if bandwidth_multiplier <= 0:
        raise ValueError("bandwidth_multiplier must be positive")
    bandwidth = (float(np.median(positive)) if len(positive) else 1.0) * bandwidth_multiplier
    return (
        np.exp(-train_distance / bandwidth).astype(np.float32),
        np.exp(-query_distance / bandwidth).astype(np.float32),
    )


def physical_free_energy_kernels(
    fit_smiles: np.ndarray, valid_smiles: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Four interpretable ligand-side contributions to binding free energy.

    The channels represent desolvation, electrostatics/protonation,
    conformational entropy/strain, and size/dispersion.  They are deliberately
    small and fixed; this is a thermodynamic decomposition, not a descriptor
    search.
    """
    def describe(smiles: np.ndarray) -> list[np.ndarray]:
        rows = [[], [], [], []]
        for text in smiles:
            mol = Chem.MolFromSmiles(str(text))
            if mol is None:
                raise ValueError(f"RDKit failed to parse {text}")
            rdPartialCharges.ComputeGasteigerCharges(mol)
            charges = np.asarray([
                float(atom.GetProp("_GasteigerCharge"))
                for atom in mol.GetAtoms()
            ])
            charges = charges[np.isfinite(charges)]
            if not len(charges):
                charges = np.zeros(1)
            heavy = max(float(mol.GetNumHeavyAtoms()), 1.0)
            rows[0].append([
                Crippen.MolLogP(mol), rdMolDescriptors.CalcTPSA(mol),
                Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol),
                rdMolDescriptors.CalcLabuteASA(mol),
            ])
            rows[1].append([
                Chem.GetFormalCharge(mol), charges.max(), charges.min(),
                np.abs(charges).sum() / heavy,
                Lipinski.NumHeteroatoms(mol) / heavy,
            ])
            rows[2].append([
                Lipinski.NumRotatableBonds(mol), Lipinski.RingCount(mol),
                Lipinski.NumAromaticRings(mol), rdMolDescriptors.CalcFractionCSP3(mol),
                Descriptors.BertzCT(mol) / heavy,
            ])
            rows[3].append([
                Descriptors.MolWt(mol), heavy, Crippen.MolMR(mol),
            ])
        return [np.asarray(channel, dtype=np.float64) for channel in rows]

    fit_channels = describe(fit_smiles)
    valid_channels = describe(valid_smiles)
    kernels = [
        _rbf_from_train_scale(fit_channel, valid_channel)
        for fit_channel, valid_channel in zip(fit_channels, valid_channels, strict=True)
    ]
    return [item[0] for item in kernels], [item[1] for item in kernels]


def train_only_balanced_cliff_weights(
    smiles: np.ndarray, y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Match the benchmark's cliff definition using fit observations only.

    Inverse-frequency weights give high- and low-gradient molecular regions
    equal total mass.  This has no fitted coefficient and never reads the
    supplied full-dataset cliff flag.
    """
    similarity, _ = moleculeace_similarity_matrix(smiles, smiles)
    np.fill_diagonal(similarity, -np.inf)
    high_gradient = np.any(
        (similarity >= 0.9) & (np.abs(y[:, None] - y[None, :]) >= 1.0), axis=1
    )
    prevalence = float(high_gradient.mean())
    if prevalence <= 0.0 or prevalence >= 1.0:
        return np.ones(len(y), dtype=np.float64), high_gradient
    weights = np.where(
        high_gradient, 0.5 / prevalence, 0.5 / (1.0 - prevalence)
    )
    return weights.astype(np.float64), high_gradient


def centered_kernel_alignment(kernel: np.ndarray, y: np.ndarray) -> float:
    centered_y = np.asarray(y, dtype=np.float64) - float(np.mean(y))
    centered_kernel = (
        kernel - kernel.mean(axis=0, keepdims=True)
        - kernel.mean(axis=1, keepdims=True) + kernel.mean()
    )
    numerator = float(centered_y @ centered_kernel @ centered_y)
    denominator = float(
        np.linalg.norm(centered_kernel) * np.dot(centered_y, centered_y)
    )
    return max(numerator / max(denominator, 1e-12), 0.0)


def _assay_transfer_models(
    profile: pd.DataFrame, molecule_to_row: dict[str, int], y: np.ndarray
) -> dict[str, tuple[float, float, float]]:
    candidates = []
    subset = profile[profile.molecule.isin(molecule_to_row)]
    for assay, group in subset.groupby("target"):
        rows = np.asarray([molecule_to_row[molecule] for molecule in group.molecule])
        values = group.value.to_numpy(float)
        if len(values) < 5 or np.std(values) < 1e-8 or np.std(y[rows]) < 1e-8:
            continue
        correlation = float(np.corrcoef(values, y[rows])[0, 1])
        reliability = max(correlation * correlation - 1.0 / (len(values) - 1), 0.0)
        if reliability <= 0:
            continue
        slope = float(np.cov(values, y[rows], ddof=0)[0, 1] / np.var(values))
        intercept = float(y[rows].mean() - slope * values.mean())
        candidates.append((reliability, str(assay), slope, intercept))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return {assay: (weight, slope, intercept)
            for weight, assay, slope, intercept in candidates[:10]}


def _assay_transfer_predict(
    profile: pd.DataFrame, molecule_ids: list[str],
    models: dict[str, tuple[float, float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    row_of = {molecule: row for row, molecule in enumerate(molecule_ids)}
    numerator = np.zeros(len(molecule_ids), dtype=np.float64)
    denominator = np.zeros(len(molecule_ids), dtype=np.float64)
    subset = profile[profile.molecule.isin(row_of) & profile.target.isin(models)]
    for record in subset.itertuples(index=False):
        weight, slope, intercept = models[str(record.target)]
        row = row_of[str(record.molecule)]
        numerator[row] += weight * (slope * float(record.value) + intercept)
        denominator[row] += weight
    prediction = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
    )
    return prediction, denominator > 0


def crossfit_assay_transfer_features(
    profile: pd.DataFrame, fit_ids: list[str], valid_ids: list[str], y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict]:
    fit_score = np.zeros(len(fit_ids), dtype=np.float64)
    fit_observed = np.zeros(len(fit_ids), dtype=bool)
    indices = np.arange(len(fit_ids))
    for train_index, holdout_index in KFold(
        n_splits=5, shuffle=True, random_state=42
    ).split(indices):
        train_ids = [fit_ids[index] for index in train_index]
        models = _assay_transfer_models(
            profile, {molecule: row for row, molecule in enumerate(train_ids)}, y[train_index]
        )
        score, observed = _assay_transfer_predict(
            profile, [fit_ids[index] for index in holdout_index], models
        )
        fit_score[holdout_index] = score
        fit_observed[holdout_index] = observed

    full_models = _assay_transfer_models(
        profile, {molecule: row for row, molecule in enumerate(fit_ids)}, y
    )
    valid_score, valid_observed = _assay_transfer_predict(profile, valid_ids, full_models)
    if fit_observed.any():
        location = float(fit_score[fit_observed].mean())
        scale = float(fit_score[fit_observed].std())
    else:
        location, scale = 0.0, 1.0
    scale = max(scale, 1e-6)
    fit_features = np.column_stack([
        fit_observed.astype(float),
        fit_observed * ((fit_score - location) / scale),
    ])
    valid_features = np.column_stack([
        valid_observed.astype(float),
        valid_observed * ((valid_score - location) / scale),
    ])
    audit = {
        "assay_transfer_models": len(full_models),
        "assay_transfer_fit_coverage": float(fit_observed.mean()),
        "assay_transfer_valid_coverage": float(valid_observed.mean()),
    }
    return fit_features, valid_features, audit


def load_target_metadata(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with gzip.open(path, "rt") as handle:
        for line in handle:
            record = json.loads(line)
            records[str(record["target_chembl_id"])] = record
    return records


def load_sequence_kmers(path: Path) -> dict[str, np.ndarray]:
    with gzip.open(path, "rt") as handle:
        sequences = json.load(handle)
    alphabet = {amino_acid: index for index, amino_acid in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    vectors: dict[str, np.ndarray] = {}
    for accession, sequence in sequences.items():
        vector = np.zeros(20 ** 3, dtype=np.float32)
        encoded = [alphabet.get(amino_acid) for amino_acid in str(sequence)]
        for left, middle, right in zip(encoded, encoded[1:], encoded[2:]):
            if left is not None and middle is not None and right is not None:
                vector[left * 400 + middle * 20 + right] += 1.0
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vectors[str(accession)] = vector / norm
    return vectors


def load_sequences(path: Path) -> dict[str, str]:
    with gzip.open(path, "rt") as handle:
        raw = json.load(handle)
    alphabet = set("ACDEFGHIKLMNPQRSTVWY")
    return {
        str(accession): "".join(letter for letter in str(sequence) if letter in alphabet)
        for accession, sequence in raw.items() if sequence
    }


def load_bindingdb_profiles(
    profile_path: Path, metadata_path: Path,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    records = []
    with gzip.open(profile_path, "rt") as handle:
        for line in handle:
            record = json.loads(line)
            records.append((str(record["smiles"]), str(record["system"]),
                            float(record["value"])))
    metadata = {}
    with gzip.open(metadata_path, "rt") as handle:
        for line in handle:
            record = json.loads(line)
            metadata[str(record["system"])] = record
    profile = pd.DataFrame(records, columns=["molecule", "target", "value"])
    profile = profile[profile.target.isin(metadata)]
    return profile, metadata


def filter_bindingdb_equivalent_targets(
    dataset: str, profile: pd.DataFrame, binding_metadata: dict[str, dict],
    chembl_metadata: dict[str, dict], sequences: dict[str, str],
) -> tuple[pd.DataFrame, list[str]]:
    benchmark = chembl_metadata[dataset.rsplit("_", 1)[0]]
    benchmark_accessions, _, benchmark_name = biological_identity(benchmark)
    benchmark_sequences = [sequences[value] for value in benchmark_accessions if value in sequences]
    aligner = Align.PairwiseAligner()
    aligner.mode = "local"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    benchmark_self = [float(aligner.score(value, value)) for value in benchmark_sequences]
    excluded = []
    for system in sorted(profile.target.unique()):
        record = binding_metadata[str(system)]
        accessions = set(map(str.upper, record.get("accessions", [])))
        names = {" ".join(str(value).lower().split()) for value in record.get("names", [])}
        same = bool(benchmark_accessions.intersection(accessions))
        same = same or bool(benchmark_name and benchmark_name in names)
        if not same and benchmark_sequences:
            for candidate in record.get("sequences", []):
                candidate = "".join(
                    letter for letter in str(candidate) if letter in "ACDEFGHIKLMNPQRSTVWY"
                )
                if not candidate:
                    continue
                candidate_self = float(aligner.score(candidate, candidate))
                for reference, reference_self in zip(
                    benchmark_sequences, benchmark_self, strict=True
                ):
                    # Normalization by the shorter sequence detects the same
                    # target represented as an isolated domain or construct.
                    coverage_score = float(aligner.score(reference, candidate)) / max(
                        min(reference_self, candidate_self), 1e-12
                    )
                    if coverage_score >= 0.8:
                        same = True
                        break
                if same:
                    break
        if same:
            excluded.append(str(system))
    return profile[~profile.target.isin(excluded)].copy(), excluded


def local_alignment_nearest_targets(
    dataset: str, target_ids: set[str], metadata: dict[str, dict],
    sequences: dict[str, str], count: int = 10,
) -> list[str]:
    """Rank profile targets by normalized local BLOSUM62 similarity."""
    aligner = Align.PairwiseAligner()
    aligner.mode = "local"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    benchmark = metadata[dataset.rsplit("_", 1)[0]]
    benchmark_accessions = [
        str(component.get("accession"))
        for component in benchmark.get("target_components", [])
        if sequences.get(str(component.get("accession")))
    ]
    if not benchmark_accessions:
        return []
    self_scores: dict[str, float] = {}

    def self_score(accession: str) -> float:
        if accession not in self_scores:
            self_scores[accession] = float(
                aligner.score(sequences[accession], sequences[accession])
            )
        return self_scores[accession]

    similarities = []
    for target_id in target_ids:
        accessions = [
            str(component.get("accession"))
            for component in metadata[target_id].get("target_components", [])
            if sequences.get(str(component.get("accession")))
        ]
        if not accessions:
            continue
        similarity = max(
            float(aligner.score(sequences[left], sequences[right]))
            / np.sqrt(self_score(left) * self_score(right))
            for left in benchmark_accessions for right in accessions
        )
        similarities.append((similarity, target_id))
    similarities.sort(key=lambda item: (-item[0], item[1]))
    return [target_id for similarity, target_id in similarities[:count] if similarity > 0]


def biological_nearest_targets(
    dataset: str, target_ids: set[str], metadata: dict[str, dict],
    sequence_kmers: dict[str, np.ndarray], count: int = 10,
) -> list[str]:
    benchmark = metadata[dataset.rsplit("_", 1)[0]]
    benchmark_vectors = [
        sequence_kmers[str(component.get("accession"))]
        for component in benchmark.get("target_components", [])
        if str(component.get("accession")) in sequence_kmers
    ]
    if not benchmark_vectors:
        return []
    similarities = []
    for target_id in target_ids:
        record = metadata[target_id]
        candidates = [
            sequence_kmers[str(component.get("accession"))]
            for component in record.get("target_components", [])
            if str(component.get("accession")) in sequence_kmers
        ]
        if candidates:
            similarity = max(float(left @ right)
                             for left in benchmark_vectors for right in candidates)
            similarities.append((similarity, target_id))
    similarities.sort(key=lambda item: (-item[0], item[1]))
    return [target_id for similarity, target_id in similarities[:count] if similarity > 0]


def load_assay_profiles(dataset: str, root: Path, split: str = "train") -> pd.DataFrame:
    records = []
    splits = ("train", "test") if split == "both" else (split,)
    for part in splits:
        with gzip.open(root / f"{dataset}.{part}_offtarget.jsonl.gz", "rt") as handle:
            for line in handle:
                record = json.loads(line)
                records.append((
                    str(record["molecule_chembl_id"]),
                    str(record["target_chembl_id"]),
                    str(record["assay_chembl_id"]),
                    float(record["pchembl_value"]),
                ))
    frame = pd.DataFrame(records, columns=["molecule", "target_entity", "target", "value"])
    return frame.groupby(["molecule", "target_entity", "target"], as_index=False).value.median()


def load_endpoint_profiles(dataset: str, root: Path, split: str = "train") -> pd.DataFrame:
    endpoint = dataset.rsplit("_", 1)[1].lower()
    records = []
    splits = ("train", "test") if split == "both" else (split,)
    for part in splits:
        with gzip.open(root / f"{dataset}.{part}_offtarget.jsonl.gz", "rt") as handle:
            for line in handle:
                record = json.loads(line)
                if str(record.get("standard_type", "")).lower() != endpoint:
                    continue
                records.append((
                    str(record["molecule_chembl_id"]),
                    str(record["target_chembl_id"]),
                    float(record["pchembl_value"]),
                ))
    frame = pd.DataFrame(records, columns=["molecule", "target", "value"])
    return frame.groupby(["molecule", "target"], as_index=False).value.median()


def load_qualitative_profiles(
    dataset: str, root: Path, assay_level: bool = False,
) -> pd.DataFrame:
    records = []
    with gzip.open(root / f"{dataset}.train_qualitative.jsonl.gz", "rt") as handle:
        for line in handle:
            record = json.loads(line)
            records.append((
                str(record["molecule_chembl_id"]),
                str(record["target_chembl_id"]),
                str(record["assay_chembl_id"]),
                float(record["binary_activity"]),
            ))
    frame = pd.DataFrame(
        records, columns=["molecule", "target_entity", "assay", "value"]
    )
    if assay_level:
        return frame.rename(columns={"assay": "target"}).groupby(
            ["molecule", "target_entity", "target"], as_index=False
        ).value.median()
    return frame.rename(columns={"target_entity": "target"}).groupby(
        ["molecule", "target"], as_index=False
    ).value.median()


def biological_identity(record: dict) -> tuple[set[str], set[str], str]:
    accessions: set[str] = set()
    genes: set[str] = set()
    for component in record.get("target_components", []):
        accession = component.get("accession")
        if accession:
            accessions.add(str(accession).upper())
        for synonym in component.get("target_component_synonyms", []):
            if str(synonym.get("syn_type", "")).startswith("GENE_SYMBOL"):
                value = synonym.get("component_synonym")
                if value:
                    genes.add(str(value).upper())
    name = " ".join(str(record.get("pref_name", "")).lower().split())
    return accessions, genes, name


def remove_equivalent_targets(
    profile: pd.DataFrame, dataset: str, metadata: dict[str, dict]
) -> tuple[pd.DataFrame, list[str]]:
    benchmark_id = dataset.rsplit("_", 1)[0]
    benchmark = metadata.get(benchmark_id)
    if benchmark is None:
        raise KeyError(f"target metadata lacks benchmark target {benchmark_id}")
    benchmark_accessions, benchmark_genes, benchmark_name = biological_identity(benchmark)
    excluded = {benchmark_id}
    for target_id in profile.target.unique():
        record = metadata.get(str(target_id))
        if record is None:
            raise KeyError(f"target metadata lacks profile target {target_id}")
        accessions, genes, name = biological_identity(record)
        if (benchmark_accessions.intersection(accessions)
                or benchmark_genes.intersection(genes)
                or (benchmark_name and name == benchmark_name)):
            excluded.add(str(target_id))
    return profile[~profile.target.isin(excluded)].copy(), sorted(excluded)


def remove_equivalent_assays(
    profile: pd.DataFrame, dataset: str, metadata: dict[str, dict]
) -> pd.DataFrame:
    # Reuse the entity-level filter while retaining assay identities as the
    # eventual feature columns.
    entities = profile[["molecule", "target_entity", "value"]].rename(
        columns={"target_entity": "target"}
    )
    filtered, _ = remove_equivalent_targets(entities, dataset, metadata)
    allowed = set(zip(filtered.molecule, filtered.target, filtered.value))
    keep = [
        (row.molecule, row.target_entity, row.value) in allowed
        for row in profile.itertuples(index=False)
    ]
    return profile.loc[keep, ["molecule", "target", "value"]].copy()


def filter_equivalent_assay_records(
    profile: pd.DataFrame, dataset: str, metadata: dict[str, dict]
) -> pd.DataFrame:
    entities = profile[["molecule", "target_entity", "value"]].rename(
        columns={"target_entity": "target"}
    )
    filtered, _ = remove_equivalent_targets(entities, dataset, metadata)
    allowed_entities = set(filtered.target.astype(str))
    return profile[profile.target_entity.astype(str).isin(allowed_entities)].copy()


def profile_design(
    fit_ids: list[str], valid_ids: list[str], profile: pd.DataFrame, y: np.ndarray
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, list[str], dict]:
    fit_set = set(fit_ids)
    fit_profile = profile[profile.molecule.isin(fit_set)]
    targets = sorted(fit_profile.target.unique())
    vocabulary = {target: col for col, target in enumerate(targets)}
    row_of_fit = {molecule: row for row, molecule in enumerate(fit_ids)}
    row_of_valid = {molecule: row for row, molecule in enumerate(valid_ids)}

    # Estimate each assay's location, scale, and relevance using fit molecules
    # only.  Subtracting E[r^2|null]=1/(n-1) is an analytic noise correction.
    locations = np.zeros(len(targets), dtype=np.float32)
    scales = np.ones(len(targets), dtype=np.float32)
    weights = np.zeros(len(targets), dtype=np.float32)
    supports = np.zeros(len(targets), dtype=np.int32)
    for target, group in fit_profile.groupby("target"):
        col = vocabulary[target]
        rows = np.asarray([row_of_fit[m] for m in group.molecule], dtype=np.int64)
        values = group.value.to_numpy(dtype=float)
        location = float(np.median(values))
        scale = float(np.std(values))
        locations[col] = location
        supports[col] = len(values)
        if len(values) < 5 or scale < 1e-6 or np.std(y[rows]) < 1e-6:
            continue
        scales[col] = scale
        correlation = float(np.corrcoef(values, y[rows])[0, 1])
        if np.isfinite(correlation):
            weights[col] = max(correlation * correlation - 1.0 / (len(values) - 1), 0.0)

    def build(ids: list[str], row_of: dict[str, int]) -> sparse.csr_matrix:
        if not targets:
            return sparse.csr_matrix((len(ids), 0), dtype=np.float32)
        rows: list[int] = []
        cols: list[int] = []
        values: list[float] = []
        subset = profile[profile.molecule.isin(set(ids))]
        for record in subset.itertuples(index=False):
            col = vocabulary.get(record.target)
            if col is None or weights[col] <= 0:
                continue
            rows.append(row_of[record.molecule])
            cols.append(col)
            values.append((float(record.value) - locations[col]) / scales[col])
        matrix = sparse.csr_matrix(
            (values, (rows, cols)), shape=(len(ids), len(targets)), dtype=np.float32
        )
        return normalize(matrix.multiply(np.sqrt(weights)[None, :]), norm="l2")

    fit = build(fit_ids, row_of_fit)
    valid = build(valid_ids, row_of_valid)
    positive = np.flatnonzero(weights > 0)
    nearest = sorted(positive, key=lambda col: (-float(weights[col]), targets[col]))[:10]
    nearest_targets = [targets[col] for col in nearest]
    audit = {
        "profile_targets": len(targets),
        "positive_weight_targets": int(len(positive)),
        "max_weight": float(weights.max(initial=0)),
        "median_positive_weight": float(np.median(weights[positive])) if len(positive) else 0.0,
        "max_support": int(supports.max(initial=0)),
        "nearest_assays": json.dumps(nearest_targets),
    }
    return fit, valid, nearest_targets, audit


def action_profile_kernel(
    fit_ids: list[str], valid_ids: list[str], profile: pd.DataFrame,
    chemistry_fit: np.ndarray, chemistry_valid: np.ndarray, y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build a profile kernel whose fixed weights explain local activity deltas.

    Weights are fit-only squared correlations between off-target assay changes
    and changes in benchmark activity on the top 24 chemically nearest pairs
    (similarity >= .35).  The finite-sample null subtraction mirrors
    ``profile_design``.  No cliff flags, validation labels, or weight scan are
    used; valid rows are only projected into the frozen fit vocabulary.
    """
    fit_set = set(fit_ids)
    valid_set = set(valid_ids)
    fit_row = {molecule: row for row, molecule in enumerate(fit_ids)}
    valid_row = {molecule: row for row, molecule in enumerate(valid_ids)}
    fit_profile = profile[profile.molecule.isin(fit_set)]
    targets = sorted(fit_profile.target.unique())
    vocabulary = {target: col for col, target in enumerate(targets)}
    if not vocabulary:
        return (np.zeros_like(chemistry_fit), np.zeros_like(chemistry_valid),
                {"action_profile_targets": 0, "action_profile_positive": 0})
    p = len(vocabulary)
    fit_values = np.zeros((len(fit_ids), p), dtype=np.float32)
    valid_values = np.zeros((len(valid_ids), p), dtype=np.float32)
    locations = np.zeros(p, dtype=np.float32)
    scales = np.ones(p, dtype=np.float32)
    for target, group in fit_profile.groupby("target"):
        col = vocabulary[target]
        values = group.value.to_numpy(float)
        locations[col] = float(np.median(values))
        scales[col] = max(float(np.std(values)), 1e-6)
        for record in group.itertuples(index=False):
            row = fit_row.get(record.molecule)
            if row is not None:
                fit_values[row, col] = (float(record.value) - locations[col]) / scales[col]
    for record in profile[profile.molecule.isin(valid_set)].itertuples(index=False):
        col = vocabulary.get(record.target)
        row = valid_row.get(record.molecule)
        if col is not None and row is not None:
            valid_values[row, col] = (float(record.value) - locations[col]) / scales[col]

    # Fixed local action graph: same top-neighbor rule as the action model.
    edges: list[tuple[int, int]] = []
    k = max(4, min(24, int(np.sqrt(len(fit_ids)))))
    for src in range(len(fit_ids)):
        order = np.argsort(chemistry_fit[src])[::-1][:k + 1]
        for dst in order:
            if int(dst) == src or float(chemistry_fit[src, dst]) < 0.35:
                continue
            edges.append((src, int(dst)))
    if not edges:
        return (np.zeros_like(chemistry_fit), np.zeros_like(chemistry_valid),
                {"action_profile_targets": p, "action_profile_positive": 0})
    src = np.asarray([item[0] for item in edges], dtype=np.int64)
    dst = np.asarray([item[1] for item in edges], dtype=np.int64)
    delta_y = y[dst] - y[src]
    weights = np.zeros(p, dtype=np.float32)
    for col in range(p):
        left, right = fit_values[src, col], fit_values[dst, col]
        observed = (left != 0.0) & (right != 0.0)
        count = int(observed.sum())
        if count < 8:
            continue
        delta_x = right[observed] - left[observed]
        if float(np.std(delta_x)) < 1e-8 or float(np.std(delta_y[observed])) < 1e-8:
            continue
        correlation = float(np.corrcoef(delta_x, delta_y[observed])[0, 1])
        if np.isfinite(correlation):
            weights[col] = max(correlation * correlation - 1.0 / (count - 1), 0.0)
    positive = int(np.count_nonzero(weights > 0))
    if positive == 0:
        return (np.zeros_like(chemistry_fit), np.zeros_like(chemistry_valid),
                {"action_profile_targets": p, "action_profile_positive": 0,
                 "action_profile_edges": len(edges)})
    fit_values *= np.sqrt(weights)[None, :]
    valid_values *= np.sqrt(weights)[None, :]
    fit_values = normalize(sparse.csr_matrix(fit_values), norm="l2")
    valid_values = normalize(sparse.csr_matrix(valid_values), norm="l2")
    return (
        (fit_values @ fit_values.T).toarray().astype(np.float32),
        (valid_values @ fit_values.T).toarray().astype(np.float32),
        {"action_profile_targets": p, "action_profile_positive": positive,
         "action_profile_edges": len(edges)},
    )


def dense_nearest_profile_kernel(
    fit_ids: list[str], valid_ids: list[str], profile: pd.DataFrame,
    nearest_features: list[str], bandwidth_multiplier: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep potency and missingness for each fit-selected neighboring assay."""
    vocabulary = {feature: col for col, feature in enumerate(nearest_features)}
    locations = np.zeros(len(vocabulary), dtype=np.float64)
    scales = np.ones(len(vocabulary), dtype=np.float64)
    fit_set = set(fit_ids)
    for feature, group in profile[
        profile.molecule.isin(fit_set) & profile.target.isin(vocabulary)
    ].groupby("target"):
        col = vocabulary[str(feature)]
        values = group.value.to_numpy(float)
        locations[col] = float(np.median(values))
        scales[col] = max(float(np.std(values)), 1e-6)

    def build(ids: list[str]) -> np.ndarray:
        row_of = {molecule: row for row, molecule in enumerate(ids)}
        design = np.zeros((len(ids), 2 * len(vocabulary)), dtype=np.float64)
        subset = profile[
            profile.molecule.isin(row_of) & profile.target.isin(vocabulary)
        ]
        for record in subset.itertuples(index=False):
            row, col = row_of[str(record.molecule)], vocabulary[str(record.target)]
            design[row, col] = (float(record.value) - locations[col]) / scales[col]
            design[row, len(vocabulary) + col] = 1.0
        return design

    if not vocabulary:
        return (np.zeros((len(fit_ids), len(fit_ids)), dtype=np.float32),
                np.zeros((len(valid_ids), len(fit_ids)), dtype=np.float32))
    return _rbf_from_train_scale(
        build(fit_ids), build(valid_ids), bandwidth_multiplier=bandwidth_multiplier
    )


def imputed_nearest_profile_kernel(
    fit_ids: list[str], valid_ids: list[str], profile: pd.DataFrame,
    nearest_features: list[str], chemistry_fit: np.ndarray,
    chemistry_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Complete sparse assay coordinates by fit-only local chemical transport."""
    vocabulary = {feature: col for col, feature in enumerate(nearest_features)}
    if not vocabulary:
        return (np.zeros_like(chemistry_fit), np.zeros_like(chemistry_valid))
    row_fit = {molecule: row for row, molecule in enumerate(fit_ids)}
    row_valid = {molecule: row for row, molecule in enumerate(valid_ids)}
    fit_design = np.zeros((len(fit_ids), 3 * len(vocabulary)), dtype=np.float64)
    valid_design = np.zeros((len(valid_ids), 3 * len(vocabulary)), dtype=np.float64)
    subset = profile[profile.target.isin(vocabulary)]
    for feature, group in subset.groupby("target"):
        col = vocabulary[str(feature)]
        fit_group = group[group.molecule.isin(row_fit)]
        if len(fit_group) < 2:
            continue
        observed_rows = np.asarray([row_fit[str(value)] for value in fit_group.molecule])
        observed_values = fit_group.value.to_numpy(float)
        location = float(np.median(observed_values))
        scale = max(float(np.std(observed_values)), 1e-6)
        standardized = (observed_values - location) / scale

        fit_weights = chemistry_fit[:, observed_rows].astype(np.float64)
        valid_weights = chemistry_valid[:, observed_rows].astype(np.float64)
        fit_design[:, col] = (
            fit_weights @ standardized / np.maximum(fit_weights.sum(axis=1), 1e-12)
        )
        valid_design[:, col] = (
            valid_weights @ standardized / np.maximum(valid_weights.sum(axis=1), 1e-12)
        )
        fit_design[:, 2 * len(vocabulary) + col] = fit_weights.max(axis=1)
        valid_design[:, 2 * len(vocabulary) + col] = valid_weights.max(axis=1)

        for record in fit_group.itertuples(index=False):
            row = row_fit[str(record.molecule)]
            fit_design[row, col] = (float(record.value) - location) / scale
            fit_design[row, len(vocabulary) + col] = 1.0
        for record in group[group.molecule.isin(row_valid)].itertuples(index=False):
            row = row_valid[str(record.molecule)]
            valid_design[row, col] = (float(record.value) - location) / scale
            valid_design[row, len(vocabulary) + col] = 1.0
    return _rbf_from_train_scale(fit_design, valid_design)


def run_one(
    dataset: str, profile_root: Path, metadata: dict[str, dict],
    sequence_kmers: dict[str, np.ndarray] | None = None,
    sequences: dict[str, str] | None = None,
    bindingdb_profile: pd.DataFrame | None = None,
    bindingdb_metadata: dict[str, dict] | None = None,
    qualitative_root: Path | None = None,
    splitter: Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]] = split_official_train,
    candidate_names: set[str] | None = None,
    profile_split: str = "train",
    artifacts_dir: Path | None = None,
    predictions_dir: Path | None = None,
) -> list[dict]:
    fit, valid = splitter(load_moleculeace(dataset))
    chemistry_fit, chemistry_valid = exact_kernel(fit, valid)
    physical_fit_views, physical_valid_views = physical_free_energy_kernels(
        fit.smiles.to_numpy(), valid.smiles.to_numpy()
    )
    physical_fit = sum(physical_fit_views) / len(physical_fit_views)
    physical_valid = sum(physical_valid_views) / len(physical_valid_views)
    physical_local_fit = sum(
        chemistry_fit * view for view in physical_fit_views
    ) / len(physical_fit_views)
    physical_local_valid = sum(
        chemistry_valid * view for view in physical_valid_views
    ) / len(physical_valid_views)
    cliff_weights, fit_high_gradient = train_only_balanced_cliff_weights(
        fit.smiles.to_numpy(), fit.target.to_numpy(float)
    )
    mapping = molecule_mapping(dataset)
    fit_ids = [mapping[s] for s in fit.smiles]
    valid_ids = [mapping[s] for s in valid.smiles]
    profile, excluded = remove_equivalent_targets(
        load_profiles(dataset, profile_root, profile_split), dataset, metadata
    )
    profile_fit, profile_valid, nearest_targets, audit = profile_design(
        fit_ids, valid_ids, profile, fit.target.to_numpy(float)
    )
    action_profile_fit, action_profile_valid, action_profile_audit = action_profile_kernel(
        fit_ids, valid_ids, profile, chemistry_fit, chemistry_valid,
        fit.target.to_numpy(float),
    )
    audit.update(action_profile_audit)
    audit["biologically_equivalent_targets_excluded"] = len(excluded)
    coverage_fit, coverage_valid, residual_fit, residual_valid, _ = matrices(
        fit_ids, valid_ids, profile
    )
    unweighted_fit = (coverage_fit + residual_fit) / 2
    unweighted_valid = (coverage_valid + residual_valid) / 2
    nearest_profile = profile[profile.target.isin(nearest_targets)]
    (nearest_coverage_fit, nearest_coverage_valid,
     nearest_residual_fit, nearest_residual_valid, _) = matrices(
        fit_ids, valid_ids, nearest_profile
    )
    nearest_fit = (nearest_coverage_fit + nearest_residual_fit) / 2
    nearest_valid = (nearest_coverage_valid + nearest_residual_valid) / 2
    dense_nearest_fit, dense_nearest_valid = dense_nearest_profile_kernel(
        fit_ids, valid_ids, profile, nearest_targets
    )
    dense_nearest1_fit, dense_nearest1_valid = dense_nearest_profile_kernel(
        fit_ids, valid_ids, profile, nearest_targets[:1]
    )
    imputed_nearest_fit, imputed_nearest_valid = imputed_nearest_profile_kernel(
        fit_ids, valid_ids, profile, nearest_targets, chemistry_fit, chemistry_valid
    )
    nearest1_profile = profile[profile.target.isin(nearest_targets[:1])]
    (nearest1_coverage_fit, nearest1_coverage_valid,
     nearest1_residual_fit, nearest1_residual_valid, _) = matrices(
        fit_ids, valid_ids, nearest1_profile
    )
    nearest1_fit = (nearest1_coverage_fit + nearest1_residual_fit) / 2
    nearest1_valid = (nearest1_coverage_valid + nearest1_residual_valid) / 2
    interaction_fit = chemistry_fit * unweighted_fit
    interaction_valid = chemistry_valid * unweighted_valid

    assay_profile = remove_equivalent_assays(
        load_assay_profiles(dataset, profile_root, profile_split), dataset, metadata
    )
    assay_fit_design, assay_valid_design, nearest_assays, assay_audit = profile_design(
        fit_ids, valid_ids, assay_profile, fit.target.to_numpy(float)
    )
    assay_coverage_fit, assay_coverage_valid, assay_residual_fit, assay_residual_valid, _ = matrices(
        fit_ids, valid_ids, assay_profile
    )
    assay_fit = (assay_coverage_fit + assay_residual_fit) / 2
    assay_valid = (assay_coverage_valid + assay_residual_valid) / 2
    nearest_assay_profile = assay_profile[assay_profile.target.isin(nearest_assays)]
    if nearest_assays:
        (nearest_assay_coverage_fit, nearest_assay_coverage_valid,
         nearest_assay_residual_fit, nearest_assay_residual_valid, _) = matrices(
            fit_ids, valid_ids, nearest_assay_profile
        )
        nearest_assay_fit = (nearest_assay_coverage_fit + nearest_assay_residual_fit) / 2
        nearest_assay_valid = (nearest_assay_coverage_valid + nearest_assay_residual_valid) / 2
    else:
        nearest_assay_fit = np.zeros_like(chemistry_fit)
        nearest_assay_valid = np.zeros_like(chemistry_valid)
    dense_nearest_assay_fit, dense_nearest_assay_valid = dense_nearest_profile_kernel(
        fit_ids, valid_ids, assay_profile, nearest_assays
    )
    dense_nearest1_assay_fit, dense_nearest1_assay_valid = dense_nearest_profile_kernel(
        fit_ids, valid_ids, assay_profile, nearest_assays[:1]
    )
    imputed_nearest_assay_fit, imputed_nearest_assay_valid = imputed_nearest_profile_kernel(
        fit_ids, valid_ids, assay_profile, nearest_assays, chemistry_fit, chemistry_valid
    )
    audit["assay_profile_features"] = assay_audit["profile_targets"]
    audit["nearest_assays_at_assay_level"] = assay_audit["nearest_assays"]
    endpoint_profile, _ = remove_equivalent_targets(
        load_endpoint_profiles(dataset, profile_root, profile_split), dataset, metadata
    )
    endpoint_design_fit, endpoint_design_valid, nearest_endpoint_targets, endpoint_audit = (
        profile_design(fit_ids, valid_ids, endpoint_profile, fit.target.to_numpy(float))
    )
    (endpoint_coverage_fit, endpoint_coverage_valid,
     endpoint_residual_fit, endpoint_residual_valid, _) = matrices(
        fit_ids, valid_ids, endpoint_profile
    )
    endpoint_fit = (endpoint_coverage_fit + endpoint_residual_fit) / 2
    endpoint_valid = (endpoint_coverage_valid + endpoint_residual_valid) / 2
    nearest_endpoint_profile = endpoint_profile[
        endpoint_profile.target.isin(nearest_endpoint_targets)
    ]
    if nearest_endpoint_targets:
        (nearest_endpoint_coverage_fit, nearest_endpoint_coverage_valid,
         nearest_endpoint_residual_fit, nearest_endpoint_residual_valid, _) = matrices(
            fit_ids, valid_ids, nearest_endpoint_profile
        )
        nearest_endpoint_fit = (nearest_endpoint_coverage_fit + nearest_endpoint_residual_fit) / 2
        nearest_endpoint_valid = (nearest_endpoint_coverage_valid + nearest_endpoint_residual_valid) / 2
    else:
        nearest_endpoint_fit = np.zeros_like(chemistry_fit)
        nearest_endpoint_valid = np.zeros_like(chemistry_valid)
    audit["endpoint_profile_features"] = endpoint_audit["profile_targets"]
    audit["fit_only_high_gradient_fraction"] = float(fit_high_gradient.mean())
    audit["fit_only_high_gradient_max_weight"] = float(cliff_weights.max())
    qualitative_path = (
        qualitative_root / f"{dataset}.train_qualitative.jsonl.gz"
        if qualitative_root is not None else Path("__missing__")
    )
    if qualitative_path.exists():
        qualitative_profile, qualitative_excluded = remove_equivalent_targets(
            load_qualitative_profiles(dataset, qualitative_root), dataset, metadata
        )
        _, _, qualitative_nearest, qualitative_audit = profile_design(
            fit_ids, valid_ids, qualitative_profile, fit.target.to_numpy(float)
        )
        (qualitative_coverage_fit, qualitative_coverage_valid,
         qualitative_residual_fit, qualitative_residual_valid, _) = matrices(
            fit_ids, valid_ids, qualitative_profile
        )
        qualitative_fit = (qualitative_coverage_fit + qualitative_residual_fit) / 2
        qualitative_valid = (qualitative_coverage_valid + qualitative_residual_valid) / 2
        qualitative_nearest_profile = qualitative_profile[
            qualitative_profile.target.isin(qualitative_nearest)
        ]
        if qualitative_nearest:
            (qualitative_nearest_coverage_fit, qualitative_nearest_coverage_valid,
             qualitative_nearest_residual_fit, qualitative_nearest_residual_valid, _) = matrices(
                fit_ids, valid_ids, qualitative_nearest_profile
            )
            qualitative_nearest_fit = (
                qualitative_nearest_coverage_fit + qualitative_nearest_residual_fit
            ) / 2
            qualitative_nearest_valid = (
                qualitative_nearest_coverage_valid + qualitative_nearest_residual_valid
            ) / 2
        else:
            qualitative_nearest_fit = np.zeros_like(chemistry_fit)
            qualitative_nearest_valid = np.zeros_like(chemistry_valid)
        qualitative_assay = remove_equivalent_assays(
            load_qualitative_profiles(dataset, qualitative_root, assay_level=True),
            dataset, metadata
        )
        _, _, qualitative_nearest_assays, qualitative_assay_audit = profile_design(
            fit_ids, valid_ids, qualitative_assay, fit.target.to_numpy(float)
        )
        (qualitative_assay_coverage_fit, qualitative_assay_coverage_valid,
         qualitative_assay_residual_fit, qualitative_assay_residual_valid, _) = matrices(
            fit_ids, valid_ids, qualitative_assay
        )
        qualitative_assay_fit = (
            qualitative_assay_coverage_fit + qualitative_assay_residual_fit
        ) / 2
        qualitative_assay_valid = (
            qualitative_assay_coverage_valid + qualitative_assay_residual_valid
        ) / 2
        qualitative_dense_fit, qualitative_dense_valid = dense_nearest_profile_kernel(
            fit_ids, valid_ids, qualitative_profile, qualitative_nearest
        )
        qualitative_dense_assay_fit, qualitative_dense_assay_valid = (
            dense_nearest_profile_kernel(
                fit_ids, valid_ids, qualitative_assay, qualitative_nearest_assays
            )
        )
        audit["qualitative_profile_targets"] = qualitative_audit["profile_targets"]
        audit["qualitative_assays"] = qualitative_assay_audit["profile_targets"]
        audit["qualitative_equivalent_targets_excluded"] = len(qualitative_excluded)
        audit["qualitative_nearest_targets"] = json.dumps(qualitative_nearest)
        audit["qualitative_nearest_assays"] = json.dumps(qualitative_nearest_assays)
    else:
        qualitative_fit = qualitative_nearest_fit = np.zeros_like(chemistry_fit)
        qualitative_assay_fit = qualitative_dense_fit = np.zeros_like(chemistry_fit)
        qualitative_dense_assay_fit = np.zeros_like(chemistry_fit)
        qualitative_valid = qualitative_nearest_valid = np.zeros_like(chemistry_valid)
        qualitative_assay_valid = qualitative_dense_valid = np.zeros_like(chemistry_valid)
        qualitative_dense_assay_valid = np.zeros_like(chemistry_valid)
        audit["qualitative_profile_targets"] = 0
        audit["qualitative_assays"] = 0
        audit["qualitative_equivalent_targets_excluded"] = 0
        audit["qualitative_nearest_targets"] = "[]"
        audit["qualitative_nearest_assays"] = "[]"
    biological_targets = biological_nearest_targets(
        dataset, set(profile.target.astype(str)), metadata, sequence_kmers or {}
    )
    biological_profile = profile[profile.target.astype(str).isin(biological_targets)]
    if biological_targets:
        (biological_coverage_fit, biological_coverage_valid,
         biological_residual_fit, biological_residual_valid, _) = matrices(
            fit_ids, valid_ids, biological_profile
        )
        biological_fit = (biological_coverage_fit + biological_residual_fit) / 2
        biological_valid = (biological_coverage_valid + biological_residual_valid) / 2
    else:
        biological_fit = np.zeros_like(chemistry_fit)
        biological_valid = np.zeros_like(chemistry_valid)
    raw_assay_profile = filter_equivalent_assay_records(
        load_assay_profiles(dataset, profile_root), dataset, metadata
    )
    biological_assay_profile = raw_assay_profile[
        raw_assay_profile.target_entity.astype(str).isin(biological_targets)
    ][["molecule", "target", "value"]]
    if len(biological_assay_profile):
        (biological_assay_coverage_fit, biological_assay_coverage_valid,
         biological_assay_residual_fit, biological_assay_residual_valid, _) = matrices(
            fit_ids, valid_ids, biological_assay_profile
        )
        biological_assay_fit = (
            biological_assay_coverage_fit + biological_assay_residual_fit
        ) / 2
        biological_assay_valid = (
            biological_assay_coverage_valid + biological_assay_residual_valid
        ) / 2
    else:
        biological_assay_fit = np.zeros_like(chemistry_fit)
        biological_assay_valid = np.zeros_like(chemistry_valid)
    audit["biological_nearest_targets"] = json.dumps(biological_targets)
    need_alignment = (
        sequences and (candidate_names is None
                       or any("alignment" in name for name in candidate_names))
    )
    alignment_targets = local_alignment_nearest_targets(
        dataset, set(profile.target.astype(str)), metadata, sequences or {}
    ) if need_alignment else []
    alignment_profile = profile[profile.target.astype(str).isin(alignment_targets)]
    if alignment_targets:
        (alignment_coverage_fit, alignment_coverage_valid,
         alignment_residual_fit, alignment_residual_valid, _) = matrices(
            fit_ids, valid_ids, alignment_profile
        )
        alignment_fit = (alignment_coverage_fit + alignment_residual_fit) / 2
        alignment_valid = (alignment_coverage_valid + alignment_residual_valid) / 2
    else:
        alignment_fit = np.zeros_like(chemistry_fit)
        alignment_valid = np.zeros_like(chemistry_valid)
    alignment_assay_profile = raw_assay_profile[
        raw_assay_profile.target_entity.astype(str).isin(alignment_targets)
    ][["molecule", "target", "value"]]
    if len(alignment_assay_profile):
        (alignment_assay_coverage_fit, alignment_assay_coverage_valid,
         alignment_assay_residual_fit, alignment_assay_residual_valid, _) = matrices(
            fit_ids, valid_ids, alignment_assay_profile
        )
        alignment_assay_fit = (
            alignment_assay_coverage_fit + alignment_assay_residual_fit
        ) / 2
        alignment_assay_valid = (
            alignment_assay_coverage_valid + alignment_assay_residual_valid
        ) / 2
    else:
        alignment_assay_fit = np.zeros_like(chemistry_fit)
        alignment_assay_valid = np.zeros_like(chemistry_valid)
    audit["local_alignment_nearest_targets"] = json.dumps(alignment_targets)
    if bindingdb_profile is not None and bindingdb_metadata is not None:
        binding_profile, binding_excluded = filter_bindingdb_equivalent_targets(
            dataset, bindingdb_profile, bindingdb_metadata, metadata, sequences or {}
        )
        binding_fit_design, binding_valid_design, binding_nearest, binding_audit = (
            profile_design(
                fit.smiles.astype(str).tolist(), valid.smiles.astype(str).tolist(),
                binding_profile, fit.target.to_numpy(float)
            )
        )
        (binding_coverage_fit, binding_coverage_valid,
         binding_residual_fit, binding_residual_valid, _) = matrices(
            fit.smiles.astype(str).tolist(), valid.smiles.astype(str).tolist(),
            binding_profile
        )
        binding_fit = (binding_coverage_fit + binding_residual_fit) / 2
        binding_valid = (binding_coverage_valid + binding_residual_valid) / 2
        binding_nearest_profile = binding_profile[
            binding_profile.target.isin(binding_nearest)
        ]
        if binding_nearest:
            (binding_nearest_coverage_fit, binding_nearest_coverage_valid,
             binding_nearest_residual_fit, binding_nearest_residual_valid, _) = matrices(
                fit.smiles.astype(str).tolist(), valid.smiles.astype(str).tolist(),
                binding_nearest_profile
            )
            binding_nearest_fit = (
                binding_nearest_coverage_fit + binding_nearest_residual_fit
            ) / 2
            binding_nearest_valid = (
                binding_nearest_coverage_valid + binding_nearest_residual_valid
            ) / 2
        else:
            binding_nearest_fit = np.zeros_like(chemistry_fit)
            binding_nearest_valid = np.zeros_like(chemistry_valid)
        binding_dense_fit, binding_dense_valid = dense_nearest_profile_kernel(
            fit.smiles.astype(str).tolist(), valid.smiles.astype(str).tolist(),
            binding_profile, binding_nearest
        )
        audit["bindingdb_profile_systems"] = binding_audit["profile_targets"]
        audit["bindingdb_equivalent_systems_excluded"] = len(binding_excluded)
        audit["bindingdb_nearest_systems"] = json.dumps(binding_nearest)
    else:
        binding_fit = binding_nearest_fit = binding_dense_fit = np.zeros_like(chemistry_fit)
        binding_valid = binding_nearest_valid = binding_dense_valid = np.zeros_like(chemistry_valid)
        audit["bindingdb_profile_systems"] = 0
        audit["bindingdb_equivalent_systems_excluded"] = 0
        audit["bindingdb_nearest_systems"] = "[]"
    channel_fit = [chemistry_fit, unweighted_fit, nearest_fit, assay_fit, nearest_assay_fit]
    channel_valid = [chemistry_valid, unweighted_valid, nearest_valid,
                     assay_valid, nearest_assay_valid]
    channel_names = ["chemistry", "target_all", "target_nearest10",
                     "assay_all", "assay_nearest10"]
    alignment_weights = np.asarray([
        centered_kernel_alignment(kernel, fit.target.to_numpy(float))
        for kernel in channel_fit
    ], dtype=np.float64)
    if alignment_weights.sum() <= 0:
        alignment_weights[0] = 1.0
    alignment_weights /= alignment_weights.sum()
    aligned_multiview_fit = sum(
        float(weight) * kernel for weight, kernel in zip(alignment_weights, channel_fit)
    )
    aligned_multiview_valid = sum(
        float(weight) * kernel for weight, kernel in zip(alignment_weights, channel_valid)
    )
    audit["channel_alignment_weights"] = json.dumps(
        dict(zip(channel_names, alignment_weights.tolist())), sort_keys=True
    )
    transfer_fit_features, transfer_valid_features, transfer_audit = (
        crossfit_assay_transfer_features(
            assay_profile, fit_ids, valid_ids, fit.target.to_numpy(float)
        )
    )
    audit.update(transfer_audit)
    transfer_fit = (transfer_fit_features @ transfer_fit_features.T / 2).astype(np.float32)
    transfer_valid = (transfer_valid_features @ transfer_fit_features.T / 2).astype(np.float32)
    hierarchical_fit = (
        10 * chemistry_fit + unweighted_fit + nearest_fit + assay_fit + nearest_assay_fit
    ) / 14
    hierarchical_valid = (
        10 * chemistry_valid + unweighted_valid + nearest_valid
        + assay_valid + nearest_assay_valid
    ) / 14
    # Fixed reliability gate for the profile block.  It uses only whether a
    # molecule has observed fit-derived profile coordinates; c=n/(n+10) is a
    # global shrinkage rule, not a validation-fitted threshold.  Pairwise
    # gating preserves PSD because it is an outer product on both train and
    # train-query kernels.
    fit_profile_count = np.asarray(profile_fit.getnnz(axis=1), dtype=np.float32)
    valid_profile_count = np.asarray(profile_valid.getnnz(axis=1), dtype=np.float32)
    fit_profile_gate = fit_profile_count / (fit_profile_count + 10.0)
    valid_profile_gate = valid_profile_count / (valid_profile_count + 10.0)
    profile_block_fit = (
        unweighted_fit + biological_fit + assay_fit + biological_assay_fit
        + dense_nearest_fit + dense_nearest_assay_fit
        + chemistry_fit * dense_nearest_fit
        + chemistry_fit * dense_nearest_assay_fit
    ) / 8
    profile_block_valid = (
        unweighted_valid + biological_valid + assay_valid + biological_assay_valid
        + dense_nearest_valid + dense_nearest_assay_valid
        + chemistry_valid * dense_nearest_valid
        + chemistry_valid * dense_nearest_assay_valid
    ) / 8
    profile_no_biological_fit = (
        unweighted_fit + assay_fit + dense_nearest_fit + dense_nearest_assay_fit
        + chemistry_fit * dense_nearest_fit
        + chemistry_fit * dense_nearest_assay_fit
    ) / 6
    profile_no_biological_valid = (
        unweighted_valid + assay_valid + dense_nearest_valid + dense_nearest_assay_valid
        + chemistry_valid * dense_nearest_valid
        + chemistry_valid * dense_nearest_assay_valid
    ) / 6
    profile_no_biological_without_interactions_fit = (
        unweighted_fit + assay_fit + dense_nearest_fit + dense_nearest_assay_fit
    ) / 4
    profile_no_biological_without_interactions_valid = (
        unweighted_valid + assay_valid + dense_nearest_valid + dense_nearest_assay_valid
    ) / 4
    profile_entity_only_fit = (
        unweighted_fit + dense_nearest_fit + chemistry_fit * dense_nearest_fit
    ) / 3
    profile_entity_only_valid = (
        unweighted_valid + dense_nearest_valid + chemistry_valid * dense_nearest_valid
    ) / 3
    profile_assay_only_fit = (
        assay_fit + dense_nearest_assay_fit + chemistry_fit * dense_nearest_assay_fit
    ) / 3
    profile_assay_only_valid = (
        assay_valid + dense_nearest_assay_valid + chemistry_valid * dense_nearest_assay_valid
    ) / 3
    profile_raw_entity_assay_fit = (unweighted_fit + assay_fit) / 2
    profile_raw_entity_assay_valid = (unweighted_valid + assay_valid) / 2
    profile_sparse_fit = (
        unweighted_fit + biological_fit + assay_fit + biological_assay_fit
    ) / 4
    profile_sparse_valid = (
        unweighted_valid + biological_valid + assay_valid + biological_assay_valid
    ) / 4
    profile_without_interactions_fit = (
        unweighted_fit + biological_fit + assay_fit + biological_assay_fit
        + dense_nearest_fit + dense_nearest_assay_fit
    ) / 6
    profile_without_interactions_valid = (
        unweighted_valid + biological_valid + assay_valid + biological_assay_valid
        + dense_nearest_valid + dense_nearest_assay_valid
    ) / 6
    profile_dense_fit = (
        dense_nearest_fit + dense_nearest_assay_fit
        + chemistry_fit * dense_nearest_fit
        + chemistry_fit * dense_nearest_assay_fit
    ) / 4
    profile_dense_valid = (
        dense_nearest_valid + dense_nearest_assay_valid
        + chemistry_valid * dense_nearest_valid
        + chemistry_valid * dense_nearest_assay_valid
    ) / 4
    gated_profile_fit = profile_block_fit * (
        fit_profile_gate[:, None] * fit_profile_gate[None, :]
    )
    gated_profile_valid = profile_block_valid * (
        valid_profile_gate[:, None] * fit_profile_gate[None, :]
    )
    # Train-only hyperparameter sensitivity candidates.  These vary one
    # parameter inside each RAG stage while retaining every model component:
    # retrieval budget, dense-view bandwidth, memory fusion weight, and SVR C.
    # They are constructed only when explicitly requested by the CV driver.
    sensitivity_candidates: list[tuple[str, np.ndarray, np.ndarray]] = []
    sensitivity_c: dict[str, float] = {}
    requested = candidate_names or set()

    def memory_block(
        dense_entity_fit: np.ndarray, dense_entity_valid: np.ndarray,
        dense_assay_fit_: np.ndarray, dense_assay_valid_: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        fit_block = (
            unweighted_fit + assay_fit + dense_entity_fit + dense_assay_fit_
            + chemistry_fit * dense_entity_fit
            + chemistry_fit * dense_assay_fit_
        ) / 6
        valid_block = (
            unweighted_valid + assay_valid + dense_entity_valid + dense_assay_valid_
            + chemistry_valid * dense_entity_valid
            + chemistry_valid * dense_assay_valid_
        ) / 6
        return fit_block, valid_block

    default_mem_fit, default_mem_valid = memory_block(
        dense_nearest_fit, dense_nearest_valid,
        dense_nearest_assay_fit, dense_nearest_assay_valid,
    )

    for budget in (1, 2, 4, 6, 8, 10):
        name = f"sensitivity_retrieval_budget_{budget}"
        if name not in requested:
            continue
        ent_fit, ent_valid = dense_nearest_profile_kernel(
            fit_ids, valid_ids, profile, nearest_targets[:budget]
        )
        ass_fit, ass_valid = dense_nearest_profile_kernel(
            fit_ids, valid_ids, assay_profile, nearest_assays[:budget]
        )
        mem_fit, mem_valid = memory_block(ent_fit, ent_valid, ass_fit, ass_valid)
        sensitivity_candidates.append(
            (name, (10 * chemistry_fit + 8 * mem_fit) / 18,
             (10 * chemistry_valid + 8 * mem_valid) / 18)
        )

    for multiplier in (0.25, 0.5, 1.0, 2.0, 4.0):
        token = str(multiplier).replace(".", "p")
        name = f"sensitivity_bandwidth_{token}"
        if name not in requested:
            continue
        ent_fit, ent_valid = dense_nearest_profile_kernel(
            fit_ids, valid_ids, profile, nearest_targets,
            bandwidth_multiplier=multiplier,
        )
        ass_fit, ass_valid = dense_nearest_profile_kernel(
            fit_ids, valid_ids, assay_profile, nearest_assays,
            bandwidth_multiplier=multiplier,
        )
        mem_fit, mem_valid = memory_block(ent_fit, ent_valid, ass_fit, ass_valid)
        sensitivity_candidates.append(
            (name, (10 * chemistry_fit + 8 * mem_fit) / 18,
             (10 * chemistry_valid + 8 * mem_valid) / 18)
        )

    for memory_weight in (0.0, 0.2, 0.35, 8 / 18, 0.6, 0.8):
        token = f"{memory_weight:.3f}".rstrip("0").rstrip(".").replace(".", "p")
        name = f"sensitivity_memory_weight_{token}"
        if name not in requested:
            continue
        sensitivity_candidates.append(
            (name,
             (1 - memory_weight) * chemistry_fit + memory_weight * default_mem_fit,
             (1 - memory_weight) * chemistry_valid + memory_weight * default_mem_valid)
        )

    default_fused_fit = (10 * chemistry_fit + 8 * default_mem_fit) / 18
    default_fused_valid = (10 * chemistry_valid + 8 * default_mem_valid) / 18
    for penalty in (0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
        token = str(penalty).replace(".", "p")
        name = f"sensitivity_svr_c_{token}"
        if name not in requested:
            continue
        sensitivity_candidates.append((name, default_fused_fit, default_fused_valid))
        sensitivity_c[name] = penalty
    aligned_fit = (profile_fit @ profile_fit.T).toarray().astype(np.float32)
    aligned_valid = (profile_valid @ profile_fit.T).toarray().astype(np.float32)
    candidates = (
        ("collision_free_binary_count_kernel", chemistry_fit, chemistry_valid),
        ("physical_free_energy_kernel", physical_fit, physical_valid),
        (
            "collision_free_plus_physical_free_energy_10to4",
            (10 * chemistry_fit + sum(physical_fit_views)) / 14,
            (10 * chemistry_valid + sum(physical_valid_views)) / 14,
        ),
        (
            "collision_free_plus_local_physical_free_energy_10to4",
            (10 * chemistry_fit + 4 * physical_local_fit) / 14,
            (10 * chemistry_valid + 4 * physical_local_valid) / 14,
        ),
        (
            "collision_free_plus_global_and_local_physics_10to4to4",
            (10 * chemistry_fit + sum(physical_fit_views) + 4 * physical_local_fit) / 18,
            (10 * chemistry_valid + sum(physical_valid_views) + 4 * physical_local_valid) / 18,
        ),
        ("target_aligned_profile_kernel", aligned_fit, aligned_valid),
        (
            "collision_free_plus_entity_filtered_profile_10to1",
            (10 * chemistry_fit + unweighted_fit) / 11,
            (10 * chemistry_valid + unweighted_valid) / 11,
        ),
        (
            "collision_free_plus_nearest10_profile_10to1",
            (10 * chemistry_fit + nearest_fit) / 11,
            (10 * chemistry_valid + nearest_valid) / 11,
        ),
        (
            "collision_free_plus_all_and_nearest10_profile_10to1to1",
            (10 * chemistry_fit + unweighted_fit + nearest_fit) / 12,
            (10 * chemistry_valid + unweighted_valid + nearest_valid) / 12,
        ),
        (
            "collision_free_plus_dense_nearest_profiles_10to1to1",
            (10 * chemistry_fit + dense_nearest_fit + dense_nearest_assay_fit) / 12,
            (10 * chemistry_valid + dense_nearest_valid + dense_nearest_assay_valid) / 12,
        ),
        (
            "collision_free_plus_multiscale_profile_10to1to1to1",
            (10 * chemistry_fit + unweighted_fit + nearest_fit + nearest1_fit) / 13,
            (10 * chemistry_valid + unweighted_valid + nearest_valid + nearest1_valid) / 13,
        ),
        (
            "collision_free_plus_profile_interaction_10to1",
            (10 * chemistry_fit + interaction_fit) / 11,
            (10 * chemistry_valid + interaction_valid) / 11,
        ),
        (
            "collision_free_plus_multiscale_profile_interaction_10to1to1to1",
            (10 * chemistry_fit + unweighted_fit + nearest_fit + interaction_fit) / 13,
            (10 * chemistry_valid + unweighted_valid + nearest_valid + interaction_valid) / 13,
        ),
        (
            "collision_free_plus_assay_profile_10to1",
            (10 * chemistry_fit + assay_fit) / 11,
            (10 * chemistry_valid + assay_valid) / 11,
        ),
        (
            "collision_free_plus_hierarchical_assay_profile_10to1to1to1to1",
            (10 * chemistry_fit + unweighted_fit + nearest_fit
             + assay_fit + nearest_assay_fit) / 14,
            (10 * chemistry_valid + unweighted_valid + nearest_valid
             + assay_valid + nearest_assay_valid) / 14,
        ),
        (
            "aligned_hierarchical_assay_profile_kernel",
            aligned_multiview_fit,
            aligned_multiview_valid,
        ),
        (
            "collision_free_plus_hierarchical_profile_transfer_10to1to1to1to1to1",
            (10 * chemistry_fit + unweighted_fit + nearest_fit
             + assay_fit + nearest_assay_fit + transfer_fit) / 15,
            (10 * chemistry_valid + unweighted_valid + nearest_valid
             + assay_valid + nearest_assay_valid + transfer_valid) / 15,
        ),
        (
            "hierarchical_profile_plus_assay_transfer_equal",
            (hierarchical_fit + transfer_fit) / 2,
            (hierarchical_valid + transfer_valid) / 2,
        ),
        (
            "assay_susceptibility_transfer_kernel",
            transfer_fit,
            transfer_valid,
        ),
        (
            "collision_free_plus_assay_susceptibility_10to1",
            (10 * chemistry_fit + transfer_fit) / 11,
            (10 * chemistry_valid + transfer_valid) / 11,
        ),
        (
            "dense_full_interaction_profile_reliability_gated_10to8",
            (10 * chemistry_fit + 8 * gated_profile_fit) / 18,
            (10 * chemistry_valid + 8 * gated_profile_valid) / 18,
        ),
        (
            "collision_free_plus_action_profile_10to1",
            (10 * chemistry_fit + action_profile_fit) / 11,
            (10 * chemistry_valid + action_profile_valid) / 11,
        ),
        (
            "collision_free_plus_action_profile_interaction_10to1",
            (10 * chemistry_fit + chemistry_fit * action_profile_fit) / 11,
            (10 * chemistry_valid + chemistry_valid * action_profile_valid) / 11,
        ),
        (
            "dense_full_plus_action_profile_18to1",
            (
                10 * chemistry_fit + unweighted_fit + biological_fit + assay_fit
                + biological_assay_fit + dense_nearest_fit + dense_nearest_assay_fit
                + chemistry_fit * dense_nearest_fit
                + chemistry_fit * dense_nearest_assay_fit + action_profile_fit
            ) / 19,
            (
                10 * chemistry_valid + unweighted_valid + biological_valid + assay_valid
                + biological_assay_valid + dense_nearest_valid + dense_nearest_assay_valid
                + chemistry_valid * dense_nearest_valid
                + chemistry_valid * dense_nearest_assay_valid + action_profile_valid
            ) / 19,
        ),
        (
            "dense_full_plus_action_profile_interaction_18to1",
            (
                10 * chemistry_fit + unweighted_fit + biological_fit + assay_fit
                + biological_assay_fit + dense_nearest_fit + dense_nearest_assay_fit
                + chemistry_fit * dense_nearest_fit
                + chemistry_fit * dense_nearest_assay_fit
                + chemistry_fit * action_profile_fit
            ) / 19,
            (
                10 * chemistry_valid + unweighted_valid + biological_valid + assay_valid
                + biological_assay_valid + dense_nearest_valid + dense_nearest_assay_valid
                + chemistry_valid * dense_nearest_valid
                + chemistry_valid * dense_nearest_assay_valid
                + chemistry_valid * action_profile_valid
            ) / 19,
        ),
        (
            "collision_free_plus_endpoint_hierarchical_profile_10to1to1to1to1",
            (10 * chemistry_fit + endpoint_fit + nearest_endpoint_fit
             + assay_fit + nearest_assay_fit) / 14,
            (10 * chemistry_valid + endpoint_valid + nearest_endpoint_valid
             + assay_valid + nearest_assay_valid) / 14,
        ),
        (
            "collision_free_plus_full_endpoint_hierarchical_profile_10to1to1to1to1to1to1",
            (10 * chemistry_fit + unweighted_fit + nearest_fit
             + endpoint_fit + nearest_endpoint_fit + assay_fit + nearest_assay_fit) / 16,
            (10 * chemistry_valid + unweighted_valid + nearest_valid
             + endpoint_valid + nearest_endpoint_valid + assay_valid + nearest_assay_valid) / 16,
        ),
        (
            "collision_free_plus_biological_hierarchical_profile_10to1to1to1to1",
            (10 * chemistry_fit + unweighted_fit + biological_fit
             + assay_fit + biological_assay_fit) / 14,
            (10 * chemistry_valid + unweighted_valid + biological_valid
             + assay_valid + biological_assay_valid) / 14,
        ),
        (
            "collision_free_plus_alignment_hierarchical_profile_10to1to1to1to1",
            (10 * chemistry_fit + unweighted_fit + alignment_fit
             + assay_fit + alignment_assay_fit) / 14,
            (10 * chemistry_valid + unweighted_valid + alignment_valid
             + assay_valid + alignment_assay_valid) / 14,
        ),
        (
            "dense_alignment_hierarchical_kernel_10to1to1to1to1to1to1",
            (10 * chemistry_fit + unweighted_fit + alignment_fit + assay_fit
             + alignment_assay_fit + dense_nearest_fit + dense_nearest_assay_fit) / 16,
            (10 * chemistry_valid + unweighted_valid + alignment_valid + assay_valid
             + alignment_assay_valid + dense_nearest_valid + dense_nearest_assay_valid) / 16,
        ),
        (
            "dense_biological_hierarchical_kernel_10to1to1to1to1to1to1",
            (10 * chemistry_fit + unweighted_fit + biological_fit + assay_fit
             + biological_assay_fit + dense_nearest_fit + dense_nearest_assay_fit) / 16,
            (10 * chemistry_valid + unweighted_valid + biological_valid + assay_valid
             + biological_assay_valid + dense_nearest_valid + dense_nearest_assay_valid) / 16,
        ),
        (
            "dense_interaction_biological_hierarchical_kernel_10to1to1to1to1to1to1",
            (10 * chemistry_fit + unweighted_fit + biological_fit + assay_fit
             + biological_assay_fit + chemistry_fit * dense_nearest_fit
             + chemistry_fit * dense_nearest_assay_fit) / 16,
            (10 * chemistry_valid + unweighted_valid + biological_valid + assay_valid
             + biological_assay_valid + chemistry_valid * dense_nearest_valid
             + chemistry_valid * dense_nearest_assay_valid) / 16,
        ),
        (
            "dense_full_interaction_biological_hierarchical_kernel_10to8",
            (10 * chemistry_fit + unweighted_fit + biological_fit + assay_fit
             + biological_assay_fit + dense_nearest_fit + dense_nearest_assay_fit
             + chemistry_fit * dense_nearest_fit
             + chemistry_fit * dense_nearest_assay_fit) / 18,
            (10 * chemistry_valid + unweighted_valid + biological_valid + assay_valid
             + biological_assay_valid + dense_nearest_valid + dense_nearest_assay_valid
             + chemistry_valid * dense_nearest_valid
             + chemistry_valid * dense_nearest_assay_valid) / 18,
        ),
        (
            "response_profile_block_kernel",
            profile_block_fit,
            profile_block_valid,
        ),
        (
            "chemistry_profile_kernel_10to8",
            (10 * chemistry_fit + 8 * profile_block_fit) / 18,
            (10 * chemistry_valid + 8 * profile_block_valid) / 18,
        ),
        (
            "chemistry_profile_no_biological_neighborhood_10to8",
            (10 * chemistry_fit + 8 * profile_no_biological_fit) / 18,
            (10 * chemistry_valid + 8 * profile_no_biological_valid) / 18,
        ),
        (
            "responsekernel_final",
            (10 * chemistry_fit + 8 * profile_no_biological_fit) / 18,
            (10 * chemistry_valid + 8 * profile_no_biological_valid) / 18,
        ),
        (
            "responsekernel_without_interactions",
            (10 * chemistry_fit + 8 * profile_no_biological_without_interactions_fit) / 18,
            (10 * chemistry_valid + 8 * profile_no_biological_without_interactions_valid) / 18,
        ),
        (
            "responsekernel_entity_only",
            (10 * chemistry_fit + 8 * profile_entity_only_fit) / 18,
            (10 * chemistry_valid + 8 * profile_entity_only_valid) / 18,
        ),
        (
            "responsekernel_assay_only",
            (10 * chemistry_fit + 8 * profile_assay_only_fit) / 18,
            (10 * chemistry_valid + 8 * profile_assay_only_valid) / 18,
        ),
        (
            "responsekernel_raw_profiles_only",
            (10 * chemistry_fit + 8 * profile_raw_entity_assay_fit) / 18,
            (10 * chemistry_valid + 8 * profile_raw_entity_assay_valid) / 18,
        ),
        (
            "chemistry_profile_sparse_only_10to8",
            (10 * chemistry_fit + 8 * profile_sparse_fit) / 18,
            (10 * chemistry_valid + 8 * profile_sparse_valid) / 18,
        ),
        (
            "chemistry_profile_no_chemistry_interactions_10to8",
            (10 * chemistry_fit + 8 * profile_without_interactions_fit) / 18,
            (10 * chemistry_valid + 8 * profile_without_interactions_valid) / 18,
        ),
        (
            "chemistry_profile_dense_only_10to8",
            (10 * chemistry_fit + 8 * profile_dense_fit) / 18,
            (10 * chemistry_valid + 8 * profile_dense_valid) / 18,
        ),
        (
            "chemistry_profile_physics_kernel_10to8to4",
            (10 * chemistry_fit + 8 * profile_block_fit + 4 * physical_local_fit) / 22,
            (10 * chemistry_valid + 8 * profile_block_valid + 4 * physical_local_valid) / 22,
        ),
        (
            "chemistry_profile_susceptibility_kernel_10to8to1",
            (10 * chemistry_fit + 8 * profile_block_fit
             + chemistry_fit * action_profile_fit) / 19,
            (10 * chemistry_valid + 8 * profile_block_valid
             + chemistry_valid * action_profile_valid) / 19,
        ),
        (
            "combined_positive_signal_kernel_10chem_8profile_4physics_1susceptibility",
            (10 * chemistry_fit + 8 * profile_block_fit + 4 * physical_local_fit
             + chemistry_fit * action_profile_fit) / 23,
            (10 * chemistry_valid + 8 * profile_block_valid + 4 * physical_local_valid
             + chemistry_valid * action_profile_valid) / 23,
        ),
        (
            "bindingdb_profile_kernel_10to1to1to1",
            (10 * chemistry_fit + binding_fit + binding_nearest_fit
             + binding_dense_fit) / 13,
            (10 * chemistry_valid + binding_valid + binding_nearest_valid
             + binding_dense_valid) / 13,
        ),
        (
            "chembl_bindingdb_dense_interaction_kernel_10to11",
            (10 * chemistry_fit + unweighted_fit + biological_fit + assay_fit
             + biological_assay_fit + dense_nearest_fit + dense_nearest_assay_fit
             + chemistry_fit * dense_nearest_fit
             + chemistry_fit * dense_nearest_assay_fit + binding_fit
             + binding_nearest_fit + binding_dense_fit) / 21,
            (10 * chemistry_valid + unweighted_valid + biological_valid + assay_valid
             + biological_assay_valid + dense_nearest_valid + dense_nearest_assay_valid
             + chemistry_valid * dense_nearest_valid
             + chemistry_valid * dense_nearest_assay_valid + binding_valid
             + binding_nearest_valid + binding_dense_valid) / 21,
        ),
        (
            "qualitative_profile_kernel_10to5",
            (10 * chemistry_fit + qualitative_fit + qualitative_nearest_fit
             + qualitative_assay_fit + qualitative_dense_fit
             + qualitative_dense_assay_fit) / 15,
            (10 * chemistry_valid + qualitative_valid + qualitative_nearest_valid
             + qualitative_assay_valid + qualitative_dense_valid
             + qualitative_dense_assay_valid) / 15,
        ),
        (
            "quantitative_qualitative_dense_interaction_kernel_10to13",
            (10 * chemistry_fit + unweighted_fit + biological_fit + assay_fit
             + biological_assay_fit + dense_nearest_fit + dense_nearest_assay_fit
             + chemistry_fit * dense_nearest_fit
             + chemistry_fit * dense_nearest_assay_fit + qualitative_fit
             + qualitative_nearest_fit + qualitative_assay_fit
             + qualitative_dense_fit + qualitative_dense_assay_fit) / 23,
            (10 * chemistry_valid + unweighted_valid + biological_valid + assay_valid
             + biological_assay_valid + dense_nearest_valid + dense_nearest_assay_valid
             + chemistry_valid * dense_nearest_valid
             + chemistry_valid * dense_nearest_assay_valid + qualitative_valid
             + qualitative_nearest_valid + qualitative_assay_valid
             + qualitative_dense_valid + qualitative_dense_assay_valid) / 23,
        ),
        (
            "dense_top1_biological_hierarchical_kernel_10to1to1to1to1to1to1",
            (10 * chemistry_fit + unweighted_fit + biological_fit + assay_fit
             + biological_assay_fit + dense_nearest1_fit
             + dense_nearest1_assay_fit) / 16,
            (10 * chemistry_valid + unweighted_valid + biological_valid + assay_valid
             + biological_assay_valid + dense_nearest1_valid
             + dense_nearest1_assay_valid) / 16,
        ),
        (
            "dense_top1_interaction_biological_hierarchical_kernel_10to1to1to1to1to1to1",
            (10 * chemistry_fit + unweighted_fit + biological_fit + assay_fit
             + biological_assay_fit + chemistry_fit * dense_nearest1_fit
             + chemistry_fit * dense_nearest1_assay_fit) / 16,
            (10 * chemistry_valid + unweighted_valid + biological_valid + assay_valid
             + biological_assay_valid + chemistry_valid * dense_nearest1_valid
             + chemistry_valid * dense_nearest1_assay_valid) / 16,
        ),
        (
            "imputed_biological_hierarchical_kernel_10to1to1to1to1to1to1",
            (10 * chemistry_fit + unweighted_fit + biological_fit + assay_fit
             + biological_assay_fit + imputed_nearest_fit
             + imputed_nearest_assay_fit) / 16,
            (10 * chemistry_valid + unweighted_valid + biological_valid + assay_valid
             + biological_assay_valid + imputed_nearest_valid
             + imputed_nearest_assay_valid) / 16,
        ),
        (
            "balanced_cliff_biological_hierarchical_kernel",
            (10 * chemistry_fit + unweighted_fit + biological_fit
             + assay_fit + biological_assay_fit) / 14,
            (10 * chemistry_valid + unweighted_valid + biological_valid
             + assay_valid + biological_assay_valid) / 14,
        ),
        (
            "physical_biological_hierarchical_kernel_10to4to1to1to1to1",
            (10 * chemistry_fit + sum(physical_fit_views) + unweighted_fit
             + biological_fit + assay_fit + biological_assay_fit) / 18,
            (10 * chemistry_valid + sum(physical_valid_views) + unweighted_valid
             + biological_valid + assay_valid + biological_assay_valid) / 18,
        ),
        (
            "local_physical_biological_hierarchical_kernel_10to4to1to1to1to1",
            (10 * chemistry_fit + 4 * physical_local_fit + unweighted_fit
             + biological_fit + assay_fit + biological_assay_fit) / 18,
            (10 * chemistry_valid + 4 * physical_local_valid + unweighted_valid
             + biological_valid + assay_valid + biological_assay_valid) / 18,
        ),
        (
            "collision_free_plus_full_biological_hierarchical_profile_10to1to1to1to1to1to1",
            (10 * chemistry_fit + unweighted_fit + nearest_fit + biological_fit
             + assay_fit + nearest_assay_fit + biological_assay_fit) / 16,
            (10 * chemistry_valid + unweighted_valid + nearest_valid + biological_valid
             + assay_valid + nearest_assay_valid + biological_assay_valid) / 16,
        ),
        (
            "collision_free_plus_target_aligned_profile_10to1",
            (10 * chemistry_fit + aligned_fit) / 11,
            (10 * chemistry_valid + aligned_valid) / 11,
        ),
    ) + tuple(sensitivity_candidates)
    rows = []
    for name, train_kernel, valid_kernel in candidates:
        if candidate_names is not None and name not in candidate_names:
            continue
        model = SVR(C=sensitivity_c.get(name, 10.0), epsilon=0.1, kernel="precomputed")
        sample_weight = cliff_weights if name.startswith("balanced_cliff_") else None
        model.fit(train_kernel, fit.target.to_numpy(float), sample_weight=sample_weight)
        prediction = model.predict(valid_kernel)
        if predictions_dir is not None:
            predictions_dir.mkdir(parents=True, exist_ok=True)
            prediction_frame = valid[
                ["dataset", "smiles", "target", "cliff_mol", "split"]
            ].copy()
            prediction_frame["prediction"] = prediction
            prediction_frame["model"] = name
            prediction_frame["observed_response_features"] = valid_profile_count
            prediction_frame["nearest_chemistry_similarity"] = chemistry_valid.max(axis=1)
            prediction_frame["nearest_response_similarity"] = profile_block_valid.max(axis=1)
            prediction_frame.to_csv(
                predictions_dir / f"{dataset}.{name}.predictions.csv", index=False
            )
        if artifacts_dir is not None:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            with (artifacts_dir / f"{dataset}.svr.pkl").open("wb") as handle:
                pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
            np.savez_compressed(
                artifacts_dir / f"{dataset}.train_kernel.npz",
                kernel=train_kernel.astype(np.float32),
                target=fit.target.to_numpy(float),
                smiles=fit.smiles.to_numpy(str),
            )
        rows.append({
            "dataset": dataset, "model": name, **audit,
            **regression_metrics(valid.target.to_numpy(float), prediction,
                                 valid.cliff_mol.to_numpy(bool)),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(MOLECULEACE_DATASETS))
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Optional frozen candidate subset; representation construction is unchanged.",
    )
    parser.add_argument("--profile-root", type=Path,
                        default=Path("data/chembl_offtarget_profiles"))
    parser.add_argument("--target-metadata", type=Path,
                        default=Path("data/chembl_target_metadata.jsonl.gz"))
    parser.add_argument("--sequence-cache", type=Path,
                        default=Path("data/uniprot_target_sequences.json.gz"))
    parser.add_argument("--bindingdb-profile", type=Path,
                        default=Path("data/bindingdb_moleculeace_profiles.jsonl.gz"))
    parser.add_argument("--bindingdb-metadata", type=Path,
                        default=Path("data/bindingdb_system_metadata.jsonl.gz"))
    parser.add_argument("--qualitative-root", type=Path,
                        default=Path("data/chembl_qualitative_profiles"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/moleculeace_profile_alignment_validation_v1"))
    args = parser.parse_args()
    metadata = load_target_metadata(args.target_metadata)
    sequence_kmers = load_sequence_kmers(args.sequence_cache) if args.sequence_cache.exists() else {}
    sequences = load_sequences(args.sequence_cache) if args.sequence_cache.exists() else {}
    if args.bindingdb_profile.exists() and args.bindingdb_metadata.exists():
        bindingdb_profile, bindingdb_metadata = load_bindingdb_profiles(
            args.bindingdb_profile, args.bindingdb_metadata
        )
    else:
        bindingdb_profile, bindingdb_metadata = None, None
    rows = [row for dataset in args.datasets
            for row in run_one(
                dataset, args.profile_root, metadata, sequence_kmers, sequences,
                bindingdb_profile, bindingdb_metadata, args.qualitative_root,
                candidate_names=set(args.models) if args.models else None,
            )]
    summary = pd.DataFrame(rows).sort_values(["model", "dataset"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "summary.validation.csv", index=False)
    macro = summary.groupby("model")[["rmse", "cliff_rmse", "noncliff_rmse"]].mean()
    payload = {name: {key: float(value) for key, value in row.items()}
               for name, row in macro.iterrows()}
    payload["protocol"] = (
        "fit-only null-corrected squared-correlation profile metric; "
        "10 exact chemical views + 1 aligned profile view; no search"
    )
    (args.output_dir / "aggregate.validation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
