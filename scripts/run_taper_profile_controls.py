from __future__ import annotations

"""Permutation and leakage controls for the frozen TAPER kernel.

The permutation control reassigns complete profile representations among
molecules within the fitting and validation partitions.  It preserves each
partition's profile multiset, coverage distribution, feature dimensionality,
kernel values, and the exact kernel construction while destroying the link
between a molecule and its biological response profile.

The leakage stress test starts from an archive in which the benchmark endpoint
has deliberately been re-injected.  Strict biological exclusion, normalized
target-name exclusion, and no exclusion are then compared under the same folds.
The no-exclusion condition is intentionally contaminated and is never a valid
predictive model.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVR

from molcliff.data import MOLECULEACE_DATASETS, load_moleculeace
from molcliff.metrics import regression_metrics
from validate_moleculeace_offtarget_kernel import load_profiles, matrices, molecule_mapping
from validate_moleculeace_operator_action_kernel import exact_kernel
from validate_moleculeace_profile_alignment_kernel import (
    biological_identity,
    dense_nearest_profile_kernel,
    load_assay_profiles,
    load_target_metadata,
    profile_design,
)


MODEL_ORDER = (
    "chemistry_only",
    "taper_shuffled_profile",
    "taper_strict_exclusion",
    "taper_target_name_only",
    "taper_no_exclusion",
)


def fold_split(frame: pd.DataFrame, fold: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool = frame[frame.split.eq("train")].reset_index(drop=True)
    indices = np.arange(len(pool))
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fit_index, valid_index = list(splitter.split(indices, pool.cliff_mol.to_numpy()))[fold]
    return (
        pool.iloc[np.sort(fit_index)].reset_index(drop=True),
        pool.iloc[np.sort(valid_index)].reset_index(drop=True),
    )


def target_ids_to_exclude(
    available: set[str], dataset: str, metadata: dict[str, dict], mode: str,
) -> set[str]:
    if mode == "none":
        return set()
    benchmark_id = dataset.rsplit("_", 1)[0]
    benchmark = metadata[benchmark_id]
    benchmark_accessions, benchmark_genes, benchmark_name = biological_identity(benchmark)
    excluded: set[str] = set()
    for target_id in available:
        record = metadata.get(str(target_id))
        if record is None:
            raise KeyError(f"target metadata lacks {target_id}")
        accessions, genes, name = biological_identity(record)
        if mode == "name":
            match = bool(benchmark_name and name == benchmark_name)
        elif mode == "strict":
            match = bool(
                target_id == benchmark_id
                or benchmark_accessions.intersection(accessions)
                or benchmark_genes.intersection(genes)
                or (benchmark_name and name == benchmark_name)
            )
        else:
            raise ValueError(f"unknown exclusion mode: {mode}")
        if match:
            excluded.add(str(target_id))
    return excluded


def inject_benchmark_endpoint(
    dataset: str,
    fit: pd.DataFrame,
    valid: pd.DataFrame,
    fit_ids: list[str],
    valid_ids: list[str],
    entity_profile: pd.DataFrame,
    assay_profile: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    benchmark_id = dataset.rsplit("_", 1)[0]
    molecule_ids = fit_ids + valid_ids
    values = np.concatenate([
        fit.target.to_numpy(float), valid.target.to_numpy(float),
    ])
    direct_entity = pd.DataFrame({
        "molecule": molecule_ids,
        "target": benchmark_id,
        "value": values,
    })
    direct_assay = pd.DataFrame({
        "molecule": molecule_ids,
        "target_entity": benchmark_id,
        "target": f"BENCHMARK_ENDPOINT::{dataset}",
        "value": values,
    })
    entity = pd.concat([entity_profile, direct_entity], ignore_index=True)
    entity = entity.groupby(["molecule", "target"], as_index=False).value.median()
    assay = pd.concat([assay_profile, direct_assay], ignore_index=True)
    assay = assay.groupby(
        ["molecule", "target_entity", "target"], as_index=False
    ).value.median()
    return entity, assay


def filter_profiles(
    entity: pd.DataFrame,
    assay: pd.DataFrame,
    dataset: str,
    metadata: dict[str, dict],
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    available = set(entity.target.astype(str)).union(assay.target_entity.astype(str))
    excluded = target_ids_to_exclude(available, dataset, metadata, mode)
    filtered_entity = entity[~entity.target.astype(str).isin(excluded)].copy()
    filtered_assay = assay[~assay.target_entity.astype(str).isin(excluded)].copy()
    return filtered_entity, filtered_assay, sorted(excluded)


def profile_components(
    fit_ids: list[str],
    valid_ids: list[str],
    y_fit: np.ndarray,
    entity_profile: pd.DataFrame,
    assay_profile: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    _, _, entity_selected, entity_audit = profile_design(
        fit_ids, valid_ids, entity_profile, y_fit
    )
    ent_cov_fit, ent_cov_valid, ent_val_fit, ent_val_valid, ent_dim = matrices(
        fit_ids, valid_ids, entity_profile
    )
    entity_sparse_fit = (ent_cov_fit + ent_val_fit) / 2
    entity_sparse_valid = (ent_cov_valid + ent_val_valid) / 2
    entity_dense_fit, entity_dense_valid = dense_nearest_profile_kernel(
        fit_ids, valid_ids, entity_profile, entity_selected
    )

    _, _, assay_selected, assay_audit = profile_design(
        fit_ids, valid_ids, assay_profile, y_fit
    )
    ass_cov_fit, ass_cov_valid, ass_val_fit, ass_val_valid, ass_dim = matrices(
        fit_ids, valid_ids, assay_profile
    )
    assay_sparse_fit = (ass_cov_fit + ass_val_fit) / 2
    assay_sparse_valid = (ass_cov_valid + ass_val_valid) / 2
    assay_dense_fit, assay_dense_valid = dense_nearest_profile_kernel(
        fit_ids, valid_ids, assay_profile, assay_selected
    )
    components = {
        "entity_sparse_fit": entity_sparse_fit,
        "entity_sparse_valid": entity_sparse_valid,
        "assay_sparse_fit": assay_sparse_fit,
        "assay_sparse_valid": assay_sparse_valid,
        "entity_dense_fit": entity_dense_fit,
        "entity_dense_valid": entity_dense_valid,
        "assay_dense_fit": assay_dense_fit,
        "assay_dense_valid": assay_dense_valid,
    }
    audit = {
        "entity_dimensions": int(ent_dim),
        "assay_dimensions": int(ass_dim),
        "selected_entity_dimensions": int(len(entity_selected)),
        "selected_assay_dimensions": int(len(assay_selected)),
        "positive_entity_dimensions": int(entity_audit["positive_weight_targets"]),
        "positive_assay_dimensions": int(assay_audit["positive_weight_targets"]),
    }
    return components, audit


def assemble_kernel(
    chemistry_fit: np.ndarray,
    chemistry_valid: np.ndarray,
    components: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    profile_fit = (
        components["entity_sparse_fit"]
        + components["assay_sparse_fit"]
        + components["entity_dense_fit"]
        + components["assay_dense_fit"]
        + chemistry_fit * components["entity_dense_fit"]
        + chemistry_fit * components["assay_dense_fit"]
    ) / 6
    profile_valid = (
        components["entity_sparse_valid"]
        + components["assay_sparse_valid"]
        + components["entity_dense_valid"]
        + components["assay_dense_valid"]
        + chemistry_valid * components["entity_dense_valid"]
        + chemistry_valid * components["assay_dense_valid"]
    ) / 6
    return (
        (10 * chemistry_fit + 8 * profile_fit) / 18,
        (10 * chemistry_valid + 8 * profile_valid) / 18,
    )


def permute_components(
    components: dict[str, np.ndarray], fit_permutation: np.ndarray,
    valid_permutation: np.ndarray,
) -> dict[str, np.ndarray]:
    shuffled: dict[str, np.ndarray] = {}
    for view in ("entity_sparse", "assay_sparse", "entity_dense", "assay_dense"):
        train = components[f"{view}_fit"]
        query = components[f"{view}_valid"]
        shuffled[f"{view}_fit"] = train[np.ix_(fit_permutation, fit_permutation)]
        shuffled[f"{view}_valid"] = query[np.ix_(valid_permutation, fit_permutation)]
        if not np.allclose(np.sort(train.ravel()), np.sort(shuffled[f"{view}_fit"].ravel())):
            raise AssertionError(f"{view}: training kernel distribution changed")
        if not np.allclose(np.sort(query.ravel()), np.sort(shuffled[f"{view}_valid"].ravel())):
            raise AssertionError(f"{view}: validation kernel distribution changed")
    return shuffled


def deterministic_rng(dataset: str, fold: int, seed: int) -> np.random.Generator:
    token = hashlib.sha256(f"{dataset}|{fold}|{seed}|TAPER-shuffle".encode()).digest()
    return np.random.default_rng(int.from_bytes(token[:8], "little"))


def evaluate(
    dataset: str,
    model_name: str,
    train_kernel: np.ndarray,
    valid_kernel: np.ndarray,
    fit: pd.DataFrame,
    valid: pd.DataFrame,
    fold: int,
    seed: int,
    audit: dict[str, object],
) -> dict[str, object]:
    model = SVR(C=10.0, epsilon=0.1, kernel="precomputed")
    model.fit(train_kernel, fit.target.to_numpy(float))
    prediction = model.predict(valid_kernel)
    return {
        "dataset": dataset,
        "fold": int(fold),
        "seed": int(seed),
        "model": model_name,
        **audit,
        **regression_metrics(
            valid.target.to_numpy(float), prediction, valid.cliff_mol.to_numpy(bool)
        ),
    }


def run_fold(
    dataset: str,
    fold: int,
    seed: int,
    profile_root: Path,
    metadata: dict[str, dict],
) -> list[dict[str, object]]:
    fit, valid = fold_split(load_moleculeace(dataset), fold, seed)
    chemistry_fit, chemistry_valid = exact_kernel(fit, valid)
    mapping = molecule_mapping(dataset)
    fit_ids = [mapping[str(smiles)] for smiles in fit.smiles]
    valid_ids = [mapping[str(smiles)] for smiles in valid.smiles]

    raw_entity = load_profiles(dataset, profile_root, "train")
    raw_assay = load_assay_profiles(dataset, profile_root, "train")
    raw_entity, raw_assay = inject_benchmark_endpoint(
        dataset, fit, valid, fit_ids, valid_ids, raw_entity, raw_assay
    )

    by_mode: dict[str, tuple[dict[str, np.ndarray], dict[str, object]]] = {}
    for mode in ("strict", "name", "none"):
        entity, assay, excluded = filter_profiles(
            raw_entity, raw_assay, dataset, metadata, mode
        )
        components, dimensions = profile_components(
            fit_ids, valid_ids, fit.target.to_numpy(float), entity, assay
        )
        direct_retained = int(
            (entity.target.astype(str) == dataset.rsplit("_", 1)[0]).sum()
        )
        by_mode[mode] = (
            components,
            {
                "exclusion_mode": mode,
                "excluded_target_entities": len(excluded),
                "direct_endpoint_records_retained": direct_retained,
                **dimensions,
            },
        )

    strict_components, strict_audit = by_mode["strict"]
    rng = deterministic_rng(dataset, fold, seed)
    fit_permutation = rng.permutation(len(fit))
    valid_permutation = rng.permutation(len(valid))
    shuffled_components = permute_components(
        strict_components, fit_permutation, valid_permutation
    )
    if any(
        strict_components[key].shape != shuffled_components[key].shape
        for key in strict_components
    ):
        raise AssertionError("profile shuffle changed feature/kernel dimensionality")

    rows = [
        evaluate(
            dataset, "chemistry_only", chemistry_fit, chemistry_valid,
            fit, valid, fold, seed,
            {"exclusion_mode": "not_applicable", "shuffle_control": False},
        )
    ]
    strict_fit, strict_valid = assemble_kernel(
        chemistry_fit, chemistry_valid, strict_components
    )
    rows.append(evaluate(
        dataset, "taper_strict_exclusion", strict_fit, strict_valid,
        fit, valid, fold, seed, {**strict_audit, "shuffle_control": False},
    ))
    shuffled_fit, shuffled_valid = assemble_kernel(
        chemistry_fit, chemistry_valid, shuffled_components
    )
    rows.append(evaluate(
        dataset, "taper_shuffled_profile", shuffled_fit, shuffled_valid,
        fit, valid, fold, seed,
        {
            **strict_audit,
            "shuffle_control": True,
            "shuffle_preserves_kernel_value_multiset": True,
            "shuffle_preserves_partition_coverage_multiset": True,
        },
    ))
    for mode, model_name in (
        ("name", "taper_target_name_only"),
        ("none", "taper_no_exclusion"),
    ):
        components, audit = by_mode[mode]
        train_kernel, valid_kernel = assemble_kernel(
            chemistry_fit, chemistry_valid, components
        )
        rows.append(evaluate(
            dataset, model_name, train_kernel, valid_kernel,
            fit, valid, fold, seed, {**audit, "shuffle_control": False},
        ))
    return rows


def bootstrap_interval(values: np.ndarray, seed: int, draws: int = 20000) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def summarize(summary: pd.DataFrame) -> dict[str, object]:
    metrics = ("rmse", "cliff_rmse", "noncliff_rmse")
    target = summary.groupby(["dataset", "model"])[list(metrics)].mean().reset_index()
    macro = target.groupby("model")[list(metrics)].agg(["mean", "std"])
    payload: dict[str, object] = {
        "protocol": "official-train-only stratified five-fold CV; seed 42",
        "models": {},
        "shuffle_control": {},
        "leakage_stress_test": {},
    }
    for model in MODEL_ORDER:
        row = macro.loc[model]
        payload["models"][model] = {
            metric: {"mean": float(row[(metric, "mean")]), "target_sd": float(row[(metric, "std")])}
            for metric in metrics
        }

    wide = target.pivot(index="dataset", columns="model", values=list(metrics))
    for offset, metric in enumerate(metrics):
        penalty = (
            wide[(metric, "taper_shuffled_profile")]
            - wide[(metric, "taper_strict_exclusion")]
        ).to_numpy(float)
        test = wilcoxon(penalty, alternative="greater", zero_method="zsplit")
        payload["shuffle_control"][metric] = {
            "mean_penalty_shuffled_minus_true": float(penalty.mean()),
            "bootstrap_95_ci": bootstrap_interval(penalty, 20260905 + offset),
            "true_profile_target_wins": int((penalty > 0).sum()),
            "targets": int(len(penalty)),
            "wilcoxon_p_one_sided": float(test.pvalue),
        }
    for model in ("taper_strict_exclusion", "taper_target_name_only", "taper_no_exclusion"):
        payload["leakage_stress_test"][model] = payload["models"][model]
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(MOLECULEACE_DATASETS))
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--profile-root", type=Path, default=Path("data/chembl_offtarget_profiles")
    )
    parser.add_argument(
        "--target-metadata", type=Path,
        default=Path("data/chembl_target_metadata.jsonl.gz"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/taper_profile_controls_cv5")
    )
    args = parser.parse_args()
    if any(fold not in range(5) for fold in args.folds):
        raise ValueError("folds must be in [0, 4]")
    metadata = load_target_metadata(args.target_metadata)
    rows: list[dict[str, object]] = []
    for dataset in args.datasets:
        for fold in args.folds:
            fold_rows = run_fold(
                dataset, fold, args.seed, args.profile_root, metadata
            )
            rows.extend(fold_rows)
            print(json.dumps({
                "dataset": dataset,
                "fold": fold,
                "rmse": {row["model"]: row["rmse"] for row in fold_rows},
                "cliff_rmse": {row["model"]: row["cliff_rmse"] for row in fold_rows},
            }, sort_keys=True), flush=True)
    summary = pd.DataFrame(rows).sort_values(["model", "dataset", "fold"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "summary.cv5.csv", index=False)
    payload = summarize(summary)
    (args.output_dir / "aggregate.cv5.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
