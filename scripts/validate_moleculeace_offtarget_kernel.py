from __future__ import annotations

"""Internal validation of a leakage-filtered ChEMBL off-target profile kernel."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.svm import SVR

from molcliff.data import (
    EXPANDED_CLIFF_ROOT,
    MOLECULEACE_ROOT,
    load_moleculeace,
    split_official_train,
)
from molcliff.metrics import regression_metrics
from validate_moleculeace_operator_action_kernel import exact_kernel


DEFAULT_DATASETS = ("CHEMBL4203_Ki", "CHEMBL2971_Ki", "CHEMBL237_EC50", "CHEMBL228_Ki")


def molecule_mapping(dataset: str) -> dict[str, str]:
    path = MOLECULEACE_ROOT / "raw" / f"{dataset}.csv"
    if not path.exists():
        path = EXPANDED_CLIFF_ROOT / "raw" / f"{dataset}.csv"
    raw = pd.read_csv(path)
    return raw.drop_duplicates("smiles").set_index("smiles").chembl_id.astype(str).to_dict()


def load_profiles(dataset: str, root: Path, split: str = "train") -> pd.DataFrame:
    records = []
    splits = ("train", "test") if split == "both" else (split,)
    for part in splits:
        suffix = f"{part}_offtarget"
        with gzip.open(root / f"{dataset}.{suffix}.jsonl.gz", "rt") as handle:
            for line in handle:
                record = json.loads(line)
                records.append((record["molecule_chembl_id"], record["target_chembl_id"],
                                float(record["pchembl_value"])))
    frame = pd.DataFrame(records, columns=["molecule", "target", "value"])
    return frame.groupby(["molecule", "target"], as_index=False).value.median()


def matrices(fit_ids: list[str], valid_ids: list[str], profile: pd.DataFrame):
    fit_set = set(fit_ids)
    fit_profile = profile[profile.molecule.isin(fit_set)]
    targets = sorted(fit_profile.target.unique())
    vocabulary = {target: col for col, target in enumerate(targets)}
    medians = fit_profile.groupby("target").value.median().to_dict()

    def build(ids: list[str]) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        if not vocabulary:
            empty = sparse.csr_matrix((len(ids), 0), dtype=np.float32)
            return empty, empty
        row_of = {molecule: row for row, molecule in enumerate(ids)}
        rows, cols, observed, potency = [], [], [], []
        for record in profile.itertuples(index=False):
            row = row_of.get(record.molecule)
            col = vocabulary.get(record.target)
            if row is None or col is None:
                continue
            rows.append(row); cols.append(col); observed.append(1.0)
            potency.append(float(record.value - medians[record.target]))
        shape = (len(ids), len(vocabulary))
        coverage = sparse.csr_matrix((observed, (rows, cols)), shape=shape, dtype=np.float32)
        residual = sparse.csr_matrix((potency, (rows, cols)), shape=shape, dtype=np.float32)
        return normalize(coverage, norm="l2"), normalize(residual, norm="l2")

    fit_coverage, fit_residual = build(fit_ids)
    valid_coverage, valid_residual = build(valid_ids)
    coverage_fit = (fit_coverage @ fit_coverage.T).toarray().astype(np.float32)
    coverage_valid = (valid_coverage @ fit_coverage.T).toarray().astype(np.float32)
    residual_fit = (fit_residual @ fit_residual.T).toarray().astype(np.float32)
    residual_valid = (valid_residual @ fit_residual.T).toarray().astype(np.float32)
    return (coverage_fit, coverage_valid, residual_fit, residual_valid,
            len(vocabulary))


def run_one(dataset: str, profile_root: Path) -> list[dict]:
    fit, valid = split_official_train(load_moleculeace(dataset))
    chemistry_fit, chemistry_valid = exact_kernel(fit, valid)
    mapping = molecule_mapping(dataset)
    fit_ids = [mapping[s] for s in fit.smiles]
    valid_ids = [mapping[s] for s in valid.smiles]
    coverage_fit, coverage_valid, residual_fit, residual_valid, dimensions = matrices(
        fit_ids, valid_ids, load_profiles(dataset, profile_root))
    profile_fit = (coverage_fit + residual_fit) / 2
    profile_valid = (coverage_valid + residual_valid) / 2
    candidates = (
        ("collision_free_binary_count_kernel", chemistry_fit, chemistry_valid),
        ("offtarget_profile_kernel", profile_fit, profile_valid),
        ("collision_free_plus_coverage_10to1",
         (10 * chemistry_fit + coverage_fit) / 11,
         (10 * chemistry_valid + coverage_valid) / 11),
        ("collision_free_plus_potency_residual_10to1",
         (10 * chemistry_fit + residual_fit) / 11,
         (10 * chemistry_valid + residual_valid) / 11),
        ("collision_free_plus_offtarget_10to1",
         (10 * chemistry_fit + profile_fit) / 11,
         (10 * chemistry_valid + profile_valid) / 11),
    )
    rows = []
    for name, train_kernel, valid_kernel in candidates:
        model = SVR(C=10.0, epsilon=0.1, kernel="precomputed")
        model.fit(train_kernel, fit.target.to_numpy(dtype=float))
        pred = model.predict(valid_kernel)
        rows.append({"dataset": dataset, "model": name, "profile_targets": dimensions,
                     **regression_metrics(valid.target.to_numpy(dtype=float), pred,
                                          valid.cliff_mol.to_numpy(dtype=bool))})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--profile-root", type=Path, default=Path("data/chembl_offtarget_profiles"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/moleculeace_offtarget_validation_pilot_v1"))
    args = parser.parse_args()
    rows = [row for dataset in args.datasets for row in run_one(dataset, args.profile_root)]
    summary = pd.DataFrame(rows).sort_values(["model", "dataset"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "summary.validation.csv", index=False)
    macro = summary.groupby("model")[["rmse", "cliff_rmse", "noncliff_rmse"]].mean()
    result = {name: {k: float(v) for k, v in row.items()} for name, row in macro.iterrows()}
    result["protocol"] = (
        "current benchmark target removed; fit-only target vocabulary/medians; "
        "equal coverage/residual profile kernel; 10 chemical + 1 profile view"
    )
    (args.output_dir / "aggregate.validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
