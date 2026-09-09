from __future__ import annotations

"""Train-only validation of a deterministic Boltzmann conformer-ensemble view."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from sklearn.svm import SVR

from molcliff.data import MOLECULEACE_DATASETS, load_moleculeace, split_official_train
from molcliff.metrics import regression_metrics
from validate_moleculeace_operator_action_kernel import exact_kernel
from validate_moleculeace_profile_alignment_kernel import _rbf_from_train_scale


RT_KCAL_298K = 0.00198720425864083 * 298.15


def conformer_summary(smiles: str) -> tuple[str, np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.pruneRmsThresh = 0.5
    params.useSmallRingTorsions = True
    params.timeout = 20
    AllChem.EmbedMultipleConfs(mol, numConfs=8, params=params)
    conformers = [conformer.GetId() for conformer in mol.GetConformers()]
    if not conformers:
        params.useRandomCoords = True
        AllChem.EmbedMultipleConfs(mol, numConfs=1, params=params)
        conformers = [conformer.GetId() for conformer in mol.GetConformers()]
    if not conformers:
        return smiles, np.zeros(18, dtype=np.float32)
    if AllChem.MMFFHasAllMoleculeParams(mol):
        optimized = AllChem.MMFFOptimizeMoleculeConfs(
            mol, numThreads=1, maxIters=200, mmffVariant="MMFF94s"
        )
    else:
        optimized = AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=1, maxIters=200)
    energies = np.asarray([float(value[1]) for value in optimized], dtype=np.float64)
    energies -= float(energies.min())
    weights = np.exp(-np.minimum(energies / RT_KCAL_298K, 80.0))
    weights /= weights.sum()
    descriptors = []
    for conformer in conformers:
        cid = int(conformer)
        descriptors.append([
            rdMolDescriptors.CalcRadiusOfGyration(mol, confId=cid),
            rdMolDescriptors.CalcAsphericity(mol, confId=cid),
            rdMolDescriptors.CalcEccentricity(mol, confId=cid),
            rdMolDescriptors.CalcInertialShapeFactor(mol, confId=cid),
            rdMolDescriptors.CalcNPR1(mol, confId=cid),
            rdMolDescriptors.CalcNPR2(mol, confId=cid),
            rdMolDescriptors.CalcPBF(mol, confId=cid),
        ])
    shape = np.asarray(descriptors, dtype=np.float64)
    mean = weights @ shape
    variance = weights @ np.square(shape - mean)
    entropy = -float(np.sum(weights * np.log(np.maximum(weights, 1e-12))))
    summary = np.concatenate([
        mean,
        np.sqrt(np.maximum(variance, 0.0)),
        np.asarray([
            entropy,
            float(1.0 / np.sum(weights * weights)),
            float(np.sum(weights * energies)),
            float(np.std(energies)),
        ]),
    ])
    return smiles, summary.astype(np.float32)


def load_or_build_cache(
    smiles: list[str], cache_path: Path, jobs: int,
) -> dict[str, np.ndarray]:
    cache: dict[str, np.ndarray] = {}
    if cache_path.exists():
        stored = np.load(cache_path, allow_pickle=True)
        cache = {
            str(key): value.astype(np.float32)
            for key, value in zip(stored["smiles"], stored["features"], strict=True)
        }
    missing = sorted(set(smiles).difference(cache))
    for start in range(0, len(missing), 128):
        batch = missing[start:start + 128]
        records = Parallel(n_jobs=jobs, verbose=10)(
            delayed(conformer_summary)(value) for value in batch
        )
        cache.update(records)
        keys = sorted(cache)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            smiles=np.asarray(keys, dtype=object),
            features=np.vstack([cache[key] for key in keys]),
        )
        print(json.dumps({
            "conformer_cache": str(cache_path),
            "cached": len(cache),
            "remaining": len(missing) - min(start + len(batch), len(missing)),
        }), flush=True)
    return cache


def run_one(dataset: str, cache: dict[str, np.ndarray]) -> list[dict]:
    fit, valid = split_official_train(load_moleculeace(dataset))
    chemistry_fit, chemistry_valid = exact_kernel(fit, valid)
    fit_shape = np.vstack([cache[str(value)] for value in fit.smiles])
    valid_shape = np.vstack([cache[str(value)] for value in valid.smiles])
    shape_fit, shape_valid = _rbf_from_train_scale(fit_shape, valid_shape)
    kernels = {
        "collision_free_binary_count_kernel": (chemistry_fit, chemistry_valid),
        "boltzmann_conformer_kernel": (shape_fit, shape_valid),
        "collision_free_plus_boltzmann_10to1": (
            (10 * chemistry_fit + shape_fit) / 11,
            (10 * chemistry_valid + shape_valid) / 11,
        ),
    }
    rows = []
    for name, (fit_kernel, valid_kernel) in kernels.items():
        model = SVR(C=10.0, epsilon=0.1, kernel="precomputed")
        model.fit(fit_kernel, fit.target.to_numpy(float))
        prediction = model.predict(valid_kernel)
        rows.append({
            "dataset": dataset, "model": name,
            **regression_metrics(
                valid.target.to_numpy(float), prediction,
                valid.cliff_mol.to_numpy(bool),
            ),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(MOLECULEACE_DATASETS))
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--cache", type=Path, default=Path("data/moleculeace_boltzmann_rdkit_v1.npz")
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/moleculeace_boltzmann_validation_v1"),
    )
    args = parser.parse_args()
    frames = [load_moleculeace(dataset) for dataset in args.datasets]
    train_smiles = sorted(set(
        value for frame in frames
        for value in frame.loc[frame.split.eq("train"), "smiles"].astype(str)
    ))
    cache = load_or_build_cache(train_smiles, args.cache, args.jobs)
    nested = Parallel(n_jobs=args.jobs)(
        delayed(run_one)(dataset, cache) for dataset in args.datasets
    )
    summary = pd.DataFrame([row for rows in nested for row in rows])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.sort_values(["model", "dataset"]).to_csv(
        args.output_dir / "summary.validation.csv", index=False
    )
    macro = summary.groupby("model")[["rmse", "cliff_rmse", "noncliff_rmse"]].mean()
    payload = {
        "protocol": (
            "official-train-only fixed seed42 validation; 8 ETKDGv3 conformers; "
            "MMFF94s/UFF; 298.15 K Boltzmann weights; zero test evaluations"
        ),
        "datasets": int(summary.dataset.nunique()),
        "cached_molecules": len(cache),
        "models": {
            name: {key: float(value) for key, value in row.items()}
            for name, row in macro.iterrows()
        },
        "official_test_evaluations": 0,
    }
    (args.output_dir / "aggregate.validation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
