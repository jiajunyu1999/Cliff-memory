from __future__ import annotations

"""Train-only test of a discrete local-curvature activity model.

The query is the origin of a local physical perturbation coordinate system.
A weighted polynomial response surface is fitted from chemically nearest fit
molecules.  Comparing degree one and degree two isolates the contribution of
directional curvature without a Hessian network, seed search, or test labels.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors, rdPartialCharges
from sklearn.svm import SVR

from molcliff.data import MOLECULEACE_DATASETS, load_moleculeace, split_official_train
from molcliff.metrics import regression_metrics
from validate_moleculeace_operator_action_kernel import exact_kernel


def physical_coordinates(smiles: np.ndarray) -> np.ndarray:
    rows = []
    for text in smiles:
        mol = Chem.MolFromSmiles(str(text))
        if mol is None:
            raise ValueError(f"RDKit failed to parse {text}")
        rdPartialCharges.ComputeGasteigerCharges(mol)
        charges = np.asarray([
            float(atom.GetProp("_GasteigerCharge")) for atom in mol.GetAtoms()
        ])
        charges = charges[np.isfinite(charges)]
        if not len(charges):
            charges = np.zeros(1)
        heavy = max(float(mol.GetNumHeavyAtoms()), 1.0)
        rows.append([
            Descriptors.MolWt(mol), Crippen.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol), Crippen.MolMR(mol),
            Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol),
            Lipinski.NumRotatableBonds(mol), Lipinski.RingCount(mol),
            Lipinski.NumAromaticRings(mol), rdMolDescriptors.CalcFractionCSP3(mol),
            Chem.GetFormalCharge(mol), float(charges.max()), float(charges.min()),
            float(np.abs(charges).sum() / heavy), heavy,
            Lipinski.NumHeteroatoms(mol) / heavy, Descriptors.BertzCT(mol) / heavy,
        ])
    return np.asarray(rows, dtype=np.float64)


def robust_scale(fit: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    location = np.median(fit, axis=0)
    scale = 1.4826 * np.median(np.abs(fit - location), axis=0)
    fallback = np.std(fit, axis=0)
    scale = np.where(scale > 1e-8, scale, np.where(fallback > 1e-8, fallback, 1.0))
    return (fit - location) / scale, (query - location) / scale


def local_response_predict(
    fit_x: np.ndarray,
    query_x: np.ndarray,
    fit_y: np.ndarray,
    similarity: np.ndarray,
    degree: int,
) -> np.ndarray:
    n_neighbors = max(64, min(192, int(4 * np.sqrt(len(fit_x)))))
    y_location = float(np.mean(fit_y))
    y_scale = max(float(np.std(fit_y)), 1e-8)
    standardized_y = (fit_y - y_location) / y_scale
    predictions = np.empty(len(query_x), dtype=np.float64)
    for row in range(len(query_x)):
        neighbors = np.argsort(similarity[row])[::-1][:n_neighbors]
        sim = np.maximum(similarity[row, neighbors].astype(np.float64), 0.0)
        delta = fit_x[neighbors] - query_x[row]
        chemical_distance = (1.0 - sim)[:, None]
        first_order = np.concatenate([delta, chemical_distance], axis=1)
        design_parts = [np.ones((len(neighbors), 1)), first_order]
        if degree == 2:
            # Diagonal curvature is identifiable with the fixed local sample
            # budget; unrestricted cross terms would turn this into a large
            # hyperparameter-sensitive Hessian fit.
            design_parts.append(first_order * first_order)
        design = np.concatenate(design_parts, axis=1)
        weights = np.maximum(sim ** 4, 1e-4)
        weighted = design * np.sqrt(weights[:, None])
        target = standardized_y[neighbors] * np.sqrt(weights)
        penalty = np.eye(design.shape[1], dtype=np.float64)
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(weighted.T @ weighted + penalty, weighted.T @ target)
        predictions[row] = coefficients[0] * y_scale + y_location
    return predictions


def run_one(dataset: str) -> list[dict]:
    fit, valid = split_official_train(load_moleculeace(dataset))
    chemistry_fit, chemistry_valid = exact_kernel(fit, valid)
    fit_x, valid_x = robust_scale(
        physical_coordinates(fit.smiles.to_numpy()),
        physical_coordinates(valid.smiles.to_numpy()),
    )
    y = fit.target.to_numpy(float)
    baseline = SVR(C=10.0, epsilon=0.1, kernel="precomputed")
    baseline.fit(chemistry_fit, y)
    predictions = {
        "collision_free_binary_count_kernel": baseline.predict(chemistry_valid),
        "local_linear_response": local_response_predict(
            fit_x, valid_x, y, chemistry_valid, degree=1
        ),
        "local_quadratic_curvature": local_response_predict(
            fit_x, valid_x, y, chemistry_valid, degree=2
        ),
    }
    return [{
        "dataset": dataset,
        "model": name,
        **regression_metrics(
            valid.target.to_numpy(float), prediction,
            valid.cliff_mol.to_numpy(bool),
        ),
    } for name, prediction in predictions.items()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(MOLECULEACE_DATASETS))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/moleculeace_local_curvature_validation_v1"),
    )
    args = parser.parse_args()
    nested = Parallel(n_jobs=args.jobs)(delayed(run_one)(dataset) for dataset in args.datasets)
    summary = pd.DataFrame([row for rows in nested for row in rows])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.sort_values(["model", "dataset"]).to_csv(
        args.output_dir / "summary.validation.csv", index=False
    )
    macro = summary.groupby("model")[["rmse", "cliff_rmse", "noncliff_rmse"]].mean()
    payload = {
        "protocol": (
            "fixed official-train 80/20 seed42 split; degree-1 versus degree-2 "
            "ablation; zero official-test evaluations"
        ),
        "datasets": int(summary.dataset.nunique()),
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
