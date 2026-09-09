from __future__ import annotations

"""Internal validation of a single pose-aware local interaction kernel."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.svm import SVR

from molcliff.data import load_moleculeace, split_official_train, tanimoto_matrix
from molcliff.metrics import regression_metrics
from run_moleculeace_local_chemistry_kernel import _generators
from validate_moleculeace_collision_free_kernel import sparse_counts, sparse_tanimoto


ELEMENT_GROUPS = ({6}, {7}, {8}, {15, 16}, {9, 17, 35, 53})


def _protein_residues(path: Path, pocket: dict) -> list[np.ndarray]:
    wanted = {(str(row["resid"]), str(row["resname"])) for row in pocket["residues"]}
    atoms: dict[tuple[str, str], list[list[float]]] = {key: [] for key in wanted}
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith("ATOM"):
            continue
        key = (line[22:27].strip(), line[17:20].strip())
        if key in atoms and (line[76:78].strip() or line[12:14].strip()).upper() != "H":
            atoms[key].append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    ordered = []
    for row in pocket["residues"]:
        key = (str(row["resid"]), str(row["resname"]))
        if not atoms[key]:
            raise RuntimeError(f"Missing pocket residue {key}")
        ordered.append(np.asarray(atoms[key], dtype=np.float32))
    return ordered


def _read_docked(root: Path, receptor_atoms: list[np.ndarray],
                 index_by_smiles: dict[str, int] | None = None,
                 files: list[Path] | None = None) -> tuple[dict[int, float], dict[int, np.ndarray]]:
    scores, contacts = {}, {}
    for path in files or sorted(root.glob("docked.[0-9][0-9].sdf.gz")):
        with gzip.open(path, "rb") as handle:
            supplier = Chem.ForwardSDMolSupplier(handle, removeHs=False)
            for mol in supplier:
                if mol is None:
                    continue
                if index_by_smiles is None:
                    index = int(mol.GetProp("_Name"))
                else:
                    smiles = mol.GetProp("SMILES")
                    if smiles not in index_by_smiles:
                        continue
                    index = index_by_smiles[smiles]
                scores[index] = float(mol.GetProp("minimizedAffinity"))
                conformer = mol.GetConformer()
                ligand_by_group = []
                for group in ELEMENT_GROUPS:
                    xyz = [list(conformer.GetAtomPosition(atom.GetIdx())) for atom in mol.GetAtoms()
                           if atom.GetAtomicNum() in group]
                    ligand_by_group.append(np.asarray(xyz, dtype=np.float32).reshape(-1, 3))
                fingerprint = np.zeros((len(receptor_atoms), len(ELEMENT_GROUPS)), dtype=np.float32)
                for residue_index, residue_xyz in enumerate(receptor_atoms):
                    for group_index, ligand_xyz in enumerate(ligand_by_group):
                        if len(ligand_xyz):
                            distance2 = ((residue_xyz[:, None] - ligand_xyz[None]) ** 2).sum(2)
                            fingerprint[residue_index, group_index] = float(distance2.min() <= 4.5 ** 2)
                contacts[index] = fingerprint.reshape(-1)
    return scores, contacts


def _rbf_by_median(train: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distance = np.abs(train[:, None] - train[None, :])
    positive = distance[distance > 0]
    scale = float(np.median(positive)) if len(positive) else 1.0
    train_kernel = np.exp(-np.square(distance / max(scale, 1e-8))).astype(np.float32)
    query_distance = np.abs(query[:, None] - train[None, :])
    query_kernel = np.exp(-np.square(query_distance / max(scale, 1e-8))).astype(np.float32)
    return train_kernel, query_kernel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="CHEMBL4203_Ki")
    parser.add_argument("--root", type=Path, default=Path("data/docking_pilot/CHEMBL4203_Ki"))
    parser.add_argument("--pockets", type=Path, default=Path("data/moleculeace_holo_pockets.json"))
    parser.add_argument("--training-docked-root", type=Path)
    parser.add_argument("--evaluation-docked-file", type=Path)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/moleculeace_docking_kernel_validation_v1"))
    parser.add_argument("--mode", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    full = load_moleculeace(args.dataset)
    if args.mode == "validation":
        fit, valid = split_official_train(full)
        evaluation_role = "validation"
    else:
        fit = full[full.split.eq("train")].reset_index(drop=True)
        valid = full[full.split.eq("test")].reset_index(drop=True)
        evaluation_role = "test"
    manifest = pd.read_csv(args.root / "manifest.csv")
    if manifest.role.tolist() != ["fit"] * len(fit) + [evaluation_role] * len(valid):
        raise RuntimeError("Docking manifest does not match the frozen validation split")
    pocket = json.loads(args.pockets.read_text())["datasets"][args.dataset]
    receptor_atoms = _protein_residues(args.root / "receptor.pdb", pocket)
    score_lookup, contact_lookup = _read_docked(args.root, receptor_atoms)
    if args.evaluation_docked_file is not None:
        constrained_scores, constrained_contacts = _read_docked(
            args.root, receptor_atoms, files=[args.evaluation_docked_file]
        )
        score_lookup.update(constrained_scores)
        contact_lookup.update(constrained_contacts)
    if args.training_docked_root is not None:
        train_lookup = {str(smiles): i for i, smiles in enumerate(fit.smiles)}
        reused_scores, reused_contacts = _read_docked(
            args.training_docked_root, receptor_atoms, train_lookup)
        score_lookup.update(reused_scores); contact_lookup.update(reused_contacts)
    if len(score_lookup) != len(manifest) or len(contact_lookup) != len(manifest):
        raise RuntimeError(f"Incomplete docking: {len(score_lookup)}/{len(manifest)}")
    scores = np.asarray([score_lookup[i] for i in range(len(manifest))], dtype=np.float32)
    contacts = np.asarray([contact_lookup[i] for i in range(len(manifest))], dtype=np.float32)

    n_fit = len(fit)
    chemistry_fit = np.zeros((n_fit, n_fit), dtype=np.float32)
    chemistry_valid = np.zeros((len(valid), n_fit), dtype=np.float32)
    for _name, generator in _generators():
        fit_count, vocabulary = sparse_counts(fit.smiles.to_numpy(), generator)
        valid_count, _ = sparse_counts(valid.smiles.to_numpy(), generator, vocabulary)
        fit_binary, valid_binary = fit_count.copy(), valid_count.copy()
        fit_binary.data[:] = 1.0; valid_binary.data[:] = 1.0
        chemistry_fit += (sparse_tanimoto(fit_binary, fit_binary)
                          + sparse_tanimoto(fit_count, fit_count)) / 10.0
        chemistry_valid += (sparse_tanimoto(valid_binary, fit_binary)
                            + sparse_tanimoto(valid_count, fit_count)) / 10.0
    contact_fit = tanimoto_matrix(contacts[:n_fit], contacts[:n_fit])
    contact_valid = tanimoto_matrix(contacts[n_fit:], contacts[:n_fit])
    vina_fit, vina_valid = _rbf_by_median(scores[:n_fit], scores[n_fit:])
    physical_fit = (10.0 * chemistry_fit + contact_fit + vina_fit) / 12.0
    physical_valid = (10.0 * chemistry_valid + contact_valid + vina_valid) / 12.0

    rows = []
    for name, train_kernel, valid_kernel in (
        ("collision_free_2d", chemistry_fit, chemistry_valid),
        ("pose_action_12view", physical_fit, physical_valid),
    ):
        model = SVR(C=10.0, epsilon=0.1, kernel="precomputed")
        model.fit(train_kernel, fit.target.to_numpy(dtype=float))
        prediction = model.predict(valid_kernel)
        metric = regression_metrics(valid.target.to_numpy(dtype=float), prediction,
                                    valid.cliff_mol.to_numpy(dtype=bool))
        rows.append({"dataset": args.dataset, "model": name, **metric})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / f"{args.dataset}.{args.mode}.csv", index=False)
    payload = {"dataset": args.dataset, "models": {row["model"]: row for row in rows},
               "docked": len(scores), "mode": args.mode,
               "test_evaluations": int(args.mode == "test"), "cnn_scoring": False,
               "vina_activity_pearson_fit": float(np.corrcoef(scores[:n_fit], fit.target)[0, 1])}
    (args.output_dir / f"{args.dataset}.{args.mode}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
