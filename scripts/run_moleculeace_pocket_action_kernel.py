from __future__ import annotations

"""A single pocket-gated local-action kernel for MoleculeACE.

Five generic local chemistry views preserve ligand similarity.  Six additional
Morgan views start only at atoms of one pharmacophore family and are weighted
by complementary residues in a label-free experimental holo pocket.  Their
weighted sum is one positive-semidefinite kernel fitted by one SVR.  The rule
has no validation/test selection, checkpoint selection, or model averaging.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem, DataStructs, RDConfig
from rdkit.Chem import ChemicalFeatures, rdFingerprintGenerator
from sklearn.svm import SVR

from molcliff.data import MOLECULEACE_DATASETS, load_moleculeace, tanimoto_matrix
from molcliff.metrics import regression_metrics


FP_SIZE = 1024
CHANNELS = ("donor", "acceptor", "hydrophobe", "aromatic", "positive", "negative")
FAMILY_TO_CHANNEL = {
    "Donor": "donor", "Acceptor": "acceptor", "Hydrophobe": "hydrophobe",
    "LumpedHydrophobe": "hydrophobe", "Aromatic": "aromatic",
    "PosIonizable": "positive", "NegIonizable": "negative",
}
# Residues whose side chains can complement each ligand pharmacophore.  This is
# a chemistry prior, fixed before reading any assay activity.
COMPLEMENT = {
    "donor": set("DENQHSTY"),
    "acceptor": set("RKHNQSTYW"),
    "hydrophobe": set("AVILMFWYPC"),
    "aromatic": set("FWYH"),
    "positive": set("DE"),
    "negative": set("KRH"),
}
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _base_generators() -> tuple[tuple[str, object], ...]:
    feature_inv = rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
    return (
        ("chiral_morgan_r1", rdFingerprintGenerator.GetMorganGenerator(
            radius=1, fpSize=FP_SIZE, includeChirality=True)),
        ("chiral_morgan_r2", rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=FP_SIZE, includeChirality=True)),
        ("chiral_morgan_r3", rdFingerprintGenerator.GetMorganGenerator(
            radius=3, fpSize=FP_SIZE, includeChirality=True)),
        ("feature_morgan_r2", rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=FP_SIZE, includeChirality=True,
            atomInvariantsGenerator=feature_inv)),
        ("chiral_atom_pair", rdFingerprintGenerator.GetAtomPairGenerator(
            fpSize=FP_SIZE, includeChirality=True)),
    )


FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(
    str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef")
)
CHANNEL_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=FP_SIZE, includeChirality=True,
)


def _mols(smiles: np.ndarray) -> list[Chem.Mol]:
    mols = [Chem.MolFromSmiles(str(value)) for value in smiles]
    if any(mol is None for mol in mols):
        raise ValueError("RDKit failed to parse a molecule")
    return mols  # type: ignore[return-value]


def _bits(mols: list[Chem.Mol], generator: object) -> np.ndarray:
    out = np.zeros((len(mols), FP_SIZE), dtype=np.float32)
    for row, mol in zip(out, mols, strict=True):
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), row)
    return out


def _channel_bits(mols: list[Chem.Mol]) -> dict[str, np.ndarray]:
    out = {name: np.zeros((len(mols), FP_SIZE), dtype=np.float32) for name in CHANNELS}
    for index, mol in enumerate(mols):
        atom_ids: dict[str, set[int]] = {name: set() for name in CHANNELS}
        for feature in FEATURE_FACTORY.GetFeaturesForMol(mol):
            channel = FAMILY_TO_CHANNEL.get(feature.GetFamily())
            if channel is not None:
                atom_ids[channel].update(int(i) for i in feature.GetAtomIds())
        for channel, ids in atom_ids.items():
            if ids:
                fp = CHANNEL_GENERATOR.GetFingerprint(mol, fromAtoms=sorted(ids))
                DataStructs.ConvertToNumpyArray(fp, out[channel][index])
    return out


def _pocket_weights(pocket: dict) -> dict[str, float]:
    scores = []
    for channel in CHANNELS:
        score = 0.0
        for residue in pocket["residues"]:
            aa = AA3_TO_1.get(str(residue["resname"]))
            if aa in COMPLEMENT[channel]:
                # Inverse distance is a parameter-free contact potential.
                score += 1.0 / max(float(residue["ligand_distance"]), 1.0)
        scores.append(score)
    scores_array = np.asarray(scores, dtype=np.float64)
    if not np.any(scores_array > 0):
        scores_array[:] = 1.0
    # Mean-one normalization fixes the total capacity of the pocket branch.
    scores_array /= scores_array.mean()
    return {channel: float(weight) for channel, weight in zip(CHANNELS, scores_array, strict=True)}


def run_one(dataset: str, output_dir: Path, pocket: dict) -> dict:
    frame = load_moleculeace(dataset)
    train = frame[frame.split.eq("train")].reset_index(drop=True)
    test = frame[frame.split.eq("test")].reset_index(drop=True)
    train_mols = _mols(train.smiles.to_numpy())
    test_mols = _mols(test.smiles.to_numpy())

    train_kernel = np.zeros((len(train), len(train)), dtype=np.float32)
    test_kernel = np.zeros((len(test), len(train)), dtype=np.float32)
    base_names = []
    for name, generator in _base_generators():
        base_names.append(name)
        x_train = _bits(train_mols, generator)
        x_test = _bits(test_mols, generator)
        train_kernel += tanimoto_matrix(x_train, x_train)
        test_kernel += tanimoto_matrix(x_test, x_train)

    weights = _pocket_weights(pocket)
    channel_train = _channel_bits(train_mols)
    channel_test = _channel_bits(test_mols)
    for channel in CHANNELS:
        train_kernel += weights[channel] * tanimoto_matrix(
            channel_train[channel], channel_train[channel]
        )
        test_kernel += weights[channel] * tanimoto_matrix(
            channel_test[channel], channel_train[channel]
        )
    # Five unit generic views plus six mean-one pocket views.
    train_kernel /= len(base_names) + len(CHANNELS)
    test_kernel /= len(base_names) + len(CHANNELS)

    model = SVR(C=10.0, epsilon=0.1, kernel="precomputed")
    model.fit(train_kernel, train.target.to_numpy(dtype=float))
    prediction = model.predict(test_kernel)
    metric = regression_metrics(
        test.target.to_numpy(dtype=float), prediction, test.cliff_mol.to_numpy(dtype=bool)
    )
    metric.update({"dataset": dataset, "model": "pocket_action_kernel", "seed": 42})
    predictions = test[["dataset", "smiles", "target", "cliff_mol", "split"]].copy()
    predictions["prediction"] = prediction
    predictions.to_csv(output_dir / f"{dataset}.pocket_action_kernel.predictions.csv", index=False)
    return {**metric, "pocket_weights": json.dumps(weights, sort_keys=True)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pockets", type=Path, default=Path("data/moleculeace_holo_pockets.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/moleculeace_pocket_action_kernel_v1"))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    payload = json.loads(args.pockets.read_text())
    missing = [name for name in MOLECULEACE_DATASETS if payload["datasets"].get(name) is None]
    if missing:
        raise RuntimeError(f"Missing holo pockets: {missing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # The RDKit feature factory is immutable but intentionally non-picklable;
    # threads safely share it and each target otherwise has independent state.
    rows = Parallel(n_jobs=args.jobs, prefer="threads")(
        delayed(run_one)(name, args.output_dir, payload["datasets"][name])
        for name in MOLECULEACE_DATASETS
    )
    summary = pd.DataFrame(rows).sort_values("dataset")
    summary.to_csv(args.output_dir / "summary.pocket_action_kernel.csv", index=False)
    aggregate = {
        "model": "pocket_action_kernel", "datasets": len(summary),
        "macro_rmse": float(summary.rmse.mean()),
        "macro_cliff_rmse": float(summary.cliff_rmse.mean()),
        "macro_noncliff_rmse": float(summary.noncliff_rmse.mean()),
        "base_views": [name for name, _ in _base_generators()],
        "pocket_views": list(CHANNELS), "seed": 42, "selection": "none",
        "ensemble": False, "uses_test_labels_as_input": False,
    }
    (args.output_dir / "aggregate.pocket_action_kernel.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(aggregate, sort_keys=True))


if __name__ == "__main__":
    main()
