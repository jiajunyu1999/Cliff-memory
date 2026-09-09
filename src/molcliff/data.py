from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol, MakeScaffoldGeneric
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOLECULEACE_ROOT = (
    PROJECT_ROOT
    / "third_party"
    / "MoleculeACE"
    / "MoleculeACE"
    / "Data"
    / "benchmark_data"
)
EXPANDED_CLIFF_ROOT = PROJECT_ROOT / "data" / "expanded_cliff_regression"

MOLECULEACE_DATASETS = (
    "CHEMBL4203_Ki", "CHEMBL2034_Ki", "CHEMBL233_Ki", "CHEMBL4616_EC50",
    "CHEMBL287_Ki", "CHEMBL218_EC50", "CHEMBL264_Ki", "CHEMBL219_Ki",
    "CHEMBL2835_Ki", "CHEMBL2147_Ki", "CHEMBL231_Ki", "CHEMBL3979_EC50",
    "CHEMBL237_EC50", "CHEMBL244_Ki", "CHEMBL4792_Ki", "CHEMBL1871_Ki",
    "CHEMBL237_Ki", "CHEMBL262_Ki", "CHEMBL2047_EC50", "CHEMBL239_EC50",
    "CHEMBL2971_Ki", "CHEMBL204_Ki", "CHEMBL214_Ki", "CHEMBL1862_Ki",
    "CHEMBL234_Ki", "CHEMBL238_Ki", "CHEMBL235_EC50", "CHEMBL4005_Ki",
    "CHEMBL236_Ki", "CHEMBL228_Ki",
)


def load_moleculeace(dataset: str, root: Path | None = None) -> pd.DataFrame:
    """Load one unmodified official MoleculeACE table.

    The molar pEC50/pKi column is used as the target.  It differs from the
    original ``y`` by a constant only, so RMSE is unchanged and positive values
    are easier to interpret as potency.
    """

    if root is None:
        if dataset in MOLECULEACE_DATASETS:
            path = MOLECULEACE_ROOT / f"{dataset}.csv"
        else:
            path = EXPANDED_CLIFF_ROOT / f"{dataset}.csv"
            if not path.exists():
                raise ValueError(f"Unknown activity-cliff dataset: {dataset}")
    else:
        path = root / f"{dataset}.csv"
    frame = pd.read_csv(path)
    target_column = "y [pEC50/pKi]" if "y [pEC50/pKi]" in frame.columns else "y"
    required = {"smiles", "split", "cliff_mol", target_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame.smiles.duplicated().any():
        raise ValueError(f"{path} contains duplicate SMILES")
    frame = frame.copy()
    frame["target"] = frame[target_column].astype(float)
    frame["dataset"] = dataset
    return frame


def split_official_train(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the one fixed, cliff-stratified validation split used locally."""

    pool = frame[frame.split.eq("train")].reset_index(drop=True)
    indices = np.arange(len(pool))
    fit_idx, valid_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=pool.cliff_mol.to_numpy(),
    )
    return (
        pool.iloc[np.sort(fit_idx)].reset_index(drop=True),
        pool.iloc[np.sort(valid_idx)].reset_index(drop=True),
    )


class Fingerprints:
    """One deterministic RDKit ECFP representation used by all local models."""

    def __init__(self, radius: int = 2, n_bits: int = 1024):
        self.radius = int(radius)
        self.n_bits = int(n_bits)
        self._bit = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius, fpSize=self.n_bits
        )
        self._count = rdFingerprintGenerator.GetMorganGenerator(radius=self.radius)

    def mols(self, smiles: list[str] | np.ndarray) -> list[Chem.Mol]:
        mols = [Chem.MolFromSmiles(str(s)) for s in smiles]
        bad = [i for i, mol in enumerate(mols) if mol is None]
        if bad:
            raise ValueError(f"RDKit failed on molecule indices {bad[:10]}")
        return mols  # type: ignore[return-value]

    def bits(self, smiles: list[str] | np.ndarray) -> np.ndarray:
        mols = self.mols(smiles)
        out = np.zeros((len(mols), self.n_bits), dtype=np.float32)
        for i, mol in enumerate(mols):
            DataStructs.ConvertToNumpyArray(self._bit.GetFingerprint(mol), out[i])
        return out

    def counts(self, smiles: list[str] | np.ndarray) -> np.ndarray:
        """Hashed count ECFP, clipped to retain a compact action magnitude."""

        mols = self.mols(smiles)
        out = np.zeros((len(mols), self.n_bits), dtype=np.float32)
        hashed = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius, fpSize=self.n_bits, countSimulation=False
        )
        for i, mol in enumerate(mols):
            fp = hashed.GetCountFingerprint(mol)
            for index, value in fp.GetNonzeroElements().items():
                out[i, int(index)] = min(float(value), 4.0)
        return out

    def scaffold_bits(self, smiles: list[str] | np.ndarray) -> np.ndarray:
        mols = self.mols(smiles)
        out = np.zeros((len(mols), self.n_bits), dtype=np.float32)
        for i, mol in enumerate(mols):
            scaffold = GetScaffoldForMol(mol)
            try:
                scaffold = MakeScaffoldGeneric(scaffold)
            except Exception:
                pass
            DataStructs.ConvertToNumpyArray(self._bit.GetFingerprint(scaffold), out[i])
        return out


def tanimoto_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Dense Tanimoto matrix for binary fingerprint arrays."""

    intersection = left @ right.T
    union = left.sum(1, keepdims=True) + right.sum(1)[None, :] - intersection
    return intersection / np.maximum(union, 1.0)


def levenshtein_similarity_matrix(
    left: list[str] | np.ndarray, right: list[str] | np.ndarray
) -> np.ndarray:
    from Levenshtein import distance

    result = np.empty((len(left), len(right)), dtype=np.float32)
    for i, a in enumerate(left):
        a = str(a)
        for j, b in enumerate(right):
            b = str(b)
            result[i, j] = 1.0 - distance(a, b) / max(len(a), len(b), 1)
    return result


def moleculeace_similarity_matrix(
    left_smiles: list[str] | np.ndarray,
    right_smiles: list[str] | np.ndarray,
    fingerprints: Fingerprints | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Cross-set version of MoleculeACE's three-view consensus similarity."""

    fp = fingerprints or Fingerprints()
    left_smiles = np.asarray(left_smiles, dtype=object)
    right_smiles = np.asarray(right_smiles, dtype=object)
    ecfp = tanimoto_matrix(fp.bits(left_smiles), fp.bits(right_smiles))
    scaffold = tanimoto_matrix(
        fp.scaffold_bits(left_smiles), fp.scaffold_bits(right_smiles)
    )
    smiles = levenshtein_similarity_matrix(left_smiles, right_smiles)
    views = {"ecfp": ecfp, "scaffold": scaffold, "smiles": smiles}
    return np.maximum.reduce([ecfp, scaffold, smiles]), views
