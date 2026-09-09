from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy import sparse

from .data import Fingerprints, tanimoto_matrix


class _SparseActionFeaturizer:
    def __init__(self, n_bits: int = 1024):
        self.n_bits = int(n_bits)
        self.bit_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
        self.count_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=n_bits, countSimulation=False
        )
        self.mol_cache: dict[str, Chem.Mol] = {}
        self.bit_cache: dict[str, set[int]] = {}
        self.count_cache: dict[str, dict[int, float]] = {}

    def mol(self, smiles: str) -> Chem.Mol:
        if smiles in self.mol_cache:
            return self.mol_cache[smiles]
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            raise ValueError(f"RDKit failed on SMILES: {smiles}")
        self.mol_cache[smiles] = mol
        return mol

    def bit_set(self, smiles: str) -> set[int]:
        if smiles not in self.bit_cache:
            self.bit_cache[smiles] = set(
                int(x) for x in self.bit_generator.GetFingerprint(self.mol(smiles)).GetOnBits()
            )
        return self.bit_cache[smiles]

    def count_items(self, smiles: str) -> dict[int, float]:
        if smiles not in self.count_cache:
            fp = self.count_generator.GetCountFingerprint(self.mol(smiles))
            self.count_cache[smiles] = {
                int(index): min(float(value), 4.0)
                for index, value in fp.GetNonzeroElements().items()
            }
        return self.count_cache[smiles]

    def transform_one(self, source_smiles: str, destination_smiles: str, source_y: float) -> sparse.csr_matrix:
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        src_bits = self.bit_set(source_smiles)
        dst_bits = self.bit_set(destination_smiles)
        src_counts = self.count_items(source_smiles)
        dst_counts = self.count_items(destination_smiles)
        bit_union = src_bits | dst_bits
        bit_inter = src_bits & dst_bits
        tanimoto = len(bit_inter) / max(len(bit_union), 1)
        keys = set(src_counts) | set(dst_counts)
        signed_mass = 0.0
        action_mass = 0.0
        for key in keys:
            value = dst_counts.get(key, 0.0) - src_counts.get(key, 0.0)
            if value:
                cols.append(int(key)); rows.append(0); data.append(float(value))
                cols.append(self.n_bits + int(key)); rows.append(0); data.append(abs(float(value)))
                signed_mass += float(value)
                action_mass += abs(float(value))
        offset = 2 * self.n_bits
        for j, value in enumerate((source_y, tanimoto, signed_mass, action_mass, float(len(keys)))):
            rows.append(0); cols.append(offset + j); data.append(float(value))
        return sparse.csr_matrix(
            (np.asarray(data, dtype=np.float32), (np.asarray(rows), np.asarray(cols))),
            shape=(1, 2 * self.n_bits + 5),
            dtype=np.float32,
        )


class ExternalSingleAnchorActionRegressor:
    """Single external action-delta model plus one deterministic nearest anchor."""

    def __init__(self, model_path: Path | str = Path("outputs/bindingdb_action_delta_sgd_v1/model.joblib")):
        self.model_path = Path(model_path)

    def fit(self, smiles: list[str], y: np.ndarray) -> "ExternalSingleAnchorActionRegressor":
        payload = joblib.load(self.model_path)
        self.model_ = payload["model"]
        self.scaler_ = payload["scaler"]
        self.n_bits_ = int(payload["n_bits"])
        self.action_featurizer_ = _SparseActionFeaturizer(n_bits=self.n_bits_)
        self.fp_ = Fingerprints(radius=2, n_bits=self.n_bits_)
        self.train_smiles_ = list(smiles)
        self.train_y_ = np.asarray(y, dtype=float)
        self.train_bits_ = self.fp_.bits(np.asarray(self.train_smiles_))
        return self

    def predict(self, smiles: list[str]) -> np.ndarray:
        query_bits = self.fp_.bits(np.asarray(smiles))
        sim = tanimoto_matrix(query_bits, self.train_bits_)
        anchor_idx = np.argmax(sim, axis=1)
        predictions = []
        for query, anchor in zip(smiles, anchor_idx, strict=True):
            source_y = float(self.train_y_[anchor])
            x = self.action_featurizer_.transform_one(
                self.train_smiles_[anchor],
                query,
                source_y,
            )
            delta = float(self.model_.predict(self.scaler_.transform(x))[0])
            predictions.append(source_y + delta)
        return np.asarray(predictions, dtype=float)
