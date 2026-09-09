from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy import sparse
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import MaxAbsScaler


class SparseActionFeaturizer:
    def __init__(self, n_bits: int = 1024):
        self.n_bits = int(n_bits)
        self.bit_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
        self.count_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=n_bits, countSimulation=False
        )
        self.mol_cache: dict[str, Chem.Mol] = {}
        self.bit_cache: dict[str, set[int]] = {}
        self.count_cache: dict[str, dict[int, float]] = {}

    def mol(self, smiles: str) -> Chem.Mol | None:
        if smiles in self.mol_cache:
            return self.mol_cache[smiles]
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        self.mol_cache[smiles] = mol
        return mol

    def bit_set(self, smiles: str) -> set[int] | None:
        if smiles in self.bit_cache:
            return self.bit_cache[smiles]
        mol = self.mol(smiles)
        if mol is None:
            return None
        bits = set(int(x) for x in self.bit_generator.GetFingerprint(mol).GetOnBits())
        self.bit_cache[smiles] = bits
        return bits

    def count_items(self, smiles: str) -> dict[int, float] | None:
        if smiles in self.count_cache:
            return self.count_cache[smiles]
        mol = self.mol(smiles)
        if mol is None:
            return None
        fp = self.count_generator.GetCountFingerprint(mol)
        counts = {
            int(index): min(float(value), 4.0)
            for index, value in fp.GetNonzeroElements().items()
        }
        self.count_cache[smiles] = counts
        return counts

    def transform(
        self,
        source_smiles: list[str],
        destination_smiles: list[str],
        source_y: np.ndarray,
    ) -> tuple[sparse.csr_matrix, np.ndarray]:
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        kept: list[int] = []
        n_cols = 2 * self.n_bits + 5
        scalar_offset = 2 * self.n_bits
        for row_idx, (src, dst) in enumerate(zip(source_smiles, destination_smiles, strict=True)):
            src_bits = self.bit_set(src)
            dst_bits = self.bit_set(dst)
            src_counts = self.count_items(src)
            dst_counts = self.count_items(dst)
            if src_bits is None or dst_bits is None or src_counts is None or dst_counts is None:
                continue
            out_row = len(kept)
            kept.append(row_idx)
            bit_union = src_bits | dst_bits
            bit_inter = src_bits & dst_bits
            tanimoto = len(bit_inter) / max(len(bit_union), 1)
            keys = set(src_counts) | set(dst_counts)
            signed_mass = 0.0
            action_mass = 0.0
            for key in keys:
                value = dst_counts.get(key, 0.0) - src_counts.get(key, 0.0)
                if value:
                    rows.append(out_row); cols.append(int(key)); data.append(float(value))
                    rows.append(out_row); cols.append(self.n_bits + int(key)); data.append(abs(float(value)))
                    signed_mass += float(value)
                    action_mass += abs(float(value))
            for offset, value in enumerate(
                (
                    float(source_y[row_idx]),
                    float(tanimoto),
                    signed_mass,
                    action_mass,
                    float(len(keys)),
                )
            ):
                rows.append(out_row); cols.append(scalar_offset + offset); data.append(value)
        matrix = sparse.csr_matrix(
            (np.asarray(data, dtype=np.float32), (np.asarray(rows), np.asarray(cols))),
            shape=(len(kept), n_cols),
            dtype=np.float32,
        )
        return matrix, np.asarray(kept, dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("../training_data_860.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/bindingdb_action_delta_sgd_v1/model.joblib"))
    parser.add_argument("--max-pairs", type=int, default=500_000)
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument("--n-bits", type=int, default=1024)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    featurizer = SparseActionFeaturizer(n_bits=args.n_bits)
    scaler = MaxAbsScaler()
    model = SGDRegressor(
        loss="huber",
        epsilon=0.2,
        alpha=1e-4,
        learning_rate="invscaling",
        eta0=0.01,
        max_iter=1,
        tol=None,
        random_state=42,
        average=True,
    )
    seen = 0
    fitted = False
    losses = []
    usecols = ["smile1", "smile2", "Label1", "Label2", "Label"]
    for chunk in pd.read_csv(args.input, usecols=usecols, chunksize=args.chunksize):
        if seen >= args.max_pairs:
            break
        if seen + len(chunk) > args.max_pairs:
            chunk = chunk.iloc[: args.max_pairs - seen]
        y1 = chunk.Label1.to_numpy(dtype=np.float32)
        y2 = chunk.Label2.to_numpy(dtype=np.float32)
        x_fwd, kept_fwd = featurizer.transform(chunk.smile1.astype(str).tolist(), chunk.smile2.astype(str).tolist(), y1)
        y_fwd = y2[kept_fwd] - y1[kept_fwd]
        x_rev, kept_rev = featurizer.transform(chunk.smile2.astype(str).tolist(), chunk.smile1.astype(str).tolist(), y2)
        y_rev = y1[kept_rev] - y2[kept_rev]
        x = sparse.vstack([x_fwd, x_rev], format="csr")
        y = np.concatenate([y_fwd, y_rev])
        if len(y) == 0:
            seen += len(chunk)
            continue
        scaler.partial_fit(x)
        x_scaled = scaler.transform(x)
        if fitted:
            pred = model.predict(x_scaled)
            losses.append(float(np.mean(np.abs(pred - y))))
        model.partial_fit(x_scaled, y)
        fitted = True
        seen += len(chunk)
        print(json.dumps({"seen_pairs": seen, "train_rows": int(len(y)), "mae_before_update": losses[-1] if losses else None}), flush=True)

    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "n_bits": args.n_bits,
            "max_pairs": args.max_pairs,
            "chunksize": args.chunksize,
            "seen_pairs": seen,
            "mean_stream_mae": float(np.mean(losses)) if losses else None,
        },
        args.output,
    )
    args.output.with_suffix(".summary.json").write_text(
        json.dumps({"seen_pairs": seen, "mean_stream_mae": float(np.mean(losses)) if losses else None}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
