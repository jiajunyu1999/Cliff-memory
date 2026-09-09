from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy import sparse
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACNET_ROOT = PROJECT_ROOT / "data" / "acnet" / "generated"


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray
    valid_seed: int
    test_seed: int


def _labels(items: list[dict]) -> np.ndarray:
    return np.asarray([int(item["Value"]) for item in items], dtype=np.int8)


def _has_two_classes(y: np.ndarray) -> bool:
    return len(np.unique(y)) == 2


def official_random_split(
    items: list[dict],
    split_rate: tuple[float, float] = (0.8, 0.1),
    seed: int = 8,
) -> Split:
    """Replicate ACNet's RandomSplitter control flow for binary tasks.

    ACNet advances the seed only when a validation or test set has a missing
    class.  That is a validity guard, not model selection, so the selected seed
    is recorded in the output for auditability.
    """

    total = len(items)
    train_num = int(total * split_rate[0])
    valid_num = int(total * split_rate[1])
    index = np.arange(total)
    y = _labels(items)

    test_seed = seed
    while True:
        random.seed(test_seed)
        random.shuffle(index)
        test = index[(train_num + valid_num) :].copy()
        if _has_two_classes(y[test]):
            break
        test_seed += 1

    remain = index[: (train_num + valid_num)].copy()
    valid_seed = seed
    while True:
        random.seed(valid_seed)
        random.shuffle(remain)
        train = remain[:train_num].copy()
        valid = remain[train_num:].copy()
        if _has_two_classes(y[valid]):
            break
        valid_seed += 1

    return Split(train=train, valid=valid, test=test, valid_seed=valid_seed, test_seed=test_seed)


def _target_sample(
    target_to_indices: dict[str, list[int]],
    candidate_targets: list[str],
    sample_size: int,
    optimal_count: int,
    seed: int,
) -> tuple[list[str], int, int, float]:
    """Replicate ACNet target-count sampling without using labels."""

    error_rate = 0.1
    tried = 0
    while True:
        tried += 1
        seed += 1
        random.seed(seed)
        chosen = random.sample(candidate_targets, sample_size)
        count = sum(len(target_to_indices[target]) for target in chosen)
        if optimal_count * (1 - error_rate) <= count <= optimal_count * (1 + error_rate):
            return chosen, count, seed, error_rate
        if tried % 5000 == 0:
            error_rate += 0.05
            sample_size = int(sample_size * 1.1)
            if sample_size >= len(candidate_targets):
                raise ValueError("target split sample size exhausted all targets")


def official_target_split(
    items: list[dict],
    split_rate: tuple[float, float] = (0.8, 0.1),
    seed: int = 8,
) -> Split:
    total = len(items)
    train_num = int(total * split_rate[0])
    valid_num = int(total * split_rate[1])
    test_num = total - train_num - valid_num
    target_to_indices: dict[str, list[int]] = {}
    for idx, item in enumerate(items):
        target_to_indices.setdefault(str(item["Target"]), []).append(idx)
    targets = list(target_to_indices)
    test_target_count = int(len(targets) * (1 - split_rate[0] - split_rate[1]))
    test_targets, _, test_seed, _ = _target_sample(
        target_to_indices, targets, test_target_count, test_num, seed
    )
    remaining = [target for target in targets if target not in set(test_targets)]
    valid_target_count = int(len(targets) * split_rate[1])
    valid_targets, _, valid_seed, _ = _target_sample(
        target_to_indices, remaining, valid_target_count, valid_num, seed
    )
    valid_set = set(valid_targets)
    test_set = set(test_targets)
    train = []
    valid = []
    test = []
    for target, indices in target_to_indices.items():
        if target in test_set:
            test.extend(indices)
        elif target in valid_set:
            valid.extend(indices)
        else:
            train.extend(indices)
    return Split(
        train=np.asarray(train, dtype=int),
        valid=np.asarray(valid, dtype=int),
        test=np.asarray(test, dtype=int),
        valid_seed=valid_seed,
        test_seed=test_seed,
    )


class PairFeaturizer:
    def __init__(self, n_bits: int = 1024, radius: int = 2):
        self.n_bits = int(n_bits)
        self.bit_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits
        )
        self.count_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits, countSimulation=False
        )
        self.mol_cache: dict[str, Chem.Mol] = {}
        self.bit_cache: dict[str, np.ndarray] = {}
        self.count_cache: dict[str, np.ndarray] = {}

    def _mol(self, smiles: str) -> Chem.Mol:
        if smiles in self.mol_cache:
            return self.mol_cache[smiles]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"RDKit failed on SMILES: {smiles}")
        self.mol_cache[smiles] = mol
        return mol

    def _bit(self, smiles: str) -> np.ndarray:
        if smiles in self.bit_cache:
            return self.bit_cache[smiles]
        out = np.zeros(self.n_bits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(
            self.bit_generator.GetFingerprint(self._mol(smiles)), out
        )
        self.bit_cache[smiles] = out
        return out

    def _count(self, smiles: str) -> np.ndarray:
        if smiles in self.count_cache:
            return self.count_cache[smiles]
        out = np.zeros(self.n_bits, dtype=np.float32)
        fp = self.count_generator.GetCountFingerprint(self._mol(smiles))
        for index, value in fp.GetNonzeroElements().items():
            out[int(index)] = min(float(value), 4.0)
        self.count_cache[smiles] = out
        return out

    def transform(self, items: list[dict], mode: str) -> np.ndarray:
        smiles = sorted({item["SMILES1"] for item in items} | {item["SMILES2"] for item in items})
        bit_cache = {smi: self._bit(smi) for smi in smiles}
        count_cache = {smi: self._count(smi) for smi in smiles}
        rows = []
        for item in items:
            b1 = bit_cache[item["SMILES1"]]
            b2 = bit_cache[item["SMILES2"]]
            c1 = count_cache[item["SMILES1"]]
            c2 = count_cache[item["SMILES2"]]
            shared = np.minimum(b1, b2)
            action = np.abs(c1 - c2)
            xor = np.abs(b1 - b2)
            inter = float(np.dot(b1, b2))
            union = float(b1.sum() + b2.sum() - inter)
            tanimoto = inter / max(union, 1.0)
            size_delta = abs(float(c1.sum() - c2.sum()))
            action_mass = float(action.sum())
            if mode == "endpoint":
                row = np.concatenate([shared, np.maximum(b1, b2), [tanimoto]])
            elif mode == "action":
                row = np.concatenate([xor, action, [tanimoto, size_delta, action_mass]])
            elif mode == "endpoint_action":
                row = np.concatenate(
                    [shared, np.maximum(b1, b2), xor, action, [tanimoto, size_delta, action_mass]]
                )
            else:
                raise ValueError(f"unknown feature mode: {mode}")
            rows.append(row.astype(np.float32, copy=False))
        return np.vstack(rows)


class SparsePairFeaturizer:
    def __init__(self, n_bits: int = 1024, radius: int = 2):
        self.n_bits = int(n_bits)
        self.bit_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits
        )
        self.count_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits, countSimulation=False
        )
        self.mol_cache: dict[str, Chem.Mol] = {}
        self.bit_cache: dict[str, np.ndarray] = {}
        self.count_cache: dict[str, dict[int, float]] = {}

    def _mol(self, smiles: str) -> Chem.Mol:
        if smiles in self.mol_cache:
            return self.mol_cache[smiles]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"RDKit failed on SMILES: {smiles}")
        self.mol_cache[smiles] = mol
        return mol

    def _bit_indices(self, smiles: str) -> np.ndarray:
        if smiles in self.bit_cache:
            return self.bit_cache[smiles]
        bits = np.asarray(
            list(self.bit_generator.GetFingerprint(self._mol(smiles)).GetOnBits()),
            dtype=np.int32,
        )
        self.bit_cache[smiles] = bits
        return bits

    def _count_items(self, smiles: str) -> dict[int, float]:
        if smiles in self.count_cache:
            return self.count_cache[smiles]
        fp = self.count_generator.GetCountFingerprint(self._mol(smiles))
        counts = {
            int(index): min(float(value), 4.0)
            for index, value in fp.GetNonzeroElements().items()
        }
        self.count_cache[smiles] = counts
        return counts

    def transform(self, items: list[dict], mode: str) -> sparse.csr_matrix:
        smiles = sorted({item["SMILES1"] for item in items} | {item["SMILES2"] for item in items})
        bit_cache = {smi: self._bit_indices(smi) for smi in smiles}
        count_cache = {smi: self._count_items(smi) for smi in smiles}
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        if mode == "endpoint":
            n_cols = 2 * self.n_bits + 1
            scalar_offset = 2 * self.n_bits
        elif mode == "action":
            n_cols = 2 * self.n_bits + 3
            scalar_offset = 2 * self.n_bits
        elif mode == "endpoint_action":
            n_cols = 4 * self.n_bits + 3
            scalar_offset = 4 * self.n_bits
        else:
            raise ValueError(f"unknown feature mode: {mode}")

        for row_idx, item in enumerate(items):
            b1 = bit_cache[item["SMILES1"]]
            b2 = bit_cache[item["SMILES2"]]
            c1 = count_cache[item["SMILES1"]]
            c2 = count_cache[item["SMILES2"]]
            set1 = set(int(x) for x in b1)
            set2 = set(int(x) for x in b2)
            shared = set1 & set2
            union = set1 | set2
            xor = set1 ^ set2
            tanimoto = len(shared) / max(len(union), 1)
            count_keys = set(c1) | set(c2)
            action_values = {
                key: abs(c1.get(key, 0.0) - c2.get(key, 0.0))
                for key in count_keys
            }
            action_values = {key: value for key, value in action_values.items() if value > 0}
            size_delta = abs(sum(c1.values()) - sum(c2.values()))
            action_mass = sum(action_values.values())

            def add_sparse(columns: set[int] | dict[int, float], offset: int = 0) -> None:
                for column, value in (
                    columns.items() if isinstance(columns, dict) else ((column, 1.0) for column in columns)
                ):
                    rows.append(row_idx)
                    cols.append(offset + int(column))
                    data.append(float(value))

            if mode == "endpoint":
                add_sparse(shared, 0)
                add_sparse(union, self.n_bits)
                rows.append(row_idx); cols.append(scalar_offset); data.append(float(tanimoto))
            elif mode == "action":
                add_sparse(xor, 0)
                add_sparse(action_values, self.n_bits)
                for j, value in enumerate((tanimoto, size_delta, action_mass)):
                    rows.append(row_idx); cols.append(scalar_offset + j); data.append(float(value))
            else:
                add_sparse(shared, 0)
                add_sparse(union, self.n_bits)
                add_sparse(xor, 2 * self.n_bits)
                add_sparse(action_values, 3 * self.n_bits)
                for j, value in enumerate((tanimoto, size_delta, action_mass)):
                    rows.append(row_idx); cols.append(scalar_offset + j); data.append(float(value))

        return sparse.csr_matrix(
            (np.asarray(data, dtype=np.float32), (np.asarray(rows), np.asarray(cols))),
            shape=(len(items), n_cols),
            dtype=np.float32,
        )


def _fit_predict(
    model_name: str, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray
) -> np.ndarray:
    if model_name == "logreg":
        scaler = StandardScaler(with_mean=False)
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            solver="liblinear",
            random_state=42,
        )
        model.fit(x_train, y_train)
        return model.predict_proba(x_test)[:, 1]
    if model_name == "sgd_logreg":
        scaler = StandardScaler(with_mean=False)
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        model = SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            max_iter=50,
            tol=1e-4,
            class_weight="balanced",
            average=True,
            random_state=42,
        )
        model.fit(x_train, y_train)
        return model.predict_proba(x_test)[:, 1]
    if model_name == "extratrees":
        if sparse.issparse(x_train):
            x_train = x_train.toarray()
            x_test = x_test.toarray()
        model = ExtraTreesClassifier(
            n_estimators=384,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        return model.predict_proba(x_test)[:, 1]
    if model_name == "hgb":
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            class_weight="balanced",
            random_state=42,
        )
        model.fit(x_train, y_train)
        return model.predict_proba(x_test)[:, 1]
    raise ValueError(f"unknown model: {model_name}")


def _metrics(y_true: np.ndarray, score: np.ndarray) -> dict[str, float]:
    pred = (score >= 0.5).astype(np.int8)
    return {
        "auc": float(roc_auc_score(y_true, score)),
        "ap": float(average_precision_score(y_true, score)),
        "acc": float(accuracy_score(y_true, pred)),
    }


def run_dataset(args: argparse.Namespace) -> None:
    data_path = ACNET_ROOT / args.data
    all_data = json.loads(data_path.read_text())
    if args.targets == "all":
        target_ids = list(all_data)
    else:
        requested = {x.strip() for x in args.targets.split(",") if x.strip()}
        target_ids = [target for target in all_data if target in requested]
        missing = requested.difference(target_ids)
        if missing:
            raise ValueError(f"targets not found: {sorted(missing)}")

    out_dir = PROJECT_ROOT / "outputs" / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    featurizer = (
        SparsePairFeaturizer(n_bits=args.n_bits)
        if args.sparse
        else PairFeaturizer(n_bits=args.n_bits)
    )
    rows: list[dict] = []

    for target_id in target_ids:
        raw_items = all_data[target_id]
        items = []
        for item in raw_items:
            try:
                featurizer._mol(item["SMILES1"])
                featurizer._mol(item["SMILES2"])
                items.append(item)
            except ValueError:
                pass
        if args.splitter == "random":
            split = official_random_split(items, seed=args.seed)
        else:
            split = official_target_split(items, seed=args.seed)
        y = _labels(items)
        prediction_rows = pd.DataFrame(
            {
                "idx": split.test,
                "target": target_id,
                "smiles1": [items[i]["SMILES1"] for i in split.test],
                "smiles2": [items[i]["SMILES2"] for i in split.test],
                "y_true": y[split.test],
            }
        )
        base_metrics: dict[str, dict[str, float]] = {}
        for mode in args.feature_modes:
            print(
                f"target={target_id} mode={mode} transform n={len(items)}",
                flush=True,
            )
            feature = featurizer.transform(items, mode)
            print(
                f"target={target_id} mode={mode} fit model={args.model}",
                flush=True,
            )
            score = _fit_predict(
                args.model,
                feature[split.train],
                y[split.train],
                feature[split.test],
            )
            met = _metrics(y[split.test], score)
            base_metrics[mode] = met
            prediction_rows[f"{mode}_score"] = score
            rows.append(
                {
                    "dataset": args.data,
                    "target": target_id,
                    "model": args.model,
                    "feature_mode": mode,
                    "n": len(items),
                    "n_train": len(split.train),
                    "n_valid": len(split.valid),
                    "n_test": len(split.test),
                    "test_pos": int(y[split.test].sum()),
                    "valid_pos": int(y[split.valid].sum()),
                    "train_pos": int(y[split.train].sum()),
                    "valid_seed": split.valid_seed,
                    "test_seed": split.test_seed,
                    "splitter": args.splitter,
                    **met,
                }
            )
            del feature
        prediction_rows.to_csv(out_dir / f"{target_id}.predictions.csv", index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    macro = (
        summary.groupby(["model", "feature_mode"], as_index=False)[["auc", "ap", "acc"]]
        .mean()
        .sort_values(["auc", "ap"], ascending=False)
    )
    wins = []
    endpoint = summary[summary.feature_mode.eq("endpoint")].set_index("target")
    for mode in args.feature_modes:
        if mode == "endpoint":
            continue
        other = summary[summary.feature_mode.eq(mode)].set_index("target")
        diff = other["auc"] - endpoint["auc"]
        wins.append(
            {
                "feature_mode": mode,
                "auc_wins": int((diff > 1e-12).sum()),
                "auc_ties": int((diff.abs() <= 1e-12).sum()),
                "auc_losses": int((diff < -1e-12).sum()),
                "mean_auc_gain": float(diff.mean()),
                "min_auc_gain": float(diff.min()),
            }
        )
    report = {
        "data": args.data,
        "model": args.model,
        "seed": args.seed,
        "targets": len(target_ids),
        "macro": macro.to_dict(orient="records"),
        "wins_vs_endpoint": wins,
    }
    (out_dir / "aggregate.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="MMP_AC_Small.json")
    parser.add_argument("--targets", default="all")
    parser.add_argument(
        "--model",
        choices=["logreg", "sgd_logreg", "extratrees", "hgb"],
        default="logreg",
    )
    parser.add_argument("--splitter", choices=["random", "target"], default="random")
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument(
        "--feature-modes",
        nargs="+",
        default=["endpoint", "action", "endpoint_action"],
        choices=["endpoint", "action", "endpoint_action"],
    )
    parser.add_argument("--n-bits", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--output", default="acnet_small_action_v1")
    args = parser.parse_args()
    run_dataset(args)


if __name__ == "__main__":
    main()
