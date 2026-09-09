from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_acnet_action.py"
SPEC = importlib.util.spec_from_file_location("run_acnet_action", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
acnet = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = acnet
SPEC.loader.exec_module(acnet)


def _toy_items() -> list[dict[str, str]]:
    items = []
    for i in range(48):
        items.append(
            {
                "SMILES1": "CCO" if i % 2 else "CCC",
                "SMILES2": "CCN" if i % 3 else "CCCl",
                "Value": str(i % 2),
            }
        )
    return items


def test_acnet_official_random_split_is_deterministic_and_valid() -> None:
    items = _toy_items()
    first = acnet.official_random_split(items, seed=8)
    second = acnet.official_random_split(items, seed=8)
    np.testing.assert_array_equal(first.train, second.train)
    np.testing.assert_array_equal(first.valid, second.valid)
    np.testing.assert_array_equal(first.test, second.test)
    labels = np.asarray([int(item["Value"]) for item in items])
    assert set(labels[first.valid]) == {0, 1}
    assert set(labels[first.test]) == {0, 1}


def test_pair_features_are_order_invariant() -> None:
    left = [{"SMILES1": "CCOc1ccccc1", "SMILES2": "CCNc1ccccc1", "Value": "1"}]
    right = [{"SMILES1": "CCNc1ccccc1", "SMILES2": "CCOc1ccccc1", "Value": "1"}]
    for featurizer_cls in [acnet.PairFeaturizer, acnet.SparsePairFeaturizer]:
        featurizer = featurizer_cls(n_bits=128)
        for mode in ["endpoint", "action", "endpoint_action"]:
            observed = featurizer.transform(left, mode)
            expected = featurizer.transform(right, mode)
            if hasattr(observed, "toarray"):
                observed = observed.toarray()
                expected = expected.toarray()
            np.testing.assert_allclose(observed, expected)


def test_target_split_keeps_targets_disjoint() -> None:
    items = []
    for target in range(20):
        for idx in range(10 + target):
            items.append(
                {
                    "SMILES1": "CCO",
                    "SMILES2": "CCN",
                    "Value": str(idx % 2),
                    "Target": str(target),
                }
            )
    split = acnet.official_target_split(items, seed=8)
    targets = np.asarray([item["Target"] for item in items])
    train_targets = set(targets[split.train])
    valid_targets = set(targets[split.valid])
    test_targets = set(targets[split.test])
    assert train_targets.isdisjoint(valid_targets)
    assert train_targets.isdisjoint(test_targets)
    assert valid_targets.isdisjoint(test_targets)
