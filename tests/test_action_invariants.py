import numpy as np

from molcliff.calibrated_action import _stable_calibration_mask
from molcliff.integrable_action import IntegrableActionRegressor
from molcliff.mmp_action import single_cuts


def test_calibration_fold_is_deterministic_and_nonempty() -> None:
    smiles = [f"C{'C' * i}O" for i in range(1, 40)]
    first = _stable_calibration_mask(smiles)
    second = _stable_calibration_mask(list(smiles))
    np.testing.assert_array_equal(first, second)
    assert 0 < first.sum() < len(first)


def test_integrable_potential_is_antisymmetric_and_cycle_closed() -> None:
    potential = {"a": -0.4, "b": 0.7, "c": 1.1}

    def delta(source: str, destination: str) -> float:
        return potential[destination] - potential[source]

    assert np.isclose(delta("a", "b"), -delta("b", "a"))
    assert np.isclose(delta("a", "b") + delta("b", "c") + delta("c", "a"), 0.0)


def test_single_cuts_are_unique_and_deterministic() -> None:
    first = single_cuts("CCOc1ccccc1")
    second = single_cuts("CCOc1ccccc1")
    assert first == second
    assert len(first) == len(set(first))
    assert first
