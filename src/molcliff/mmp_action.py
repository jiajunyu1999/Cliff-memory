from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.rdMMPA import FragmentMol
from sklearn.svm import SVR

from .data import Fingerprints


@dataclass(frozen=True)
class Cut:
    core: str
    substituent: str


def _canonical_fragment(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def single_cuts(smiles: str) -> tuple[Cut, ...]:
    """Enumerate deterministic one-bond MMP core/substituent decompositions."""

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ()
    found: set[tuple[str, str]] = set()
    for _unused_core, components in FragmentMol(mol, maxCuts=1, resultsAsMols=False):
        pieces = str(components).split(".")
        if len(pieces) != 2:
            continue
        parsed = [(piece, Chem.MolFromSmiles(piece)) for piece in pieces]
        if any(part is None for _, part in parsed):
            continue
        parsed.sort(key=lambda item: (item[1].GetNumHeavyAtoms(), item[0]), reverse=True)  # type: ignore[union-attr]
        core = _canonical_fragment(parsed[0][0])
        substituent = _canonical_fragment(parsed[1][0])
        if core and substituent:
            found.add((core, substituent))
    return tuple(Cut(core, substituent) for core, substituent in sorted(found))


class MMPActionMemoryRegressor:
    """Transport labels through repeated one-cut matched-pair actions.

    A transform is trusted only when it was observed on at least two *other*
    cores.  This blocks endpoint memorization and makes every correction an
    explicit claim that a chemical action transfers across contexts.
    """

    def __init__(self, min_other_cores: int = 2, core_residual_fallback: bool = False):
        self.min_other_cores = int(min_other_cores)
        self.core_residual_fallback = bool(core_residual_fallback)
        self.fp = Fingerprints(radius=2, n_bits=1024)
        self.core_fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

    def fit(self, smiles: list[str], y: np.ndarray) -> "MMPActionMemoryRegressor":
        self.smiles_ = list(smiles)
        self.y_ = np.asarray(y, dtype=float)
        self.bits_ = self.fp.bits(np.asarray(smiles))
        self.base_ = SVR(C=10.0, epsilon=0.1, gamma=0.01, kernel="rbf")
        self.base_.fit(self.bits_, self.y_)
        self.base_train_prediction_ = np.asarray(self.base_.predict(self.bits_), dtype=float)
        self.cuts_ = [single_cuts(s) for s in smiles]
        groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for index, cuts in enumerate(self.cuts_):
            for cut in cuts:
                groups[cut.core].append((index, cut.substituent))
        self.core_groups_ = groups
        observations: dict[tuple[str, str], list[tuple[float, str]]] = defaultdict(list)
        for core, members in groups.items():
            unique = sorted(set(members))
            for left in range(len(unique)):
                i, sub_i = unique[left]
                for right in range(left + 1, len(unique)):
                    j, sub_j = unique[right]
                    if i == j or sub_i == sub_j:
                        continue
                    delta = float(self.y_[j] - self.y_[i])
                    observations[(sub_i, sub_j)].append((delta, core))
                    observations[(sub_j, sub_i)].append((-delta, core))
        self.observations_ = observations
        self.core_fps_: dict[str, object] = {}
        return self

    def _core_similarity(self, left: str, right: str) -> float:
        def get(smiles: str):
            if smiles not in self.core_fps_:
                mol = Chem.MolFromSmiles(smiles)
                self.core_fps_[smiles] = self.core_fp.GetFingerprint(mol)
            return self.core_fps_[smiles]

        return float(DataStructs.TanimotoSimilarity(get(left), get(right)))

    def _transfer(self, core: str, source_sub: str, destination_sub: str) -> tuple[float, float, int] | None:
        candidates = [
            (delta, seen_core)
            for delta, seen_core in self.observations_.get((source_sub, destination_sub), ())
            if seen_core != core
        ]
        distinct_cores = {seen_core for _, seen_core in candidates}
        if len(distinct_cores) < self.min_other_cores:
            return None
        values = np.asarray([delta for delta, _ in candidates], dtype=float)
        similarities = np.asarray(
            [self._core_similarity(core, seen_core) for _, seen_core in candidates], dtype=float
        )
        weights = np.exp(4.0 * (similarities - similarities.max()))
        effect = float(np.sum(weights * values) / np.sum(weights))
        deviation = float(np.sum(weights * np.abs(values - effect)) / np.sum(weights))
        support = len(distinct_cores)
        return effect, deviation, support

    def predict_components(self, smiles: list[str]) -> dict[str, np.ndarray]:
        query_bits = self.fp.bits(np.asarray(smiles))
        base = np.asarray(self.base_.predict(query_bits), dtype=float)
        action = base.copy()
        confidence = np.zeros(len(smiles), dtype=float)
        support = np.zeros(len(smiles), dtype=int)
        candidate_count = np.zeros(len(smiles), dtype=int)
        for query_index, query_smiles in enumerate(smiles):
            votes = []
            for cut in single_cuts(query_smiles):
                for anchor_index, source_sub in self.core_groups_.get(cut.core, ()):
                    if source_sub == cut.substituent:
                        continue
                    transfer = self._transfer(cut.core, source_sub, cut.substituent)
                    if transfer is None:
                        if self.core_residual_fallback:
                            # The exact retained core is observed, but this
                            # substitution is not transferable yet.  Carry
                            # only the anchor's base-model residual; the base
                            # model itself supplies the unknown action delta.
                            core_size = len(self.core_groups_[cut.core])
                            reliability = 0.2 * core_size / (core_size + 4.0)
                            estimate = base[query_index] + (
                                self.y_[anchor_index] - self.base_train_prediction_[anchor_index]
                            )
                            votes.append((estimate, reliability, 0))
                        continue
                    effect, deviation, n_support = transfer
                    reliability = n_support / (n_support + 2.0) / (1.0 + deviation)
                    votes.append((self.y_[anchor_index] + effect, reliability, n_support))
            if votes:
                values = np.asarray([v[0] for v in votes], dtype=float)
                weights = np.asarray([v[1] for v in votes], dtype=float)
                action[query_index] = float(np.sum(values * weights) / np.sum(weights))
                # Multiple agreeing reconstruction paths raise confidence;
                # disagreement automatically lowers it.
                disagreement = float(np.sum(weights * np.abs(values - action[query_index])) / np.sum(weights))
                confidence[query_index] = min(0.75, float(weights.mean() / (1.0 + disagreement)))
                support[query_index] = max(v[2] for v in votes)
                candidate_count[query_index] = len(votes)
        blended = (1.0 - confidence) * base + confidence * action
        return {
            "base": base,
            "action": action,
            "blended": blended,
            "confidence": confidence,
            "support": support.astype(float),
            "candidate_count": candidate_count.astype(float),
        }

    def predict(self, smiles: list[str]) -> np.ndarray:
        return self.predict_components(smiles)["blended"]


class CoreAnchoredMMPRegressor(MMPActionMemoryRegressor):
    """MMP transport with a conservative exact-core residual fallback."""

    def __init__(self, min_other_cores: int = 2):
        super().__init__(min_other_cores=min_other_cores, core_residual_fallback=True)
