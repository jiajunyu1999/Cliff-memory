from __future__ import annotations

"""Build validation poses by fixing each query's common core to a fit anchor."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdFMCS

from molcliff.data import load_moleculeace, split_official_train
from validate_moleculeace_operator_action_kernel import exact_kernel


def load_poses(root: Path) -> dict[int, Chem.Mol]:
    poses: dict[int, Chem.Mol] = {}
    for path in sorted(root.glob("docked.[0-9][0-9].sdf.gz")):
        with gzip.open(path, "rb") as handle:
            for mol in Chem.ForwardSDMolSupplier(handle, removeHs=False):
                if mol is not None:
                    poses[int(mol.GetProp("_Name"))] = mol
    return poses


def constrained_pose(smiles: str, anchor: Chem.Mol) -> tuple[Chem.Mol, int]:
    query_heavy = Chem.MolFromSmiles(smiles)
    anchor_heavy = Chem.RemoveHs(anchor)
    mcs = rdFMCS.FindMCS(
        [query_heavy, anchor_heavy], timeout=5, ringMatchesRingOnly=True,
        completeRingsOnly=True, matchValences=True,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrder,
    )
    pattern = Chem.MolFromSmarts(mcs.smartsString)
    query_match = query_heavy.GetSubstructMatch(pattern)
    anchor_match = anchor_heavy.GetSubstructMatch(pattern)
    if len(query_match) < 3 or len(query_match) != len(anchor_match):
        raise RuntimeError("No usable common core")

    query = Chem.AddHs(query_heavy)
    anchor_conf = anchor_heavy.GetConformer()
    coordinate_map = {
        int(query_index): anchor_conf.GetAtomPosition(int(anchor_index))
        for query_index, anchor_index in zip(query_match, anchor_match, strict=True)
    }
    status = AllChem.EmbedMolecule(
        query, coordMap=coordinate_map, randomSeed=42, useRandomCoords=True,
        enforceChirality=True,
    )
    if status != 0:
        raise RuntimeError("Constrained embedding failed")
    AllChem.AlignMol(
        query, anchor_heavy,
        atomMap=list(zip(map(int, query_match), map(int, anchor_match), strict=True)),
    )
    properties = AllChem.MMFFGetMoleculeProperties(query)
    if properties is not None:
        force_field = AllChem.MMFFGetMoleculeForceField(query, properties)
    else:
        force_field = AllChem.UFFGetMoleculeForceField(query)
    for atom_index in query_match:
        force_field.AddFixedPoint(int(atom_index))
    force_field.Initialize()
    force_field.Minimize(maxIts=300)
    return query, len(query_match)


def main() -> None:
    RDLogger.DisableLog("rdApp.warning")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="CHEMBL4203_Ki")
    parser.add_argument("--docking-root", type=Path, default=Path("data/docking_pilot"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("data/core_constrained_docking"))
    args = parser.parse_args()
    fit, valid = split_official_train(load_moleculeace(args.dataset))
    chemistry_fit, chemistry_valid = exact_kernel(fit, valid)
    poses = load_poses(args.docking_root / args.dataset)
    if len(poses) != len(fit) + len(valid):
        raise RuntimeError(f"Incomplete anchor docking cache: {len(poses)}")

    output = args.output_root / args.dataset
    output.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(output / "queries.constrained.sdf"))
    core_sizes = []
    failures = 0
    for valid_index, row in enumerate(valid.itertuples(index=False)):
        anchor_index = int(np.argmax(chemistry_valid[valid_index]))
        try:
            mol, core_size = constrained_pose(str(row.smiles), poses[anchor_index])
        except RuntimeError:
            # Retain the independently docked pose only when a meaningful
            # common-core coordinate constraint cannot be constructed.
            mol = Chem.Mol(poses[len(fit) + valid_index])
            core_size = 0
            failures += 1
        mol.SetProp("_Name", str(len(fit) + valid_index))
        mol.SetProp("SMILES", str(row.smiles))
        mol.SetIntProp("ANCHOR_INDEX", anchor_index)
        mol.SetIntProp("COMMON_CORE_HEAVY_ATOMS", core_size)
        writer.write(mol)
        core_sizes.append(core_size)
    writer.close()
    audit = {
        "dataset": args.dataset, "queries": len(valid), "failures": failures,
        "median_common_core_heavy_atoms": float(np.median(core_sizes)),
        "mean_common_core_heavy_atoms": float(np.mean(core_sizes)),
        "uses_test": False,
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
