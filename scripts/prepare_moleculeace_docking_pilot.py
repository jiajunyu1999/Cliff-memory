from __future__ import annotations

"""Prepare one leakage-free MoleculeACE internal-validation docking pilot."""

import argparse
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from molcliff.data import load_moleculeace, split_official_train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="CHEMBL4203_Ki")
    parser.add_argument("--pockets", type=Path, default=Path("data/moleculeace_holo_pockets.json"))
    parser.add_argument("--pdb-dir", type=Path, default=Path("data/holo_pdb"))
    parser.add_argument("--output-root", type=Path, default=Path("data/docking_pilot"))
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--mode", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    full = load_moleculeace(args.dataset)
    if args.mode == "validation":
        fit, evaluation = split_official_train(full)
        evaluation_role = "validation"
    else:
        fit = full[full.split.eq("train")].reset_index(drop=True)
        evaluation = full[full.split.eq("test")].reset_index(drop=True)
        evaluation_role = "test"
    fit = fit.copy(); fit["role"] = "fit"
    evaluation = evaluation.copy(); evaluation["role"] = evaluation_role
    frame = pd.concat([fit, evaluation], ignore_index=True)

    pocket = json.loads(args.pockets.read_text())["datasets"][args.dataset]
    pdb_id, chain = pocket["pdb_id"], pocket["chain"]
    ligand = pocket["ligand"]
    source = (args.pdb_dir / f"{pdb_id}.pdb").read_text(errors="ignore").splitlines()
    output = args.output_root / args.dataset
    output.mkdir(parents=True, exist_ok=True)
    receptor_lines, reference_lines = [], []
    for line in source:
        if line.startswith("ATOM") and line[21].strip() == chain:
            receptor_lines.append(line)
        elif (line.startswith("HETATM") and line[17:20].strip() == ligand["resname"]
              and line[21].strip() == ligand["chain"]
              and line[22:27].strip() == ligand["resid"]
              and line[16].strip() in {"", "A"}):
            reference_lines.append(line)
    if not receptor_lines or not reference_lines:
        raise RuntimeError("Failed to isolate receptor chain or reference ligand")
    (output / "receptor.pdb").write_text("\n".join(receptor_lines + ["TER", "END"]) + "\n")
    (output / "reference_ligand.pdb").write_text("\n".join(reference_lines + ["END"]) + "\n")

    writer = Chem.SDWriter(str(output / "queries.sdf"))
    chunk_writers = [Chem.SDWriter(str(output / f"queries.{i:02d}.sdf"))
                     for i in range(args.chunks)]
    evaluation_writers = [Chem.SDWriter(str(output / f"queries_evaluation.{i:02d}.sdf"))
                          for i in range(args.chunks)]
    manifest = []
    for index, row in enumerate(frame.itertuples()):
        mol = Chem.AddHs(Chem.MolFromSmiles(str(row.smiles)))
        params = AllChem.ETKDGv3(); params.randomSeed = 42
        status = AllChem.EmbedMolecule(mol, params)
        if status != 0:
            status = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
        if status != 0:
            raise RuntimeError(f"Conformer generation failed for row {index}")
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass
        name = f"{index:05d}"
        mol.SetProp("_Name", name)
        mol.SetProp("SMILES", str(row.smiles))
        mol.SetProp("ROLE", str(row.role))
        writer.write(mol)
        chunk_writers[index % args.chunks].write(mol)
        if row.role != "fit":
            evaluation_writers[index % args.chunks].write(mol)
        manifest.append({"name": name, "smiles": row.smiles, "role": row.role,
                         "target": row.target, "cliff_mol": bool(row.cliff_mol)})
    writer.close()
    for chunk_writer in chunk_writers:
        chunk_writer.close()
    for evaluation_writer in evaluation_writers:
        evaluation_writer.close()
    pd.DataFrame(manifest).to_csv(output / "manifest.csv", index=False)
    metadata = {
        "dataset": args.dataset, "pdb_id": pdb_id, "chain": chain,
        "reference_ligand": ligand, "molecules": len(frame),
        "fit": len(fit), "evaluation": len(evaluation), "mode": args.mode, "seed": 42,
        "chunks": args.chunks,
        "uses_official_test": args.mode == "test",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
