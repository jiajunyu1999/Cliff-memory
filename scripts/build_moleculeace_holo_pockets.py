from __future__ import annotations

"""Build label-free experimental holo pockets for MoleculeACE targets.

PDBe's UniProt-to-PDB best-structure mapping supplies candidates.  Among a
fixed number of unique entries, selection prefers the largest non-polymer
small molecule, then chain coverage and resolution.  No assay labels are read.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


EXCLUDED = {
    "HOH", "DOD", "SO4", "PO4", "GOL", "EDO", "PEG", "PG4", "ACT",
    "FMT", "ACE", "TRS", "MES", "HEP", "BME", "DMS", "MPD", "IOD",
    "CL", "BR", "NA", "K", "CA", "MG", "MN", "ZN", "FE", "CU", "CO",
    # Canonical and common modified polymer residues.  Many modified residues
    # are serialized as HETATM and must not become putative holo ligands.
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "ASX", "GLX", "SEC", "PYL", "MSE", "SEP", "TPO", "PTR", "TYS", "CGU",
    "CSO", "CSD", "CME", "HYP", "MLY", "KCX", "LLP", "PCA", "FME", "ORN",
}


def _download(session: requests.Session, url: str, path: Path) -> str | None:
    if path.exists() and path.stat().st_size > 1000:
        return path.read_text(errors="ignore")
    response = session.get(url, timeout=60)
    if response.status_code != 200 or len(response.content) < 1000:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return response.text


def _atom(line: str) -> dict | None:
    try:
        return {
            "name": line[12:16].strip(), "resname": line[17:20].strip(),
            "altloc": line[16].strip(),
            "chain": line[21].strip(), "resid": line[22:27].strip(),
            "xyz": np.asarray([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
            "element": (line[76:78].strip() or line[12:14].strip()).upper(),
        }
    except (ValueError, IndexError):
        return None


def _extract(text: str, target_chain: str) -> list[dict]:
    protein = []
    ligands: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for line in text.splitlines():
        # Keep one conformer so alternate-location records do not inflate a
        # residue's heavy-atom count.  Blank and A are the conventional first
        # conformer choices in legacy PDB files.
        if len(line) > 16 and line[16].strip() not in {"", "A"}:
            continue
        if line.startswith("ATOM") and line[21].strip() == target_chain:
            atom = _atom(line)
            if atom is not None and atom["element"] != "H":
                protein.append(atom)
        elif line.startswith("HETATM"):
            atom = _atom(line)
            if atom is not None and atom["element"] != "H" and atom["resname"] not in EXCLUDED:
                ligands[(atom["resname"], atom["chain"], atom["resid"])].append(atom)
    if not protein:
        return []
    protein_xyz = np.vstack([a["xyz"] for a in protein])
    candidates = []
    for key, atoms in ligands.items():
        unique_atoms = {atom["name"]: atom for atom in atoms}
        atoms = list(unique_atoms.values())
        if not 12 <= len(atoms) <= 80:
            continue
        ligand_xyz = np.vstack([a["xyz"] for a in atoms])
        min_distance = float(np.sqrt(((ligand_xyz[:, None] - protein_xyz[None]) ** 2).sum(2)).min())
        if min_distance <= 5.0:
            candidates.append((len(atoms), key, atoms))
    if not candidates:
        return []
    _size, ligand_key, ligand_atoms = max(candidates, key=lambda item: item[0])
    ligand_xyz = np.vstack([a["xyz"] for a in ligand_atoms])
    residues: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for atom in protein:
        residues[(atom["resid"], atom["resname"])].append(atom)
    pocket = []
    for (resid, resname), atoms in residues.items():
        xyz = np.vstack([a["xyz"] for a in atoms])
        distance = float(np.sqrt(((xyz[:, None] - ligand_xyz[None]) ** 2).sum(2)).min())
        if distance <= 6.0:
            ca = next((a["xyz"] for a in atoms if a["name"] == "CA"), xyz.mean(0))
            pocket.append({"resid": resid, "resname": resname,
                           "coord": [round(float(v), 3) for v in ca],
                           "ligand_distance": round(distance, 3)})
    pocket.sort(key=lambda row: (row["ligand_distance"], row["resid"]))
    # Reject crystallization additives that merely touch one exposed loop.
    # A drug-binding pocket should surround the ligand on several residues.
    if len(pocket) < 8:
        return []
    return [{"ligand": {"resname": ligand_key[0], "chain": ligand_key[1],
                         "resid": ligand_key[2], "heavy_atoms": len(ligand_atoms)},
             "residues": pocket}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=Path, default=Path("data/moleculeace_target_context.csv"))
    parser.add_argument("--pdb-dir", type=Path, default=Path("data/holo_pdb"))
    parser.add_argument("--output", type=Path, default=Path("data/moleculeace_holo_pockets.json"))
    parser.add_argument("--max-entries", type=int, default=24)
    args = parser.parse_args()
    context = pd.read_csv(args.context)
    session = requests.Session(); session.headers["User-Agent"] = "PC-GAM-cliff-research/1.0"
    session.mount("https://", HTTPAdapter(max_retries=Retry(
        total=5, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",))))
    by_accession = {}
    for accession in sorted(context.uniprot_accession.unique()):
        mapping_url = f"https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/{accession}"
        try:
            response = session.get(mapping_url, timeout=60)
        except requests.RequestException:
            response = None
        entries = response.json().get(accession, []) if response is not None and response.status_code == 200 else []
        seen, valid = set(), []
        for item in entries:
            pdb_id = item["pdb_id"].lower()
            if pdb_id in seen or len(seen) >= args.max_entries:
                continue
            seen.add(pdb_id)
            path = args.pdb_dir / f"{pdb_id}.pdb"
            text = _download(session, f"https://files.rcsb.org/download/{pdb_id}.pdb", path)
            if text is None:
                continue
            extracted = _extract(text, str(item["chain_id"]))
            if not extracted:
                continue
            resolution = item.get("resolution")
            valid.append({"pdb_id": pdb_id, "chain": str(item["chain_id"]),
                          "coverage": float(item.get("coverage") or 0.0),
                          "resolution": float(resolution) if resolution else 99.0,
                          **extracted[0]})
        if valid:
            # Prefer a substantial drug-like ligand over high-coverage entries
            # containing only a modified residue; coverage and resolution then
            # break ties without consulting activity data.
            chosen = max(valid, key=lambda row: (row["ligand"]["heavy_atoms"],
                                                  row["coverage"], -row["resolution"]))
            by_accession[accession] = chosen
            print(json.dumps({"accession": accession, "pdb": chosen["pdb_id"],
                              "ligand": chosen["ligand"]["resname"],
                              "pocket_residues": len(chosen["residues"])}), flush=True)
        else:
            by_accession[accession] = None
            print(json.dumps({"accession": accession, "pdb": None}), flush=True)
        time.sleep(0.05)
    result = {"selection": {"source": "PDBe best_structures + RCSB PDB",
                             "max_entries": args.max_entries, "ligand_heavy_atoms": [12, 80],
                             "ligand_contact_angstrom": 5.0, "pocket_radius_angstrom": 6.0,
                             "minimum_pocket_residues": 8,
                             "uses_activity_labels": False},
              "accessions": by_accession,
              "datasets": {row.dataset: by_accession[row.uniprot_accession]
                           for row in context.itertuples()}}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    covered = sum(value is not None for value in by_accession.values())
    print(json.dumps({"unique_targets": len(by_accession), "covered": covered,
                      "coverage": covered / len(by_accession)}))


if __name__ == "__main__":
    main()
