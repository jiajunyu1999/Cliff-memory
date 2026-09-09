from __future__ import annotations

"""Extract frozen, label-free PSICHIC interaction features for MoleculeACE.

This script deliberately processes only the requested benchmark split and never
loads its activity column.  The default checkpoint is the PDBBind-only model;
the human multitask checkpoint is excluded because its training pairs overlap
MoleculeACE.
"""

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger

from molcliff.data import MOLECULEACE_DATASETS, load_moleculeace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PSICHIC = PROJECT_ROOT.parents[2] / "PSICHIC"
RDLogger.DisableLog("rdApp.warning")


def expose_psichic_modules(psichic_root: Path) -> None:
    """Expose submodules without executing PSICHIC's training-only utils imports."""
    sys.path.insert(0, str(psichic_root))
    if "utils" not in sys.modules:
        package = types.ModuleType("utils")
        package.__path__ = [str(psichic_root / "utils")]
        sys.modules["utils"] = package


def load_model(psichic_root: Path, device: torch.device):
    expose_psichic_modules(psichic_root)
    from models.net import net

    weight_root = psichic_root / "trained_weights" / "PDBv2020_PSICHIC"
    config = json.loads((weight_root / "config.json").read_text())
    degree = torch.load(weight_root / "degree.pt", map_location="cpu")
    params = config["params"]
    tasks = config["tasks"]
    model = net(
        degree["ligand_deg"], degree["protein_deg"],
        mol_in_channels=params["mol_in_channels"],
        prot_in_channels=params["prot_in_channels"],
        prot_evo_channels=params["prot_evo_channels"],
        hidden_channels=params["hidden_channels"],
        pre_layers=params["pre_layers"], post_layers=params["post_layers"],
        aggregators=params["aggregators"], scalers=params["scalers"],
        total_layer=params["total_layer"], K=params["K"], heads=params["heads"],
        dropout=params["dropout"], dropout_attn_score=params["dropout_attn_score"],
        regression_head=tasks["regression_task"],
        classification_head=tasks["classification_task"],
        multiclassification_head=tasks["mclassification_task"], device=device,
    ).to(device)
    model.load_state_dict(torch.load(weight_root / "model.pt", map_location=device))
    model.eval()
    return model


def extract_one(dataset: str, psichic_root: Path, output_root: Path,
                device: torch.device, batch_size: int) -> None:
    expose_psichic_modules(psichic_root)
    from utils.dataset import ProteinMoleculeDataset
    from utils.ligand_init import ligand_init
    from utils.protein_init import protein_init
    from utils.utils import DataLoader

    context = pd.read_csv(PROJECT_ROOT / "data" / "moleculeace_target_context.csv")
    sequence = context.set_index("dataset").loc[dataset, "sequence"]
    frame = load_moleculeace(dataset)
    # Feature extraction sees structures and split membership, never activities.
    train = frame.loc[frame.split.eq("train"), ["smiles"]].reset_index(drop=True)
    pairs = pd.DataFrame({"Ligand": train.smiles, "Protein": sequence})

    protein_cache = output_root / "protein_graphs" / f"{dataset}.pt"
    protein_cache.parent.mkdir(parents=True, exist_ok=True)
    if protein_cache.exists():
        protein_dict = torch.load(protein_cache, map_location="cpu")
    else:
        protein_dict = protein_init([sequence])
        torch.save(protein_dict, protein_cache)
    ligand_dict = ligand_init(pairs.Ligand.tolist())
    data = ProteinMoleculeDataset(pairs, ligand_dict, protein_dict, device=device)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False,
                        follow_batch=["mol_x", "clique_x", "prot_node_aa"])
    model = load_model(psichic_root, device)
    scores, features = [], []
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            pred, _, _, _, _, _, info = model(
                mol_x=batch.mol_x, mol_x_feat=batch.mol_x_feat,
                bond_x=batch.mol_edge_attr, atom_edge_index=batch.mol_edge_index,
                clique_x=batch.clique_x, clique_edge_index=batch.clique_edge_index,
                atom2clique_index=batch.atom2clique_index,
                residue_x=batch.prot_node_aa, residue_evo_x=batch.prot_node_evo,
                residue_edge_index=batch.prot_edge_index,
                residue_edge_weight=batch.prot_edge_weight,
                mol_batch=batch.mol_x_batch, prot_batch=batch.prot_node_aa_batch,
                clique_batch=batch.clique_x_batch,
            )
            scores.append(pred.reshape(-1).cpu().numpy())
            features.append(info["interaction_fingerprint"].cpu().numpy())
    output_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_root / f"{dataset}.train.npz",
        smiles=train.smiles.to_numpy(dtype=str),
        score=np.concatenate(scores).astype(np.float32),
        feature=np.concatenate(features).astype(np.float32),
        checkpoint=np.array("PDBv2020_PSICHIC/model.pt"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(MOLECULEACE_DATASETS))
    parser.add_argument("--psichic-root", type=Path, default=DEFAULT_PSICHIC)
    parser.add_argument("--output-root", type=Path,
                        default=PROJECT_ROOT / "data" / "moleculeace_psichic_pdb2020")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    unknown = sorted(set(args.datasets).difference(MOLECULEACE_DATASETS))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    device = torch.device(args.device)
    for dataset in args.datasets:
        print(f"Extracting {dataset} on {device}", flush=True)
        extract_one(dataset, args.psichic_root.resolve(), args.output_root.resolve(),
                    device, args.batch_size)


if __name__ == "__main__":
    main()
