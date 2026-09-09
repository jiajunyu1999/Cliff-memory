import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_holo_pockets_cover_every_dataset_with_valid_residues() -> None:
    payload = json.loads((ROOT / "data/moleculeace_holo_pockets.json").read_text())
    assert len(payload["accessions"]) == 29
    assert len(payload["datasets"]) == 30
    assert all(value is not None for value in payload["accessions"].values())
    for pocket in payload["accessions"].values():
        assert len(pocket["residues"]) >= 8
        assert 12 <= pocket["ligand"]["heavy_atoms"] <= 80
        assert all(residue["ligand_distance"] <= 6.0 for residue in pocket["residues"])


def test_holo_pocket_construction_is_activity_label_free() -> None:
    payload = json.loads((ROOT / "data/moleculeace_holo_pockets.json").read_text())
    assert payload["selection"]["uses_activity_labels"] is False
