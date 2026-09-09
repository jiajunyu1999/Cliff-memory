from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pandas as pd

from molcliff.data import MOLECULEACE_DATASETS, MOLECULEACE_ROOT, PROJECT_ROOT


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def main() -> None:
    assets = []
    for name in MOLECULEACE_DATASETS:
        path = MOLECULEACE_ROOT / f"{name}.csv"
        frame = pd.read_csv(path)
        assets.append(
            {
                "collection": "MoleculeACE",
                "file": str(path.relative_to(PROJECT_ROOT)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "rows": len(frame),
                "train": int(frame.split.eq("train").sum()),
                "test": int(frame.split.eq("test").sum()),
                "test_cliff": int(frame.loc[frame.split.eq("test"), "cliff_mol"].sum()),
            }
        )
    acnet_root = PROJECT_ROOT / "data" / "acnet" / "raw"
    for path in sorted(acnet_root.glob("*")):
        assets.append(
            {
                "collection": "ACNet",
                "file": str(path.relative_to(PROJECT_ROOT)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    generated_root = PROJECT_ROOT / "data" / "acnet" / "generated"
    for path in sorted(generated_root.glob("*.json")):
        assets.append(
            {
                "collection": "ACNet-generated",
                "file": str(path.relative_to(PROJECT_ROOT)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    dablander_root = (
        PROJECT_ROOT / "third_party" / "QSAR-activity-cliff-experiments" / "data"
    )
    for path in sorted(dablander_root.glob("*/*.csv")):
        assets.append(
            {
                "collection": "QSAR-activity-cliff-experiments",
                "file": str(path.relative_to(PROJECT_ROOT)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "rows": len(pd.read_csv(path)),
            }
        )
    pocket_manifest = PROJECT_ROOT / "data" / "moleculeace_holo_pockets.json"
    if pocket_manifest.exists():
        payload = json.loads(pocket_manifest.read_text())
        assets.append({
            "collection": "MoleculeACE-holo-pockets",
            "file": str(pocket_manifest.relative_to(PROJECT_ROOT)),
            "sha256": sha256(pocket_manifest),
            "bytes": pocket_manifest.stat().st_size,
            "selected_structures": len(payload["accessions"]),
        })
        selected = sorted({value["pdb_id"] for value in payload["accessions"].values()})
        for pdb_id in selected:
            path = PROJECT_ROOT / "data" / "holo_pdb" / f"{pdb_id}.pdb"
            assets.append({
                "collection": "RCSB-selected-holo-PDB",
                "file": str(path.relative_to(PROJECT_ROOT)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "pdb_id": pdb_id,
            })
    revisions = {}
    for path in sorted((PROJECT_ROOT / "third_party").iterdir()):
        if (path / ".git").exists():
            revisions[path.name] = git_revision(path)
    manifest = {"revisions": revisions, "assets": assets}
    out = PROJECT_ROOT / "data" / "asset_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(out), "assets": len(assets), "revisions": revisions}, indent=2))


if __name__ == "__main__":
    main()
