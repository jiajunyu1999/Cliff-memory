from __future__ import annotations

"""Build a reproducible target-context table from primary ChEMBL/UniProt APIs.

The MoleculeACE dataset names identify ChEMBL targets, but the benchmark files
contain no protein sequence.  This script resolves each identifier once and
stores the exact accessions and canonical sequences used by the model.
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd


CHEMBL_URL = "https://www.ebi.ac.uk/chembl/api/data/target/{target}.json"
UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/{accession}.fasta"


def _read_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "molecule-cliffs-research/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _read_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "molecule-cliffs-research/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def _canonical_sequence(accession: str) -> str:
    fasta = _read_text(UNIPROT_URL.format(accession=accession))
    lines = [line.strip() for line in fasta.splitlines() if line and not line.startswith(">")]
    sequence = "".join(lines)
    if not sequence:
        raise RuntimeError(f"UniProt returned an empty sequence for {accession}")
    return sequence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("third_party/MoleculeACE/MoleculeACE/Data/benchmark_data/metadata/datasets.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/moleculeace_target_context.csv"))
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata)
    rows: list[dict] = []
    sequence_cache: dict[str, str] = {}
    for record in metadata.to_dict("records"):
        chembl_id = str(record["ChEMBL ID"])
        payload = _read_json(CHEMBL_URL.format(target=chembl_id))
        components = [
            component
            for component in payload.get("target_components", [])
            if component.get("component_type") == "PROTEIN" and component.get("accession")
        ]
        if len(components) != 1:
            raise RuntimeError(
                f"Expected one protein component for {chembl_id}, found {len(components)}"
            )
        component = components[0]
        accession = str(component["accession"])
        if accession not in sequence_cache:
            sequence_cache[accession] = _canonical_sequence(accession)
            time.sleep(0.05)
        sequence = sequence_cache[accession]
        rows.append(
            {
                "dataset": str(record["Dataset"]),
                "chembl_id": chembl_id,
                "target_name": str(payload.get("pref_name", record["Target name"])),
                "organism": str(payload.get("organism", "")),
                "receptor_class": str(record["Receptor Class"]),
                "measurement_type": str(record["Type"]),
                "uniprot_accession": accession,
                "sequence": sequence,
                "sequence_length": len(sequence),
            }
        )
        print(json.dumps({"dataset": record["Dataset"], "accession": accession, "length": len(sequence)}))

    output = pd.DataFrame(rows)
    if output.dataset.duplicated().any() or len(output) != len(metadata):
        raise RuntimeError("Target-context table does not map one-to-one to MoleculeACE datasets")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    manifest = {
        "rows": len(output),
        "unique_chembl_targets": int(output.chembl_id.nunique()),
        "unique_uniprot_accessions": int(output.uniprot_accession.nunique()),
        "chembl_endpoint": CHEMBL_URL,
        "uniprot_endpoint": UNIPROT_URL,
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
