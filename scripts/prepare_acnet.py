from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACNET_CODE = PROJECT_ROOT / "third_party" / "ACNet" / "ACNet"
RAW = PROJECT_ROOT / "data" / "acnet" / "raw"
OUTPUT = PROJECT_ROOT / "data" / "acnet" / "generated"


def main() -> None:
    sys.path.insert(0, str(ACNET_CODE))
    from ACComponents.ACDataset import DataPreprocess as preprocess
    from ACComponents.ACDataset.DataUtils import Config

    OUTPUT.mkdir(parents=True, exist_ok=True)
    preprocess.OriginDatasetAddrAll = str(RAW / "all_smiles_target.csv")
    preprocess.OriginDatasetAddrPos = str(RAW / "mmp_ac_s_distinct.csv")
    preprocess.OriginDatasetAddrNeg = str(RAW / "mmp_ac_s_neg_distinct.csv")
    preprocess.GeneratedDatasetAddrAll = str(OUTPUT / "MMP_AC.json")
    preprocess.GeneratedDatasetAddrLarge = str(OUTPUT / "MMP_AC_Large.json")
    preprocess.GeneratedDatasetAddrMedium = str(OUTPUT / "MMP_AC_Medium.json")
    preprocess.GeneratedDatasetAddrSmall = str(OUTPUT / "MMP_AC_Small.json")
    preprocess.GeneratedDatasetAddrFew = str(OUTPUT / "MMP_AC_Few.json")
    preprocess.DiscardedDatasetAddr = str(OUTPUT / "MMP_AC_Discarded.json")
    preprocess.GeneratedDatasetAddrMixed = str(OUTPUT / "MMP_AC_Mixed.json")
    preprocess.GeneratedDatasetAddrMixedScreened = str(OUTPUT / "MMP_AC_Mixed_Screened.json")
    preprocess.ACDatasetPreprocess(Config())

    summary = {}
    for path in sorted(OUTPUT.glob("*.json")):
        content = json.loads(path.read_text())
        summary[path.name] = {
            "targets": len(content),
            "pairs": sum(len(items) for items in content.values()),
            "bytes": path.stat().st_size,
        }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

