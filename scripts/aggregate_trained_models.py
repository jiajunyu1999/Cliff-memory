from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-targets", type=int, default=30)
    args = parser.parse_args()
    models = sorted(args.parts_dir.glob("*/*.svr.pkl"))
    kernels = sorted(args.parts_dir.glob("*/*.train_kernel.npz"))
    if len(models) != args.expected_targets or len(kernels) != args.expected_targets:
        raise RuntimeError(f"incomplete artifacts: models={len(models)}, kernels={len(kernels)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for source in models + kernels:
        shutil.copy2(source, args.output_dir / source.name)
    metrics = pd.concat(
        [pd.read_csv(path) for path in sorted(args.parts_dir.glob("*/train_metrics.csv"))],
        ignore_index=True,
    ).sort_values("dataset")
    metrics.to_csv(args.output_dir / "train_metrics.csv", index=False)
    recipes = [json.loads(path.read_text()) for path in args.parts_dir.glob("*/recipe.json")]
    recipe = recipes[0]
    recipe.update({"datasets": int(metrics.dataset.nunique()),
                   "artifacts": "one SVR and train Gram matrix per target"})
    (args.output_dir / "recipe.json").write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"datasets": int(metrics.dataset.nunique()),
                      "output_dir": str(args.output_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
