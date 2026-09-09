from __future__ import annotations

"""Fail loudly unless every primary reported baseline has local result evidence."""

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CLASSICAL = {
    "linear_ecfp", "svm_ecfp", "hist_gradient_boosting_ecfp", "mlp_ecfp",
    "extratrees_ecfp", "random_forest_ecfp",
}
GRAPH = {"chemprop", "gine_residual", "gine_nodenorm", "gine_pairnorm", "graphcliff"}


def csv_report(
    path: Path, expected_models: set[str], expected_targets: int, expected_folds: int,
    collection: str | None = None,
) -> dict:
    if not path.exists():
        return {"path": str(path), "ok": False, "reason": "missing"}
    frame = pd.read_csv(path)
    if collection is not None:
        if "collection" not in frame:
            return {"path": str(path), "ok": False, "reason": "collection column missing"}
        frame = frame[frame.collection.eq(collection)].copy()
    present = set(frame.model.astype(str)) if "model" in frame else set()
    targets = frame.dataset.nunique() if "dataset" in frame else 0
    observed = {
        name: int((frame.model == name).sum())
        for name in sorted(present.intersection(expected_models))
    }
    expected_rows_per_model = expected_targets * expected_folds
    ok = (
        expected_models.issubset(present)
        and targets == expected_targets
        and all(observed.get(name, 0) == expected_rows_per_model for name in expected_models)
        and frame[["rmse", "cliff_rmse", "noncliff_rmse"]].notna().all().all()
    )
    return {
        "path": str(path), "collection": collection, "ok": bool(ok), "targets": int(targets),
        "expected_targets": expected_targets, "rows": int(len(frame)),
        "models": sorted(present), "expected_models": sorted(expected_models),
        "rows_per_model": observed, "expected_rows_per_model": expected_rows_per_model,
    }


def mtpnet_report(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path), "ok": False, "reason": "missing"}
    frame = pd.read_csv(path)
    metrics = {"rmse", "cliff_rmse", "noncliff_rmse"}
    ok = len(frame) == 30 and frame.dataset.nunique() == 30 and metrics.issubset(frame) and frame[list(metrics)].notna().all().all()
    return {"path": str(path), "ok": bool(ok), "rows": int(len(frame)), "targets": int(frame.dataset.nunique())}


def acnet_report(paths: list[Path], expected_models: set[str]) -> dict:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        return {"paths": [str(path) for path in paths], "ok": False,
                "reason": "missing", "missing": missing}
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    models = set(frame.model.astype(str))
    counts = frame.groupby("model").target.nunique().to_dict()
    finite = frame[["auc", "ap"]].notna().all().all()
    ok = expected_models.issubset(models) and all(counts.get(model) == 190 for model in expected_models) and finite
    return {"paths": [str(path) for path in paths], "ok": bool(ok), "rows": int(len(frame)),
            "models": sorted(models), "expected_models": sorted(expected_models),
            "targets_per_model": {str(k): int(v) for k, v in counts.items()}}


def dablander_report(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path), "ok": False, "reason": "missing"}
    frame = pd.read_csv(path)
    expected = {"svm", "calibrated_mmp"}
    counts = frame.groupby(["dataset", "model"]).size()
    ok = (set(frame.model) == expected and frame.dataset.nunique() == 3
          and len(frame) == 12 and (counts == 2).all()
          and frame[["rmse", "cliff_rmse", "noncliff_rmse"]].notna().all().all())
    return {"path": str(path), "ok": bool(ok), "rows": int(len(frame)),
            "datasets": int(frame.dataset.nunique()), "models": sorted(set(frame.model))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "outputs" / "baseline_runs")
    parser.add_argument("--mtpnet-dir", type=Path, default=ROOT / "outputs" / "baseline_smoke" / "mtpnet_epoch50")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "moleculeace_classical_initial": csv_report(
            ROOT / "outputs" / "cross_dataset_regression_baselines" / "summary.csv",
            {"linear_ecfp", "extratrees_ecfp", "random_forest_ecfp"}, 30, 15,
            collection="MoleculeACE",
        ),
        "moleculeace_graph": csv_report(args.runs_dir / "moleculeace_graph_full.csv", GRAPH, 30, 5),
        "external_graph": csv_report(args.runs_dir / "external_graph_full.csv", GRAPH, 5, 5),
        "moleculeace_classical_recent": csv_report(args.runs_dir / "moleculeace_classical_recent.csv", {"svm_ecfp", "hist_gradient_boosting_ecfp", "mlp_ecfp"}, 30, 15),
        "mtpnet": mtpnet_report(args.mtpnet_dir / "summary.mtpnet_epoch50.csv"),
        "acnet_classical": acnet_report(
            [ROOT / "outputs/baseline_smoke/acnet_classical_all.csv"], CLASSICAL),
        "acnet_graph": acnet_report([
            ROOT / "outputs/cross_dataset_graph_baselines/acnet_gpu0.csv",
            ROOT / "outputs/cross_dataset_graph_baselines/acnet_gpu1.csv",
            args.runs_dir / "acnet_chemprop_full.csv",
        ], GRAPH),
        "acnet_edit_model": acnet_report(
            [ROOT / "outputs/acnet_all190_action_sgd_v1/summary.csv"], {"sgd_logreg"}),
        "dablander_external": dablander_report(
            ROOT / "outputs/dablander_external_v2_local/summary.csv"),
    }
    report["ok"] = all(value["ok"] for value in report.values())
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    if not report["ok"]:
        raise SystemExit("baseline audit incomplete")


if __name__ == "__main__":
    main()
