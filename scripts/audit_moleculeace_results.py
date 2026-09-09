from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


RESULTS = {
    "svm_official": "outputs/moleculeace_official_baselines/summary.svm_official.csv",
    "tanimoto_svm": "outputs/moleculeace_tanimoto_svm_v1/summary.tanimoto_svm.csv",
    "mmp_memory": "outputs/moleculeace_mmp_v1/summary.mmp_memory.csv",
    "calibrated_mmp": "outputs/moleculeace_calibrated_mmp_v1/summary.calibrated_mmp.csv",
    "multitarget_action": "outputs/moleculeace_multitarget_action_v1/summary.multitarget_action.csv",
    "multitarget_action_balanced": "outputs/moleculeace_multitarget_action_balanced_v2/summary.multitarget_action.csv",
    "target_action_chemberta_esm": "outputs/moleculeace_target_action_v1/summary.target_action.csv",
    "target_action_unimol_esm": "outputs/moleculeace_target_action_unimol_v1/summary.target_action.csv",
    "multiscale_tanimoto": "outputs/moleculeace_multiscale_kernel_v1/summary.multiscale_tanimoto.csv",
    "local_chemistry_kernel": "outputs/moleculeace_local_chemistry_kernel_v1/summary.local_chemistry_kernel.csv",
    "action_salience_kernel": "outputs/moleculeace_action_salience_kernel_v1/summary.action_salience_kernel.csv",
    "deshrunk_local_chemistry_kernel": "outputs/moleculeace_deshrunk_kernel_v1/summary.deshrunk_kernel.csv",
    "conditional_mmp": "outputs/moleculeace_conditional_mmp_v1/summary.conditional_mmp.csv",
    "mtpnet_released_epoch50": "outputs/moleculeace_mtpnet_epoch50_audit/summary.mtpnet_epoch50.csv",
    "pocket_action_kernel": "outputs/moleculeace_pocket_action_kernel_v1/summary.pocket_action_kernel.csv",
    "fixed_cliffloss_kernel": "outputs/moleculeace_fixed_cliffloss_kernel_v1/summary.fixed_cliffloss_kernel.csv",
    "binary_count_local_kernel": "outputs/moleculeace_count_kernel_v1/summary.binary_count_local_kernel.csv",
    "collision_free_binary_count_kernel": "outputs/moleculeace_collision_free_kernel_v1/summary.collision_free_kernel.csv",
    "action_ensemble": "outputs/action_ensemble_v1/summary.action_ensemble.csv",
    "guarded_action_ensemble": "outputs/action_ensemble_guarded_v3/summary.action_ensemble.csv",
}

NONCOMPLIANT = {
    "action_ensemble", "guarded_action_ensemble", "calibrated_mmp", "mmp_memory",
    "mtpnet_released_epoch50",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("reports/results"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for model, path in RESULTS.items():
        frame = pd.read_csv(path)
        frame["model"] = model
        frame["compliant_single_model"] = model not in NONCOMPLIANT
        frames.append(frame)
    all_results = pd.concat(frames, ignore_index=True)
    metrics = ["rmse", "cliff_rmse", "noncliff_rmse"]
    macro = all_results.groupby(["model", "compliant_single_model"], sort=False)[metrics].mean().reset_index()
    macro.to_csv(args.output_dir / "macro_metrics.csv", index=False)

    baseline = all_results[all_results.model.eq("svm_official")].set_index("dataset")
    comparisons = []
    for model, frame in all_results.groupby("model", sort=False):
        current = frame.set_index("dataset")
        record: dict[str, object] = {"model": model}
        for metric in metrics:
            gain = baseline[metric] - current[metric]
            record[f"{metric}_mean_gain_vs_svm"] = float(gain.mean())
            record[f"{metric}_wins_vs_svm"] = int((gain > 1e-12).sum())
            record[f"{metric}_ties_vs_svm"] = int((gain.abs() <= 1e-12).sum())
            record[f"{metric}_losses_vs_svm"] = int((gain < -1e-12).sum())
        comparisons.append(record)
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(args.output_dir / "wins_vs_svm.csv", index=False)

    best_name = "collision_free_binary_count_kernel"
    best = all_results[all_results.model.eq(best_name)].set_index("dataset")
    per_target = baseline[metrics].add_suffix("_svm").join(
        best[metrics].add_suffix(f"_{best_name}")
    )
    for metric in metrics:
        per_target[f"{metric}_gain"] = (
            per_target[f"{metric}_svm"]
            - per_target[f"{metric}_{best_name}"]
        )
    per_target.reset_index().to_csv(args.output_dir / f"{best_name}_vs_svm.csv", index=False)

    report = {
        "macro": macro.set_index("model").to_dict(orient="index"),
        "hard_gate": {
            "baseline": "svm_official",
            "required_relative_gain": {"rmse": 0.10, "cliff_rmse": 0.20},
            "required_max_rmse": float(baseline.rmse.mean() * 0.90),
            "required_max_cliff_rmse": float(baseline.cliff_rmse.mean() * 0.80),
            "best_compliant_observed": best_name,
            "best_compliant_passes": {
                "rmse": bool(best.rmse.mean() <= baseline.rmse.mean() * 0.90),
                "cliff_rmse": bool(best.cliff_rmse.mean() <= baseline.cliff_rmse.mean() * 0.80),
            },
        },
        "published_graphcliff": {
            "rmse": 0.665,
            "cliff_rmse": 0.757,
            "noncliff_rmse": 0.619,
        },
        "comparison_vs_svm": comparison.set_index("model").to_dict(orient="index"),
        "claim_boundary": (
            "The collision-free binary+count local kernel is the strongest compliant single model. "
            "It was frozen on one internal validation comparison before one official-test run, "
            "but it does not reach the requested 10% overall "
            "and 20% cliff-RMSE gains."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
