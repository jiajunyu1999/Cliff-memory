from __future__ import annotations

"""Summarize repeated CV, restricted representations, and coverage sensitivity."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from figure_style import COLORS, clean_axis, save_figure, set_publication_style


BASELINE = "collision_free_binary_count_kernel"
FULL = "responsekernel_final"
LABELS = {
    BASELINE: "Chemistry only",
    "chemistry_profile_dense_only_10to8": "Dense profiles only",
    "responsekernel_assay_only": "Assay profiles only",
    "responsekernel_entity_only": "Entity profiles only",
    "responsekernel_raw_profiles_only": "Sparse profiles only",
    "responsekernel_without_interactions": "No chem.–profile interaction",
}
ABLATION_ORDER = list(LABELS)


def target_bootstrap(delta: pd.DataFrame, metric: str, draws: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    target_means = delta.groupby("dataset")[metric].mean().to_numpy(float)
    values = rng.choice(target_means, size=(draws, len(target_means)), replace=True).mean(axis=1)
    lo, hi = np.quantile(values, [0.025, 0.975])
    return {"mean": float(delta[metric].mean()), "ci95": [float(lo), float(hi)]}


def paired_delta(frame: pd.DataFrame, baseline: str = BASELINE,
                 primary: str = FULL) -> pd.DataFrame:
    keys = ["dataset", "fold", "seed"]
    metrics = ["rmse", "cliff_rmse", "noncliff_rmse"]
    wide = frame[frame.model.isin([baseline, primary])].pivot(
        index=keys, columns="model", values=metrics
    )
    rows = wide.index.to_frame(index=False)
    for metric in metrics:
        rows[metric] = wide[(metric, baseline)].to_numpy() - wide[(metric, primary)].to_numpy()
    return rows


def _cluster_ci(values: pd.Series, groups: pd.Series, draws: int = 12000,
                seed: int = 20260904) -> tuple[float, float]:
    grouped = pd.DataFrame({"value": values, "group": groups}).groupby("group").value.mean()
    rng = np.random.default_rng(seed)
    sampled = rng.choice(grouped.to_numpy(float), size=(draws, len(grouped)), replace=True)
    return tuple(np.quantile(sampled.mean(axis=1), [0.025, 0.975]))


def save_ablation_figure(frame: pd.DataFrame, path: Path) -> None:
    """Target-level distributions for restricted representation comparisons."""
    order = [name for name in ABLATION_ORDER if name in set(frame.model)]
    keys = ["dataset", "fold", "seed"]
    metrics = ["rmse", "cliff_rmse"]
    models = order + [FULL]
    wide = frame[frame.model.isin(models)].pivot(index=keys, columns="model", values=metrics)
    target_values: dict[tuple[str, str], np.ndarray] = {}
    for model in order:
        for metric in metrics:
            penalty = wide[(metric, model)] - wide[(metric, FULL)]
            target_values[(model, metric)] = (
                penalty.groupby(wide.index.get_level_values("dataset")).mean().to_numpy(float)
            )

    fig, ax = plt.subplots(figsize=(7.05, 4.15), facecolor="white")
    ax.set_facecolor("white")
    center = np.arange(len(order))[::-1]
    offsets = {"rmse": 0.18, "cliff_rmse": -0.18}
    colors = {"rmse": "#4DBBD5", "cliff_rmse": "#E64B35"}
    labels = {"rmse": "Overall", "cliff_rmse": "Cliff"}
    rng = np.random.default_rng(20260904)
    legend_handles = []
    for metric in metrics:
        positions = center + offsets[metric]
        values = [target_values[(model, metric)] for model in order]
        box = ax.boxplot(
            values, positions=positions, orientation="horizontal", widths=0.28,
            patch_artist=True, showfliers=False, manage_ticks=False,
            medianprops={"color": "#202124", "linewidth": 1.25},
            whiskerprops={"color": colors[metric], "linewidth": 1.0},
            capprops={"color": colors[metric], "linewidth": 1.0},
            boxprops={"edgecolor": colors[metric], "linewidth": 1.0},
        )
        for patch in box["boxes"]:
            patch.set_facecolor(colors[metric])
            patch.set_alpha(0.30)
        for position, values_i in zip(positions, values):
            jitter = rng.uniform(-0.045, 0.045, size=len(values_i))
            ax.scatter(
                values_i, position + jitter, s=10, color=colors[metric],
                alpha=0.46, edgecolors="none", zorder=3,
            )
        legend_handles.append(
            plt.Line2D(
                [0], [0], marker="s", linestyle="none", markersize=7,
                markerfacecolor=colors[metric], markeredgecolor=colors[metric],
                alpha=0.65, label=labels[metric],
            )
        )

    ax.axvline(0, color="#555555", lw=0.9, linestyle=(0, (4, 3)), zorder=1)
    ax.set_yticks(center, [LABELS[name] for name in order])
    ax.set_xlabel(r"$\Delta$RMSE relative to complete model", fontsize=14)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12, length=0)
    ax.grid(True, axis="x", linestyle="--", alpha=0.25, color="gray", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#202124")
    ax.spines["bottom"].set_linewidth(0.8)
    ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=12, ncol=2)
    fig.subplots_adjust(left=0.40, right=0.985, top=0.96, bottom=0.18)
    save_figure(fig, path)


def save_seed_figure(delta: pd.DataFrame, path: Path) -> None:
    rows = []
    seeds = sorted(delta.seed.unique())
    for metric in ("rmse", "cliff_rmse"):
        for i, seed in enumerate(seeds):
            part = delta[delta.seed.eq(seed)]
            lo, hi = _cluster_ci(part[metric], part.dataset, seed=20260904+i)
            rows.append((metric, f"seed {int(seed)}", part[metric].mean(), lo, hi))
        lo, hi = _cluster_ci(delta[metric], delta.dataset, seed=20261000)
        rows.append((metric, "pooled", delta[metric].mean(), lo, hi))
    plot = pd.DataFrame(rows, columns=["metric", "split", "mean", "lo", "hi"])

    fig, axes = plt.subplots(1, 2, figsize=(3.42, 2.14), sharex=True, sharey=True)
    labels = [f"seed {int(x)}" for x in seeds] + ["pooled"]
    display = [f"Split {i+1}" for i in range(len(seeds))] + ["Pooled"]
    y = np.arange(len(labels))[::-1]
    for ax, metric, title in zip(axes, ("rmse", "cliff_rmse"), ("Overall", "Cliff")):
        part = plot[plot.metric.eq(metric)].set_index("split").loc[labels]
        accent = COLORS["blue"] if metric == "cliff_rmse" else COLORS["ink"]
        for j, (_, row) in enumerate(part.iterrows()):
            pooled = j == len(labels)-1
            color = accent if pooled else "#777777"
            ax.errorbar(row["mean"], y[j],
                        xerr=[[row["mean"]-row["lo"]], [row["hi"]-row["mean"]]],
                        fmt="D" if pooled else "o", ms=4.1 if pooled else 3.0,
                        color=color, ecolor=color, elinewidth=1.0 if pooled else .75,
                        capsize=1.8, markeredgecolor="white", markeredgewidth=.3, zorder=3)
        ax.axvline(0, color="#666666", lw=.7, linestyle=(0, (3, 2)))
        ax.axhline(.5, color="#BDBDBD", lw=.6)
        pooled_row = part.iloc[-1]
        ax.set_title(f"{title}\n{pooled_row['mean']:.3f} [{pooled_row['lo']:.3f}, {pooled_row['hi']:.3f}]",
                     pad=3, fontsize=7.6)
        clean_axis(ax, grid="x")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, display)
    axes[0].get_yticklabels()[-1].set_fontweight("bold")
    axes[1].tick_params(labelleft=False)
    axes[0].set_xlim(-.008, .112)
    fig.supxlabel("Paired RMSE reduction", fontsize=7.4, y=.012)
    fig.subplots_adjust(left=.27, right=.99, top=.88, bottom=.26, wspace=.18)
    save_figure(fig, path)


def coverage_analysis(frame: pd.DataFrame, delta: pd.DataFrame, path: Path,
                      primary: str = FULL) -> dict:
    full = frame[frame.model.eq(primary)].copy()
    coverage = full.groupby(["dataset", "fold", "seed"], as_index=False).agg(
        profile_features=("positive_weight_targets", "max"),
        transfer_coverage=("assay_transfer_valid_coverage", "max"),
    )
    merged = delta.merge(coverage, on=["dataset", "fold", "seed"], how="left")
    target = merged.groupby("dataset", as_index=False).agg(
        profile_features=("profile_features", "mean"),
        transfer_coverage=("transfer_coverage", "mean"),
        cliff_gain=("cliff_rmse", "mean"),
        overall_gain=("rmse", "mean"),
    )
    rho, pvalue = spearmanr(target.profile_features, target.cliff_gain)
    target["coverage_quartile"] = pd.qcut(
        target.profile_features.rank(method="first"), 4,
        labels=["Q1 sparse", "Q2", "Q3", "Q4 dense"]
    )
    cohort = target.groupby("coverage_quartile", observed=True).agg(
        targets=("dataset", "size"), cliff_gain=("cliff_gain", "mean"),
        overall_gain=("overall_gain", "mean")
    ).reset_index()

    rng = np.random.default_rng(20260904)
    bootstrap_rho = []
    for _ in range(12000):
        sample = target.iloc[rng.integers(0, len(target), len(target))]
        value = spearmanr(sample.profile_features, sample.cliff_gain).statistic
        if np.isfinite(value):
            bootstrap_rho.append(value)
    rho_ci = np.quantile(bootstrap_rho, [.025, .975])
    leave_one_out = []
    for omitted in target.dataset:
        reduced = target[target.dataset.ne(omitted)]
        rho_without, p_without = spearmanr(reduced.profile_features, reduced.cliff_gain)
        leave_one_out.append({"omitted_target": str(omitted),
                              "rho": float(rho_without), "p": float(p_without)})
    loo_rhos = np.asarray([item["rho"] for item in leave_one_out])
    largest_gain = str(target.loc[target.cliff_gain.idxmax(), "dataset"])

    fig, ax = plt.subplots(figsize=(3.42, 2.60))
    ax.scatter(target.profile_features, target.cliff_gain, s=22, facecolor="#777777",
               edgecolor="white", linewidth=.35, zorder=3)
    special = target[target.dataset.eq(largest_gain)].iloc[0]
    ax.scatter(special.profile_features, special.cliff_gain, s=30,
               facecolor=COLORS["blue"], edgecolor="white", linewidth=.4, zorder=4)
    ax.annotate(largest_gain.replace("_", " ") + "\n(largest gain)",
                (special.profile_features, special.cliff_gain), xytext=(7, -8),
                textcoords="offset points", ha="left", va="top", fontsize=7.1,
                color=COLORS["blue"], arrowprops=dict(arrowstyle="-", lw=.55,
                color=COLORS["blue"]))
    ax.axhline(0, color="#555555", lw=.7, linestyle=(0, (3, 2)))
    ax.set_xscale("log")
    ax.set_xticks([1, 10, 100], labels=["1", "10", "100"])
    ax.set_xlim(max(.8, target.profile_features.min()*.75), target.profile_features.max()*1.70)
    ax.set_xlabel("Fit-only informative profile coordinates")
    ax.set_ylabel("Paired cliff-RMSE reduction")
    ax.text(.03, .96,
            f"Spearman $\\rho$={rho:.2f} [{rho_ci[0]:.2f}, {rho_ci[1]:.2f}], $p$={pvalue:.2f}\n"
            f"leave-one-target-out $\\rho$ range {loo_rhos.min():.2f} to {loo_rhos.max():.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.2,
            color=COLORS["ink"])
    clean_axis(ax)
    fig.subplots_adjust(left=.20, right=.98, top=.97, bottom=.21)
    save_figure(fig, path)
    return {
        "spearman_rho": float(rho), "spearman_p": float(pvalue),
        "spearman_ci95": [float(x) for x in rho_ci],
        "largest_gain_target": largest_gain,
        "leave_one_target_out_rho_range": [float(loo_rhos.min()), float(loo_rhos.max())],
        "leave_one_target_out": leave_one_out,
        "cohorts": cohort.to_dict(orient="records"),
        "targets": target.to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", type=Path, nargs="+", required=True)
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/results/extended_evidence"))
    parser.add_argument("--figure-dir", type=Path, default=Path("paper/figures"))
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument("--primary-model", default=FULL)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    set_publication_style()

    repeated = pd.concat([pd.read_csv(path) for path in args.cv], ignore_index=True)
    ablation = pd.read_csv(args.ablation)
    delta = paired_delta(repeated, args.baseline, args.primary_model)
    delta.to_csv(args.output_dir / "paired_fold_deltas.csv", index=False)
    payload = {"datasets": int(delta.dataset.nunique()),
               "seeds": sorted(int(x) for x in delta.seed.unique()),
               "paired_folds": int(len(delta)), "metrics": {}}
    for offset, metric in enumerate(["rmse", "cliff_rmse", "noncliff_rmse"]):
        stats = target_bootstrap(delta, metric, args.bootstrap_draws, 20260904 + offset)
        target_delta = delta.groupby("dataset")[metric].mean()
        test = wilcoxon(target_delta, alternative="greater")
        stats.update({"target_wins": int((target_delta > 0).sum()),
                      "target_losses": int((target_delta < 0).sum()),
                      "wilcoxon_statistic": float(test.statistic),
                      "wilcoxon_p_one_sided": float(test.pvalue)})
        payload["metrics"][metric] = stats
    payload["coverage"] = coverage_analysis(
        repeated, delta, args.figure_dir / "fig_coverage_gain.pdf", args.primary_model
    )
    interaction_wide = ablation[ablation.model.isin([FULL, "responsekernel_without_interactions"])].pivot(
        index=["dataset", "fold", "seed"], columns="model", values=["rmse", "cliff_rmse"]
    )
    payload["interaction_ablation"] = {}
    for metric in ("rmse", "cliff_rmse"):
        target_penalty = (
            interaction_wide[(metric, "responsekernel_without_interactions")]
            - interaction_wide[(metric, FULL)]
        ).groupby("dataset").mean()
        test = wilcoxon(target_penalty, alternative="greater", zero_method="zsplit")
        payload["interaction_ablation"][metric] = {
            "mean_penalty": float(target_penalty.mean()),
            "target_wins": int((target_penalty > 0).sum()),
            "targets": int(len(target_penalty)),
            "wilcoxon_p_one_sided": float(test.pvalue),
        }
    save_ablation_figure(ablation, args.figure_dir / "fig_component_ablation.pdf")
    save_seed_figure(delta, args.figure_dir / "fig_repeated_seed_gain.pdf")
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
