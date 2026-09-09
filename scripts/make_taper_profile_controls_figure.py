from __future__ import annotations

"""Draw the TAPER profile-shuffle and leakage-stress controls."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_style import COLORS, clean_axis, panel_label, save_figure, set_publication_style


METRICS = {
    "rmse": "Overall RMSE",
    "cliff_rmse": "Cliff RMSE",
}
FILTERS = (
    ("taper_strict_exclusion", "Strict bio."),
    ("taper_target_name_only", "Name only"),
    ("taper_no_exclusion", "No exclusion"),
)


def target_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(["dataset", "model"])[list(METRICS)].mean().reset_index()


def bootstrap(values: np.ndarray, seed: int, draws: int = 20000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return tuple(float(x) for x in np.quantile(sampled, [0.025, 0.975]))


def draw(frame: pd.DataFrame, output: Path) -> None:
    target = target_summary(frame)
    wide = target.pivot(index="dataset", columns="model", values=list(METRICS))
    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.65), facecolor="white")

    ax = axes[0]
    rng = np.random.default_rng(20260905)
    penalties = []
    for index, metric in enumerate(METRICS):
        values = (
            wide[(metric, "taper_shuffled_profile")]
            - wide[(metric, "taper_strict_exclusion")]
        ).to_numpy(float)
        penalties.append(values)
        jitter = rng.uniform(-0.095, 0.095, len(values))
        ax.scatter(
            np.full(len(values), index) + jitter, values,
            s=11, color=COLORS["blue"] if index == 0 else COLORS["red"],
            alpha=0.36, edgecolor="none", zorder=3,
        )
    boxes = ax.boxplot(
        penalties, positions=np.arange(len(penalties)), widths=0.42,
        patch_artist=True, showfliers=False,
        medianprops={"color": "#202124", "linewidth": 1.25},
        whiskerprops={"color": "#555555", "linewidth": 0.8},
        capprops={"color": "#555555", "linewidth": 0.8},
        boxprops={"edgecolor": "#555555", "linewidth": 0.8},
    )
    for patch, color in zip(boxes["boxes"], (COLORS["blue"], COLORS["red"])):
        patch.set_facecolor(color)
        patch.set_alpha(0.18)
    ax.axhline(0, color="#3A3A3A", linewidth=0.85, linestyle=(0, (3, 2)), zorder=1)
    ax.set_xticks(range(len(METRICS)), list(METRICS.values()))
    ax.set_ylabel(r"Shuffled $-$ true profile RMSE")
    clean_axis(ax, grid="y")
    panel_label(ax, "a", x=-0.13, y=1.04)

    ax = axes[1]
    x = np.arange(len(FILTERS))
    styles = {"rmse": (COLORS["blue"], "o"), "cliff_rmse": (COLORS["red"], "s")}
    seed = 20261000
    for metric, label in METRICS.items():
        means, low, high = [], [], []
        for model, _ in FILTERS:
            values = target[target.model.eq(model)][metric].to_numpy(float)
            lo, hi = bootstrap(values, seed)
            seed += 1
            means.append(float(values.mean())); low.append(lo); high.append(hi)
        means = np.asarray(means)
        error = np.vstack([means - np.asarray(low), np.asarray(high) - means])
        color, marker = styles[metric]
        ax.errorbar(
            x[:2], means[:2], yerr=error[:, :2], color=color, marker=marker, markersize=4.8,
            markeredgecolor="white", markeredgewidth=0.65, linewidth=1.5,
            capsize=2.4, label=label, zorder=4,
        )
        ax.errorbar(
            x[2], means[2], yerr=error[:, 2:3], color=color, marker=marker, markersize=4.8,
            markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.1,
            linewidth=0, capsize=2.4, zorder=4,
        )
    ax.set_xticks(x, [label for _, label in FILTERS], rotation=0, ha="center")
    ax.set_ylabel("Target-macro RMSE")
    ax.legend(frameon=False, loc="upper right", fontsize=7.0, handlelength=1.5)
    clean_axis(ax, grid="y")
    panel_label(ax, "b", x=-0.13, y=1.04)
    ax.annotate(
        "leakage\ncontrol",
        xy=(2, 0.199), xytext=(1.56, 0.285),
        arrowprops={"arrowstyle": "-", "color": COLORS["muted"], "linewidth": 0.65},
        ha="left", va="center", fontsize=6.8, color=COLORS["muted"],
    )

    fig.subplots_adjust(left=0.105, right=0.985, top=0.94, bottom=0.21, wspace=0.36)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=420, bbox_inches="tight", facecolor="white")
    save_figure(fig, output, dpi=360)


def write_table(frame: pd.DataFrame, output: Path) -> None:
    target = target_summary(frame)
    display = (
        ("chemistry_only", "Chemistry only"),
        ("taper_shuffled_profile", "TAPER with shuffled profiles"),
        ("taper_strict_exclusion", r"TAPER, strict exclusion"),
        ("taper_target_name_only", r"TAPER, target-name exclusion"),
        ("taper_no_exclusion", r"TAPER, no exclusion$^{\dagger}$"),
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Response-profile permutation and leakage stress controls on 30 MoleculeACE targets under five-fold train-only evaluation. Values are target-macro RMSE. $^{\dagger}$The no-exclusion condition deliberately re-injects the benchmark endpoint and is an invalid, contaminated upper bound.}",
        r"\label{tab:profile-controls}",
        r"\footnotesize",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Condition & Overall & Cliff & Non-cliff\\",
        r"\midrule",
    ]
    for model, label in display:
        part = target[target.model.eq(model)]
        values = part[["rmse", "cliff_rmse"]].mean()
        noncliff = frame.groupby(["dataset", "model"]).noncliff_rmse.mean().reset_index()
        noncliff_value = noncliff[noncliff.model.eq(model)].noncliff_rmse.mean()
        lines.append(
            f"{label} & {values.rmse:.4f} & {values.cliff_rmse:.4f} & {noncliff_value:.4f}\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary", type=Path,
        default=Path("outputs/taper_profile_controls_cv5/summary.cv5.csv"),
    )
    parser.add_argument(
        "--figure", type=Path,
        default=Path("paper/figures/fig_profile_controls.pdf"),
    )
    parser.add_argument(
        "--table", type=Path,
        default=Path("paper/tables/profile_controls.tex"),
    )
    args = parser.parse_args()
    frame = pd.read_csv(args.summary)
    expected = set(model for model, _ in (
        ("chemistry_only", ""), ("taper_shuffled_profile", ""),
        ("taper_strict_exclusion", ""), ("taper_target_name_only", ""),
        ("taper_no_exclusion", ""),
    ))
    if set(frame.model.unique()) != expected or frame.dataset.nunique() != 30:
        raise RuntimeError("profile control figure requires five models on all 30 targets")
    draw(frame, args.figure)
    write_table(frame, args.table)
    print(json.dumps({
        "figure": str(args.figure), "table": str(args.table),
        "targets": int(frame.dataset.nunique()), "rows": int(len(frame)),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
