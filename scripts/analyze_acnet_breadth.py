from __future__ import annotations

"""Paired statistics and restrained breadth figure for the 190 ACNet targets."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from figure_style import COLORS, clean_axis, panel_label, save_figure, set_publication_style


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--model", default="extratrees_ecfp")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/results/acnet_breadth"))
    parser.add_argument("--figure", type=Path, default=Path("paper/figures/fig_acnet_190_breadth.pdf"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    set_publication_style()

    frame = pd.read_csv(args.summary)
    if "model" in frame.columns:
        frame = frame[frame.model.eq(args.model)].copy()
    if frame.target.nunique() != 190:
        raise RuntimeError(f"expected 190 targets for {args.model}, found {frame.target.nunique()}")
    wide = frame.pivot(index="target", columns="feature_mode", values=["auc", "ap", "acc"])
    delta = pd.DataFrame(index=wide.index)
    for metric in ("auc", "ap", "acc"):
        delta[f"{metric}_gain"] = wide[(metric, "endpoint_action")] - wide[(metric, "endpoint")]
    delta.reset_index().to_csv(args.output_dir / "paired_target_deltas.csv", index=False)

    rng = np.random.default_rng(20260904)
    payload = {"targets": int(len(delta)), "comparison": "endpoint_action minus endpoint", "metrics": {}}
    for metric in ("auc", "ap", "acc"):
        values = delta[f"{metric}_gain"].to_numpy(float)
        bootstrap = rng.choice(values, (20000, len(values)), replace=True).mean(axis=1)
        test = wilcoxon(values, alternative="greater", zero_method="zsplit")
        payload["metrics"][metric] = {
            "mean_gain": float(values.mean()), "median_gain": float(np.median(values)),
            "ci95": [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])],
            "wins": int((values > 1e-12).sum()), "ties": int((np.abs(values) <= 1e-12).sum()),
            "losses": int((values < -1e-12).sum()),
            "wilcoxon_p_one_sided": float(test.pvalue),
        }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    fig, axes = plt.subplots(2, 1, figsize=(3.42, 3.55),
                             gridspec_kw={"height_ratios": [1.35, 1]})

    # Primary task-level evidence: an empirical CDF exposes zeros and both tails.
    ap = np.sort(delta.ap_gain.to_numpy(float))
    ecdf = np.arange(1, len(ap)+1) / len(ap)
    axes[0].step(ap, ecdf, where="post", color=COLORS["blue"], lw=1.25)
    axes[0].axvline(0, color="#555555", lw=.75, linestyle=(0, (3, 2)))
    axes[0].set_xlim(min(-.5, ap.min()-.03), max(.85, ap.max()+.03))
    axes[0].set_ylim(0, 1.01)
    axes[0].set_xlabel("Paired average-precision gain")
    axes[0].set_ylabel("Fraction of targets")
    stats = payload["metrics"]["ap"]
    axes[0].text(.98, .05,
                 f"win / tie / loss = {stats['wins']} / {stats['ties']} / {stats['losses']}\n"
                 f"median = {stats['median_gain']:.3f}",
                 transform=axes[0].transAxes, ha="right", va="bottom", fontsize=7.2)
    clean_axis(axes[0])
    panel_label(axes[0], "a", x=-.18, y=1.00)

    # Mean paired effects with bootstrap intervals; an open marker flags a CI crossing zero.
    specs = [("auc", "ROC–AUC"), ("ap", "Average precision"), ("acc", "Accuracy")]
    y = np.arange(3)[::-1]
    for yy, (metric, label) in zip(y, specs):
        item = payload["metrics"][metric]
        mean = item["mean_gain"]; lo, hi = item["ci95"]
        crosses = lo <= 0 <= hi
        color = COLORS["blue"] if metric == "ap" else COLORS["ink"]
        axes[1].errorbar(mean, yy, xerr=[[mean-lo], [hi-mean]], fmt="o", ms=4.2,
                         mfc="white" if crosses else color, mec=color, mew=.9,
                         color=color, ecolor=color, elinewidth=.9, capsize=2.1, zorder=3)
    axes[1].axvline(0, color="#555555", lw=.75, linestyle=(0, (3, 2)))
    axes[1].set_yticks(y, [x[1] for x in specs])
    interval_values = [
        bound
        for metric in ("auc", "ap", "acc")
        for bound in payload["metrics"][metric]["ci95"]
    ]
    margin = max(max(interval_values) - min(interval_values), 0.01) * 0.12
    axes[1].set_xlim(min(0.0, min(interval_values)) - margin,
                     max(interval_values) + margin)
    axes[1].set_xlabel("Mean paired gain (95% bootstrap CI)")
    clean_axis(axes[1], grid="x")
    axes[1].spines["left"].set_visible(False)
    axes[1].tick_params(axis="y", length=0)
    panel_label(axes[1], "b", x=-.18, y=1.00)
    fig.subplots_adjust(left=.23, right=.985, top=.98, bottom=.12, hspace=.52)
    save_figure(fig, args.figure)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
