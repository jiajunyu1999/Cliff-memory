from __future__ import annotations

"""Plot one-parameter-at-a-time sensitivity for every TAPER component."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_style import COLORS, clean_axis, panel_label, save_figure, set_publication_style


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "moleculeace_hyperparameter_sensitivity_parts"
OUT = ROOT / "reports" / "results" / "hyperparameter_sensitivity"

PANELS = [
    {
        "key": "retrieval_budget", "prefix": "sensitivity_retrieval_budget_",
        "values": [1, 2, 4, 6, 8, 10], "selected": 10,
        "xlabel": r"Selected profile coordinates, $q_{\rm resp}$",
    },
    {
        "key": "bandwidth", "prefix": "sensitivity_bandwidth_",
        "values": [0.25, 0.5, 1.0, 2.0, 4.0], "selected": 1.0,
        "xlabel": r"Dense-view bandwidth, $\tau/\tau_{\rm med}$",
    },
    {
        "key": "memory_weight", "prefix": "sensitivity_memory_weight_",
        "values": [0.2, 0.35, 0.444, 0.6, 0.8], "selected": 0.444,
        "xlabel": r"Profile mixture weight, $\lambda_{\rm prof}$",
    },
    {
        "key": "svr_c", "prefix": "sensitivity_svr_c_",
        "values": [0.3, 1.0, 3.0, 10.0, 30.0, 100.0], "selected": 10.0,
        "xlabel": r"SVR penalty, $C$",
    },
]


def model_name(prefix: str, value: float | int) -> str:
    if prefix.endswith("memory_weight_"):
        token = f"{float(value):.3f}".rstrip("0").rstrip(".")
    else:
        token = str(value)
    token = token.replace(".", "p")
    return prefix + token


def bootstrap(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(20000, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def aggregate() -> pd.DataFrame:
    paths = sorted(SOURCE.glob("*/summary.cv5.csv"))
    if len(paths) != 30:
        raise RuntimeError(f"expected 30 target result files, found {len(paths)}")
    raw = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    if raw.dataset.nunique() != 30 or set(raw.fold.unique()) != set(range(5)):
        raise RuntimeError("sensitivity audit requires 30 targets and five folds")
    target = raw.groupby(["dataset", "model"])[["rmse", "cliff_rmse"]].mean().reset_index()
    rows: list[dict] = []
    seed = 20260905
    for panel in PANELS:
        for value in panel["values"]:
            name = model_name(panel["prefix"], value)
            part = target[target.model.eq(name)]
            if len(part) != 30:
                raise RuntimeError(f"{name} has {len(part)} targets, expected 30")
            for metric in ("rmse", "cliff_rmse"):
                values = part[metric].to_numpy(float)
                lo, hi = bootstrap(values, seed)
                seed += 1
                rows.append({
                    "module": panel["key"], "parameter": float(value),
                    "model": name, "metric": metric, "mean": float(values.mean()),
                    "ci_low": float(lo), "ci_high": float(hi), "targets": len(values),
                })
    return pd.DataFrame(rows)


def draw(summary: pd.DataFrame) -> None:
    set_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(6.85, 4.65))
    styles = {
        "rmse": ("Overall RMSE", COLORS["blue"], "o"),
        "cliff_rmse": ("Cliff RMSE", COLORS["red"], "s"),
    }
    for index, (ax, panel) in enumerate(zip(axes.flat, PANELS, strict=True)):
        part = summary[summary.module.eq(panel["key"])]
        x = np.asarray(panel["values"], dtype=float)
        categorical = panel["key"] in {"bandwidth", "memory_weight"}
        x_plot = np.arange(len(x), dtype=float) if categorical else x
        for metric, (label, color, marker) in styles.items():
            curve = part[part.metric.eq(metric)].set_index("parameter").loc[x]
            mean = curve["mean"].to_numpy(float)
            lo = curve["ci_low"].to_numpy(float)
            hi = curve["ci_high"].to_numpy(float)
            ax.fill_between(x_plot, lo, hi, color=color, alpha=0.10, linewidth=0, zorder=1)
            ax.plot(x_plot, mean, color=color, marker=marker, markersize=4.2,
                    markeredgecolor="white", markeredgewidth=0.6, label=label, zorder=3)
            chosen = int(np.argmin(np.abs(x - panel["selected"])))
            ax.scatter(x_plot[chosen], mean[chosen], s=47, facecolors="none",
                       edgecolors=COLORS["orange"], linewidths=1.2, zorder=5)
        chosen = int(np.argmin(np.abs(x - panel["selected"])))
        selected_x = x_plot[chosen]
        ax.axvline(selected_x, color=COLORS["orange"], lw=0.8,
                   ls=(0, (2, 2)), alpha=0.72, zorder=0)
        if panel["key"] == "svr_c":
            ax.set_xscale("log")
            ax.set_xticks(x, ["0.3", "1", "3", "10", "30", "100"])
        elif categorical:
            labels = [f"{value:g}" for value in x]
            ax.set_xticks(x_plot, labels)
        else:
            ax.set_xticks(x)
        ax.set_xlabel(panel["xlabel"])
        ax.set_ylabel("Target-macro RMSE" if index % 2 == 0 else "")
        clean_axis(ax, grid="y")
        panel_label(ax, chr(ord("a") + index), x=-0.15, y=1.03)
        ax.margins(x=0.06, y=0.12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 1.01),
               ncol=2, frameon=False, handlelength=2.0, columnspacing=1.6)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.11, top=0.92,
                        wspace=0.22, hspace=0.40)
    fig.savefig(OUT / "fig_sensitivity_preview.png", dpi=320,
                bbox_inches="tight", facecolor="white")
    save_figure(fig, ROOT / "paper" / "figures" / "fig_sensitivity.pdf")


def main() -> None:
    summary = aggregate()
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "summary.csv", index=False)
    payload = {
        "protocol": "official-train-only, five-fold CV, seed 42",
        "aggregation": "fold mean within target, then macro mean over 30 targets",
        "uncertainty": "95% bootstrap interval over targets",
        "design": "one parameter varied at a time with all model components retained",
    }
    (OUT / "audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    draw(summary)


if __name__ == "__main__":
    main()
