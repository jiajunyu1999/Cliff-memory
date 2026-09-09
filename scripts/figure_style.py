"""Shared visual language for publication figures.

The palette is colour-blind safe, typography is sized for a two-column paper,
and vector text remains editable in exported PDFs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


COLORS = {
    "ink": "#222222",
    "muted": "#666666",
    "grid": "#D9D9D9",
    "paper": "#FFFFFF",
    "panel": "#F7F7F7",
    "navy": "#2F5597",
    "blue": "#2F5597",
    "blue_light": "#E7ECF4",
    "teal": "#2F5597",
    "teal_light": "#E7ECF4",
    "orange": "#D97706",
    "orange_light": "#F7E8CF",
    "red": "#B4443C",
    "red_light": "#F3E3E1",
    "purple": "#6B5B8E",
    "grey": "#9A9A9A",
    "grey_light": "#EEEEEE",
}


def set_publication_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.5,
        "axes.titleweight": "semibold",
        "axes.labelcolor": COLORS["ink"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "xtick.labelsize": 7.6,
        "ytick.labelsize": 7.6,
        "xtick.color": COLORS["muted"],
        "ytick.color": COLORS["muted"],
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.fontsize": 7.6,
        "legend.frameon": False,
        "figure.facecolor": COLORS["paper"],
        "axes.facecolor": COLORS["paper"],
        "savefig.facecolor": COLORS["paper"],
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.035,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "lines.linewidth": 1.35,
        "patch.linewidth": 0.8,
    })


def clean_axis(ax: plt.Axes, grid: str | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=COLORS["grid"], linewidth=0.55, alpha=0.65)
        ax.set_axisbelow(True)
    ax.tick_params(length=3, pad=2)


def panel_label(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.05) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10.5, fontweight="bold", color=COLORS["ink"])


def save_figure(fig: plt.Figure, path: Path, *, dpi: int = 320) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
