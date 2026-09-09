from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

from figure_style import COLORS, save_figure, set_publication_style
from molecule_drawing import render_aligned_pair_grid


def arrow(ax, start, end, color="#333333", lw=0.9):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8,
                                 lw=lw, color=color, shrinkA=1, shrinkB=1))


def square_box(ax, xy, width, height, text, edge="#555555", face="white",
               fontsize=7.4, weight="normal", linestyle="-"):
    ax.add_patch(Rectangle(xy, width, height, facecolor=face, edgecolor=edge,
                           linewidth=.85, linestyle=linestyle))
    ax.text(xy[0]+width/2, xy[1]+height/2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=COLORS["ink"], linespacing=1.15)


def matrix(ax, x, y, width, height, color, values, missing=None):
    values = np.asarray(values, dtype=float)
    rows, cols = values.shape
    missing = np.zeros_like(values, dtype=bool) if missing is None else np.asarray(missing)
    for r in range(rows):
        for c in range(cols):
            alpha = .10 + .72*values[r, c]
            face = "white" if missing[r, c] else color
            ax.add_patch(Rectangle((x+c*width/cols, y+(rows-r-1)*height/rows),
                                   width/cols, height/rows, facecolor=face,
                                   alpha=1 if missing[r, c] else alpha,
                                   edgecolor="#B8B8B8", linewidth=.35))
    ax.add_patch(Rectangle((x, y), width, height, fill=False, edgecolor="#555555", linewidth=.7))


def panel_heading(ax, x, label, title):
    ax.text(x, .94, label, ha="left", va="top", fontsize=10, fontweight="bold")
    ax.text(x+.025, .94, title, ha="left", va="top", fontsize=9.0, fontweight="bold")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("paper/figures/fig_method_schematic.pdf"))
    args = parser.parse_args()
    set_publication_style()
    fig, ax = plt.subplots(figsize=(6.55, 2.55))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    panel_heading(ax, .015, "a", "Local cliff")
    panel_heading(ax, .260, "b", "Target-excluded profile")
    panel_heading(ax, .755, "c", "TAPER predictor")
    ax.plot([.238, .238], [.08, .92], color="#C8C8C8", lw=.7)
    ax.plot([.738, .738], [.08, .92], color="#C8C8C8", lw=.7)

    # (a) A real, localized MoleculeACE cliff transformation.
    left = "NC(=O)c1c(I)cccc1F"
    right = "NC(=O)c1c(F)cccc1F"
    image = render_aligned_pair_grid(left, right, panel_size=(520, 330))
    iax = ax.inset_axes([.018, .42, .202, .31])
    iax.imshow(image); iax.axis("off")
    arrow(ax, (.116, .55), (.135, .55), COLORS["ink"], 1.0)
    ax.text(.126, .65, "I → F", ha="center", va="center", fontsize=7.1,
            color=COLORS["orange"], fontweight="bold")
    ax.text(.067, .33, "$p$Activity 7.28", ha="center", va="center", fontsize=7.4)
    ax.text(.185, .33, "$p$Activity 5.30", ha="center", va="center", fontsize=7.4)
    ax.text(.126, .20, "local moiety shown", ha="center",
            fontsize=7.2, color=COLORS["muted"])
    ax.text(.126, .115, "$|\\Delta y|=1.98$", ha="center", fontsize=8.1,
            color=COLORS["ink"], fontweight="bold")

    # (b) Fit-fold-only transformations are indicated once by a dashed enclosure.
    ax.add_patch(Rectangle((.270, .13), .450, .685, facecolor="none",
                           edgecolor="#777777", linewidth=.7, linestyle=(0, (3, 2))))
    ax.text(.712, .795, "fit-fold transformations", ha="right", va="top", fontsize=7.2,
            color=COLORS["muted"])
    square_box(ax, (.285, .56), .100, .125, "local\nenvironments", edge=COLORS["blue"], fontsize=7.0)
    # binary/count feature vectors
    for k, yy in enumerate((.590, .615, .640)):
        for j in range(7):
            ax.add_patch(Rectangle((.410+j*.012, yy), .009, .014,
                                   facecolor=COLORS["blue"] if (j+k)%3 else "white",
                                   edgecolor=COLORS["blue"], linewidth=.35))
    ax.text(.450, .555, "binary / count", ha="center", va="top", fontsize=7.2)
    matrix(ax, .520, .555, .070, .115, COLORS["blue"],
           [[1,.6,.2],[.6,1,.4],[.2,.4,1]])
    ax.text(.555, .535, "Kchem", ha="center", va="top", fontsize=7.3,
            fontstyle="italic")
    arrow(ax, (.385,.622), (.407,.622), COLORS["ink"])
    arrow(ax, (.495,.622), (.518,.622), COLORS["ink"])

    missing = [[0,1,0,1],[1,0,1,0],[0,0,1,1]]
    matrix(ax, .285, .315, .078, .130, COLORS["orange"],
           [[.2,0,.8,0],[0,.7,0,.3],[.9,.4,0,0]], missing)
    ax.text(.324, .275, "ChEMBL archive", ha="center", va="top", fontsize=7.2)
    ax.text(.442, .490, "biological target exclusion", ha="center", va="center",
            fontsize=6.8, color=COLORS["red"])
    arrow(ax, (.365,.380), (.392,.380), COLORS["ink"])
    square_box(ax, (.395, .315), .130, .130, "entity / assay profile\ncoverage + potency",
               edge=COLORS["orange"], fontsize=6.3)
    matrix(ax, .575, .322, .070, .115, COLORS["orange"],
           [[1,.3,.1],[.3,1,.6],[.1,.6,1]])
    ax.text(.610, .298, "Kprof", ha="center", va="top", fontsize=7.5,
            fontstyle="italic")
    arrow(ax, (.527,.380), (.572,.380), COLORS["ink"])
    ax.text(.495, .205, "Kprof ∝ Uent + Uassay + Dent + Dassay",
            ha="center", va="center", fontsize=7.2, fontstyle="italic")
    ax.text(.495, .155, "+ Kchem ⊙ (Dent + Dassay)",
            ha="center", va="center", fontsize=7.2, fontstyle="italic")

    # (c) Only symbols needed for the computation remain in the final panel.
    matrix(ax, .765, .555, .050, .085, COLORS["blue"], [[1,.5],[.5,1]])
    matrix(ax, .765, .330, .050, .085, COLORS["orange"], [[1,.3],[.3,1]])
    ax.text(.823, .597, "Kchem", ha="left", va="center", fontsize=7.5,
            fontstyle="italic")
    ax.text(.823, .372, "Kprof", ha="left", va="center", fontsize=7.5,
            fontstyle="italic")
    arrow(ax, (.875,.597), (.895,.490)); arrow(ax, (.875,.372), (.895,.445))
    square_box(ax, (.895, .395), .070, .135, "Kλ", edge=COLORS["ink"], fontsize=9, weight="bold")
    arrow(ax, (.930,.393), (.930,.325))
    ax.text(.930, .285, "ε-SVR", ha="center", va="center", fontsize=7.5)
    arrow(ax, (.930,.245), (.930,.185))
    ax.text(.930, .135, "ŷ", ha="center", va="center", fontsize=10, fontweight="bold")

    fig.subplots_adjust(left=.01, right=.995, top=.99, bottom=.02)
    save_figure(fig, args.output)


if __name__ == "__main__":
    main()
