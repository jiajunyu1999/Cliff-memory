from __future__ import annotations

"""Annotate fit-selected response coordinates using cached ChEMBL metadata.

This is an enrichment description of the coordinates selected by the frozen
TAPER protocol; it is not a causal attribution of affinity.
"""

import ast
import gzip
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data/chembl_target_metadata.jsonl.gz"
RUNS = [ROOT / "outputs/responsekernel_final_cv5_seed7/summary.cv5.csv",
        ROOT / "outputs/responsekernel_final_cv5_seed73/summary.cv5.csv"]
OUT = ROOT / "reports/results/extended_evidence"
FIG = ROOT / "paper/figures/fig_response_biology.pdf"


def family(name: str) -> str:
    text = name.lower()
    if "kinase" in text:
        return "Kinase"
    if "ion channel" in text or "channel" in text:
        return "Ion channel"
    if "transporter" in text or "transport protein" in text:
        return "Transporter"
    if "receptor" in text or "adrenergic" in text or "dopamine" in text or "serotonin" in text:
        return "Receptor"
    if any(x in text for x in ["protease", "reductase", "dehydrogenase", "phosphatase", "synthase", "oxidase", "hydrolase", "enzyme"]):
        return "Enzyme"
    return "Other"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    names = {}
    with gzip.open(META, "rt") as handle:
        for line in handle:
            record = json.loads(line)
            names[str(record["target_chembl_id"])] = str(record.get("pref_name") or "Unknown")
    counter: Counter[tuple[str, str]] = Counter()
    for path in RUNS:
        frame = pd.read_csv(path)
        frame = frame[frame.model.eq("responsekernel_final")]
        for row in frame.itertuples(index=False):
            for target in ast.literal_eval(row.nearest_assays):
                name = names.get(str(target), "Unknown")
                counter[(str(target), name)] += 1
    rows = [{"target_chembl_id": tid, "target_name": name, "selection_count": count,
             "family": family(name)} for (tid, name), count in counter.items()]
    ranked = pd.DataFrame(rows).sort_values(["selection_count", "target_name"], ascending=[False, True])
    ranked.to_csv(OUT / "response_coordinate_annotations.csv", index=False)
    counts = ranked.groupby("family", as_index=False).selection_count.sum().sort_values("selection_count", ascending=False)
    counts.to_csv(OUT / "response_coordinate_family_counts.csv", index=False)
    # The right panel deliberately excludes records whose preferred-name field
    # is a status/species/non-protein label; they remain counted transparently
    # in the left-panel ``Other'' category.
    top = ranked[ranked.family.ne("Other")].head(15)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.labelsize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.0), facecolor="white",
                             gridspec_kw={"width_ratios": [1, 1.3]})
    palette = {"Kinase": "#4DBBD5", "Receptor": "#00A087", "Enzyme": "#E64B35",
               "Ion channel": "#8E6C8A", "Transporter": "#F28E2B", "Other": "#6C7A89"}
    c = counts.iloc[::-1]
    axes[0].barh(c.family, c.selection_count, color=[palette.get(x, "#6C7A89") for x in c.family], zorder=3)
    axes[0].set_xlabel("Selections across target-folds")
    top_plot = top.iloc[::-1]
    labels = [x if len(x) <= 32 else x[:29] + "…" for x in top_plot.target_name]
    axes[1].barh(labels, top_plot.selection_count, color="#4DBBD5", zorder=3)
    axes[1].set_xlabel("Selection count")
    for ax, letter in zip(axes, "ab"):
        ax.set_facecolor("white"); ax.grid(axis="x", ls="--", lw=.45, alpha=.28, zorder=0)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.text(-.15, 1.04, letter, transform=ax.transAxes, fontweight="bold", fontsize=11)
    fig.tight_layout(w_pad=1.3)
    fig.savefig(FIG, bbox_inches="tight")
    fig.savefig(OUT / "fig_response_biology.png", dpi=350, bbox_inches="tight")
    (OUT / "response_coordinate_annotation_summary.json").write_text(json.dumps({
        "unique_coordinates": int(len(ranked)), "selection_events": int(ranked.selection_count.sum()),
        "top_coordinates": top[["target_chembl_id", "target_name", "selection_count", "family"]].to_dict(orient="records")
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
