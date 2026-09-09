from __future__ import annotations

"""Benchmark frozen-model scoring and draw the performance--latency trade-off."""

import argparse
import json
import pickle
import platform
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from figure_style import save_figure, set_publication_style
from molcliff.baselines import make_regressor
from molcliff.data import Fingerprints, MOLECULEACE_DATASETS, load_moleculeace


ROOT = Path(__file__).resolve().parents[1]
DISPLAY = {
    "svm_official": "ECFP-SVM",
    "rf_official": "ECFP-RF",
    "gbm_official": "ECFP-GBM",
    "linear_ecfp": "Ridge-ECFP",
    "extratrees_ecfp": "ExtraTrees-ECFP",
    "random_forest_ecfp": "RF-ECFP",
    "responsekernel_final": "TAPER",
}
PALETTE = {
    "svm_official": "#4DBBD5",
    "rf_official": "#8491B4",
    "gbm_official": "#91D1C2",
    "linear_ecfp": "#3C5488",
    "extratrees_ecfp": "#F39B7F",
    "random_forest_ecfp": "#8491B4",
    "responsekernel_final": "#E64B35",
}
MARKERS = {
    "svm_official": "o",
    "rf_official": "s",
    "gbm_official": "^",
    "linear_ecfp": "P",
    "extratrees_ecfp": "X",
    "random_forest_ecfp": "h",
    "responsekernel_final": "D",
}


def _latency_ms(model: object, query: np.ndarray, repeats: int) -> float:
    """Median synchronized wall-clock latency per molecule."""
    model.predict(query)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        model.predict(query)
        samples.append((time.perf_counter_ns() - start) / 1e6 / len(query))
    return float(np.median(samples))


def benchmark(batch_size: int, repeats: int, model_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    fingerprints = Fingerprints(radius=2, n_bits=1024)
    with threadpool_limits(limits=1):
        for dataset in MOLECULEACE_DATASETS:
            frame = load_moleculeace(dataset)
            train = frame[frame.split.eq("train")].reset_index(drop=True)
            x_train = fingerprints.bits(train.smiles.to_numpy())
            query_rows = min(batch_size, len(train))
            y_train = train.target.to_numpy(float)
            test = frame[frame.split.eq("test")].reset_index(drop=True)
            x_test = fingerprints.bits(test.smiles.to_numpy())
            y_test = test.target.to_numpy(float)
            for name in (
                "svm_official", "rf_official", "gbm_official",
                "linear_ecfp", "extratrees_ecfp", "random_forest_ecfp",
            ):
                if name == "linear_ecfp":
                    model = Ridge(alpha=1.0)
                elif name == "extratrees_ecfp":
                    model = ExtraTreesRegressor(
                        n_estimators=192, max_features="sqrt", min_samples_leaf=2,
                        random_state=42, n_jobs=1,
                    )
                elif name == "random_forest_ecfp":
                    model = RandomForestRegressor(
                        n_estimators=192, max_features="sqrt", min_samples_leaf=2,
                        random_state=42, n_jobs=1,
                    )
                else:
                    model = make_regressor(name, dataset=dataset)
                if hasattr(model, "n_jobs"):
                    model.set_params(n_jobs=1)
                model.fit(x_train, y_train)
                latency = _latency_ms(model, x_train[:query_rows], repeats)
                item = {
                    "dataset": dataset,
                    "model": name,
                    "batch_size": query_rows,
                    "latency_ms_per_molecule": latency,
                }
                if name in {"linear_ecfp", "extratrees_ecfp", "random_forest_ecfp"}:
                    item["official_rmse"] = float(
                        np.sqrt(np.mean((model.predict(x_test) - y_test) ** 2))
                    )
                rows.append(item)

            artifact_dir = model_root / dataset
            with (artifact_dir / f"{dataset}.svr.pkl").open("rb") as handle:
                response_model = pickle.load(handle)
            artifact = np.load(artifact_dir / f"{dataset}.train_kernel.npz")
            response_query = artifact["kernel"][:query_rows]
            latency = _latency_ms(response_model, response_query, repeats)
            rows.append({
                "dataset": dataset,
                "model": "responsekernel_final",
                "batch_size": query_rows,
                "latency_ms_per_molecule": latency,
            })
    return pd.DataFrame(rows)


def add_performance(frame: pd.DataFrame) -> pd.DataFrame:
    published = pd.read_csv(
        ROOT / "third_party" / "MoleculeACE" / "MoleculeACE"
        / "Data" / "results" / "MoleculeACE_results.csv"
    )
    performance: dict[str, float] = {}
    for algorithm, name in (
        ("SVM", "svm_official"), ("RF", "rf_official"), ("GBM", "gbm_official")
    ):
        subset = published[
            published.algorithm.eq(algorithm)
            & published.descriptor.eq("ECFP")
            & published.augmentation.eq(0)
        ]
        if subset.dataset.nunique() != len(MOLECULEACE_DATASETS):
            raise RuntimeError(f"Incomplete published performance for {algorithm}")
        performance[name] = float(subset.rmse.mean())
    response = pd.read_csv(
        ROOT / "outputs" / "responsekernel_final_official_test_v1" / "summary.test.csv"
    )
    performance["responsekernel_final"] = float(response.rmse.mean())
    for name in ("linear_ecfp", "extratrees_ecfp", "random_forest_ecfp"):
        values = frame.loc[frame.model.eq(name), "official_rmse"]
        if values.empty:
            raise RuntimeError(f"Missing official RMSE for {name}")
        performance[name] = float(values.mean())
    frame = frame.copy()
    frame["macro_rmse"] = frame.model.map(performance)
    return frame


def draw(frame: pd.DataFrame, path: Path) -> None:
    summary = frame.groupby("model").latency_ms_per_molecule.agg(
        median="median",
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75),
    )
    summary["macro_rmse"] = frame.groupby("model").macro_rmse.first()
    order = [
        "gbm_official", "responsekernel_final", "linear_ecfp",
        "extratrees_ecfp", "random_forest_ecfp", "rf_official", "svm_official",
    ]
    summary = summary.loc[[name for name in order if name in summary.index]]

    set_publication_style()
    fig, ax = plt.subplots(figsize=(7.05, 4.70), facecolor="white")
    ax.set_facecolor("white")
    annotation_offsets = {
        "gbm_official": (8, -15), "responsekernel_final": (8, 10),
        "linear_ecfp": (8, 12), "extratrees_ecfp": (8, -14),
        "random_forest_ecfp": (8, -2), "rf_official": (8, 11),
        "svm_official": (8, -2),
    }
    for name, row in summary.iterrows():
        xerr = np.asarray([[row["median"] - row["q1"]], [row["q3"] - row["median"]]])
        ax.errorbar(
            row["median"], row["macro_rmse"], xerr=xerr,
            fmt=MARKERS[name], markersize=9.0 if name == "responsekernel_final" else 7.5,
            color=PALETTE[name], ecolor=PALETTE[name], elinewidth=1.25,
            capsize=3.0, markeredgecolor="white", markeredgewidth=0.8, zorder=4,
        )
        ax.annotate(
            DISPLAY[name], (row["median"], row["macro_rmse"]),
            xytext=annotation_offsets[name], textcoords="offset points",
            fontsize=12, color="#202124", ha="left", va="center",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Frozen-model scoring latency (ms molecule$^{-1}$)", fontsize=14)
    ax.set_ylabel("MoleculeACE macro RMSE", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, linestyle="--", alpha=0.25, color="gray", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#202124")
    ax.spines["bottom"].set_color("#202124")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.margins(x=0.22, y=0.16)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.97, bottom=0.19)
    save_figure(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument(
        "--model-root", type=Path,
        default=Path("outputs/responsekernel_final_model_parts"),
    )
    parser.add_argument(
        "--output-csv", type=Path,
        default=Path("reports/results/inference_efficiency.csv"),
    )
    parser.add_argument(
        "--figure", type=Path,
        default=Path("paper/figures/fig_inference_efficiency.pdf"),
    )
    args = parser.parse_args()
    frame = add_performance(benchmark(args.batch_size, args.repeats, args.model_root))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)
    draw(frame, args.figure)
    metadata = {
        "protocol": "single-thread frozen-model predict; precomputed representations; "
                    "median of repeated batches; training excluded",
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "targets": len(MOLECULEACE_DATASETS),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "summary": frame.groupby("model").agg(
            latency_median=("latency_ms_per_molecule", "median"),
            latency_q1=("latency_ms_per_molecule", lambda x: x.quantile(0.25)),
            latency_q3=("latency_ms_per_molecule", lambda x: x.quantile(0.75)),
            macro_rmse=("macro_rmse", "first"),
        ).to_dict(orient="index"),
    }
    args.output_csv.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
