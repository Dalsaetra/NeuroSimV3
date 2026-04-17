from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap


BRUNEL_ORDER = ["AI", "SI", "AR", "SR"]
BRUNEL_CLASS_TO_CODE = {name: idx for idx, name in enumerate(BRUNEL_ORDER)}
BRUNEL_CODE_TO_CLASS = {value: key for key, value in BRUNEL_CLASS_TO_CODE.items()}
DEFAULT_CYCLE_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
BRUNEL_COLORS = {
    brunel_class: DEFAULT_CYCLE_COLORS[idx]
    for idx, brunel_class in enumerate(BRUNEL_ORDER)
}
BRUNEL_LISTED_CMAP = ListedColormap([BRUNEL_COLORS[name] for name in BRUNEL_ORDER])
BRUNEL_NORM = BoundaryNorm(np.arange(-0.5, len(BRUNEL_ORDER) + 0.5, 1.0), BRUNEL_LISTED_CMAP.N)
TOPOLOGY_MARKERS = {"random_fixed": "o", "spatial": "^"}
NORMALIZATION_ORDER = ["none", "rate", "soft_in"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze week12 plasticity/normalization sweep outputs and generate summary plots."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "plasticity_normalization_sweep",
        help="Directory containing summary.csv from the sweep.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional explicit path to summary.csv. Overrides --results-dir/summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to <results-dir>/analysis.",
    )
    return parser.parse_args()


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        if column in {"run_id", "topology", "topology_label", "condition_name", "normalization_kind", "details_json", "status", "error"}:
            continue
        try:
            out[column] = pd.to_numeric(out[column])
        except (ValueError, TypeError):
            pass
    return out


def _classify_brunel(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["brunel_irregular"] = (
        (out["recovery__Fano_median_300ms"] > 0.99)
        & (out["recovery__ISI_CV_mean_E"] > 0.99)
    )
    out["brunel_synchronous"] = (
        (out["recovery__mean_noise_corr_50ms"] > 0.05)
        | (out["recovery__psd_peak_ratio"] > 200.0)
    )
    out["brunel_class"] = np.select(
        [
            ~out["brunel_synchronous"] & ~out["brunel_irregular"],
            ~out["brunel_synchronous"] & out["brunel_irregular"],
            out["brunel_synchronous"] & ~out["brunel_irregular"],
            out["brunel_synchronous"] & out["brunel_irregular"],
        ],
        ["AR", "AI", "SR", "SI"],
        default="unknown",
    )
    out["brunel_code"] = out["brunel_class"].map(BRUNEL_CLASS_TO_CODE)
    return out


def _add_condition_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["apply_every_ms_int"] = out["apply_every_ms"].round().astype("Int64")
    out["eta_label"] = out["eta_E"].map(lambda value: "" if pd.isna(value) else f"{value:g}")
    out["exp_label"] = out["normalization_exponent"].map(lambda value: "" if pd.isna(value) else f"{value:g}")

    def label_row(row: pd.Series) -> str:
        kind = row["normalization_kind"]
        if kind == "none":
            return "Clopath only"
        if kind == "rate":
            return f"Rate {int(row['apply_every_ms'])} ms, eta={row['eta_E']:g}"
        if kind == "soft_in":
            return f"Soft-in {int(row['apply_every_ms'])} ms, exp={row['normalization_exponent']:g}"
        return str(row["condition_name"])

    out["condition_label"] = out.apply(label_row, axis=1)
    return out


def _attach_weight_change_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "mean_abs_weight_change" in out.columns:
        out["plot_mean_abs_weight_change"] = out["mean_abs_weight_change"]
        out["plot_mean_abs_weight_change_source"] = "stored"
    else:
        out["plot_mean_abs_weight_change"] = out["weight_change_mean"].abs()
        out["plot_mean_abs_weight_change_source"] = "abs_of_signed_mean_fallback"
    return out


def load_summary(summary_path: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_path)
    df = _coerce_numeric(df)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    df = _classify_brunel(df)
    df = _add_condition_labels(df)
    df = _attach_weight_change_columns(df)
    return df


def _save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _aggregate_condition_metrics(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["topology", "normalization_kind", "apply_every_ms", "eta_E", "normalization_exponent"]
    metrics = {
        "recovery__rate_mean_Hz": ["mean", "std"],
        "recovery__rate_mean_Hz_E": ["mean", "std"],
        "recovery__participation_frac_total": ["mean", "std"],
        "recovery__ISI_CV_mean_E": ["mean", "std"],
        "recovery__Fano_median_300ms": ["mean", "std"],
        "recovery__mean_noise_corr_50ms": ["mean", "std"],
        "recovery__psd_peak_ratio": ["mean", "std"],
        "plot_mean_abs_weight_change": ["mean", "std"],
        "weight_change_std": ["mean", "std"],
    }
    grouped = df.groupby(group_cols, dropna=False).agg(metrics)
    grouped.columns = ["__".join(col).strip("_") for col in grouped.columns.to_flat_index()]
    grouped = grouped.reset_index()

    class_counts = (
        df.groupby(group_cols + ["brunel_class"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    merged = grouped.merge(class_counts, on=group_cols, how="left")
    for brunel_class in BRUNEL_ORDER:
        if brunel_class not in merged.columns:
            merged[brunel_class] = 0
    total = merged[BRUNEL_ORDER].sum(axis=1).replace(0, np.nan)
    for brunel_class in BRUNEL_ORDER:
        merged[f"frac_{brunel_class}"] = merged[brunel_class] / total
    return merged


def _aggregate_class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby(["topology", "normalization_kind", "brunel_class"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for brunel_class in BRUNEL_ORDER:
        if brunel_class not in counts.columns:
            counts[brunel_class] = 0
    total = counts[BRUNEL_ORDER].sum(axis=1)
    for brunel_class in BRUNEL_ORDER:
        counts[f"frac_{brunel_class}"] = counts[brunel_class] / total.replace(0, np.nan)
    return counts


def _ensure_numeric_order(values: pd.Series) -> list[float]:
    return sorted(float(v) for v in values.dropna().unique())


def _draw_numeric_heatmap(
    ax: plt.Axes,
    subset: pd.DataFrame,
    row_col: str,
    col_col: str,
    value_col: str,
    title: str,
    fmt: str = ".2f",
    cmap: str = "viridis",
) -> None:
    if subset.empty:
        ax.axis("off")
        ax.set_title(f"{title}\n(no data)")
        return

    pivot = subset.pivot_table(index=row_col, columns=col_col, values=value_col, aggfunc="mean")
    pivot = pivot.sort_index().sort_index(axis=1)
    image = ax.imshow(pivot.values, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{float(v):g}" for v in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{float(v):g}" for v in pivot.index])
    ax.set_xlabel(col_col)
    ax.set_ylabel(row_col)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            label = "nan" if pd.isna(value) else format(float(value), fmt)
            ax.text(j, i, label, ha="center", va="center", color="white", fontsize=8)

    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def _draw_class_heatmap(
    ax: plt.Axes,
    subset: pd.DataFrame,
    row_col: str,
    col_col: str,
    title: str,
) -> None:
    if subset.empty:
        ax.axis("off")
        ax.set_title(f"{title}\n(no data)")
        return

    rows = _ensure_numeric_order(subset[row_col])
    cols = _ensure_numeric_order(subset[col_col])
    grid = np.full((len(rows), len(cols)), np.nan, dtype=float)
    labels = np.full((len(rows), len(cols)), "", dtype=object)

    for i, row_value in enumerate(rows):
        for j, col_value in enumerate(cols):
            cell = subset[(subset[row_col] == row_value) & (subset[col_col] == col_value)]
            if cell.empty:
                continue
            mode_class = cell["brunel_class"].mode().iat[0]
            grid[i, j] = BRUNEL_CLASS_TO_CODE[mode_class]
            labels[i, j] = mode_class

    image = ax.imshow(grid, aspect="auto", cmap=BRUNEL_LISTED_CMAP, norm=BRUNEL_NORM)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([f"{value:g}" for value in cols])
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([f"{value:g}" for value in rows])
    ax.set_xlabel(col_col)
    ax.set_ylabel(row_col)

    for i in range(len(rows)):
        for j in range(len(cols)):
            if labels[i, j]:
                ax.text(j, i, labels[i, j], ha="center", va="center", color="white", fontsize=9, fontweight="bold")

    cbar = plt.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        ticks=np.arange(len(BRUNEL_ORDER)),
        boundaries=np.arange(-0.5, len(BRUNEL_ORDER) + 0.5, 1.0),
    )
    cbar.set_ticklabels(BRUNEL_ORDER)


def plot_brunel_phase_panels(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for brunel_class in BRUNEL_ORDER:
        color = BRUNEL_COLORS[brunel_class]
        subset = df[df["brunel_class"] == brunel_class]
        for topology, marker in TOPOLOGY_MARKERS.items():
            topo_subset = subset[subset["topology"] == topology]
            axes[0].scatter(
                topo_subset["recovery__ISI_CV_mean_E"],
                topo_subset["recovery__Fano_median_300ms"],
                s=50,
                alpha=0.8,
                color=color,
                marker=marker,
                label=f"{brunel_class} / {topology}",
            )
            axes[1].scatter(
                topo_subset["recovery__mean_noise_corr_50ms"],
                topo_subset["recovery__psd_peak_ratio"],
                s=50,
                alpha=0.8,
                color=color,
                marker=marker,
                label=f"{brunel_class} / {topology}",
            )

    axes[0].axvline(0.99, color="black", linestyle="--", linewidth=1)
    axes[0].axhline(0.99, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Recovery ISI_CV_mean_E")
    axes[0].set_ylabel("Recovery Fano median 300 ms")
    axes[0].set_title("Irregularity Criteria")

    axes[1].axvline(0.05, color="black", linestyle="--", linewidth=1)
    axes[1].axhline(200.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Recovery mean noise corr 50 ms")
    axes[1].set_ylabel("Recovery PSD peak ratio")
    axes[1].set_title("Synchrony Criteria")

    handles, labels = axes[1].get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    fig.legend(dedup.values(), dedup.keys(), loc="upper center", bbox_to_anchor=(0.5, 0.85), ncol=4, frameon=False)
    fig.suptitle("Brunel Classification Phase Panels", y=0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_persistence_and_weights(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for brunel_class in BRUNEL_ORDER:
        color = BRUNEL_COLORS[brunel_class]
        subset = df[df["brunel_class"] == brunel_class]
        axes[0, 0].scatter(
            subset["recovery__participation_frac_total"],
            subset["recovery__rate_mean_Hz"],
            s=55,
            alpha=0.8,
            color=color,
            label=brunel_class,
        )
        axes[0, 1].scatter(
            subset["plot_mean_abs_weight_change"],
            subset["recovery__rate_mean_Hz"],
            s=55,
            alpha=0.8,
            color=color,
            label=brunel_class,
        )

    axes[0, 0].set_xlabel("Recovery participation frac total")
    axes[0, 0].set_ylabel("Recovery mean rate (Hz)")
    axes[0, 0].set_title("Persistence Proxy: Participation vs Rate")

    axes[0, 1].set_xlabel("Mean absolute weight change")
    axes[0, 1].set_ylabel("Recovery mean rate (Hz)")
    axes[0, 1].set_title("Weight Change vs Recovery Rate")

    norm_groups = []
    norm_labels = []
    for kind in NORMALIZATION_ORDER:
        subset = df[df["normalization_kind"] == kind]
        if subset.empty:
            continue
        norm_groups.append(subset["recovery__rate_mean_Hz"].values)
        norm_labels.append(kind)
    axes[1, 0].boxplot(norm_groups, tick_labels=norm_labels, showfliers=False)
    axes[1, 0].set_ylabel("Recovery mean rate (Hz)")
    axes[1, 0].set_title("Recovery Rate by Normalization Family")

    topo_groups = []
    topo_labels = []
    for topology in sorted(df["topology"].unique()):
        subset = df[df["topology"] == topology]
        topo_groups.append(subset["weight_change_std"].values)
        topo_labels.append(topology)
    axes[1, 1].boxplot(topo_groups, tick_labels=topo_labels, showfliers=False)
    axes[1, 1].set_ylabel("Weight change std")
    axes[1, 1].set_title("Weight Change Spread by Topology")

    axes[0, 0].legend(frameon=False)
    fig.suptitle("Persistence and Weight-Change Overview", y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(df: pd.DataFrame, out_path: Path) -> None:
    topologies = sorted(df["topology"].unique())
    fig, axes = plt.subplots(1, len(topologies), figsize=(7 * len(topologies), 5), squeeze=False)

    for ax, topology in zip(axes[0], topologies):
        subset = df[df["topology"] == topology]
        summary = (
            subset.groupby(["normalization_kind", "brunel_class"])
            .size()
            .unstack(fill_value=0)
            .reindex(NORMALIZATION_ORDER)
            .fillna(0)
        )
        total = summary.sum(axis=1).replace(0, np.nan)
        bottom = np.zeros(len(summary), dtype=float)
        x = np.arange(len(summary.index))

        for brunel_class in BRUNEL_ORDER:
            values = (summary.get(brunel_class, 0) / total).fillna(0).values
            ax.bar(x, values, bottom=bottom, color=BRUNEL_COLORS[brunel_class], label=brunel_class)
            bottom += values

        ax.set_xticks(x)
        ax.set_xticklabels(summary.index, rotation=0)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction of runs")
        ax.set_title(f"Brunel Class Fractions: {topology}")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_rate_heatmaps(df: pd.DataFrame, out_path: Path) -> None:
    rate_df = df[df["normalization_kind"] == "rate"].copy()
    topologies = sorted(df["topology"].unique())
    fig, axes = plt.subplots(len(topologies), 4, figsize=(20, 5 * len(topologies)), squeeze=False)

    for row_idx, topology in enumerate(topologies):
        subset = rate_df[rate_df["topology"] == topology]
        _draw_numeric_heatmap(
            axes[row_idx, 0],
            subset,
            "apply_every_ms",
            "eta_E",
            "recovery__rate_mean_Hz",
            f"{topology}: mean recovery rate",
            fmt=".1f",
        )
        _draw_numeric_heatmap(
            axes[row_idx, 1],
            subset,
            "apply_every_ms",
            "eta_E",
            "recovery__participation_frac_total",
            f"{topology}: recovery participation",
            fmt=".2f",
        )
        _draw_numeric_heatmap(
            axes[row_idx, 2],
            subset,
            "apply_every_ms",
            "eta_E",
            "plot_mean_abs_weight_change",
            f"{topology}: mean absolute weight change",
            fmt=".2f",
        )
        _draw_class_heatmap(
            axes[row_idx, 3],
            subset,
            "apply_every_ms",
            "eta_E",
            f"{topology}: modal Brunel class",
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_soft_in_heatmaps(df: pd.DataFrame, out_path: Path) -> None:
    soft_df = df[df["normalization_kind"] == "soft_in"].copy()
    topologies = sorted(df["topology"].unique())
    fig, axes = plt.subplots(len(topologies), 4, figsize=(20, 5 * len(topologies)), squeeze=False)

    for row_idx, topology in enumerate(topologies):
        subset = soft_df[soft_df["topology"] == topology]
        _draw_numeric_heatmap(
            axes[row_idx, 0],
            subset,
            "apply_every_ms",
            "normalization_exponent",
            "recovery__rate_mean_Hz",
            f"{topology}: mean recovery rate",
            fmt=".1f",
        )
        _draw_numeric_heatmap(
            axes[row_idx, 1],
            subset,
            "apply_every_ms",
            "normalization_exponent",
            "recovery__participation_frac_total",
            f"{topology}: recovery participation",
            fmt=".2f",
        )
        _draw_numeric_heatmap(
            axes[row_idx, 2],
            subset,
            "apply_every_ms",
            "normalization_exponent",
            "plot_mean_abs_weight_change",
            f"{topology}: mean absolute weight change",
            fmt=".2f",
        )
        _draw_class_heatmap(
            axes[row_idx, 3],
            subset,
            "apply_every_ms",
            "normalization_exponent",
            f"{topology}: modal Brunel class",
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_path = args.summary_csv if args.summary_csv is not None else args.results_dir / "summary.csv"
    output_dir = args.output_dir if args.output_dir is not None else args.results_dir / "analysis"
    figures_dir = output_dir / "figures"

    df = load_summary(summary_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    _save_dataframe(df, output_dir / "summary_with_brunel.csv")
    _save_dataframe(_aggregate_condition_metrics(df), output_dir / "aggregated_condition_metrics.csv")
    _save_dataframe(_aggregate_class_distribution(df), output_dir / "brunel_class_distribution.csv")

    plot_brunel_phase_panels(df, figures_dir / "brunel_phase_panels.png")
    plot_persistence_and_weights(df, figures_dir / "persistence_and_weights.png")
    plot_class_distribution(df, figures_dir / "brunel_class_distribution.png")
    plot_rate_heatmaps(df, figures_dir / "rate_normalization_heatmaps.png")
    plot_soft_in_heatmaps(df, figures_dir / "soft_in_normalization_heatmaps.png")

    print(f"Loaded {len(df)} successful runs from {summary_path}")
    print(f"Wrote analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()
