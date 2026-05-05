"""Analyze final-paper normalization/topology grid-search outputs.

The script expects the CSV files produced by grid_search_normalization_topology.py.
It aggregates across seeds for each topology, J, g condition, then writes
summary tables and figure files.

Example:

    python FinalPaper/analyze_normalization_grid.py

    python FinalPaper/analyze_normalization_grid.py \
        --results-dir FinalPaper/results/normalization_grid \
        --output-dir FinalPaper/results/normalization_grid_analysis
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch, Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "FinalPaper" / "results" / "normalization_grid"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "FinalPaper" / "results" / "normalization_grid_analysis"

GROUP_COLUMNS = ["topology", "j_gain", "g_gain"]
CONDITION_COLUMNS = [*GROUP_COLUMNS]

BRUNEL_ORDER = ["AI", "SI", "AR", "SR"]
BRUNEL_COLORS = {
    "AI": "#4C78A8",
    "SI": "#72B7B2",
    "AR": "#F58518",
    "SR": "#E45756",
    "NA": "#D9D9D9",
}

PERSISTENCE_REGEN_ORDER = ["EN", "ER", "PN", "PR"]
PERSISTENCE_REGEN_COLORS = {
    "EN": "#D9D9D9",
    "ER": "#59A14F",
    "PN": "#B07AA1",
    "PR": "#E15759",
    "NA": "#F0F0F0",
}

BIFURCATION_PHASE_ORDER = ["Q", "A", "SO"]
BIFURCATION_PHASE_COLORS = {
    "Q": "#D9D9D9",
    "A": "#4E79A7",
    "SO": "#E15759",
    "NA": "#F0F0F0",
}

BIFURCATION_REGION_ORDER = ["monostable", "bistable", "oscillatory"]
BIFURCATION_REGION_COLORS = {
    "monostable": "#D9D9D9",
    "bistable": "#F4A261",
    "oscillatory": "#1D3557",
    "NA": "#F0F0F0",
}

AXIS_LABELS = {
    "J": ("J global normalization gain", "J"),
    "g": ("g inhibitory-input normalization gain", "g"),
    "g_b_target": ("g b-target normalization gain", "g_b"),
    "E_delay_or_lambda": ("E delay mean (fixed, ms) / E spatial lambda", "E delay/lambda"),
    "I_delay_or_lambda": ("I delay mean (fixed, ms) / I spatial lambda", "I delay/lambda"),
    "E_threshold_variance": ("E neuron Vt variance", "E Vt var"),
    "I_threshold_variance": ("I neuron Vt variance", "I Vt var"),
    "n_neurons": ("number of neurons", "n"),
    "inhibitory_fraction": ("inhibitory neuron fraction", "I fraction"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--fano-column",
        default="observation.Fano_median_300ms",
        help="Fano metric from regime_rows_all.csv to plot as the Fano heatmap.",
    )
    parser.add_argument(
        "--regime-metric-columns",
        nargs="+",
        default=[
            "observation.ISI_CV_mean_E",
            "observation.ISI_CV_mean_I",
            "observation.psd_peak_ratio",
            "observation.pop_spec_entropy_norm",
            "observation.psd_peak_freq_hz",
            "observation.mean_noise_corr_50ms",
            "observation.potjans_diesmann_synchrony_3ms",
            "observation.participation_frac_total_E",
            "observation.participation_frac_total_I",
            "observation.rate_mean_Hz_E",
            "observation.rate_mean_Hz_I",
        ],
        help="Additional regime metrics from regime_rows_all.csv to plot as heatmaps.",
    )
    parser.add_argument(
        "--separation-columns",
        nargs="+",
        default=["effective_rank_norm_pooled", "effective_rank_norm_mean"],
        help="Separation metrics to plot from separation_summary_all.csv.",
    )
    parser.add_argument(
        "--memory-columns",
        nargs="+",
        default=[
            "memory_capacity",
            "memory_capacity_norm_delay_bound",
            "memory_capacity_mean_r2",
            "memory_first_negative_delay_ms",
        ],
        help="Memory metrics to plot from memory_summary_all.csv.",
    )
    parser.add_argument(
        "--fallback-memory-delay-bin-ms",
        type=float,
        default=30.0,
        help=(
            "Bin width used to derive memory_first_negative_delay_ms from old memory_summary CSVs "
            "that have r2_test_by_delay but no delays_ms column."
        ),
    )
    parser.add_argument(
        "--generalization-columns",
        nargs="+",
        default=[
            "between_within_scatter_ratio",
            "separability_index",
            "linear_readout_test_accuracy",
            "linear_readout_balanced_accuracy",
            "class_mean_effective_rank_norm",
            "mean_pairwise_class_distance",
            "trial_mean_rate_Hz_mean",
        ],
        help="Generalization metrics to plot from generalization_summary_all.csv.",
    )
    parser.add_argument(
        "--phase-slices",
        choices=["j", "g", "both"],
        default="both",
        help="Bifurcation phase slices to plot: fixed J, fixed g, or both.",
    )
    parser.add_argument("--brunel-cv-threshold", type=float, default=0.99)
    parser.add_argument("--brunel-fano-threshold", type=float, default=0.99)
    parser.add_argument("--brunel-corr-threshold", type=float, default=0.05)
    parser.add_argument("--brunel-peak-ratio-threshold", type=float, default=250.0)
    parser.add_argument("--brunel-entropy-norm-threshold", type=float, default=0.8)
    parser.add_argument("--brunel-bursty-fano-threshold", type=float, default=3.0)
    parser.add_argument(
        "--brunel-corr-only-hatch-threshold",
        type=float,
        default=0.5,
        help="Seed fraction of correlation-only synchronous seeds required to hatch Brunel map cells.",
    )
    parser.add_argument(
        "--use-saved-brunel-labels",
        action="store_true",
        help="Use brunel.brunel_class saved during simulation instead of recomputing labels from observation metrics.",
    )
    parser.add_argument(
        "--phase-region-osc-fraction",
        type=float,
        default=0.5,
        help="Seed fraction required to call an input-amplitude point oscillatory in the summary phase-region plots.",
    )
    parser.add_argument(
        "--phase-region-bistable-tolerance",
        type=float,
        default=1e-12,
        help="Minimum absolute difference between up/down active thresholds used to mark a path-dependent/bistable window.",
    )
    parser.add_argument(
        "--phase-region-bistable-fraction",
        type=float,
        default=0.5,
        help="Seed fraction required to call an input-amplitude point bistable in the summary phase-region plots.",
    )
    parser.add_argument(
        "--grid-row-label",
        default=None,
        help="Override heatmap row-axis label. By default inferred from x_name in the result CSVs.",
    )
    parser.add_argument(
        "--grid-column-label",
        default=None,
        help="Override heatmap column-axis label. By default inferred from y_name in the result CSVs.",
    )
    return parser.parse_args()


def read_table(results_dir: Path, stem: str) -> pd.DataFrame:
    all_path = results_dir / f"{stem}_all.csv"
    if all_path.exists():
        return pd.read_csv(all_path)

    job_paths = sorted(results_dir.glob(f"{stem}_job*.csv"))
    if not job_paths:
        raise FileNotFoundError(f"Could not find {all_path} or {stem}_job*.csv in {results_dir}")
    return pd.concat([pd.read_csv(path) for path in job_paths], ignore_index=True)


def read_optional_table(results_dir: Path, stem: str) -> pd.DataFrame:
    try:
        return read_table(results_dir, stem)
    except FileNotFoundError:
        return pd.DataFrame()


def ensure_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    if df.empty:
        return df
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def parse_array_cell(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(float, copy=False)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=float)
    if pd.isna(value):
        return np.asarray([], dtype=float)
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return np.asarray([], dtype=float)
    return np.asarray(parsed, dtype=float)


def first_negative_memory_delay(
    r2_values: np.ndarray,
    *,
    delays_bins: np.ndarray | None = None,
    delays_ms: np.ndarray | None = None,
    fallback_delay_bin_ms: float = 30.0,
) -> tuple[float, float]:
    r2 = np.asarray(r2_values, dtype=float).reshape(-1)
    if r2.size == 0:
        return np.nan, np.nan

    if delays_bins is None or np.asarray(delays_bins).size != r2.size:
        delays_bins_arr = np.arange(1, r2.size + 1, dtype=float)
    else:
        delays_bins_arr = np.asarray(delays_bins, dtype=float).reshape(-1)

    if delays_ms is None or np.asarray(delays_ms).size != r2.size:
        delays_ms_arr = delays_bins_arr * float(fallback_delay_bin_ms)
    else:
        delays_ms_arr = np.asarray(delays_ms, dtype=float).reshape(-1)

    valid = np.isfinite(r2) & np.isfinite(delays_bins_arr) & np.isfinite(delays_ms_arr)
    if not np.any(valid):
        return np.nan, np.nan

    order = np.argsort(delays_bins_arr[valid])
    r2_valid = r2[valid][order]
    delays_bins_valid = delays_bins_arr[valid][order]
    delays_ms_valid = delays_ms_arr[valid][order]
    if r2_valid[0] < 0.0:
        return 0.0, 0.0

    negative_idx = np.flatnonzero(r2_valid < 0.0)
    if negative_idx.size == 0:
        return np.nan, np.nan
    first_idx = int(negative_idx[0])
    return float(delays_bins_valid[first_idx]), float(delays_ms_valid[first_idx])


def add_memory_first_negative_delay_metrics(
    memory: pd.DataFrame,
    *,
    fallback_delay_bin_ms: float = 30.0,
) -> pd.DataFrame:
    if memory.empty or "r2_test_by_delay" not in memory.columns:
        return memory

    memory = memory.copy()
    for column in ("memory_first_negative_delay_bins", "memory_first_negative_delay_ms"):
        if column not in memory.columns:
            memory[column] = np.nan
        memory[column] = pd.to_numeric(memory[column], errors="coerce")

    needs_fill = memory["memory_first_negative_delay_ms"].isna()
    for idx, row in memory[needs_fill].iterrows():
        r2 = parse_array_cell(row.get("r2_test_by_delay"))
        delays_bins = parse_array_cell(row.get("delays_bins")) if "delays_bins" in memory.columns else None
        delays_ms = parse_array_cell(row.get("delays_ms")) if "delays_ms" in memory.columns else None
        first_bins, first_ms = first_negative_memory_delay(
            r2,
            delays_bins=delays_bins,
            delays_ms=delays_ms,
            fallback_delay_bin_ms=fallback_delay_bin_ms,
        )
        memory.at[idx, "memory_first_negative_delay_bins"] = first_bins
        memory.at[idx, "memory_first_negative_delay_ms"] = first_ms
    return memory


def coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(bool)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False})
        .fillna(False)
        .astype(bool)
    )


def format_axis_value(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):g}"


def pretty_label(text: str) -> str:
    replacements = {
        "r2": "R2",
        "psd": "PSD",
        "isi": "ISI",
        "cv": "CV",
        "hz": "Hz",
        "ms": "ms",
        "ai": "AI",
        "si": "SI",
        "ar": "AR",
        "sr": "SR",
        "vt": "Vt",
    }
    words = str(text).replace("_", " ").replace(".", " ").split()
    pretty_words = []
    for word in words:
        key = word.lower()
        pretty_words.append(replacements.get(key, word))
    label = " ".join(pretty_words)
    if not label:
        return ""
    return label[:1].upper() + label[1:]


def metric_title(column: str, *, statistic: str = "mean over seeds") -> str:
    label = pretty_label(column)
    if label.lower().startswith("observation "):
        label = label[len("observation "):]
    if label.lower().startswith("transient "):
        label = label[len("transient "):]
    return f"{label} {statistic}"


def format_metric_value(value: float) -> str:
    if pd.isna(value):
        return ""
    value = float(value)
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def safe_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() else "_" for ch in str(value).strip())
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_") or "axis"


def axis_label_from_name(axis_name: str | None, fallback_name: str) -> tuple[str, str]:
    if axis_name and str(axis_name) in AXIS_LABELS:
        return AXIS_LABELS[str(axis_name)]
    if axis_name and str(axis_name).strip():
        raw = str(axis_name).strip()
        return raw.replace("_", " "), raw
    return AXIS_LABELS[fallback_name]


def first_unique_value(dfs: Iterable[pd.DataFrame], column: str, default=None):
    for df in dfs:
        if df is None or df.empty or column not in df.columns:
            continue
        values = [value for value in df[column].dropna().astype(str).unique() if value and value.lower() != "nan"]
        if values:
            return values[0] if len(values) == 1 else "mixed"
    return default


def infer_grid_axis_labels(
    dfs: Iterable[pd.DataFrame],
    *,
    row_override: str | None = None,
    col_override: str | None = None,
) -> dict[str, dict[str, str]]:
    row_name = first_unique_value(dfs, "x_name", "J")
    col_name = first_unique_value(dfs, "y_name", "g")
    row_label, row_short = axis_label_from_name(row_name, "J")
    col_label, col_short = axis_label_from_name(col_name, "g")
    if row_override:
        row_label = row_override
    if col_override:
        col_label = col_override
    return {
        "j_gain": {
            "name": str(row_name),
            "label": row_label,
            "short": row_short,
            "token": safe_token(row_short),
        },
        "g_gain": {
            "name": str(col_name),
            "label": col_label,
            "short": col_short,
            "token": safe_token(col_short),
        },
    }


def infer_sweep_name(dfs: Iterable[pd.DataFrame]) -> str:
    return str(first_unique_value(dfs, "sweep", "normalization"))


def axis_labels_for_topology(
    axis_labels: dict[str, dict[str, str]],
    *,
    sweep_name: str,
    topology: str,
) -> dict[str, dict[str, str]]:
    labels = {axis: dict(values) for axis, values in axis_labels.items()}
    if sweep_name == "delay":
        if topology == "fixed":
            labels["j_gain"].update(
                {
                    "label": "E delay mean (ms)",
                    "short": "E delay ms",
                    "token": "E_delay_ms",
                }
            )
            labels["g_gain"].update(
                {
                    "label": "I delay mean (ms)",
                    "short": "I delay ms",
                    "token": "I_delay_ms",
                }
            )
        elif topology == "spatial":
            labels["j_gain"].update(
                {
                    "label": "E spatial lambda",
                    "short": "E lambda",
                    "token": "E_lambda",
                }
            )
            labels["g_gain"].update(
                {
                    "label": "I spatial lambda",
                    "short": "I lambda",
                    "token": "I_lambda",
                }
            )
    return labels


def majority_summary(
    df: pd.DataFrame,
    *,
    value_column: str,
    categories: list[str],
    group_columns: list[str] = CONDITION_COLUMNS,
) -> pd.DataFrame:
    rows = []
    for group_key, group in df.groupby(group_columns, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        counts = group[value_column].fillna("NA").astype(str).value_counts()
        n = int(counts.sum())
        if n == 0:
            mode_value = "NA"
            mode_count = 0
        else:
            sorted_counts = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            mode_value, mode_count = sorted_counts[0]

        row = {column: value for column, value in zip(group_columns, group_key)}
        row[f"{value_column}_mode"] = mode_value
        row[f"{value_column}_mode_count"] = int(mode_count)
        row[f"{value_column}_mode_fraction"] = float(mode_count / n) if n else np.nan
        row["n_seeds"] = n
        for category in categories:
            row[f"{value_column}_frac_{category}"] = float(counts.get(category, 0) / n) if n else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def numeric_summary(
    df: pd.DataFrame,
    *,
    value_columns: list[str],
    group_columns: list[str] = CONDITION_COLUMNS,
) -> pd.DataFrame:
    available = [column for column in value_columns if column in df.columns]
    if not available:
        return pd.DataFrame(columns=group_columns)
    summary = (
        df.groupby(group_columns, dropna=False)[available]
        .agg(["mean", "std", "median", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join([str(part) for part in col if str(part)])
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns
    ]
    return summary


def derive_persistence_regen_code(df: pd.DataFrame) -> pd.Series:
    persistent = coerce_bool(df["persistence.is_persistent"])
    persistent_regenerative = coerce_bool(df["regeneration.is_persistent_regenerative"])
    spread_regenerative = coerce_bool(df["regeneration.is_spread_regenerative"])
    regenerative = np.where(persistent, persistent_regenerative, spread_regenerative)
    persistence_prefix = np.where(persistent, "P", "E").astype(str)
    regeneration_suffix = np.where(regenerative, "R", "N").astype(str)
    return pd.Series(
        np.char.add(persistence_prefix, regeneration_suffix),
        index=df.index,
    )


def derive_bifurcation_phase(df: pd.DataFrame) -> pd.Series:
    active = coerce_bool(df["is_active"])
    strong = coerce_bool(df["is_strongly_oscillatory"])
    return pd.Series(np.where(strong, "SO", np.where(active, "A", "Q")), index=df.index)


def recompute_brunel_labels(
    df: pd.DataFrame,
    *,
    cv_threshold: float = 0.99,
    fano_threshold: float = 0.99,
    corr_threshold: float = 0.05,
    peak_ratio_threshold: float = 250.0,
    entropy_norm_threshold: float = 0.85,
    bursty_fano_threshold: float = 3.0,
) -> pd.DataFrame:
    out = df.copy()
    fano = pd.to_numeric(out.get("observation.Fano_median_300ms", 0.0), errors="coerce").fillna(0.0)
    isi_cv_e = pd.to_numeric(
        out.get("observation.ISI_CV_mean_E", out.get("observation.ISI_CV_mean", 0.0)),
        errors="coerce",
    ).fillna(0.0)
    corr = pd.to_numeric(out.get("observation.mean_noise_corr_50ms", 0.0), errors="coerce").fillna(0.0)
    peak_ratio = pd.to_numeric(out.get("observation.psd_peak_ratio", 0.0), errors="coerce").fillna(0.0)
    entropy_norm = pd.to_numeric(
        out.get("observation.pop_spec_entropy_norm", np.inf),
        errors="coerce",
    ).fillna(np.inf)

    irregular = (fano > float(fano_threshold)) & (isi_cv_e > float(cv_threshold))
    synchronous_by_corr = corr > float(corr_threshold)
    oscillatory = ((peak_ratio > float(peak_ratio_threshold)) & (entropy_norm < float(entropy_norm_threshold)))
    oscillatory = oscillatory | (peak_ratio > float(peak_ratio_threshold + 100.0))
    oscillatory = oscillatory | (entropy_norm < float(entropy_norm_threshold - 0.1))
    synchronous = synchronous_by_corr | oscillatory
    bursty_individual = fano > float(bursty_fano_threshold)

    out["brunel.saved_class"] = out.get("brunel.brunel_class", pd.Series(index=out.index, dtype=object))
    out["brunel.brunel_irregular"] = irregular.astype(bool)
    out["brunel.brunel_synchronous"] = synchronous.astype(bool)
    out["brunel.synchronous_by_corr"] = synchronous_by_corr.astype(bool)
    out["brunel.synchronous_by_oscillation"] = oscillatory.astype(bool)
    out["brunel.oscillatory"] = oscillatory.astype(bool)
    out["brunel.bursty_individual"] = bursty_individual.astype(bool)
    out["brunel.brunel_class"] = np.select(
        [
            (~synchronous) & irregular,
            synchronous & irregular,
            (~synchronous) & (~irregular),
        ],
        ["AI", "SI", "AR"],
        default="SR",
    )
    return out


def build_bifurcation_phase_regions(
    phase_summary: pd.DataFrame,
    bifurcation_summary: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    *,
    oscillatory_fraction_threshold: float = 0.5,
    bistable_fraction_threshold: float = 0.5,
    bistable_tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Build monostable/bistable/oscillatory labels over topology, J, g, amplitude."""
    if phase_summary.empty:
        return pd.DataFrame()

    pivot = (
        phase_summary.pivot_table(
            index=[*CONDITION_COLUMNS, "input_amplitude"],
            columns="sweep_direction",
            values="strongly_oscillatory_fraction",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={"up": "strongly_oscillatory_fraction_up", "down": "strongly_oscillatory_fraction_down"})
    )
    for column in ("strongly_oscillatory_fraction_up", "strongly_oscillatory_fraction_down"):
        if column not in pivot.columns:
            pivot[column] = np.nan
        pivot[column] = pd.to_numeric(pivot[column], errors="coerce")

    condition_amplitudes = phase_summary[[*CONDITION_COLUMNS, "input_amplitude"]].drop_duplicates()
    threshold_seed = bifurcation_summary[
        [
            *CONDITION_COLUMNS,
            "seed",
            "active_up_threshold",
            "active_down_threshold",
        ]
    ].copy()
    threshold_seed["active_up_threshold"] = pd.to_numeric(threshold_seed["active_up_threshold"], errors="coerce")
    threshold_seed["active_down_threshold"] = pd.to_numeric(threshold_seed["active_down_threshold"], errors="coerce")
    bistable_seed = condition_amplitudes.merge(threshold_seed, on=CONDITION_COLUMNS, how="left")
    up_seed = bistable_seed["active_up_threshold"]
    down_seed = bistable_seed["active_down_threshold"]
    lo_seed = np.minimum(up_seed, down_seed)
    hi_seed = np.maximum(up_seed, down_seed)
    width_seed = np.abs(up_seed - down_seed)
    bistable_seed["is_bistable_seed"] = (
        up_seed.notna()
        & down_seed.notna()
        & (width_seed > float(bistable_tolerance))
        & (bistable_seed["input_amplitude"] >= lo_seed)
        & (bistable_seed["input_amplitude"] < hi_seed)
    )
    bistable_fraction = (
        bistable_seed.groupby([*CONDITION_COLUMNS, "input_amplitude"], dropna=False)
        .agg(
            bistable_fraction=("is_bistable_seed", "mean"),
            bistable_seed_count=("is_bistable_seed", "sum"),
            bistable_n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )

    threshold_cols = [
        *CONDITION_COLUMNS,
        "active_up_threshold_mean",
        "active_down_threshold_mean",
        "active_hysteresis_width_mean",
    ]
    existing_threshold_cols = [column for column in threshold_cols if column in threshold_summary.columns]
    regions = pivot.merge(
        threshold_summary[existing_threshold_cols],
        on=CONDITION_COLUMNS,
        how="left",
    )
    regions = regions.merge(
        bistable_fraction,
        on=[*CONDITION_COLUMNS, "input_amplitude"],
        how="left",
    )

    regions["strongly_oscillatory_fraction_max"] = regions[
        ["strongly_oscillatory_fraction_up", "strongly_oscillatory_fraction_down"]
    ].max(axis=1)
    regions["is_oscillatory_region"] = (
        regions["strongly_oscillatory_fraction_max"] >= float(oscillatory_fraction_threshold)
    )

    regions["bistable_fraction"] = pd.to_numeric(regions["bistable_fraction"], errors="coerce").fillna(0.0)
    regions["is_bistable_region"] = regions["bistable_fraction"] >= float(bistable_fraction_threshold)

    regions["phase_region"] = np.select(
        [regions["is_oscillatory_region"], regions["is_bistable_region"]],
        ["oscillatory", "bistable"],
        default="monostable",
    )
    regions["phase_region_code"] = regions["phase_region"].map(
        {name: idx for idx, name in enumerate(BIFURCATION_REGION_ORDER)}
    )
    return regions


def pivot_for_grid(df: pd.DataFrame, value_column: str, *, row_column="j_gain", col_column="g_gain"):
    pivot = df.pivot(index=row_column, columns=col_column, values=value_column)
    pivot = pivot.sort_index().sort_index(axis=1)
    return pivot


def dataframe_map(df: pd.DataFrame, func):
    if hasattr(df, "map"):
        return df.map(func)
    return df.applymap(func)


def plot_numeric_heatmap(
    df: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    output_path: Path,
    row_label: str = "J global normalization gain",
    col_label: str = "g inhibitory-input normalization gain",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_label: str | None = None,
    dpi: int = 180,
) -> None:
    if value_column not in df.columns:
        return
    pivot = pivot_for_grid(df, value_column)
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(7.0, 5.6), constrained_layout=True)
    image = ax.imshow(pivot.to_numpy(dtype=float), origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(col_label)
    ax.set_ylabel(row_label)
    ax.set_xticks(np.arange(pivot.shape[1]), [format_axis_value(v) for v in pivot.columns])
    ax.set_yticks(np.arange(pivot.shape[0]), [format_axis_value(v) for v in pivot.index])

    for yi in range(pivot.shape[0]):
        for xi in range(pivot.shape[1]):
            value = pivot.iat[yi, xi]
            if pd.notna(value):
                ax.text(xi, yi, format_metric_value(value), ha="center", va="center", fontsize=8, color="black")

    cbar = fig.colorbar(image, ax=ax, shrink=0.9)
    cbar.set_label(colorbar_label or pretty_label(value_column))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def metric_color_limits(column: str, topology: str | None = None) -> tuple[float | None, float | None]:
    key = column.lower()
    topology_key = "" if topology is None else str(topology).lower()
    relative_range_metrics = {
        "separability_index",
        "memory_capacity_mean_r2",
        "memory_capacity_norm_delay_bound",
    }
    if key in relative_range_metrics:
        return None, None
    if "accuracy" in key or "norm" in key or "fraction" in key or "frac" in key:
        return 0.0, 1.0
    if "r2" in key:
        return 0.0, 1.0
    if topology_key == "fixed" and key in {"memory_capacity_mean_r2", "memory_capacity_norm_delay_bound"}:
        return None, None
    if "pop_spec_entropy_norm" in key:
        return 0.0, 1.0
    return None, None


def regime_metric_cmap(column: str) -> str:
    key = column.lower()
    if "fano" in key:
        return "YlGnBu"
    if "rate_mean" in key or "psd_peak_freq" in key:
        return "plasma"
    if "psd" in key or "spec_entropy" in key:
        return "cividis"
    if "synchrony" in key:
        return "magma"
    if "corr" in key:
        return "coolwarm"
    return "viridis"


def plot_categorical_heatmap(
    df: pd.DataFrame,
    *,
    mode_column: str,
    fraction_column: str,
    categories: list[str],
    colors: dict[str, str],
    title: str,
    output_path: Path,
    row_label: str = "J global normalization gain",
    col_label: str = "g inhibitory-input normalization gain",
    dpi: int = 180,
    hatch_fraction_column: str | None = None,
    hatch_threshold: float = 0.5,
    hatch_label: str | None = None,
) -> None:
    if mode_column not in df.columns:
        return

    plot_df = df.copy()
    plot_df[mode_column] = plot_df[mode_column].fillna("NA").astype(str)
    category_to_code = {category: idx for idx, category in enumerate(categories)}
    category_to_code["NA"] = len(categories)
    ordered_categories = [*categories, "NA"]

    pivot_cat = pivot_for_grid(plot_df, mode_column)
    pivot_frac = pivot_for_grid(plot_df, fraction_column) if fraction_column in plot_df.columns else None
    pivot_hatch = None
    if hatch_fraction_column and hatch_fraction_column in plot_df.columns:
        pivot_hatch = pivot_for_grid(plot_df, hatch_fraction_column).reindex(
            index=pivot_cat.index,
            columns=pivot_cat.columns,
        )
    numeric = dataframe_map(pivot_cat, lambda value: category_to_code.get(str(value), category_to_code["NA"]))

    cmap = ListedColormap([colors.get(category, "#F0F0F0") for category in ordered_categories])
    norm = BoundaryNorm(np.arange(-0.5, len(ordered_categories) + 0.5, 1.0), cmap.N)

    fig, ax = plt.subplots(figsize=(7.0, 5.6), constrained_layout=True)
    ax.imshow(numeric.to_numpy(dtype=float), origin="lower", aspect="auto", cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.set_xlabel(col_label)
    ax.set_ylabel(row_label)
    ax.set_xticks(np.arange(pivot_cat.shape[1]), [format_axis_value(v) for v in pivot_cat.columns])
    ax.set_yticks(np.arange(pivot_cat.shape[0]), [format_axis_value(v) for v in pivot_cat.index])

    has_hatch = False
    if pivot_hatch is not None:
        for yi in range(pivot_cat.shape[0]):
            for xi in range(pivot_cat.shape[1]):
                value = pivot_hatch.iat[yi, xi]
                if pd.notna(value) and float(value) >= float(hatch_threshold):
                    ax.add_patch(
                        Rectangle(
                            (xi - 0.5, yi - 0.5),
                            1.0,
                            1.0,
                            facecolor="none",
                            edgecolor=(0.0, 0.0, 0.0, 0.45),
                            hatch="/",
                            linewidth=0.75,
                            alpha=0.55,
                        )
                    )
                    has_hatch = True

    for yi in range(pivot_cat.shape[0]):
        for xi in range(pivot_cat.shape[1]):
            label = str(pivot_cat.iat[yi, xi])
            if pivot_frac is not None and pd.notna(pivot_frac.iat[yi, xi]):
                label = f"{label}\n{float(pivot_frac.iat[yi, xi]):.0%}"
            ax.text(xi, yi, label, ha="center", va="center", fontsize=8, color="black")

    handles = [Patch(facecolor=colors.get(category, "#F0F0F0"), edgecolor="black", label=category) for category in ordered_categories]
    if has_hatch and hatch_label:
        handles.append(Patch(facecolor="white", edgecolor=(0.0, 0.0, 0.0, 0.45), hatch="//////", label=hatch_label, alpha=0.55))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_phase_slice(
    phase_summary: pd.DataFrame,
    *,
    topology: str,
    direction: str,
    fixed_column: str,
    varying_column: str,
    fixed_value: float,
    output_path: Path,
    axis_labels: dict[str, dict[str, str]],
    dpi: int = 180,
) -> None:
    subset = phase_summary[
        (phase_summary["topology"] == topology)
        & (phase_summary["sweep_direction"] == direction)
        & np.isclose(phase_summary[fixed_column].astype(float), float(fixed_value))
    ].copy()
    if subset.empty:
        return

    row_column = varying_column
    col_column = "input_amplitude"
    mode_column = "phase_mode"
    fraction_column = "phase_mode_fraction"
    category_to_code = {category: idx for idx, category in enumerate(BIFURCATION_PHASE_ORDER)}
    category_to_code["NA"] = len(BIFURCATION_PHASE_ORDER)
    ordered_categories = [*BIFURCATION_PHASE_ORDER, "NA"]

    pivot_cat = subset.pivot(index=row_column, columns=col_column, values=mode_column).sort_index().sort_index(axis=1)
    pivot_frac = subset.pivot(index=row_column, columns=col_column, values=fraction_column).reindex(index=pivot_cat.index, columns=pivot_cat.columns)
    numeric = dataframe_map(pivot_cat, lambda value: category_to_code.get(str(value), category_to_code["NA"]))

    cmap = ListedColormap([BIFURCATION_PHASE_COLORS.get(category, "#F0F0F0") for category in ordered_categories])
    norm = BoundaryNorm(np.arange(-0.5, len(ordered_categories) + 0.5, 1.0), cmap.N)

    fig, ax = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)
    ax.imshow(numeric.to_numpy(dtype=float), origin="lower", aspect="auto", cmap=cmap, norm=norm)
    fixed_label = axis_labels[fixed_column]["short"]
    varying_label = axis_labels[varying_column]["label"]
    ax.set_title(
        f"{str(topology).capitalize()} bifurcation phase, {direction}, "
        f"{fixed_label}={format_axis_value(fixed_value)}"
    )
    ax.set_xlabel("Input amplitude")
    ax.set_ylabel(varying_label)
    ax.set_xticks(np.arange(pivot_cat.shape[1]), [format_axis_value(v) for v in pivot_cat.columns])
    ax.set_yticks(np.arange(pivot_cat.shape[0]), [format_axis_value(v) for v in pivot_cat.index])

    for yi in range(pivot_cat.shape[0]):
        for xi in range(pivot_cat.shape[1]):
            label = str(pivot_cat.iat[yi, xi])
            if pd.notna(pivot_frac.iat[yi, xi]):
                label = f"{label}\n{float(pivot_frac.iat[yi, xi]):.0%}"
            ax.text(xi, yi, label, ha="center", va="center", fontsize=8)

    handles = [
        Patch(facecolor=BIFURCATION_PHASE_COLORS.get(category, "#F0F0F0"), edgecolor="black", label=category)
        for category in ordered_categories
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_phase_region_grid(
    phase_regions: pd.DataFrame,
    *,
    topology: str,
    varying_column: str,
    fixed_column: str,
    output_path: Path,
    axis_labels: dict[str, dict[str, str]],
    dpi: int = 180,
) -> None:
    topo_regions = phase_regions[phase_regions["topology"] == topology].copy()
    if topo_regions.empty:
        return

    varying_values = sorted(topo_regions[varying_column].dropna().unique())
    fixed_values = sorted(topo_regions[fixed_column].dropna().unique())
    amplitudes = sorted(topo_regions["input_amplitude"].dropna().unique())
    if not varying_values or not fixed_values or not amplitudes:
        return

    n_panels = len(fixed_values)
    n_panel_rows = 2 if n_panels >= 6 and n_panels % 2 == 0 else 1
    n_panel_cols = int(np.ceil(n_panels / n_panel_rows))
    fig, axes = plt.subplots(
        n_panel_rows,
        n_panel_cols,
        figsize=(3.25 * n_panel_cols, 4.3 * n_panel_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes_flat = axes.ravel()

    ordered = [*BIFURCATION_REGION_ORDER, "NA"]
    cmap = ListedColormap([BIFURCATION_REGION_COLORS[name] for name in ordered])
    norm = BoundaryNorm(np.arange(-0.5, len(ordered) + 0.5, 1.0), cmap.N)
    region_to_code = {name: idx for idx, name in enumerate(ordered)}

    for axis, fixed_value in zip(axes_flat, fixed_values):
        subset = topo_regions[np.isclose(topo_regions[fixed_column].astype(float), float(fixed_value))]
        pivot = (
            subset.pivot_table(
                index=varying_column,
                columns="input_amplitude",
                values="phase_region",
                aggfunc="first",
            )
            .reindex(index=varying_values, columns=amplitudes)
        )
        matrix = dataframe_map(
            pivot,
            lambda value: region_to_code.get(str(value), region_to_code["NA"]),
        ).to_numpy(dtype=float)
        axis.imshow(matrix, origin="lower", aspect="auto", cmap=cmap, norm=norm)

        fixed_label = axis_labels[fixed_column]["short"]
        axis.set_title(f"{fixed_label}={format_axis_value(fixed_value)}", fontsize=10)
        axis.set_xlabel("Input amplitude")
        axis.set_xticks(range(len(amplitudes)))
        axis.set_xticklabels([format_axis_value(value) for value in amplitudes], rotation=45, ha="right", fontsize=8)

    for axis in axes_flat[len(fixed_values):]:
        axis.set_visible(False)

    varying_label = axis_labels[varying_column]["label"]
    fixed_label = axis_labels[fixed_column]["label"]
    for axis in axes[:, 0]:
        axis.set_ylabel(varying_label)
        axis.set_yticks(range(len(varying_values)))
        axis.set_yticklabels([format_axis_value(value) for value in varying_values], fontsize=8)

    fig.suptitle(
        f"{str(topology).capitalize()}: phase regions in input amplitude vs {varying_label}\n"
        f"Panels fix {fixed_label}; oscillatory > bistable > monostable",
        fontsize=13,
        y=0.98,
    )
    legend_handles = [
        Patch(facecolor=BIFURCATION_REGION_COLORS[name], edgecolor="none", label=name)
        for name in BIFURCATION_REGION_ORDER
    ]
    fig.legend(handles=legend_handles, loc="upper right", frameon=False)
    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def save_summary_md(
    output_dir: Path,
    *,
    regime_rows: pd.DataFrame,
    phase_summary: pd.DataFrame,
    numeric_tables: dict[str, pd.DataFrame],
    sweep_name: str,
    axis_labels: dict[str, dict[str, str]],
    brunel_thresholds: dict[str, float],
    using_saved_brunel_labels: bool,
    brunel_corr_only_hatch_threshold: float,
) -> None:
    lines = [
        "# Grid Analysis",
        "",
        "Categorical maps use seed-majority labels. The text in each cell is the majority class and the fraction of seeds in that class.",
        "",
        "## Sweep Axes",
        "",
        f"- Sweep: {sweep_name}",
        f"- Heatmap rows: {axis_labels['j_gain']['label']} (`j_gain` column)",
        f"- Heatmap columns: {axis_labels['g_gain']['label']} (`g_gain` column)",
        "",
        "## Brunel Classification",
        "",
        f"- Source: {'saved simulation labels' if using_saved_brunel_labels else 'recomputed from stored observation metrics'}",
        f"- Brunel map hatch: `brunel.corr_only_synchronous_fraction >= {brunel_corr_only_hatch_threshold:g}`",
    ]
    if not using_saved_brunel_labels:
        for name, value in brunel_thresholds.items():
            lines.append(f"- {name}: {value:g}")

    lines.extend(
        [
            "",
            "## Input Counts",
            "",
            f"- Regime rows: {len(regime_rows)}",
            f"- Bifurcation phase rows: {len(phase_summary)}",
        ]
    )
    for name, table in numeric_tables.items():
        lines.append(f"- {name}: {len(table)} summary rows")

    if "topology" in regime_rows.columns:
        lines.extend(["", "## Seeds Per Topology", ""])
        seed_counts = regime_rows.groupby("topology")["seed"].nunique()
        for topology, count in seed_counts.items():
            lines.append(f"- {topology}: {int(count)} seeds")

    lines.extend(
        [
            "",
            "## Bifurcation Phase Labels",
            "",
            "- Q: inactive/quiescent below the active-rate threshold",
            "- A: active but not strongly oscillatory",
            "- SO: strongly oscillatory",
            "",
            "## Bifurcation Region Labels",
            "",
            "- monostable: not in a ramp-dependent active-threshold window and not oscillatory",
            "- bistable: enough seeds have input amplitude between that seed's ramp-up and ramp-down active thresholds",
            "- oscillatory: strongly oscillatory in at least one ramp direction",
            "",
            "## Persistence/Regeneration Labels",
            "",
            "- PR: persistent regenerative",
            "- PN: persistent non-regenerative",
            "- ER: evoked/transient regenerative",
            "- EN: evoked/transient non-regenerative",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    regime = read_table(results_dir, "regime_rows")
    separation = read_table(results_dir, "separation_summary")
    memory = read_table(results_dir, "memory_summary")
    generalization = read_optional_table(results_dir, "generalization_summary")
    bifurcation_rows = read_table(results_dir, "bifurcation_rows")
    bifurcation_summary = read_table(results_dir, "bifurcation_summary")

    for df in (regime, separation, memory, generalization, bifurcation_rows, bifurcation_summary):
        ensure_numeric(df, ["j_gain", "g_gain", "seed", "input_amplitude"])
    memory = add_memory_first_negative_delay_metrics(
        memory,
        fallback_delay_bin_ms=args.fallback_memory_delay_bin_ms,
    )
    raw_tables = [regime, separation, memory, generalization, bifurcation_rows, bifurcation_summary]
    axis_labels = infer_grid_axis_labels(
        raw_tables,
        row_override=args.grid_row_label,
        col_override=args.grid_column_label,
    )
    sweep_name = infer_sweep_name(raw_tables)

    if not args.use_saved_brunel_labels:
        regime = recompute_brunel_labels(
            regime,
            cv_threshold=args.brunel_cv_threshold,
            fano_threshold=args.brunel_fano_threshold,
            corr_threshold=args.brunel_corr_threshold,
            peak_ratio_threshold=args.brunel_peak_ratio_threshold,
            entropy_norm_threshold=args.brunel_entropy_norm_threshold,
            bursty_fano_threshold=args.brunel_bursty_fano_threshold,
        )
    regime["pr_er_class"] = derive_persistence_regen_code(regime)
    if "brunel.synchronous_by_corr" not in regime.columns:
        corr = pd.to_numeric(regime.get("observation.mean_noise_corr_50ms", 0.0), errors="coerce").fillna(0.0)
        regime["brunel.synchronous_by_corr"] = corr > float(args.brunel_corr_threshold)
    if "brunel.oscillatory" not in regime.columns:
        peak_ratio = pd.to_numeric(regime.get("observation.psd_peak_ratio", 0.0), errors="coerce").fillna(0.0)
        entropy_norm = pd.to_numeric(
            regime.get("observation.pop_spec_entropy_norm", np.inf),
            errors="coerce",
        ).fillna(np.inf)
        regime["brunel.oscillatory"] = (
            ((peak_ratio > float(args.brunel_peak_ratio_threshold)) & (entropy_norm < float(args.brunel_entropy_norm_threshold)))
            | (peak_ratio > float(args.brunel_peak_ratio_threshold + 100.0))
        )
    synchronous_class = regime["brunel.brunel_class"].fillna("").astype(str).isin(["SI", "SR"])
    regime["brunel.corr_only_synchronous"] = (
        synchronous_class
        & coerce_bool(regime["brunel.synchronous_by_corr"])
        & ~coerce_bool(regime["brunel.oscillatory"])
    )
    bifurcation_rows["phase"] = derive_bifurcation_phase(bifurcation_rows)

    brunel_summary = majority_summary(
        regime,
        value_column="brunel.brunel_class",
        categories=BRUNEL_ORDER,
    )
    regime["brunel.corr_only_synchronous_numeric"] = coerce_bool(regime["brunel.corr_only_synchronous"]).astype(float)
    corr_only_summary = (
        regime.groupby(CONDITION_COLUMNS, dropna=False)["brunel.corr_only_synchronous_numeric"]
        .mean()
        .reset_index()
        .rename(columns={"brunel.corr_only_synchronous_numeric": "brunel.corr_only_synchronous_fraction"})
    )
    brunel_summary = brunel_summary.merge(corr_only_summary, on=CONDITION_COLUMNS, how="left")
    pr_er_summary = majority_summary(
        regime,
        value_column="pr_er_class",
        categories=PERSISTENCE_REGEN_ORDER,
    )

    regime_metric_columns = []
    for column in [args.fano_column, *args.regime_metric_columns]:
        if column not in regime_metric_columns:
            regime_metric_columns.append(column)

    fano_summary = numeric_summary(regime, value_columns=[args.fano_column])
    regime_metric_summary = numeric_summary(regime, value_columns=regime_metric_columns)
    separation_summary = numeric_summary(separation, value_columns=args.separation_columns)
    memory_summary = numeric_summary(memory, value_columns=args.memory_columns)
    generalization_summary = numeric_summary(generalization, value_columns=args.generalization_columns)

    phase_summary = majority_summary(
        bifurcation_rows,
        value_column="phase",
        categories=BIFURCATION_PHASE_ORDER,
        group_columns=[*CONDITION_COLUMNS, "sweep_direction", "input_amplitude"],
    )
    phase_summary = phase_summary.rename(
        columns={
            "phase_mode": "phase_mode",
            "phase_mode_fraction": "phase_mode_fraction",
        }
    )
    phase_numeric = (
        bifurcation_rows.assign(
            is_active_numeric=coerce_bool(bifurcation_rows["is_active"]).astype(float),
            is_strongly_oscillatory_numeric=coerce_bool(bifurcation_rows["is_strongly_oscillatory"]).astype(float),
        )
        .groupby([*CONDITION_COLUMNS, "sweep_direction", "input_amplitude"], dropna=False)
        .agg(
            active_fraction=("is_active_numeric", "mean"),
            strongly_oscillatory_fraction=("is_strongly_oscillatory_numeric", "mean"),
            rate_mean_Hz_mean=("rate_mean_Hz", "mean"),
            rate_mean_Hz_std=("rate_mean_Hz", "std"),
            n_seeds_numeric=("seed", "nunique"),
        )
        .reset_index()
    )
    phase_summary = phase_summary.merge(
        phase_numeric,
        on=[*CONDITION_COLUMNS, "sweep_direction", "input_amplitude"],
        how="left",
    )

    threshold_columns = [
        "active_up_threshold",
        "active_down_threshold",
        "active_hysteresis_width",
        "strong_oscillatory_up_threshold",
        "strong_oscillatory_down_threshold",
        "strong_oscillatory_hysteresis_width",
        "rate_hysteresis_area",
    ]
    threshold_summary = numeric_summary(bifurcation_summary, value_columns=threshold_columns)
    phase_regions = build_bifurcation_phase_regions(
        phase_summary,
        bifurcation_summary,
        threshold_summary,
        oscillatory_fraction_threshold=args.phase_region_osc_fraction,
        bistable_fraction_threshold=args.phase_region_bistable_fraction,
        bistable_tolerance=args.phase_region_bistable_tolerance,
    )

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    brunel_summary.to_csv(tables_dir / "brunel_class_seed_summary.csv", index=False)
    pr_er_summary.to_csv(tables_dir / "persistence_regeneration_seed_summary.csv", index=False)
    fano_summary.to_csv(tables_dir / "fano_seed_summary.csv", index=False)
    regime_metric_summary.to_csv(tables_dir / "regime_metric_seed_summary.csv", index=False)
    separation_summary.to_csv(tables_dir / "separation_seed_summary.csv", index=False)
    memory_summary.to_csv(tables_dir / "memory_seed_summary.csv", index=False)
    generalization_summary.to_csv(tables_dir / "generalization_seed_summary.csv", index=False)
    phase_summary.to_csv(tables_dir / "bifurcation_phase_seed_summary.csv", index=False)
    threshold_summary.to_csv(tables_dir / "bifurcation_threshold_seed_summary.csv", index=False)
    phase_regions.to_csv(tables_dir / "bifurcation_phase_regions.csv", index=False)

    figures_dir = output_dir / "figures"
    for topology in sorted(regime["topology"].dropna().unique()):
        topology_axis_labels = axis_labels_for_topology(
            axis_labels,
            sweep_name=sweep_name,
            topology=str(topology),
        )
        brunel_top = brunel_summary[brunel_summary["topology"] == topology]
        plot_categorical_heatmap(
            brunel_top,
            mode_column="brunel.brunel_class_mode",
            fraction_column="brunel.brunel_class_mode_fraction",
            categories=BRUNEL_ORDER,
            colors=BRUNEL_COLORS,
            title=f"{str(topology).capitalize()}: Brunel class majority over seeds",
            output_path=figures_dir / "regime_maps" / f"{topology}_brunel_class_map.png",
            row_label=topology_axis_labels["j_gain"]["label"],
            col_label=topology_axis_labels["g_gain"]["label"],
            dpi=args.dpi,
            hatch_fraction_column="brunel.corr_only_synchronous_fraction",
            hatch_threshold=args.brunel_corr_only_hatch_threshold,
            hatch_label="~O",
        )

        pr_top = pr_er_summary[pr_er_summary["topology"] == topology]
        plot_categorical_heatmap(
            pr_top,
            mode_column="pr_er_class_mode",
            fraction_column="pr_er_class_mode_fraction",
            categories=PERSISTENCE_REGEN_ORDER,
            colors=PERSISTENCE_REGEN_COLORS,
            title=f"{str(topology).capitalize()}: persistence/regeneration majority over seeds",
            output_path=figures_dir / "regime_maps" / f"{topology}_persistence_regeneration_map.png",
            row_label=topology_axis_labels["j_gain"]["label"],
            col_label=topology_axis_labels["g_gain"]["label"],
            dpi=args.dpi,
        )

        regime_topology_dir = figures_dir / "regime_maps" / safe_token(str(topology))
        regime_metric_top = regime_metric_summary[regime_metric_summary["topology"] == topology]
        for column in regime_metric_columns:
            value_column = f"{column}_mean"
            if value_column not in regime_metric_top.columns:
                continue
            output_name = "fano" if column == args.fano_column else safe_token(column.replace("observation.", ""))
            vmin, vmax = metric_color_limits(column, topology)
            plot_numeric_heatmap(
                regime_metric_top,
                value_column=value_column,
                title=f"{str(topology).capitalize()}: {metric_title(column)}",
                output_path=regime_topology_dir / f"{topology}_{output_name}_heatmap.png",
                row_label=topology_axis_labels["j_gain"]["label"],
                col_label=topology_axis_labels["g_gain"]["label"],
                cmap=regime_metric_cmap(column),
                vmin=vmin,
                vmax=vmax,
                colorbar_label=pretty_label(column.replace("observation.", "")),
                dpi=args.dpi,
            )

        for column in args.separation_columns:
            sep_top = separation_summary[separation_summary["topology"] == topology]
            plot_numeric_heatmap(
                sep_top,
                value_column=f"{column}_mean",
                title=f"{str(topology).capitalize()}: {metric_title(column)}",
                output_path=figures_dir / "reservoir_maps" / f"{topology}_{column}_heatmap.png",
                row_label=topology_axis_labels["j_gain"]["label"],
                col_label=topology_axis_labels["g_gain"]["label"],
                cmap="viridis",
                vmin=metric_color_limits(column, topology)[0],
                vmax=metric_color_limits(column, topology)[1],
                colorbar_label=pretty_label(column),
                dpi=args.dpi,
            )

        for column in args.memory_columns:
            mem_top = memory_summary[memory_summary["topology"] == topology]
            vmin, vmax = metric_color_limits(column, topology)
            plot_numeric_heatmap(
                mem_top,
                value_column=f"{column}_mean",
                title=f"{str(topology).capitalize()}: {metric_title(column)}",
                output_path=figures_dir / "reservoir_maps" / f"{topology}_{column}_heatmap.png",
                row_label=topology_axis_labels["j_gain"]["label"],
                col_label=topology_axis_labels["g_gain"]["label"],
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                colorbar_label=pretty_label(column),
                dpi=args.dpi,
            )

        for column in args.generalization_columns:
            gen_top = generalization_summary[generalization_summary["topology"] == topology]
            vmin, vmax = metric_color_limits(column, topology)
            plot_numeric_heatmap(
                gen_top,
                value_column=f"{column}_mean",
                title=f"{str(topology).capitalize()}: {metric_title(column)}",
                output_path=figures_dir / "generalization_maps" / f"{topology}_{column}_heatmap.png",
                row_label=topology_axis_labels["j_gain"]["label"],
                col_label=topology_axis_labels["g_gain"]["label"],
                cmap="cividis",
                vmin=vmin,
                vmax=vmax,
                colorbar_label=pretty_label(column),
                dpi=args.dpi,
            )

        threshold_top = threshold_summary[threshold_summary["topology"] == topology]
        for column in threshold_columns:
            plot_numeric_heatmap(
                threshold_top,
                value_column=f"{column}_mean",
                title=f"{str(topology).capitalize()}: {metric_title(column)}",
                output_path=figures_dir / "bifurcation_thresholds" / f"{topology}_{column}_heatmap.png",
                row_label=topology_axis_labels["j_gain"]["label"],
                col_label=topology_axis_labels["g_gain"]["label"],
                cmap="plasma",
                colorbar_label=pretty_label(column),
                dpi=args.dpi,
            )

        phase_top = phase_summary[phase_summary["topology"] == topology]
        directions = sorted(phase_top["sweep_direction"].dropna().unique())
        row_token = topology_axis_labels["j_gain"]["token"]
        col_token = topology_axis_labels["g_gain"]["token"]
        if args.phase_slices in ("j", "both"):
            for direction in directions:
                for fixed_j in sorted(phase_top["j_gain"].dropna().unique()):
                    plot_phase_slice(
                        phase_summary,
                        topology=topology,
                        direction=direction,
                        fixed_column="j_gain",
                        varying_column="g_gain",
                        fixed_value=float(fixed_j),
                        axis_labels=topology_axis_labels,
                        output_path=figures_dir
                        / "bifurcation_phase_slices"
                        / f"{topology}_{direction}_fixed_{row_token}_{format_axis_value(fixed_j)}.png",
                        dpi=args.dpi,
                    )

        if args.phase_slices in ("g", "both"):
            for direction in directions:
                for fixed_g in sorted(phase_top["g_gain"].dropna().unique()):
                    plot_phase_slice(
                        phase_summary,
                        topology=topology,
                        direction=direction,
                        fixed_column="g_gain",
                        varying_column="j_gain",
                        fixed_value=float(fixed_g),
                        axis_labels=topology_axis_labels,
                        output_path=figures_dir
                        / "bifurcation_phase_slices"
                        / f"{topology}_{direction}_fixed_{col_token}_{format_axis_value(fixed_g)}.png",
                        dpi=args.dpi,
                    )

        plot_phase_region_grid(
            phase_regions,
            topology=topology,
            varying_column="g_gain",
            fixed_column="j_gain",
            axis_labels=topology_axis_labels,
            output_path=figures_dir / "bifurcation_phase_regions" / f"{topology}_phase_regions_vs_{col_token}.png",
            dpi=args.dpi,
        )
        plot_phase_region_grid(
            phase_regions,
            topology=topology,
            varying_column="j_gain",
            fixed_column="g_gain",
            axis_labels=topology_axis_labels,
            output_path=figures_dir / "bifurcation_phase_regions" / f"{topology}_phase_regions_vs_{row_token}.png",
            dpi=args.dpi,
        )

    save_summary_md(
        output_dir,
        regime_rows=regime,
        phase_summary=phase_summary,
        sweep_name=sweep_name,
        axis_labels=axis_labels,
        brunel_thresholds={
            "CV threshold": args.brunel_cv_threshold,
            "Fano threshold": args.brunel_fano_threshold,
            "Correlation threshold": args.brunel_corr_threshold,
            "PSD peak-ratio threshold": args.brunel_peak_ratio_threshold,
            "Spectral-entropy norm threshold": args.brunel_entropy_norm_threshold,
            "Bursty Fano threshold": args.brunel_bursty_fano_threshold,
        },
        using_saved_brunel_labels=args.use_saved_brunel_labels,
        brunel_corr_only_hatch_threshold=args.brunel_corr_only_hatch_threshold,
        numeric_tables={
            "Fano": fano_summary,
            "Regime metrics": regime_metric_summary,
            "Separation": separation_summary,
            "Memory": memory_summary,
            "Generalization": generalization_summary,
            "Bifurcation thresholds": threshold_summary,
            "Bifurcation phase regions": phase_regions,
        },
    )

    manifest = {
        "results_dir": str(results_dir),
        "output_dir": str(output_dir),
        "tables": sorted(str(path.relative_to(output_dir)) for path in tables_dir.glob("*.csv")),
        "figures": sorted(str(path.relative_to(output_dir)) for path in figures_dir.rglob("*.png")),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote analysis outputs to {output_dir}")
    print(f"Figures: {len(manifest['figures'])}")
    print(f"Tables: {len(manifest['tables'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
