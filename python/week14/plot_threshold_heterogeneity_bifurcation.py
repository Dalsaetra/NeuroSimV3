import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "week14" / "results" / "threshold_heterogeneity_bifurcation_sweep.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "week14" / "results" / "threshold_heterogeneity_bifurcation_plots"

PARAMETER_COLUMNS = ["variance_ss4_vt", "variance_b_vt"]
GROUP_COLUMNS = ["topology", *PARAMETER_COLUMNS]
CURVE_GROUP_COLUMNS = [*GROUP_COLUMNS, "sweep_direction", "input_amplitude"]
SEED_GROUP_COLUMNS = [*GROUP_COLUMNS, "topology_seed", "sweep_direction"]

CATEGORY_SPECS = {
    "strong": {
        "seed_column": "is_strongly_oscillatory",
        "fraction_column": "strong_fraction",
        "label": "strongly oscillatory",
        "color": "#1d3557",
    },
    "weak": {
        "seed_column": "is_weakly_oscillatory",
        "fraction_column": "weak_fraction",
        "label": "weakly oscillatory",
        "color": "#457b9d",
    },
    "sync_only": {
        "seed_column": "is_synchronous_only",
        "fraction_column": "sync_only_fraction",
        "label": "synchronous only",
        "color": "#e9c46a",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot and summarize threshold-heterogeneity bifurcation sweeps. "
            "Rate bifurcations use mean firing rate and oscillatory regions use "
            "PSD peak ratio / noise-correlation criteria."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input CSV from bifurcation_threshold_heterogeneity.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated plots and summary tables.",
    )
    parser.add_argument(
        "--rate-column",
        type=str,
        default="rate_mean_Hz",
        help="Metric column used to define the rate bifurcation threshold.",
    )
    parser.add_argument(
        "--rate-threshold-hz",
        type=float,
        default=1.0,
        help="Minimum mean firing rate used to declare the system active.",
    )
    parser.add_argument(
        "--psd-peak-ratio-threshold",
        type=float,
        default=200.0,
        help="Oscillatory criterion: psd_peak_ratio must exceed this value.",
    )
    parser.add_argument(
        "--strong-entropy-threshold",
        type=float,
        default=7.5,
        help="Strong oscillation criterion: pop_spec_entropy must be below this value.",
    )
    parser.add_argument(
        "--noise-corr-threshold",
        type=float,
        default=0.05,
        help="Synchronous-only criterion: mean_noise_corr_50ms or mean_noise_corr_10ms must exceed this value.",
    )
    parser.add_argument(
        "--osc-fraction-threshold",
        type=float,
        default=0.5,
        help="Seed fraction required to classify a curve-level condition as oscillatory.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="PNG resolution.")
    return parser.parse_args()


def load_results(input_path, psd_peak_ratio_threshold, strong_entropy_threshold, noise_corr_threshold):
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    required_columns = {
        "topology",
        "topology_seed",
        "sweep_direction",
        "input_amplitude",
        "variance_ss4_vt",
        "variance_b_vt",
        "psd_peak_ratio",
        "mean_noise_corr_50ms",
        "mean_noise_corr_10ms",
        "pop_spec_entropy",
    }
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    numeric_columns = [
        "topology_seed",
        "input_amplitude",
        "variance_ss4_vt",
        "variance_b_vt",
        "direction_step_index",
        "global_block_index",
        "rate_mean_Hz",
        "rate_mean_Hz_E",
        "rate_mean_Hz_I",
        "ISI_CV_mean",
        "participation_frac_total",
        "psd_peak_ratio",
        "mean_noise_corr_50ms",
        "mean_noise_corr_10ms",
        "pop_spec_entropy",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["sweep_direction"] = df["sweep_direction"].astype(str).str.lower()
    df["is_strongly_oscillatory"] = (
        (df["psd_peak_ratio"] > float(psd_peak_ratio_threshold))
        & (df["pop_spec_entropy"] < float(strong_entropy_threshold))
    )
    df["is_weakly_oscillatory"] = (
        (df["psd_peak_ratio"] > float(psd_peak_ratio_threshold))
        & (df["pop_spec_entropy"] >= float(strong_entropy_threshold))
    )
    df["is_oscillatory"] = df["is_strongly_oscillatory"] | df["is_weakly_oscillatory"]
    df["is_synchronous_only"] = (
        ~df["is_oscillatory"]
        & (
            (df["mean_noise_corr_50ms"] > float(noise_corr_threshold))
            | (df["mean_noise_corr_10ms"] > float(noise_corr_threshold))
        )
    )
    df["is_coordinated"] = (
        df["is_strongly_oscillatory"] | df["is_weakly_oscillatory"] | df["is_synchronous_only"]
    )
    for column in (
        "is_strongly_oscillatory",
        "is_weakly_oscillatory",
        "is_synchronous_only",
        "is_oscillatory",
        "is_coordinated",
    ):
        df[f"{column}_numeric"] = df[column].astype(float)
    return df


def aggregate_rate_curves(df, rate_column):
    if rate_column not in df.columns:
        raise ValueError(f"Requested rate column '{rate_column}' not found in CSV.")

    grouped = (
        df.groupby(CURVE_GROUP_COLUMNS, dropna=False)[rate_column]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": f"{rate_column}_mean",
                "std": f"{rate_column}_std",
                "count": "seed_count",
            }
        )
    )
    grouped[f"{rate_column}_std"] = grouped[f"{rate_column}_std"].fillna(0.0)
    return grouped


def aggregate_oscillation_curves(df):
    grouped = (
        df.groupby(CURVE_GROUP_COLUMNS, dropna=False)
        .agg(
            strong_fraction=("is_strongly_oscillatory_numeric", "mean"),
            strong_fraction_std=("is_strongly_oscillatory_numeric", "std"),
            strong_seed_count=("is_strongly_oscillatory_numeric", "sum"),
            weak_fraction=("is_weakly_oscillatory_numeric", "mean"),
            weak_fraction_std=("is_weakly_oscillatory_numeric", "std"),
            weak_seed_count=("is_weakly_oscillatory_numeric", "sum"),
            sync_only_fraction=("is_synchronous_only_numeric", "mean"),
            sync_only_fraction_std=("is_synchronous_only_numeric", "std"),
            sync_only_seed_count=("is_synchronous_only_numeric", "sum"),
            coordinated_fraction=("is_coordinated_numeric", "mean"),
            coordinated_fraction_std=("is_coordinated_numeric", "std"),
            coordinated_seed_count=("is_coordinated_numeric", "sum"),
            seed_count=("is_coordinated_numeric", "count"),
            psd_peak_ratio_mean=("psd_peak_ratio", "mean"),
            psd_peak_ratio_std=("psd_peak_ratio", "std"),
            mean_noise_corr_50ms_mean=("mean_noise_corr_50ms", "mean"),
            mean_noise_corr_50ms_std=("mean_noise_corr_50ms", "std"),
            mean_noise_corr_10ms_mean=("mean_noise_corr_10ms", "mean"),
            mean_noise_corr_10ms_std=("mean_noise_corr_10ms", "std"),
            pop_spec_entropy_mean=("pop_spec_entropy", "mean"),
            pop_spec_entropy_std=("pop_spec_entropy", "std"),
        )
        .reset_index()
    )
    for column in (
        "strong_fraction_std",
        "weak_fraction_std",
        "sync_only_fraction_std",
        "coordinated_fraction_std",
        "psd_peak_ratio_std",
        "mean_noise_corr_50ms_std",
        "mean_noise_corr_10ms_std",
        "pop_spec_entropy_std",
    ):
        grouped[column] = grouped[column].fillna(0.0)
    return grouped


def detect_direction_threshold(group_df, value_column, threshold_value):
    ordered = group_df.sort_values("input_amplitude")
    above = ordered[ordered[value_column] >= threshold_value]
    if above.empty:
        return pd.Series(
            {
                "threshold_found": False,
                "threshold_amplitude": np.nan,
                "threshold_signal_value": np.nan,
                "lowest_above_threshold_amplitude": np.nan,
                "highest_above_threshold_amplitude": np.nan,
            }
        )

    threshold_row = above.iloc[0]
    return pd.Series(
        {
            "threshold_found": True,
            "threshold_amplitude": float(threshold_row["input_amplitude"]),
            "threshold_signal_value": float(threshold_row[value_column]),
            "lowest_above_threshold_amplitude": float(above["input_amplitude"].min()),
            "highest_above_threshold_amplitude": float(above["input_amplitude"].max()),
        }
    )


def compute_curve_thresholds(summary_df, value_column, threshold_value, prefix):
    records = []
    grouping_columns = [*GROUP_COLUMNS, "sweep_direction"]
    for group_key, group_df in summary_df.groupby(grouping_columns, dropna=False, sort=False):
        threshold_row = detect_direction_threshold(
            group_df,
            value_column=value_column,
            threshold_value=threshold_value,
        ).to_dict()
        records.append(dict(zip(grouping_columns, group_key)) | threshold_row)

    thresholds = pd.DataFrame.from_records(records)
    if thresholds.empty:
        thresholds = pd.DataFrame(columns=[*grouping_columns])
    thresholds = thresholds.rename(
        columns={
            "threshold_found": f"{prefix}_threshold_found",
            "threshold_amplitude": f"{prefix}_threshold_amplitude",
            "threshold_signal_value": f"{prefix}_threshold_signal_value",
            "lowest_above_threshold_amplitude": f"{prefix}_lowest_above_threshold_amplitude",
            "highest_above_threshold_amplitude": f"{prefix}_highest_above_threshold_amplitude",
        }
    )
    return thresholds


def compute_seed_thresholds(df, value_column, threshold_value, prefix):
    records = []
    for group_key, group_df in df.groupby(SEED_GROUP_COLUMNS, dropna=False, sort=False):
        threshold_row = detect_direction_threshold(
            group_df,
            value_column=value_column,
            threshold_value=threshold_value,
        ).to_dict()
        records.append(dict(zip(SEED_GROUP_COLUMNS, group_key)) | threshold_row)

    seed_thresholds = pd.DataFrame.from_records(records)
    if seed_thresholds.empty:
        seed_thresholds = pd.DataFrame(columns=SEED_GROUP_COLUMNS)
    seed_thresholds = seed_thresholds.rename(
        columns={
            "threshold_found": f"seed_{prefix}_threshold_found",
            "threshold_amplitude": f"seed_{prefix}_threshold_amplitude",
            "threshold_signal_value": f"seed_{prefix}_threshold_signal_value",
            "lowest_above_threshold_amplitude": f"seed_{prefix}_lowest_above_threshold_amplitude",
            "highest_above_threshold_amplitude": f"seed_{prefix}_highest_above_threshold_amplitude",
        }
    )
    return seed_thresholds


def summarize_seed_thresholds(seed_thresholds, prefix):
    amp_column = f"seed_{prefix}_threshold_amplitude"
    summary = (
        seed_thresholds.groupby([*GROUP_COLUMNS, "sweep_direction"], dropna=False)
        .agg(
            seed_count=("topology_seed", "nunique"),
            seed_threshold_found_count=(amp_column, lambda s: int(s.notna().sum())),
            seed_threshold_mean=(amp_column, "mean"),
            seed_threshold_std=(amp_column, "std"),
            seed_threshold_min=(amp_column, "min"),
            seed_threshold_max=(amp_column, "max"),
        )
        .reset_index()
    )
    summary["seed_threshold_std"] = summary["seed_threshold_std"].fillna(0.0)
    summary = summary.rename(
        columns={
            "seed_threshold_found_count": f"seed_{prefix}_threshold_found_count",
            "seed_threshold_mean": f"seed_{prefix}_threshold_mean",
            "seed_threshold_std": f"seed_{prefix}_threshold_std",
            "seed_threshold_min": f"seed_{prefix}_threshold_min",
            "seed_threshold_max": f"seed_{prefix}_threshold_max",
        }
    )
    return summary


def summarize_seed_hysteresis(seed_thresholds, prefix):
    amp_column = f"seed_{prefix}_threshold_amplitude"
    pivot = (
        seed_thresholds.pivot_table(
            index=[*GROUP_COLUMNS, "topology_seed"],
            columns="sweep_direction",
            values=amp_column,
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    if "up" not in pivot.columns:
        pivot["up"] = np.nan
    if "down" not in pivot.columns:
        pivot["down"] = np.nan
    pivot[f"seed_{prefix}_hysteresis_width"] = pivot["up"] - pivot["down"]
    pivot = pivot.rename(
        columns={
            "up": f"seed_{prefix}_threshold_up",
            "down": f"seed_{prefix}_threshold_down",
        }
    )

    summary = (
        pivot.groupby(GROUP_COLUMNS, dropna=False)
        .agg(
            seed_hysteresis_count=("topology_seed", "nunique"),
            seed_hysteresis_mean=(f"seed_{prefix}_hysteresis_width", "mean"),
            seed_hysteresis_std=(f"seed_{prefix}_hysteresis_width", "std"),
        )
        .reset_index()
    )
    summary["seed_hysteresis_std"] = summary["seed_hysteresis_std"].fillna(0.0)
    summary = summary.rename(
        columns={
            "seed_hysteresis_count": f"seed_{prefix}_hysteresis_count",
            "seed_hysteresis_mean": f"seed_{prefix}_hysteresis_mean",
            "seed_hysteresis_std": f"seed_{prefix}_hysteresis_std",
        }
    )
    return pivot, summary


def build_transition_summary(curve_thresholds, seed_hysteresis_summary, prefix):
    amp_column = f"{prefix}_threshold_amplitude"
    pivot = (
        curve_thresholds.pivot_table(
            index=GROUP_COLUMNS,
            columns="sweep_direction",
            values=amp_column,
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    if "up" not in pivot.columns:
        pivot["up"] = np.nan
    if "down" not in pivot.columns:
        pivot["down"] = np.nan

    pivot = pivot.rename(
        columns={
            "up": f"curve_{prefix}_threshold_up",
            "down": f"curve_{prefix}_threshold_down",
        }
    )
    pivot[f"curve_{prefix}_hysteresis_width"] = (
        pivot[f"curve_{prefix}_threshold_up"] - pivot[f"curve_{prefix}_threshold_down"]
    )
    pivot = pivot.merge(seed_hysteresis_summary, on=GROUP_COLUMNS, how="left")
    return pivot


def build_phase_classification(rate_transition_summary, oscillation_summary, osc_fraction_threshold):
    oscillation_curve = oscillation_summary.copy()
    oscillation_curve["curve_is_strong"] = oscillation_curve["strong_fraction"] >= float(osc_fraction_threshold)
    oscillation_curve["curve_is_weak"] = oscillation_curve["weak_fraction"] >= float(osc_fraction_threshold)
    oscillation_curve["curve_is_sync_only"] = (
        oscillation_curve["sync_only_fraction"] >= float(osc_fraction_threshold)
    )

    pivot = (
        oscillation_curve.pivot_table(
            index=[*GROUP_COLUMNS, "input_amplitude"],
            columns="sweep_direction",
            values=[
                "curve_is_strong",
                "curve_is_weak",
                "curve_is_sync_only",
                "strong_fraction",
                "weak_fraction",
                "sync_only_fraction",
                "coordinated_fraction",
            ],
            aggfunc="first",
        )
        .reset_index()
    )
    pivot.columns = [
        "_".join(str(part) for part in col if str(part) != "").rstrip("_")
        if isinstance(col, tuple)
        else col
        for col in pivot.columns
    ]

    for column in (
        "curve_is_strong_up",
        "curve_is_strong_down",
        "curve_is_weak_up",
        "curve_is_weak_down",
        "curve_is_sync_only_up",
        "curve_is_sync_only_down",
        "strong_fraction_up",
        "strong_fraction_down",
        "weak_fraction_up",
        "weak_fraction_down",
        "sync_only_fraction_up",
        "sync_only_fraction_down",
        "coordinated_fraction_up",
        "coordinated_fraction_down",
    ):
        if column not in pivot.columns:
            pivot[column] = np.nan

    for column in (
        "curve_is_strong_up",
        "curve_is_strong_down",
        "curve_is_weak_up",
        "curve_is_weak_down",
        "curve_is_sync_only_up",
        "curve_is_sync_only_down",
    ):
        pivot[column] = pivot[column].fillna(False).astype(bool)
    for column in (
        "strong_fraction_up",
        "strong_fraction_down",
        "weak_fraction_up",
        "weak_fraction_down",
        "sync_only_fraction_up",
        "sync_only_fraction_down",
        "coordinated_fraction_up",
        "coordinated_fraction_down",
    ):
        pivot[column] = pd.to_numeric(pivot[column], errors="coerce")

    phase = pivot.merge(
        rate_transition_summary[
            [
                *GROUP_COLUMNS,
                "curve_rate_threshold_up",
                "curve_rate_threshold_down",
                "curve_rate_hysteresis_width",
            ]
        ],
        on=GROUP_COLUMNS,
        how="left",
    )

    phase["strong_any_direction"] = phase["curve_is_strong_up"] | phase["curve_is_strong_down"]
    phase["weak_any_direction"] = phase["curve_is_weak_up"] | phase["curve_is_weak_down"]
    phase["sync_only_any_direction"] = (
        phase["curve_is_sync_only_up"] | phase["curve_is_sync_only_down"]
    )
    phase["strong_max_fraction"] = phase[["strong_fraction_up", "strong_fraction_down"]].max(axis=1)
    phase["weak_max_fraction"] = phase[["weak_fraction_up", "weak_fraction_down"]].max(axis=1)
    phase["sync_only_max_fraction"] = phase[["sync_only_fraction_up", "sync_only_fraction_down"]].max(axis=1)
    phase["coordinated_max_fraction"] = phase[
        ["coordinated_fraction_up", "coordinated_fraction_down"]
    ].max(axis=1)

    phase["in_hysteresis_window"] = (
        phase["curve_rate_threshold_up"].notna()
        & phase["curve_rate_threshold_down"].notna()
        & (phase["curve_rate_threshold_up"] > phase["curve_rate_threshold_down"])
        & (phase["input_amplitude"] >= phase["curve_rate_threshold_down"])
        & (phase["input_amplitude"] < phase["curve_rate_threshold_up"])
    )

    phase["phase_label"] = np.select(
        [
            phase["strong_any_direction"],
            (~phase["strong_any_direction"]) & phase["weak_any_direction"],
            (~phase["strong_any_direction"]) & (~phase["weak_any_direction"]) & phase["sync_only_any_direction"],
            phase["in_hysteresis_window"],
        ],
        ["strongly oscillatory", "weakly oscillatory", "synchronous only", "bistable"],
        default="monostable",
    )
    phase["phase_code"] = phase["phase_label"].map(
        {
            "monostable": 0,
            "bistable": 1,
            "synchronous only": 2,
            "weakly oscillatory": 3,
            "strongly oscillatory": 4,
        }
    )
    return phase


def save_tables(
    output_dir,
    rate_summary,
    rate_curve_thresholds,
    rate_seed_thresholds,
    rate_seed_threshold_summary,
    rate_seed_hysteresis_by_seed,
    rate_transition_summary,
    oscillation_summary,
    category_outputs,
    phase_classification,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    rate_summary.to_csv(output_dir / "aggregated_rate_curves.csv", index=False)
    rate_curve_thresholds.to_csv(output_dir / "curve_bifurcation_thresholds.csv", index=False)
    rate_seed_thresholds.to_csv(output_dir / "seed_bifurcation_thresholds.csv", index=False)
    rate_seed_threshold_summary.to_csv(output_dir / "seed_bifurcation_threshold_summary.csv", index=False)
    rate_seed_hysteresis_by_seed.to_csv(output_dir / "seed_hysteresis_by_seed.csv", index=False)
    rate_transition_summary.to_csv(output_dir / "hysteresis_summary.csv", index=False)

    oscillation_summary.to_csv(output_dir / "aggregated_oscillation_curves.csv", index=False)
    for category_key, category_data in category_outputs.items():
        category_data["curve_thresholds"].to_csv(
            output_dir / f"curve_{category_key}_thresholds.csv",
            index=False,
        )
        category_data["seed_thresholds"].to_csv(
            output_dir / f"seed_{category_key}_thresholds.csv",
            index=False,
        )
        category_data["seed_threshold_summary"].to_csv(
            output_dir / f"seed_{category_key}_threshold_summary.csv",
            index=False,
        )
        category_data["seed_hysteresis_by_seed"].to_csv(
            output_dir / f"seed_{category_key}_hysteresis_by_seed.csv",
            index=False,
        )
        category_data["transition_summary"].to_csv(
            output_dir / f"{category_key}_hysteresis_summary.csv",
            index=False,
        )

    phase_classification.to_csv(output_dir / "phase_classification_by_condition_amplitude.csv", index=False)


def format_value(value):
    if pd.isna(value):
        return "NA"
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def draw_heatmap(ax, matrix, x_values, y_values, title, cmap, text_color_rule=None, vmin=None, vmax=None):
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("variance_b_vt")
    ax.set_ylabel("variance_ss4_vt")
    ax.set_xticks(range(len(x_values)))
    ax.set_yticks(range(len(y_values)))
    ax.set_xticklabels([format_value(v) for v in x_values], rotation=45, ha="right")
    ax.set_yticklabels([format_value(v) for v in y_values])

    for y_index in range(len(y_values)):
        for x_index in range(len(x_values)):
            value = matrix[y_index, x_index]
            text = format_value(value)
            if callable(text_color_rule):
                color = text_color_rule(value)
            else:
                color = "black"
            ax.text(x_index, y_index, text, ha="center", va="center", color=color, fontsize=8)
    return image


def plot_rate_grid(rate_summary, rate_curve_thresholds, topology, rate_column, rate_threshold_hz, output_dir, dpi):
    topo_rates = rate_summary[rate_summary["topology"] == topology].copy()
    topo_thresholds = rate_curve_thresholds[rate_curve_thresholds["topology"] == topology].copy()
    if topo_rates.empty:
        return

    ss4_values = sorted(topo_rates["variance_ss4_vt"].dropna().unique())
    b_values = sorted(topo_rates["variance_b_vt"].dropna().unique())
    amplitudes = sorted(topo_rates["input_amplitude"].dropna().unique())
    amplitude_to_x = {amp: idx for idx, amp in enumerate(amplitudes)}

    fig, axes = plt.subplots(
        len(ss4_values),
        len(b_values),
        figsize=(3.3 * len(b_values), 2.6 * len(ss4_values)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    colors = {"up": "#1f77b4", "down": "#d62728"}
    styles = {"up": "-", "down": "--"}
    legend_handles = []
    legend_labels = []

    for row_idx, ss4_value in enumerate(ss4_values):
        for col_idx, b_value in enumerate(b_values):
            ax = axes[row_idx][col_idx]
            subset = topo_rates[
                (topo_rates["variance_ss4_vt"] == ss4_value)
                & (topo_rates["variance_b_vt"] == b_value)
            ].copy()

            for direction in ("up", "down"):
                curve = subset[subset["sweep_direction"] == direction].sort_values("input_amplitude")
                if curve.empty:
                    continue

                x = [amplitude_to_x[value] for value in curve["input_amplitude"]]
                y = curve[f"{rate_column}_mean"].to_numpy()
                y_std = curve[f"{rate_column}_std"].to_numpy()

                line = ax.plot(
                    x,
                    y,
                    marker="o",
                    markersize=3.5,
                    linewidth=1.5,
                    color=colors[direction],
                    linestyle=styles[direction],
                    label=f"{direction} ramp",
                )[0]
                ax.fill_between(x, np.maximum(y - y_std, 0.0), y + y_std, color=colors[direction], alpha=0.12)

                if f"{direction} ramp" not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(f"{direction} ramp")

                threshold_row = topo_thresholds[
                    (topo_thresholds["variance_ss4_vt"] == ss4_value)
                    & (topo_thresholds["variance_b_vt"] == b_value)
                    & (topo_thresholds["sweep_direction"] == direction)
                ]
                if not threshold_row.empty and pd.notna(threshold_row.iloc[0]["rate_threshold_amplitude"]):
                    threshold_amp = float(threshold_row.iloc[0]["rate_threshold_amplitude"])
                    ax.axvline(
                        amplitude_to_x[threshold_amp],
                        color=colors[direction],
                        linestyle=":",
                        linewidth=1.0,
                        alpha=0.7,
                    )

            ax.axhline(rate_threshold_hz, color="black", linewidth=1.0, linestyle="-.")
            ax.set_title(f"E={format_value(ss4_value)}, I={format_value(b_value)}", fontsize=9)
            ax.grid(alpha=0.18, linewidth=0.5)

            if row_idx == len(ss4_values) - 1:
                ax.set_xticks(range(len(amplitudes)))
                ax.set_xticklabels([format_value(value) for value in amplitudes], rotation=45, ha="right", fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(f"{rate_column}\nHz", fontsize=9)

    fig.suptitle(
        f"{topology}: mean-rate curves across threshold heterogeneity\n"
        f"Threshold = lowest amplitude with {rate_column} >= {rate_threshold_hz:.2f} Hz",
        fontsize=13,
        y=0.995,
    )
    fig.legend(legend_handles, legend_labels, loc="upper right", frameon=False)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.965))
    fig.savefig(output_dir / f"rate_vs_amplitude_grid_{topology}.png", dpi=dpi)
    plt.close(fig)


def plot_category_fraction_grid(
    oscillation_summary,
    category_curve_thresholds,
    category_key,
    topology,
    category_fraction_threshold,
    output_dir,
    dpi,
):
    category_spec = CATEGORY_SPECS[category_key]
    category_label = category_spec["label"]
    fraction_column = category_spec["fraction_column"]
    topo_osc = oscillation_summary[oscillation_summary["topology"] == topology].copy()
    topo_thresholds = category_curve_thresholds[category_curve_thresholds["topology"] == topology].copy()
    if topo_osc.empty:
        return

    ss4_values = sorted(topo_osc["variance_ss4_vt"].dropna().unique())
    b_values = sorted(topo_osc["variance_b_vt"].dropna().unique())
    amplitudes = sorted(topo_osc["input_amplitude"].dropna().unique())
    amplitude_to_x = {amp: idx for idx, amp in enumerate(amplitudes)}

    fig, axes = plt.subplots(
        len(ss4_values),
        len(b_values),
        figsize=(3.3 * len(b_values), 2.6 * len(ss4_values)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    colors = {"up": category_spec["color"], "down": "#e76f51"}
    styles = {"up": "-", "down": "--"}
    legend_handles = []
    legend_labels = []

    for row_idx, ss4_value in enumerate(ss4_values):
        for col_idx, b_value in enumerate(b_values):
            ax = axes[row_idx][col_idx]
            subset = topo_osc[
                (topo_osc["variance_ss4_vt"] == ss4_value)
                & (topo_osc["variance_b_vt"] == b_value)
            ].copy()

            for direction in ("up", "down"):
                curve = subset[subset["sweep_direction"] == direction].sort_values("input_amplitude")
                if curve.empty:
                    continue

                x = [amplitude_to_x[value] for value in curve["input_amplitude"]]
                y = curve[fraction_column].to_numpy()
                y_std = curve[f"{fraction_column}_std"].to_numpy()

                line = ax.plot(
                    x,
                    y,
                    marker="o",
                    markersize=3.5,
                    linewidth=1.5,
                    color=colors[direction],
                    linestyle=styles[direction],
                    label=f"{direction} ramp",
                )[0]
                ax.fill_between(
                    x,
                    np.clip(y - y_std, 0.0, 1.0),
                    np.clip(y + y_std, 0.0, 1.0),
                    color=colors[direction],
                    alpha=0.12,
                )

                if f"{direction} ramp" not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(f"{direction} ramp")

                threshold_row = topo_thresholds[
                    (topo_thresholds["variance_ss4_vt"] == ss4_value)
                    & (topo_thresholds["variance_b_vt"] == b_value)
                    & (topo_thresholds["sweep_direction"] == direction)
                ]
                if not threshold_row.empty and pd.notna(threshold_row.iloc[0][f"{category_key}_threshold_amplitude"]):
                    threshold_amp = float(threshold_row.iloc[0][f"{category_key}_threshold_amplitude"])
                    ax.axvline(
                        amplitude_to_x[threshold_amp],
                        color=colors[direction],
                        linestyle=":",
                        linewidth=1.0,
                        alpha=0.7,
                    )

            ax.axhline(category_fraction_threshold, color="black", linewidth=1.0, linestyle="-.")
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"E={format_value(ss4_value)}, I={format_value(b_value)}", fontsize=9)
            ax.grid(alpha=0.18, linewidth=0.5)

            if row_idx == len(ss4_values) - 1:
                ax.set_xticks(range(len(amplitudes)))
                ax.set_xticklabels([format_value(value) for value in amplitudes], rotation=45, ha="right", fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(f"{category_label}\nseed fraction", fontsize=9)

    fig.suptitle(
        f"{topology}: {category_label} region across threshold heterogeneity\n"
        f"Curve threshold at fraction >= {category_fraction_threshold:.2f}",
        fontsize=13,
        y=0.995,
    )
    fig.legend(legend_handles, legend_labels, loc="upper right", frameon=False)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.965))
    fig.savefig(output_dir / f"{category_key}_vs_amplitude_grid_{topology}.png", dpi=dpi)
    plt.close(fig)


def plot_transition_heatmaps(transition_summary, topology, prefix, title_prefix, output_name, output_dir, dpi):
    topo_summary = transition_summary[transition_summary["topology"] == topology].copy()
    if topo_summary.empty:
        return

    ss4_values = sorted(topo_summary["variance_ss4_vt"].dropna().unique())
    b_values = sorted(topo_summary["variance_b_vt"].dropna().unique())

    def pivot_matrix(value_column):
        pivot = topo_summary.pivot_table(
            index="variance_ss4_vt",
            columns="variance_b_vt",
            values=value_column,
            aggfunc="first",
        )
        pivot = pivot.reindex(index=ss4_values, columns=b_values)
        return pivot.to_numpy(dtype=float)

    up_matrix = pivot_matrix(f"curve_{prefix}_threshold_up")
    down_matrix = pivot_matrix(f"curve_{prefix}_threshold_down")
    hysteresis_matrix = pivot_matrix(f"curve_{prefix}_hysteresis_width")

    finite_thresholds = np.concatenate(
        [up_matrix[np.isfinite(up_matrix)], down_matrix[np.isfinite(down_matrix)]]
    )
    threshold_vmin = float(np.min(finite_thresholds)) if finite_thresholds.size else None
    threshold_vmax = float(np.max(finite_thresholds)) if finite_thresholds.size else None

    finite_hysteresis = hysteresis_matrix[np.isfinite(hysteresis_matrix)]
    if finite_hysteresis.size:
        h_abs = float(np.max(np.abs(finite_hysteresis)))
        hysteresis_vmin = -h_abs
        hysteresis_vmax = h_abs
    else:
        hysteresis_vmin = None
        hysteresis_vmax = None

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    threshold_text = lambda value: "white" if np.isfinite(value) else "black"
    hysteresis_text = lambda value: "black"

    im_up = draw_heatmap(
        axes[0],
        up_matrix,
        b_values,
        ss4_values,
        "Ramp Up Threshold",
        cmap="viridis",
        text_color_rule=threshold_text,
        vmin=threshold_vmin,
        vmax=threshold_vmax,
    )
    im_down = draw_heatmap(
        axes[1],
        down_matrix,
        b_values,
        ss4_values,
        "Ramp Down Threshold",
        cmap="viridis",
        text_color_rule=threshold_text,
        vmin=threshold_vmin,
        vmax=threshold_vmax,
    )
    im_hyst = draw_heatmap(
        axes[2],
        hysteresis_matrix,
        b_values,
        ss4_values,
        "Hysteresis Width (Up - Down)",
        cmap="coolwarm",
        text_color_rule=hysteresis_text,
        vmin=hysteresis_vmin,
        vmax=hysteresis_vmax,
    )

    cbar1 = fig.colorbar(im_up, ax=axes[:2], fraction=0.028, pad=0.02)
    cbar1.set_label("Input amplitude")
    cbar2 = fig.colorbar(im_hyst, ax=axes[2], fraction=0.046, pad=0.04)
    cbar2.set_label("Amplitude difference")

    fig.suptitle(f"{topology}: {title_prefix}", fontsize=13)
    fig.subplots_adjust(left=0.07, right=0.94, bottom=0.14, top=0.84, wspace=0.38)
    fig.savefig(output_dir / f"{output_name}_{topology}.png", dpi=dpi)
    plt.close(fig)


def plot_top_hysteresis_table(transition_summary, topology, prefix, title, output_name, output_dir, dpi, max_rows=10):
    topo_summary = transition_summary[transition_summary["topology"] == topology].copy()
    if topo_summary.empty:
        return

    width_column = f"curve_{prefix}_hysteresis_width"
    up_column = f"curve_{prefix}_threshold_up"
    down_column = f"curve_{prefix}_threshold_down"
    seed_mean_column = f"seed_{prefix}_hysteresis_mean"
    seed_std_column = f"seed_{prefix}_hysteresis_std"

    table_df = topo_summary.sort_values(width_column, ascending=False).head(max_rows).copy()
    display_df = table_df[
        [
            "variance_ss4_vt",
            "variance_b_vt",
            up_column,
            down_column,
            width_column,
            seed_mean_column,
            seed_std_column,
        ]
    ].rename(
        columns={
            "variance_ss4_vt": "E var",
            "variance_b_vt": "I var",
            up_column: "Up thr",
            down_column: "Down thr",
            width_column: "Curve width",
            seed_mean_column: "Seed width mean",
            seed_std_column: "Seed width std",
        }
    )

    for column in display_df.columns:
        display_df[column] = display_df[column].map(format_value)

    fig_height = 0.55 * (len(display_df) + 2)
    fig, ax = plt.subplots(figsize=(8.8, max(fig_height, 2.5)))
    ax.axis("off")
    ax.set_title(title, fontsize=13, pad=12)
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.2)
    fig.tight_layout()
    fig.savefig(output_dir / f"{output_name}_{topology}.png", dpi=dpi)
    plt.close(fig)


def plot_phase_slice_grid(phase_classification, topology, varying_column, fixed_column, output_dir, dpi):
    topo_phase = phase_classification[phase_classification["topology"] == topology].copy()
    if topo_phase.empty:
        return

    varying_values = sorted(topo_phase[varying_column].dropna().unique())
    fixed_values = sorted(topo_phase[fixed_column].dropna().unique())
    amplitudes = sorted(topo_phase["input_amplitude"].dropna().unique())

    fig, axes = plt.subplots(
        1,
        len(fixed_values),
        figsize=(3.1 * len(fixed_values), 4.4),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    cmap = ListedColormap(["#d9d9d9", "#f4a261", "#e9c46a", "#457b9d", "#1d3557"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

    for axis, fixed_value in zip(axes, fixed_values):
        subset = topo_phase[topo_phase[fixed_column] == fixed_value].copy()
        matrix = (
            subset.pivot_table(
                index=varying_column,
                columns="input_amplitude",
                values="phase_code",
                aggfunc="first",
            )
            .reindex(index=varying_values, columns=amplitudes)
            .to_numpy(dtype=float)
        )
        axis.imshow(matrix, origin="lower", aspect="auto", cmap=cmap, norm=norm)
        axis.set_title(f"{fixed_column}={format_value(fixed_value)}", fontsize=10)
        axis.set_xlabel("input amplitude")
        axis.set_xticks(range(len(amplitudes)))
        axis.set_xticklabels([format_value(value) for value in amplitudes], rotation=45, ha="right", fontsize=8)

    axes[0].set_ylabel(varying_column)
    axes[0].set_yticks(range(len(varying_values)))
    axes[0].set_yticklabels([format_value(value) for value in varying_values], fontsize=8)

    title_vary = "variance_ss4_vt" if varying_column == "variance_ss4_vt" else "variance_b_vt"
    title_fixed = "variance_b_vt" if fixed_column == "variance_b_vt" else "variance_ss4_vt"
    fig.suptitle(
        f"{topology}: phase regions in input amplitude vs {title_vary}\n"
        f"Panels are fixed {title_fixed}; strong > weak > synchronous-only > bistable > monostable",
        fontsize=13,
        y=0.98,
    )

    legend_handles = [
        Patch(facecolor="#d9d9d9", edgecolor="none", label="monostable"),
        Patch(facecolor="#f4a261", edgecolor="none", label="bistable"),
        Patch(facecolor="#e9c46a", edgecolor="none", label="synchronous only"),
        Patch(facecolor="#457b9d", edgecolor="none", label="weakly oscillatory"),
        Patch(facecolor="#1d3557", edgecolor="none", label="strongly oscillatory"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", frameon=False)
    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.92))

    suffix = "variance_ss4" if varying_column == "variance_ss4_vt" else "variance_b"
    fig.savefig(output_dir / f"phase_regions_vs_{suffix}_{topology}.png", dpi=dpi)
    plt.close(fig)


def write_summary_markdown(
    output_dir,
    input_path,
    rate_column,
    rate_threshold_hz,
    psd_peak_ratio_threshold,
    strong_entropy_threshold,
    noise_corr_threshold,
    category_fraction_threshold,
    rate_transition_summary,
    category_outputs,
):
    lines = [
        "# Threshold Heterogeneity Bifurcation Summary",
        "",
        f"- Input CSV: `{input_path}`",
        f"- Rate threshold metric: `{rate_column}`",
        f"- Active-state criterion: `{rate_column} >= {rate_threshold_hz:.3f} Hz`",
        (
            f"- Strongly oscillatory per seed: `psd_peak_ratio > {psd_peak_ratio_threshold:.3f}` "
            f"and `pop_spec_entropy < {strong_entropy_threshold:.3f}`"
        ),
        (
            f"- Weakly oscillatory per seed: `psd_peak_ratio > {psd_peak_ratio_threshold:.3f}` "
            f"and `pop_spec_entropy >= {strong_entropy_threshold:.3f}`"
        ),
        (
            f"- Synchronous only per seed: not oscillatory, and "
            f"`mean_noise_corr_50ms > {noise_corr_threshold:.3f}` or "
            f"`mean_noise_corr_10ms > {noise_corr_threshold:.3f}`"
        ),
        f"- Curve-level category threshold: category seed fraction >= {category_fraction_threshold:.3f}",
        "",
        "## Files",
        "",
        "- `aggregated_rate_curves.csv`: seed-averaged rate curves vs input amplitude.",
        "- `curve_bifurcation_thresholds.csv`: rate thresholds detected from the mean curve for each ramp.",
        "- `hysteresis_summary.csv`: up/down rate thresholds and rate-hysteresis widths.",
        "- `aggregated_oscillation_curves.csv`: strong/weak/synchronous-only seed fractions plus averaged PSD/correlation/entropy metrics.",
        "- `curve_strong_thresholds.csv`, `curve_weak_thresholds.csv`, `curve_sync_only_thresholds.csv`: category onset thresholds for each ramp.",
        "- `strong_hysteresis_summary.csv`, `weak_hysteresis_summary.csv`, `sync_only_hysteresis_summary.csv`: up/down category onset thresholds and widths.",
        "- `phase_classification_by_condition_amplitude.csv`: monostable / bistable / synchronous-only / weakly oscillatory / strongly oscillatory labels.",
        "",
        "## Largest Rate Hysteresis Widths",
        "",
    ]

    top_rate = rate_transition_summary.sort_values("curve_rate_hysteresis_width", ascending=False).head(12)
    if top_rate.empty:
        lines.append("No rate hysteresis rows were available.")
    else:
        lines.append("| topology | variance_ss4_vt | variance_b_vt | up | down | width |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for _, row in top_rate.iterrows():
            lines.append(
                "| "
                f"{row['topology']} | "
                f"{format_value(row['variance_ss4_vt'])} | "
                f"{format_value(row['variance_b_vt'])} | "
                f"{format_value(row['curve_rate_threshold_up'])} | "
                f"{format_value(row['curve_rate_threshold_down'])} | "
                f"{format_value(row['curve_rate_hysteresis_width'])} |"
            )

    for category_key in ("strong", "weak", "sync_only"):
        category_data = category_outputs[category_key]
        transition_summary = category_data["transition_summary"]
        category_label = CATEGORY_SPECS[category_key]["label"].title()
        lines.extend(["", f"## Largest {category_label} Hysteresis Widths", ""])
        top_rows = transition_summary.sort_values(
            f"curve_{category_key}_hysteresis_width",
            ascending=False,
        ).head(12)
        if top_rows.empty:
            lines.append(f"No {category_label.lower()} hysteresis rows were available.")
            continue

        lines.append("| topology | variance_ss4_vt | variance_b_vt | up | down | width |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for _, row in top_rows.iterrows():
            lines.append(
                "| "
                f"{row['topology']} | "
                f"{format_value(row['variance_ss4_vt'])} | "
                f"{format_value(row['variance_b_vt'])} | "
                f"{format_value(row[f'curve_{category_key}_threshold_up'])} | "
                f"{format_value(row[f'curve_{category_key}_threshold_down'])} | "
                f"{format_value(row[f'curve_{category_key}_hysteresis_width'])} |"
            )

    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(
        args.input.resolve(),
        psd_peak_ratio_threshold=args.psd_peak_ratio_threshold,
        strong_entropy_threshold=args.strong_entropy_threshold,
        noise_corr_threshold=args.noise_corr_threshold,
    )

    rate_summary = aggregate_rate_curves(df, args.rate_column)
    rate_curve_thresholds = compute_curve_thresholds(
        rate_summary,
        value_column=f"{args.rate_column}_mean",
        threshold_value=args.rate_threshold_hz,
        prefix="rate",
    )
    rate_seed_thresholds = compute_seed_thresholds(
        df,
        value_column=args.rate_column,
        threshold_value=args.rate_threshold_hz,
        prefix="rate",
    )
    rate_seed_threshold_summary = summarize_seed_thresholds(rate_seed_thresholds, prefix="rate")
    rate_seed_hysteresis_by_seed, rate_seed_hysteresis_summary = summarize_seed_hysteresis(
        rate_seed_thresholds,
        prefix="rate",
    )
    rate_curve_thresholds = rate_curve_thresholds.merge(
        rate_seed_threshold_summary,
        on=[*GROUP_COLUMNS, "sweep_direction"],
        how="left",
    )
    rate_transition_summary = build_transition_summary(
        rate_curve_thresholds,
        rate_seed_hysteresis_summary,
        prefix="rate",
    )

    oscillation_summary = aggregate_oscillation_curves(df)
    category_outputs = {}
    for category_key, category_spec in CATEGORY_SPECS.items():
        curve_thresholds = compute_curve_thresholds(
            oscillation_summary,
            value_column=category_spec["fraction_column"],
            threshold_value=args.osc_fraction_threshold,
            prefix=category_key,
        )
        seed_thresholds = compute_seed_thresholds(
            df,
            value_column=f"{category_spec['seed_column']}_numeric",
            threshold_value=0.5,
            prefix=category_key,
        )
        seed_threshold_summary = summarize_seed_thresholds(seed_thresholds, prefix=category_key)
        seed_hysteresis_by_seed, seed_hysteresis_summary = summarize_seed_hysteresis(
            seed_thresholds,
            prefix=category_key,
        )
        curve_thresholds = curve_thresholds.merge(
            seed_threshold_summary,
            on=[*GROUP_COLUMNS, "sweep_direction"],
            how="left",
        )
        transition_summary = build_transition_summary(
            curve_thresholds,
            seed_hysteresis_summary,
            prefix=category_key,
        )
        category_outputs[category_key] = {
            "curve_thresholds": curve_thresholds,
            "seed_thresholds": seed_thresholds,
            "seed_threshold_summary": seed_threshold_summary,
            "seed_hysteresis_by_seed": seed_hysteresis_by_seed,
            "transition_summary": transition_summary,
        }

    phase_classification = build_phase_classification(
        rate_transition_summary=rate_transition_summary,
        oscillation_summary=oscillation_summary,
        osc_fraction_threshold=args.osc_fraction_threshold,
    )

    save_tables(
        output_dir=output_dir,
        rate_summary=rate_summary,
        rate_curve_thresholds=rate_curve_thresholds,
        rate_seed_thresholds=rate_seed_thresholds,
        rate_seed_threshold_summary=rate_seed_threshold_summary,
        rate_seed_hysteresis_by_seed=rate_seed_hysteresis_by_seed,
        rate_transition_summary=rate_transition_summary,
        oscillation_summary=oscillation_summary,
        category_outputs=category_outputs,
        phase_classification=phase_classification,
    )

    for topology in sorted(df["topology"].dropna().unique()):
        plot_rate_grid(
            rate_summary=rate_summary,
            rate_curve_thresholds=rate_curve_thresholds,
            topology=topology,
            rate_column=args.rate_column,
            rate_threshold_hz=args.rate_threshold_hz,
            output_dir=output_dir,
            dpi=args.dpi,
        )
        plot_transition_heatmaps(
            transition_summary=rate_transition_summary,
            topology=topology,
            prefix="rate",
            title_prefix="bifurcation thresholds from mean rate",
            output_name="bifurcation_threshold_heatmaps",
            output_dir=output_dir,
            dpi=args.dpi,
        )
        plot_top_hysteresis_table(
            transition_summary=rate_transition_summary,
            topology=topology,
            prefix="rate",
            title=f"{topology}: largest rate hysteresis widths",
            output_name="top_hysteresis_table",
            output_dir=output_dir,
            dpi=args.dpi,
        )
        for category_key, category_spec in CATEGORY_SPECS.items():
            plot_category_fraction_grid(
                oscillation_summary=oscillation_summary,
                category_curve_thresholds=category_outputs[category_key]["curve_thresholds"],
                category_key=category_key,
                topology=topology,
                category_fraction_threshold=args.osc_fraction_threshold,
                output_dir=output_dir,
                dpi=args.dpi,
            )
            plot_transition_heatmaps(
                transition_summary=category_outputs[category_key]["transition_summary"],
                topology=topology,
                prefix=category_key,
                title_prefix=f"{category_spec['label']} onset thresholds",
                output_name=f"{category_key}_threshold_heatmaps",
                output_dir=output_dir,
                dpi=args.dpi,
            )
            plot_top_hysteresis_table(
                transition_summary=category_outputs[category_key]["transition_summary"],
                topology=topology,
                prefix=category_key,
                title=f"{topology}: largest {category_spec['label']} hysteresis widths",
                output_name=f"top_{category_key}_hysteresis_table",
                output_dir=output_dir,
                dpi=args.dpi,
            )
        plot_phase_slice_grid(
            phase_classification=phase_classification,
            topology=topology,
            varying_column="variance_ss4_vt",
            fixed_column="variance_b_vt",
            output_dir=output_dir,
            dpi=args.dpi,
        )
        plot_phase_slice_grid(
            phase_classification=phase_classification,
            topology=topology,
            varying_column="variance_b_vt",
            fixed_column="variance_ss4_vt",
            output_dir=output_dir,
            dpi=args.dpi,
        )

    write_summary_markdown(
        output_dir=output_dir,
        input_path=args.input.resolve(),
        rate_column=args.rate_column,
        rate_threshold_hz=args.rate_threshold_hz,
        psd_peak_ratio_threshold=args.psd_peak_ratio_threshold,
        strong_entropy_threshold=args.strong_entropy_threshold,
        noise_corr_threshold=args.noise_corr_threshold,
        category_fraction_threshold=args.osc_fraction_threshold,
        rate_transition_summary=rate_transition_summary,
        category_outputs=category_outputs,
    )
    print(f"Wrote analysis outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
