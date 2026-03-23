import argparse
import csv
import json
import math
import textwrap
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors


CLASS_ORDER = ("AI", "AR", "SI", "SR", "QUIESCENT", "NON_PERSISTENT")
CLASS_TO_CODE = {name: idx for idx, name in enumerate(CLASS_ORDER)}
METRIC_COLUMNS = (
    "persistence_fraction",
    "ssai_fraction",
    "active_fraction",
    "synchronous_fraction",
    "rate_late_Hz_E_mean",
    "rate_late_Hz_I_mean",
    "rate_late_Hz_mean",
    "rate_post_Hz_E_mean",
    "rate_post_Hz_I_mean",
    "rate_post_Hz_mean",
    "ISI_CV_mean_mean",
    "Fano_median_300ms_mean",
    "peak_ratio_mean",
    "mean_noise_corr_50ms_mean",
    "oscillation_freq_hz_mean",
)

PAPER_PARAMETER_LABELS = {
    "ampa_strength": "AMPA Strength",
    "nmda_strength": "NMDA Strength",
    "gabaa_strength": "GABA_A Strength",
    "gabab_strength": "GABA_B Strength",
    "inhibitory_strength": "Inhibitory Strength",
    "delay_scale": "Delay Scale",
    "delay_mean_exc_ms": "Mean Excitatory Delay (ms)",
    "delay_mean_inh_ms": "Mean Inhibitory Delay (ms)",
    "recurrent_exc_sigma": "Recurrent Excitatory Sigma",
    "e_normalization": "Excitatory Normalization",
    "i_normalization": "Inhibitory Normalization",
    "spatial_ei_lambda_E": "Excitatory Lambda",
    "spatial_ei_lambda_I": "Inhibitory Lambda",
    "sigma_E": "Excitatory Sigma",
    "sigma_I": "Inhibitory Sigma",
}

PAPER_METRIC_LABELS = {
    "Fano_median_300ms_mean": "Fano Factor (300 ms)",
    "ISI_CV_mean_mean": "ISI CV",
    "active_fraction": "Active Fraction",
    "mean_noise_corr_50ms_mean": "Mean Pairwise Correlation (50 ms)",
    "oscillation_freq_hz_mean": "Oscillation Frequency (Hz)",
    "peak_ratio_mean": "PSD Peak Ratio",
    "persistence_fraction": "Persistence Fraction",
    "rate_late_Hz_mean": "Late Firing Rate (Hz)",
    "rate_post_Hz_mean": "Post-Stimulus Firing Rate (Hz)",
    "ssai_fraction": "SSAI Fraction",
    "synchronous_fraction": "Synchronous Fraction",
}

PAPER_SECTION_LABELS = {
    "dense_random_topology": "Dense Random Topology",
    "realistic_topology": "Realistic Topology",
}

PAPER_VALUE_LABELS = {
    "median_abs_spearman": "Median |Spearman rho|",
    "median_effect_rel_range": "Median Relative Range",
    "median_signed_pearson": "Median Signed Pearson r",
}

PAPER_CLASS_LABELS = {
    "AI": "AI",
    "AR": "AR",
    "SI": "SI",
    "SR": "SR",
    "QUIESCENT": "Quiescent",
    "NON_PERSISTENT": "Non-Persistent",
}

METRIC_COMPARISON_ORDER = [
    "active_fraction",
    "persistence_fraction",
    "ssai_fraction",
    "synchronous_fraction",
    "peak_ratio_mean",
    "mean_noise_corr_50ms_mean",
    "oscillation_freq_hz_mean",
    "rate_post_Hz_mean",
    "rate_late_Hz_mean",
    "ISI_CV_mean_mean",
    "Fano_median_300ms_mean",
]

PARAMETER_COMPARISON_ORDER = [
    "ampa_strength",
    "nmda_strength",
    "gabaa_strength",
    "gabab_strength",
    "delay_mean_exc_ms",
    "delay_mean_inh_ms",
    "delay_scale",
    "recurrent_exc_sigma",
    "inhibitory_strength",
    "e_normalization",
    "i_normalization",
    "spatial_ei_lambda_E",
    "spatial_ei_lambda_I",
    "sigma_E",
    "sigma_I",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate batch sweep2d outputs across families/trials for paper-style results, "
            "including aggregate phase plots and parameter impact rankings."
        )
    )
    parser.add_argument("--batch-dir", type=str, required=True, help="Batch sweep directory produced by run_analysis_code_sweeps.py")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory; defaults to <batch-dir>/paper_analysis")
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include jobs with nonzero return codes if their summary files exist. Default is successful jobs only.",
    )
    return parser


def _read_csv(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [_coerce_row(row) for row in rows]


def _coerce_row(row: Mapping[str, str]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in row.items():
        if value is None or value == "":
            out[key] = ""
            continue
        try:
            if any(ch in value for ch in (".", "e", "E")):
                out[key] = float(value)
            else:
                out[key] = int(value)
            continue
        except ValueError:
            out[key] = value
    return out


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _mean(values: Sequence[object]) -> float:
    vals = []
    for value in values:
        try:
            val = float(value)
        except Exception:
            continue
        if math.isfinite(val):
            vals.append(val)
    return float(np.mean(vals)) if vals else float("nan")


def _nanmedian(values: Sequence[object]) -> float:
    vals = []
    for value in values:
        try:
            val = float(value)
        except Exception:
            continue
        if math.isfinite(val):
            vals.append(val)
    return float(np.median(vals)) if vals else float("nan")


def _mode(values: Sequence[object]) -> object:
    items = [value for value in values if value != ""]
    if not items:
        return ""
    counts = Counter(items)
    return sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[0][0]


def _sorted_axis(values: Sequence[object]) -> List[object]:
    return sorted(set(values), key=lambda x: float(x) if isinstance(x, (int, float)) else str(x))


def _format_tick(value: object) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.3g}"
    return str(value)


def _parse_job_meta(batch_dir: Path, job_dir: Path) -> Dict[str, object]:
    rel = job_dir.relative_to(batch_dir)
    parts = rel.parts
    if len(parts) < 3:
        raise ValueError(f"Unexpected batch job path: {job_dir}")
    section = parts[0]
    family_part = parts[1]
    job_slug = parts[2]
    family_match = family_part.split("_")
    family_id = int(family_match[1])
    trial_number = int(family_match[3])
    return {
        "section": section,
        "family_id": family_id,
        "trial_number": trial_number,
        "job_slug": job_slug,
    }


def _successful_job_dirs(batch_dir: Path, include_failed: bool) -> List[Path]:
    summary_dirs = sorted({path.parent.resolve() for path in batch_dir.rglob("sweep2d_summary.csv")})
    results_path = batch_dir / "batch_results.json"
    if not results_path.exists():
        return summary_dirs

    results = json.loads(results_path.read_text(encoding="utf-8"))
    returncodes: Dict[Path, int] = {}
    for row in results:
        out_dir = Path(str(row["out_dir"])).resolve()
        returncodes[out_dir] = int(row.get("returncode", 0))

    if include_failed:
        return summary_dirs

    keep = []
    for job_dir in summary_dirs:
        if returncodes.get(job_dir, 0) == 0:
            keep.append(job_dir)
    return keep


def _load_job_rows(batch_dir: Path, include_failed: bool) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    summary_rows: List[Dict[str, object]] = []
    detail_rows: List[Dict[str, object]] = []
    for job_dir in _successful_job_dirs(batch_dir, include_failed):
        meta = _parse_job_meta(batch_dir, job_dir)
        job_summary = _read_csv(job_dir / "sweep2d_summary.csv")
        job_detail = _read_csv(job_dir / "sweep2d_detail.csv")
        for row in job_summary:
            row.update(meta)
            row["pair_key"] = f"{row['x_name']}__vs__{row['y_name']}__{job_slug_topology(meta['job_slug'])}"
        for row in job_detail:
            row.update(meta)
            row["pair_key"] = f"{row['x_name']}__vs__{row['y_name']}__{job_slug_topology(meta['job_slug'])}"
        summary_rows.extend(job_summary)
        detail_rows.extend(job_detail)
    return summary_rows, detail_rows


def job_slug_topology(job_slug: str) -> str:
    if job_slug.endswith("_spatial_ei"):
        return "spatial_ei"
    if job_slug.endswith("_baseline"):
        return "baseline"
    return job_slug.split("_")[-1]


def _pair_group_key(row: Mapping[str, object]) -> Tuple[str, str, str, str]:
    return (str(row["section"]), str(row["x_name"]), str(row["y_name"]), str(row.get("topology", job_slug_topology(str(row["job_slug"])))))


def _aggregate_pair_surface(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[object, ...], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["section"]),
                str(row["x_name"]),
                str(row["y_name"]),
                str(row.get("topology", job_slug_topology(str(row["job_slug"])))),
                row["x_axis_value"],
                row["y_axis_value"],
            )
        ].append(row)

    out: List[Dict[str, object]] = []
    for key in sorted(grouped.keys(), key=lambda item: tuple(str(v) for v in item)):
        group = grouped[key]
        class_counts = Counter(str(row["mode_activity_class"]) for row in group)
        persistent_counts = Counter(str(row["mode_persistent_activity_class"]) for row in group)
        dominant_class, dominant_count = sorted(class_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        dominant_persistent, dominant_persistent_count = sorted(persistent_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        total = max(1, len(group))
        class_probs = [count / total for count in class_counts.values() if count > 0]
        persistent_probs = [count / total for count in persistent_counts.values() if count > 0]
        out.append(
            {
                "section": key[0],
                "x_name": key[1],
                "y_name": key[2],
                "topology": key[3],
                "x_axis_value": key[4],
                "y_axis_value": key[5],
                "n_job_surfaces": int(total),
                "mode_activity_class": dominant_class,
                "mode_persistent_activity_class": dominant_persistent,
                "activity_class_consensus": float(dominant_count / total),
                "persistent_class_consensus": float(dominant_persistent_count / total),
                "n_distinct_activity_classes": int(sum(1 for count in class_counts.values() if count > 0)),
                "n_distinct_persistent_classes": int(sum(1 for count in persistent_counts.values() if count > 0)),
                "activity_class_entropy": float(-sum(prob * math.log2(prob) for prob in class_probs)),
                "persistent_class_entropy": float(-sum(prob * math.log2(prob) for prob in persistent_probs)),
                **{f"{cls}_cell_fraction": float(np.mean([1.0 if row["mode_activity_class"] == cls else 0.0 for row in group])) for cls in CLASS_ORDER},
                **{metric: _mean([row.get(metric, float("nan")) for row in group]) for metric in METRIC_COLUMNS},
            }
        )
    return out


def _make_grid(rows: Sequence[Mapping[str, object]], value_key: str) -> Tuple[List[object], List[object], np.ndarray]:
    x_vals = _sorted_axis([row["x_axis_value"] for row in rows])
    y_vals = _sorted_axis([row["y_axis_value"] for row in rows])
    x_lookup = {value: idx for idx, value in enumerate(x_vals)}
    y_lookup = {value: idx for idx, value in enumerate(y_vals)}
    grid = np.full((len(y_vals), len(x_vals)), np.nan, dtype=float)
    for row in rows:
        x_idx = x_lookup[row["x_axis_value"]]
        y_idx = y_lookup[row["y_axis_value"]]
        value = row.get(value_key, float("nan"))
        try:
            grid[y_idx, x_idx] = float(value)
        except Exception:
            grid[y_idx, x_idx] = np.nan
    return x_vals, y_vals, grid


def _plot_heatmap(
    *,
    grid: np.ndarray,
    x_vals: Sequence[object],
    y_vals: Sequence[object],
    title: str,
    x_label: str,
    y_label: str,
    out_path: Path,
    cmap: str = "viridis",
    colorbar_label: str = "",
    class_mode: bool = False,
    vmin=None,
    vmax=None,
) -> None:
    wrapped_title = "\n".join(textwrap.wrap(str(title), width=64)) if title else ""
    fig, ax = plt.subplots(figsize=(8.8, 8.8), constrained_layout=True)
    if class_mode:
        discrete_cmap = plt.get_cmap(cmap, len(CLASS_ORDER))
        bounds = np.arange(-0.5, len(CLASS_ORDER) + 0.5, 1.0)
        norm = colors.BoundaryNorm(bounds, discrete_cmap.N)
        im = ax.imshow(grid, origin="lower", aspect="equal", cmap=discrete_cmap, norm=norm)
    else:
        im = ax.imshow(grid, origin="lower", aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(x_vals)))
    ax.set_yticks(np.arange(len(y_vals)))
    ax.set_xticklabels([_format_tick(v) for v in x_vals], rotation=45, ha="right")
    ax.set_yticklabels([_format_tick(v) for v in y_vals])
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_title(wrapped_title, pad=12, fontsize=15)
    ax.tick_params(axis="both", labelsize=12)
    cbar = fig.colorbar(im, ax=ax, boundaries=bounds if class_mode else None, shrink=0.78, fraction=0.05, pad=0.03)
    cbar.set_label(colorbar_label, fontsize=12)
    cbar.ax.tick_params(labelsize=11)
    if class_mode:
        cbar.set_ticks(list(CLASS_TO_CODE.values()))
        cbar.set_ticklabels(list(CLASS_TO_CODE.keys()))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _pair_slug(section: str, x_name: str, y_name: str, topology: str) -> str:
    return f"{section}__{x_name}_vs_{y_name}__{topology}".lower().replace(" ", "_")


def _slugify(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _paper_parameter_label(name: str) -> str:
    return PAPER_PARAMETER_LABELS.get(name, name.replace("_", " "))


def _paper_metric_label(name: str) -> str:
    return PAPER_METRIC_LABELS.get(name, name.replace("_", " "))


def _paper_section_label(name: str) -> str:
    return PAPER_SECTION_LABELS.get(name, name.replace("_", " ").title())


def _paper_value_label(name: str) -> str:
    return PAPER_VALUE_LABELS.get(name, name)


def _paper_class_label(name: str) -> str:
    return PAPER_CLASS_LABELS.get(name, name)


def _strip_units(label: str) -> str:
    return re.sub(r"\s*\([^)]*\)", "", label).strip()


def _axis_caption(parameter_name: str, mode: str | None) -> str:
    base = _paper_parameter_label(parameter_name)
    if str(mode).lower() == "relative":
        return f"{_strip_units(base)} Relative Factor"
    return base


def _comparison_parameter_key(section: str, parameter: str) -> str:
    if section == "realistic_topology" and parameter in {"inhibitory_strength", "i_normalization"}:
        return "merged_inhibitory_normalization"
    return parameter


def _comparison_parameter_label(section: str, parameter: str, *, paper_labels: bool) -> str:
    if section == "realistic_topology" and parameter == "merged_inhibitory_normalization":
        return "Inhibitory Normalization" if paper_labels else "merged_inhibitory_normalization"
    return _paper_parameter_label(parameter) if paper_labels else parameter


def _ordered_metrics(metric_names: Sequence[str]) -> List[str]:
    order_lookup = {name: idx for idx, name in enumerate(METRIC_COMPARISON_ORDER)}
    return sorted(set(metric_names), key=lambda name: (order_lookup.get(name, len(order_lookup)), _paper_metric_label(name), name))


def _ordered_parameters(parameter_names: Sequence[str]) -> List[str]:
    order_lookup = {name: idx for idx, name in enumerate(PARAMETER_COMPARISON_ORDER)}
    def _order_name(name: str) -> str:
        if name == "merged_inhibitory_normalization":
            return "i_normalization"
        return name
    return sorted(
        set(parameter_names),
        key=lambda name: (order_lookup.get(_order_name(name), len(order_lookup)), _paper_parameter_label(_order_name(name)), name),
    )


def _aggregate_comparison_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    section: str,
    value_key: str,
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in rows:
        parameter = _comparison_parameter_key(section, str(row["parameter"]))
        metric = str(row["metric"])
        try:
            value = float(row[value_key])
        except Exception:
            continue
        if math.isfinite(value):
            grouped[(parameter, metric)].append(value)

    out: List[Dict[str, object]] = []
    for (parameter, metric), values in grouped.items():
        out.append(
            {
                "parameter": parameter,
                "metric": metric,
                value_key: _nanmedian(values),
            }
        )
    return out


def _rank_transform(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def _spearman(values_x: Sequence[float], values_y: Sequence[float]) -> float:
    x = np.asarray(values_x, dtype=float)
    y = np.asarray(values_y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    xr = _rank_transform(x)
    yr = _rank_transform(y)
    return float(np.corrcoef(xr, yr)[0, 1])


def _pearson(values_x: Sequence[float], values_y: Sequence[float]) -> float:
    x = np.asarray(values_x, dtype=float)
    y = np.asarray(values_y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _axis_profile(rows: Sequence[Mapping[str, object]], axis: str, metric: str) -> List[Tuple[float, float]]:
    axis_key = f"{axis}_axis_value"
    other_key = "y_axis_value" if axis == "x" else "x_axis_value"
    grouped: Dict[object, List[float]] = defaultdict(list)
    for row in rows:
        try:
            axis_val = float(row[axis_key])
            metric_val = float(row[metric])
        except Exception:
            continue
        if math.isfinite(axis_val) and math.isfinite(metric_val):
            grouped[axis_val].append(metric_val)
    return sorted((axis_val, float(np.mean(vals))) for axis_val, vals in grouped.items())


def _parameter_impact_rows(summary_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    by_surface: Dict[Tuple[object, ...], List[Mapping[str, object]]] = defaultdict(list)
    for row in summary_rows:
        key = (row["section"], row["family_id"], row["trial_number"], row["job_slug"])
        by_surface[key].append(row)

    impact_rows: List[Dict[str, object]] = []
    for key, surface_rows in by_surface.items():
        if not surface_rows:
            continue
        meta = surface_rows[0]
        for axis in ("x", "y"):
            param_name = str(meta[f"{axis}_name"])
            axis_values = [float(row[f"{axis}_axis_value"]) for row in surface_rows]
            if all(val > 0 for val in axis_values):
                transformed_axis = np.log10(np.asarray(axis_values, dtype=float))
            else:
                transformed_axis = np.asarray(axis_values, dtype=float)
            for metric in METRIC_COLUMNS:
                profile = _axis_profile(surface_rows, axis, metric)
                if len(profile) < 2:
                    continue
                xs = np.asarray([item[0] for item in profile], dtype=float)
                ys = np.asarray([item[1] for item in profile], dtype=float)
                x_for_corr = np.log10(xs) if np.all(xs > 0) else xs
                impact_rows.append(
                    {
                        "section": meta["section"],
                        "topology": meta.get("topology", job_slug_topology(str(meta["job_slug"]))),
                        "family_id": int(meta["family_id"]),
                        "trial_number": int(meta["trial_number"]),
                        "job_slug": meta["job_slug"],
                        "parameter": param_name,
                        "metric": metric,
                        "effect_range": float(np.nanmax(ys) - np.nanmin(ys)),
                        "effect_rel_range": float((np.nanmax(ys) - np.nanmin(ys)) / (abs(np.nanmean(ys)) + 1e-9)),
                        "spearman_rho": _spearman(x_for_corr, ys),
                        "pearson_r": _pearson(x_for_corr, ys),
                    }
                )
    return impact_rows


def _aggregate_impact(impact_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in impact_rows:
        grouped[(str(row["parameter"]), str(row["metric"]))].append(row)

    out: List[Dict[str, object]] = []
    for key in sorted(grouped.keys()):
        group = grouped[key]
        out.append(
            {
                "parameter": key[0],
                "metric": key[1],
                "n_surfaces": int(len(group)),
                "median_abs_spearman": _nanmedian([abs(float(row["spearman_rho"])) for row in group]),
                "median_signed_pearson": _nanmedian([float(row["pearson_r"]) for row in group]),
                "median_abs_pearson": _nanmedian([abs(float(row["pearson_r"])) for row in group]),
                "median_effect_range": _nanmedian([float(row["effect_range"]) for row in group]),
                "median_effect_rel_range": _nanmedian([float(row["effect_rel_range"]) for row in group]),
            }
        )
    return out


def _impact_heatmap(
    rows: Sequence[Mapping[str, object]],
    value_key: str,
    out_path: Path,
    title: str,
    *,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    paper_labels: bool = False,
) -> None:
    parameters = sorted({str(row["parameter"]) for row in rows})
    metrics = sorted({str(row["metric"]) for row in rows})
    wrapped_title = "\n".join(textwrap.wrap(str(title), width=64)) if title else ""
    grid = np.full((len(metrics), len(parameters)), np.nan, dtype=float)
    x_lookup = {name: idx for idx, name in enumerate(parameters)}
    y_lookup = {name: idx for idx, name in enumerate(metrics)}
    for row in rows:
        grid[y_lookup[str(row["metric"])], x_lookup[str(row["parameter"])]] = float(row[value_key])
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(parameters)), max(6, 0.45 * len(metrics))))
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(parameters)))
    ax.set_yticks(np.arange(len(metrics)))
    xlabels = [_paper_parameter_label(name) for name in parameters] if paper_labels else parameters
    ylabels = [_paper_metric_label(name) for name in metrics] if paper_labels else metrics
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticklabels(ylabels)
    ax.set_title(wrapped_title, pad=10)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(_paper_value_label(value_key) if paper_labels else value_key)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _impact_comparison_heatmap(
    *,
    section_to_rows: Mapping[str, Sequence[Mapping[str, object]]],
    value_key: str,
    out_path: Path,
    title: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    paper_labels: bool = False,
) -> None:
    sections = list(sorted(section_to_rows.keys()))
    if not sections:
        return

    section_rows_agg = {
        section: _aggregate_comparison_rows(rows, section=section, value_key=value_key) for section, rows in section_to_rows.items()
    }
    metrics = _ordered_metrics([str(row["metric"]) for rows in section_rows_agg.values() for row in rows])
    max_parameters = max(
        (len(_ordered_parameters([str(row["parameter"]) for row in rows])) for rows in section_rows_agg.values()),
        default=1,
    )
    ncols = len(sections)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(max(8, 0.75 * max_parameters * ncols), max(6, 0.45 * len(metrics) + 2)),
        squeeze=False,
        constrained_layout=True,
    )

    if vmin is None or vmax is None:
        auto_vmin = float("inf")
        auto_vmax = float("-inf")
        for rows in section_rows_agg.values():
            for row in rows:
                try:
                    value = float(row[value_key])
                except Exception:
                    continue
                if math.isfinite(value):
                    auto_vmin = min(auto_vmin, value)
                    auto_vmax = max(auto_vmax, value)
        if not math.isfinite(auto_vmin) or not math.isfinite(auto_vmax):
            auto_vmin, auto_vmax = 0.0, 1.0
        if vmin is None:
            vmin = auto_vmin
        if vmax is None:
            vmax = auto_vmax

    for ax, section in zip(axes[0], sections):
        rows = section_rows_agg[section]
        parameters = _ordered_parameters([str(row["parameter"]) for row in rows])
        grid = np.full((len(metrics), len(parameters)), np.nan, dtype=float)
        x_lookup = {name: idx for idx, name in enumerate(parameters)}
        y_lookup = {name: idx for idx, name in enumerate(metrics)}
        for row in rows:
            try:
                grid[y_lookup[str(row["metric"])], x_lookup[str(row["parameter"])]] = float(row[value_key])
            except Exception:
                continue
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(parameters)))
        ax.set_yticks(np.arange(len(metrics)))
        xlabels = [_comparison_parameter_label(section, name, paper_labels=paper_labels) for name in parameters]
        ylabels = [_paper_metric_label(name) for name in metrics] if paper_labels else metrics
        ax.set_xticklabels(xlabels, rotation=45, ha="right")
        ax.set_yticklabels(ylabels)
        ax.set_title(_paper_section_label(section) if paper_labels else section.replace("_", " "))

    fig.suptitle(title)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label(_paper_value_label(value_key) if paper_labels else value_key)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _top_parameters_by_metric(rows: Sequence[Mapping[str, object]], top_k: int = 5) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["metric"])].append(row)

    out: Dict[str, List[Dict[str, object]]] = {}
    for metric, metric_rows in grouped.items():
        ranked = sorted(
            metric_rows,
            key=lambda row: (
                -float(row["median_abs_spearman"]) if math.isfinite(float(row["median_abs_spearman"])) else float("inf"),
                -float(row["median_effect_rel_range"]) if math.isfinite(float(row["median_effect_rel_range"])) else float("inf"),
                str(row["parameter"]),
            ),
        )
        out[metric] = [
            {
                "parameter": str(row["parameter"]),
                "n_surfaces": int(row["n_surfaces"]),
                "median_abs_spearman": float(row["median_abs_spearman"]),
                "median_effect_rel_range": float(row["median_effect_rel_range"]),
                "median_effect_range": float(row["median_effect_range"]),
            }
            for row in ranked[:top_k]
        ]
    return out


def main() -> None:
    args = _build_parser().parse_args()
    batch_dir = Path(args.batch_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (batch_dir / "paper_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows, detail_rows = _load_job_rows(batch_dir, args.include_failed)
    if not summary_rows:
        raise ValueError(f"No successful sweep2d summaries found under {batch_dir}")

    job_manifest = []
    for row in summary_rows:
        job_manifest.append(
            {
                "section": row["section"],
                "family_id": row["family_id"],
                "trial_number": row["trial_number"],
                "job_slug": row["job_slug"],
                "pair_key": row["pair_key"],
                "topology": row.get("topology", job_slug_topology(str(row["job_slug"]))),
                "x_name": row["x_name"],
                "y_name": row["y_name"],
            }
        )
    manifest_unique = []
    seen = set()
    for row in job_manifest:
        key = tuple(row.items())
        if key in seen:
            continue
        seen.add(key)
        manifest_unique.append(row)
    _write_csv(out_dir / "job_manifest.csv", manifest_unique)

    pair_groups: Dict[Tuple[str, str, str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in summary_rows:
        pair_groups[_pair_group_key(row)].append(row)
    pair_axis_modes: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}
    for row in detail_rows:
        key = _pair_group_key(row)
        entry = pair_axis_modes.setdefault(key, {})
        if "x_mode" in row and "x_mode" not in entry and row["x_mode"] != "":
            entry["x_mode"] = str(row["x_mode"])
        if "y_mode" in row and "y_mode" not in entry and row["y_mode"] != "":
            entry["y_mode"] = str(row["y_mode"])

    pair_overview: List[Dict[str, object]] = []
    aggregate_phase_all: List[Dict[str, object]] = []
    phase_dir = out_dir / "pair_phase_plots"
    phase_dir.mkdir(parents=True, exist_ok=True)

    for pair_key, rows in sorted(pair_groups.items()):
        section, x_name, y_name, topology = pair_key
        surface_rows = _aggregate_pair_surface(rows)
        aggregate_phase_all.extend(surface_rows)
        pair_slug = _pair_slug(section, x_name, y_name, topology)
        pair_dir = phase_dir / pair_slug
        pair_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(pair_dir / "aggregate_phase_table.csv", surface_rows)

        x_vals, y_vals, class_grid = _make_grid(
            [
                {**row, "mode_activity_class_code": CLASS_TO_CODE.get(str(row["mode_activity_class"]), np.nan)}
                for row in surface_rows
            ],
            value_key="mode_activity_class_code",
        )
        _, _, persistence_grid = _make_grid(surface_rows, value_key="persistence_fraction")
        _, _, ssai_grid = _make_grid(surface_rows, value_key="ssai_fraction")
        _, _, consensus_grid = _make_grid(surface_rows, value_key="activity_class_consensus")
        _, _, entropy_grid = _make_grid(surface_rows, value_key="activity_class_entropy")
        _, _, cv_grid = _make_grid(surface_rows, value_key="ISI_CV_mean_mean")
        _, _, fano_grid = _make_grid(surface_rows, value_key="Fano_median_300ms_mean")
        _, _, peak_grid = _make_grid(surface_rows, value_key="peak_ratio_mean")
        _, _, corr_grid = _make_grid(surface_rows, value_key="mean_noise_corr_50ms_mean")
        _, _, rate_post_e_grid = _make_grid(surface_rows, value_key="rate_post_Hz_E_mean")
        _, _, rate_post_i_grid = _make_grid(surface_rows, value_key="rate_post_Hz_I_mean")
        _, _, sync_frac_grid = _make_grid(surface_rows, value_key="synchronous_fraction")
        _, _, osc_freq_grid = _make_grid(surface_rows, value_key="oscillation_freq_hz_mean")
        osc_freq_sync_grid = np.array(osc_freq_grid, copy=True)
        osc_freq_sync_grid[sync_frac_grid < 0.5] = np.nan

        axis_modes = pair_axis_modes.get(pair_key, {})
        x_label = _axis_caption(x_name, axis_modes.get("x_mode"))
        y_label = _axis_caption(y_name, axis_modes.get("y_mode"))
        title_prefix = f"{_paper_section_label(section)}: {_paper_parameter_label(x_name)} vs {_paper_parameter_label(y_name)}"
        _plot_heatmap(
            grid=class_grid,
            x_vals=x_vals,
            y_vals=y_vals,
            title=f"{title_prefix}\nAggregate Activity Class",
            x_label=x_label,
            y_label=y_label,
            out_path=pair_dir / "aggregate_mode_activity_class.png",
            cmap="tab20",
            colorbar_label="Class",
            class_mode=True,
        )
        _plot_heatmap(
            grid=persistence_grid,
            x_vals=x_vals,
            y_vals=y_vals,
            title=f"{title_prefix}\nMean Persistence Fraction",
            x_label=x_label,
            y_label=y_label,
            out_path=pair_dir / "aggregate_persistence_fraction.png",
            colorbar_label="Fraction",
            vmin=0.0,
            vmax=1.0,
        )
        _plot_heatmap(
            grid=ssai_grid,
            x_vals=x_vals,
            y_vals=y_vals,
            title=f"{title_prefix}\nMean SSAI Fraction",
            x_label=x_label,
            y_label=y_label,
            out_path=pair_dir / "aggregate_ssai_fraction.png",
            colorbar_label="Fraction",
            vmin=0.0,
            vmax=1.0,
        )
        _plot_heatmap(
            grid=consensus_grid,
            x_vals=x_vals,
            y_vals=y_vals,
            title=f"{title_prefix}\nFamily Consensus on Activity Class",
            x_label=x_label,
            y_label=y_label,
            out_path=pair_dir / "aggregate_activity_consensus.png",
            colorbar_label="Consensus",
            vmin=0.0,
            vmax=1.0,
        )
        _plot_heatmap(
            grid=entropy_grid,
            x_vals=x_vals,
            y_vals=y_vals,
            title=f"{title_prefix}\nActivity-Class Entropy Across Families",
            x_label=x_label,
            y_label=y_label,
            out_path=pair_dir / "aggregate_activity_class_entropy.png",
            colorbar_label="Entropy (bits)",
            vmin=0.0,
        )
        _plot_heatmap(
            grid=cv_grid,
            x_vals=x_vals,
            y_vals=y_vals,
            title=f"{title_prefix}\nMean ISI CV",
            x_label=x_label,
            y_label=y_label,
            out_path=pair_dir / "aggregate_isi_cv.png",
            colorbar_label="ISI CV",
        )
        _plot_heatmap(
            grid=fano_grid,
            x_vals=x_vals,
            y_vals=y_vals,
            title=f"{title_prefix}\nMean Fano Factor (300 ms)",
            x_label=x_label,
            y_label=y_label,
            out_path=pair_dir / "aggregate_fano_300ms.png",
            colorbar_label="Fano Factor",
        )
        _plot_heatmap(
            grid=peak_grid,
            x_vals=x_vals,
            y_vals=y_vals,
            title=f"{title_prefix}\nMean PSD Peak Ratio",
            x_label=x_label,
            y_label=y_label,
            out_path=pair_dir / "aggregate_peak_ratio.png",
            colorbar_label="PSD Peak Ratio",
        )
        _plot_heatmap(
            grid=corr_grid,
            x_vals=x_vals,
            y_vals=y_vals,
            title=f"{title_prefix}\nMean Pairwise Correlation (50 ms)",
            x_label=x_label,
            y_label=y_label,
            out_path=pair_dir / "aggregate_noise_corr.png",
            colorbar_label="Correlation",
        )
        _plot_heatmap(
            grid=rate_post_e_grid,
            x_vals=x_vals,
            y_vals=y_vals,
            title=f"{title_prefix}\nMean Excitatory Firing Rate",
            x_label=x_label,
            y_label=y_label,
            out_path=pair_dir / "aggregate_rate_post_exc.png",
            colorbar_label="Firing Rate (Hz)",
        )
        _plot_heatmap(
            grid=rate_post_i_grid,
            x_vals=x_vals,
            y_vals=y_vals,
            title=f"{title_prefix}\nMean Inhibitory Firing Rate",
            x_label=x_label,
            y_label=y_label,
            out_path=pair_dir / "aggregate_rate_post_inh.png",
            colorbar_label="Firing Rate (Hz)",
        )
        _plot_heatmap(
            grid=osc_freq_sync_grid,
            x_vals=x_vals,
            y_vals=y_vals,
            title=f"{title_prefix}\nMean Global Oscillation Frequency",
            x_label=x_label,
            y_label=y_label,
            out_path=pair_dir / "aggregate_oscillation_frequency.png",
            colorbar_label="Frequency (Hz)",
        )
        for cls in CLASS_ORDER:
            _, _, class_fraction_grid = _make_grid(surface_rows, value_key=f"{cls}_cell_fraction")
            _plot_heatmap(
                grid=class_fraction_grid,
                x_vals=x_vals,
                y_vals=y_vals,
                title=f"{title_prefix}\nFraction of Families Classified as {_paper_class_label(cls)}",
                x_label=x_label,
                y_label=y_label,
                out_path=pair_dir / f"aggregate_{cls.lower()}_fraction.png",
                colorbar_label="Fraction",
                vmin=0.0,
                vmax=1.0,
            )

        pair_overview.append(
            {
                "section": section,
                "topology": topology,
                "x_name": x_name,
                "y_name": y_name,
                "n_job_surfaces": int(len({(row['family_id'], row['trial_number'], row['job_slug']) for row in rows})),
                "n_phase_cells": int(len(surface_rows)),
                "mean_persistence_fraction": _mean([row["persistence_fraction"] for row in surface_rows]),
                "mean_ssai_fraction": _mean([row["ssai_fraction"] for row in surface_rows]),
                "mean_activity_consensus": _mean([row["activity_class_consensus"] for row in surface_rows]),
                "mean_activity_class_entropy": _mean([row["activity_class_entropy"] for row in surface_rows]),
                "AI_cell_fraction_mean": _mean([row["AI_cell_fraction"] for row in surface_rows]),
                "AR_cell_fraction_mean": _mean([row["AR_cell_fraction"] for row in surface_rows]),
                "SI_cell_fraction_mean": _mean([row["SI_cell_fraction"] for row in surface_rows]),
                "SR_cell_fraction_mean": _mean([row["SR_cell_fraction"] for row in surface_rows]),
                "mean_ISI_CV": _mean([row["ISI_CV_mean_mean"] for row in surface_rows]),
                "mean_Fano_300ms": _mean([row["Fano_median_300ms_mean"] for row in surface_rows]),
                "mean_peak_ratio": _mean([row["peak_ratio_mean"] for row in surface_rows]),
                "mean_noise_corr_50ms": _mean([row["mean_noise_corr_50ms_mean"] for row in surface_rows]),
                "pair_dir": str(pair_dir),
            }
        )

    _write_csv(out_dir / "aggregate_phase_table_all_pairs.csv", aggregate_phase_all)
    _write_csv(out_dir / "pair_overview.csv", pair_overview)

    impact_rows = _parameter_impact_rows(summary_rows)
    impact_summary = _aggregate_impact(impact_rows)
    _write_csv(out_dir / "parameter_impact_per_surface.csv", impact_rows)
    _write_csv(out_dir / "parameter_impact_summary.csv", impact_summary)
    _impact_heatmap(
        impact_summary,
        value_key="median_abs_spearman",
        out_path=out_dir / "parameter_impact_abs_spearman.png",
        title="Parameter Impact on Metrics: Median |Spearman rho|",
    )
    _impact_heatmap(
        impact_summary,
        value_key="median_effect_rel_range",
        out_path=out_dir / "parameter_impact_relative_range.png",
        title="Parameter Impact on Metrics: Median Relative Range",
    )
    _impact_heatmap(
        impact_summary,
        value_key="median_signed_pearson",
        out_path=out_dir / "parameter_impact_signed_pearson.png",
        title="Parameter Impact on Metrics: Median Signed Pearson r",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )

    impact_by_section_dir = out_dir / "parameter_impact_by_section"
    impact_by_section_dir.mkdir(parents=True, exist_ok=True)
    section_reports: Dict[str, Dict[str, object]] = {}
    section_groups: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in impact_rows:
        section_groups[str(row["section"])].append(row)
    section_summaries: Dict[str, List[Dict[str, object]]] = {}

    for section, section_rows in sorted(section_groups.items()):
        section_slug = _slugify(section)
        section_dir = impact_by_section_dir / section_slug
        section_dir.mkdir(parents=True, exist_ok=True)
        section_summary = _aggregate_impact(section_rows)
        section_summaries[section] = section_summary
        _write_csv(section_dir / "parameter_impact_per_surface.csv", section_rows)
        _write_csv(section_dir / "parameter_impact_summary.csv", section_summary)
        _impact_heatmap(
            section_summary,
            value_key="median_abs_spearman",
            out_path=section_dir / "parameter_impact_abs_spearman.png",
            title=f"{section}: Median |Spearman rho|",
        )
        _impact_heatmap(
            section_summary,
            value_key="median_effect_rel_range",
            out_path=section_dir / "parameter_impact_relative_range.png",
            title=f"{section}: Median Relative Range",
        )
        _impact_heatmap(
            section_summary,
            value_key="median_signed_pearson",
            out_path=section_dir / "parameter_impact_signed_pearson.png",
            title=f"{section}: Median Signed Pearson r",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
        )
        section_reports[section] = {
            "n_surface_axes": int(len(section_rows)),
            "top_parameters_by_metric": _top_parameters_by_metric(section_summary),
            "summary_csv": str(section_dir / "parameter_impact_summary.csv"),
        }

    _impact_comparison_heatmap(
        section_to_rows=section_summaries,
        value_key="median_abs_spearman",
        out_path=out_dir / "parameter_impact_abs_spearman_side_by_side.png",
        title="Parameter Impact Comparison by Topology: Median |Spearman rho|",
    )
    _impact_comparison_heatmap(
        section_to_rows=section_summaries,
        value_key="median_abs_spearman",
        out_path=out_dir / "parameter_impact_abs_spearman_side_by_side_paper.png",
        title="Parameter Impact Comparison by Topology: Median |Spearman rho|",
        paper_labels=True,
    )
    _impact_comparison_heatmap(
        section_to_rows=section_summaries,
        value_key="median_effect_rel_range",
        out_path=out_dir / "parameter_impact_relative_range_side_by_side.png",
        title="Parameter Impact Comparison by Topology: Median Relative Range",
    )
    _impact_comparison_heatmap(
        section_to_rows=section_summaries,
        value_key="median_effect_rel_range",
        out_path=out_dir / "parameter_impact_relative_range_side_by_side_paper.png",
        title="Parameter Impact Comparison by Topology: Median Relative Range",
        paper_labels=True,
    )
    _impact_comparison_heatmap(
        section_to_rows=section_summaries,
        value_key="median_signed_pearson",
        out_path=out_dir / "parameter_impact_signed_pearson_side_by_side.png",
        title="Parameter Impact Comparison by Topology: Median Signed Pearson r",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )
    _impact_comparison_heatmap(
        section_to_rows=section_summaries,
        value_key="median_signed_pearson",
        out_path=out_dir / "parameter_impact_signed_pearson_side_by_side_paper.png",
        title="Parameter Impact Comparison by Topology: Median Signed Pearson r",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        paper_labels=True,
    )

    report = {
        "batch_dir": str(batch_dir),
        "n_summary_rows": int(len(summary_rows)),
        "n_detail_rows": int(len(detail_rows)),
        "n_successful_job_surfaces": int(len(manifest_unique)),
        "n_parameter_pairs": int(len(pair_groups)),
        "aggregate_phase_pairs": sorted(_pair_slug(*key) for key in pair_groups.keys()),
        "top_parameters_by_metric": _top_parameters_by_metric(impact_summary),
        "top_parameters_by_section": section_reports,
    }
    _write_json(out_dir / "paper_results_report.json", report)


if __name__ == "__main__":
    main()
