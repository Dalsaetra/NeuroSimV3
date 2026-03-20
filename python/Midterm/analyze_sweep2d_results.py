import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors


CLASS_ORDER = ("AI", "AR", "SI", "SR", "QUIESCENT", "NON_PERSISTENT")
CLASS_TO_CODE = {name: idx for idx, name in enumerate(CLASS_ORDER)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze 2D sweep outputs into phase diagrams, summary tables, and per-trial reports."
        )
    )
    parser.add_argument(
        "--sweep2d-dir",
        type=str,
        required=True,
        help="Directory containing sweep2d_detail.csv / sweep2d_summary.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory; defaults to <sweep2d-dir>/analysis.",
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


def _group_by_trial(rows: Sequence[Mapping[str, object]]) -> Dict[int, List[Mapping[str, object]]]:
    grouped: Dict[int, List[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(int(row["trial_number"]), []).append(row)
    return grouped


def _sorted_unique(values: Sequence[object]) -> List[object]:
    return sorted(set(values), key=lambda x: (str(type(x)), float(x) if isinstance(x, (int, float)) else str(x)))


def _mode(values: Sequence[object]) -> object:
    items = [value for value in values if value != ""]
    if not items:
        return ""
    counts = Counter(items)
    return sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[0][0]


def _mean(values: Sequence[object]) -> float:
    vals = []
    for value in values:
        if value == "":
            continue
        try:
            val = float(value)
        except Exception:
            continue
        if math.isfinite(val):
            vals.append(val)
    return float(np.mean(vals)) if vals else float("nan")


def _axis_value(row: Mapping[str, object], prefix: str) -> object:
    explicit = row.get(f"{prefix}_axis_value", "")
    if explicit != "":
        return explicit
    mode = str(row.get(f"{prefix}_mode", ""))
    factor = row.get(f"{prefix}_factor", "")
    if mode == "relative" and factor != "":
        return factor
    return row.get(f"{prefix}_value", "")


def _axis_label(rows: Sequence[Mapping[str, object]], prefix: str) -> str:
    name = str(rows[0].get(f"{prefix}_name", prefix))
    mode = str(rows[0].get(f"{prefix}_mode", ""))
    return f"{name} (relative factor)" if mode == "relative" else name


def _format_tick_label(value: object) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.3g}"
    return str(value)


def _aggregate_detail_rows(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            int(row["trial_number"]),
            str(row["x_name"]),
            int(row.get("x_index", 0)),
            str(row["y_name"]),
            int(row.get("y_index", 0)),
        )
        grouped[key].append(row)

    out: List[Dict[str, object]] = []
    for key in sorted(grouped.keys()):
        group = grouped[key]
        out.append(
            {
                "trial_number": key[0],
                "x_name": key[1],
                "x_index": key[2],
                "x_value": _axis_value(group[0], "x"),
                "x_mode": _mode([row.get("x_mode", "") for row in group]),
                "y_name": key[3],
                "y_index": key[4],
                "y_value": _axis_value(group[0], "y"),
                "y_mode": _mode([row.get("y_mode", "") for row in group]),
                "mode_persistent_activity_class": _mode([row.get("persistent_activity_class", "") for row in group]),
                "persistence_fraction": _mean([row.get("is_persistent", 0) for row in group]),
                "ssai_fraction": _mean([row.get("is_ssai", 0) for row in group]),
                "synchronous_fraction": _mean([row.get("is_synchronous", 0) for row in group]),
                "rate_late_Hz_mean": _mean([row.get("rate_late_Hz", float("nan")) for row in group]),
                "rate_post_Hz_mean": _mean([row.get("rate_post_Hz", float("nan")) for row in group]),
                "ISI_CV_mean_mean": _mean([row.get("ISI_CV_mean", float("nan")) for row in group]),
                "peak_ratio_mean": _mean([row.get("peak_ratio", float("nan")) for row in group]),
                "oscillation_freq_hz_mean": _mean([row.get("oscillation_freq_hz", float("nan")) for row in group]),
            }
        )
    return out


def _make_grid(
    rows: Sequence[Mapping[str, object]],
    *,
    x_key: str,
    y_key: str,
    value_key: str,
    value_transform=None,
):
    x_indexed = all("x_index" in row for row in rows)
    y_indexed = all("y_index" in row for row in rows)
    if x_indexed:
        x_pairs = sorted({(int(row["x_index"]), row[x_key]) for row in rows}, key=lambda item: item[0])
        x_vals = [pair[1] for pair in x_pairs]
        x_lookup = {pair[0]: idx for idx, pair in enumerate(x_pairs)}
    else:
        x_vals = _sorted_unique([row[x_key] for row in rows])
        x_lookup = {value: idx for idx, value in enumerate(x_vals)}
    if y_indexed:
        y_pairs = sorted({(int(row["y_index"]), row[y_key]) for row in rows}, key=lambda item: item[0])
        y_vals = [pair[1] for pair in y_pairs]
        y_lookup = {pair[0]: idx for idx, pair in enumerate(y_pairs)}
    else:
        y_vals = _sorted_unique([row[y_key] for row in rows])
        y_lookup = {value: idx for idx, value in enumerate(y_vals)}
    grid = np.full((len(y_vals), len(x_vals)), np.nan, dtype=float)
    for row in rows:
        x_idx = x_lookup[int(row["x_index"])] if x_indexed else x_lookup[row[x_key]]
        y_idx = y_lookup[int(row["y_index"])] if y_indexed else y_lookup[row[y_key]]
        value = row.get(value_key, float("nan"))
        if value_transform is not None:
            value = value_transform(value)
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
    vmin=None,
    vmax=None,
    colorbar_label: str = "",
    class_mode: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    if class_mode:
        discrete_cmap = plt.get_cmap(cmap, len(CLASS_ORDER))
        bounds = np.arange(-0.5, len(CLASS_ORDER) + 0.5, 1.0)
        norm = colors.BoundaryNorm(bounds, discrete_cmap.N)
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap=discrete_cmap, norm=norm)
    else:
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(x_vals)))
    ax.set_yticks(np.arange(len(y_vals)))
    ax.set_xticklabels([_format_tick_label(v) for v in x_vals], rotation=45, ha="right")
    ax.set_yticklabels([_format_tick_label(v) for v in y_vals])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, boundaries=bounds if class_mode else None)
    cbar.set_label(colorbar_label)
    if class_mode:
        cbar.set_ticks(list(CLASS_TO_CODE.values()))
        cbar.set_ticklabels(list(CLASS_TO_CODE.keys()))

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _phase_table(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        out.append(
            {
                "trial_number": int(row["trial_number"]),
                "x_name": row["x_name"],
                "x_value": row["x_value"],
                "x_mode": row.get("x_mode", ""),
                "y_name": row["y_name"],
                "y_value": row["y_value"],
                "y_mode": row.get("y_mode", ""),
                "mode_persistent_activity_class": row.get("mode_persistent_activity_class", ""),
                "persistence_fraction": row.get("persistence_fraction", float("nan")),
                "ssai_fraction": row.get("ssai_fraction", float("nan")),
                "synchronous_fraction": row.get("synchronous_fraction", float("nan")),
                "rate_late_Hz_mean": row.get("rate_late_Hz_mean", float("nan")),
                "rate_post_Hz_mean": row.get("rate_post_Hz_mean", float("nan")),
                "ISI_CV_mean_mean": row.get("ISI_CV_mean_mean", float("nan")),
                "peak_ratio_mean": row.get("peak_ratio_mean", float("nan")),
                "oscillation_freq_hz_mean": row.get("oscillation_freq_hz_mean", float("nan")),
            }
        )
    return out


def _trial_report(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    if not rows:
        return {}
    x_name = str(rows[0]["x_name"])
    y_name = str(rows[0]["y_name"])
    best_persistence = max(rows, key=lambda r: float(r.get("persistence_fraction", -1.0)))
    best_ssai = max(rows, key=lambda r: float(r.get("ssai_fraction", -1.0)))
    dominant_classes = {}
    for cls in CLASS_ORDER:
        dominant_classes[cls] = int(sum(1 for row in rows if row.get("mode_persistent_activity_class") == cls))
    return {
        "trial_number": int(rows[0]["trial_number"]),
        "x_name": x_name,
        "y_name": y_name,
        "n_grid_points": int(len(rows)),
        "best_persistence_point": {
            "x_value": best_persistence["x_value"],
            "y_value": best_persistence["y_value"],
            "persistence_fraction": best_persistence.get("persistence_fraction", float("nan")),
            "mode_class": best_persistence.get("mode_persistent_activity_class", ""),
        },
        "best_ssai_point": {
            "x_value": best_ssai["x_value"],
            "y_value": best_ssai["y_value"],
            "ssai_fraction": best_ssai.get("ssai_fraction", float("nan")),
            "mode_class": best_ssai.get("mode_persistent_activity_class", ""),
        },
        "dominant_class_counts": dominant_classes,
    }


def _analyze_trial(rows: Sequence[Mapping[str, object]], out_dir: Path) -> Dict[str, object]:
    trial_number = int(rows[0]["trial_number"])
    trial_dir = out_dir / f"trial_{trial_number}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    x_label = _axis_label(rows, "x")
    y_label = _axis_label(rows, "y")

    phase_rows = _phase_table(rows)
    _write_csv(trial_dir / "phase_table.csv", phase_rows)

    x_vals, y_vals, class_grid = _make_grid(
        rows,
        x_key="x_value",
        y_key="y_value",
        value_key="mode_persistent_activity_class",
        value_transform=lambda x: CLASS_TO_CODE.get(str(x), np.nan),
    )
    _, _, persistence_grid = _make_grid(rows, x_key="x_value", y_key="y_value", value_key="persistence_fraction")
    _, _, ssai_grid = _make_grid(rows, x_key="x_value", y_key="y_value", value_key="ssai_fraction")
    _, _, sync_grid = _make_grid(rows, x_key="x_value", y_key="y_value", value_key="synchronous_fraction")
    _, _, late_rate_grid = _make_grid(rows, x_key="x_value", y_key="y_value", value_key="rate_late_Hz_mean")
    _, _, cv_grid = _make_grid(rows, x_key="x_value", y_key="y_value", value_key="ISI_CV_mean_mean")
    _, _, peak_grid = _make_grid(rows, x_key="x_value", y_key="y_value", value_key="peak_ratio_mean")
    _, _, freq_grid = _make_grid(rows, x_key="x_value", y_key="y_value", value_key="oscillation_freq_hz_mean")

    _plot_heatmap(
        grid=class_grid,
        x_vals=x_vals,
        y_vals=y_vals,
        title=f"Trial {trial_number}: Mode Persistent Class",
        x_label=x_label,
        y_label=y_label,
        out_path=trial_dir / "phase_mode_class.png",
        cmap="tab20",
        vmin=0,
        vmax=len(CLASS_ORDER) - 1,
        colorbar_label="Class",
        class_mode=True,
    )
    _plot_heatmap(
        grid=persistence_grid,
        x_vals=x_vals,
        y_vals=y_vals,
        title=f"Trial {trial_number}: Persistence Fraction",
        x_label=x_label,
        y_label=y_label,
        out_path=trial_dir / "phase_persistence_fraction.png",
        vmin=0.0,
        vmax=1.0,
        colorbar_label="Fraction",
    )
    _plot_heatmap(
        grid=ssai_grid,
        x_vals=x_vals,
        y_vals=y_vals,
        title=f"Trial {trial_number}: SSAI Fraction",
        x_label=x_label,
        y_label=y_label,
        out_path=trial_dir / "phase_ssai_fraction.png",
        vmin=0.0,
        vmax=1.0,
        colorbar_label="Fraction",
    )
    _plot_heatmap(
        grid=sync_grid,
        x_vals=x_vals,
        y_vals=y_vals,
        title=f"Trial {trial_number}: Synchrony Fraction",
        x_label=x_label,
        y_label=y_label,
        out_path=trial_dir / "phase_synchrony_fraction.png",
        vmin=0.0,
        vmax=1.0,
        colorbar_label="Fraction",
    )
    _plot_heatmap(
        grid=late_rate_grid,
        x_vals=x_vals,
        y_vals=y_vals,
        title=f"Trial {trial_number}: Late Rate Mean",
        x_label=x_label,
        y_label=y_label,
        out_path=trial_dir / "phase_rate_late_hz.png",
        colorbar_label="Hz",
    )
    _plot_heatmap(
        grid=cv_grid,
        x_vals=x_vals,
        y_vals=y_vals,
        title=f"Trial {trial_number}: ISI CV Mean",
        x_label=x_label,
        y_label=y_label,
        out_path=trial_dir / "phase_isi_cv.png",
        colorbar_label="CV",
    )
    _plot_heatmap(
        grid=peak_grid,
        x_vals=x_vals,
        y_vals=y_vals,
        title=f"Trial {trial_number}: PSD Peak Ratio Mean",
        x_label=x_label,
        y_label=y_label,
        out_path=trial_dir / "phase_peak_ratio.png",
        colorbar_label="Peak ratio",
    )
    _plot_heatmap(
        grid=freq_grid,
        x_vals=x_vals,
        y_vals=y_vals,
        title=f"Trial {trial_number}: Oscillation Frequency Mean",
        x_label=x_label,
        y_label=y_label,
        out_path=trial_dir / "phase_oscillation_frequency.png",
        colorbar_label="Hz",
    )

    report = _trial_report(rows)
    _write_json(trial_dir / "trial_report.json", report)
    return report


def main() -> None:
    args = _build_parser().parse_args()
    sweep2d_dir = Path(args.sweep2d_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (sweep2d_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_rows = _read_csv(sweep2d_dir / "sweep2d_detail.csv")
    summary_rows = _aggregate_detail_rows(detail_rows) if detail_rows else _read_csv(sweep2d_dir / "sweep2d_summary.csv")
    grouped_summary = _group_by_trial(summary_rows)
    grouped_detail = _group_by_trial(detail_rows)

    phase_table_all = _phase_table(summary_rows)
    _write_csv(out_dir / "phase_table_all_trials.csv", phase_table_all)

    reports = []
    for trial_number in sorted(grouped_summary.keys()):
        report = _analyze_trial(grouped_summary[trial_number], out_dir)
        reports.append(report)

    aggregate = {
        "n_trials": int(len(grouped_summary)),
        "n_summary_rows": int(len(summary_rows)),
        "n_detail_rows": int(len(detail_rows)),
        "trials": reports,
        "trial_numbers_with_detail": sorted(int(x) for x in grouped_detail.keys()),
    }
    _write_json(out_dir / "sweep2d_analysis_report.json", aggregate)


if __name__ == "__main__":
    main()
