import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize representative family metrics across families into a compact table "
            "with mean +- std and range."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Path to analyzed_trial_metrics.csv or a representative-only metrics table.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Output CSV path; defaults to <input stem>_across_families.csv",
    )
    parser.add_argument(
        "--analysis-dir",
        type=str,
        default=None,
        help="Analysis directory containing family_*/representative.json; defaults to the input CSV parent directory.",
    )
    parser.add_argument("--exc-weight", type=float, default=0.8, help="Weight for excitatory metrics when combining E/I pairs")
    parser.add_argument("--inh-weight", type=float, default=0.2, help="Weight for inhibitory metrics when combining E/I pairs")
    return parser


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_float(value: str) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _fmt(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.4g}"


def _read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_representative_values(analysis_dir: Path) -> Dict[str, List[float]]:
    params_by_name: Dict[str, List[float]] = {}
    objectives_by_name: Dict[str, List[float]] = {}
    for rep_path in sorted(analysis_dir.glob("family_*_trial_*/representative.json")):
        payload = _read_json(rep_path)
        for name, value in dict(payload.get("params", {})).items():
            params_by_name.setdefault(str(name), []).append(float(value))
        for name, value in dict(payload.get("objective_values", {})).items():
            objectives_by_name.setdefault(str(name), []).append(float(value))

    combined: Dict[str, List[float]] = {}
    for name, values in params_by_name.items():
        combined[f"param.{name}"] = values
    for name, values in objectives_by_name.items():
        combined[f"objective.{name}"] = values
    return combined


def _combine_metric_rows(rows: Sequence[Dict[str, str]], exc_weight: float, inh_weight: float) -> Dict[str, List[float]]:
    if not rows:
        return {}
    columns = list(rows[0].keys())
    ignored = {"cluster_id", "trial_number", "repeat_seed", "analysis_t_start_ms", "analysis_t_stop_ms"}
    allowed_scalar = {
        "mean_noise_corr_50ms",
        "psd_peak_ratio",
    }
    allowed_weighted_bases = {
        "ISI_CV_mean",
        "Fano_median_300ms",
        "mean_voltage_mV",
    }
    allowed_rate_bases = {"rate_mean_Hz"}

    combined: Dict[str, List[float]] = {}
    used = set()
    for col in columns:
        if col in ignored or col in used:
            continue
        if col.endswith("_E"):
            base = col[:-2]
            partner = f"{base}_I"
            if partner in columns:
                if base in allowed_rate_bases:
                    combined[col] = [_parse_float(row[col]) for row in rows]
                    combined[partner] = [_parse_float(row[partner]) for row in rows]
                elif base in allowed_weighted_bases:
                    combined[base] = [
                        exc_weight * _parse_float(row[col]) + inh_weight * _parse_float(row[partner])
                        for row in rows
                    ]
                else:
                    used.add(col)
                    used.add(partner)
                    continue
                used.add(col)
                used.add(partner)
                continue
        if col.endswith("_I") and f"{col[:-2]}_E" in columns:
            continue
        if col in allowed_scalar:
            combined[col] = [_parse_float(row[col]) for row in rows]
            used.add(col)
    return combined


def _metric_summary(values: Sequence[float]) -> Dict[str, str]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return {"mean+-std": "", "range": ""}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=0))
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    return {
        "mean+-std": f"{_fmt(mean)} +- {_fmt(std)}",
        "range": f"{_fmt(lo)}-{_fmt(hi)}",
    }


def _display_name(metric_name: str) -> str:
    labels = {
        "Fano_median_300ms": "Fano factor (300 ms)",
        "ISI_CV_mean": "ISI CV",
        "mean_noise_corr_50ms": "Mean pairwise correlation (50 ms)",
        "mean_voltage_mV": "Mean membrane potential (mV)",
        "psd_peak_ratio": "PSD peak ratio",
        "rate_mean_Hz_E": "Excitatory firing rate (Hz)",
        "rate_mean_Hz_I": "Inhibitory firing rate (Hz)",
        "objective.asynchrony": "Asynchrony objective",
        "objective.persistence": "Persistence objective",
        "objective.regime": "Regime objective",
        "objective.transmitter_balance": "Transmitter balance objective",
        "param.A_AMPA": "AMPA amplitude",
        "param.A_GABA_A": "GABA_A amplitude",
        "param.A_GABA_B": "GABA_B amplitude",
        "param.A_NMDA": "NMDA amplitude",
        "param.external_amplitude": "External amplitude",
        "param.g_AMPA_max": "AMPA maximal conductance",
        "param.g_GABA_A_max": "GABA_A maximal conductance",
        "param.g_GABA_B_max": "GABA_B maximal conductance",
        "param.g_NMDA_max": "NMDA maximal conductance",
        "param.inhibitory_nmda_weight": "Inhibitory NMDA weight",
        "param.inhibitory_scale_g": "Inhibitory strength",
        "param.recurrent_exc_lognorm_sigma": "Recurrent excitatory sigma",
    }
    return labels.get(metric_name, metric_name)


def main() -> None:
    args = _build_parser().parse_args()
    input_csv = Path(args.input_csv).resolve()
    output_csv = Path(args.output_csv).resolve() if args.output_csv else input_csv.with_name(f"{input_csv.stem}_across_families.csv")
    analysis_dir = Path(args.analysis_dir).resolve() if args.analysis_dir else input_csv.parent
    rows = _read_csv(input_csv)
    metric_map = _combine_metric_rows(rows, float(args.exc_weight), float(args.inh_weight))
    metric_map.update(_load_representative_values(analysis_dir))

    summary_rows: List[Dict[str, str]] = []
    ordered_names = [
        "objective.persistence",
        "objective.asynchrony",
        "objective.regime",
        "objective.transmitter_balance",
        "param.g_AMPA_max",
        "param.g_NMDA_max",
        "param.g_GABA_A_max",
        "param.g_GABA_B_max",
        "param.A_AMPA",
        "param.A_NMDA",
        "param.A_GABA_A",
        "param.A_GABA_B",
        "param.inhibitory_scale_g",
        "param.external_amplitude",
        "param.recurrent_exc_lognorm_sigma",
        "param.inhibitory_nmda_weight",
        "ISI_CV_mean",
        "Fano_median_300ms",
        "mean_noise_corr_50ms",
        "psd_peak_ratio",
        "rate_mean_Hz_E",
        "rate_mean_Hz_I",
    ]
    for metric_name in [name for name in ordered_names if name in metric_map]:
        summary = _metric_summary(metric_map[metric_name])
        summary_rows.append(
            {
                "metric": _display_name(metric_name),
                "mean+-std": summary["mean+-std"],
                "range": summary["range"],
            }
        )

    _write_csv(output_csv, summary_rows)
    print(output_csv)


if __name__ == "__main__":
    main()
