import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


METRIC_COLUMNS = [
    "is_active",
    "is_persistent",
    "is_ssai",
    "is_synchronous",
    "rate_post_Hz",
    "rate_post_Hz_E",
    "rate_post_Hz_I",
    "rate_late_Hz",
    "rate_late_Hz_E",
    "rate_late_Hz_I",
    "ISI_CV_mean",
    "ISI_CV_mean_E",
    "ISI_CV_mean_I",
    "Fano_median_300ms",
    "peak_ratio",
    "mean_noise_corr_50ms",
    "oscillation_freq_hz",
    "participation_frac_total",
    "pop_spec_entropy",
]

PAPER_LABELS = {
    "is_active": "Active Fraction",
    "is_persistent": "Persistence Fraction",
    "is_ssai": "SSAI Fraction",
    "is_synchronous": "Synchronous Fraction",
    "rate_post_Hz": "Post-Stimulus Firing Rate",
    "rate_post_Hz_E": "Post-Stimulus Excitatory Firing Rate",
    "rate_post_Hz_I": "Post-Stimulus Inhibitory Firing Rate",
    "rate_late_Hz": "Late Firing Rate",
    "rate_late_Hz_E": "Late Excitatory Firing Rate",
    "rate_late_Hz_I": "Late Inhibitory Firing Rate",
    "ISI_CV_mean": "ISI CV",
    "ISI_CV_mean_E": "Excitatory ISI CV",
    "ISI_CV_mean_I": "Inhibitory ISI CV",
    "Fano_median_300ms": "Fano Factor (300 ms)",
    "peak_ratio": "PSD Peak Ratio",
    "mean_noise_corr_50ms": "Mean Pairwise Correlation (50 ms)",
    "oscillation_freq_hz": "Oscillation Frequency (Hz)",
    "participation_frac_total": "Participation Fraction",
    "pop_spec_entropy": "Population Spectrum Entropy",
    "baseline": "Dense Random Topology",
    "spatial_ei": "Realistic Topology",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize batch sweep detail metrics by topology.")
    parser.add_argument("--batch-dir", type=str, required=True, help="Batch sweep directory containing sweep2d_detail.csv files")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory; defaults to <batch-dir>/paper_analysis",
    )
    return parser


def _fmt(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.4g}"


def _load_detail_df(batch_dir: Path) -> pd.DataFrame:
    files = sorted(batch_dir.rglob("sweep2d_detail.csv"))
    if not files:
        raise ValueError(f"No sweep2d_detail.csv files found under {batch_dir}")
    return pd.concat([pd.read_csv(path) for path in files], ignore_index=True)


def _summary_rows(df: pd.DataFrame) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for topology, topo_df in df.groupby("topology"):
        row: Dict[str, object] = {
            "topology": str(topology),
            "topology_label": PAPER_LABELS.get(str(topology), str(topology)),
            "n_runs": int(len(topo_df)),
        }
        for metric in METRIC_COLUMNS:
            if metric not in topo_df.columns:
                continue
            values = pd.to_numeric(topo_df[metric], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(np.mean(finite)) if finite.size else float("nan")
            row[f"{metric}_std"] = float(np.std(finite, ddof=0)) if finite.size else float("nan")
        rows.append(row)
    return rows


def _compact_rows(summary_rows: Sequence[Dict[str, object]]) -> List[Dict[str, str]]:
    topology_order = ["baseline", "spatial_ei"]
    rows: List[Dict[str, str]] = []
    for summary in sorted(summary_rows, key=lambda row: topology_order.index(row["topology"]) if row["topology"] in topology_order else 99):
        row: Dict[str, str] = {
            "topology": str(summary["topology_label"]),
            "n_runs": str(summary["n_runs"]),
        }
        for metric in METRIC_COLUMNS:
            mean_key = f"{metric}_mean"
            std_key = f"{metric}_std"
            if mean_key not in summary:
                continue
            row[PAPER_LABELS.get(metric, metric)] = f"{_fmt(float(summary[mean_key]))} +- {_fmt(float(summary[std_key]))}"
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _build_parser().parse_args()
    batch_dir = Path(args.batch_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (batch_dir / "paper_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_detail_df(batch_dir)
    summary_rows = _summary_rows(df)
    compact_rows = _compact_rows(summary_rows)

    _write_csv(out_dir / "topology_metric_summary_raw.csv", summary_rows)
    _write_csv(out_dir / "topology_metric_summary_compact.csv", compact_rows)
    print(out_dir / "topology_metric_summary_compact.csv")


if __name__ == "__main__":
    main()
