import argparse
import ast
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


CLASS_ORDER = ("AI", "AR", "SI", "SR", "QUIESCENT", "NON_PERSISTENT")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize robustness post-analysis outputs into family-level tables, "
            "class-transition summaries, parameter-delta tables, and overview plots."
        )
    )
    parser.add_argument(
        "--robustness-dir",
        type=str,
        required=True,
        help="Directory containing robustness_detail.csv and robustness_summary.json.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory; defaults to <robustness-dir>/analysis.",
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
        if key == "params_snapshot":
            try:
                out[key] = ast.literal_eval(value)
            except Exception:
                out[key] = value
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


def _group_rows(rows: Sequence[Mapping[str, object]], key: str) -> Dict[object, List[Mapping[str, object]]]:
    out: Dict[object, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        out[row.get(key)].append(row)
    return out


def _safe_mean(values: Iterable[object]) -> float:
    vals = []
    for value in values:
        if isinstance(value, str):
            continue
        val = float(value)
        if math.isfinite(val):
            vals.append(val)
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _safe_std(values: Iterable[object]) -> float:
    vals = []
    for value in values:
        if isinstance(value, str):
            continue
        val = float(value)
        if math.isfinite(val):
            vals.append(val)
    if len(vals) < 2:
        return 0.0
    return float(np.std(vals))


def _fraction(rows: Sequence[Mapping[str, object]], key: str, target: object = 1) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([1.0 if row.get(key) == target else 0.0 for row in rows]))


def _mode(rows: Sequence[Mapping[str, object]], key: str) -> str:
    counts = Counter(str(row.get(key, "")) for row in rows)
    if not counts:
        return ""
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _class_counts(rows: Sequence[Mapping[str, object]], key: str) -> Dict[str, int]:
    counts = Counter(str(row.get(key, "")) for row in rows)
    for cls in CLASS_ORDER:
        counts.setdefault(cls, 0)
    return {cls: int(counts[cls]) for cls in CLASS_ORDER}


def _family_overview(detail_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    by_family = _group_rows(detail_rows, "family_id")
    out: List[Dict[str, object]] = []
    for family_id in sorted(by_family.keys()):
        fam_rows = by_family[family_id]
        member_rows = [row for row in fam_rows if row.get("analysis_type") == "family_member"]
        perm_rows = [row for row in fam_rows if row.get("analysis_type") != "family_member"]
        row = {
            "family_id": int(family_id),
            "family_size": int(_safe_mean(r.get("family_size", 0) for r in fam_rows)),
            "n_member_runs": int(len(member_rows)),
            "n_perturbation_runs": int(len(perm_rows)),
            "member_mode_class": _mode(member_rows, "persistent_activity_class"),
            "perm_mode_class": _mode(perm_rows, "persistent_activity_class"),
            "member_persistence_fraction": _fraction(member_rows, "is_persistent"),
            "perm_persistence_fraction": _fraction(perm_rows, "is_persistent"),
            "member_ssai_fraction": _fraction(member_rows, "is_ssai"),
            "perm_ssai_fraction": _fraction(perm_rows, "is_ssai"),
            "member_sync_fraction": _fraction(member_rows, "is_synchronous"),
            "perm_sync_fraction": _fraction(perm_rows, "is_synchronous"),
            "member_rate_late_Hz_mean": _safe_mean(r.get("rate_late_Hz", float("nan")) for r in member_rows),
            "perm_rate_late_Hz_mean": _safe_mean(r.get("rate_late_Hz", float("nan")) for r in perm_rows),
            "member_rate_late_Hz_std": _safe_std(r.get("rate_late_Hz", float("nan")) for r in member_rows),
            "perm_rate_late_Hz_std": _safe_std(r.get("rate_late_Hz", float("nan")) for r in perm_rows),
            "member_peak_ratio_mean": _safe_mean(r.get("peak_ratio", float("nan")) for r in member_rows),
            "perm_peak_ratio_mean": _safe_mean(r.get("peak_ratio", float("nan")) for r in perm_rows),
            "member_to_perm_persistence_drop": _fraction(member_rows, "is_persistent") - _fraction(perm_rows, "is_persistent"),
            "member_to_perm_ssai_drop": _fraction(member_rows, "is_ssai") - _fraction(perm_rows, "is_ssai"),
        }
        out.append(row)
    return out


def _transition_table(detail_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    by_family = _group_rows(detail_rows, "family_id")
    out: List[Dict[str, object]] = []
    for family_id in sorted(by_family.keys()):
        fam_rows = by_family[family_id]
        for analysis_type in sorted({str(row.get("analysis_type")) for row in fam_rows}):
            subset = [row for row in fam_rows if row.get("analysis_type") == analysis_type]
            if not subset:
                continue
            counts = _class_counts(subset, "persistent_activity_class")
            row = {
                "family_id": int(family_id),
                "analysis_type": analysis_type,
                "n_runs": int(len(subset)),
            }
            for cls in CLASS_ORDER:
                row[f"class_count.{cls}"] = int(counts[cls])
                row[f"class_fraction.{cls}"] = float(counts[cls] / max(1, len(subset)))
            out.append(row)
    return out


def _member_trial_summary(detail_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple, List[Mapping[str, object]]] = defaultdict(list)
    for row in detail_rows:
        if row.get("analysis_type") != "family_member":
            continue
        grouped[(int(row["family_id"]), int(row["trial_number"]))].append(row)

    out: List[Dict[str, object]] = []
    for (family_id, trial_number), rows in sorted(grouped.items()):
        out.append(
            {
                "family_id": family_id,
                "trial_number": trial_number,
                "n_runs": int(len(rows)),
                "mode_class": _mode(rows, "persistent_activity_class"),
                "persistence_fraction": _fraction(rows, "is_persistent"),
                "ssai_fraction": _fraction(rows, "is_ssai"),
                "sync_fraction": _fraction(rows, "is_synchronous"),
                "rate_late_Hz_mean": _safe_mean(r.get("rate_late_Hz", float("nan")) for r in rows),
                "rate_late_Hz_std": _safe_std(r.get("rate_late_Hz", float("nan")) for r in rows),
                "peak_ratio_mean": _safe_mean(r.get("peak_ratio", float("nan")) for r in rows),
                "isi_cv_mean": _safe_mean(r.get("ISI_CV_mean", float("nan")) for r in rows),
            }
        )
    return out


def _parameter_delta_summary(detail_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    member_examples: Dict[int, Mapping[str, object]] = {}
    for row in detail_rows:
        if row.get("analysis_type") == "family_member" and isinstance(row.get("params_snapshot"), dict):
            family_id = int(row["family_id"])
            member_examples.setdefault(family_id, row)

    perm_rows = [row for row in detail_rows if row.get("analysis_type") != "family_member"]
    if not perm_rows:
        return []

    parameter_names = sorted(
        {
            name
            for row in perm_rows
            for name in (row.get("params_snapshot", {}) if isinstance(row.get("params_snapshot"), dict) else {})
        }
    )

    grouped: Dict[tuple, List[Mapping[str, object]]] = defaultdict(list)
    for row in perm_rows:
        grouped[(int(row["family_id"]), int(row.get("permutation_idx", -1)))].append(row)

    out: List[Dict[str, object]] = []
    for (family_id, permutation_idx), rows in sorted(grouped.items()):
        ref = member_examples.get(family_id, {})
        ref_params = ref.get("params_snapshot", {}) if isinstance(ref, dict) else {}
        perm_params = rows[0].get("params_snapshot", {}) if isinstance(rows[0].get("params_snapshot"), dict) else {}
        summary = {
            "family_id": family_id,
            "permutation_idx": permutation_idx,
            "n_runs": int(len(rows)),
            "mode_class": _mode(rows, "persistent_activity_class"),
            "persistence_fraction": _fraction(rows, "is_persistent"),
            "ssai_fraction": _fraction(rows, "is_ssai"),
            "rate_late_Hz_mean": _safe_mean(r.get("rate_late_Hz", float("nan")) for r in rows),
        }
        for name in parameter_names:
            if name not in perm_params:
                continue
            perm_value = float(perm_params[name])
            ref_value = float(ref_params.get(name, float("nan"))) if isinstance(ref_params, dict) and name in ref_params else float("nan")
            summary[f"param.{name}"] = perm_value
            summary[f"delta.{name}"] = perm_value - ref_value if math.isfinite(ref_value) else float("nan")
        out.append(summary)
    return out


def _plot_retention(overview_rows: Sequence[Mapping[str, object]], out_path: Path) -> None:
    if not overview_rows:
        return
    family_ids = [int(row["family_id"]) for row in overview_rows]
    member = [float(row["member_persistence_fraction"]) for row in overview_rows]
    perm = [float(row["perm_persistence_fraction"]) for row in overview_rows]
    x = np.arange(len(family_ids))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, member, width=width, label="Member replay")
    ax.bar(x + width / 2, perm, width=width, label="Parameter permutation")
    ax.set_xticks(x)
    ax.set_xticklabels([str(fid) for fid in family_ids])
    ax.set_xlabel("Family ID")
    ax.set_ylabel("Persistence fraction")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Persistence Retention by Family")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_class_mix(transitions: Sequence[Mapping[str, object]], out_path: Path) -> None:
    if not transitions:
        return
    labels = [f"F{int(row['family_id'])}-{str(row['analysis_type'])[:4]}" for row in transitions]
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels), dtype=float)

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap("tab20")
    for idx, cls in enumerate(CLASS_ORDER):
        vals = np.array([float(row.get(f"class_fraction.{cls}", 0.0)) for row in transitions], dtype=float)
        ax.bar(x, vals, bottom=bottom, color=cmap(idx), label=cls)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Activity-Class Composition")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_permutation_scatter(delta_rows: Sequence[Mapping[str, object]], out_path: Path) -> None:
    if not delta_rows:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    xs = [float(row.get("rate_late_Hz_mean", float("nan"))) for row in delta_rows]
    ys = [float(row.get("persistence_fraction", float("nan"))) for row in delta_rows]
    family_ids = [int(row["family_id"]) for row in delta_rows]
    scatter = ax.scatter(xs, ys, c=family_ids, cmap="tab10", s=60, alpha=0.85)
    ax.set_xlabel("Late firing rate mean (Hz)")
    ax.set_ylabel("Persistence fraction")
    ax.set_title("Permutation Outcomes")
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Family ID")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = _build_parser().parse_args()
    robustness_dir = Path(args.robustness_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (robustness_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_rows = _read_csv(robustness_dir / "robustness_detail.csv")
    summary_json_path = robustness_dir / "robustness_summary.json"
    summary_json = {}
    if summary_json_path.exists():
        with summary_json_path.open("r", encoding="utf-8") as f:
            summary_json = json.load(f)

    family_overview = _family_overview(detail_rows)
    transitions = _transition_table(detail_rows)
    member_trial_rows = _member_trial_summary(detail_rows)
    parameter_delta_rows = _parameter_delta_summary(detail_rows)

    _write_csv(out_dir / "family_overview.csv", family_overview)
    _write_csv(out_dir / "class_transitions.csv", transitions)
    _write_csv(out_dir / "member_trial_summary.csv", member_trial_rows)
    _write_csv(out_dir / "permutation_parameter_delta_summary.csv", parameter_delta_rows)

    report = {
        "n_detail_rows": int(len(detail_rows)),
        "n_families": int(len({int(row["family_id"]) for row in detail_rows})),
        "family_overview": family_overview,
        "input_summary": summary_json,
    }
    _write_json(out_dir / "robustness_analysis_report.json", report)

    _plot_retention(family_overview, out_dir / "persistence_retention.png")
    _plot_class_mix(transitions, out_dir / "activity_class_mix.png")
    _plot_permutation_scatter(parameter_delta_rows, out_dir / "permutation_outcomes.png")


if __name__ == "__main__":
    main()
