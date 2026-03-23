import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Mapping, Tuple


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reformat family_summary_statistics.csv into a cleaner transposed table with "
            "'mean +- std' and 'range' columns for each family."
        )
    )
    parser.add_argument("--input-csv", type=str, required=True, help="Path to family_summary_statistics.csv")
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Output CSV path; defaults to <input stem>_clean_transposed.csv",
    )
    return parser


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Mapping[str, str]]) -> None:
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


def _fmt_number(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.4g}"


def _clean_feature_name(base_name: str) -> str:
    return base_name.replace("metric.", "").replace("objective.", "").replace("param.", "")


def _collect_base_names(columns: List[str]) -> List[str]:
    base_names = set()
    suffixes = (".mean", ".std", ".min", ".max")
    for col in columns:
        for suffix in suffixes:
            if col.endswith(suffix):
                base_names.add(col[: -len(suffix)])
                break
    return sorted(base_names)


def _family_sort_key(row: Mapping[str, str]) -> Tuple[int, str]:
    cluster_str = row.get("cluster_id", "")
    try:
        return (int(cluster_str), cluster_str)
    except ValueError:
        return (10**9, cluster_str)


def _collapse_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not rows:
        return []
    rows = sorted(rows, key=_family_sort_key)
    base_names = _collect_base_names(list(rows[0].keys()))
    out_rows: List[Dict[str, str]] = []

    for base_name in base_names:
        clean_name = _clean_feature_name(base_name)
        mean_std_row: Dict[str, str] = {"feature": clean_name, "summary": "mean +- std"}
        range_row: Dict[str, str] = {"feature": clean_name, "summary": "range"}
        for row in rows:
            family_label = f"family_{int(row['cluster_id']):02d}"
            mean_val = _parse_float(row.get(f"{base_name}.mean", ""))
            std_val = _parse_float(row.get(f"{base_name}.std", ""))
            min_val = _parse_float(row.get(f"{base_name}.min", ""))
            max_val = _parse_float(row.get(f"{base_name}.max", ""))
            mean_std_row[family_label] = f"{_fmt_number(mean_val)} +- {_fmt_number(std_val)}".strip()
            if not math.isfinite(mean_val) and not math.isfinite(std_val):
                mean_std_row[family_label] = ""
            range_row[family_label] = ""
            if math.isfinite(min_val) or math.isfinite(max_val):
                range_row[family_label] = f"{_fmt_number(min_val)}-{_fmt_number(max_val)}"
        out_rows.append(mean_std_row)
        out_rows.append(range_row)
    return out_rows


def main() -> None:
    args = _build_parser().parse_args()
    input_csv = Path(args.input_csv).resolve()
    output_csv = Path(args.output_csv).resolve() if args.output_csv else input_csv.with_name(f"{input_csv.stem}_clean_transposed.csv")
    rows = _read_csv(input_csv)
    collapsed_rows = _collapse_rows(rows)
    _write_csv(output_csv, collapsed_rows)
    print(output_csv)


if __name__ == "__main__":
    main()
