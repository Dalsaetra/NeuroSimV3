import argparse
import csv
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "week14" / "results" / "threshold_voltage_heterogeneity_grid_search.csv"
JOB_FILE_RE = re.compile(r"^(?P<stem>.+)\.job(?P<job>\d+)of(?P<total>\d+)(?P<suffix>\.[^.]+)$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge per-job threshold heterogeneity grid search CSV files into one CSV."
    )
    parser.add_argument(
        "--input-base",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Base CSV path used by the sweep script before job suffixes are added. "
            "Example: week14/results/threshold_voltage_heterogeneity_grid_search.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Merged CSV destination. Defaults to --input-base.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the merged output if it already exists.",
    )
    return parser.parse_args()


def trial_key(row):
    return (
        row.get("topology", ""),
        row.get("topology_seed", ""),
        row.get("variance_ss4_vt", ""),
        row.get("variance_b_vt", ""),
    )


def discover_job_files(input_base):
    input_base = input_base.resolve()
    parent = input_base.parent
    stem = input_base.stem
    suffix = input_base.suffix
    pattern = f"{stem}.job*of*{suffix}"
    candidates = sorted(parent.glob(pattern))

    matched = []
    for path in candidates:
        match = JOB_FILE_RE.match(path.name)
        if not match:
            continue
        if match.group("stem") != stem or match.group("suffix") != suffix:
            continue
        matched.append((int(match.group("job")), int(match.group("total")), path))

    if not matched:
        raise FileNotFoundError(f"No job CSVs found for base path {input_base}")

    totals = sorted({total for _, total, _ in matched})
    if len(totals) != 1:
        raise ValueError(f"Inconsistent total-job counts across files: {totals}")
    total_jobs = totals[0]

    job_numbers = sorted(job for job, _, _ in matched)
    expected = list(range(1, total_jobs + 1))
    if job_numbers != expected:
        raise ValueError(
            f"Missing or unexpected job files. Expected jobs {expected}, found {job_numbers}."
        )

    return [path for _, _, path in sorted(matched)]


def merge_job_files(job_files):
    merged_rows = []
    seen_keys = set()
    fieldnames = None

    for path in job_files:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"CSV {path} has no header.")
            if fieldnames is None:
                fieldnames = reader.fieldnames
            elif reader.fieldnames != fieldnames:
                raise ValueError(f"Header mismatch in {path}.")

            for row in reader:
                key = trial_key(row)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                merged_rows.append(row)

    merged_rows.sort(
        key=lambda row: (
            row.get("topology", ""),
            int(row.get("topology_seed", "0")),
            float(row.get("variance_ss4_vt", "0")),
            float(row.get("variance_b_vt", "0")),
        )
    )
    return fieldnames, merged_rows


def write_merged_csv(output_path, fieldnames, rows, overwrite):
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    input_base = args.input_base.resolve()
    output_path = args.output.resolve() if args.output is not None else input_base

    job_files = discover_job_files(input_base)
    fieldnames, merged_rows = merge_job_files(job_files)
    write_merged_csv(output_path, fieldnames, merged_rows, overwrite=args.overwrite)

    print(f"Merged {len(job_files)} files into {output_path} ({len(merged_rows)} unique rows).", flush=True)


if __name__ == "__main__":
    main()
