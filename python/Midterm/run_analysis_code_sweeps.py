import argparse
import csv
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


def _load_post_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("ssai_post_nsga_analysis", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import analysis script from '{script_path}'.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text.strip().lower()).strip("_")
    return slug or "job"


@dataclass
class CommandTemplate:
    section: str
    line_number: int
    raw_line: str
    tokens: List[str]


@dataclass
class SweepJob:
    section: str
    section_slug: str
    family_id: int
    trial_number: int
    template_line_number: int
    template_command: str
    job_slug: str
    out_dir: str
    log_path: str
    argv: List[str]


def _parse_analysis_codes(path: Path) -> List[CommandTemplate]:
    templates: List[CommandTemplate] = []
    current_section = "default"
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            current_section = line.lstrip("#").strip() or current_section
            continue
        tokens = shlex.split(line, posix=True)
        templates.append(
            CommandTemplate(
                section=current_section,
                line_number=line_number,
                raw_line=line,
                tokens=tokens,
            )
        )
    return templates


def _load_family_representatives(cluster_summary_path: Path, family_ids: Optional[Sequence[int]]) -> List[Tuple[int, int]]:
    summary = json.loads(cluster_summary_path.read_text(encoding="utf-8"))
    selected = None if family_ids is None else {int(fid) for fid in family_ids}
    reps: List[Tuple[int, int]] = []
    for family in summary.get("families", []):
        family_id = int(family["cluster_id"])
        if selected is not None and family_id not in selected:
            continue
        reps.append((family_id, int(family["representative_trial_number"])))
    if not reps:
        raise ValueError("No family representatives matched the requested family IDs.")
    return reps


def _template_job_slug(template: CommandTemplate, args_namespace) -> str:
    x_name = getattr(args_namespace, "x_sweep", "x").split(":", 1)[0].split("=", 1)[0]
    y_name = getattr(args_namespace, "y_sweep", "y").split(":", 1)[0].split("=", 1)[0]
    topology = getattr(args_namespace, "topology", "baseline")
    command_name = getattr(args_namespace, "command", "analysis")
    return _slugify(f"{command_name}_{x_name}_vs_{y_name}__{topology}")


def _validate_template_tokens(template: CommandTemplate) -> None:
    forbidden = {"--run-dir", "--family-analysis-dir", "--trial-numbers", "--family-ids", "--max-families", "--out-dir"}
    for token in template.tokens:
        if token in forbidden:
            raise ValueError(
                f"analysis_codes line {template.line_number} must not include '{token}'; "
                "the batch runner sets run/selection/output arguments itself."
            )


def _build_jobs(
    *,
    post,
    script_path: Path,
    run_dir: Path,
    family_analysis_dir: Path,
    templates: Sequence[CommandTemplate],
    reps: Sequence[Tuple[int, int]],
    batch_out_dir: Path,
    max_jobs: Optional[int],
) -> List[SweepJob]:
    parser = post._build_parser()
    cfg = post._search_config_from_run(run_dir)
    trials = post._load_pareto_trials(run_dir)
    trial_by_number: Dict[int, Mapping[str, object]] = {int(trial["number"]): trial for trial in trials}
    jobs: List[SweepJob] = []

    for family_id, trial_number in reps:
        if trial_number not in trial_by_number:
            raise ValueError(f"Representative trial {trial_number} for family {family_id} was not found in pareto_front.json.")
        trial = trial_by_number[trial_number]
        base_seed = int(trial.get("user_attrs", {}).get("trial_seed", cfg.base_seed + 1009 * (trial_number + 1)))
        for template in templates:
            _validate_template_tokens(template)
            section_slug = _slugify(template.section)
            base_out_dir = batch_out_dir / section_slug / f"family_{family_id:02d}_trial_{trial_number}"
            tmp_tokens = list(template.tokens)
            tmp_tokens.extend(
                [
                    "--run-dir",
                    str(run_dir),
                    "--family-analysis-dir",
                    str(family_analysis_dir),
                    "--trial-numbers",
                    str(trial_number),
                    "--out-dir",
                    str(base_out_dir / "placeholder"),
                ]
            )
            try:
                parsed = parser.parse_args(tmp_tokens)
            except SystemExit as exc:
                raise ValueError(f"analysis_codes line {template.line_number} failed argparse validation: {template.raw_line}") from exc
            if parsed.command != "sweep2d":
                raise ValueError(f"analysis_codes line {template.line_number} must use 'sweep2d'.")

            x_sweep = post._parse_sweep_spec(parsed.x_sweep)
            y_sweep = post._parse_sweep_spec(parsed.y_sweep)
            topology = str(parsed.topology)
            post._resolve_sweep_value(
                cfg,
                trial["params"],
                x_sweep,
                x_sweep.values[0],
                topology=topology,
                run_seed=base_seed,
            )
            post._resolve_sweep_value(
                cfg,
                trial["params"],
                y_sweep,
                y_sweep.values[0],
                topology=topology,
                run_seed=base_seed,
            )

            job_slug = _template_job_slug(template, parsed)
            out_dir = base_out_dir / job_slug
            argv = [
                sys.executable,
                str(script_path),
                *template.tokens,
                "--run-dir",
                str(run_dir),
                "--family-analysis-dir",
                str(family_analysis_dir),
                "--trial-numbers",
                str(trial_number),
                "--out-dir",
                str(out_dir),
            ]
            jobs.append(
                SweepJob(
                    section=template.section,
                    section_slug=section_slug,
                    family_id=family_id,
                    trial_number=trial_number,
                    template_line_number=template.line_number,
                    template_command=template.raw_line,
                    job_slug=job_slug,
                    out_dir=str(out_dir),
                    log_path=str(out_dir / "runner.log"),
                    argv=argv,
                )
            )
            if max_jobs is not None and len(jobs) >= max_jobs:
                return jobs
    return jobs


def _write_manifest(batch_out_dir: Path, jobs: Sequence[SweepJob]) -> None:
    batch_out_dir.mkdir(parents=True, exist_ok=True)
    manifest_json = batch_out_dir / "batch_jobs.json"
    manifest_csv = batch_out_dir / "batch_jobs.csv"
    manifest_json.write_text(json.dumps([asdict(job) for job in jobs], indent=2), encoding="utf-8")
    if jobs:
        with manifest_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(asdict(jobs[0]).keys()))
            writer.writeheader()
            writer.writerows(asdict(job) for job in jobs)


def _replace_out_dir_in_argv(argv: Sequence[str], new_out_dir: str) -> List[str]:
    parts = list(argv)
    for idx, token in enumerate(parts[:-1]):
        if token == "--out-dir":
            parts[idx + 1] = new_out_dir
            return parts
    raise ValueError("Job argv did not contain --out-dir.")


def _load_failed_jobs_from_batch(batch_dir: Path, rerun_suffix: str) -> List[SweepJob]:
    results_path = batch_dir / "batch_results.json"
    jobs_path = batch_dir / "batch_jobs.json"
    if not results_path.exists() or not jobs_path.exists():
        raise ValueError(f"Expected both batch_results.json and batch_jobs.json in '{batch_dir}'.")

    results = json.loads(results_path.read_text(encoding="utf-8"))
    failed = [row for row in results if int(row.get("returncode", 0)) != 0]
    if not failed:
        return []

    jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
    job_by_log_path = {str(job["log_path"]): job for job in jobs}
    rerun_jobs: List[SweepJob] = []
    for row in failed:
        source = job_by_log_path.get(str(row["log_path"]))
        if source is None:
            raise ValueError(f"Could not find job manifest entry for failed log '{row['log_path']}'.")
        out_dir = Path(source["out_dir"])
        rerun_out_dir = out_dir.parent / f"{out_dir.name}__{rerun_suffix}"
        rerun_jobs.append(
            SweepJob(
                section=str(source["section"]),
                section_slug=str(source["section_slug"]),
                family_id=int(source["family_id"]),
                trial_number=int(source["trial_number"]),
                template_line_number=int(source["template_line_number"]),
                template_command=str(source["template_command"]),
                job_slug=f"{source['job_slug']}__{rerun_suffix}",
                out_dir=str(rerun_out_dir),
                log_path=str(rerun_out_dir / "runner.log"),
                argv=_replace_out_dir_in_argv(source["argv"], str(rerun_out_dir)),
            )
        )
    return rerun_jobs


def _run_job(job: SweepJob) -> Dict[str, object]:
    out_dir = Path(job.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(job.argv, capture_output=True, text=True)
    log_lines = [
        f"command: {' '.join(shlex.quote(part) for part in job.argv)}",
        f"returncode: {completed.returncode}",
        "",
        "[stdout]",
        completed.stdout,
        "",
        "[stderr]",
        completed.stderr,
    ]
    Path(job.log_path).write_text("\n".join(log_lines), encoding="utf-8")
    return {
        "family_id": int(job.family_id),
        "trial_number": int(job.trial_number),
        "section": job.section,
        "job_slug": job.job_slug,
        "out_dir": job.out_dir,
        "log_path": job.log_path,
        "returncode": int(completed.returncode),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Expand sweep2d commands from analysis_codes.txt across family representatives, "
            "validate them up front, and run them in parallel subprocesses."
        )
    )
    parser.add_argument("--run-dir", type=str, required=False)
    parser.add_argument("--family-analysis-dir", type=str, required=False)
    parser.add_argument("--analysis-codes", type=str, required=False)
    parser.add_argument("--family-ids", type=int, nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2))))
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--batch-out-dir", type=str, default=None)
    parser.add_argument(
        "--rerun-failures-from",
        type=str,
        default=None,
        help="Existing batch output directory containing batch_jobs.json and batch_results.json.",
    )
    parser.add_argument(
        "--rerun-suffix",
        type=str,
        default="rerun",
        help="Suffix appended to each failed job folder when using --rerun-failures-from.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.rerun_failures_from:
        batch_out_dir = Path(args.rerun_failures_from).resolve()
        jobs = _load_failed_jobs_from_batch(batch_out_dir, args.rerun_suffix)
        if not jobs:
            print(f"No failed jobs found in {batch_out_dir}")
            return
        print(f"Loaded {len(jobs)} failed jobs from {batch_out_dir}")
    else:
        missing = [name for name in ("run_dir", "family_analysis_dir", "analysis_codes") if not getattr(args, name)]
        if missing:
            raise ValueError(f"Missing required arguments for fresh batch mode: {', '.join(missing)}")
        run_dir = Path(args.run_dir).resolve()
        family_analysis_dir = Path(args.family_analysis_dir).resolve()
        analysis_codes_path = Path(args.analysis_codes).resolve()
        script_path = Path(__file__).resolve().parent / "ssai_post_nsga_analysis.py"
        batch_out_dir = (
            Path(args.batch_out_dir).resolve()
            if args.batch_out_dir
            else (run_dir / "post_nsga_analysis" / "batch_sweep2d_run16")
        )

        post = _load_post_module(script_path)
        templates = _parse_analysis_codes(analysis_codes_path)
        if not templates:
            raise ValueError(f"No runnable commands were found in '{analysis_codes_path}'.")
        reps = _load_family_representatives(family_analysis_dir / "cluster_summary.json", args.family_ids)
        jobs = _build_jobs(
            post=post,
            script_path=script_path,
            run_dir=run_dir,
            family_analysis_dir=family_analysis_dir,
            templates=templates,
            reps=reps,
            batch_out_dir=batch_out_dir,
            max_jobs=args.max_jobs,
        )
        _write_manifest(batch_out_dir, jobs)

        print(f"Validated {len(jobs)} jobs across {len(reps)} families.")
        print(f"Manifest written to {batch_out_dir}")
    if args.validate_only:
        return

    results: List[Dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        future_to_job = {executor.submit(_run_job, job): job for job in jobs}
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            result = future.result()
            results.append(result)
            status = "OK" if int(result["returncode"]) == 0 else "FAIL"
            print(
                f"[{status}] family={job.family_id} trial={job.trial_number} "
                f"section='{job.section}' job='{job.job_slug}'"
            )

    results_path = batch_out_dir / "batch_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    failures = [result for result in results if int(result["returncode"]) != 0]
    if failures:
        raise SystemExit(f"{len(failures)} jobs failed. See batch_results.json and per-job runner.log files.")


if __name__ == "__main__":
    main()
