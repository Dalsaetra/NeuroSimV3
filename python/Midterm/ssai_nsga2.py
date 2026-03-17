import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import optuna
from optuna.samplers import NSGAIISampler
from optuna.trial import TrialState

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from midterm.ssai_nsga2_objective import OBJECTIVE_NAMES, PARAM_SPECS, SSAIMultiObjective, SearchConfig, config_to_dict
except ModuleNotFoundError:
    from ssai_nsga2_objective import OBJECTIVE_NAMES, PARAM_SPECS, SSAIMultiObjective, SearchConfig, config_to_dict


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "NSGA-II search for self-sustained asynchronous activity using the week4-style "
            "simulation regime plus a transmitter-balance objective."
        )
    )
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds.")
    parser.add_argument("--study-name", type=str, default="midterm_ssai_nsga2")
    parser.add_argument("--storage", type=str, default="sqlite:///midterm/ssai_nsga2.db")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--mutation-prob", type=float, default=None)
    parser.add_argument("--crossover-prob", type=float, default=0.9)
    parser.add_argument("--swapping-prob", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="midterm/nsga2_runs")
    parser.add_argument("--save-every", type=int, default=1, help="Checkpoint trials.csv every N completed trials.")
    parser.add_argument("--show-pareto", action="store_true")

    parser.add_argument("--n-neurons", type=int, default=1000)
    parser.add_argument("--n-excit", type=int, default=800)
    parser.add_argument("--n-out", type=int, default=100)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--dt-ms", type=float, default=0.1)
    parser.add_argument("--ext-on-ms", type=float, default=1000.0)
    parser.add_argument("--total-ms", type=float, default=3000.0)
    parser.add_argument("--delay-mean-exc-ms", type=float, default=1.5)
    parser.add_argument("--delay-std-exc-ms", type=float, default=0.3)
    parser.add_argument("--delay-mean-inh-ms", type=float, default=1.5)
    parser.add_argument("--delay-std-inh-ms", type=float, default=0.3)
    parser.add_argument("--recurrent-exc-mu", type=float, default=0.0)
    parser.add_argument("--recurrent-exc-wmax", type=float, default=100.0)
    return parser


def _serialize_trial(trial: optuna.trial.FrozenTrial) -> Dict[str, object]:
    return {
        "number": int(trial.number),
        "values": [float(v) for v in trial.values] if trial.values is not None else None,
        "params": dict(trial.params),
        "user_attrs": dict(trial.user_attrs),
        "state": trial.state.name,
    }


def _write_trials_csv(trials: Iterable[optuna.trial.FrozenTrial], out_path: Path) -> None:
    trial_list = [t for t in trials if t.values is not None]
    if not trial_list:
        return

    param_keys = sorted({k for t in trial_list for k in t.params.keys()})
    attr_keys = sorted({k for t in trial_list for k in t.user_attrs.keys()})

    fieldnames = ["number", "state"]
    fieldnames += [f"objective.{name}" for name in OBJECTIVE_NAMES]
    fieldnames += [f"param.{k}" for k in param_keys]
    fieldnames += [f"metric.{k}" for k in attr_keys]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for trial in trial_list:
            row = {
                "number": int(trial.number),
                "state": trial.state.name,
            }
            row.update(
                {
                    f"objective.{name}": float(trial.values[i])
                    for i, name in enumerate(OBJECTIVE_NAMES)
                    if trial.values is not None and i < len(trial.values)
                }
            )
            row.update({f"param.{k}": v for k, v in trial.params.items()})
            row.update({f"metric.{k}": v for k, v in trial.user_attrs.items()})
            writer.writerow(row)


def _pick_compromise_trial(pareto_trials: List[optuna.trial.FrozenTrial]) -> Optional[optuna.trial.FrozenTrial]:
    feasible_trials = [
        trial
        for trial in pareto_trials
        if trial.values is not None and all(float(value) >= 0.0 for value in trial.values)
    ]

    if not feasible_trials:
        return None
    if len(feasible_trials) == 1:
        return feasible_trials[0]

    value_columns = list(zip(*[[float(v) for v in t.values] for t in feasible_trials]))
    mins = [min(col) for col in value_columns]
    maxs = [max(col) for col in value_columns]

    best_trial = None
    best_score = None
    for trial in feasible_trials:
        score = 0.0
        for i, value in enumerate(trial.values):
            lo = mins[i]
            hi = maxs[i]
            score += 1.0 if hi == lo else (float(value) - lo) / (hi - lo)
        if best_score is None or score > best_score:
            best_score = score
            best_trial = trial
    return best_trial


def _print_pareto(pareto_trials: List[optuna.trial.FrozenTrial]) -> None:
    print("\nPareto front")
    for trial in pareto_trials:
        parts = [f"trial={trial.number}"]
        parts.extend(f"{name}={float(trial.values[i]):.4f}" for i, name in enumerate(OBJECTIVE_NAMES))
        print(" ".join(parts))


def _checkpoint_trials_csv(study: optuna.Study, out_dir: Path, save_every: int):
    def _callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state != TrialState.COMPLETE:
            return
        complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE and t.values is not None]
        if len(complete_trials) % save_every != 0:
            return
        _write_trials_csv(complete_trials, out_dir / "trials.csv")

    return _callback


def _prepare_storage_url(storage: str) -> str:
    if not storage.startswith("sqlite:///"):
        return storage

    db_path_str = storage[len("sqlite:///") :]
    if not db_path_str:
        raise ValueError("SQLite storage URL must include a database path.")

    if os.name == "nt" and db_path_str.startswith("/") and len(db_path_str) > 2 and db_path_str[2] == ":":
        db_path = Path(db_path_str.lstrip("/"))
    else:
        db_path = Path(db_path_str)

    if not db_path.is_absolute():
        db_path = REPO_ROOT / db_path

    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path.as_posix()}"


def main() -> None:
    args = _build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    storage_url = _prepare_storage_url(args.storage)

    if args.n_neurons <= 1:
        raise ValueError("--n-neurons must be > 1.")
    if not (0 < args.n_excit < args.n_neurons):
        raise ValueError("--n-excit must be between 1 and n-neurons - 1.")
    if not (0 < args.n_out <= args.n_neurons):
        raise ValueError("--n-out must be between 1 and n-neurons.")
    if args.n_repeats <= 0:
        raise ValueError("--n-repeats must be >= 1.")
    if args.save_every <= 0:
        raise ValueError("--save-every must be >= 1.")
    if args.dt_ms <= 0.0:
        raise ValueError("--dt-ms must be > 0.")
    if args.ext_on_ms < 0.0 or args.total_ms <= 0.0 or args.ext_on_ms > args.total_ms:
        raise ValueError("--ext-on-ms must satisfy 0 <= ext-on-ms <= total-ms.")

    cfg = SearchConfig(
        n_neurons=int(args.n_neurons),
        n_excit=int(args.n_excit),
        n_out=int(args.n_out),
        n_repeats=int(args.n_repeats),
        dt_ms=float(args.dt_ms),
        ext_on_ms=float(args.ext_on_ms),
        total_ms=float(args.total_ms),
        delay_mean_exc_ms=float(args.delay_mean_exc_ms),
        delay_std_exc_ms=float(args.delay_std_exc_ms),
        delay_mean_inh_ms=float(args.delay_mean_inh_ms),
        delay_std_inh_ms=float(args.delay_std_inh_ms),
        recurrent_exc_mu=float(args.recurrent_exc_mu),
        recurrent_exc_wmax=float(args.recurrent_exc_wmax),
        base_seed=int(args.seed),
    )
    objective = SSAIMultiObjective(cfg)

    sampler = NSGAIISampler(
        population_size=int(args.population_size),
        mutation_prob=args.mutation_prob,
        crossover_prob=float(args.crossover_prob),
        swapping_prob=float(args.swapping_prob),
        seed=int(args.seed),
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        load_if_exists=True,
        directions=["maximize"] * len(OBJECTIVE_NAMES),
        sampler=sampler,
    )

    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        timeout=args.timeout,
        gc_after_trial=True,
        callbacks=[_checkpoint_trials_csv(study, out_dir, int(args.save_every))],
    )

    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE and t.values is not None]
    pareto_trials = sorted(
        study.best_trials,
        key=lambda t: tuple(float(v) for v in t.values),
        reverse=True,
    )
    compromise_trial = _pick_compromise_trial(pareto_trials)

    _write_trials_csv(complete_trials, out_dir / "trials.csv")
    _write_trials_csv(pareto_trials, out_dir / "pareto_front.csv")

    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "search_config": config_to_dict(cfg),
                "param_specs": PARAM_SPECS,
                "study_name": args.study_name,
                "storage": storage_url,
                "seed": int(args.seed),
                "population_size": int(args.population_size),
                "save_every": int(args.save_every),
                "n_trials_requested": int(args.n_trials),
                "n_jobs": int(args.n_jobs),
                "timeout": args.timeout,
            },
            f,
            indent=2,
            sort_keys=True,
        )

    with (out_dir / "pareto_front.json").open("w", encoding="utf-8") as f:
        json.dump([_serialize_trial(t) for t in pareto_trials], f, indent=2, sort_keys=True)

    if compromise_trial is not None:
        with (out_dir / "compromise_result.json").open("w", encoding="utf-8") as f:
            json.dump(_serialize_trial(compromise_trial), f, indent=2, sort_keys=True)

    summary = {
        "study_name": study.study_name,
        "n_trials_total": int(len(study.trials)),
        "n_trials_complete": int(len(complete_trials)),
        "n_pareto": int(len(pareto_trials)),
        "best_objective_trials": {
            name: _serialize_trial(max(complete_trials, key=lambda t, idx=i: float(t.values[idx])))
            for i, name in enumerate(OBJECTIVE_NAMES)
        } if complete_trials else {},
        "compromise_trial": _serialize_trial(compromise_trial) if compromise_trial is not None else None,
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.show_pareto:
        _print_pareto(pareto_trials)


if __name__ == "__main__":
    main()
