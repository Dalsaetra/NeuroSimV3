import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ssai_optuna import PARAM_SPECS, SSAIObjective, SearchConfig

try:
    import cma
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Failed to import cma. Activate the 'neuro' conda env and install 'cma'.") from exc


_WORKER_OBJECTIVE = None


class DummyTrial:
    def __init__(self, number: int, params: Dict[str, float]):
        self.number = int(number)
        self._params = dict(params)
        self.user_attrs: Dict[str, float] = {}
        self.intermediate_values: Dict[int, float] = {}

    def suggest_float(self, name: str, low: float, high: float, log: bool = False) -> float:
        if name not in self._params:
            raise KeyError(f"Missing parameter '{name}' for dummy trial.")
        value = float(self._params[name])
        if value < low or value > high:
            raise ValueError(f"Parameter '{name}'={value} out of bounds [{low}, {high}].")
        return value

    def report(self, value: float, step: int) -> None:
        self.intermediate_values[int(step)] = float(value)

    def should_prune(self) -> bool:
        return False

    def set_user_attr(self, key: str, value: float) -> None:
        self.user_attrs[key] = float(value)


def _decode_unit_vector(x: np.ndarray) -> Dict[str, float]:
    params: Dict[str, float] = {}
    for i, (name, low, high, use_log) in enumerate(PARAM_SPECS):
        z = float(np.clip(x[i], 0.0, 1.0))
        if use_log:
            lo = math.log(low)
            hi = math.log(high)
            params[name] = float(math.exp(lo + z * (hi - lo)))
        else:
            params[name] = float(low + z * (high - low))
    return params


def _init_worker(seed: int) -> None:
    global _WORKER_OBJECTIVE
    _WORKER_OBJECTIVE = SSAIObjective(SearchConfig(base_seed=int(seed)))


def _evaluate_candidate(task: Tuple[int, int, List[float], int]) -> Dict[str, object]:
    generation, eval_id, x, seed = task
    x_arr = np.asarray(x, dtype=float)
    params = _decode_unit_vector(x_arr)

    global _WORKER_OBJECTIVE
    if _WORKER_OBJECTIVE is None:
        _WORKER_OBJECTIVE = SSAIObjective(SearchConfig(base_seed=int(seed)))

    trial = DummyTrial(number=eval_id, params=params)
    score = float(_WORKER_OBJECTIVE(trial))
    return {
        "generation": int(generation),
        "eval_id": int(eval_id),
        "score": score,
        "loss": -score,
        "params": params,
        "metrics": dict(trial.user_attrs),
        "intermediate": dict(trial.intermediate_values),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CMA-ES search for SSAI using the objective in ssai_optuna.py")
    parser.add_argument("--max-evals", type=int, default=200)
    parser.add_argument("--popsize", type=int, default=16)
    parser.add_argument("--sigma0", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default="week4/cmaes_analysis")
    parser.add_argument("--save-every", type=int, default=1)
    return parser


def _write_csv(rows: List[Dict[str, object]], out_path: Path) -> None:
    if not rows:
        return

    fieldnames = ["generation", "eval_id", "score", "loss"]

    param_keys = sorted({k for r in rows for k in r["params"].keys()})
    metric_keys = sorted({k for r in rows for k in r["metrics"].keys()})
    fieldnames += [f"param.{k}" for k in param_keys]
    fieldnames += [f"metric.{k}" for k in metric_keys]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            row = {
                "generation": r["generation"],
                "eval_id": r["eval_id"],
                "score": r["score"],
                "loss": r["loss"],
            }
            row.update({f"param.{k}": v for k, v in r["params"].items()})
            row.update({f"metric.{k}": v for k, v in r["metrics"].items()})
            writer.writerow(row)


def main() -> None:
    args = _build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x0 = np.full(len(PARAM_SPECS), 0.5, dtype=float)
    options = {
        "bounds": [0.0, 1.0],
        "seed": int(args.seed),
        "popsize": int(args.popsize),
        "verb_disp": 1,
    }
    es = cma.CMAEvolutionStrategy(x0.tolist(), float(args.sigma0), options)

    all_rows: List[Dict[str, object]] = []
    best_row = None
    eval_id = 0
    generation = 0

    objective = SSAIObjective(SearchConfig(base_seed=int(args.seed))) if args.n_workers <= 1 else None

    pool = None
    if args.n_workers > 1:
        pool = ProcessPoolExecutor(max_workers=int(args.n_workers), initializer=_init_worker, initargs=(int(args.seed),))

    try:
        while eval_id < int(args.max_evals) and not es.stop():
            generation += 1
            candidates = es.ask()
            tasks = []
            for x in candidates:
                if eval_id >= int(args.max_evals):
                    break
                tasks.append((generation, eval_id, list(x), int(args.seed)))
                eval_id += 1

            if args.n_workers > 1:
                results = list(pool.map(_evaluate_candidate, tasks))
            else:
                results = []
                for g, eid, x, seed in tasks:
                    params = _decode_unit_vector(np.asarray(x, dtype=float))
                    trial = DummyTrial(number=eid, params=params)
                    score = float(objective(trial))
                    results.append(
                        {
                            "generation": g,
                            "eval_id": eid,
                            "score": score,
                            "loss": -score,
                            "params": params,
                            "metrics": dict(trial.user_attrs),
                            "intermediate": dict(trial.intermediate_values),
                        }
                    )

            losses = [float(r["loss"]) for r in results]
            used_candidates = candidates[: len(losses)]
            es.tell(used_candidates, losses)

            all_rows.extend(results)
            local_best = max(results, key=lambda r: float(r["score"]))
            if best_row is None or float(local_best["score"]) > float(best_row["score"]):
                best_row = local_best

            if generation % int(args.save_every) == 0:
                _write_csv(all_rows, out_dir / "cmaes_trials.csv")

            print(
                f"gen={generation} evals={eval_id} best_gen={local_best['score']:.4f} best_global={best_row['score']:.4f}"
            )

    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    _write_csv(all_rows, out_dir / "cmaes_trials.csv")

    summary = {
        "max_evals": int(args.max_evals),
        "evaluated": int(len(all_rows)),
        "best_score": float(best_row["score"]) if best_row is not None else None,
        "best_params": best_row["params"] if best_row is not None else None,
        "best_metrics": best_row["metrics"] if best_row is not None else None,
        "seed": int(args.seed),
        "popsize": int(args.popsize),
        "sigma0": float(args.sigma0),
        "n_workers": int(args.n_workers),
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    with (out_dir / "best_result.json").open("w", encoding="utf-8") as f:
        json.dump(best_row, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))
    if best_row is not None:
        print("\nBest parameters")
        for name, _, _, _ in PARAM_SPECS:
            print(f"{name}: {best_row['params'].get(name)}")


if __name__ == "__main__":
    main()
