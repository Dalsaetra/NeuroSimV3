import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd


def _flatten_attrs(prefix: str, attrs: dict) -> dict:
    out = {}
    for k, v in attrs.items():
        out[f"{prefix}.{k}"] = v
    return out


def _top_trials_dataframe(study: optuna.Study, top_n: int) -> pd.DataFrame:
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed = sorted(completed, key=lambda t: t.value if t.value is not None else float("-inf"), reverse=True)
    top = completed[:top_n]

    rows = []
    for t in top:
        row = {
            "number": t.number,
            "value": t.value,
            "datetime_start": t.datetime_start,
            "datetime_complete": t.datetime_complete,
            "duration_s": (t.duration.total_seconds() if t.duration is not None else None),
        }
        row.update(_flatten_attrs("param", t.params))
        row.update(_flatten_attrs("metric", t.user_attrs))
        rows.append(row)

    return pd.DataFrame(rows)


def _save_plots(
    study: optuna.Study,
    out_dir: Path,
    history_q_low: float = 0.05,
    history_q_high: float = 0.95,
    history_symlog: bool = True,
) -> None:
    try:
        from optuna.visualization.matplotlib import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate,
            plot_slice,
        )
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Skipping plots: matplotlib-based Optuna plots unavailable ({e}).")
        return

    # Optimization history
    ax = plot_optimization_history(study)
    values = [t.value for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    if values:
        lo = float(np.quantile(values, history_q_low))
        hi = float(np.quantile(values, history_q_high))
        if hi > lo:
            pad = 0.1 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)
        if history_symlog:
            ax.set_yscale("symlog", linthresh=max(1e-3, 0.1 * max(abs(lo), abs(hi), 1.0)))
    ax.figure.savefig(out_dir / "optimization_history.png", dpi=160, bbox_inches="tight")
    plt.close(ax.figure)

    # Param importances
    ax = plot_param_importances(study)
    ax.figure.savefig(out_dir / "param_importances.png", dpi=160, bbox_inches="tight")
    plt.close(ax.figure)

    # Parallel coordinates
    ax = plot_parallel_coordinate(study)
    ax.figure.savefig(out_dir / "parallel_coordinate.png", dpi=160, bbox_inches="tight")
    plt.close(ax.figure)

    # Slice
    ax = plot_slice(study)
    # Optuna may return a single Axes or an ndarray of Axes for slice plots.
    if isinstance(ax, np.ndarray):
        fig = ax.flat[0].figure
    else:
        fig = ax.figure
    fig.savefig(out_dir / "slice.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze Optuna study results from ssai_optuna.db")
    p.add_argument("--storage", type=str, default="sqlite:///ssai_optuna.db")
    p.add_argument("--study-name", type=str, default="ssai_optuna")
    p.add_argument("--out-dir", type=str, default="optuna_analysis")
    p.add_argument("--top-n", type=int, default=25)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--history-q-low", type=float, default=0.05)
    p.add_argument("--history-q-high", type=float, default=0.95)
    p.add_argument("--history-linear", action="store_true", help="Use linear y-axis for optimization history.")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    study = optuna.load_study(study_name=args.study_name, storage=args.storage)

    trials_df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "datetime_start", "datetime_complete", "duration"))
    trials_csv = out_dir / "trials.csv"
    trials_df.to_csv(trials_csv, index=False)

    top_df = _top_trials_dataframe(study, top_n=args.top_n)
    top_csv = out_dir / "top_trials.csv"
    top_df.to_csv(top_csv, index=False)

    summary = {
        "study_name": study.study_name,
        "storage": args.storage,
        "n_trials": len(study.trials),
        "n_complete": int(sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)),
        "n_pruned": int(sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)),
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "best_user_attrs": study.best_trial.user_attrs,
        "trials_csv": str(trials_csv),
        "top_trials_csv": str(top_csv),
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    if not args.no_plots:
        _save_plots(
            study,
            out_dir,
            history_q_low=args.history_q_low,
            history_q_high=args.history_q_high,
            history_symlog=not args.history_linear,
        )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
