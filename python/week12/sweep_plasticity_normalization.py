from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.connectome import Connectome
from src.external_inputs import PoissonInput
from src.network_generators import (
    generate_random_fixed_indegree_ei_network,
    generate_spatial_ei_network,
)
from src.neuron_population import NeuronPopulation
from src.neuron_templates import neuron_type_IZ
from src.normalization import PeriodicIncomingWeightNormalizer
from src.overhead import Simulation, SimulationStats


EXCITATORY_TYPE = "ss4"
INHIBITORY_TYPE = "b"
TYPE_FRACTIONS = {"ss4": 0.8, "b": 0.2}
INHIBITORY_TYPES = ("b",)
WEIGHT_DIST_BY_NTYPE = {"ss4": "lognormal", "b": "normal"}
OUTDEGREE_CONFIG_BY_TYPE = {
    "ss4": {"dist": "lognormal", "params": (2.65, 0.8)},
    "b": {"dist": "neg-bin", "params": (50, 40)},
}
P0_BY_PAIR = {"EE": 0.5, "EI": 0.5, "IE": 0.5, "II": 0.5}
LAMBDA_BY_PRECLASS = {"E": 0.2, "I": 0.2}
RATE_NORM_INTERVALS_MS = (100.0, 500.0, 1000.0)
RATE_NORM_ETAS = (0.0005, 0.001, 0.01)
SOFT_IN_NORM_EXPONENTS = (0.25, 0.5, 0.75, 1.0)
DEFAULT_SEEDS = (1234, 1235, 1236)
V_EXT = 50.0
POISSON_AMPLITUDE = 0.5
METRIC_BIN_MS_PARTICIPATION = 300.0
HIST_BINS = 64

TOPOLOGY_CONFIGS = {
    "random_fixed": {
        "label": "simple_fixed_indegree",
        "target_rate_hz": {"ss4": 3.5, "b": 19.0},
        "normalize_target_in_E": {"ss4": 50.0, "b": 20.0},
        "normalize_target_in_I": {"ss4": 250.0, "b": 25.0},
    },
    "spatial": {
        "label": "realistic_spatial",
        "target_rate_hz": {"ss4": 8.0, "b": 18.0},
        "normalize_target_in_E": {"ss4": 50.0, "b": 10.0},
        "normalize_target_in_I": {"ss4": 500.0, "b": 50.0},
    },
}


def _seed_sequence_for_run(base_seed: int, topology_name: str) -> np.random.SeedSequence:
    topology_code = 0 if topology_name == "random_fixed" else 1
    return np.random.SeedSequence([int(base_seed), topology_code, 20260401])


def _build_network(topology_name: str, n_neurons: int, network_seed: int):
    cfg = TOPOLOGY_CONFIGS[topology_name]
    if topology_name == "random_fixed":
        return generate_random_fixed_indegree_ei_network(
            n_neurons=n_neurons,
            indegree=100,
            type_fractions=TYPE_FRACTIONS,
            inhibitory_types=INHIBITORY_TYPES,
            delay_mean_E=10.0,
            delay_std_E=3.0,
            delay_mean_I=1.5,
            delay_std_I=0.45,
            mu_E=0.0,
            sigma_E=1.6,
            mu_I=30.0,
            sigma_I=6.0,
            weight_dist_by_ntype=WEIGHT_DIST_BY_NTYPE,
            normalize_mode="in",
            normalize_target_in_E=cfg["normalize_target_in_E"],
            normalize_target_in_I=cfg["normalize_target_in_I"],
            seed=network_seed,
        )

    if topology_name == "spatial":
        return generate_spatial_ei_network(
            n_neurons=n_neurons,
            type_fractions=TYPE_FRACTIONS,
            inhibitory_types=INHIBITORY_TYPES,
            mu_E=0.0,
            sigma_E=1.6,
            mu_I=30.0,
            sigma_I=6.0,
            p0_by_pair=P0_BY_PAIR,
            lambda_by_preclass=LAMBDA_BY_PRECLASS,
            distance_scale=20.0,
            weight_dist_by_ntype=WEIGHT_DIST_BY_NTYPE,
            outdegree_config_by_type=OUTDEGREE_CONFIG_BY_TYPE,
            normalize_mode="in",
            normalize_target_in_E=cfg["normalize_target_in_E"],
            normalize_target_in_I=cfg["normalize_target_in_I"],
            seed=network_seed,
        )

    raise ValueError(f"Unknown topology '{topology_name}'.")


def _build_connectome_from_graph(G, dt_ms: float):
    n_neurons = G.number_of_nodes()
    neuron_types = [EXCITATORY_TYPE, INHIBITORY_TYPE]
    inhibitory = [False, True]
    threshold_decay = np.exp(-dt_ms / 5.0)

    pop = NeuronPopulation(n_neurons, neuron_types, inhibitory, threshold_decay)
    max_synapses = max(dict(G.out_degree()).values())
    connectome = Connectome(max_synapses, pop)
    connectome.nx_to_connectome(G)
    return connectome, pop


def _build_initial_state(n_neurons: int, init_seed: int):
    rng = np.random.default_rng(init_seed)
    Vs = rng.uniform(-100.0, -70.0, size=n_neurons)
    us = rng.uniform(0.0, 400.0, size=n_neurons)
    spikes = np.zeros(n_neurons, dtype=bool)
    Ts = np.zeros_like(spikes)
    return (Vs, us, spikes.copy(), Ts.copy())


def _build_input_rate_vector(pop: NeuronPopulation, input_mask_seed: int) -> np.ndarray:
    rng = np.random.default_rng(input_mask_seed)
    excit_idx = np.flatnonzero(~pop.inhibitory_mask.astype(bool))
    n_stim = excit_idx.size // 2
    selected = rng.choice(excit_idx, size=n_stim, replace=False)
    rates = np.zeros(pop.n_neurons, dtype=float)
    rates[selected] = V_EXT
    return rates


def _build_clopath_thresholds(pop: NeuronPopulation):
    resting = np.zeros(pop.n_neurons, dtype=float)
    threshold = np.zeros(pop.n_neurons, dtype=float)
    for i in range(pop.n_neurons):
        ntype = pop.type_from_neuron_index(i)
        resting[i] = neuron_type_IZ[ntype][5]
        threshold[i] = neuron_type_IZ[ntype][6]
    return resting, threshold


def _build_simulation(
    connectome: Connectome,
    pop: NeuronPopulation,
    dt_ms: float,
    state0,
    rate_normalization: dict | None,
):
    nmda_weight = np.ones(pop.n_neurons, dtype=float)
    nmda_weight[pop.inhibitory_mask.astype(bool)] = 0.959685703507305 * 0.5

    resting_potentials, threshold_potentials = _build_clopath_thresholds(pop)
    excit_mask = ~pop.inhibitory_mask.astype(bool)
    initial_weights = connectome.W[~connectome.NC]
    initial_rows = np.where(~connectome.NC)[0]
    excit_initial_weights = initial_weights[~pop.inhibitory_mask[initial_rows]]
    max_exc_weight = float(np.max(excit_initial_weights)) if excit_initial_weights.size > 0 else None

    return Simulation(
        connectome,
        dt_ms,
        stepper_type="euler_det",
        state0=state0,
        synapse_kwargs={"LT_scale": 1.0, "NMDA_weight": nmda_weight},
        synapse_type="standard",
        enable_state_logger=False,
        enable_debug_logger=False,
        rate_normalization=rate_normalization,
        enable_plasticity=True,
        plasticity="clopath",
        plasticity_reward_type="online",
        plasticity_kwargs={
            "plastic_source_mask": excit_mask,
            "weight_multiplicity": None,
            "max_weight": max_exc_weight,
            "weight_update_scale": 1.0,
            "theta_minus": resting_potentials,
            "theta_plus": threshold_potentials,
        },
    )


def _capture_sim_state(stats: SimulationStats, sim: Simulation):
    stats.Vs.append(sim.neuron_states.V.copy())
    stats.spikes.append(sim.neuron_states.spike.copy())
    stats.ts.append(float(sim.t_now))


def _run_steps(
    sim: Simulation,
    steps: int,
    dt_ms: float,
    *,
    poisson: PoissonInput | None = None,
    incoming_normalizer: PeriodicIncomingWeightNormalizer | None = None,
    stats: SimulationStats | None = None,
):
    for _ in range(steps):
        spike_ext = None if poisson is None else poisson(dt_ms)
        sim.step(spike_ext=spike_ext, reward=1.0)
        if incoming_normalizer is not None:
            incoming_normalizer.update()
        if stats is not None:
            _capture_sim_state(stats, sim)


def _disable_learning(sim: Simulation):
    sim.plasticity = None
    sim.plasticity_step = None
    sim.rate_normalizer = None


def _extract_valid_weights(connectome: Connectome):
    valid = ~connectome.NC
    rows, cols = np.where(valid)
    weights = connectome.W[rows, cols].astype(float, copy=True)
    src_inhibitory = connectome.neuron_population.inhibitory_mask[rows].astype(bool, copy=True)
    return weights, src_inhibitory


def _weight_summary(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            "count": 0.0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "p01": 0.0,
            "p05": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    return {
        "count": float(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def _build_weight_histogram(before: np.ndarray, after: np.ndarray, hist_bins: int) -> dict[str, list[float] | list[int]]:
    combined = np.concatenate((before, after)) if before.size or after.size else np.zeros(1, dtype=float)
    max_weight = float(np.max(combined)) if combined.size > 0 else 0.0
    if max_weight <= 0.0:
        linear_edges = np.linspace(0.0, 1.0, hist_bins + 1)
        before_counts = np.zeros(hist_bins, dtype=int)
        after_counts = np.zeros(hist_bins, dtype=int)
    else:
        linear_edges = np.linspace(0.0, max_weight, hist_bins + 1)
        before_counts, _ = np.histogram(before, bins=linear_edges)
        after_counts, _ = np.histogram(after, bins=linear_edges)

    positive = combined[combined > 0.0]
    if positive.size == 0:
        log_edges = np.logspace(-6, 0, hist_bins + 1)
        before_log_counts = np.zeros(hist_bins, dtype=int)
        after_log_counts = np.zeros(hist_bins, dtype=int)
    else:
        log_min = float(np.min(positive))
        log_max = float(np.max(positive))
        if log_min == log_max:
            log_max = log_min * 1.0001
        log_edges = np.logspace(np.log10(log_min), np.log10(log_max), hist_bins + 1)
        before_log_counts, _ = np.histogram(before[before > 0.0], bins=log_edges)
        after_log_counts, _ = np.histogram(after[after > 0.0], bins=log_edges)

    return {
        "linear_bin_edges": linear_edges.tolist(),
        "before_linear_counts": before_counts.astype(int).tolist(),
        "after_linear_counts": after_counts.astype(int).tolist(),
        "log_bin_edges": log_edges.tolist(),
        "before_log_counts": before_log_counts.astype(int).tolist(),
        "after_log_counts": after_log_counts.astype(int).tolist(),
    }


def _weight_details(before_weights: np.ndarray, after_weights: np.ndarray, src_inhibitory: np.ndarray, hist_bins: int):
    delta = after_weights - before_weights
    exc_mask = ~src_inhibitory
    inh_mask = src_inhibitory

    return {
        "all": {
            "before_summary": _weight_summary(before_weights),
            "after_summary": _weight_summary(after_weights),
            "histogram": _build_weight_histogram(before_weights, after_weights, hist_bins),
        },
        "exc_source": {
            "before_summary": _weight_summary(before_weights[exc_mask]),
            "after_summary": _weight_summary(after_weights[exc_mask]),
            "histogram": _build_weight_histogram(before_weights[exc_mask], after_weights[exc_mask], hist_bins),
        },
        "inh_source": {
            "before_summary": _weight_summary(before_weights[inh_mask]),
            "after_summary": _weight_summary(after_weights[inh_mask]),
            "histogram": _build_weight_histogram(before_weights[inh_mask], after_weights[inh_mask], hist_bins),
        },
        "delta_summary": {
            "min": float(np.min(delta)) if delta.size > 0 else 0.0,
            "max": float(np.max(delta)) if delta.size > 0 else 0.0,
            "mean": float(np.mean(delta)) if delta.size > 0 else 0.0,
            "mean_abs": float(np.mean(np.abs(delta))) if delta.size > 0 else 0.0,
            "std": float(np.std(delta)) if delta.size > 0 else 0.0,
        },
    }


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")


def _is_scalar_like(value) -> bool:
    return np.isscalar(value) or isinstance(value, np.generic)


def _flatten_scalar_metrics(prefix: str, metrics: dict) -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        if _is_scalar_like(value):
            flat[f"{prefix}{key}"] = float(value)
    return flat


def _flatten_weight_summary(prefix: str, summary: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}{key}": float(value) for key, value in summary.items()}


def _trial_conditions():
    conditions = [
        {
            "condition_name": "clopath_only",
            "normalization_kind": "none",
            "apply_every_ms": None,
            "eta_E": None,
            "eta_I": None,
            "normalization_exponent": None,
        }
    ]

    for interval_ms in RATE_NORM_INTERVALS_MS:
        for eta in RATE_NORM_ETAS:
            conditions.append(
                {
                    "condition_name": f"clopath_rate_norm_{int(interval_ms)}ms_eta_{eta}",
                    "normalization_kind": "rate",
                    "apply_every_ms": float(interval_ms),
                    "eta_E": float(eta),
                    "eta_I": float(eta),
                    "normalization_exponent": None,
                }
            )

    for interval_ms in RATE_NORM_INTERVALS_MS:
        for exponent in SOFT_IN_NORM_EXPONENTS:
            conditions.append(
                {
                    "condition_name": f"clopath_soft_in_norm_{int(interval_ms)}ms_exp_{exponent}",
                    "normalization_kind": "soft_in",
                    "apply_every_ms": float(interval_ms),
                    "eta_E": None,
                    "eta_I": None,
                    "normalization_exponent": float(exponent),
                }
            )

    return conditions


def _build_trials(seeds: tuple[int, ...]):
    trials = []
    for topology_name in TOPOLOGY_CONFIGS:
        for seed in seeds:
            for condition in _trial_conditions():
                trial = {
                    "topology": topology_name,
                    "seed": int(seed),
                }
                trial.update(condition)
                trials.append(trial)
    return trials


def _run_single_trial(trial: dict, args_dict: dict) -> dict[str, float | str]:
    t0 = time.perf_counter()
    topology_name = str(trial["topology"])
    topology_cfg = TOPOLOGY_CONFIGS[topology_name]
    seed = int(trial["seed"])
    dt_ms = float(args_dict["dt"])
    sim_ms = float(args_dict["sim_ms"])
    recovery_ms = float(args_dict["recovery_ms"])
    output_dir = Path(args_dict["output_dir"])
    details_dir = output_dir / "details"
    hist_bins = int(args_dict["hist_bins"])

    run_id = f"{topology_name}__{trial['condition_name']}__seed_{seed}"
    row: dict[str, float | str] = {
        "run_id": run_id,
        "topology": topology_name,
        "topology_label": topology_cfg["label"],
        "seed": seed,
        "condition_name": str(trial["condition_name"]),
        "normalization_kind": str(trial["normalization_kind"]),
        "apply_every_ms": "" if trial["apply_every_ms"] is None else float(trial["apply_every_ms"]),
        "eta_E": "" if trial["eta_E"] is None else float(trial["eta_E"]),
        "eta_I": "" if trial["eta_I"] is None else float(trial["eta_I"]),
        "normalization_exponent": "" if trial["normalization_exponent"] is None else float(trial["normalization_exponent"]),
        "details_json": str((details_dir / f"{run_id}.json").resolve()),
    }

    try:
        seed_seq = _seed_sequence_for_run(seed, topology_name)
        network_seed, init_seed, input_mask_seed, poisson_seed = [
            int(s.generate_state(1, dtype=np.uint32)[0]) for s in seed_seq.spawn(4)
        ]

        G = _build_network(topology_name, int(args_dict["n_neurons"]), network_seed)
        connectome, pop = _build_connectome_from_graph(G, dt_ms)
        state0 = _build_initial_state(pop.n_neurons, init_seed)

        rate_normalization = None
        if trial["normalization_kind"] == "rate":
            initial_weights, _ = _extract_valid_weights(connectome)
            max_weight = float(np.max(initial_weights)) if initial_weights.size > 0 else None
            rate_normalization = {
                "target_rate_hz": topology_cfg["target_rate_hz"],
                "eta_E": float(trial["eta_E"]),
                "eta_I": float(trial["eta_I"]),
                "rate_method": "window",
                "rate_window_ms": float(trial["apply_every_ms"]),
                "apply_every_ms": float(trial["apply_every_ms"]),
                "min_weight": 0.0,
                "max_weight": max_weight,
            }

        sim = _build_simulation(connectome, pop, dt_ms, state0, rate_normalization)

        incoming_normalizer = None
        if trial["normalization_kind"] == "soft_in":
            incoming_normalizer = PeriodicIncomingWeightNormalizer(
                connectome,
                dt_ms=dt_ms,
                target_in_E=topology_cfg["normalize_target_in_E"],
                target_in_I=topology_cfg["normalize_target_in_I"],
                apply_every_ms=float(trial["apply_every_ms"]),
                normalization_exponent=float(trial["normalization_exponent"]),
            )

        before_weights, src_inhibitory = _extract_valid_weights(connectome)

        input_rates = _build_input_rate_vector(pop, input_mask_seed)
        poisson = PoissonInput(
            pop.n_neurons,
            rate=input_rates,
            amplitude=POISSON_AMPLITUDE,
            rng=np.random.default_rng(poisson_seed),
        )
        active_steps = int(round(sim_ms / dt_ms))
        _run_steps(sim, active_steps, dt_ms, poisson=poisson, incoming_normalizer=incoming_normalizer, stats=None)

        after_weights, src_inhibitory_after = _extract_valid_weights(connectome)
        if not np.array_equal(src_inhibitory, src_inhibitory_after):
            raise RuntimeError("Source inhibitory masks changed unexpectedly across the simulation.")

        _disable_learning(sim)
        recovery_stats = SimulationStats()
        recovery_stats.inhibitory_mask = pop.inhibitory_mask.copy()
        _capture_sim_state(recovery_stats, sim)
        recovery_steps = int(round(recovery_ms / dt_ms))
        _run_steps(sim, recovery_steps, dt_ms, poisson=None, incoming_normalizer=None, stats=recovery_stats)
        recovery_metrics = recovery_stats.compute_metrics(dt_ms, bin_ms_participation=METRIC_BIN_MS_PARTICIPATION)

        weight_details = _weight_details(before_weights, after_weights, src_inhibitory, hist_bins)
        details_payload = {
            "run_id": run_id,
            "config": {
                "seed": seed,
                "network_seed": network_seed,
                "init_seed": init_seed,
                "input_mask_seed": input_mask_seed,
                "poisson_seed": poisson_seed,
                "topology": topology_name,
                "topology_label": topology_cfg["label"],
                "n_neurons": int(args_dict["n_neurons"]),
                "dt_ms": dt_ms,
                "sim_ms": sim_ms,
                "recovery_ms": recovery_ms,
                "v_ext_hz": V_EXT,
                "poisson_amplitude": POISSON_AMPLITUDE,
                "stimulated_fraction_of_excitatory": 0.5,
                "condition": dict(trial),
                "target_rate_hz": topology_cfg["target_rate_hz"],
                "normalize_target_in_E": topology_cfg["normalize_target_in_E"],
                "normalize_target_in_I": topology_cfg["normalize_target_in_I"],
            },
            "recovery_metrics": recovery_metrics,
            "weight_details": weight_details,
            "network_summary": {
                "n_edges": int(before_weights.size),
                "n_excitatory": int(np.sum(~pop.inhibitory_mask.astype(bool))),
                "n_inhibitory": int(np.sum(pop.inhibitory_mask.astype(bool))),
            },
            "elapsed_sec": float(time.perf_counter() - t0),
        }
        _write_json(details_dir / f"{run_id}.json", details_payload)

        row.update(_flatten_scalar_metrics("recovery__", recovery_metrics))
        row.update(_flatten_weight_summary("weights_all_before__", weight_details["all"]["before_summary"]))
        row.update(_flatten_weight_summary("weights_all_after__", weight_details["all"]["after_summary"]))
        row["weight_change_min"] = float(weight_details["delta_summary"]["min"])
        row["weight_change_max"] = float(weight_details["delta_summary"]["max"])
        row["weight_change_mean"] = float(weight_details["delta_summary"]["mean"])
        row["mean_abs_weight_change"] = float(weight_details["delta_summary"]["mean_abs"])
        row["weight_change_std"] = float(weight_details["delta_summary"]["std"])
        row["n_edges"] = int(before_weights.size)
        row["status"] = "ok"
        row["elapsed_sec"] = float(time.perf_counter() - t0)
        return row
    except Exception as exc:
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["elapsed_sec"] = float(time.perf_counter() - t0)
        return row


def _write_summary_csv(rows: list[dict[str, float | str]], output_path: Path):
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the week12 Clopath plus normalization sweep across both network topologies."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "plasticity_normalization_sweep",
        help="Directory containing summary.csv and per-run detail JSON files.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=min(4, max(1, os.cpu_count() or 1)),
        help="Number of worker processes. Defaults to min(4, cpu_count).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Base seeds for repeated trials.",
    )
    parser.add_argument(
        "--n-neurons",
        type=int,
        default=1000,
        help="Neuron count for each generated network.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Simulation timestep in ms.",
    )
    parser.add_argument(
        "--sim-ms",
        type=float,
        default=5000.0,
        help="Learning-phase duration in ms with stimulation on.",
    )
    parser.add_argument(
        "--recovery-ms",
        type=float,
        default=1000.0,
        help="Post-learning duration in ms with plasticity and normalization off.",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=HIST_BINS,
        help="Histogram bins saved for before/after weight distributions.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap on the total number of runs. Useful for smoke tests.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    trials = _build_trials(tuple(int(seed) for seed in args.seeds))
    if args.max_runs is not None:
        trials = trials[: int(args.max_runs)]
    if len(trials) == 0:
        raise ValueError("No trials selected.")

    args_dict = {
        "output_dir": str(args.output_dir),
        "n_neurons": int(args.n_neurons),
        "dt": float(args.dt),
        "sim_ms": float(args.sim_ms),
        "recovery_ms": float(args.recovery_ms),
        "hist_bins": int(args.hist_bins),
    }

    print(f"Running {len(trials)} runs with up to {args.max_workers} workers...")
    rows: list[dict[str, float | str]] = []
    completed = 0
    progress_step = max(1, math.ceil(len(trials) / 20))

    if args.max_workers == 1:
        for trial in trials:
            rows.append(_run_single_trial(trial, args_dict))
            completed += 1
            if completed % progress_step == 0 or completed == len(trials):
                print(f"Completed {completed}/{len(trials)}")
    else:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(_run_single_trial, trial, args_dict) for trial in trials]
            for future in as_completed(futures):
                rows.append(future.result())
                completed += 1
                if completed % progress_step == 0 or completed == len(trials):
                    print(f"Completed {completed}/{len(trials)}")

    rows.sort(
        key=lambda row: (
            str(row["topology"]),
            str(row["condition_name"]),
            float(row["seed"]),
        )
    )
    summary_path = args.output_dir / "summary.csv"
    _write_summary_csv(rows, summary_path)
    print(f"Wrote {len(rows)} rows to {summary_path}")


if __name__ == "__main__":
    main()
