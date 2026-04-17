import argparse
import csv
import gc
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.connectome import Connectome
from src.external_inputs import PoissonInput
from src.neuron_population import NeuronPopulation
from src.overhead import Simulation
from src.network_generators import (
    generate_random_fixed_indegree_ei_network,
    generate_spatial_ei_network,
)


DEFAULT_VARIANCES = (0.0, 0.5, 1.0, 3.0, 7.0)
DEFAULT_INPUT_AMPLITUDES = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0, 10.0, 50.0)
TOPOLOGY_CHOICES = ("spatial", "fixed")
HETEROGENEITY_PARAM_INDEX = 6
HETEROGENEITY_PARAM_NAME = "Vt"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Persistent ramp-up/ramp-down external-input sweep for mapping bifurcations "
            "across threshold-voltage heterogeneity conditions in the week14 setup."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "week14" / "results" / "threshold_heterogeneity_bifurcation_sweep.csv",
        help="CSV path for per-block results.",
    )
    parser.add_argument(
        "--topologies",
        nargs="+",
        choices=TOPOLOGY_CHOICES,
        default=list(TOPOLOGY_CHOICES),
        help="Topology families to sweep.",
    )
    parser.add_argument(
        "--variance-values",
        nargs="+",
        type=float,
        default=list(DEFAULT_VARIANCES),
        help="Variance values for Gaussian threshold-voltage heterogeneity.",
    )
    parser.add_argument(
        "--input-amplitudes",
        nargs="+",
        type=float,
        default=list(DEFAULT_INPUT_AMPLITUDES),
        help="Ramp amplitudes for the Poisson input. The full schedule is ramp-up followed by ramp-down.",
    )
    parser.add_argument("--seed-start", type=int, default=1234, help="First topology seed.")
    parser.add_argument(
        "--seed-count",
        type=int,
        default=3,
        help="Number of topology seeds per condition.",
    )
    parser.add_argument("--n-neurons", type=int, default=1000, help="Neuron count.")
    parser.add_argument("--fixed-indegree", type=int, default=100, help="Indegree for fixed topology.")
    parser.add_argument("--dt-ms", type=float, default=0.1, help="Simulation timestep in ms.")
    parser.add_argument(
        "--block-ms",
        type=float,
        default=1500.0,
        help="Duration in ms for each tested input amplitude block.",
    )
    parser.add_argument(
        "--metric-window-ms",
        type=float,
        default=1000.0,
        help="Use only the last metric-window-ms of each block when calling compute_metrics.",
    )
    parser.add_argument("--input-rate-hz", type=float, default=50.0, help="Poisson input rate in Hz.")
    parser.add_argument(
        "--truncate-std",
        type=float,
        default=2.0,
        help="Gaussian truncation in units of standard deviations.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output CSV instead of resuming/appending.",
    )
    parser.add_argument(
        "--job-index",
        type=int,
        default=0,
        help="0-based job index for partitioning the full sweep across multiple workers.",
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=1,
        help="Total number of disjoint jobs to split the full sweep into.",
    )
    return parser.parse_args()


def format_float_key(value):
    return f"{float(value):.6g}"


def scalarize_metric(value):
    if isinstance(value, np.generic):
        return value.item()
    if np.isscalar(value):
        return value
    return None


def steps_for_interval(duration_ms, dt_ms):
    n_steps = int(round(float(duration_ms) / float(dt_ms)))
    if n_steps <= 0:
        raise ValueError("Duration must produce at least one simulation step.")
    realized = n_steps * float(dt_ms)
    if not np.isclose(realized, float(duration_ms), atol=1e-9, rtol=0.0):
        raise ValueError(
            f"Duration {duration_ms} ms is not an integer multiple of dt {dt_ms} ms."
        )
    return n_steps


def build_sweep_schedule(amplitudes):
    amplitudes = [float(a) for a in amplitudes]
    schedule = []
    for step_index, amplitude in enumerate(amplitudes):
        schedule.append(
            {
                "sweep_direction": "up",
                "direction_step_index": step_index,
                "input_amplitude": amplitude,
            }
        )
    for step_index, amplitude in enumerate(reversed(amplitudes)):
        schedule.append(
            {
                "sweep_direction": "down",
                "direction_step_index": step_index,
                "input_amplitude": float(amplitude),
            }
        )
    return schedule


def get_completed_keys(output_path):
    completed = set()
    fieldnames = None
    if not output_path.exists():
        return completed, fieldnames

    with output_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        for row in reader:
            completed.add(
                (
                    row["topology"],
                    row["topology_seed"],
                    row["variance_ss4_vt"],
                    row["variance_b_vt"],
                    row["sweep_direction"],
                    row["direction_step_index"],
                    row["input_amplitude"],
                )
            )
    return completed, fieldnames


def resolve_output_path(output_path, job_index, num_jobs):
    output_path = output_path.resolve()
    if num_jobs <= 1:
        return output_path

    return output_path.with_name(
        f"{output_path.stem}.job{job_index + 1:02d}of{num_jobs:02d}{output_path.suffix}"
    )


def get_topology_config(topology_name):
    common = {
        "type_fractions": {"ss4": 0.8, "b": 0.2},
        "inhibitory_types": ("b",),
        "weight_dist_by_ntype": {"ss4": "lognormal", "b": "normal"},
        "mu_E": 0.0,
        "sigma_E": 1.6,
        "mu_I": 30.0,
        "sigma_I": 6.0,
        "normalize_mode": "in",
    }

    if topology_name == "spatial":
        return {
            **common,
            "generator": generate_spatial_ei_network,
            "generator_name": "generate_spatial_ei_network",
            "normalize_target_in_E": {"ss4": 50.0, "b": 10.0},
            "normalize_target_in_I": {"ss4": 500.0, "b": 50.0},
            "generator_kwargs": {
                "p0_by_pair": {"EE": 0.5, "EI": 0.5, "IE": 0.5, "II": 0.5},
                "lambda_by_preclass": {"E": 0.2, "I": 0.2},
                "distance_scale": 20.0,
                "outdegree_config_by_type": {
                    "ss4": {"dist": "lognormal", "params": (2.65, 0.8)},
                    "b": {"dist": "neg-bin", "params": (50, 40)},
                },
            },
        }

    if topology_name == "fixed":
        return {
            **common,
            "generator": generate_random_fixed_indegree_ei_network,
            "generator_name": "generate_random_fixed_indegree_ei_network",
            "normalize_target_in_E": {"ss4": 50.0, "b": 20.0},
            "normalize_target_in_I": {"ss4": 250.0, "b": 25.0},
            "generator_kwargs": {
                "indegree": 100,
                "delay_mean_E": 10.0,
                "delay_std_E": 3.0,
                "delay_mean_I": 1.5,
                "delay_std_I": 0.45,
            },
        }

    raise ValueError(f"Unknown topology '{topology_name}'.")


def build_graph(topology_name, topology_seed, n_neurons, fixed_indegree):
    topo_cfg = get_topology_config(topology_name)
    generator_kwargs = dict(topo_cfg["generator_kwargs"])
    if topology_name == "fixed":
        generator_kwargs["indegree"] = min(fixed_indegree, n_neurons - 1)

    G = topo_cfg["generator"](
        n_neurons=n_neurons,
        seed=topology_seed,
        type_fractions=topo_cfg["type_fractions"],
        inhibitory_types=topo_cfg["inhibitory_types"],
        weight_dist_by_ntype=topo_cfg["weight_dist_by_ntype"],
        mu_E=topo_cfg["mu_E"],
        sigma_E=topo_cfg["sigma_E"],
        mu_I=topo_cfg["mu_I"],
        sigma_I=topo_cfg["sigma_I"],
        normalize_mode=topo_cfg["normalize_mode"],
        normalize_target_in_E=topo_cfg["normalize_target_in_E"],
        normalize_target_in_I=topo_cfg["normalize_target_in_I"],
        **generator_kwargs,
    )
    return G, topo_cfg


def build_heterogeneity(var_ss4, var_b, truncate_std):
    return {
        "ss4": {
            "params": [HETEROGENEITY_PARAM_INDEX],
            "distributions": [("gaussian", None, float(var_ss4), float(truncate_std))],
        },
        "b": {
            HETEROGENEITY_PARAM_INDEX: ("gaussian", None, float(var_b), float(truncate_std)),
        },
    }


def build_connectome_from_graph(G, threshold_decay, heterogeneity, pop_rng):
    neuron_types = ["ss4", "b"]
    inhibitory = [False, True]
    pop = NeuronPopulation(
        G.number_of_nodes(),
        neuron_types,
        inhibitory,
        threshold_decay,
        heterogeneity=heterogeneity,
        rng=pop_rng,
    )

    max_synapses = max(dict(G.out_degree()).values())
    connectome = Connectome(max_synapses, pop)
    connectome.nx_to_connectome(G)
    return pop, connectome


def build_nmda_weight(population):
    nmda_weight = np.ones(population.n_neurons, dtype=float)
    nmda_weight[population.inhibitory_mask.astype(bool)] = 0.959685703507305 * 0.5
    return nmda_weight


def build_input_mask(population):
    inhib_mask = population.inhibitory_mask.astype(bool)
    excit_mask = ~inhib_mask
    input_mask = excit_mask.copy()
    input_mask[: min(500, input_mask.size)] = False
    return input_mask


def build_initial_state(n_neurons, init_rng):
    Vs = init_rng.uniform(-100.0, -70.0, size=n_neurons)
    us = init_rng.uniform(0.0, 400.0, size=n_neurons)
    spikes = np.zeros(n_neurons, dtype=bool)
    Ts = np.zeros(n_neurons, dtype=float)
    return Vs, us, spikes, Ts


def build_simulation(connectome, dt_ms, init_rng):
    state0 = build_initial_state(connectome.neuron_population.n_neurons, init_rng)
    nmda_weight = build_nmda_weight(connectome.neuron_population)
    sim = Simulation(
        connectome,
        dt_ms,
        stepper_type="euler_det",
        state0=state0,
        synapse_type="standard",
        synapse_kwargs={"LT_scale": 1.0, "NMDA_weight": nmda_weight},
        enable_state_logger=True,
        enable_debug_logger=False,
        enable_plasticity=False,
        rate_normalization=None,
    )
    return sim


def trial_seeds(base_seed, topology_seed, topology_idx, var_ss4_idx, var_b_idx):
    seed_sequence = np.random.SeedSequence(
        [int(base_seed), int(topology_seed), int(topology_idx), int(var_ss4_idx), int(var_b_idx)]
    )
    pop_seq, init_seq, poisson_seq = seed_sequence.spawn(3)
    return (
        np.random.default_rng(pop_seq),
        np.random.default_rng(init_seq),
        np.random.default_rng(poisson_seq),
    )


def block_key(trial, schedule_item):
    return (
        trial["topology_name"],
        str(trial["topology_seed"]),
        format_float_key(trial["var_ss4"]),
        format_float_key(trial["var_b"]),
        schedule_item["sweep_direction"],
        str(schedule_item["direction_step_index"]),
        format_float_key(schedule_item["input_amplitude"]),
    )


def write_row(output_path, row, fieldnames, writer_state):
    if writer_state["handle"] is None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer_state["handle"] = output_path.open("a", newline="", encoding="utf-8")
        writer_state["writer"] = csv.DictWriter(writer_state["handle"], fieldnames=fieldnames)
        if writer_state["handle"].tell() == 0:
            writer_state["writer"].writeheader()

    writer_state["writer"].writerow(row)
    writer_state["handle"].flush()


def enumerate_trials(topologies, variance_values, topology_seeds):
    for topology_idx, topology_name in enumerate(topologies):
        for var_ss4_idx, var_ss4 in enumerate(variance_values):
            for var_b_idx, var_b in enumerate(variance_values):
                for topology_seed in topology_seeds:
                    yield {
                        "topology_idx": topology_idx,
                        "topology_name": topology_name,
                        "var_ss4_idx": var_ss4_idx,
                        "var_ss4": var_ss4,
                        "var_b_idx": var_b_idx,
                        "var_b": var_b,
                        "topology_seed": topology_seed,
                    }


def main():
    args = parse_args()

    if args.n_neurons <= 1:
        raise ValueError("n_neurons must be greater than 1.")
    if args.seed_count <= 0:
        raise ValueError("seed_count must be greater than 0.")
    if args.num_jobs <= 0:
        raise ValueError("num_jobs must be greater than 0.")
    if not 0 <= args.job_index < args.num_jobs:
        raise ValueError("job_index must satisfy 0 <= job_index < num_jobs.")
    if "fixed" in args.topologies and args.fixed_indegree >= args.n_neurons:
        raise ValueError("fixed_indegree must be smaller than n_neurons.")
    if args.metric_window_ms <= 0.0:
        raise ValueError("metric_window_ms must be greater than 0.")
    if args.block_ms <= 0.0:
        raise ValueError("block_ms must be greater than 0.")
    if args.metric_window_ms > args.block_ms:
        raise ValueError("metric_window_ms cannot exceed block_ms.")

    steps_for_interval(args.block_ms, args.dt_ms)
    schedule = build_sweep_schedule(args.input_amplitudes)

    output_path = resolve_output_path(args.output, args.job_index, args.num_jobs)
    if args.overwrite and output_path.exists():
        output_path.unlink()

    completed_keys, existing_fieldnames = get_completed_keys(output_path)
    writer_state = {"handle": None, "writer": None}

    topology_seeds = [args.seed_start + i for i in range(args.seed_count)]
    all_trials = list(enumerate_trials(args.topologies, args.variance_values, topology_seeds))
    job_trials = [trial for idx, trial in enumerate(all_trials) if idx % args.num_jobs == args.job_index]
    total_trials = len(job_trials)

    start_all = time.time()
    completed_now = 0

    print(
        f"Job {args.job_index + 1}/{args.num_jobs}: "
        f"{total_trials} assigned conditions, {len(schedule)} blocks per condition, output={output_path}",
        flush=True,
    )

    try:
        for trial in job_trials:
            expected_keys = {block_key(trial, schedule_item) for schedule_item in schedule}
            topology_name = trial["topology_name"]
            topology_seed = trial["topology_seed"]
            var_ss4 = trial["var_ss4"]
            var_b = trial["var_b"]

            if expected_keys.issubset(completed_keys):
                completed_now += 1
                print(
                    f"[skip {completed_now}/{total_trials}] job={args.job_index + 1}/{args.num_jobs} "
                    f"topology={topology_name} seed={topology_seed} var_ss4={var_ss4} var_b={var_b}",
                    flush=True,
                )
                continue

            t0_condition = time.time()
            threshold_decay = float(np.exp(-args.dt_ms / 5.0))
            heterogeneity = build_heterogeneity(var_ss4, var_b, args.truncate_std)
            pop_rng, init_rng, poisson_rng = trial_seeds(
                args.seed_start,
                topology_seed,
                trial["topology_idx"],
                trial["var_ss4_idx"],
                trial["var_b_idx"],
            )

            G, topo_cfg = build_graph(
                topology_name=topology_name,
                topology_seed=topology_seed,
                n_neurons=args.n_neurons,
                fixed_indegree=args.fixed_indegree,
            )
            population, connectome = build_connectome_from_graph(
                G,
                threshold_decay=threshold_decay,
                heterogeneity=heterogeneity,
                pop_rng=pop_rng,
            )
            sim = build_simulation(connectome=connectome, dt_ms=args.dt_ms, init_rng=init_rng)
            input_mask = build_input_mask(population)
            block_steps = steps_for_interval(args.block_ms, args.dt_ms)

            for global_block_index, schedule_item in enumerate(schedule):
                block_start_ms = float(sim.t_now)
                input_amplitude = schedule_item["input_amplitude"]
                poisson = PoissonInput(
                    connectome.neuron_population.n_neurons,
                    rate=args.input_rate_hz * input_mask,
                    amplitude=input_amplitude,
                    rng=poisson_rng,
                )

                for _ in range(block_steps):
                    sim.step(spike_ext=poisson(args.dt_ms), reward=1.0)

                block_stop_ms = float(sim.t_now)
                metric_t_start_ms = block_stop_ms - float(args.metric_window_ms)
                metric_t_stop_ms = block_stop_ms
                metrics = sim.stats.compute_metrics(
                    args.dt_ms,
                    bin_ms_participation=300.0,
                    t_start_ms=metric_t_start_ms,
                    t_stop_ms=metric_t_stop_ms,
                )

                scalar_metrics = {}
                for key, value in metrics.items():
                    scalar_value = scalarize_metric(value)
                    if scalar_value is not None:
                        scalar_metrics[key] = scalar_value

                row = {
                    "job_index": args.job_index,
                    "num_jobs": args.num_jobs,
                    "topology": topology_name,
                    "generator_name": topo_cfg["generator_name"],
                    "topology_seed": topology_seed,
                    "variance_ss4_vt": format_float_key(var_ss4),
                    "variance_b_vt": format_float_key(var_b),
                    "truncate_std": args.truncate_std,
                    "heterogeneity_param_index": HETEROGENEITY_PARAM_INDEX,
                    "heterogeneity_param_name": HETEROGENEITY_PARAM_NAME,
                    "n_neurons": args.n_neurons,
                    "fixed_indegree": min(args.fixed_indegree, args.n_neurons - 1),
                    "dt_ms": args.dt_ms,
                    "block_ms": args.block_ms,
                    "metric_window_ms": args.metric_window_ms,
                    "input_rate_hz": args.input_rate_hz,
                    "input_amplitude": format_float_key(input_amplitude),
                    "sweep_direction": schedule_item["sweep_direction"],
                    "direction_step_index": schedule_item["direction_step_index"],
                    "global_block_index": global_block_index,
                    "block_start_ms": round(block_start_ms, 6),
                    "block_stop_ms": round(block_stop_ms, 6),
                    "metric_t_start_ms": round(metric_t_start_ms, 6),
                    "metric_t_stop_ms": round(metric_t_stop_ms, 6),
                    "normalize_mode_topology": topo_cfg["normalize_mode"],
                    "normalize_target_in_E_ss4": topo_cfg["normalize_target_in_E"]["ss4"],
                    "normalize_target_in_E_b": topo_cfg["normalize_target_in_E"]["b"],
                    "normalize_target_in_I_ss4": topo_cfg["normalize_target_in_I"]["ss4"],
                    "normalize_target_in_I_b": topo_cfg["normalize_target_in_I"]["b"],
                    "runtime_plasticity": False,
                    "runtime_rate_normalization": False,
                    "elapsed_sec": round(time.time() - t0_condition, 6),
                }
                row.update(scalar_metrics)

                key = block_key(trial, schedule_item)
                if key not in completed_keys:
                    fieldnames = existing_fieldnames or list(row.keys())
                    write_row(output_path, row, fieldnames, writer_state)
                    existing_fieldnames = fieldnames
                    completed_keys.add(key)

                print(
                    f"[block {global_block_index + 1}/{len(schedule)}] "
                    f"topology={topology_name} seed={topology_seed} "
                    f"var_ss4={var_ss4} var_b={var_b} "
                    f"dir={schedule_item['sweep_direction']} amp={input_amplitude} "
                    f"t=({block_start_ms:.1f},{block_stop_ms:.1f})",
                    flush=True,
                )

                if global_block_index < len(schedule) - 1:
                    sim.reset_stats()
                del metrics, scalar_metrics, row, poisson

            completed_now += 1
            print(
                f"[done {completed_now}/{total_trials}] job={args.job_index + 1}/{args.num_jobs} "
                f"topology={topology_name} seed={topology_seed} var_ss4={var_ss4} var_b={var_b} "
                f"elapsed={time.time() - t0_condition:.2f}s",
                flush=True,
            )

            del G, population, connectome, sim, input_mask, pop_rng, init_rng, poisson_rng
            gc.collect()
    finally:
        if writer_state["handle"] is not None:
            writer_state["handle"].close()

    total_elapsed = time.time() - start_all
    print(f"Finished. Wrote results to {output_path} in {total_elapsed:.2f}s.", flush=True)


if __name__ == "__main__":
    main()
