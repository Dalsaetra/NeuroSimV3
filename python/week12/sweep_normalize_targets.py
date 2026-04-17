from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import random
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
from src.network_generators import generate_spatial_ei_network
from src.neuron_population import NeuronPopulation
from src.overhead import Simulation


DELAY_MEAN_E = 10.0
DELAY_STD_E = DELAY_MEAN_E * 0.3
DELAY_MEAN_I = 1.5
DELAY_STD_I = DELAY_MEAN_I * 0.3
V_EXT = 50.0

EXCITATORY_TYPE = "ss4"
INHIBITORY_TYPE = "b"
SEED = 1234
DT = 0.1
SIM_LEN_MS = 1000.0
METRIC_BIN_MS_PARTICIPATION = 300.0
DEFAULT_IN_E_SS4_VALUES = (5.0, 10.0, 20.0, 30.0, 50.0)
DEFAULT_IN_E_B_VALUES = (0.5, 1.0, 5.0, 10.0, 20.0)
DEFAULT_IN_I_SS4_VALUES = (50.0, 100.0, 250.0, 500.0, 1000.0)
DEFAULT_IN_I_B_VALUES = (10.0, 25.0, 50.0, 100.0, 250.0)


def _build_network(normalize_target_in_E: dict[str, float], normalize_target_in_I: dict[str, float]):
    n_neurons = int(1000 * 1.0)
    type_fractions = {"ss4": 0.8, "b": 0.2}
    inhibitory_types = ("b",)
    p0_by_pair = {"EE": 0.5, "EI": 0.5, "IE": 0.5, "II": 0.5}
    lambda_by_preclass = {"E": 0.2, "I": 0.05}
    weight_dist_by_ntype = {"ss4": "lognormal", "b": "normal"}
    outdegree_config_by_type = {
        "ss4": {"dist": "lognormal", "params": (2.65, 0.8)},
        "b": {"dist": "neg-bin", "params": (50, 40)},
    }

    return generate_spatial_ei_network(
        n_neurons=n_neurons,
        type_fractions=type_fractions,
        inhibitory_types=inhibitory_types,
        mu_E=0.0,
        sigma_E=1.6,
        mu_I=30.0,
        sigma_I=6.0,
        p0_by_pair=p0_by_pair,
        lambda_by_preclass=lambda_by_preclass,
        distance_scale=20.0,
        weight_dist_by_ntype=weight_dist_by_ntype,
        outdegree_config_by_type=outdegree_config_by_type,
        normalize_mode="in",
        normalize_target_in_E=normalize_target_in_E,
        normalize_target_in_I=normalize_target_in_I,
        seed=SEED,
    )


def _build_simulation(G):
    n_neurons = G.number_of_nodes()
    neuron_types = [EXCITATORY_TYPE, INHIBITORY_TYPE]
    inhibitory = [False, True]
    threshold_decay = np.exp(-DT / 5)

    pop = NeuronPopulation(n_neurons, neuron_types, inhibitory, threshold_decay)

    max_synapses = max(dict(G.out_degree()).values())
    connectome = Connectome(max_synapses, pop)
    connectome.nx_to_connectome(G)

    nmda_weight = np.ones(connectome.neuron_population.n_neurons, dtype=float)
    nmda_weight[pop.inhibitory_mask.astype(bool)] = 0.959685703507305 * 0.5 * 0.1

    out_degrees = dict(G.out_degree())
    top_neurons = sorted(out_degrees, key=out_degrees.get, reverse=True)[: int(0.05 * len(out_degrees))]
    random.Random(SEED).shuffle(top_neurons)

    rng = np.random.default_rng(SEED)
    Vs = rng.uniform(-100, -70, size=n_neurons)
    us = rng.uniform(0, 400, size=n_neurons)
    spikes = np.zeros(n_neurons, dtype=bool)
    Ts = np.zeros_like(spikes)
    state0 = (Vs, us, spikes.copy(), Ts.copy())

    sim = Simulation(
        connectome,
        DT,
        stepper_type="euler_det",
        state0=state0,
        enable_plasticity=False,
        synapse_kwargs={"LT_scale": 1.0, "NMDA_weight": nmda_weight},
        synapse_type="standard",
        enable_state_logger=True,
        enable_debug_logger=False,
    )
    sim.configure_output_readout(output_neuron_indices=top_neurons, output_dim=2, rate_window_ms=200.0)
    return sim, pop


def _run_simulation(sim, pop):
    input_mask = (~pop.inhibitory_mask.astype(bool)).astype(float)
    poisson = PoissonInput(pop.n_neurons, rate=V_EXT * input_mask, amplitude=0.5)
    steps = int(SIM_LEN_MS / DT)
    for _ in range(steps):
        sim.step(spike_ext=poisson(DT), reward=1.0)


def _serialize_metric_value(value):
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return json.dumps(value)
    return value


def run_single_combo(combo: tuple[float, float, float, float]) -> dict[str, float | str]:
    e_ss4, e_b, i_ss4, i_b = combo
    row: dict[str, float | str] = {
        "normalize_target_in_E_ss4": float(e_ss4),
        "normalize_target_in_E_b": float(e_b),
        "normalize_target_in_I_ss4": float(i_ss4),
        "normalize_target_in_I_b": float(i_b),
    }
    t0 = time.perf_counter()

    try:
        G = _build_network(
            normalize_target_in_E={"ss4": float(e_ss4), "b": float(e_b)},
            normalize_target_in_I={"ss4": float(i_ss4), "b": float(i_b)},
        )
        sim, pop = _build_simulation(G)
        _run_simulation(sim, pop)
        metrics = sim.stats.compute_metrics(DT, bin_ms_participation=METRIC_BIN_MS_PARTICIPATION)
        row.update({key: _serialize_metric_value(value) for key, value in metrics.items()})
        row["status"] = "ok"
        row["elapsed_sec"] = float(time.perf_counter() - t0)
        return row
    except Exception as exc:
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["elapsed_sec"] = float(time.perf_counter() - t0)
        return row


def write_rows_to_csv(rows: list[dict[str, float | str]], output_path: Path) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep week12 normalize_target_in_E/I parameters over a 5x5x5x5 grid."
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of parallel worker processes. Defaults to os.cpu_count().",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "week12_normalize_target_sweep.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--in-e-ss4-values",
        type=float,
        nargs="+",
        default=list(DEFAULT_IN_E_SS4_VALUES),
        help="Sweep values for normalize_target_in_E['ss4'].",
    )
    parser.add_argument(
        "--in-e-b-values",
        type=float,
        nargs="+",
        default=list(DEFAULT_IN_E_B_VALUES),
        help="Sweep values for normalize_target_in_E['b'].",
    )
    parser.add_argument(
        "--in-i-ss4-values",
        type=float,
        nargs="+",
        default=list(DEFAULT_IN_I_SS4_VALUES),
        help="Sweep values for normalize_target_in_I['ss4'].",
    )
    parser.add_argument(
        "--in-i-b-values",
        type=float,
        nargs="+",
        default=list(DEFAULT_IN_I_B_VALUES),
        help="Sweep values for normalize_target_in_I['b'].",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_e_ss4_values = tuple(float(v) for v in args.in_e_ss4_values)
    in_e_b_values = tuple(float(v) for v in args.in_e_b_values)
    in_i_ss4_values = tuple(float(v) for v in args.in_i_ss4_values)
    in_i_b_values = tuple(float(v) for v in args.in_i_b_values)
    combos = list(itertools.product(in_e_ss4_values, in_e_b_values, in_i_ss4_values, in_i_b_values))
    n_workers = args.max_workers
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1))
    if n_workers <= 0:
        raise ValueError("max_workers must be >= 1")

    rows: list[dict[str, float | str]] = []
    print(f"Running {len(combos)} combinations with {n_workers} workers...")

    completed = 0
    progress_step = max(1, math.ceil(len(combos) / 20))

    if n_workers == 1:
        for combo in combos:
            row = run_single_combo(combo)
            rows.append(row)
            completed += 1
            if completed % progress_step == 0 or completed == len(combos):
                print(f"Completed {completed}/{len(combos)}")
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_combo = {executor.submit(run_single_combo, combo): combo for combo in combos}
            for future in as_completed(future_to_combo):
                row = future.result()
                rows.append(row)
                completed += 1
                if completed % progress_step == 0 or completed == len(combos):
                    print(f"Completed {completed}/{len(combos)}")

    rows.sort(
        key=lambda row: (
            float(row["normalize_target_in_E_ss4"]),
            float(row["normalize_target_in_E_b"]),
            float(row["normalize_target_in_I_ss4"]),
            float(row["normalize_target_in_I_b"]),
        )
    )
    write_rows_to_csv(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
