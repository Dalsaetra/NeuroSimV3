"""Grid search over topology and parameter axes for final-paper analyses.

The default sweep is the original normalization sweep. Its parameterization is
relative to the values used in reservoir_tests/reservoir_tests.ipynb:

    normalize_target_in_E_scaled = base_E * J
    normalize_target_in_I_scaled = base_I * J * g

Additional sweeps keep the same analysis pipeline and base normalization:

    --sweep b-target-normalization
        J scales all four normalization constants; g scales both E->b and I->b
        incoming normalization targets.

    --sweep delay
        Fixed topology should use --fixed-e-delay-values and
        --fixed-i-delay-values, in ms. Delay std is --delay-std-fraction times
        the corresponding mean. Spatial topology should use
        --spatial-e-lambda-values and --spatial-i-lambda-values.
        Generic --x-values/--y-values still work as a fallback for either
        topology.

    --sweep threshold-heterogeneity
        x/y are Gaussian Vt heterogeneity variances for ss4(E) and b(I).

    --sweep size-inhibition
        x is n_neurons and y is the inhibitory fraction.

Run examples:

    python FinalPaper/grid_search_normalization_topology.py --j-values 0.5 1.0 2.0 --g-values 0.5 1.0 2.0

    python FinalPaper/grid_search_normalization_topology.py --j-values 0.8 1.0 --g-values 1.0 1.5 --job-index 0 --num-jobs 4

    python FinalPaper/grid_search_normalization_topology.py --sweep delay --topologies fixed spatial --fixed-e-delay-values 5 10 15 --fixed-i-delay-values 1 1.5 2 --spatial-e-lambda-values 0.1 0.2 0.4 --spatial-i-lambda-values 0.05 0.1 0.2

    python FinalPaper/grid_search_normalization_topology.py --output-dir FinalPaper/results/normalization_grid --augment-existing --add-j-values 0.6 --add-g-values 0.4
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.connectome import Connectome
from src.network_generators import (
    generate_random_fixed_indegree_ei_network,
    generate_spatial_ei_network,
)
from src.neuron_population import NeuronPopulation
from src.overhead import Simulation


TOPOLOGY_BASE_NORMALIZATION = {
    "fixed": {
        "normalize_target_in_E": {"ss4": 50.0, "b": 20.0},
        "normalize_target_in_I": {"ss4": 250.0, "b": 25.0},
    },
    "spatial": {
        "normalize_target_in_E": {"ss4": 50.0, "b": 10.0},
        "normalize_target_in_I": {"ss4": 500.0, "b": 50.0},
    },
}


TYPE_FRACTIONS = {"ss4": 0.8, "b": 0.2}
INHIBITORY_TYPES = ("b",)
WEIGHT_DIST_BY_NTYPE = {"ss4": "lognormal", "b": "normal"}
P0_BY_PAIR = {"EE": 0.5, "EI": 0.5, "IE": 0.5, "II": 0.5}
LAMBDA_BY_PRECLASS = {"E": 0.2, "I": 0.2}
OUTDEGREE_CONFIG_BY_TYPE = {
    "ss4": {"dist": "lognormal", "params": (2.65, 0.8)},
    "b": {"dist": "neg-bin", "params": (50, 40)},
}

SWEEP_AXIS_NAMES = {
    "normalization": ("J", "g"),
    "b-target-normalization": ("J", "g_b_target"),
    "delay": ("E_delay_or_lambda", "I_delay_or_lambda"),
    "threshold-heterogeneity": ("E_threshold_variance", "I_threshold_variance"),
    "size-inhibition": ("n_neurons", "inhibitory_fraction"),
}

SWEEP_CHOICES = tuple(SWEEP_AXIS_NAMES.keys())
DEFAULT_THRESHOLD_VARIANCE_E = 3.0
DEFAULT_THRESHOLD_VARIANCE_I = 1.0
DEFAULT_THRESHOLD_TRUNCATE_STD = 2.0


NOTEBOOK_ANALYSIS_SETTINGS = {
    "regime": {
        "warmup_ms": 500.0,
        "observation_ms": 2000.0,
        "transient_ms": 2000.0,
        "input_fraction": 0.3,
        "input_pool": "E",
        "input_rate_hz": 50.0,
        "input_amplitude": 0.5,
        "plot_raster": False,
        "store_simulation_stats": False,
    },
    "bifurcation": {
        "input_rate_hz": 50.0,
        "warmup_ms": 500.0,
        "measurement_ms": 1000.0,
        "input_fraction": 0.3,
        "input_pool": "E",
    },
    "separation": {
        "n_trials": 50,
        "input_fraction": 0.6,
        "input_pool": "E",
        "disjoint_state_pool": True,
        "state_fraction": 1.0,
        "state_pool": "E",
        "input_active_probability": 0.5,
        "input_rate_hz": 50.0,
        "input_amplitude": 0.5,
        "warmup_ms": 500.0,
        "measurement_ms": 500.0,
        "tau_ms": 30.0,
        "bin_ms": 20.0,
        "store_states": False,
    },
    "memory": {
        "input_fraction": 0.3,
        "input_pool": "E",
        "state_fraction": 1.0,
        "state_pool": "E",
        "disjoint_state_pool": True,
        "input_rate_min_hz": 1.0,
        "input_rate_max_hz": 200.0,
        "input_amplitude": 2.0,
        "washout_ms": 1000.0,
        "measurement_ms": 30000.0,
        "tau_ms": 150.0,
        "bin_ms": 30.0,
        "max_delay_bins": 50,
        "train_fraction": 0.7,
        "ridge_alpha": 1000.0,
        "store_states": False,
        "store_readouts": False,
    },
    "generalization": {
        "n_classes": 5,
        "trials_per_class": 10,
        "input_fraction": 0.3,
        "input_pool": "E",
        "state_fraction": 1.0,
        "state_pool": "E",
        "disjoint_state_pool": True,
        "input_rate_min_hz": 5.0,
        "input_rate_max_hz": 200.0,
        "input_rate_noise_fraction": 0.1,
        "input_amplitude": 2.0,
        "warmup_ms": 250.0,
        "measurement_ms": 500.0,
        "tau_ms": 30.0,
        "bin_ms": 30.0,
        "state_summary": "mean",
        "train_fraction": 0.7,
        "ridge_alpha": 1.0,
        "store_states": False,
        "store_rate_vectors": False,
    },
}


@dataclass(frozen=True)
class Condition:
    condition_id: int
    sweep: str
    topology: str
    x_name: str
    x_value: float
    y_name: str
    y_value: float
    j_gain: float
    g_gain: float
    seed: int
    n_neurons: int | None = None
    inhibitory_fraction: float | None = None

    @property
    def key(self) -> str:
        if self.sweep == "normalization":
            return (
                f"topology={self.topology}|J={self.x_value:.8g}|"
                f"g={self.y_value:.8g}|seed={self.seed}"
            )
        return (
            f"sweep={self.sweep}|topology={self.topology}|"
            f"{self.x_name}={self.x_value:.8g}|{self.y_name}={self.y_value:.8g}|seed={self.seed}"
        )


def condition_json_filename(condition: Condition) -> str:
    """Stable filename for a condition, independent of grid-axis expansion order."""
    digest = hashlib.sha1(condition.key.encode("utf-8")).hexdigest()[:16]
    return f"condition_{digest}.json"


def parse_float_list(values: list[str] | None) -> list[float]:
    if not values:
        return []
    parsed = []
    for value in values:
        for piece in str(value).replace(",", " ").split():
            parsed.append(float(piece))
    return parsed


def merge_float_values(*value_lists: list[float]) -> list[float]:
    values = []
    for value_list in value_lists:
        for value in value_list or []:
            values.append(float(value))
    return sorted(set(values))


def merge_int_values(*value_lists: list[int]) -> list[int]:
    values = []
    for value_list in value_lists:
        for value in value_list or []:
            values.append(int(value))
    return sorted(set(values))


def merge_str_values(*value_lists: list[str]) -> list[str]:
    values = []
    seen = set()
    for value_list in value_lists:
        for value in value_list or []:
            text = str(value)
            if text not in seen:
                values.append(text)
                seen.add(text)
    return values


def read_grid_configs(output_dir: Path) -> list[dict[str, Any]]:
    configs = []
    for path in sorted(output_dir.glob("grid_config_job*.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                config = json.load(handle)
            if isinstance(config, dict):
                config["_config_path"] = str(path)
                configs.append(config)
        except Exception:
            continue
    return configs


def config_list_values(configs: list[dict[str, Any]], key: str) -> list[Any]:
    values = []
    for config in configs:
        item = config.get(key)
        if isinstance(item, list):
            values.extend(item)
    return values


def config_first_value(configs: list[dict[str, Any]], key: str, default=None):
    for config in configs:
        if key in config and config[key] is not None:
            return config[key]
    return default


def config_float_values(configs: list[dict[str, Any]], key: str) -> list[float]:
    return merge_float_values([float(value) for value in config_list_values(configs, key)])


def config_int_values(configs: list[dict[str, Any]], key: str) -> list[int]:
    return merge_int_values([int(value) for value in config_list_values(configs, key)])


def config_str_values(configs: list[dict[str, Any]], key: str) -> list[str]:
    return merge_str_values([str(value) for value in config_list_values(configs, key)])


def resolved_type_fractions(inhibitory_fraction: float | None = None) -> dict[str, float]:
    if inhibitory_fraction is None:
        return dict(TYPE_FRACTIONS)
    inhibitory_fraction = float(inhibitory_fraction)
    if inhibitory_fraction <= 0.0 or inhibitory_fraction >= 1.0:
        raise ValueError("inhibitory_fraction must be in the open interval (0, 1).")
    return {"ss4": float(1.0 - inhibitory_fraction), "b": inhibitory_fraction}


def threshold_heterogeneity(
    variance_E: float = DEFAULT_THRESHOLD_VARIANCE_E,
    variance_I: float = DEFAULT_THRESHOLD_VARIANCE_I,
    truncate_std: float = DEFAULT_THRESHOLD_TRUNCATE_STD,
) -> dict[str, Any]:
    return {
        "ss4": {
            "params": [6],
            "distributions": [
                ("gaussian", None, float(variance_E), float(truncate_std)),
            ],
        },
        "b": {
            6: ("gaussian", None, float(variance_I), float(truncate_std)),
        },
    }


def scaled_normalization(
    topology: str,
    j_gain: float,
    g_gain: float,
    *,
    sweep: str = "normalization",
) -> tuple[dict[str, float], dict[str, float]]:
    if sweep not in ("normalization", "b-target-normalization"):
        j_gain = 1.0
        g_gain = 1.0
    base = TOPOLOGY_BASE_NORMALIZATION[topology]
    normalize_target_in_E = {
        ntype: float(value) * float(j_gain)
        for ntype, value in base["normalize_target_in_E"].items()
    }
    normalize_target_in_I = {
        ntype: float(value) * float(j_gain)
        for ntype, value in base["normalize_target_in_I"].items()
    }
    if sweep == "normalization":
        normalize_target_in_I = {
            ntype: float(value) * float(g_gain)
            for ntype, value in normalize_target_in_I.items()
        }
    elif sweep == "b-target-normalization":
        if "b" in normalize_target_in_E:
            normalize_target_in_E["b"] *= float(g_gain)
        if "b" in normalize_target_in_I:
            normalize_target_in_I["b"] *= float(g_gain)
    return normalize_target_in_E, normalize_target_in_I


def normalization_label(topology: str, normalize_target_in_E: dict[str, float], normalize_target_in_I: dict[str, float]) -> str:
    return (
        f"{topology}_"
        f"E_ss4{normalize_target_in_E['ss4']:g}_b{normalize_target_in_E['b']:g}_"
        f"I_ss4{normalize_target_in_I['ss4']:g}_b{normalize_target_in_I['b']:g}"
    )


def build_graph(
    condition: Condition,
    default_n_neurons: int,
    *,
    delay_std_fraction: float = 0.3,
):
    normalize_target_in_E, normalize_target_in_I = scaled_normalization(
        condition.topology,
        condition.j_gain,
        condition.g_gain,
        sweep=condition.sweep,
    )
    n_neurons = int(condition.n_neurons if condition.n_neurons is not None else default_n_neurons)
    type_fractions = resolved_type_fractions(condition.inhibitory_fraction)
    graph_config: dict[str, Any] = {
        "type_fractions": type_fractions,
        "delay_std_fraction": float(delay_std_fraction),
    }

    if condition.topology == "fixed":
        delay_mean_E = 10.0
        delay_mean_I = 1.5
        if condition.sweep == "delay":
            delay_mean_E = float(condition.x_value)
            delay_mean_I = float(condition.y_value)
        delay_std_E = max(1e-12, float(delay_mean_E) * float(delay_std_fraction))
        delay_std_I = max(1e-12, float(delay_mean_I) * float(delay_std_fraction))
        graph_config.update(
            {
                "fixed_indegree": int(min(100, n_neurons - 1)),
                "delay_mean_E": float(delay_mean_E),
                "delay_std_E": float(delay_std_E),
                "delay_mean_I": float(delay_mean_I),
                "delay_std_I": float(delay_std_I),
            }
        )
        graph = generate_random_fixed_indegree_ei_network(
            n_neurons=n_neurons,
            indegree=min(100, n_neurons - 1),
            type_fractions=type_fractions,
            inhibitory_types=INHIBITORY_TYPES,
            delay_mean_E=delay_mean_E,
            delay_std_E=delay_std_E,
            delay_mean_I=delay_mean_I,
            delay_std_I=delay_std_I,
            mu_E=0.0,
            sigma_E=1.6,
            mu_I=30.0,
            sigma_I=6.0,
            weight_dist_by_ntype=WEIGHT_DIST_BY_NTYPE,
            normalize_mode="in",
            normalize_target_in_E=normalize_target_in_E,
            normalize_target_in_I=normalize_target_in_I,
            seed=condition.seed,
        )
    elif condition.topology == "spatial":
        lambda_by_preclass = dict(LAMBDA_BY_PRECLASS)
        if condition.sweep == "delay":
            lambda_by_preclass = {"E": float(condition.x_value), "I": float(condition.y_value)}
        graph_config["lambda_by_preclass"] = lambda_by_preclass
        graph = generate_spatial_ei_network(
            n_neurons=n_neurons,
            type_fractions=type_fractions,
            inhibitory_types=INHIBITORY_TYPES,
            mu_E=0.0,
            sigma_E=1.6,
            mu_I=30.0,
            sigma_I=6.0,
            p0_by_pair=P0_BY_PAIR,
            lambda_by_preclass=lambda_by_preclass,
            distance_scale=20.0,
            weight_dist_by_ntype=WEIGHT_DIST_BY_NTYPE,
            outdegree_config_by_type=OUTDEGREE_CONFIG_BY_TYPE,
            normalize_mode="in",
            normalize_target_in_E=normalize_target_in_E,
            normalize_target_in_I=normalize_target_in_I,
            seed=condition.seed,
        )
    else:
        raise ValueError(f"Unknown topology: {condition.topology}")

    return graph, normalize_target_in_E, normalize_target_in_I, graph_config


def build_simulation(
    graph,
    *,
    dt_ms: float,
    seed: int,
    heterogeneity: dict[str, Any] | None = None,
) -> Simulation:
    np.random.seed(int(seed))
    n_neurons = graph.number_of_nodes()
    threshold_decay = np.exp(-dt_ms / 5.0)
    if heterogeneity is None:
        heterogeneity = threshold_heterogeneity()
    pop = NeuronPopulation(
        n_neurons,
        ["ss4", "b"],
        [False, True],
        threshold_decay,
        heterogeneity,
    )

    max_synapses = max(dict(graph.out_degree()).values())
    connectome = Connectome(max_synapses, pop)
    connectome.nx_to_connectome(graph.copy())

    rng = np.random.default_rng(seed)
    state0 = (
        rng.uniform(-100.0, -70.0, size=n_neurons),
        rng.uniform(0.0, 400.0, size=n_neurons),
        np.zeros(n_neurons, dtype=bool),
        np.zeros(n_neurons, dtype=bool),
    )

    nmda_weight = np.ones(n_neurons, dtype=float)
    nmda_weight[pop.inhibitory_mask.astype(bool)] = 0.959685703507305 * 0.5

    return Simulation(
        connectome,
        dt_ms,
        stepper_type="euler_det",
        state0=state0,
        synapse_kwargs={"LT_scale": 1.0, "NMDA_weight": nmda_weight},
        synapse_type="standard",
        enable_debug_logger=False,
        rate_normalization=None,
        enable_plasticity=False,
        plasticity="clopath",
        plasticity_reward_type="online",
        plasticity_kwargs={},
    )


def is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool, np.integer, np.floating, np.bool_))


def make_json_safe(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    if not is_scalar(value):
        return repr(value)
    return value


def flatten_scalars(prefix: str, data: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}{key}" if prefix else str(key)
        if is_scalar(value):
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                flat[name] = value
            elif isinstance(value, (np.integer, np.floating, np.bool_)):
                flat[name] = value.item()
            else:
                flat[name] = value
        elif isinstance(value, dict):
            flat.update(flatten_scalars(f"{name}.", value))
    return flat


def compact_metric_dict(metrics: dict[str, Any], *, keep_arrays: set[str] | None = None) -> dict[str, Any]:
    keep_arrays = set() if keep_arrays is None else set(keep_arrays)
    compact = {}
    for key, value in dict(metrics).items():
        if is_scalar(value):
            compact[key] = value.item() if isinstance(value, np.generic) else value
        elif key in keep_arrays:
            compact[key] = make_json_safe(value)
    return compact


def count_or_list(values: Any) -> int:
    if values is None:
        return 0
    try:
        return len(values)
    except TypeError:
        return int(values)


def index_count(result_section: dict[str, Any], indices_key: str, count_key: str) -> int:
    if count_key in result_section:
        return int(result_section[count_key])
    return count_or_list(result_section.get(indices_key, []))


def compact_outputs_for_json(outputs: dict[str, Any]) -> dict[str, Any]:
    """Keep per-condition JSON small while preserving aggregate-table inputs."""
    compact = {
        "condition": outputs.get("condition", {}),
        "settings": outputs.get("settings", {}),
    }

    regime = outputs.get("regime", {})
    regime_metrics = regime.get("metrics", {})
    compact["regime"] = {
        "classification": regime.get("classification"),
        "brunel": compact_metric_dict(regime.get("brunel", {})),
        "persistence": compact_metric_dict(regime.get("persistence", {})),
        "regeneration": compact_metric_dict(regime.get("regeneration", {})),
        "metrics": {
            name: compact_metric_dict(value)
            for name, value in regime_metrics.items()
            if isinstance(value, dict)
        },
        "config": regime.get("config", {}),
    }

    bifurcation = outputs.get("bifurcation", {})
    compact["bifurcation"] = {
        "rows": bifurcation.get("rows", []),
        "summary": bifurcation.get("summary", {}),
        "hysteresis_curve": bifurcation.get("hysteresis_curve", []),
        "config": bifurcation.get("config", {}),
    }

    separation = outputs.get("separation", {})
    compact["separation"] = {
        "metrics": compact_metric_dict(separation.get("metrics", {})),
        "n_input_neurons": count_or_list(separation.get("input_indices", [])),
        "n_state_neurons": count_or_list(separation.get("state_indices", [])),
        "config": separation.get("config", {}),
    }

    memory = outputs.get("memory", {})
    compact["memory"] = {
        "metrics": compact_metric_dict(
            memory.get("metrics", {}),
            keep_arrays={"delays_bins", "delays_ms", "r2_test_by_delay", "r2_capacity_by_delay"},
        ),
        "n_input_neurons": count_or_list(memory.get("input_indices", [])),
        "n_state_neurons": count_or_list(memory.get("state_indices", [])),
        "mean_state_rate_Hz": memory.get("mean_state_rate_Hz"),
        "config": memory.get("config", {}),
    }

    generalization = outputs.get("generalization")
    if generalization is not None:
        compact["generalization"] = {
            "metrics": compact_metric_dict(
                generalization.get("metrics", {}),
                keep_arrays={
                    "confusion_matrix",
                    "per_class_accuracy",
                    "class_counts",
                    "classes",
                    "fisher_eigenvalues",
                    "within_scatter_trace_by_class",
                },
            ),
            "n_input_neurons": count_or_list(generalization.get("input_indices", [])),
            "n_state_neurons": count_or_list(generalization.get("state_indices", [])),
            "trial_mean_rates_Hz": make_json_safe(generalization.get("trial_mean_rates_Hz", [])),
            "config": generalization.get("config", {}),
        }

    return compact


def base_row(condition: Condition, normalize_target_in_E: dict[str, float], normalize_target_in_I: dict[str, float]) -> dict[str, Any]:
    row = asdict(condition)
    row["condition_key"] = condition.key
    row["normalize_E_ss4"] = normalize_target_in_E["ss4"]
    row["normalize_E_b"] = normalize_target_in_E["b"]
    row["normalize_I_ss4"] = normalize_target_in_I["ss4"]
    row["normalize_I_b"] = normalize_target_in_I["b"]
    return row


def condition_heterogeneity(condition: Condition, *, truncate_std: float) -> dict[str, Any]:
    if condition.sweep == "threshold-heterogeneity":
        return threshold_heterogeneity(
            variance_E=float(condition.x_value),
            variance_I=float(condition.y_value),
            truncate_std=float(truncate_std),
        )
    return threshold_heterogeneity(truncate_std=float(truncate_std))


def run_condition(
    condition: Condition,
    *,
    n_neurons: int,
    dt_ms: float,
    input_amplitudes: list[float],
    delay_std_fraction: float,
    threshold_truncate_std: float,
    show_progress: bool,
    per_condition_dir: Path,
) -> dict[str, Any]:
    started = time.time()

    def log_stage(stage: str, stage_started: float) -> float:
        if show_progress:
            print(f"  {stage} finished in {time.time() - stage_started:.1f} s", flush=True)
        return time.time()

    stage_started = time.time()
    graph, normalize_target_in_E, normalize_target_in_I, graph_config = build_graph(
        condition,
        n_neurons,
        delay_std_fraction=delay_std_fraction,
    )
    stage_started = log_stage("Graph build", stage_started)
    label = normalization_label(condition.topology, normalize_target_in_E, normalize_target_in_I)
    topology_type = "spatial" if condition.topology == "spatial" else "nonspatial"
    heterogeneity = condition_heterogeneity(condition, truncate_std=threshold_truncate_std)

    metadata = base_row(condition, normalize_target_in_E, normalize_target_in_I)
    metadata["normalization_condition"] = label
    metadata["n_neurons"] = int(graph.number_of_nodes())
    metadata["dt_ms"] = float(dt_ms)
    metadata["n_edges"] = int(graph.number_of_edges())
    metadata["delay_std_fraction"] = float(delay_std_fraction)
    metadata["threshold_truncate_std"] = float(threshold_truncate_std)
    metadata.update(flatten_scalars("graph.", graph_config))
    metadata["threshold_variance_E"] = (
        float(condition.x_value) if condition.sweep == "threshold-heterogeneity" else DEFAULT_THRESHOLD_VARIANCE_E
    )
    metadata["threshold_variance_I"] = (
        float(condition.y_value) if condition.sweep == "threshold-heterogeneity" else DEFAULT_THRESHOLD_VARIANCE_I
    )

    outputs: dict[str, Any] = {
        "condition": metadata,
        "settings": NOTEBOOK_ANALYSIS_SETTINGS,
    }

    sim = build_simulation(graph, dt_ms=dt_ms, seed=condition.seed, heterogeneity=heterogeneity)
    regime_settings = dict(NOTEBOOK_ANALYSIS_SETTINGS["regime"])
    regime = sim.analyze_regime_persistence_regeneration(
        **regime_settings,
        topology_type=topology_type,
        seed=condition.seed,
        show_progress=show_progress,
    )
    outputs["regime"] = regime
    stage_started = log_stage("Regime analysis", stage_started)

    sim = build_simulation(graph, dt_ms=dt_ms, seed=condition.seed, heterogeneity=heterogeneity)
    bifurcation_settings = dict(NOTEBOOK_ANALYSIS_SETTINGS["bifurcation"])
    bifurcation = sim.analyze_input_amplitude_bifurcation(
        input_amplitudes=input_amplitudes,
        **bifurcation_settings,
        metadata={
            **metadata,
            "topology_type": topology_type,
        },
        seed=condition.seed,
        show_progress=show_progress,
    )
    outputs["bifurcation"] = bifurcation
    stage_started = log_stage("Bifurcation analysis", stage_started)

    sim = build_simulation(graph, dt_ms=dt_ms, seed=condition.seed, heterogeneity=heterogeneity)
    separation_settings = dict(NOTEBOOK_ANALYSIS_SETTINGS["separation"])
    separation = sim.analyze_separation_property(
        **separation_settings,
        seed=condition.seed,
        show_progress=show_progress,
    )
    outputs["separation"] = separation
    stage_started = log_stage("Separation analysis", stage_started)

    sim = build_simulation(graph, dt_ms=dt_ms, seed=condition.seed, heterogeneity=heterogeneity)
    generalization_settings = dict(NOTEBOOK_ANALYSIS_SETTINGS["generalization"])
    generalization = sim.analyze_generalization_property(
        **generalization_settings,
        seed=condition.seed,
        show_progress=show_progress,
    )
    outputs["generalization"] = generalization
    stage_started = log_stage("Generalization analysis", stage_started)

    sim = build_simulation(graph, dt_ms=dt_ms, seed=condition.seed, heterogeneity=heterogeneity)
    memory_settings = dict(NOTEBOOK_ANALYSIS_SETTINGS["memory"])
    memory = sim.analyze_memory_capacity(
        **memory_settings,
        seed=condition.seed,
        show_progress=show_progress,
    )
    outputs["memory"] = memory
    stage_started = log_stage("Memory analysis", stage_started)

    metadata["elapsed_s"] = time.time() - started
    metadata["status"] = "ok"

    per_condition_dir.mkdir(parents=True, exist_ok=True)
    condition_json = per_condition_dir / condition_json_filename(condition)
    metadata["condition_json"] = str(condition_json)
    compact_outputs = compact_outputs_for_json(outputs)
    with condition_json.open("w", encoding="utf-8") as handle:
        json.dump(make_json_safe(compact_outputs), handle, separators=(",", ":"))
    log_stage("Condition JSON write", stage_started)

    return outputs


def make_rows(result: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    condition = result["condition"]

    regime_row = dict(condition)
    regime = result["regime"]
    regime_row["classification"] = regime.get("classification")
    regime_row.update(flatten_scalars("brunel.", regime.get("brunel", {})))
    regime_row.update(flatten_scalars("persistence.", regime.get("persistence", {})))
    regime_row.update(flatten_scalars("regeneration.", regime.get("regeneration", {})))
    regime_row.update(flatten_scalars("observation.", regime.get("metrics", {}).get("observation", {})))
    regime_row.update(flatten_scalars("transient.", regime.get("metrics", {}).get("transient", {})))

    separation_row = dict(condition)
    separation_row.update(flatten_scalars("", result["separation"].get("metrics", {})))
    separation_row["n_input_neurons"] = index_count(result["separation"], "input_indices", "n_input_neurons")
    separation_row["n_state_neurons"] = index_count(result["separation"], "state_indices", "n_state_neurons")

    memory_row = dict(condition)
    memory_metrics = result["memory"].get("metrics", {})
    memory_row.update(flatten_scalars("", memory_metrics))
    for key in ("delays_bins", "delays_ms", "r2_test_by_delay", "r2_capacity_by_delay"):
        if key in memory_metrics:
            memory_row[key] = json.dumps(make_json_safe(memory_metrics[key]))
    memory_row["n_input_neurons"] = index_count(result["memory"], "input_indices", "n_input_neurons")
    memory_row["n_state_neurons"] = index_count(result["memory"], "state_indices", "n_state_neurons")

    generalization_rows = []
    generalization = result.get("generalization")
    if generalization is not None:
        generalization_row = dict(condition)
        generalization_metrics = generalization.get("metrics", {})
        generalization_row.update(flatten_scalars("", generalization_metrics))
        for key in (
            "confusion_matrix",
            "per_class_accuracy",
            "class_counts",
            "classes",
            "fisher_eigenvalues",
            "within_scatter_trace_by_class",
        ):
            if key in generalization_metrics:
                generalization_row[key] = json.dumps(make_json_safe(generalization_metrics[key]))
        generalization_row["n_input_neurons"] = index_count(
            generalization,
            "input_indices",
            "n_input_neurons",
        )
        generalization_row["n_state_neurons"] = index_count(
            generalization,
            "state_indices",
            "n_state_neurons",
        )
        trial_rates = generalization.get("trial_mean_rates_Hz")
        if trial_rates is not None:
            trial_rates_arr = np.asarray(trial_rates, dtype=float)
            if trial_rates_arr.size:
                generalization_row["trial_mean_rate_Hz_mean"] = float(np.nanmean(trial_rates_arr))
                generalization_row["trial_mean_rate_Hz_std"] = float(np.nanstd(trial_rates_arr))
                generalization_row["trial_mean_rate_Hz_min"] = float(np.nanmin(trial_rates_arr))
                generalization_row["trial_mean_rate_Hz_max"] = float(np.nanmax(trial_rates_arr))
        generalization_rows.append(generalization_row)

    bifurcation_rows = []
    for row in result["bifurcation"].get("rows", []):
        merged = dict(condition)
        merged.update(row)
        bifurcation_rows.append(merged)

    bifurcation_summary_row = dict(condition)
    bifurcation_summary_row.update(flatten_scalars("", result["bifurcation"].get("summary", {})))

    hysteresis_rows = []
    for row in result["bifurcation"].get("hysteresis_curve", []):
        merged = dict(condition)
        merged.update(row)
        hysteresis_rows.append(merged)

    return {
        "conditions": [dict(condition)],
        "regime_rows": [regime_row],
        "separation_summary": [separation_row],
        "memory_summary": [memory_row],
        "generalization_summary": generalization_rows,
        "bifurcation_rows": bifurcation_rows,
        "bifurcation_summary": [bifurcation_summary_row],
        "bifurcation_hysteresis": hysteresis_rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_per_condition_json(output_dir: Path, suffix: str = "all") -> None:
    rows_by_table: dict[str, list[dict[str, Any]]] = {
        "conditions": [],
        "regime_rows": [],
        "separation_summary": [],
        "memory_summary": [],
        "generalization_summary": [],
        "bifurcation_rows": [],
        "bifurcation_summary": [],
        "bifurcation_hysteresis": [],
    }

    paths = sorted((output_dir / "per_condition").glob("condition_*.json"))
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        table_rows = make_rows(result)
        for table_name, rows in table_rows.items():
            rows_by_table[table_name].extend(rows)

    for table_name, rows in rows_by_table.items():
        write_csv(output_dir / f"{table_name}_{suffix}.csv", rows)

    print(f"Aggregated {len(paths)} condition files into {output_dir}")


def read_completed_keys(output_dir: Path) -> set[str]:
    keys = set()
    for path in (output_dir / "per_condition").glob("condition_*.json"):
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            key = data.get("condition", {}).get("condition_key")
            status = data.get("condition", {}).get("status")
            if key and status == "ok":
                keys.add(str(key))
        except Exception:
            continue
    return keys


def build_conditions(
    *,
    sweep: str,
    topologies: list[str],
    x_values: list[float],
    y_values: list[float],
    seeds: list[int],
    topology_value_grids: dict[str, tuple[list[float], list[float]]] | None = None,
) -> list[Condition]:
    if sweep not in SWEEP_AXIS_NAMES:
        raise ValueError(f"Unknown sweep: {sweep}")
    x_name, y_name = SWEEP_AXIS_NAMES[sweep]
    conditions = []
    condition_id = 0
    for topology in topologies:
        topo_x_values, topo_y_values = x_values, y_values
        if topology_value_grids is not None and topology in topology_value_grids:
            topo_x_values, topo_y_values = topology_value_grids[topology]
        if not topo_x_values or not topo_y_values:
            raise ValueError(f"No sweep values were provided for topology '{topology}'.")

        for x_value in topo_x_values:
            for y_value in topo_y_values:
                for seed in seeds:
                    j_gain = 1.0
                    g_gain = 1.0
                    n_neurons = None
                    inhibitory_fraction = None
                    if sweep in ("normalization", "b-target-normalization"):
                        j_gain = float(x_value)
                        g_gain = float(y_value)
                    elif sweep == "size-inhibition":
                        n_neurons = int(round(float(x_value)))
                        inhibitory_fraction = float(y_value)
                        if n_neurons <= 1:
                            raise ValueError("n_neurons values in size-inhibition sweep must be > 1.")
                        resolved_type_fractions(inhibitory_fraction)

                    conditions.append(
                        Condition(
                            condition_id=condition_id,
                            sweep=sweep,
                            topology=topology,
                            x_name=x_name,
                            x_value=float(x_value),
                            y_name=y_name,
                            y_value=float(y_value),
                            j_gain=float(x_value),
                            g_gain=float(y_value),
                            seed=int(seed),
                            n_neurons=n_neurons,
                            inhibitory_fraction=inhibitory_fraction,
                        )
                    )
                    condition_id += 1
    return conditions


def resolve_delay_topology_value_grids(
    *,
    topologies: list[str],
    generic_x_values: list[float],
    generic_y_values: list[float],
    fixed_e_delay_values: list[float],
    fixed_i_delay_values: list[float],
    spatial_e_lambda_values: list[float],
    spatial_i_lambda_values: list[float],
) -> dict[str, tuple[list[float], list[float]]]:
    grids: dict[str, tuple[list[float], list[float]]] = {}
    for topology in topologies:
        if topology == "fixed":
            x_values = fixed_e_delay_values or generic_x_values
            y_values = fixed_i_delay_values or generic_y_values
            if not x_values or not y_values:
                raise ValueError(
                    "Delay sweep for fixed topology needs --fixed-e-delay-values and "
                    "--fixed-i-delay-values, or generic --x-values/--y-values."
                )
            grids[topology] = (x_values, y_values)
        elif topology == "spatial":
            x_values = spatial_e_lambda_values or generic_x_values
            y_values = spatial_i_lambda_values or generic_y_values
            if not x_values or not y_values:
                raise ValueError(
                    "Delay sweep for spatial topology needs --spatial-e-lambda-values and "
                    "--spatial-i-lambda-values, or generic --x-values/--y-values."
                )
            grids[topology] = (x_values, y_values)
        else:
            raise ValueError(f"Unknown topology: {topology}")
    return grids


def main() -> int:
    raw_argv = sys.argv[1:]
    provided_flags = {arg.split("=", 1)[0] for arg in raw_argv if arg.startswith("--")}

    def flag_provided(name: str) -> bool:
        return name in provided_flags

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", choices=SWEEP_CHOICES, default="normalization")
    parser.add_argument(
        "--x-values",
        nargs="+",
        help=(
            "First sweep-axis values. Meaning depends on --sweep: J, E delay/lambda, "
            "E threshold variance, or n_neurons."
        ),
    )
    parser.add_argument(
        "--y-values",
        nargs="+",
        help=(
            "Second sweep-axis values. Meaning depends on --sweep: g, I delay/lambda, "
            "I threshold variance, or inhibitory fraction."
        ),
    )
    parser.add_argument("--j-values", nargs="+", help="Alias for --x-values in normalization sweeps.")
    parser.add_argument("--g-values", nargs="+", help="Alias for --y-values in normalization sweeps.")
    parser.add_argument(
        "--augment-existing",
        action="store_true",
        help=(
            "Read previous grid_config_job*.json files in --output-dir, merge in "
            "--add-* values, and skip already-completed conditions."
        ),
    )
    parser.add_argument("--add-x-values", nargs="+", help="Additional first-axis values for --augment-existing.")
    parser.add_argument("--add-y-values", nargs="+", help="Additional second-axis values for --augment-existing.")
    parser.add_argument("--add-j-values", nargs="+", help="Alias for --add-x-values in normalization sweeps.")
    parser.add_argument("--add-g-values", nargs="+", help="Alias for --add-y-values in normalization sweeps.")
    parser.add_argument(
        "--fixed-e-delay-values",
        nargs="+",
        help="Fixed-topology excitatory delay means in ms for --sweep delay.",
    )
    parser.add_argument(
        "--fixed-i-delay-values",
        nargs="+",
        help="Fixed-topology inhibitory delay means in ms for --sweep delay.",
    )
    parser.add_argument(
        "--spatial-e-lambda-values",
        nargs="+",
        help="Spatial-topology excitatory lambda_by_preclass values for --sweep delay.",
    )
    parser.add_argument(
        "--spatial-i-lambda-values",
        nargs="+",
        help="Spatial-topology inhibitory lambda_by_preclass values for --sweep delay.",
    )
    parser.add_argument("--topologies", nargs="+", choices=["fixed", "spatial"], default=["fixed", "spatial"])
    parser.add_argument("--input-amplitudes", nargs="+", default=["0.0", "0.2", "1.0", "5.0", "50.0"])
    parser.add_argument("--output-dir", default=str(ROOT / "FinalPaper" / "results" / "normalization_grid"))
    parser.add_argument("--n-neurons", type=int, default=1000)
    parser.add_argument("--dt-ms", type=float, default=0.1)
    parser.add_argument(
        "--delay-std-fraction",
        type=float,
        default=0.3,
        help="Fixed-topology delay std as a fraction of the delay mean for --sweep delay.",
    )
    parser.add_argument(
        "--threshold-truncate-std",
        type=float,
        default=DEFAULT_THRESHOLD_TRUNCATE_STD,
        help="Gaussian truncation in standard deviations for threshold heterogeneity.",
    )
    parser.add_argument("--seed-start", type=int, default=1234)
    parser.add_argument("--seed-count", type=int, default=1)
    parser.add_argument("--job-index", type=int, default=0)
    parser.add_argument("--num-jobs", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only rebuild aggregate CSV files from per_condition/*.json in --output-dir.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.aggregate_only:
        aggregate_per_condition_json(output_dir)
        return 0

    existing_configs: list[dict[str, Any]] = []
    if args.augment_existing:
        existing_configs = read_grid_configs(output_dir)
        if not existing_configs:
            raise FileNotFoundError(
                f"--augment-existing was requested, but no grid_config_job*.json files were found in {output_dir}"
            )
        if not flag_provided("--sweep"):
            args.sweep = str(config_first_value(existing_configs, "sweep", args.sweep))
        if not flag_provided("--topologies"):
            prior_topologies = config_str_values(existing_configs, "topologies")
            if prior_topologies:
                args.topologies = prior_topologies
        if not flag_provided("--n-neurons"):
            args.n_neurons = int(config_first_value(existing_configs, "n_neurons", args.n_neurons))
        if not flag_provided("--dt-ms"):
            args.dt_ms = float(config_first_value(existing_configs, "dt_ms", args.dt_ms))
        if not flag_provided("--delay-std-fraction"):
            args.delay_std_fraction = float(
                config_first_value(existing_configs, "delay_std_fraction", args.delay_std_fraction)
            )
        if not flag_provided("--threshold-truncate-std"):
            args.threshold_truncate_std = float(
                config_first_value(existing_configs, "threshold_truncate_std", args.threshold_truncate_std)
            )

    x_values = parse_float_list(args.x_values)
    y_values = parse_float_list(args.y_values)
    if not x_values:
        x_values = parse_float_list(args.j_values)
    if not y_values:
        y_values = parse_float_list(args.g_values)
    add_x_values = parse_float_list(args.add_x_values)
    add_y_values = parse_float_list(args.add_y_values)
    if not add_x_values:
        add_x_values = parse_float_list(args.add_j_values)
    if not add_y_values:
        add_y_values = parse_float_list(args.add_g_values)
    fixed_e_delay_values = parse_float_list(args.fixed_e_delay_values)
    fixed_i_delay_values = parse_float_list(args.fixed_i_delay_values)
    spatial_e_lambda_values = parse_float_list(args.spatial_e_lambda_values)
    spatial_i_lambda_values = parse_float_list(args.spatial_i_lambda_values)
    input_amplitudes = parse_float_list(args.input_amplitudes)

    if args.augment_existing:
        prior_x_values = config_float_values(existing_configs, "x_values") or config_float_values(
            existing_configs, "j_values"
        )
        prior_y_values = config_float_values(existing_configs, "y_values") or config_float_values(
            existing_configs, "g_values"
        )
        x_values = merge_float_values(prior_x_values, x_values, add_x_values)
        y_values = merge_float_values(prior_y_values, y_values, add_y_values)
        if not flag_provided("--input-amplitudes"):
            prior_input_amplitudes = config_float_values(existing_configs, "input_amplitudes")
            if prior_input_amplitudes:
                input_amplitudes = prior_input_amplitudes
        if not flag_provided("--fixed-e-delay-values"):
            fixed_e_delay_values = config_float_values(existing_configs, "fixed_e_delay_values")
        if not flag_provided("--fixed-i-delay-values"):
            fixed_i_delay_values = config_float_values(existing_configs, "fixed_i_delay_values")
        if not flag_provided("--spatial-e-lambda-values"):
            spatial_e_lambda_values = config_float_values(existing_configs, "spatial_e_lambda_values")
        if not flag_provided("--spatial-i-lambda-values"):
            spatial_i_lambda_values = config_float_values(existing_configs, "spatial_i_lambda_values")

    topology_value_grids = None
    if args.sweep == "delay":
        topology_value_grids = resolve_delay_topology_value_grids(
            topologies=args.topologies,
            generic_x_values=x_values,
            generic_y_values=y_values,
            fixed_e_delay_values=fixed_e_delay_values,
            fixed_i_delay_values=fixed_i_delay_values,
            spatial_e_lambda_values=spatial_e_lambda_values,
            spatial_i_lambda_values=spatial_i_lambda_values,
        )
        # Keep x/y config fields useful even when topology-specific grids are used.
        if not x_values:
            x_values = sorted({value for values, _ in topology_value_grids.values() for value in values})
        if not y_values:
            y_values = sorted({value for _, values in topology_value_grids.values() for value in values})

    if args.sweep != "delay" and (not x_values or not y_values):
        raise ValueError(
            "Both sweep axes must contain at least one value. Use --x-values/--y-values, "
            "or --j-values/--g-values for normalization sweeps."
        )
    if not input_amplitudes:
        raise ValueError("--input-amplitudes must contain at least one value.")
    if args.delay_std_fraction < 0:
        raise ValueError("--delay-std-fraction must be >= 0.")
    if args.threshold_truncate_std <= 0:
        raise ValueError("--threshold-truncate-std must be > 0.")
    if args.num_jobs < 1:
        raise ValueError("--num-jobs must be >= 1.")
    if not (0 <= args.job_index < args.num_jobs):
        raise ValueError("--job-index must satisfy 0 <= job-index < num-jobs.")

    if args.augment_existing and not flag_provided("--seed-start") and not flag_provided("--seed-count"):
        seeds = config_int_values(existing_configs, "seeds")
        if not seeds:
            seeds = [args.seed_start + i for i in range(args.seed_count)]
    else:
        seeds = [args.seed_start + i for i in range(args.seed_count)]
    all_conditions = build_conditions(
        sweep=args.sweep,
        topologies=args.topologies,
        x_values=x_values,
        y_values=y_values,
        seeds=seeds,
        topology_value_grids=topology_value_grids,
    )
    assigned_job_conditions = [
        condition
        for idx, condition in enumerate(all_conditions)
        if idx % args.num_jobs == args.job_index
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    per_condition_dir = output_dir / "per_condition"
    completed = set() if args.overwrite else read_completed_keys(output_dir)
    if args.overwrite:
        job_conditions = assigned_job_conditions
        global_pending_count = len(all_conditions)
    else:
        job_conditions = [condition for condition in assigned_job_conditions if condition.key not in completed]
        global_pending_count = sum(1 for condition in all_conditions if condition.key not in completed)
    if args.augment_existing and not args.overwrite:
        print(
            f"Augment mode: loaded {len(existing_configs)} previous grid config file(s); "
            f"{global_pending_count} total expanded-grid conditions are not completed yet; "
            f"this job has {len(job_conditions)} pending of {len(assigned_job_conditions)} assigned conditions.",
            flush=True,
        )

    config = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(ROOT),
        "augment_existing": bool(args.augment_existing),
        "add_x_values": add_x_values,
        "add_y_values": add_y_values,
        "sweep": args.sweep,
        "x_name": SWEEP_AXIS_NAMES[args.sweep][0],
        "y_name": SWEEP_AXIS_NAMES[args.sweep][1],
        "x_values": x_values,
        "y_values": y_values,
        "j_values": x_values,
        "g_values": y_values,
        "topology_value_grids": topology_value_grids,
        "fixed_e_delay_values": fixed_e_delay_values,
        "fixed_i_delay_values": fixed_i_delay_values,
        "spatial_e_lambda_values": spatial_e_lambda_values,
        "spatial_i_lambda_values": spatial_i_lambda_values,
        "topologies": args.topologies,
        "input_amplitudes": input_amplitudes,
        "n_neurons": args.n_neurons,
        "dt_ms": args.dt_ms,
        "delay_std_fraction": args.delay_std_fraction,
        "threshold_truncate_std": args.threshold_truncate_std,
        "seeds": seeds,
        "job_index": args.job_index,
        "num_jobs": args.num_jobs,
        "base_normalization": TOPOLOGY_BASE_NORMALIZATION,
        "analysis_settings": NOTEBOOK_ANALYSIS_SETTINGS,
    }
    with (output_dir / f"grid_config_job{args.job_index:03d}.json").open("w", encoding="utf-8") as handle:
        json.dump(make_json_safe(config), handle, indent=2)

    rows_by_table: dict[str, list[dict[str, Any]]] = {
        "conditions": [],
        "regime_rows": [],
        "separation_summary": [],
        "memory_summary": [],
        "generalization_summary": [],
        "bifurcation_rows": [],
        "bifurcation_summary": [],
        "bifurcation_hysteresis": [],
        "errors": [],
    }

    print(
        f"Running job {args.job_index}/{args.num_jobs}: "
        f"{len(job_conditions)} pending conditions "
        f"({len(assigned_job_conditions)} assigned of {len(all_conditions)} total expanded-grid conditions)"
    )
    print(f"Output directory: {output_dir}")

    for local_idx, condition in enumerate(job_conditions, start=1):
        print(f"[{local_idx}/{len(job_conditions)}] Running {condition.key}")
        started = time.time()
        try:
            result = run_condition(
                condition,
                n_neurons=args.n_neurons,
                dt_ms=args.dt_ms,
                input_amplitudes=input_amplitudes,
                delay_std_fraction=args.delay_std_fraction,
                threshold_truncate_std=args.threshold_truncate_std,
                show_progress=not args.no_progress,
                per_condition_dir=per_condition_dir,
            )
            row_started = time.time()
            table_rows = make_rows(result)
            for table_name, rows in table_rows.items():
                rows_by_table[table_name].extend(rows)
            if not args.no_progress:
                print(f"  Row extraction finished in {time.time() - row_started:.1f} s", flush=True)
            print(f"Finished {condition.key} in {time.time() - started:.1f} s")
        except Exception as exc:
            error_row = asdict(condition)
            error_row["condition_key"] = condition.key
            error_row["status"] = "error"
            error_row["error_type"] = type(exc).__name__
            error_row["error"] = str(exc)
            error_row["traceback"] = traceback.format_exc()
            rows_by_table["errors"].append(error_row)
            print(f"ERROR in {condition.key}: {exc}")

    suffix = f"job{args.job_index:03d}_of_{args.num_jobs:03d}"
    for table_name, rows in rows_by_table.items():
        write_started = time.time()
        write_csv(output_dir / f"{table_name}_{suffix}.csv", rows)
        if rows and not args.no_progress:
            print(
                f"Wrote {table_name}_{suffix}.csv "
                f"({len(rows)} rows) in {time.time() - write_started:.1f} s",
                flush=True,
            )

    if args.num_jobs == 1:
        aggregate_per_condition_json(output_dir)

    print("Done.")
    print("Per-job CSV files can be concatenated by table name for cross-job analysis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
