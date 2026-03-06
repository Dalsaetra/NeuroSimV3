import argparse
import csv
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.connectome import Connectome
from src.external_inputs import PoissonInput
from src.network_generators import generate_spatial_ei_network
from src.neuron_population import NeuronPopulation
from src.overhead import Simulation

try:
    import cma
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Failed to import cma. Activate the environment and install 'cma'.") from exc


_WORKER_OBJECTIVE = None


DIM_ORDER = [
    "ss4_fraction",
    "p0_EE",
    "p0_EI",
    "p0_IE",
    "p0_II",
    "lambda_E",
    "lambda_I",
    "normalize_target_out_E",
    "normalize_target_out_I",
    "weight_dist_ss4",
    "weight_dist_b",
    "distance_scale",
    "outdist_ss4",
    "outparam1_ss4",
    "outparam2_ss4",
    "outdist_b",
    "outparam1_b",
    "outparam2_b",
    "mu_E",
    "sigma_E",
    "mu_I",
    "sigma_I",
]


@dataclass
class SearchConfig:
    n_neurons: int = 150
    dt_ms: float = 0.1
    total_ms: float = 400.0
    total_ms_jitter: float = 0.0
    t_start_ms: float = 150.0
    p_inputs: int = 12
    input_fraction_exc: float = 0.20
    input_rate_min_hz: float = 30.0
    input_rate_max_hz: float = 100.0
    input_amp_min: float = 0.5
    input_amp_max: float = 20.0
    high_rate_threshold_hz: float = 60.0
    high_rate_penalty_weight: float = 0.1
    excitatory_type: str = "ss4"
    inhibitory_type: str = "b"
    normalize_mode: str = "out"
    stepper_type: str = "euler_det"
    synapse_type: str = "standard"
    lt_scale: float = 1.0
    inhibitory_nmda_weight: float = 0.959685703507305 * 0.5
    base_seed: int = 1234


def _unit_to_linear(z: float, low: float, high: float) -> float:
    zz = float(np.clip(z, 0.0, 1.0))
    return float(low + zz * (high - low))


def _unit_to_log(z: float, low: float, high: float) -> float:
    zz = float(np.clip(z, 0.0, 1.0))
    lo = math.log(float(low))
    hi = math.log(float(high))
    return float(math.exp(lo + zz * (hi - lo)))


def _effective_rank(matrix: np.ndarray) -> float:
    if matrix.size == 0:
        return 0.0
    svals = np.linalg.svd(matrix, compute_uv=False, full_matrices=False)
    svals = svals[svals > 1e-12]
    if svals.size == 0:
        return 0.0
    probs = svals / np.sum(svals)
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    return float(np.exp(entropy))


def _decode_outdegree_config(dist_z: float, p1_z: float, p2_z: float) -> Dict[str, object]:
    if dist_z >= 0.5:
        return {
            "dist": "lognormal",
            "params": (
                _unit_to_linear(p1_z, 0.0, 3.2),
                _unit_to_linear(p2_z, 0.2, 1.2),
            ),
        }
    return {
        "dist": "neg-bin",
        "params": (
            _unit_to_linear(p1_z, 5.0, 120.0),
            _unit_to_linear(p2_z, 3.0, 80.0),
        ),
    }


def _decode_weight_config(mu_z: float, sigma_z: float, dist_name: str, *, inhibitory: bool) -> Tuple[float, float]:
    dist = str(dist_name).lower()
    if dist == "normal":
        if inhibitory:
            mu = _unit_to_linear(mu_z, 5.0, 80.0)
            sigma = _unit_to_linear(sigma_z, 1.0, 50.0)
        else:
            mu = _unit_to_linear(mu_z, 0.0, 100.0)
            sigma = _unit_to_linear(sigma_z, 0.5, 50.0)
        return float(mu), float(sigma)

    # lognormal (default)
    if inhibitory:
        mu = _unit_to_linear(mu_z, -2.0, 4.0)
        sigma = _unit_to_linear(sigma_z, 0.2, 2.5)
    else:
        mu = _unit_to_linear(mu_z, -2.0, 4.0)
        sigma = _unit_to_linear(sigma_z, 0.2, 2.5)
    return float(mu), float(sigma)


def _decode_unit_vector(x: np.ndarray) -> Dict[str, object]:
    if x.shape[0] != len(DIM_ORDER):
        raise ValueError(f"Expected {len(DIM_ORDER)} dimensions, got {x.shape[0]}.")

    z = {name: float(np.clip(x[i], 0.0, 1.0)) for i, name in enumerate(DIM_ORDER)}

    ss4_fraction = _unit_to_linear(z["ss4_fraction"], 0.70, 0.90)
    type_fractions = {
        "ss4": float(ss4_fraction),
        "b": float(1.0 - ss4_fraction),
    }

    p0_by_pair = {
        "EE": _unit_to_linear(z["p0_EE"], 0.10, 0.90),
        "EI": _unit_to_linear(z["p0_EI"], 0.10, 0.90),
        "IE": _unit_to_linear(z["p0_IE"], 0.10, 0.90),
        "II": _unit_to_linear(z["p0_II"], 0.10, 0.90),
    }
    lambda_by_preclass = {
        "E": _unit_to_log(z["lambda_E"], 0.05, 0.50),
        "I": _unit_to_log(z["lambda_I"], 0.01, 0.50),
    }

    normalize_target_out_e = _unit_to_log(z["normalize_target_out_E"], 2.0, 100.0)
    normalize_target_out_i = _unit_to_log(z["normalize_target_out_I"], 2.0, 1000.0)

    weight_dist_by_ntype = {
        "ss4": "lognormal" if z["weight_dist_ss4"] >= 0.5 else "normal",
        "b": "lognormal" if z["weight_dist_b"] >= 0.5 else "normal",
    }

    outdegree_config_by_type = {
        "ss4": _decode_outdegree_config(z["outdist_ss4"], z["outparam1_ss4"], z["outparam2_ss4"]),
        "b": _decode_outdegree_config(z["outdist_b"], z["outparam1_b"], z["outparam2_b"]),
    }

    mu_e, sigma_e = _decode_weight_config(
        z["mu_E"],
        z["sigma_E"],
        weight_dist_by_ntype["ss4"],
        inhibitory=False,
    )
    mu_i, sigma_i = _decode_weight_config(
        z["mu_I"],
        z["sigma_I"],
        weight_dist_by_ntype["b"],
        inhibitory=True,
    )

    generator_kwargs = {
        "n_neurons": None,  # filled by objective cfg
        "type_fractions": type_fractions,
        "inhibitory_types": ("b",),
        "mu_E": float(mu_e),
        "sigma_E": float(sigma_e),
        "mu_I": float(mu_i),
        "sigma_I": float(sigma_i),
        "p0_by_pair": p0_by_pair,
        "lambda_by_preclass": lambda_by_preclass,
        "distance_scale": _unit_to_linear(z["distance_scale"], 5.0, 40.0),
        "weight_dist_by_ntype": weight_dist_by_ntype,
        "outdegree_config_by_type": outdegree_config_by_type,
        "normalize_mode": "out",
        "normalize_target_out_E": float(normalize_target_out_e),
        "normalize_target_out_I": float(normalize_target_out_i),
    }

    params_flat = {
        "type_fraction_ss4": float(type_fractions["ss4"]),
        "type_fraction_b": float(type_fractions["b"]),
        "p0_EE": float(p0_by_pair["EE"]),
        "p0_EI": float(p0_by_pair["EI"]),
        "p0_IE": float(p0_by_pair["IE"]),
        "p0_II": float(p0_by_pair["II"]),
        "lambda_E": float(lambda_by_preclass["E"]),
        "lambda_I": float(lambda_by_preclass["I"]),
        "normalize_target_out_E": float(normalize_target_out_e),
        "normalize_target_out_I": float(normalize_target_out_i),
        "weight_dist_ss4": weight_dist_by_ntype["ss4"],
        "weight_dist_b": weight_dist_by_ntype["b"],
        "distance_scale": float(generator_kwargs["distance_scale"]),
        "outdeg_ss4_dist": str(outdegree_config_by_type["ss4"]["dist"]),
        "outdeg_ss4_param1": float(outdegree_config_by_type["ss4"]["params"][0]),
        "outdeg_ss4_param2": float(outdegree_config_by_type["ss4"]["params"][1]),
        "outdeg_b_dist": str(outdegree_config_by_type["b"]["dist"]),
        "outdeg_b_param1": float(outdegree_config_by_type["b"]["params"][0]),
        "outdeg_b_param2": float(outdegree_config_by_type["b"]["params"][1]),
        "mu_E": float(mu_e),
        "sigma_E": float(sigma_e),
        "mu_I": float(mu_i),
        "sigma_I": float(sigma_i),
    }

    return {
        "generator_kwargs": generator_kwargs,
        "params_flat": params_flat,
    }


def _default_unit_vector() -> np.ndarray:
    x0 = np.full(len(DIM_ORDER), 0.5, dtype=float)
    idx = {name: i for i, name in enumerate(DIM_ORDER)}
    x0[idx["weight_dist_ss4"]] = 0.9
    x0[idx["weight_dist_b"]] = 0.1
    x0[idx["distance_scale"]] = 0.43
    x0[idx["outdist_ss4"]] = 0.9
    x0[idx["outparam1_ss4"]] = 0.54
    x0[idx["outparam2_ss4"]] = 0.56
    x0[idx["outdist_b"]] = 0.1
    x0[idx["outparam1_b"]] = 0.30
    x0[idx["outparam2_b"]] = 0.43
    x0[idx["mu_I"]] = 0.67
    x0[idx["sigma_I"]] = 0.56
    x0[idx["normalize_target_out_I"]] = 0.52
    x0[idx["normalize_target_out_E"]] = 0.51
    return x0


class SpatialEIObjective:
    def __init__(self, cfg: SearchConfig):
        self.cfg = cfg

    def _build_connectome(self, generator_kwargs: Dict[str, object], seed: int) -> Tuple[Connectome, NeuronPopulation]:
        cfg = self.cfg
        graph_kwargs = dict(generator_kwargs)
        graph_kwargs["n_neurons"] = int(cfg.n_neurons)
        graph_kwargs["seed"] = int(seed)

        graph = generate_spatial_ei_network(**graph_kwargs)

        neuron_types = [cfg.excitatory_type, cfg.inhibitory_type]
        inhibitory = [False, True]
        threshold_decay = np.exp(-cfg.dt_ms / 5.0)
        pop = NeuronPopulation(cfg.n_neurons, neuron_types, inhibitory, threshold_decay)

        max_synapses = max(dict(graph.out_degree()).values())
        connectome = Connectome(max_synapses, pop)
        connectome.nx_to_connectome(graph)
        return connectome, pop

    def _sample_state0(self, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = int(self.cfg.n_neurons)
        vs = rng.uniform(-100.0, -70.0, size=n)
        us = rng.uniform(0.0, 400.0, size=n)
        spikes = np.zeros(n, dtype=bool)
        ts = np.zeros(n, dtype=float)
        return vs, us, spikes, ts

    def _run_trial(
        self,
        connectome: Connectome,
        nmda_weight: np.ndarray,
        state0: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        output_indices: np.ndarray,
        input_group_a: np.ndarray,
        input_group_b: np.ndarray,
        rate_a: float,
        amp_a: float,
        rate_b: float,
        amp_b: float,
        total_ms: float,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, float]:
        cfg = self.cfg
        state0_local = (
            state0[0].copy(),
            state0[1].copy(),
            state0[2].copy(),
            state0[3].copy(),
        )

        sim = Simulation(
            connectome,
            cfg.dt_ms,
            stepper_type=cfg.stepper_type,
            state0=state0_local,
            enable_plasticity=False,
            synapse_type=cfg.synapse_type,
            synapse_kwargs={"LT_scale": cfg.lt_scale, "NMDA_weight": nmda_weight},
            enable_state_logger=False,
            enable_debug_logger=False,
        )

        rates = np.zeros(cfg.n_neurons, dtype=float)
        amps = np.zeros(cfg.n_neurons, dtype=float)
        rates[input_group_a] = float(rate_a)
        amps[input_group_a] = float(amp_a)
        rates[input_group_b] = float(rate_b)
        amps[input_group_b] = float(amp_b)

        poisson = PoissonInput(cfg.n_neurons, rate=rates, amplitude=amps, rng=rng)

        warmup_steps = int(round(cfg.t_start_ms / cfg.dt_ms))
        total_steps = int(round(float(total_ms) / cfg.dt_ms))
        total_steps = max(total_steps, warmup_steps + 1)
        post_steps = max(1, total_steps - warmup_steps)

        spike_counts = np.zeros(output_indices.size, dtype=float)
        for step in range(total_steps):
            sim.step(spike_ext=poisson(cfg.dt_ms))
            if step >= warmup_steps:
                spike_counts += sim.neuron_states.spike[output_indices].astype(float)

        norm = float(np.linalg.norm(spike_counts))
        if norm > 0:
            state_vec = spike_counts / norm
        else:
            state_vec = spike_counts.copy()

        mean_rate_hz = float(np.sum(spike_counts) / output_indices.size / (post_steps * cfg.dt_ms / 1000.0))
        return state_vec, mean_rate_hz

    def evaluate(self, x: np.ndarray, trial_seed: int) -> Tuple[float, Dict[str, float], Dict[str, object]]:
        cfg = self.cfg
        rng = np.random.default_rng(trial_seed)
        decoded = _decode_unit_vector(x)
        generator_kwargs = decoded["generator_kwargs"]

        connectome, pop = self._build_connectome(generator_kwargs, seed=int(rng.integers(0, 2**31 - 1)))

        nmda_weight = np.ones(cfg.n_neurons, dtype=float)
        nmda_weight[pop.inhibitory_mask.astype(bool)] = float(cfg.inhibitory_nmda_weight)

        excitatory_indices = np.flatnonzero(~pop.inhibitory_mask.astype(bool))
        if excitatory_indices.size < 2:
            raise ValueError("Need at least two excitatory neurons to build input/output sets.")
        output_indices = excitatory_indices.copy()

        n_input = int(round(cfg.input_fraction_exc * excitatory_indices.size))
        n_input = max(2, min(n_input, excitatory_indices.size))
        input_neurons = rng.choice(excitatory_indices, size=n_input, replace=False)
        input_neurons = rng.permutation(input_neurons)

        split = n_input // 2
        if split <= 0:
            split = 1
        if split >= n_input:
            split = n_input - 1
        input_group_a = input_neurons[:split]
        input_group_b = input_neurons[split:]

        state0 = self._sample_state0(rng)

        states: List[np.ndarray] = []
        trial_rates: List[float] = []
        trial_total_ms: List[float] = []
        for _ in range(int(cfg.p_inputs)):
            rate_a = _unit_to_log(rng.random(), cfg.input_rate_min_hz, cfg.input_rate_max_hz)
            amp_a = _unit_to_log(rng.random(), cfg.input_amp_min, cfg.input_amp_max)
            rate_b = _unit_to_log(rng.random(), cfg.input_rate_min_hz, cfg.input_rate_max_hz)
            amp_b = _unit_to_log(rng.random(), cfg.input_amp_min, cfg.input_amp_max)
            if cfg.total_ms_jitter > 0.0:
                total_ms_this_trial = float(cfg.total_ms + rng.uniform(-cfg.total_ms_jitter, cfg.total_ms_jitter))
            else:
                total_ms_this_trial = float(cfg.total_ms)
            total_ms_this_trial = max(float(cfg.t_start_ms + cfg.dt_ms), total_ms_this_trial)
            trial_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
            state_vec, mean_rate_hz = self._run_trial(
                connectome=connectome,
                nmda_weight=nmda_weight,
                state0=state0,
                output_indices=output_indices,
                input_group_a=input_group_a,
                input_group_b=input_group_b,
                rate_a=rate_a,
                amp_a=amp_a,
                rate_b=rate_b,
                amp_b=amp_b,
                total_ms=total_ms_this_trial,
                rng=trial_rng,
            )
            states.append(state_vec)
            trial_rates.append(mean_rate_hz)
            trial_total_ms.append(total_ms_this_trial)

        state_matrix = np.vstack(states)
        eff_rank = _effective_rank(state_matrix)
        rank_max = float(min(state_matrix.shape[0], state_matrix.shape[1]))
        eff_rank_norm = float(eff_rank / rank_max) if rank_max > 0 else 0.0

        mean_rate_hz = float(np.mean(trial_rates))
        excess_ratio = max(0.0, mean_rate_hz - cfg.high_rate_threshold_hz) / max(cfg.high_rate_threshold_hz, 1e-6)
        rate_penalty = float(cfg.high_rate_penalty_weight * (excess_ratio ** 2))

        score = float(eff_rank_norm - rate_penalty)

        metrics = {
            "effective_rank": float(eff_rank),
            "effective_rank_norm": float(eff_rank_norm),
            "mean_output_rate_hz": float(mean_rate_hz),
            "high_rate_penalty": float(rate_penalty),
            "rank_max": float(rank_max),
            "n_exc_output": float(output_indices.size),
            "n_input": float(n_input),
            "n_input_group_a": float(input_group_a.size),
            "n_input_group_b": float(input_group_b.size),
            "mean_total_ms": float(np.mean(trial_total_ms)),
            "std_total_ms": float(np.std(trial_total_ms)),
        }
        return score, metrics, decoded["params_flat"]


def _init_worker(cfg_dict: Dict[str, object]) -> None:
    global _WORKER_OBJECTIVE
    _WORKER_OBJECTIVE = SpatialEIObjective(SearchConfig(**cfg_dict))


def _evaluate_candidate(task: Tuple[int, int, List[float], int, Dict[str, object]]) -> Dict[str, object]:
    generation, eval_id, x, seed, cfg_dict = task
    x_arr = np.asarray(x, dtype=float)
    trial_seed = int(seed + 1009 * (eval_id + 1) + 100_003 * generation)

    global _WORKER_OBJECTIVE
    if _WORKER_OBJECTIVE is None:
        _WORKER_OBJECTIVE = SpatialEIObjective(SearchConfig(**cfg_dict))

    try:
        score, metrics, params = _WORKER_OBJECTIVE.evaluate(x_arr, trial_seed)
        error_msg = ""
    except Exception as exc:
        score = -1e6
        metrics = {
            "effective_rank": 0.0,
            "effective_rank_norm": 0.0,
            "mean_output_rate_hz": 0.0,
            "high_rate_penalty": 1e6,
        }
        params = {}
        error_msg = str(exc)

    return {
        "generation": int(generation),
        "eval_id": int(eval_id),
        "score": float(score),
        "loss": float(-score),
        "params": params,
        "metrics": metrics,
        "error": error_msg,
    }


def _write_csv(rows: List[Dict[str, object]], out_path: Path) -> None:
    if not rows:
        return

    fieldnames = ["generation", "eval_id", "score", "loss", "error"]
    param_keys = sorted({k for row in rows for k in row["params"].keys()})
    metric_keys = sorted({k for row in rows for k in row["metrics"].keys()})
    fieldnames += [f"param.{k}" for k in param_keys]
    fieldnames += [f"metric.{k}" for k in metric_keys]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {
                "generation": row["generation"],
                "eval_id": row["eval_id"],
                "score": row["score"],
                "loss": row["loss"],
                "error": row.get("error", ""),
            }
            out.update({f"param.{k}": v for k, v in row["params"].items()})
            out.update({f"metric.{k}": v for k, v in row["metrics"].items()})
            writer.writerow(out)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CMA-ES search over generate_spatial_ei_network parameters using "
            "effective-rank separation fitness with high-rate penalty."
        )
    )
    parser.add_argument("--max-evals", type=int, default=120)
    parser.add_argument("--popsize", type=int, default=12)
    parser.add_argument("--sigma0", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default="week8/cmaes_spatial_ei")
    parser.add_argument("--save-every", type=int, default=1)

    parser.add_argument("--n-neurons", type=int, default=150)
    parser.add_argument("--p-inputs", type=int, default=12)
    parser.add_argument("--total-ms", type=float, default=400.0)
    parser.add_argument("--total-ms-jitter", type=float, default=0.0,
                        help="Uniform per-trial jitter (ms) applied as total_ms +/- jitter.",
                        )
    parser.add_argument("--t-start-ms", type=float, default=150.0)
    parser.add_argument("--dt-ms", type=float, default=0.1)

    parser.add_argument("--input-rate-min", type=float, default=30.0)
    parser.add_argument("--input-rate-max", type=float, default=100.0)
    parser.add_argument("--input-amp-min", type=float, default=0.5)
    parser.add_argument("--input-amp-max", type=float, default=20.0)
    parser.add_argument("--high-rate-threshold", type=float, default=60.0)
    parser.add_argument("--high-rate-penalty-weight", type=float, default=0.1)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.t_start_ms >= args.total_ms:
        raise ValueError("--t-start-ms must be strictly less than --total-ms.")
    if args.total_ms_jitter < 0:
        raise ValueError("--total-ms-jitter must be >= 0.")
    if args.input_rate_min <= 0 or args.input_rate_max <= 0 or args.input_amp_min <= 0 or args.input_amp_max <= 0:
        raise ValueError("Input rate and amplitude ranges must be > 0.")
    if args.input_rate_min > args.input_rate_max:
        raise ValueError("--input-rate-min cannot exceed --input-rate-max.")
    if args.input_amp_min > args.input_amp_max:
        raise ValueError("--input-amp-min cannot exceed --input-amp-max.")

    cfg = SearchConfig(
        n_neurons=int(args.n_neurons),
        dt_ms=float(args.dt_ms),
        total_ms=float(args.total_ms),
        total_ms_jitter=float(args.total_ms_jitter),
        t_start_ms=float(args.t_start_ms),
        p_inputs=int(args.p_inputs),
        input_rate_min_hz=float(args.input_rate_min),
        input_rate_max_hz=float(args.input_rate_max),
        input_amp_min=float(args.input_amp_min),
        input_amp_max=float(args.input_amp_max),
        high_rate_threshold_hz=float(args.high_rate_threshold),
        high_rate_penalty_weight=float(args.high_rate_penalty_weight),
        base_seed=int(args.seed),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x0 = _default_unit_vector()
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
    cfg_dict = asdict(cfg)

    objective = SpatialEIObjective(cfg) if args.n_workers <= 1 else None
    pool = None
    if args.n_workers > 1:
        pool = ProcessPoolExecutor(
            max_workers=int(args.n_workers),
            initializer=_init_worker,
            initargs=(cfg_dict,),
        )

    try:
        while eval_id < int(args.max_evals) and not es.stop():
            generation += 1
            candidates = es.ask()
            tasks = []
            for x in candidates:
                if eval_id >= int(args.max_evals):
                    break
                tasks.append((generation, eval_id, list(x), int(args.seed), cfg_dict))
                eval_id += 1

            if args.n_workers > 1:
                results = list(pool.map(_evaluate_candidate, tasks))
            else:
                results = []
                for g, eid, x, seed, _ in tasks:
                    x_arr = np.asarray(x, dtype=float)
                    trial_seed = int(seed + 1009 * (eid + 1) + 100_003 * g)
                    try:
                        score, metrics, params = objective.evaluate(x_arr, trial_seed)
                        error_msg = ""
                    except Exception as exc:
                        score = -1e6
                        metrics = {
                            "effective_rank": 0.0,
                            "effective_rank_norm": 0.0,
                            "mean_output_rate_hz": 0.0,
                            "high_rate_penalty": 1e6,
                        }
                        params = {}
                        error_msg = str(exc)
                    results.append(
                        {
                            "generation": int(g),
                            "eval_id": int(eid),
                            "score": float(score),
                            "loss": float(-score),
                            "params": params,
                            "metrics": metrics,
                            "error": error_msg,
                        }
                    )

            losses = [float(r["loss"]) for r in results]
            es.tell(candidates[: len(losses)], losses)

            all_rows.extend(results)
            local_best = max(results, key=lambda r: float(r["score"]))
            if best_row is None or float(local_best["score"]) > float(best_row["score"]):
                best_row = local_best

            if generation % int(args.save_every) == 0:
                _write_csv(all_rows, out_dir / "cmaes_trials.csv")

            print(
                f"gen={generation} evals={eval_id} "
                f"best_gen={local_best['score']:.4f} best_global={best_row['score']:.4f}"
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
        "config": cfg_dict,
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    with (out_dir / "best_result.json").open("w", encoding="utf-8") as f:
        json.dump(best_row, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
