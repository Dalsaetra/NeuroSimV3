import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import networkx as nx
import numpy as np
import optuna

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.connectome import Connectome
from src.external_inputs import PoissonInput
from src.network_weight_distributor import assign_lognormal_weights_for_ntype
from src.neuron_population import NeuronPopulation
from src.overhead import Simulation


PARAM_SPECS = [
    ("g_AMPA_max", 1.0, 1500.0, False),
    ("g_NMDA_max", 1.0, 1500.0, False),
    ("g_GABA_A_max", 1.0, 1500.0, False),
    ("g_GABA_B_max", 1.0, 1500.0, False),
    ("A_AMPA", 0.0001, 0.1, False),
    ("A_NMDA", 0.0001, 0.1, False),
    ("A_GABA_A", 0.0001, 0.1, False),
    ("A_GABA_B", 0.0001, 0.1, False),
    ("inhibitory_scale_g", 0.1, 1000.0, False),
    ("v_ext", 10.0, 200, False),
    ("external_amplitude", 0.5, 50.0, False),
    ("recurrent_exc_lognorm_sigma", 0.1, 3.0, False),
    ("inhibitory_nmda_weight", 0.0, 1.0, False),
]

TRANSMITTERS = ("AMPA", "NMDA", "GABA_A", "GABA_B")
OBJECTIVE_NAMES = (
    "persistence",
    "asynchrony",
    "regime",
    "transmitter_balance",
)
STD_SUMMARY_KEYS = (
    "score_ssai",
    "score_balance",
    "score_persistence",
    "score_asynchrony",
    "score_regime",
    "ISI_CV_mean",
    "Fano_median_300ms",
    "mean_noise_corr_50ms",
    "rate_post_Hz",
    "rate_late_Hz",
    "rate_post_exc_Hz",
    "rate_post_inh_Hz",
    "rate_late_exc_Hz",
    "rate_late_inh_Hz",
    "mean_voltage_post_mV",
    "balance_min_fraction",
    "balance_total_effective_strength",
    "conductance_mean_AMPA",
    "conductance_mean_NMDA",
    "conductance_mean_GABA_A",
    "conductance_mean_GABA_B",
)


def suggest_params(trial: optuna.Trial) -> Dict[str, float]:
    params: Dict[str, float] = {}
    for name, low, high, use_log in PARAM_SPECS:
        params[name] = trial.suggest_float(name, low, high, log=use_log)
    return params


@dataclass
class SearchConfig:
    n_neurons: int = 1000
    n_excit: int = 800
    n_out: int = 100
    excitatory_type: str = "ss4"
    inhibitory_type: str = "b"
    dt_ms: float = 0.1
    ext_on_ms: float = 1000.0
    total_ms: float = 3000.0
    n_repeats: int = 3
    delay_mean_exc_ms: float = 1.5
    delay_std_exc_ms: float = 0.3
    delay_mean_inh_ms: float = 1.5
    delay_std_inh_ms: float = 0.3
    recurrent_exc_mu: float = 0.0
    recurrent_exc_wmax: float = 100.0
    state_v_min: float = -100.0
    state_v_max: float = -70.0
    state_u_min: float = 0.0
    state_u_max: float = 400.0
    transmitter_balance_entropy_weight: float = 0.5
    transmitter_balance_min_weight: float = 0.5
    transmitter_balance_target_fraction: float = 0.25
    reject_rate_late_low_hz: float = 0.5
    reject_rate_late_exc_high_hz: float = 10.0
    reject_rate_late_inh_high_hz: float = 50.0
    reject_rate_late_extreme_low_hz: float = 0.05
    reject_rate_late_exc_extreme_high_hz: float = 20.0
    reject_rate_late_inh_extreme_high_hz: float = 100.0
    late_rate_low_penalty_weight: float = 4.0
    late_rate_exc_high_penalty_weight: float = 4.0
    late_rate_inh_high_penalty_weight: float = 4.0
    reject_objective_value: float = -100.0
    base_seed: int = 1234


def config_to_dict(cfg: SearchConfig) -> Dict[str, object]:
    return asdict(cfg)


def _safe_tanh(x: float) -> float:
    return float(np.tanh(float(x)))


def _compute_peak_ratio(freq_hz: np.ndarray, psd: np.ndarray) -> float:
    if freq_hz is None or psd is None:
        return 0.0
    if len(freq_hz) == 0 or len(psd) == 0:
        return 0.0

    band_mask = (freq_hz >= 2.0) & (freq_hz <= 120.0)
    if not np.any(band_mask):
        return 0.0

    band = psd[band_mask]
    median = float(np.median(band))
    if median <= 0:
        return 0.0
    peak = float(np.max(band))
    return peak / (median + 1e-12)


def _nmda_voltage_factor(voltage_mV: float) -> float:
    mg = 1.0
    return float(1.0 / (1.0 + 0.28 * mg * np.exp(-0.062 * float(voltage_mV))))


def _aggregate_repeat_metrics(repeat_metrics: Tuple[Dict[str, float], ...]) -> Dict[str, float]:
    if not repeat_metrics:
        return {}

    aggregated: Dict[str, float] = {}
    numeric_keys = sorted(
        {
            key
            for metrics in repeat_metrics
            for key, value in metrics.items()
            if isinstance(value, (int, float, np.integer, np.floating))
        }
    )

    for key in numeric_keys:
        values = np.array(
            [
                float(metrics[key])
                for metrics in repeat_metrics
                if key in metrics and isinstance(metrics[key], (int, float, np.integer, np.floating))
            ],
            dtype=float,
        )
        if values.size == 0:
            continue
        aggregated[key] = float(np.mean(values))
        if key in STD_SUMMARY_KEYS:
            aggregated[f"{key}_std"] = float(np.std(values))

    return aggregated


class SSAIMultiObjective:
    def __init__(self, cfg: SearchConfig):
        self.cfg = cfg
        self.template_graph = self._build_template_graph(seed=cfg.base_seed)
        self.template_topology_metrics = self._compute_topology_metrics(self.template_graph)

    def _build_template_graph(self, seed: int) -> nx.DiGraph:
        rng = np.random.default_rng(seed)
        g = nx.DiGraph()

        for i in range(self.cfg.n_neurons):
            is_inhib = i >= self.cfg.n_excit
            g.add_node(
                i,
                inhibitory=is_inhib,
                ntype=self.cfg.inhibitory_type if is_inhib else self.cfg.excitatory_type,
                layer=0,
            )

        for i in range(self.cfg.n_neurons):
            targets = rng.choice(self.cfg.n_neurons, size=self.cfg.n_out, replace=False)
            is_inhib = i >= self.cfg.n_excit
            if is_inhib:
                delays = np.maximum(
                    0.1,
                    rng.normal(
                        self.cfg.delay_mean_inh_ms,
                        self.cfg.delay_std_inh_ms,
                        size=self.cfg.n_out,
                    ),
                )
            else:
                delays = np.maximum(
                    0.1,
                    rng.normal(
                        self.cfg.delay_mean_exc_ms,
                        self.cfg.delay_std_exc_ms,
                        size=self.cfg.n_out,
                    ),
                )

            for target, delay in zip(targets, delays):
                g.add_edge(i, int(target), weight=1.0, distance=float(delay))

        return g

    def _compute_topology_metrics(self, graph: nx.DiGraph) -> Dict[str, float]:
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        out_deg = np.array([graph.out_degree(n) for n in graph.nodes()], dtype=float)
        in_deg = np.array([graph.in_degree(n) for n in graph.nodes()], dtype=float)
        delays = np.array([data.get("distance", 0.0) for _, _, data in graph.edges(data=True)], dtype=float)
        exc_delays = []
        inh_delays = []
        for u, _, data in graph.edges(data=True):
            if u >= self.cfg.n_excit:
                inh_delays.append(float(data.get("distance", 0.0)))
            else:
                exc_delays.append(float(data.get("distance", 0.0)))

        possible_edges = float(n_nodes * n_nodes) if n_nodes > 0 else 1.0
        reciprocity = nx.reciprocity(graph)
        return {
            "topology_n_nodes": float(n_nodes),
            "topology_n_edges": float(n_edges),
            "topology_density_observed": float(n_edges / possible_edges),
            "topology_out_degree_mean": float(np.mean(out_deg)) if out_deg.size else 0.0,
            "topology_out_degree_std": float(np.std(out_deg)) if out_deg.size else 0.0,
            "topology_in_degree_mean": float(np.mean(in_deg)) if in_deg.size else 0.0,
            "topology_in_degree_std": float(np.std(in_deg)) if in_deg.size else 0.0,
            "topology_reciprocity": float(reciprocity) if reciprocity is not None else 0.0,
            "topology_delay_mean_ms": float(np.mean(delays)) if delays.size else 0.0,
            "topology_delay_mean_exc_ms": float(np.mean(exc_delays)) if exc_delays else 0.0,
            "topology_delay_mean_inh_ms": float(np.mean(inh_delays)) if inh_delays else 0.0,
        }

    def _build_connectome(self, graph: nx.DiGraph, inhibitory_nmda_weight: float):
        dt_ms = self.cfg.dt_ms
        n_neurons = graph.number_of_nodes()
        neuron_types = [self.cfg.excitatory_type, self.cfg.inhibitory_type]
        inhibitory = [False, True]
        threshold_decay = np.exp(-dt_ms / 5.0)

        pop = NeuronPopulation(n_neurons, neuron_types, inhibitory, threshold_decay)
        max_synapses = max(dict(graph.out_degree()).values())

        connectome = Connectome(max_synapses, pop)
        connectome.nx_to_connectome(graph)

        nmda_weight = np.ones(n_neurons, dtype=float)
        nmda_weight[pop.inhibitory_mask.astype(bool)] = inhibitory_nmda_weight
        return connectome, pop, nmda_weight

    def _initialize_state(self, n_neurons: int, rng: np.random.Generator):
        vs = rng.uniform(self.cfg.state_v_min, self.cfg.state_v_max, size=n_neurons)
        us = rng.uniform(self.cfg.state_u_min, self.cfg.state_u_max, size=n_neurons)
        spikes = np.zeros(n_neurons, dtype=bool)
        ts = np.zeros_like(spikes)
        return vs, us, spikes, ts

    def _score_ssai(self, sim: Simulation) -> Tuple[float, Dict[str, float]]:
        cfg = self.cfg

        post_stats = sim.stats.compute_metrics(
            cfg.dt_ms,
            bin_ms_participation=300,
            t_start_ms=cfg.ext_on_ms,
            t_stop_ms=cfg.total_ms,
        )
        late_stats = sim.stats.compute_metrics(
            cfg.dt_ms,
            bin_ms_participation=300,
            t_start_ms=max(cfg.ext_on_ms, cfg.total_ms - 300.0),
            t_stop_ms=cfg.total_ms,
        )

        isi_cv_mean = float(post_stats.get("ISI_CV_mean", 0.0))
        fano_300 = float(post_stats.get("Fano_median_300ms", 0.0))
        noise_corr_50 = float(post_stats.get("mean_noise_corr_50ms", 0.0))
        pop_entropy = float(post_stats.get("pop_spec_entropy", 0.0))
        rate_post = float(post_stats.get("rate_mean_Hz", 0.0))
        rate_late = float(late_stats.get("rate_mean_Hz", 0.0))
        rate_post_exc = float(post_stats.get("rate_mean_Hz_E", 0.0))
        rate_post_inh = float(post_stats.get("rate_mean_Hz_I", 0.0))
        rate_late_exc = float(late_stats.get("rate_mean_Hz_E", 0.0))
        rate_late_inh = float(late_stats.get("rate_mean_Hz_I", 0.0))

        peak_ratio = _compute_peak_ratio(
            post_stats.get("pop_psd_freq_hz", np.array([])),
            post_stats.get("pop_psd", np.array([])),
        )

        v_mean = -60.0
        voltages = sim.stats.voltages()
        if voltages.size > 0:
            t_ms = sim.stats.times_ms(dt_ms=cfg.dt_ms)
            v_mask = (t_ms >= cfg.ext_on_ms) & (t_ms <= cfg.total_ms)
            if np.any(v_mask):
                v_mean = float(np.mean(voltages[:, v_mask]))

        s_cv = _safe_tanh(max(0.0, isi_cv_mean) / 2.0)
        s_fano = _safe_tanh(max(0.0, fano_300) / 4.0)
        s_persist = _safe_tanh(min(rate_post, rate_late) / 5.0)
        s_rate = _safe_tanh(((rate_post + rate_late) * 0.5) / 20.0)
        s_entropy = _safe_tanh(max(0.0, pop_entropy) / 8.0)

        p_corr = max(0.0, noise_corr_50 - 0.10)
        p_peak = max(0.0, peak_ratio - 10.0) / 10.0
        p_vhigh = max(0.0, (v_mean + 30.0) / 30.0)

        score = (
            2.0 * s_cv
            + 1.5 * s_fano
            + 3.0 * s_persist
            + 2.0 * s_rate
            + 1.0 * s_entropy
            - 2.0 * p_corr
            - 1.5 * p_peak
            - 1.0 * p_vhigh
        )

        if rate_late < 0.5:
            score -= (0.5 - rate_late) * 2.0
        if rate_late < 2.0:
            score -= (2.0 - rate_late) * 1.0

        diagnostics = {
            "score_ssai": float(score),
            "ISI_CV_mean": isi_cv_mean,
            "Fano_median_300ms": fano_300,
            "mean_noise_corr_50ms": noise_corr_50,
            "pop_spec_entropy": pop_entropy,
            "rate_post_Hz": rate_post,
            "rate_late_Hz": rate_late,
            "rate_post_exc_Hz": rate_post_exc,
            "rate_post_inh_Hz": rate_post_inh,
            "rate_late_exc_Hz": rate_late_exc,
            "rate_late_inh_Hz": rate_late_inh,
            "psd_peak_ratio": float(peak_ratio),
            "mean_voltage_post_mV": float(v_mean),
            "p_corr": float(p_corr),
            "p_peak": float(p_peak),
            "p_vhigh": float(p_vhigh),
        }
        return float(score), diagnostics

    def _score_persistence(self, diagnostics: Dict[str, float]) -> float:
        rate_post = float(diagnostics.get("rate_post_Hz", 0.0))
        rate_late = float(diagnostics.get("rate_late_Hz", 0.0))
        persist_core = _safe_tanh(min(rate_post, rate_late) / 5.0)
        late_support = _safe_tanh(rate_late / 5.0)
        return float(0.7 * persist_core + 0.3 * late_support)

    def _score_asynchrony(self, diagnostics: Dict[str, float]) -> float:
        isi_cv_mean = float(diagnostics.get("ISI_CV_mean", 0.0))
        fano_300 = float(diagnostics.get("Fano_median_300ms", 0.0))
        noise_corr_50 = float(diagnostics.get("mean_noise_corr_50ms", 0.0))
        peak_ratio = float(diagnostics.get("psd_peak_ratio", 0.0))

        s_cv = _safe_tanh(max(0.0, isi_cv_mean) / 2.0)
        s_fano = _safe_tanh(max(0.0, fano_300) / 4.0)
        p_corr = max(0.0, noise_corr_50 - 0.10)
        p_peak = max(0.0, peak_ratio - 10.0) / 10.0

        return float(1.5 * s_cv + 1.0 * s_fano - 1.5 * p_corr - 1.0 * p_peak)

    def _score_regime(self, diagnostics: Dict[str, float]) -> float:
        rate_post = float(diagnostics.get("rate_post_Hz", 0.0))
        rate_late = float(diagnostics.get("rate_late_Hz", 0.0))
        pop_entropy = float(diagnostics.get("pop_spec_entropy", 0.0))
        v_mean = float(diagnostics.get("mean_voltage_post_mV", -60.0))

        mean_rate = 0.5 * (rate_post + rate_late)
        rate_target = 12.0
        rate_band = 10.0
        rate_term = 1.0 - min(1.0, abs(mean_rate - rate_target) / rate_band)
        entropy_term = _safe_tanh(max(0.0, pop_entropy) / 8.0)
        p_vhigh = max(0.0, (v_mean + 30.0) / 30.0)
        return float(1.5 * rate_term + 1.0 * entropy_term - 1.0 * p_vhigh)

    def _hard_reject(self, diagnostics: Dict[str, float]) -> Tuple[bool, str]:
        rate_late = float(diagnostics.get("rate_late_Hz", 0.0))
        rate_late_exc = float(diagnostics.get("rate_late_exc_Hz", 0.0))
        rate_late_inh = float(diagnostics.get("rate_late_inh_Hz", 0.0))
        if rate_late < self.cfg.reject_rate_late_extreme_low_hz:
            return True, "late_rate_too_low"
        if rate_late_exc > self.cfg.reject_rate_late_exc_extreme_high_hz:
            return True, "late_exc_rate_too_high"
        if rate_late_inh > self.cfg.reject_rate_late_inh_extreme_high_hz:
            return True, "late_inh_rate_too_high"
        return False, ""

    def _late_rate_soft_penalty(self, diagnostics: Dict[str, float]) -> Tuple[float, str]:
        rate_late = float(diagnostics.get("rate_late_Hz", 0.0))
        rate_late_exc = float(diagnostics.get("rate_late_exc_Hz", 0.0))
        rate_late_inh = float(diagnostics.get("rate_late_inh_Hz", 0.0))
        if rate_late < self.cfg.reject_rate_late_low_hz:
            deficit = (self.cfg.reject_rate_late_low_hz - rate_late) / max(self.cfg.reject_rate_late_low_hz, 1e-9)
            return float(self.cfg.late_rate_low_penalty_weight * deficit * deficit), "late_rate_low_penalty"
        penalty = 0.0
        reasons = []
        if rate_late_exc > self.cfg.reject_rate_late_exc_high_hz:
            excess_exc = (rate_late_exc - self.cfg.reject_rate_late_exc_high_hz) / max(
                self.cfg.reject_rate_late_exc_high_hz,
                1e-9,
            )
            penalty += float(self.cfg.late_rate_exc_high_penalty_weight * excess_exc * excess_exc)
            reasons.append("late_exc_rate_high_penalty")
        if rate_late_inh > self.cfg.reject_rate_late_inh_high_hz:
            excess_inh = (rate_late_inh - self.cfg.reject_rate_late_inh_high_hz) / max(
                self.cfg.reject_rate_late_inh_high_hz,
                1e-9,
            )
            penalty += float(self.cfg.late_rate_inh_high_penalty_weight * excess_inh * excess_inh)
            reasons.append("late_inh_rate_high_penalty")
        return float(penalty), ",".join(reasons)

    def _score_transmitter_balance(
        self,
        params: Dict[str, float],
        conductance_means: Dict[str, float],
    ) -> Tuple[float, Dict[str, float]]:
        target_fraction = float(self.cfg.transmitter_balance_target_fraction)

        param_strengths = {
            "AMPA": float(params["g_AMPA_max"]),
            "NMDA": float(params["g_NMDA_max"]),
            "GABA_A": float(params["g_GABA_A_max"]),
            "GABA_B": float(params["g_GABA_B_max"]),
        }

        effective_strengths = {
            name: float(conductance_means.get(name, 0.0)) for name in TRANSMITTERS
        }
        total_effective = float(sum(effective_strengths.values()))

        if total_effective <= 0.0:
            diagnostics = {"score_balance": 0.0}
            for name in TRANSMITTERS:
                diagnostics[f"balance_param_strength_{name}"] = float(param_strengths[name])
                diagnostics[f"balance_effective_strength_{name}"] = 0.0
                diagnostics[f"balance_fraction_{name}"] = 0.0
            diagnostics["balance_entropy_norm"] = 0.0
            diagnostics["balance_min_fraction"] = 0.0
            diagnostics["balance_total_effective_strength"] = 0.0
            return 0.0, diagnostics

        fractions = {
            name: float(effective_strengths[name] / total_effective) for name in TRANSMITTERS
        }
        frac_vec = np.array([fractions[name] for name in TRANSMITTERS], dtype=float)
        entropy = -float(np.sum(frac_vec * np.log(frac_vec + 1e-12)))
        entropy_norm = float(entropy / math.log(len(TRANSMITTERS)))
        min_fraction = float(np.min(frac_vec))
        min_fraction_norm = float(np.clip(min_fraction / target_fraction, 0.0, 1.0))

        balance_score = (
            self.cfg.transmitter_balance_entropy_weight * entropy_norm
            + self.cfg.transmitter_balance_min_weight * min_fraction_norm
        )

        diagnostics = {
            "score_balance": float(balance_score),
            "balance_entropy_norm": float(entropy_norm),
            "balance_min_fraction": float(min_fraction),
            "balance_total_effective_strength": float(total_effective),
        }
        for name in TRANSMITTERS:
            diagnostics[f"balance_param_strength_{name}"] = float(param_strengths[name])
            diagnostics[f"balance_effective_strength_{name}"] = float(effective_strengths[name])
            diagnostics[f"balance_fraction_{name}"] = float(fractions[name])

        return float(balance_score), diagnostics

    def _set_trial_attrs(self, trial: optuna.Trial, values: Dict[str, object]) -> None:
        for key, value in values.items():
            if isinstance(value, (np.floating, np.integer)):
                trial.set_user_attr(key, float(value))
            elif isinstance(value, (float, int, str, bool)):
                trial.set_user_attr(key, value)

    def __call__(self, trial: optuna.Trial) -> Tuple[float, float, float, float]:
        cfg = self.cfg
        trial_seed = cfg.base_seed + 1009 * (trial.number + 1)
        trial_rng = np.random.default_rng(trial_seed)
        params = suggest_params(trial)

        syn_g_ampa_max = params["g_AMPA_max"]
        syn_g_nmda_max = params["g_NMDA_max"]
        syn_g_gabaa_max = params["g_GABA_A_max"]
        syn_g_gabab_max = params["g_GABA_B_max"]

        a_ampa = params["A_AMPA"]
        a_nmda = params["A_NMDA"]
        a_gabaa = params["A_GABA_A"]
        a_gabab = params["A_GABA_B"]

        inhib_g = params["inhibitory_scale_g"]
        v_ext = params["v_ext"]
        ext_amp = params["external_amplitude"]
        recurrent_exc_sigma = params["recurrent_exc_lognorm_sigma"]
        inhibitory_nmda_weight = params["inhibitory_nmda_weight"]

        graph = self.template_graph.copy()

        for u, _, data in graph.edges(data=True):
            data["weight"] = float(inhib_g if u >= cfg.n_excit else 1.0)

        assign_lognormal_weights_for_ntype(
            graph,
            cfg.excitatory_type,
            mu=cfg.recurrent_exc_mu,
            sigma=recurrent_exc_sigma,
            w_max=cfg.recurrent_exc_wmax,
            rng=trial_rng,
        )

        connectome, pop, nmda_weight = self._build_connectome(
            graph,
            inhibitory_nmda_weight=inhibitory_nmda_weight,
        )

        ext_amp_vec = np.full(cfg.n_neurons, ext_amp, dtype=float)
        ext_amp_vec[pop.inhibitory_mask.astype(bool)] = 0.0

        ext_on_steps = int(round(cfg.ext_on_ms / cfg.dt_ms))
        total_steps = int(round(cfg.total_ms / cfg.dt_ms))
        ext_off_steps = max(0, total_steps - ext_on_steps)

        repeat_metrics = []
        repeat_reject_reasons = []
        for repeat_idx in range(cfg.n_repeats):
            repeat_seed = trial_seed + 7919 * repeat_idx
            rng = np.random.default_rng(repeat_seed)
            state0 = self._initialize_state(cfg.n_neurons, rng)

            sim = Simulation(
                connectome,
                cfg.dt_ms,
                stepper_type="euler_det",
                state0=state0,
                enable_plasticity=False,
                synapse_type="standard",
                synapse_kwargs={"LT_scale": 1.0, "NMDA_weight": nmda_weight},
                enable_debug_logger=False,
            )

            syn = sim.synapse_dynamics
            syn.g_AMPA_max = syn_g_ampa_max
            syn.g_NMDA_max = syn_g_nmda_max
            syn.g_GABA_A_max = syn_g_gabaa_max
            syn.g_GABA_B_max = syn_g_gabab_max
            syn.A_AMPA = a_ampa
            syn.A_NMDA = a_nmda
            syn.A_GABA_A = a_gabaa
            syn.A_GABA_B = a_gabab

            poisson = PoissonInput(cfg.n_neurons, rate=v_ext, amplitude=ext_amp_vec, rng=rng)

            for _ in range(ext_on_steps):
                sim.step(spike_ext=poisson(cfg.dt_ms))

            conductance_sums = {name: 0.0 for name in TRANSMITTERS}
            for _ in range(ext_off_steps):
                sim.step()
                conductance_sums["AMPA"] += float(np.mean(syn.g_AMPA * syn.g_AMPA_max))
                conductance_sums["NMDA"] += float(np.mean(syn.g_NMDA * syn.g_NMDA_max))
                conductance_sums["GABA_A"] += float(np.mean(syn.g_GABA_A * syn.g_GABA_A_max))
                conductance_sums["GABA_B"] += float(np.mean(syn.g_GABA_B * syn.g_GABA_B_max))

            conductance_means = {
                name: float(conductance_sums[name] / ext_off_steps) if ext_off_steps > 0 else 0.0
                for name in TRANSMITTERS
            }

            metrics: Dict[str, float] = {
            }
            ssai_score, ssai_diagnostics = self._score_ssai(sim)
            metrics.update(ssai_diagnostics)
            reject, reject_reason = self._hard_reject(ssai_diagnostics)
            if reject:
                persistence_score = float(cfg.reject_objective_value)
                asynchrony_score = float(cfg.reject_objective_value)
                regime_score = float(cfg.reject_objective_value)
                balance_score = float(cfg.reject_objective_value)
                balance_diagnostics = {
                    "score_balance": float(balance_score),
                    "balance_entropy_norm": 0.0,
                    "balance_min_fraction": 0.0,
                    "balance_total_effective_strength": 0.0,
                }
                for name in TRANSMITTERS:
                    balance_diagnostics[f"balance_param_strength_{name}"] = float(params[f"g_{name}_max"])
                    balance_diagnostics[f"balance_effective_strength_{name}"] = 0.0
                    balance_diagnostics[f"balance_fraction_{name}"] = 0.0
                soft_penalty = 0.0
                soft_penalty_reason = ""
            else:
                persistence_score = self._score_persistence(ssai_diagnostics)
                asynchrony_score = self._score_asynchrony(ssai_diagnostics)
                regime_score = self._score_regime(ssai_diagnostics)
                balance_score, balance_diagnostics = self._score_transmitter_balance(
                    params=params,
                    conductance_means=conductance_means,
                )
                soft_penalty, soft_penalty_reason = self._late_rate_soft_penalty(ssai_diagnostics)
                if soft_penalty > 0.0:
                    persistence_score -= 1.5 * soft_penalty
                    regime_score -= 1.5 * soft_penalty
                    asynchrony_score -= 0.5 * soft_penalty
                    balance_score -= 0.25 * soft_penalty

            metrics["score_ssai"] = float(ssai_score)
            metrics["score_persistence"] = float(persistence_score)
            metrics["score_asynchrony"] = float(asynchrony_score)
            metrics["score_regime"] = float(regime_score)
            metrics["score_balance"] = float(balance_score)
            metrics["hard_reject"] = float(1.0 if reject else 0.0)
            metrics["late_rate_soft_penalty"] = float(soft_penalty)
            for name in TRANSMITTERS:
                metrics[f"conductance_mean_{name}"] = float(conductance_means[name])
            metrics.update(balance_diagnostics)
            repeat_metrics.append(metrics)
            repeat_reject_reasons.append(reject_reason or soft_penalty_reason)

        aggregated_metrics = _aggregate_repeat_metrics(tuple(repeat_metrics))
        persistence_score = float(aggregated_metrics["score_persistence"])
        asynchrony_score = float(aggregated_metrics["score_asynchrony"])
        regime_score = float(aggregated_metrics["score_regime"])
        balance_score = float(aggregated_metrics["score_balance"])

        diagnostics: Dict[str, object] = {
            "trial_seed": int(trial_seed),
            "n_repeats": int(cfg.n_repeats),
            "score_persistence": float(persistence_score),
            "score_asynchrony": float(asynchrony_score),
            "score_regime": float(regime_score),
            "score_balance": float(balance_score),
            "network_density_target": float(cfg.n_out / cfg.n_neurons),
            "n_out": int(cfg.n_out),
        }
        diagnostics.update(self.template_topology_metrics)
        diagnostics.update(aggregated_metrics)
        for repeat_idx, metrics in enumerate(repeat_metrics):
            diagnostics[f"repeat_{repeat_idx}_seed"] = int(trial_seed + 7919 * repeat_idx)
            diagnostics[f"repeat_{repeat_idx}_score_persistence"] = float(metrics["score_persistence"])
            diagnostics[f"repeat_{repeat_idx}_score_asynchrony"] = float(metrics["score_asynchrony"])
            diagnostics[f"repeat_{repeat_idx}_score_regime"] = float(metrics["score_regime"])
            diagnostics[f"repeat_{repeat_idx}_score_balance"] = float(metrics["score_balance"])
            diagnostics[f"repeat_{repeat_idx}_score_ssai"] = float(metrics["score_ssai"])
            diagnostics[f"repeat_{repeat_idx}_reject_reason"] = repeat_reject_reasons[repeat_idx]

        self._set_trial_attrs(trial, diagnostics)
        return (
            float(persistence_score),
            float(asynchrony_score),
            float(regime_score),
            float(balance_score),
        )
