from __future__ import annotations

import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import networkx as nx
import numpy as np

try:
    import optuna
except ModuleNotFoundError:  # pragma: no cover - post-analysis can run without Optuna installed
    optuna = None

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
    ("A_AMPA", 0.0001, 0.05, True),
    ("A_NMDA", 0.0001, 0.05, True),
    ("A_GABA_A", 0.0001, 0.05, True),
    ("A_GABA_B", 0.0001, 0.05, True),
    ("inhibitory_scale_g", 0.1, 1000.0, False),
    ("external_amplitude", 0.1, 20.0, True),
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
    "score_balance",
    "score_persistence",
    "score_asynchrony",
    "score_regime",
    "ISI_CV_mean",
    "Fano_median_300ms",
    "mean_noise_corr_50ms",
    "psd_peak_ratio",
    "rate_post_Hz",
    "rate_late_Hz",
    "rate_post_exc_Hz",
    "rate_post_inh_Hz",
    "rate_late_exc_Hz",
    "rate_late_inh_Hz",
    "mean_voltage_post_mV",
    "voltage_rest_var_post_exc_mV2",
    "voltage_rest_var_post_inh_mV2",
    "conductance_load_post",
    "saturation_freq_AMPA",
    "saturation_freq_NMDA",
    "saturation_freq_GABA_A",
    "saturation_freq_GABA_B",
    "saturation_freq_mean",
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
    external_rate_hz: float = 50.0
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
    reject_rate_late_extreme_low_hz: float = 0.05
    persistence_rate_min_hz: float = 1.0
    persistence_rate_exc_max_hz: float = 10.0
    persistence_rate_inh_max_hz: float = 100.0
    regime_voltage_var_exc_scale_mV2: float = 100.0
    regime_voltage_var_inh_scale_mV2: float = 100.0
    regime_voltage_var_exc_weight: float = 1.0
    regime_voltage_var_inh_weight: float = 0.2
    regime_conductance_load_scale: float = 100.0
    regime_conductance_load_weight: float = 1.0
    regime_saturation_threshold: float = 0.999
    regime_saturation_weight: float = 10000.0
    asynchrony_cv_weight: float = 1.0
    asynchrony_fano_weight: float = 1.0
    asynchrony_corr_weight: float = 1.0
    asynchrony_peak_ratio_threshold: float = 200.0
    asynchrony_peak_ratio_scale: float = 200.0
    asynchrony_peak_ratio_weight: float = 1.0
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

    def _compute_diagnostics(self, sim: Simulation) -> Dict[str, float]:
        cfg = self.cfg

        post_stats = sim.stats.compute_metrics(
            cfg.dt_ms,
            t_start_ms=cfg.ext_on_ms,
            t_stop_ms=cfg.total_ms,
        )
        late_stats = sim.stats.compute_metrics(
            cfg.dt_ms,
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
        v_rest_var_exc = 0.0
        v_rest_var_inh = 0.0
        voltages = sim.stats.voltages()
        if voltages.size > 0:
            t_ms = sim.stats.times_ms(dt_ms=cfg.dt_ms)
            v_mask = (t_ms >= cfg.ext_on_ms) & (t_ms <= cfg.total_ms)
            if np.any(v_mask):
                voltages_post = voltages[:, v_mask]
                v_mean = float(np.mean(voltages_post))
                inhib_mask = np.asarray(sim.stats.inhibitory_mask, dtype=bool)
                exc_mask = ~inhib_mask
                if np.any(exc_mask):
                    exc_dev = voltages_post[exc_mask] - (-60.0)
                    v_rest_var_exc = float(np.mean(exc_dev * exc_dev))
                if np.any(inhib_mask):
                    inh_dev = voltages_post[inhib_mask] - (-55.0)
                    v_rest_var_inh = float(np.mean(inh_dev * inh_dev))

        p_corr = max(0.0, noise_corr_50 - 0.10)
        p_vhigh = max(0.0, (v_mean + 30.0) / 30.0)

        diagnostics = {
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
            "voltage_rest_var_post_exc_mV2": float(v_rest_var_exc),
            "voltage_rest_var_post_inh_mV2": float(v_rest_var_inh),
            "p_corr": float(p_corr),
            "p_vhigh": float(p_vhigh),
        }
        return diagnostics

    def _score_persistence(self, diagnostics: Dict[str, float]) -> float:
        rate_post = float(diagnostics.get("rate_post_Hz", 0.0))
        rate_late = float(diagnostics.get("rate_late_Hz", 0.0))
        rate_post_exc = float(diagnostics.get("rate_post_exc_Hz", 0.0))
        rate_late_exc = float(diagnostics.get("rate_late_exc_Hz", 0.0))
        rate_post_inh = float(diagnostics.get("rate_post_inh_Hz", 0.0))
        rate_late_inh = float(diagnostics.get("rate_late_inh_Hz", 0.0))

        def _low_penalty(rate_hz: float, threshold_hz: float) -> float:
            if rate_hz >= threshold_hz:
                return 0.0
            deficit = (threshold_hz - rate_hz) / max(threshold_hz, 1e-9)
            return float(deficit * deficit)

        def _high_penalty(rate_hz: float, threshold_hz: float) -> float:
            if rate_hz <= threshold_hz:
                return 0.0
            excess = (rate_hz - threshold_hz) / max(threshold_hz, 1e-9)
            return float(excess * excess)

        low_penalty = 0.5 * (
            _low_penalty(rate_post, self.cfg.persistence_rate_min_hz)
            + _low_penalty(rate_late, self.cfg.persistence_rate_min_hz)
        )
        high_exc_penalty = 0.5 * (
            _high_penalty(rate_post_exc, self.cfg.persistence_rate_exc_max_hz)
            + _high_penalty(rate_late_exc, self.cfg.persistence_rate_exc_max_hz)
        )
        high_inh_penalty = 0.5 * (
            _high_penalty(rate_post_inh, self.cfg.persistence_rate_inh_max_hz)
            + _high_penalty(rate_late_inh, self.cfg.persistence_rate_inh_max_hz)
        )

        return float(1.0 - low_penalty - high_exc_penalty - high_inh_penalty)

    def _score_asynchrony(self, diagnostics: Dict[str, float]) -> float:
        isi_cv_mean = float(diagnostics.get("ISI_CV_mean", 0.0))
        fano_300 = float(diagnostics.get("Fano_median_300ms", 0.0))
        noise_corr_50 = float(diagnostics.get("mean_noise_corr_50ms", 0.0))
        peak_ratio = float(diagnostics.get("psd_peak_ratio", 0.0))

        s_cv = _safe_tanh(max(0.0, isi_cv_mean) / 2.0)
        s_fano = _safe_tanh(max(0.0, fano_300) / 2.0)
        p_corr = max(0.0, noise_corr_50 - 0.05) / 0.05
        p_peak = max(0.0, peak_ratio - self.cfg.asynchrony_peak_ratio_threshold) / max(
            self.cfg.asynchrony_peak_ratio_scale,
            1e-9,
        )

        return float(
            self.cfg.asynchrony_cv_weight * s_cv
            + self.cfg.asynchrony_fano_weight * s_fano
            - self.cfg.asynchrony_corr_weight * p_corr
            - self.cfg.asynchrony_peak_ratio_weight * p_peak
        )

    def _score_regime(
        self,
        diagnostics: Dict[str, float],
        conductance_means: Dict[str, float],
        saturation_freqs: Dict[str, float],
    ) -> float:
        v_var_exc = float(diagnostics.get("voltage_rest_var_post_exc_mV2", 0.0))
        v_var_inh = float(diagnostics.get("voltage_rest_var_post_inh_mV2", 0.0))
        conductance_load = float(sum(float(conductance_means.get(name, 0.0)) for name in TRANSMITTERS))
        saturation_mean = float(
            np.mean([float(saturation_freqs.get(name, 0.0)) for name in TRANSMITTERS])
        )

        p_var_exc = max(0.0, v_var_exc) / max(self.cfg.regime_voltage_var_exc_scale_mV2, 1e-9)
        p_var_inh = max(0.0, v_var_inh) / max(self.cfg.regime_voltage_var_inh_scale_mV2, 1e-9)
        p_cond = math.log1p(max(0.0, conductance_load) / max(self.cfg.regime_conductance_load_scale, 1e-9))
        p_sat = max(0.0, saturation_mean)

        return float(
            1.0
            - (
                self.cfg.regime_voltage_var_exc_weight * p_var_exc
                + self.cfg.regime_voltage_var_inh_weight * p_var_inh
                + self.cfg.regime_conductance_load_weight * p_cond
                + self.cfg.regime_saturation_weight * p_sat
            )
        ) / 10.0

    def _hard_reject(self, diagnostics: Dict[str, float]) -> Tuple[bool, str]:
        rate_late = float(diagnostics.get("rate_late_Hz", 0.0))
        if rate_late < self.cfg.reject_rate_late_extreme_low_hz:
            return True, "late_rate_too_low"
        return False, ""

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

    def evaluate_params(
        self,
        params: Dict[str, float],
        trial_seed: int | None = None,
    ) -> Dict[str, object]:
        cfg = self.cfg
        trial_seed = cfg.base_seed if trial_seed is None else int(trial_seed)
        trial_rng = np.random.default_rng(trial_seed)

        syn_g_ampa_max = float(params["g_AMPA_max"])
        syn_g_nmda_max = float(params["g_NMDA_max"])
        syn_g_gabaa_max = float(params["g_GABA_A_max"])
        syn_g_gabab_max = float(params["g_GABA_B_max"])

        a_ampa = float(params["A_AMPA"])
        a_nmda = float(params["A_NMDA"])
        a_gabaa = float(params["A_GABA_A"])
        a_gabab = float(params["A_GABA_B"])

        inhib_g = float(params["inhibitory_scale_g"])
        ext_amp = float(params["external_amplitude"])
        recurrent_exc_sigma = float(params["recurrent_exc_lognorm_sigma"])
        inhibitory_nmda_weight = float(params["inhibitory_nmda_weight"])

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

            poisson = PoissonInput(cfg.n_neurons, rate=cfg.external_rate_hz, amplitude=ext_amp_vec, rng=rng)

            for _ in range(ext_on_steps):
                sim.step(spike_ext=poisson(cfg.dt_ms))

            conductance_sums = {name: 0.0 for name in TRANSMITTERS}
            saturation_sums = {name: 0.0 for name in TRANSMITTERS}
            for _ in range(ext_off_steps):
                sim.step()
                conductance_sums["AMPA"] += float(np.mean(syn.g_AMPA * syn.g_AMPA_max))
                conductance_sums["NMDA"] += float(np.mean(syn.g_NMDA * syn.g_NMDA_max))
                conductance_sums["GABA_A"] += float(np.mean(syn.g_GABA_A * syn.g_GABA_A_max))
                conductance_sums["GABA_B"] += float(np.mean(syn.g_GABA_B * syn.g_GABA_B_max))
                saturation_sums["AMPA"] += float(np.mean(syn.g_AMPA >= cfg.regime_saturation_threshold))
                saturation_sums["NMDA"] += float(np.mean(syn.g_NMDA >= cfg.regime_saturation_threshold))
                saturation_sums["GABA_A"] += float(np.mean(syn.g_GABA_A >= cfg.regime_saturation_threshold))
                saturation_sums["GABA_B"] += float(np.mean(syn.g_GABA_B >= cfg.regime_saturation_threshold))

            conductance_means = {
                name: float(conductance_sums[name] / ext_off_steps) if ext_off_steps > 0 else 0.0
                for name in TRANSMITTERS
            }
            saturation_freqs = {
                name: float(saturation_sums[name] / ext_off_steps) if ext_off_steps > 0 else 0.0
                for name in TRANSMITTERS
            }

            metrics: Dict[str, float] = {}
            diagnostics = self._compute_diagnostics(sim)
            metrics.update(diagnostics)
            reject, reject_reason = self._hard_reject(diagnostics)
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
            else:
                persistence_score = self._score_persistence(diagnostics)
                asynchrony_score = self._score_asynchrony(diagnostics)
                regime_score = self._score_regime(diagnostics, conductance_means, saturation_freqs)
                balance_score, balance_diagnostics = self._score_transmitter_balance(
                    params=params,
                    conductance_means=conductance_means,
                )

            metrics["score_persistence"] = float(persistence_score)
            metrics["score_asynchrony"] = float(asynchrony_score)
            metrics["score_regime"] = float(regime_score)
            metrics["score_balance"] = float(balance_score)
            metrics["hard_reject"] = float(1.0 if reject else 0.0)
            metrics["late_rate_soft_penalty"] = 0.0
            metrics["conductance_load_post"] = float(sum(conductance_means[name] for name in TRANSMITTERS))
            for name in TRANSMITTERS:
                metrics[f"conductance_mean_{name}"] = float(conductance_means[name])
                metrics[f"saturation_freq_{name}"] = float(saturation_freqs[name])
            metrics["saturation_freq_mean"] = float(np.mean(list(saturation_freqs.values())))
            metrics.update(balance_diagnostics)
            repeat_metrics.append(metrics)
            repeat_reject_reasons.append(reject_reason)

        aggregated_metrics = _aggregate_repeat_metrics(tuple(repeat_metrics))
        objectives = {
            "persistence": float(aggregated_metrics["score_persistence"]),
            "asynchrony": float(aggregated_metrics["score_asynchrony"]),
            "regime": float(aggregated_metrics["score_regime"]),
            "transmitter_balance": float(aggregated_metrics["score_balance"]),
        }

        diagnostics_out: Dict[str, object] = {
            "trial_seed": int(trial_seed),
            "n_repeats": int(cfg.n_repeats),
            "score_persistence": objectives["persistence"],
            "score_asynchrony": objectives["asynchrony"],
            "score_regime": objectives["regime"],
            "score_balance": objectives["transmitter_balance"],
            "network_density_target": float(cfg.n_out / cfg.n_neurons),
            "n_out": int(cfg.n_out),
        }
        diagnostics_out.update(self.template_topology_metrics)
        diagnostics_out.update(aggregated_metrics)
        for repeat_idx, metrics in enumerate(repeat_metrics):
            diagnostics_out[f"repeat_{repeat_idx}_seed"] = int(trial_seed + 7919 * repeat_idx)
            diagnostics_out[f"repeat_{repeat_idx}_score_persistence"] = float(metrics["score_persistence"])
            diagnostics_out[f"repeat_{repeat_idx}_score_asynchrony"] = float(metrics["score_asynchrony"])
            diagnostics_out[f"repeat_{repeat_idx}_score_regime"] = float(metrics["score_regime"])
            diagnostics_out[f"repeat_{repeat_idx}_score_balance"] = float(metrics["score_balance"])
            diagnostics_out[f"repeat_{repeat_idx}_reject_reason"] = repeat_reject_reasons[repeat_idx]

        return {
            "objectives": objectives,
            "diagnostics": diagnostics_out,
            "repeat_metrics": repeat_metrics,
        }

    def __call__(self, trial: optuna.Trial) -> Tuple[float, float, float, float]:
        cfg = self.cfg
        trial_seed = cfg.base_seed + 1009 * (trial.number + 1)
        params = suggest_params(trial)
        evaluation = self.evaluate_params(params, trial_seed=trial_seed)
        diagnostics = dict(evaluation["diagnostics"])
        objectives = dict(evaluation["objectives"])
        self._set_trial_attrs(trial, diagnostics)
        return (
            float(objectives["persistence"]),
            float(objectives["asynchrony"]),
            float(objectives["regime"]),
            float(objectives["transmitter_balance"]),
        )


def week4_reference_params() -> Dict[str, float]:
    return {
        "g_AMPA_max": 943.8767019547389 * 0.5,
        "g_NMDA_max": 1132.4613811288566 * 0.5,
        "g_GABA_A_max": 477.1128477169971 * 2.0,
        "g_GABA_B_max": 978.2386456218012 * 0.5,
        "A_AMPA": 0.1965658831686625 * 0.1,
        "A_NMDA": 0.011641539779921259 * 0.05,
        "A_GABA_A": 0.00010575509751513417 * 10.0,
        "A_GABA_B": 0.00010290509538049661,
        "inhibitory_scale_g": 1.2984752590298583 * 60.0,
        "external_amplitude": 2.44625509556019,
        "recurrent_exc_lognorm_sigma": 1.643570,
        "inhibitory_nmda_weight": 0.959685703507305 * 0.5,
    }


def week4_reference_config(n_repeats: int = 1, base_seed: int = 1234) -> SearchConfig:
    return SearchConfig(
        n_neurons=1000,
        n_excit=800,
        n_out=100,
        excitatory_type="ss4",
        inhibitory_type="b",
        dt_ms=0.1,
        ext_on_ms=500.0,
        total_ms=2000.0,
        external_rate_hz=0.45527031369449,
        n_repeats=n_repeats,
        delay_mean_exc_ms=10.0,
        delay_std_exc_ms=5.0,
        delay_mean_inh_ms=1.5,
        delay_std_inh_ms=0.3,
        base_seed=base_seed,
    )


def evaluate_parameter_set(
    params: Dict[str, float],
    cfg: SearchConfig | None = None,
    trial_seed: int | None = None,
) -> Dict[str, object]:
    objective = SSAIMultiObjective(cfg or SearchConfig())
    return objective.evaluate_params(params=params, trial_seed=trial_seed)


def evaluate_week4_reference(
    n_repeats: int = 1,
    base_seed: int = 1234,
    trial_seed: int | None = None,
) -> Dict[str, object]:
    cfg = week4_reference_config(n_repeats=n_repeats, base_seed=base_seed)
    params = week4_reference_params()
    seed = base_seed if trial_seed is None else int(trial_seed)
    return evaluate_parameter_set(params=params, cfg=cfg, trial_seed=seed)
