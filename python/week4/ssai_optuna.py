import argparse
import json
import os
from dataclasses import dataclass

import networkx as nx
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from src.connectome import Connectome
from src.external_inputs import PoissonInput
from src.network_weight_distributor import assign_lognormal_weights_for_ntype
from src.neuron_population import NeuronPopulation
from src.overhead import Simulation

PARAM_SPECS = [
    ("g_AMPA_max", 10.0, 1000.0, True),
    ("g_NMDA_max", 50.0, 1500.0, True),
    ("g_GABA_A_max", 10.0, 1000.0, True),
    ("g_GABA_B_max", 1.0, 1000.0, True),
    ("A_AMPA", 0.0001, 0.2, True),
    ("A_NMDA", 0.0001, 0.2, True),
    ("A_GABA_A", 0.0001, 0.2, True),
    ("A_GABA_B", 0.0001, 0.2, True),
    ("inhibitory_scale_g", 1.0, 50.0, False),
    ("v_ext", 0.01, 0.5, False),
    ("external_amplitude", 1.0, 1000.0, False),
    ("recurrent_exc_lognorm_sigma", 0.1, 2.0, False),
    ("inhibitory_nmda_weight", 0.0, 1.0, False),
]


def suggest_params(trial: optuna.Trial) -> dict:
    params = {}
    for name, low, high, use_log in PARAM_SPECS:
        params[name] = trial.suggest_float(name, low, high, log=use_log)
    return params


@dataclass
class SearchConfig:
    n_neurons: int = 1000
    n_excit: int = 800
    n_out: int = 150
    excitatory_type: str = "ss4"
    inhibitory_type: str = "b"
    dt_ms: float = 0.1
    ext_on_ms: float = 750.0
    total_ms: float = 1750.0
    delay_mean: float = 1.5
    delay_std: float = 0.3
    recurrent_exc_mu: float = 0.0
    recurrent_exc_sigma: float = 1.0
    recurrent_exc_wmax: float = 100.0
    base_seed: int = 1234


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


class SSAIObjective:
    def __init__(self, cfg: SearchConfig):
        self.cfg = cfg
        self.template_graph = self._build_template_graph(seed=cfg.base_seed)

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
                delays = np.maximum(0.1, rng.normal(1.5, 0.3, size=self.cfg.n_out))
            else:
                delays = np.maximum(
                    0.1,
                    rng.normal(self.cfg.delay_mean, self.cfg.delay_std, size=self.cfg.n_out),
                )

            for target, delay in zip(targets, delays):
                g.add_edge(i, int(target), weight=1.0, distance=float(delay))

        return g

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
        vs = rng.uniform(-100.0, -70.0, size=n_neurons)
        us = rng.uniform(0.0, 400.0, size=n_neurons)
        spikes = np.zeros(n_neurons, dtype=bool)
        ts = np.zeros_like(spikes)
        return vs, us, spikes, ts

    def _score_trial(self, sim: Simulation):
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
            "score": float(score),
            "ISI_CV_mean": isi_cv_mean,
            "Fano_median_300ms": fano_300,
            "mean_noise_corr_50ms": noise_corr_50,
            "pop_spec_entropy": pop_entropy,
            "rate_post_Hz": rate_post,
            "rate_late_Hz": rate_late,
            "psd_peak_ratio": float(peak_ratio),
            "mean_voltage_post_mV": float(v_mean),
            "p_corr": float(p_corr),
            "p_peak": float(p_peak),
            "p_vhigh": float(p_vhigh),
        }
        return float(score), diagnostics

    def __call__(self, trial: optuna.Trial) -> float:
        cfg = self.cfg
        trial_seed = cfg.base_seed + 1009 * (trial.number + 1)
        rng = np.random.default_rng(trial_seed)
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

        for u, v, data in graph.edges(data=True):
            data["weight"] = float(inhib_g if u >= cfg.n_excit else 1.0)

        assign_lognormal_weights_for_ntype(
            graph,
            cfg.excitatory_type,
            mu=cfg.recurrent_exc_mu,
            sigma=recurrent_exc_sigma,
            w_max=cfg.recurrent_exc_wmax,
            rng=rng,
        )

        connectome, pop, nmda_weight = self._build_connectome(
            graph, inhibitory_nmda_weight=inhibitory_nmda_weight
        )
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

        ext_amp_vec = np.full(cfg.n_neurons, ext_amp, dtype=float)
        ext_amp_vec[pop.inhibitory_mask.astype(bool)] = 0.0

        poisson = PoissonInput(cfg.n_neurons, rate=v_ext, amplitude=ext_amp_vec, rng=rng)

        ext_on_steps = int(round(cfg.ext_on_ms / cfg.dt_ms))
        total_steps = int(round(cfg.total_ms / cfg.dt_ms))
        ext_off_steps = max(0, total_steps - ext_on_steps)

        for _ in range(ext_on_steps):
            sim.step(spike_ext=poisson(cfg.dt_ms))

        trial.report(float(sim.stats.compute_metrics(cfg.dt_ms, t_start_ms=cfg.ext_on_ms, t_stop_ms=cfg.ext_on_ms + 100.0).get("rate_mean_Hz", 0.0)), step=1)
        if trial.should_prune():
            raise optuna.TrialPruned()

        for _ in range(ext_off_steps):
            sim.step()

        score, diagnostics = self._score_trial(sim)

        for k, v in diagnostics.items():
            trial.set_user_attr(k, float(v))

        return score


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for self-sustained asynchronous activity.")
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds.")
    parser.add_argument("--study-name", type=str, default="ssai_optuna")
    parser.add_argument("--storage", type=str, default="sqlite:///ssai_optuna.db")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--show-best", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    cfg = SearchConfig(base_seed=args.seed)
    objective = SSAIObjective(cfg)

    sampler = TPESampler(seed=args.seed, multivariate=True)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=1)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        timeout=args.timeout,
        gc_after_trial=True,
    )

    best = {
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "best_user_attrs": study.best_trial.user_attrs,
        "study_name": study.study_name,
        "n_trials": len(study.trials),
    }
    print(json.dumps(best, indent=2, sort_keys=True))

    if args.show_best:
        print("\nBest trial summary")
        print(f"number={study.best_trial.number}")
        print(f"value={study.best_trial.value:.6f}")
        print("Best parameters")
        for name, _, _, _ in PARAM_SPECS:
            print(f"{name}: {study.best_trial.params.get(name)}")


if __name__ == "__main__":
    main()
