import argparse
import csv
import json
import math
import sys
from dataclasses import fields
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from Midterm.ssai_nsga2_objective import OBJECTIVE_NAMES, PARAM_SPECS, SSAIMultiObjective, SearchConfig
except ModuleNotFoundError:
    from ssai_nsga2_objective import OBJECTIVE_NAMES, PARAM_SPECS, SSAIMultiObjective, SearchConfig

from src.external_inputs import PoissonInput
from src.utilities import bin_counts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster Pareto-front solutions from an NSGA-II run and regenerate representative "
            "family plots (raster, voltage history, single-neuron voltages, synaptic traces)."
        )
    )
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory containing config.json and pareto_front.json")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory; defaults to <run-dir>/family_analysis")
    parser.add_argument("--n-clusters", type=int, default=4, help="Number of solution families to form")
    parser.add_argument(
        "--cluster-features",
        type=str,
        default="params+objectives",
        choices=("params", "params+objectives"),
        help="Feature set used for clustering",
    )
    parser.add_argument(
        "--embedding-method",
        type=str,
        default="pca",
        choices=("pca", "umap"),
        help="2D visualization method for the clustering space",
    )
    parser.add_argument("--max-families", type=int, default=None, help="Optional cap on number of plotted families")
    parser.add_argument(
        "--trial-numbers",
        type=int,
        nargs="+",
        default=None,
        help="Specific Pareto trial numbers to render. If set, these override family representative selection.",
    )
    parser.add_argument(
        "--top-neurons",
        type=int,
        default=4,
        help="Retained for compatibility; fixed neuron index sampling is used for detailed plots.",
    )
    parser.add_argument("--plot-start-ms", type=float, default=0.0)
    parser.add_argument("--plot-stop-ms", type=float, default=None)
    return parser


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _search_config_from_run(run_dir: Path) -> SearchConfig:
    cfg_json = _load_json(run_dir / "config.json")
    cfg_dict = dict(cfg_json["search_config"])
    valid_keys = {field.name for field in fields(SearchConfig)}
    filtered_cfg = {key: value for key, value in cfg_dict.items() if key in valid_keys}
    return SearchConfig(**filtered_cfg)


def _load_pareto_trials(run_dir: Path) -> List[Dict[str, object]]:
    return list(_load_json(run_dir / "pareto_front.json"))


def _filter_nonnegative_trials(trials: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    keep = []
    for trial in trials:
        values = trial.get("values", None)
        if values is None:
            continue
        if all(float(v) >= 0.0 for v in values):
            keep.append(trial)
    return keep


def _transform_param_to_unit(name: str, value: float) -> float:
    for spec_name, low, high, use_log in PARAM_SPECS:
        if spec_name != name:
            continue
        if use_log:
            lo = math.log(low)
            hi = math.log(high)
            return float((math.log(max(float(value), low)) - lo) / (hi - lo))
        return float((float(value) - low) / (high - low))
    raise KeyError(f"Unknown parameter '{name}'.")


def _extract_feature_matrix(trials: Sequence[Dict[str, object]], mode: str) -> Tuple[np.ndarray, List[str]]:
    rows = []
    for trial in tqdm(trials, desc="Extracting clustering features", unit="trial"):
        params = trial["params"]
        row = [_transform_param_to_unit(name, params[name]) for name, _, _, _ in PARAM_SPECS]
        if mode == "params+objectives":
            values = [float(v) for v in trial["values"]]
            row.extend(values)
        rows.append(row)

    X = np.asarray(rows, dtype=float)
    names = [f"param.{name}" for name, _, _, _ in PARAM_SPECS]
    if mode == "params+objectives":
        names += [f"objective.{name}" for name in OBJECTIVE_NAMES]
    return X, names


def _standardize(X: np.ndarray) -> np.ndarray:
    if X.size == 0:
        return X
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True)
    sigma[sigma < 1e-12] = 1.0
    return (X - mu) / sigma


def _kmeans(X: np.ndarray, n_clusters: int, seed: int = 1234, max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_samples = X.shape[0]
    n_clusters = max(1, min(int(n_clusters), n_samples))
    initial_idx = rng.choice(n_samples, size=n_clusters, replace=False)
    centroids = X[initial_idx].copy()
    labels = np.zeros(n_samples, dtype=int)

    for _ in range(max_iter):
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for k in range(n_clusters):
            mask = labels == k
            if np.any(mask):
                centroids[k] = X[mask].mean(axis=0)
            else:
                centroids[k] = X[rng.integers(0, n_samples)]
    return labels, centroids


def _pca_2d(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if X.ndim != 2:
        raise ValueError("PCA input must be 2D.")
    if X.shape[0] == 0:
        return np.zeros((0, 2), dtype=float), np.zeros(2, dtype=float)
    if X.shape[1] == 0:
        return np.zeros((X.shape[0], 2), dtype=float), np.zeros(2, dtype=float)

    Xc = X - X.mean(axis=0, keepdims=True)
    _, svals, vt = np.linalg.svd(Xc, full_matrices=False)
    n_components = min(2, vt.shape[0])
    components = vt[:n_components]
    coords = Xc @ components.T
    if n_components < 2:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 2 - n_components), dtype=float)])

    var = (svals ** 2) / max(1, X.shape[0] - 1)
    total_var = float(np.sum(var))
    explained = np.zeros(2, dtype=float)
    if total_var > 0.0:
        explained[:n_components] = var[:n_components] / total_var
    return coords[:, :2], explained


def _umap_2d(X: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import umap
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "UMAP visualization requires the 'umap-learn' package. "
            "Install it in the active environment or use --embedding-method pca."
        ) from exc

    if X.ndim != 2:
        raise ValueError("UMAP input must be 2D.")
    if X.shape[0] == 0:
        return np.zeros((0, 2), dtype=float), np.zeros(2, dtype=float)
    if X.shape[0] == 1:
        return np.zeros((1, 2), dtype=float), np.zeros(2, dtype=float)

    n_neighbors = max(2, min(15, X.shape[0] - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=int(seed),
    )
    coords = reducer.fit_transform(X)
    return np.asarray(coords, dtype=float), np.zeros(2, dtype=float)


def _plot_embedding_map(
    coords: np.ndarray,
    labels: np.ndarray,
    reps: Dict[int, int],
    trials: Sequence[Dict[str, object]],
    explained: np.ndarray,
    method: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    unique_labels = sorted(set(int(x) for x in labels.tolist()))
    cmap = plt.get_cmap("tab10")

    for plot_idx, cluster_id in enumerate(unique_labels):
        mask = labels == cluster_id
        pts = coords[mask]
        color = cmap(plot_idx % 10)
        ax.scatter(pts[:, 0], pts[:, 1], s=50, alpha=0.75, color=color, label=f"family {cluster_id}")

    for cluster_id, rep_idx in sorted(reps.items()):
        x, y = coords[rep_idx]
        trial_number = int(trials[rep_idx]["number"])
        ax.scatter([x], [y], s=180, facecolors="none", edgecolors="black", linewidths=1.8)
        ax.text(x, y, str(trial_number), fontsize=9, ha="left", va="bottom")

    if method == "pca":
        ax.set_title("PCA Map of Clustering Space")
        ax.set_xlabel(f"PC1 ({100.0 * explained[0]:.1f}% var)")
        ax.set_ylabel(f"PC2 ({100.0 * explained[1]:.1f}% var)")
    else:
        ax.set_title("UMAP Map of Clustering Space")
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _normalized_objective_sums(trials: Sequence[Dict[str, object]]) -> np.ndarray:
    if not trials:
        return np.zeros(0, dtype=float)
    values = np.asarray([[float(v) for v in trial["values"]] for trial in trials], dtype=float)
    mins = values.min(axis=0, keepdims=True)
    maxs = values.max(axis=0, keepdims=True)
    spans = maxs - mins
    spans[spans < 1e-12] = 1.0
    normalized = (values - mins) / spans
    return normalized.sum(axis=1)


def _representatives_by_objective_sum(
    trials: Sequence[Dict[str, object]],
    labels: np.ndarray,
) -> Dict[int, int]:
    reps: Dict[int, int] = {}
    objective_sums = _normalized_objective_sums(trials)
    for cluster_id in sorted(set(int(x) for x in labels.tolist())):
        idx = np.flatnonzero(labels == cluster_id)
        if idx.size == 0:
            continue
        best_local = idx[np.argmax(objective_sums[idx])]
        reps[cluster_id] = int(best_local)
    return reps


def _prepare_simulation(objective: SSAIMultiObjective, params: Dict[str, float], repeat_seed: int):
    cfg = objective.cfg
    graph = objective.template_graph.copy()
    trial_rng = np.random.default_rng(int(repeat_seed))

    inhib_g = float(params["inhibitory_scale_g"])
    recurrent_exc_sigma = float(params["recurrent_exc_lognorm_sigma"])
    inhibitory_nmda_weight = float(params["inhibitory_nmda_weight"])

    for u, _, data in graph.edges(data=True):
        data["weight"] = float(inhib_g if u >= cfg.n_excit else 1.0)

    from src.network_weight_distributor import assign_lognormal_weights_for_ntype

    assign_lognormal_weights_for_ntype(
        graph,
        cfg.excitatory_type,
        mu=cfg.recurrent_exc_mu,
        sigma=recurrent_exc_sigma,
        w_max=cfg.recurrent_exc_wmax,
        rng=trial_rng,
    )

    connectome, pop, nmda_weight = objective._build_connectome(graph, inhibitory_nmda_weight=inhibitory_nmda_weight)
    state0 = objective._initialize_state(cfg.n_neurons, trial_rng)

    from src.overhead import Simulation

    sim = Simulation(
        connectome,
        cfg.dt_ms,
        stepper_type="euler_det",
        state0=state0,
        enable_plasticity=False,
        synapse_type="standard",
        synapse_kwargs={"LT_scale": 1.0, "NMDA_weight": nmda_weight},
        enable_debug_logger=True,
    )

    syn = sim.synapse_dynamics
    syn.g_AMPA_max = float(params["g_AMPA_max"])
    syn.g_NMDA_max = float(params["g_NMDA_max"])
    syn.g_GABA_A_max = float(params["g_GABA_A_max"])
    syn.g_GABA_B_max = float(params["g_GABA_B_max"])
    syn.A_AMPA = float(params["A_AMPA"])
    syn.A_NMDA = float(params["A_NMDA"])
    syn.A_GABA_A = float(params["A_GABA_A"])
    syn.A_GABA_B = float(params["A_GABA_B"])

    ext_amp_vec = np.full(cfg.n_neurons, float(params["external_amplitude"]), dtype=float)
    ext_amp_vec[pop.inhibitory_mask.astype(bool)] = 0.0
    poisson = PoissonInput(cfg.n_neurons, rate=float(cfg.external_rate_hz), amplitude=ext_amp_vec, rng=trial_rng)

    ext_on_steps = int(round(cfg.ext_on_ms / cfg.dt_ms))
    total_steps = int(round(cfg.total_ms / cfg.dt_ms))
    ext_off_steps = max(0, total_steps - ext_on_steps)

    for _ in range(ext_on_steps):
        sim.step(spike_ext=poisson(cfg.dt_ms))
    for _ in range(ext_off_steps):
        sim.step()

    return sim


def _choose_repeat_seed(trial: Dict[str, object], cfg: SearchConfig) -> int:
    attrs = dict(trial.get("user_attrs", {}))
    best_seed = int(attrs.get("trial_seed", cfg.base_seed))
    best_score = None
    for idx in range(cfg.n_repeats):
        seed_key = f"repeat_{idx}_seed"
        objective_keys = [
            f"repeat_{idx}_score_persistence",
            f"repeat_{idx}_score_asynchrony",
            f"repeat_{idx}_score_regime",
            f"repeat_{idx}_score_balance",
        ]
        if all(key in attrs for key in objective_keys) and seed_key in attrs:
            score = float(sum(float(attrs[key]) for key in objective_keys))
            if best_score is None or score > best_score:
                best_score = score
                best_seed = int(attrs[seed_key])
    return best_seed


def _select_neurons(sim, top_neurons: int) -> List[int]:
    n_neurons = int(sim.connectome.neuron_population.n_neurons)
    if n_neurons <= 0:
        return []
    candidates = [
        0,
        n_neurons // 3,
        (2 * n_neurons) // 3,
        n_neurons - 1,
    ]
    selected = []
    seen = set()
    for idx in candidates:
        idx = max(0, min(n_neurons - 1, int(idx)))
        if idx not in seen:
            selected.append(idx)
            seen.add(idx)
    return selected


def _time_mask(times_ms: np.ndarray, t_start_ms: float, t_stop_ms: float | None) -> np.ndarray:
    mask = times_ms >= t_start_ms
    if t_stop_ms is not None:
        mask &= times_ms <= t_stop_ms
    return mask


def _fano_median_for_mask(S: np.ndarray, neuron_mask: np.ndarray, dt_ms: float, bin_ms: float) -> float:
    if S.size == 0 or not np.any(neuron_mask):
        return 0.0
    S_sel = S[neuron_mask]
    if S_sel.size == 0:
        return 0.0
    bin_steps = max(1, int(round(float(bin_ms) / float(dt_ms))))
    counts = bin_counts(S_sel, bin_steps=bin_steps)
    if counts.shape[1] < 2:
        return 0.0
    mu = counts.mean(axis=1)
    var = counts.var(axis=1, ddof=1)
    fanos = np.where(mu > 0, var / mu, np.nan)
    active_mask = S_sel.sum(axis=1) > 0
    valid = np.isfinite(fanos) & active_mask
    return float(np.median(fanos[valid])) if np.any(valid) else 0.0


def _analyzed_metrics(sim, t_start_ms: float, t_stop_ms: float | None) -> Dict[str, float]:
    stats = sim.stats.compute_metrics(
        sim.dt,
        t_start_ms=t_start_ms,
        t_stop_ms=t_stop_ms,
    )
    times = np.asarray(sim.stats.ts, dtype=float)
    time_mask = _time_mask(times, t_start_ms, t_stop_ms)
    spikes = sim.stats.spikes_bool()
    if spikes.size == 0:
        return {}
    spikes = spikes[:, time_mask]
    inhib_mask = np.asarray(sim.stats.inhibitory_mask, dtype=bool)
    exc_mask = ~inhib_mask
    voltages = sim.stats.voltages()
    mean_voltage_exc = 0.0
    mean_voltage_inh = 0.0
    if voltages.size > 0:
        voltages = voltages[:, time_mask]
        if voltages.size > 0:
            if np.any(exc_mask):
                mean_voltage_exc = float(np.mean(voltages[exc_mask]))
            if np.any(inhib_mask):
                mean_voltage_inh = float(np.mean(voltages[inhib_mask]))

    metrics = {
        "analysis_t_start_ms": float(t_start_ms),
        "analysis_t_stop_ms": float(t_stop_ms if t_stop_ms is not None else times[-1]),
        "rate_mean_Hz_E": float(stats.get("rate_mean_Hz_E", 0.0)),
        "rate_mean_Hz_I": float(stats.get("rate_mean_Hz_I", 0.0)),
        "ISI_CV_mean_E": float(stats.get("ISI_CV_mean_E", 0.0)),
        "ISI_CV_mean_I": float(stats.get("ISI_CV_mean_I", 0.0)),
        "mean_noise_corr_50ms": float(stats.get("mean_noise_corr_50ms", 0.0)),
        "psd_peak_ratio": float(stats.get("psd_peak_ratio", 0.0)),
        "mean_voltage_mV_E": mean_voltage_exc,
        "mean_voltage_mV_I": mean_voltage_inh,
    }
    for bin_ms in (2, 10, 50, 100, 300, 500, 1000):
        metrics[f"Fano_median_{bin_ms}ms_E"] = _fano_median_for_mask(spikes, exc_mask, sim.dt, float(bin_ms))
        metrics[f"Fano_median_{bin_ms}ms_I"] = _fano_median_for_mask(spikes, inhib_mask, sim.dt, float(bin_ms))
    return metrics


def _print_metric_summary(family_label: str, metrics: Dict[str, float]) -> None:
    bins = (10, 50, 100, 300, 500, 1000)
    fano_parts = []
    for bin_ms in bins:
        fano_parts.append(
            f"{bin_ms}ms(E={metrics.get(f'Fano_median_{bin_ms}ms_E', 0.0):.3f},"
            f"I={metrics.get(f'Fano_median_{bin_ms}ms_I', 0.0):.3f})"
        )
    print(
        f"{family_label}: "
        f"rate_Hz(E={metrics.get('rate_mean_Hz_E', 0.0):.3f}, I={metrics.get('rate_mean_Hz_I', 0.0):.3f}) | "
        f"ISI_CV(E={metrics.get('ISI_CV_mean_E', 0.0):.3f}, I={metrics.get('ISI_CV_mean_I', 0.0):.3f}) | "
        f"noise_corr_50ms={metrics.get('mean_noise_corr_50ms', 0.0):.3f} | "
        f"psd_peak_ratio={metrics.get('psd_peak_ratio', 0.0):.3f} | "
        f"Vmean_mV(E={metrics.get('mean_voltage_mV_E', 0.0):.3f}, I={metrics.get('mean_voltage_mV_I', 0.0):.3f}) | "
        f"Fano " + ", ".join(fano_parts)
    )


def _plot_type_voltage_history(sim, out_path: Path, t_start_ms: float, t_stop_ms: float | None) -> None:
    Vs = np.asarray(sim.stats.Vs)
    times = np.asarray(sim.stats.ts, dtype=float)
    mask = _time_mask(times, t_start_ms, t_stop_ms)
    Vs = Vs[mask]
    times = times[mask]
    pop = sim.connectome.neuron_population
    types = pop.neuron_types

    fig, axes = plt.subplots(len(types), 1, figsize=(12, 3 * len(types)), sharex=True)
    if len(types) == 1:
        axes = [axes]
    for ax, type_name in zip(axes, types):
        idx = pop.get_neurons_from_type(type_name)
        if len(idx) == 0:
            continue
        ax.plot(times, Vs[:, idx], alpha=0.08, linewidth=0.5)
        ax.plot(times, Vs[:, idx].mean(axis=1), color="black", linewidth=2)
        ax.set_title(f"Voltage history: {type_name}")
        ax.set_ylabel("V (mV)")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Time (ms)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_neuron_voltage_history(sim, neuron_indices: Sequence[int], out_path: Path, t_start_ms: float, t_stop_ms: float | None) -> None:
    Vs = np.asarray(sim.stats.Vs)
    times = np.asarray(sim.stats.ts, dtype=float)
    mask = _time_mask(times, t_start_ms, t_stop_ms)
    Vs = Vs[mask]
    times = times[mask]

    fig, axes = plt.subplots(len(neuron_indices), 1, figsize=(12, max(2.5, 2.5 * len(neuron_indices))), sharex=True)
    if len(neuron_indices) == 1:
        axes = [axes]
    for ax, neuron_idx in zip(axes, neuron_indices):
        ax.plot(times, Vs[:, neuron_idx], linewidth=1.0)
        ax.set_ylabel(f"n{neuron_idx}\nV")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Time (ms)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_synaptic_traces(sim, neuron_indices: Sequence[int], out_path: Path, t_start_ms: float, t_stop_ms: float | None) -> None:
    times = np.asarray(sim.stats.ts, dtype=float)
    mask = _time_mask(times, t_start_ms, t_stop_ms)
    s_ampa = np.asarray(sim.debug_logger.s_ampa)[mask]
    s_nmda = np.asarray(sim.debug_logger.s_nmda)[mask]
    s_gaba_a = np.asarray(sim.debug_logger.s_gaba_a)[mask]
    s_gaba_b = np.asarray(sim.debug_logger.s_gaba_b)[mask]
    times = times[mask]

    fig, axes = plt.subplots(len(neuron_indices), 1, figsize=(12, max(3.0, 3.0 * len(neuron_indices))), sharex=True)
    if len(neuron_indices) == 1:
        axes = [axes]
    for ax, neuron_idx in zip(axes, neuron_indices):
        ax.plot(times, s_ampa[:, neuron_idx], label="AMPA", linewidth=1.0)
        ax.plot(times, s_nmda[:, neuron_idx], label="NMDA", linewidth=1.0)
        ax.plot(times, s_gaba_a[:, neuron_idx], label="GABA_A", linewidth=1.0)
        ax.plot(times, s_gaba_b[:, neuron_idx], label="GABA_B", linewidth=1.0)
        ax.set_ylabel(f"n{neuron_idx}\ng")
        ax.grid(alpha=0.2)
    axes[0].legend(loc="upper right", ncol=4)
    axes[-1].set_xlabel("Time (ms)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_raster(sim, out_path: Path, t_start_ms: float, t_stop_ms: float | None, title: str) -> None:
    sim.plot_spike_raster(
        dt_ms=sim.dt,
        t_start_ms=t_start_ms,
        t_stop_ms=t_stop_ms,
        figsize=(12, 6),
        title=title,
        save_path=str(out_path),
    )


def _raster_points(sim, t_start_ms: float, t_stop_ms: float | None) -> Tuple[np.ndarray, np.ndarray]:
    spikes = sim.stats.spikes_bool()
    times = np.asarray(sim.stats.ts, dtype=float)
    if spikes.size == 0 or times.size == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=int)
    mask = _time_mask(times, t_start_ms, t_stop_ms)
    if not np.any(mask):
        return np.zeros(0, dtype=float), np.zeros(0, dtype=int)
    spikes = spikes[:, mask]
    times = times[mask]
    neuron_idx, time_idx = np.nonzero(spikes)
    return times[time_idx], neuron_idx


def _plot_combined_family_rasters(
    raster_panels: Sequence[Dict[str, object]],
    out_path: Path,
    t_start_ms: float,
    t_stop_ms: float | None,
) -> None:
    if not raster_panels:
        return
    fig, axes = plt.subplots(len(raster_panels), 1, figsize=(12, max(3.0, 2.4 * len(raster_panels))), sharex=True)
    if len(raster_panels) == 1:
        axes = [axes]
    for ax, panel in zip(axes, raster_panels):
        spike_times = np.asarray(panel["spike_times_ms"], dtype=float)
        neuron_idx = np.asarray(panel["neuron_indices"], dtype=int)
        inhibitory_mask = np.asarray(panel["inhibitory_mask"], dtype=bool)
        if spike_times.size > 0 and inhibitory_mask.size > 0:
            spike_is_inhib = inhibitory_mask[neuron_idx]
            exc_mask = ~spike_is_inhib
            if np.any(exc_mask):
                ax.scatter(
                    spike_times[exc_mask],
                    neuron_idx[exc_mask],
                    s=2,
                    c="tab:blue",
                    alpha=0.75,
                    linewidths=0,
                    label="Excitatory",
                )
            if np.any(spike_is_inhib):
                ax.scatter(
                    spike_times[spike_is_inhib],
                    neuron_idx[spike_is_inhib],
                    s=2,
                    c="tab:orange",
                    alpha=0.75,
                    linewidths=0,
                    label="Inhibitory",
                )
        ax.set_ylabel(f"F{int(panel['cluster_id'])}\nNeuron")
        ax.set_title(str(panel["title"]))
        ax.grid(alpha=0.15)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(loc="upper right", ncol=2)
    axes[-1].set_xlabel("Time (ms)")
    axes[-1].set_xlim(left=float(t_start_ms), right=float(t_stop_ms) if t_stop_ms is not None else None)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _summary_stats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return {
            "min": float("nan"),
            "max": float("nan"),
            "span": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
        }
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "span": float(np.max(arr) - np.min(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
    }


def _family_summary_rows(
    trials: Sequence[Dict[str, object]],
    labels: np.ndarray,
    member_metric_rows: Sequence[Dict[str, float]],
) -> List[Dict[str, object]]:
    trial_map = {int(trial["number"]): trial for trial in trials}
    metrics_by_trial = {int(row["trial_number"]): row for row in member_metric_rows}
    summary_rows: List[Dict[str, object]] = []
    metric_names = (
        "ISI_CV_mean_E",
        "ISI_CV_mean_I",
        "Fano_median_300ms_E",
        "Fano_median_300ms_I",
        "mean_noise_corr_50ms",
        "rate_mean_Hz_E",
        "rate_mean_Hz_I",
        "psd_peak_ratio",
    )

    for cluster_id in sorted(set(int(x) for x in labels.tolist())):
        member_idx = np.flatnonzero(labels == cluster_id)
        member_trials = [trials[i] for i in member_idx]
        row: Dict[str, object] = {
            "cluster_id": int(cluster_id),
            "n_members": int(len(member_trials)),
        }
        for name, _, _, _ in PARAM_SPECS:
            stats = _summary_stats([float(trial["params"][name]) for trial in member_trials])
            for stat_name, stat_value in stats.items():
                row[f"param.{name}.{stat_name}"] = stat_value
        for obj_idx, obj_name in enumerate(OBJECTIVE_NAMES):
            stats = _summary_stats([float(trial["values"][obj_idx]) for trial in member_trials])
            for stat_name, stat_value in stats.items():
                row[f"objective.{obj_name}.{stat_name}"] = stat_value

        member_metric_values = [metrics_by_trial[int(trial["number"])] for trial in member_trials if int(trial["number"]) in metrics_by_trial]
        row["n_members_with_metrics"] = int(len(member_metric_values))
        for metric_name in metric_names:
            stats = _summary_stats([float(metric_row.get(metric_name, float("nan"))) for metric_row in member_metric_values])
            for stat_name, stat_value in stats.items():
                row[f"metric.{metric_name}.{stat_name}"] = stat_value
        summary_rows.append(row)
    return summary_rows


def _cluster_summary(trials: Sequence[Dict[str, object]], labels: np.ndarray, reps: Dict[int, int]) -> Dict[str, object]:
    objective_sums = _normalized_objective_sums(trials)
    out = {"n_families": int(len(reps)), "families": []}
    for cluster_id, rep_idx in sorted(reps.items()):
        member_idx = np.flatnonzero(labels == cluster_id)
        rep_trial = trials[rep_idx]
        values = [float(v) for v in rep_trial["values"]]
        out["families"].append(
            {
                "cluster_id": int(cluster_id),
                "size": int(member_idx.size),
                "representative_trial_number": int(rep_trial["number"]),
                "representative_normalized_objective_sum": float(objective_sums[rep_idx]),
                "representative_objectives": {name: values[i] for i, name in enumerate(OBJECTIVE_NAMES)},
                "member_trial_numbers": [int(trials[i]["number"]) for i in member_idx],
            }
        )
    return out


def _trial_index_by_number(trials: Sequence[Dict[str, object]]) -> Dict[int, int]:
    return {int(trial["number"]): idx for idx, trial in enumerate(trials)}


def main() -> None:
    args = _build_parser().parse_args()
    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (run_dir / "family_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _search_config_from_run(run_dir)
    all_trials = _load_pareto_trials(run_dir)
    if not all_trials:
        raise ValueError(f"No Pareto trials found in {run_dir}.")
    trials = list(all_trials)
    nonnegative_trials = _filter_nonnegative_trials(all_trials)

    X_raw, feature_names = _extract_feature_matrix(trials, args.cluster_features)
    X = _standardize(X_raw)
    labels, centroids = _kmeans(X, args.n_clusters, seed=cfg.base_seed)
    reps = _representatives_by_objective_sum(trials, labels)
    if args.embedding_method == "pca":
        embedding_coords, embedding_explained = _pca_2d(X)
    else:
        embedding_coords, embedding_explained = _umap_2d(X, seed=cfg.base_seed)

    summary = _cluster_summary(trials, labels, reps)
    summary["n_trials_input"] = int(len(all_trials))
    summary["n_trials_nonnegative"] = int(len(nonnegative_trials))
    summary["embedding_method"] = args.embedding_method
    with (out_dir / "cluster_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    _plot_embedding_map(
        embedding_coords,
        labels,
        reps,
        trials,
        embedding_explained,
        args.embedding_method,
        out_dir / f"clustering_{args.embedding_method}_map.png",
    )

    assignments = []
    for idx, trial in enumerate(tqdm(trials, desc="Preparing family assignments", unit="trial")):
        row = {
            "trial_number": int(trial["number"]),
            "cluster_id": int(labels[idx]),
            f"{args.embedding_method}_x": float(embedding_coords[idx, 0]),
            f"{args.embedding_method}_y": float(embedding_coords[idx, 1]),
        }
        row.update({f"objective.{name}": float(trial["values"][i]) for i, name in enumerate(OBJECTIVE_NAMES)})
        row.update({f"param.{k}": v for k, v in trial["params"].items()})
        assignments.append(row)

    fieldnames = sorted({k for row in assignments for k in row.keys()})
    with (out_dir / "family_assignments.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(assignments)

    objective = SSAIMultiObjective(cfg)
    representative_index_map = dict(reps)
    member_metric_rows = []
    for cluster_id in tqdm(sorted(set(int(x) for x in labels.tolist())), desc="Analyzing all family members", unit="family"):
        member_idx = np.flatnonzero(labels == cluster_id)
        for idx in member_idx:
            trial = trials[int(idx)]
            repeat_seed = _choose_repeat_seed(trial, cfg)
            sim = _prepare_simulation(objective, trial["params"], repeat_seed=repeat_seed)
            metric_t_start_ms = max(float(cfg.ext_on_ms), float(args.plot_start_ms))
            metric_t_stop_ms = args.plot_stop_ms
            analyzed_metrics = _analyzed_metrics(sim, metric_t_start_ms, metric_t_stop_ms)
            row = {
                "cluster_id": int(cluster_id),
                "trial_number": int(trial["number"]),
                "repeat_seed": int(repeat_seed),
                "is_representative": int(int(idx) == int(representative_index_map.get(cluster_id, -1))),
            }
            row.update(analyzed_metrics)
            member_metric_rows.append(row)

    if member_metric_rows:
        member_metric_fields = sorted({key for row in member_metric_rows for key in row.keys()})
        with (out_dir / "family_member_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=member_metric_fields)
            writer.writeheader()
            writer.writerows(member_metric_rows)

        family_summary_rows = _family_summary_rows(trials, labels, member_metric_rows)
        family_summary_fields = sorted({key for row in family_summary_rows for key in row.keys()})
        with (out_dir / "family_summary_statistics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=family_summary_fields)
            writer.writeheader()
            writer.writerows(family_summary_rows)

    if args.trial_numbers:
        index_map = _trial_index_by_number(trials)
        missing = [trial_number for trial_number in args.trial_numbers if trial_number not in index_map]
        if missing:
            raise ValueError(f"Requested trial numbers not found on Pareto front: {missing}")
        family_items = [(int(labels[index_map[trial_number]]), index_map[trial_number]) for trial_number in args.trial_numbers]
    else:
        family_items = sorted(reps.items())
        if args.max_families is not None:
            family_items = family_items[: args.max_families]

    analyzed_metric_rows = []
    combined_raster_panels = []
    for cluster_id, rep_idx in tqdm(family_items, desc="Rendering representative families", unit="family"):
        trial = trials[rep_idx]
        repeat_seed = _choose_repeat_seed(trial, cfg)
        sim = _prepare_simulation(objective, trial["params"], repeat_seed=repeat_seed)
        neuron_indices = _select_neurons(sim, top_neurons=args.top_neurons)
        metric_t_start_ms = max(float(cfg.ext_on_ms), float(args.plot_start_ms))
        metric_t_stop_ms = args.plot_stop_ms

        family_dir = out_dir / f"family_{cluster_id:02d}_trial_{int(trial['number'])}"
        family_dir.mkdir(parents=True, exist_ok=True)

        title = (
            f"Family {cluster_id} | trial {trial['number']} | "
            + " ".join(f"{name}={float(trial['values'][i]):.3f}" for i, name in enumerate(OBJECTIVE_NAMES))
        )
        _plot_raster(sim, family_dir / "raster.png", args.plot_start_ms, args.plot_stop_ms, title)
        _plot_type_voltage_history(sim, family_dir / "voltage_types.png", args.plot_start_ms, args.plot_stop_ms)
        _plot_neuron_voltage_history(sim, neuron_indices, family_dir / "voltage_neurons.png", args.plot_start_ms, args.plot_stop_ms)
        _plot_synaptic_traces(sim, neuron_indices, family_dir / "synaptic_traces.png", args.plot_start_ms, args.plot_stop_ms)
        analyzed_metrics = _analyzed_metrics(sim, metric_t_start_ms, metric_t_stop_ms)

        with (family_dir / "representative.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "cluster_id": int(cluster_id),
                    "trial_number": int(trial["number"]),
                    "repeat_seed": int(repeat_seed),
                    "objective_values": {name: float(trial["values"][i]) for i, name in enumerate(OBJECTIVE_NAMES)},
                    "params": dict(trial["params"]),
                    "top_neurons": neuron_indices,
                    "feature_names": feature_names,
                    "cluster_feature_space": args.cluster_features,
                },
                f,
                indent=2,
                sort_keys=True,
            )
        with (family_dir / "summary_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(analyzed_metrics, f, indent=2, sort_keys=True)

        metric_row = {
            "cluster_id": int(cluster_id),
            "trial_number": int(trial["number"]),
            "repeat_seed": int(repeat_seed),
        }
        metric_row.update(analyzed_metrics)
        analyzed_metric_rows.append(metric_row)
        _print_metric_summary(f"family {cluster_id:02d} trial {int(trial['number'])}", analyzed_metrics)
        spike_times_ms, neuron_idx = _raster_points(sim, args.plot_start_ms, args.plot_stop_ms)
        combined_raster_panels.append(
            {
                "cluster_id": int(cluster_id),
                "trial_number": int(trial["number"]),
                "spike_times_ms": spike_times_ms,
                "neuron_indices": neuron_idx,
                "inhibitory_mask": np.asarray(sim.stats.inhibitory_mask, dtype=bool),
                "title": f"Family {int(cluster_id)} | trial {int(trial['number'])}",
            }
        )

    if analyzed_metric_rows:
        metric_fields = sorted({key for row in analyzed_metric_rows for key in row.keys()})
        with (out_dir / "analyzed_trial_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=metric_fields)
            writer.writeheader()
            writer.writerows(analyzed_metric_rows)
    _plot_combined_family_rasters(
        combined_raster_panels,
        out_dir / "representative_rasters_combined.png",
        args.plot_start_ms,
        args.plot_stop_ms,
    )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
