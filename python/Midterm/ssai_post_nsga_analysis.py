import argparse
import csv
import json
import subprocess
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

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
from src.network_generators import generate_spatial_ei_network
from src.network_generators import spatial_pa_directed_var_out_reciprocal
from src.network_weight_distributor import assign_lognormal_weights_for_ntype
from src.overhead import Simulation


TOPOLOGY_CHOICES = ("baseline", "erdos_renyi", "small_world", "spatial_pa", "spatial_ei")


@dataclass
class ClassificationConfig:
    persistent_rate_hz: float = 0.5
    active_rate_hz: float = 0.05
    irregular_cv_threshold: float = 0.8
    synchronous_noise_corr_threshold: float = 0.05
    synchronous_peak_ratio_threshold: float = 200.0
    sync_band_low_hz: float = 2.0
    sync_band_high_hz: float = 120.0
    late_window_ms: float = 300.0


@dataclass
class SweepDefinition:
    name: str
    values: List[object]
    scale: str = "linear"
    mode: str = "absolute"
    spec: str = ""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Post-NSGA analysis for SSAI families: robustness via parameter permutation, "
            "1D parameter sweeps, and 2D phase-diagram sweeps."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for subparser in (
        subparsers.add_parser("robustness", help="Rerun family members and parameter permutations."),
        subparsers.add_parser("sweep1d", help="Run 1D sweeps around selected Pareto solutions."),
        subparsers.add_parser("sweep2d", help="Run pairwise sweeps to build phase-diagram tables."),
    ):
        _add_common_run_args(subparser)
        _add_common_selection_args(subparser)
        _add_common_classification_args(subparser)

    robustness = subparsers.choices["robustness"]
    robustness.add_argument("--n-seeds", type=int, default=3, help="Repeat each family member this many times.")
    robustness.add_argument(
        "--n-permutations",
        type=int,
        default=12,
        help="Number of family-level parameter mosaics to generate per family.",
    )
    robustness.add_argument(
        "--permutation-mode",
        type=str,
        default="independent",
        choices=("independent", "cyclic"),
        help="How parameter values are recombined across family members.",
    )
    robustness.add_argument(
        "--min-family-size",
        type=int,
        default=2,
        help="Families smaller than this still get baseline reruns, but permutation runs are skipped.",
    )
    robustness.add_argument(
        "--representative-only",
        action="store_true",
        help="Only rerun the representative trial from each family instead of all family members.",
    )
    robustness.add_argument(
        "--representative-perturbation-pct",
        type=float,
        default=None,
        help=(
            "If set, generate representative-only perturbations by jittering every evolved parameter "
            "within +/- this percent of the representative value. Uses --n-permutations as the number of "
            "perturbed variants per family."
        ),
    )

    sweep1d = subparsers.choices["sweep1d"]
    sweep1d.add_argument(
        "--sweep",
        action="append",
        required=True,
        help=(
            "Sweep spec. Continuous: name:start:stop:num[:linear|log][:absolute|relative]. "
            "Discrete: name=v1,v2,v3. Examples: inhibitory_strength:50:800:9, "
            "delay_scale:0.5:2.0:7, topology=baseline,small_world,spatial_pa, "
            "ampa_nmda_ratio:0.25:4:7:log, inhibitory_strength:0.5:1.5:7:relative."
        ),
    )
    sweep1d.add_argument("--n-seeds", type=int, default=3, help="Repeat each sweep point this many times.")

    sweep2d = subparsers.choices["sweep2d"]
    sweep2d.add_argument("--x-sweep", type=str, required=True, help="Sweep spec for x-axis.")
    sweep2d.add_argument("--y-sweep", type=str, required=True, help="Sweep spec for y-axis.")
    sweep2d.add_argument("--n-seeds", type=int, default=3, help="Repeat each sweep point this many times.")

    return parser


def _add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", type=str, required=True, help="NSGA-II output directory.")
    parser.add_argument(
        "--family-analysis-dir",
        type=str,
        default=None,
        help="Directory with cluster_summary.json; defaults to <run-dir>/family_analysis.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory; defaults to <run-dir>/post_nsga_analysis/<command>.",
    )
    parser.add_argument(
        "--topology",
        type=str,
        default="baseline",
        choices=TOPOLOGY_CHOICES,
        help="Topology backend to use for all runs in this command unless a sweep overrides it.",
    )
    parser.add_argument(
        "--delay-std-frac-exc",
        type=float,
        default=None,
        help="If set, use delay_std_exc_ms = delay_mean_exc_ms * this fraction after all mean/scale adjustments.",
    )
    parser.add_argument(
        "--delay-std-frac-inh",
        type=float,
        default=None,
        help="If set, use delay_std_inh_ms = delay_mean_inh_ms * this fraction after all mean/scale adjustments.",
    )


def _add_common_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--trial-numbers",
        type=int,
        nargs="+",
        default=None,
        help="Explicit Pareto trial numbers to analyze. Overrides representative-trial selection.",
    )
    parser.add_argument(
        "--family-ids",
        type=int,
        nargs="+",
        default=None,
        help="Subset of family IDs from family_analysis/cluster_summary.json.",
    )
    parser.add_argument(
        "--max-families",
        type=int,
        default=None,
        help="Optional cap on representative families when trial numbers are not supplied.",
    )


def _add_common_classification_args(parser: argparse.ArgumentParser) -> None:
    defaults = ClassificationConfig()
    parser.add_argument("--persistent-rate-hz", type=float, default=defaults.persistent_rate_hz)
    parser.add_argument("--active-rate-hz", type=float, default=defaults.active_rate_hz)
    parser.add_argument("--irregular-cv-threshold", type=float, default=defaults.irregular_cv_threshold)
    parser.add_argument(
        "--synchronous-noise-corr-threshold",
        type=float,
        default=defaults.synchronous_noise_corr_threshold,
    )
    parser.add_argument(
        "--synchronous-peak-ratio-threshold",
        type=float,
        default=defaults.synchronous_peak_ratio_threshold,
    )
    parser.add_argument("--sync-band-low-hz", type=float, default=defaults.sync_band_low_hz)
    parser.add_argument("--sync-band-high-hz", type=float, default=defaults.sync_band_high_hz)
    parser.add_argument("--late-window-ms", type=float, default=defaults.late_window_ms)


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _search_config_from_run(run_dir: Path) -> SearchConfig:
    cfg_json = _load_json(run_dir / "config.json")
    return SearchConfig(**dict(cfg_json["search_config"]))


def _load_pareto_trials(run_dir: Path) -> List[Dict[str, object]]:
    return list(_load_json(run_dir / "pareto_front.json"))


def _trial_index_by_number(trials: Sequence[Dict[str, object]]) -> Dict[int, Dict[str, object]]:
    return {int(trial["number"]): dict(trial) for trial in trials}


def _default_family_analysis_dir(run_dir: Path, family_analysis_dir: Optional[str]) -> Path:
    if family_analysis_dir:
        return Path(family_analysis_dir).resolve()
    return (run_dir / "family_analysis").resolve()


def _load_family_summary(path: Path) -> Dict[str, object]:
    cluster_summary_path = path / "cluster_summary.json"
    if not cluster_summary_path.exists():
        raise FileNotFoundError(
            f"Expected family summary at {cluster_summary_path}. Run Midterm/analyze_solution_families.py first "
            "or provide --trial-numbers."
        )
    return dict(_load_json(cluster_summary_path))


def _resolve_family_items(
    *,
    family_analysis_dir: Path,
    trials: Sequence[Dict[str, object]],
    family_ids: Optional[Sequence[int]],
    max_families: Optional[int],
) -> List[Dict[str, object]]:
    summary = _load_family_summary(family_analysis_dir)
    families = list(summary.get("families", []))
    if family_ids:
        wanted = {int(x) for x in family_ids}
        families = [fam for fam in families if int(fam["cluster_id"]) in wanted]
    families = sorted(families, key=lambda x: int(x["cluster_id"]))
    if max_families is not None:
        families = families[: int(max_families)]
    if not families:
        raise ValueError("No families matched the selection.")

    trial_map = _trial_index_by_number(trials)
    out = []
    for family in families:
        member_numbers = [int(x) for x in family.get("member_trial_numbers", [])]
        rep_number = int(family["representative_trial_number"])
        missing = [num for num in member_numbers + [rep_number] if num not in trial_map]
        if missing:
            raise ValueError(
                f"Family {family['cluster_id']} references trial numbers missing from pareto_front.json: {missing}"
            )
        out.append(
            {
                "cluster_id": int(family["cluster_id"]),
                "representative_trial_number": rep_number,
                "representative_trial": trial_map[rep_number],
                "member_trial_numbers": member_numbers,
                "member_trials": [trial_map[num] for num in member_numbers],
            }
        )
    return out


def _resolve_target_trials(
    *,
    run_dir: Path,
    family_analysis_dir: Path,
    trials: Sequence[Dict[str, object]],
    trial_numbers: Optional[Sequence[int]],
    family_ids: Optional[Sequence[int]],
    max_families: Optional[int],
) -> List[Dict[str, object]]:
    trial_map = _trial_index_by_number(trials)
    if trial_numbers:
        missing = [int(num) for num in trial_numbers if int(num) not in trial_map]
        if missing:
            raise ValueError(f"Requested trial numbers not found in pareto_front.json: {missing}")
        return [trial_map[int(num)] for num in trial_numbers]

    summary_path = family_analysis_dir / "cluster_summary.json"
    if summary_path.exists():
        family_items = _resolve_family_items(
            family_analysis_dir=family_analysis_dir,
            trials=trials,
            family_ids=family_ids,
            max_families=max_families,
        )
        return [dict(item["representative_trial"]) for item in family_items]

    compromise_path = run_dir / "compromise_result.json"
    if compromise_path.exists():
        compromise = dict(_load_json(compromise_path))
        trial_number = int(compromise["number"])
        if trial_number in trial_map:
            return [trial_map[trial_number]]
        return [compromise]

    if not trials:
        raise ValueError("No candidate trials are available.")
    best = max(trials, key=lambda t: float(sum(float(v) for v in t["values"])))
    return [dict(best)]


def _build_classification_cfg(args: argparse.Namespace) -> ClassificationConfig:
    return ClassificationConfig(
        persistent_rate_hz=float(args.persistent_rate_hz),
        active_rate_hz=float(args.active_rate_hz),
        irregular_cv_threshold=float(args.irregular_cv_threshold),
        synchronous_noise_corr_threshold=float(args.synchronous_noise_corr_threshold),
        synchronous_peak_ratio_threshold=float(args.synchronous_peak_ratio_threshold),
        sync_band_low_hz=float(args.sync_band_low_hz),
        sync_band_high_hz=float(args.sync_band_high_hz),
        late_window_ms=float(args.late_window_ms),
    )


def _safe_tanh(x: float) -> float:
    return float(np.tanh(float(x)))


def _compute_peak_ratio(freq_hz: np.ndarray, psd: np.ndarray, band_low_hz: float, band_high_hz: float) -> float:
    if freq_hz is None or psd is None:
        return 0.0
    if len(freq_hz) == 0 or len(psd) == 0:
        return 0.0
    band_mask = (freq_hz >= float(band_low_hz)) & (freq_hz <= float(band_high_hz))
    if not np.any(band_mask):
        return 0.0
    band = np.asarray(psd[band_mask], dtype=float)
    baseline = float(np.median(band))
    if baseline <= 0:
        return 0.0
    return float(np.max(band) / (baseline + 1e-12))


def _dominant_frequency_hz(freq_hz: np.ndarray, psd: np.ndarray, band_low_hz: float, band_high_hz: float) -> float:
    if freq_hz is None or psd is None:
        return float("nan")
    if len(freq_hz) == 0 or len(psd) == 0:
        return float("nan")
    band_mask = (freq_hz >= float(band_low_hz)) & (freq_hz <= float(band_high_hz))
    if not np.any(band_mask):
        return float("nan")
    freq_band = np.asarray(freq_hz[band_mask], dtype=float)
    psd_band = np.asarray(psd[band_mask], dtype=float)
    if psd_band.size == 0:
        return float("nan")
    return float(freq_band[int(np.argmax(psd_band))])


def _base_node_graph(cfg: SearchConfig) -> nx.DiGraph:
    graph = nx.DiGraph()
    for i in range(cfg.n_neurons):
        is_inhib = i >= cfg.n_excit
        graph.add_node(
            i,
            inhibitory=bool(is_inhib),
            ntype=cfg.inhibitory_type if is_inhib else cfg.excitatory_type,
            layer=0,
        )
    return graph


def _node_is_inhibitory(graph: nx.DiGraph, node: int, cfg: SearchConfig) -> bool:
    return bool(graph.nodes[node].get("inhibitory", int(node) >= cfg.n_excit))


def _baseline_topology(cfg: SearchConfig, seed: int) -> nx.DiGraph:
    objective = SSAIMultiObjective(cfg)
    return objective._build_template_graph(seed=seed)


def _erdos_renyi_topology(cfg: SearchConfig, seed: int) -> nx.DiGraph:
    p = min(1.0, max(0.0, float(cfg.n_out) / max(1, cfg.n_neurons - 1)))
    raw = nx.gnp_random_graph(cfg.n_neurons, p, seed=seed, directed=True)
    raw.remove_edges_from(nx.selfloop_edges(raw))
    graph = _base_node_graph(cfg)
    graph.add_edges_from((int(u), int(v), {"weight": 1.0, "distance": 1.0}) for u, v in raw.edges())
    return graph


def _small_world_topology(cfg: SearchConfig, seed: int) -> nx.DiGraph:
    k = max(2, min(cfg.n_neurons - 1, int(cfg.n_out)))
    if k % 2 == 1:
        if k < cfg.n_neurons - 1:
            k += 1
        else:
            k -= 1
    raw = nx.watts_strogatz_graph(cfg.n_neurons, max(2, k), 0.15, seed=seed)
    graph = _base_node_graph(cfg)
    for u, v in raw.edges():
        graph.add_edge(int(u), int(v), weight=1.0, distance=1.0)
        graph.add_edge(int(v), int(u), weight=1.0, distance=1.0)
    return graph


def _spatial_pa_topology(cfg: SearchConfig, seed: int) -> nx.DiGraph:
    k_lo = max(1, int(round(0.7 * cfg.n_out)))
    k_hi = max(k_lo, int(round(1.3 * cfg.n_out)))
    raw = spatial_pa_directed_var_out_reciprocal(
        n=cfg.n_neurons,
        k_out=(k_lo, k_hi),
        reciprocity=0.12,
        reciprocity_local=1.25,
        alpha_dist=8.0,
        seed=seed,
    )
    graph = _base_node_graph(cfg)
    graph.add_edges_from((int(u), int(v), {"weight": 1.0, "distance": 1.0}) for u, v in raw.edges())
    return graph


def _mean_outgoing_weight_per_edge(graph: nx.DiGraph, *, inhibitory: bool, cfg: SearchConfig) -> float:
    vals = []
    for node in graph.nodes():
        if _node_is_inhibitory(graph, node, cfg) != inhibitory:
            continue
        out_deg = int(graph.out_degree(node))
        if out_deg <= 0:
            continue
        out_sum = sum(float(graph[node][nbr].get("weight", 0.0)) for nbr in graph.successors(node))
        vals.append(out_sum / float(out_deg))
    return float(np.mean(vals)) if vals else 0.0


def _mean_inverse_outdegree(graph: nx.DiGraph, *, inhibitory: bool, cfg: SearchConfig) -> float:
    vals = []
    for node in graph.nodes():
        if _node_is_inhibitory(graph, node, cfg) != inhibitory:
            continue
        out_deg = int(graph.out_degree(node))
        if out_deg <= 0:
            continue
        vals.append(1.0 / float(out_deg))
    return float(np.mean(vals)) if vals else 0.0


def _baseline_weight_reference(cfg: SearchConfig, params: Mapping[str, float], seed: int) -> Tuple[float, float]:
    graph = _baseline_topology(cfg, seed)
    rng = np.random.default_rng(int(seed))
    _apply_weight_configuration(graph, cfg, params, rng)
    ref_exc = _mean_outgoing_weight_per_edge(graph, inhibitory=False, cfg=cfg)
    ref_inh = _mean_outgoing_weight_per_edge(graph, inhibitory=True, cfg=cfg)
    return ref_exc, ref_inh


def _spatial_ei_defaults(cfg: SearchConfig, params: Mapping[str, float], seed: int) -> Dict[str, float]:
    exc_frac = float(cfg.n_excit) / float(cfg.n_neurons)
    type_fractions = {
        cfg.excitatory_type: exc_frac,
        cfg.inhibitory_type: max(0.0, 1.0 - exc_frac),
    }
    inhibitory_types = (cfg.inhibitory_type,)
    weight_dist_by_ntype = {
        cfg.excitatory_type: "lognormal",
        cfg.inhibitory_type: "normal",
    }
    outdegree_config_by_type = {
        cfg.excitatory_type: {"dist": "lognormal", "params": (2.65, 0.8)},
        cfg.inhibitory_type: {"dist": "neg-bin", "params": (50, 40)},
    }
    ref_exc, ref_inh = _baseline_weight_reference(cfg, params, seed)
    generator_kwargs = {
        "n_neurons": cfg.n_neurons,
        "type_fractions": type_fractions,
        "inhibitory_types": inhibitory_types,
        "p0_by_pair": {"EE": 0.5, "EI": 0.5, "IE": 0.5, "II": 0.5},
        "lambda_by_preclass": {"E": 0.2, "I": 0.05},
        "distance_scale": 20.0,
        "mu_E": float(cfg.recurrent_exc_mu),
        "sigma_E": float(params["recurrent_exc_lognorm_sigma"]),
        "mu_I": 30.0,
        "sigma_I": 6.0,
        "weight_dist_by_ntype": weight_dist_by_ntype,
        "outdegree_config_by_type": outdegree_config_by_type,
        "seed": int(seed),
    }
    graph_probe = generate_spatial_ei_network(
        **generator_kwargs,
        normalize_mode=None,
        normalize_target_out_E=None,
        normalize_target_out_I=None,
    )
    invdeg_exc = _mean_inverse_outdegree(graph_probe, inhibitory=False, cfg=cfg)
    invdeg_inh = _mean_inverse_outdegree(graph_probe, inhibitory=True, cfg=cfg)
    return {
        "ref_exc": float(ref_exc),
        "ref_inh": float(ref_inh),
        "invdeg_exc": float(invdeg_exc),
        "invdeg_inh": float(invdeg_inh),
        "target_out_E": float(ref_exc / max(invdeg_exc, 1e-12)),
        "target_out_I": float(ref_inh / max(invdeg_inh, 1e-12)),
        "lambda_E": 0.2,
        "lambda_I": 0.05,
        "sigma_E": float(params["recurrent_exc_lognorm_sigma"]),
        "sigma_I": 6.0,
    }


def _spatial_ei_generator_kwargs(
    cfg: SearchConfig,
    params: Mapping[str, float],
    seed: int,
    structural: Mapping[str, object],
) -> Dict[str, object]:
    exc_frac = float(cfg.n_excit) / float(cfg.n_neurons)
    type_fractions = {
        cfg.excitatory_type: exc_frac,
        cfg.inhibitory_type: max(0.0, 1.0 - exc_frac),
    }
    inhibitory_types = (cfg.inhibitory_type,)
    weight_dist_by_ntype = {
        cfg.excitatory_type: "lognormal",
        cfg.inhibitory_type: "normal",
    }
    outdegree_config_by_type = {
        cfg.excitatory_type: {"dist": "lognormal", "params": (2.65, 0.8)},
        cfg.inhibitory_type: {"dist": "neg-bin", "params": (50, 40)},
    }
    defaults = _spatial_ei_defaults(cfg, params, seed)
    lambda_e_raw = structural.get("spatial_ei_lambda_E")
    lambda_i_raw = structural.get("spatial_ei_lambda_I")
    sigma_e_raw = structural.get("spatial_ei_sigma_E")
    sigma_i_raw = structural.get("spatial_ei_sigma_I")
    distance_scale_factor_raw = structural.get("delay_scale")
    lambda_e = float(defaults["lambda_E"] if lambda_e_raw is None else lambda_e_raw)
    lambda_i = float(defaults["lambda_I"] if lambda_i_raw is None else lambda_i_raw)
    sigma_e = float(defaults["sigma_E"] if sigma_e_raw is None else sigma_e_raw)
    sigma_i = float(defaults["sigma_I"] if sigma_i_raw is None else sigma_i_raw)
    distance_scale_factor = float(1.0 if distance_scale_factor_raw is None else distance_scale_factor_raw)
    distance_scale = 20.0 * distance_scale_factor
    return {
        "n_neurons": cfg.n_neurons,
        "type_fractions": type_fractions,
        "inhibitory_types": inhibitory_types,
        "p0_by_pair": {"EE": 0.5, "EI": 0.5, "IE": 0.5, "II": 0.5},
        "lambda_by_preclass": {"E": lambda_e, "I": lambda_i},
        "distance_scale": distance_scale,
        "mu_E": float(cfg.recurrent_exc_mu),
        "sigma_E": sigma_e,
        "mu_I": 30.0,
        "sigma_I": sigma_i,
        "weight_dist_by_ntype": weight_dist_by_ntype,
        "outdegree_config_by_type": outdegree_config_by_type,
        "seed": int(seed),
    }


def _spatial_ei_topology(
    cfg: SearchConfig,
    params: Mapping[str, float],
    seed: int,
    structural: Optional[Mapping[str, object]] = None,
) -> nx.DiGraph:
    structural = {} if structural is None else dict(structural)
    defaults = _spatial_ei_defaults(cfg, params, seed)
    generator_kwargs = _spatial_ei_generator_kwargs(cfg, params, seed, structural)
    target_out_e_raw = structural.get("spatial_ei_normalize_target_out_E")
    target_out_i_raw = structural.get("spatial_ei_normalize_target_out_I")
    target_out_e = float(defaults["target_out_E"] if target_out_e_raw is None else target_out_e_raw)
    target_out_i = float(defaults["target_out_I"] if target_out_i_raw is None else target_out_i_raw)

    graph = generate_spatial_ei_network(
        **generator_kwargs,
        normalize_mode="out",
        normalize_target_out_E=target_out_e,
        normalize_target_out_I=target_out_i,
    )
    graph.graph["topology_ref_exc_out_weight_mean"] = float(defaults["ref_exc"])
    graph.graph["topology_ref_inh_out_weight_mean"] = float(defaults["ref_inh"])
    graph.graph["topology_probe_invdeg_exc_mean"] = float(defaults["invdeg_exc"])
    graph.graph["topology_probe_invdeg_inh_mean"] = float(defaults["invdeg_inh"])
    graph.graph["topology_target_out_E"] = float(target_out_e)
    graph.graph["topology_target_out_I"] = float(target_out_i)
    graph.graph["topology_lambda_E"] = float(generator_kwargs["lambda_by_preclass"]["E"])
    graph.graph["topology_lambda_I"] = float(generator_kwargs["lambda_by_preclass"]["I"])
    graph.graph["topology_distance_scale"] = float(generator_kwargs["distance_scale"])
    graph.graph["topology_sigma_E"] = float(generator_kwargs["sigma_E"])
    graph.graph["topology_sigma_I"] = float(generator_kwargs["sigma_I"])
    graph.graph["topology_matched_exc_out_weight_mean"] = float(
        _mean_outgoing_weight_per_edge(graph, inhibitory=False, cfg=cfg)
    )
    graph.graph["topology_matched_inh_out_weight_mean"] = float(
        _mean_outgoing_weight_per_edge(graph, inhibitory=True, cfg=cfg)
    )
    return graph


def _build_topology_graph(
    cfg: SearchConfig,
    params: Mapping[str, float],
    topology: str,
    seed: int,
    structural: Optional[Mapping[str, object]] = None,
) -> nx.DiGraph:
    topology = str(topology).lower()
    if topology == "baseline":
        return _baseline_topology(cfg, seed)
    if topology == "erdos_renyi":
        return _erdos_renyi_topology(cfg, seed)
    if topology == "small_world":
        return _small_world_topology(cfg, seed)
    if topology == "spatial_pa":
        return _spatial_pa_topology(cfg, seed)
    if topology == "spatial_ei":
        return _spatial_ei_topology(cfg, params, seed, structural)
    raise KeyError(f"Unsupported topology '{topology}'.")


def _apply_delay_distribution(
    graph: nx.DiGraph,
    cfg: SearchConfig,
    rng: np.random.Generator,
    *,
    delay_scale: float,
    delay_std_frac_exc: Optional[float] = None,
    delay_std_frac_inh: Optional[float] = None,
) -> None:
    exc_mean = max(0.1, float(cfg.delay_mean_exc_ms) * float(delay_scale))
    inh_mean = max(0.1, float(cfg.delay_mean_inh_ms) * float(delay_scale))
    if delay_std_frac_exc is None:
        exc_std = max(0.0, float(cfg.delay_std_exc_ms) * float(delay_scale))
    else:
        exc_std = max(0.0, exc_mean * float(delay_std_frac_exc))
    if delay_std_frac_inh is None:
        inh_std = max(0.0, float(cfg.delay_std_inh_ms) * float(delay_scale))
    else:
        inh_std = max(0.0, inh_mean * float(delay_std_frac_inh))

    for u, _, data in graph.edges(data=True):
        if _node_is_inhibitory(graph, u, cfg):
            distance = max(0.1, rng.normal(loc=inh_mean, scale=inh_std))
        else:
            distance = max(0.1, rng.normal(loc=exc_mean, scale=exc_std))
        data["distance"] = float(distance)


def _apply_weight_configuration(
    graph: nx.DiGraph,
    cfg: SearchConfig,
    params: Mapping[str, float],
    rng: np.random.Generator,
) -> None:
    inhib_g = float(params["inhibitory_scale_g"])
    for u, _, data in graph.edges(data=True):
        data["weight"] = float(inhib_g if _node_is_inhibitory(graph, u, cfg) else 1.0)

    assign_lognormal_weights_for_ntype(
        graph,
        cfg.excitatory_type,
        mu=cfg.recurrent_exc_mu,
        sigma=float(params["recurrent_exc_lognorm_sigma"]),
        w_max=cfg.recurrent_exc_wmax,
        rng=rng,
    )


def _initialize_state(cfg: SearchConfig, rng: np.random.Generator):
    vs = rng.uniform(cfg.state_v_min, cfg.state_v_max, size=cfg.n_neurons)
    us = rng.uniform(cfg.state_u_min, cfg.state_u_max, size=cfg.n_neurons)
    spikes = np.zeros(cfg.n_neurons, dtype=bool)
    ts = np.zeros_like(spikes)
    return vs, us, spikes, ts


def _apply_knob_value(
    params: MutableMapping[str, float],
    cfg_dict: MutableMapping[str, object],
    structural: MutableMapping[str, object],
    knob_name: str,
    knob_value: object,
) -> None:
    name = str(knob_name)
    if name in params:
        params[name] = float(knob_value)
        return

    if name in ("weight_heterogeneity", "recurrent_exc_sigma"):
        params["recurrent_exc_lognorm_sigma"] = float(knob_value)
        return
    if name == "inhibitory_strength":
        params["inhibitory_scale_g"] = float(knob_value)
        return
    if name in ("ampa_strength", "g_ampa_max"):
        params["g_AMPA_max"] = float(knob_value)
        return
    if name in ("nmda_strength", "g_nmda_max"):
        params["g_NMDA_max"] = float(knob_value)
        return
    if name in ("gabaa_strength", "g_gaba_a_max"):
        params["g_GABA_A_max"] = float(knob_value)
        return
    if name in ("gabab_strength", "g_gaba_b_max"):
        params["g_GABA_B_max"] = float(knob_value)
        return
    if name in ("axonal_delay_scale", "delay_scale"):
        structural["delay_scale"] = float(knob_value)
        return
    if name == "external_rate_hz":
        cfg_dict["external_rate_hz"] = float(knob_value)
        return
    if name in ("delay_std_frac_exc", "delay_std_fraction_exc"):
        structural["delay_std_frac_exc"] = float(knob_value)
        return
    if name in ("delay_std_frac_inh", "delay_std_fraction_inh"):
        structural["delay_std_frac_inh"] = float(knob_value)
        return
    if name in ("delay_mean_exc_ms", "delay_mean_inh_ms", "delay_std_exc_ms", "delay_std_inh_ms"):
        cfg_dict[name] = float(knob_value)
        return
    if name == "topology":
        structural["topology"] = str(knob_value)
        return
    if name in ("spatial_ei_normalize_target_out_E", "e_normalization"):
        structural["spatial_ei_normalize_target_out_E"] = float(knob_value)
        return
    if name in ("spatial_ei_normalize_target_out_I", "i_normalization"):
        structural["spatial_ei_normalize_target_out_I"] = float(knob_value)
        return
    if name in ("spatial_ei_lambda_E", "lambda_E"):
        structural["spatial_ei_lambda_E"] = float(knob_value)
        return
    if name in ("spatial_ei_lambda_I", "lambda_I"):
        structural["spatial_ei_lambda_I"] = float(knob_value)
        return
    if name in ("spatial_ei_sigma_E", "sigma_E"):
        structural["spatial_ei_sigma_E"] = float(knob_value)
        return
    if name in ("spatial_ei_sigma_I", "sigma_I"):
        structural["spatial_ei_sigma_I"] = float(knob_value)
        return
    if name == "ampa_nmda_ratio":
        ratio = max(1e-9, float(knob_value))
        total = float(params["g_AMPA_max"] + params["g_NMDA_max"])
        params["g_NMDA_max"] = float(total / (1.0 + ratio))
        params["g_AMPA_max"] = float(total - params["g_NMDA_max"])
        return
    if name == "gabaa_gabab_ratio":
        ratio = max(1e-9, float(knob_value))
        total = float(params["g_GABA_A_max"] + params["g_GABA_B_max"])
        params["g_GABA_B_max"] = float(total / (1.0 + ratio))
        params["g_GABA_A_max"] = float(total - params["g_GABA_B_max"])
        return

    raise KeyError(
        f"Unsupported sweep knob '{name}'. Supported knobs include direct evolved params, "
        "weight_heterogeneity, recurrent_exc_sigma, inhibitory_strength, ampa_strength, nmda_strength, "
        "gabaa_strength, gabab_strength, delay_scale, external_rate_hz, delay_mean_* / delay_std_*, "
        "delay_std_frac_exc / delay_std_frac_inh, ampa_nmda_ratio, gabaa_gabab_ratio, topology, "
        "and spatial_ei_* knobs."
    )


def _current_knob_value(
    base_cfg: SearchConfig,
    base_params: Mapping[str, float],
    knob_name: str,
    *,
    topology: str = "baseline",
    run_seed: Optional[int] = None,
) -> object:
    name = str(knob_name)
    if name in base_params:
        return float(base_params[name])
    if name in ("weight_heterogeneity", "recurrent_exc_sigma"):
        return float(base_params["recurrent_exc_lognorm_sigma"])
    if name == "inhibitory_strength":
        return float(base_params["inhibitory_scale_g"])
    if name in ("ampa_strength", "g_ampa_max"):
        return float(base_params["g_AMPA_max"])
    if name in ("nmda_strength", "g_nmda_max"):
        return float(base_params["g_NMDA_max"])
    if name in ("gabaa_strength", "g_gaba_a_max"):
        return float(base_params["g_GABA_A_max"])
    if name in ("gabab_strength", "g_gaba_b_max"):
        return float(base_params["g_GABA_B_max"])
    if name in ("axonal_delay_scale", "delay_scale"):
        return 1.0
    if name == "external_rate_hz":
        return float(getattr(base_cfg, "external_rate_hz", 50.0))
    if name in ("delay_std_frac_exc", "delay_std_fraction_exc"):
        mean = max(float(getattr(base_cfg, "delay_mean_exc_ms", 0.0)), 1e-12)
        return float(getattr(base_cfg, "delay_std_exc_ms", 0.0)) / mean
    if name in ("delay_std_frac_inh", "delay_std_fraction_inh"):
        mean = max(float(getattr(base_cfg, "delay_mean_inh_ms", 0.0)), 1e-12)
        return float(getattr(base_cfg, "delay_std_inh_ms", 0.0)) / mean
    if name in ("delay_mean_exc_ms", "delay_mean_inh_ms", "delay_std_exc_ms", "delay_std_inh_ms"):
        return float(getattr(base_cfg, name))
    if name == "topology":
        return "baseline"
    if topology == "spatial_ei" and name in {
        "spatial_ei_normalize_target_out_E",
        "e_normalization",
        "spatial_ei_normalize_target_out_I",
        "i_normalization",
        "spatial_ei_lambda_E",
        "lambda_E",
        "spatial_ei_lambda_I",
        "lambda_I",
        "spatial_ei_sigma_E",
        "sigma_E",
        "spatial_ei_sigma_I",
        "sigma_I",
    }:
        defaults = _spatial_ei_defaults(base_cfg, base_params, int(base_cfg.base_seed if run_seed is None else run_seed))
        mapping = {
            "spatial_ei_normalize_target_out_E": "target_out_E",
            "e_normalization": "target_out_E",
            "spatial_ei_normalize_target_out_I": "target_out_I",
            "i_normalization": "target_out_I",
            "spatial_ei_lambda_E": "lambda_E",
            "lambda_E": "lambda_E",
            "spatial_ei_lambda_I": "lambda_I",
            "lambda_I": "lambda_I",
            "spatial_ei_sigma_E": "sigma_E",
            "sigma_E": "sigma_E",
            "spatial_ei_sigma_I": "sigma_I",
            "sigma_I": "sigma_I",
        }
        return float(defaults[mapping[name]])
    if name == "ampa_nmda_ratio":
        denom = max(float(base_params["g_NMDA_max"]), 1e-12)
        return float(base_params["g_AMPA_max"]) / denom
    if name == "gabaa_gabab_ratio":
        denom = max(float(base_params["g_GABA_B_max"]), 1e-12)
        return float(base_params["g_GABA_A_max"]) / denom
    raise KeyError(f"Unsupported sweep knob '{name}'.")


def _resolve_sweep_value(
    base_cfg: SearchConfig,
    base_params: Mapping[str, float],
    sweep: SweepDefinition,
    raw_value: object,
    *,
    topology: str = "baseline",
    run_seed: Optional[int] = None,
) -> Tuple[object, float]:
    if topology == "spatial_ei" and str(sweep.name) in {"delay_scale", "axonal_delay_scale"} and sweep.mode != "relative":
        raise ValueError(
            "For topology='spatial_ei', delay_scale must be swept in relative mode because it maps to generator "
            "distance_scale multiplicatively."
        )
    if sweep.mode != "relative":
        return raw_value, float("nan")
    base_value = _current_knob_value(base_cfg, base_params, sweep.name, topology=topology, run_seed=run_seed)
    if isinstance(base_value, str):
        raise ValueError(f"Relative sweep is not supported for non-numeric knob '{sweep.name}'.")
    factor = float(raw_value)
    return float(base_value) * factor, float(base_value)


def _run_single_variant(
    *,
    base_cfg: SearchConfig,
    base_params: Mapping[str, float],
    run_seed: int,
    classifier_cfg: ClassificationConfig,
    delay_std_frac_exc: Optional[float] = None,
    delay_std_frac_inh: Optional[float] = None,
    knob_values: Optional[Sequence[Tuple[str, object]]] = None,
) -> Dict[str, object]:
    params = {str(k): float(v) for k, v in base_params.items()}
    cfg_dict = asdict(base_cfg)
    structural: Dict[str, object] = {
        "topology": "baseline",
        "delay_scale": 1.0,
        "delay_std_frac_exc": delay_std_frac_exc,
        "delay_std_frac_inh": delay_std_frac_inh,
        "spatial_ei_normalize_target_out_E": None,
        "spatial_ei_normalize_target_out_I": None,
        "spatial_ei_lambda_E": None,
        "spatial_ei_lambda_I": None,
        "spatial_ei_sigma_E": None,
        "spatial_ei_sigma_I": None,
    }
    knob_values = [] if knob_values is None else list(knob_values)

    for knob_name, knob_value in knob_values:
        _apply_knob_value(params, cfg_dict, structural, knob_name, knob_value)

    cfg = SearchConfig(**cfg_dict)
    objective = SSAIMultiObjective(cfg)
    rng = np.random.default_rng(int(run_seed))
    topology_name = str(structural["topology"])
    graph = _build_topology_graph(cfg, params, topology_name, int(run_seed), structural)
    if topology_name != "spatial_ei":
        _apply_delay_distribution(
            graph,
            cfg,
            rng,
            delay_scale=float(structural["delay_scale"]),
            delay_std_frac_exc=(
                None if structural.get("delay_std_frac_exc") is None else float(structural["delay_std_frac_exc"])
            ),
            delay_std_frac_inh=(
                None if structural.get("delay_std_frac_inh") is None else float(structural["delay_std_frac_inh"])
            ),
        )
        _apply_weight_configuration(graph, cfg, params, rng)

    inhibitory_nmda_weight = float(params["inhibitory_nmda_weight"])
    connectome, pop, nmda_weight = objective._build_connectome(graph, inhibitory_nmda_weight=inhibitory_nmda_weight)
    state0 = _initialize_state(cfg, rng)

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
    external_rate_hz = float(getattr(cfg, "external_rate_hz", params.get("v_ext", 50.0)))
    poisson = PoissonInput(cfg.n_neurons, rate=external_rate_hz, amplitude=ext_amp_vec, rng=rng)

    ext_on_steps = int(round(cfg.ext_on_ms / cfg.dt_ms))
    total_steps = int(round(cfg.total_ms / cfg.dt_ms))
    ext_off_steps = max(0, total_steps - ext_on_steps)

    for _ in range(ext_on_steps):
        sim.step(spike_ext=poisson(cfg.dt_ms))
    for _ in range(ext_off_steps):
        sim.step()

    result = _extract_activity_metrics(
        sim=sim,
        cfg=cfg,
        params=params,
        classifier_cfg=classifier_cfg,
        topology=str(structural["topology"]),
        delay_scale=float(structural["delay_scale"]),
    )
    result.update(
        {
            "topology_ref_exc_out_weight_mean": float(graph.graph.get("topology_ref_exc_out_weight_mean", float("nan"))),
            "topology_ref_inh_out_weight_mean": float(graph.graph.get("topology_ref_inh_out_weight_mean", float("nan"))),
            "delay_std_frac_exc": float(
                structural["delay_std_frac_exc"] if structural.get("delay_std_frac_exc") is not None else float("nan")
            ),
            "delay_std_frac_inh": float(
                structural["delay_std_frac_inh"] if structural.get("delay_std_frac_inh") is not None else float("nan")
            ),
            "topology_distance_scale": float(graph.graph.get("topology_distance_scale", float("nan"))),
            "topology_probe_invdeg_exc_mean": float(graph.graph.get("topology_probe_invdeg_exc_mean", float("nan"))),
            "topology_probe_invdeg_inh_mean": float(graph.graph.get("topology_probe_invdeg_inh_mean", float("nan"))),
            "topology_target_out_E": float(graph.graph.get("topology_target_out_E", float("nan"))),
            "topology_target_out_I": float(graph.graph.get("topology_target_out_I", float("nan"))),
            "topology_lambda_E": float(graph.graph.get("topology_lambda_E", float("nan"))),
            "topology_lambda_I": float(graph.graph.get("topology_lambda_I", float("nan"))),
            "topology_sigma_E": float(graph.graph.get("topology_sigma_E", float("nan"))),
            "topology_sigma_I": float(graph.graph.get("topology_sigma_I", float("nan"))),
            "topology_matched_exc_out_weight_mean": float(
                graph.graph.get("topology_matched_exc_out_weight_mean", float("nan"))
            ),
            "topology_matched_inh_out_weight_mean": float(
                graph.graph.get("topology_matched_inh_out_weight_mean", float("nan"))
            ),
        }
    )
    return result


def _extract_activity_metrics(
    *,
    sim,
    cfg: SearchConfig,
    params: Mapping[str, float],
    classifier_cfg: ClassificationConfig,
    topology: str,
    delay_scale: float,
) -> Dict[str, object]:
    late_start_ms = max(float(cfg.ext_on_ms), float(cfg.total_ms) - float(classifier_cfg.late_window_ms))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        post_stats = sim.stats.compute_metrics(
            cfg.dt_ms,
            bin_ms_participation=300.0,
            t_start_ms=float(cfg.ext_on_ms),
            t_stop_ms=float(cfg.total_ms),
        )
        late_stats = sim.stats.compute_metrics(
            cfg.dt_ms,
            bin_ms_participation=300.0,
            t_start_ms=late_start_ms,
            t_stop_ms=float(cfg.total_ms),
        )

    post_freq = np.asarray(post_stats.get("pop_psd_freq_hz", np.array([])), dtype=float)
    post_psd = np.asarray(post_stats.get("pop_psd", np.array([])), dtype=float)
    peak_ratio = _compute_peak_ratio(
        post_freq,
        post_psd,
        classifier_cfg.sync_band_low_hz,
        classifier_cfg.sync_band_high_hz,
    )
    peak_freq = _dominant_frequency_hz(
        post_freq,
        post_psd,
        classifier_cfg.sync_band_low_hz,
        classifier_cfg.sync_band_high_hz,
    )

    rate_post = float(post_stats.get("rate_mean_Hz", 0.0))
    rate_late = float(late_stats.get("rate_mean_Hz", 0.0))
    isi_cv = float(post_stats.get("ISI_CV_mean", 0.0))
    noise_corr = float(post_stats.get("mean_noise_corr_50ms", 0.0))
    fano_300 = float(post_stats.get("Fano_median_300ms", 0.0))

    active = rate_post >= float(classifier_cfg.active_rate_hz)
    persistent = rate_late >= float(classifier_cfg.persistent_rate_hz)
    irregular = isi_cv >= float(classifier_cfg.irregular_cv_threshold)
    synchronous = (
        peak_ratio >= float(classifier_cfg.synchronous_peak_ratio_threshold)
        or noise_corr >= float(classifier_cfg.synchronous_noise_corr_threshold)
    )

    if not active:
        activity_class = "QUIESCENT"
    else:
        activity_class = ("S" if synchronous else "A") + ("I" if irregular else "R")

    persistent_activity_class = activity_class if persistent and activity_class in {"AI", "AR", "SI", "SR"} else "NON_PERSISTENT"
    oscillation_freq_hz = float(peak_freq) if synchronous and np.isfinite(peak_freq) else float("nan")

    persistence_score = float(0.7 * _safe_tanh(min(rate_post, rate_late) / 5.0) + 0.3 * _safe_tanh(rate_late / 5.0))
    asynchrony_score = float(
        1.5 * _safe_tanh(max(0.0, isi_cv) / 2.0)
        + 1.0 * _safe_tanh(max(0.0, fano_300) / 4.0)
        - 1.5 * max(0.0, noise_corr - 0.10)
        - 1.0 * max(0.0, peak_ratio - 10.0) / 10.0
    )

    return {
        "topology": str(topology),
        "delay_scale": float(delay_scale),
        "activity_class": activity_class,
        "persistent_activity_class": persistent_activity_class,
        "is_active": int(active),
        "is_persistent": int(persistent),
        "is_irregular": int(irregular),
        "is_synchronous": int(synchronous),
        "is_ssai": int(persistent_activity_class == "AI"),
        "oscillation_freq_hz": oscillation_freq_hz,
        "peak_ratio": float(peak_ratio),
        "score_persistence_proxy": persistence_score,
        "score_asynchrony_proxy": asynchrony_score,
        "rate_post_Hz": rate_post,
        "rate_late_Hz": rate_late,
        "rate_post_Hz_E": float(post_stats.get("rate_mean_Hz_E", 0.0)),
        "rate_post_Hz_I": float(post_stats.get("rate_mean_Hz_I", 0.0)),
        "rate_late_Hz_E": float(late_stats.get("rate_mean_Hz_E", 0.0)),
        "rate_late_Hz_I": float(late_stats.get("rate_mean_Hz_I", 0.0)),
        "ISI_CV_mean": float(isi_cv),
        "ISI_CV_mean_E": float(post_stats.get("ISI_CV_mean_E", 0.0)),
        "ISI_CV_mean_I": float(post_stats.get("ISI_CV_mean_I", 0.0)),
        "Fano_median_300ms": float(fano_300),
        "mean_noise_corr_50ms": float(noise_corr),
        "pop_spec_entropy": float(post_stats.get("pop_spec_entropy", 0.0)),
        "participation_frac_total": float(post_stats.get("participation_frac_total", 0.0)),
        "params_snapshot": dict(params),
    }


def _mode_or_nan(values: Iterable[object]) -> object:
    values = list(values)
    if not values:
        return ""
    counts = Counter(values)
    return sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[0][0]


def _mean_or_nan(values: Iterable[object]) -> float:
    vals = []
    for value in values:
        if isinstance(value, str):
            continue
        val = float(value)
        if np.isfinite(val):
            vals.append(val)
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _summarize_rows(rows: Sequence[Mapping[str, object]], group_keys: Sequence[str]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[object, ...], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(name) for name in group_keys)
        grouped[key].append(row)

    summary_rows: List[Dict[str, object]] = []
    for key in sorted(grouped.keys(), key=lambda x: tuple(str(v) for v in x)):
        group = grouped[key]
        out = {name: key[idx] for idx, name in enumerate(group_keys)}
        out["n_runs"] = int(len(group))
        out["mode_activity_class"] = _mode_or_nan(row.get("activity_class", "") for row in group)
        out["mode_persistent_activity_class"] = _mode_or_nan(row.get("persistent_activity_class", "") for row in group)
        out["persistence_fraction"] = _mean_or_nan(row.get("is_persistent", 0) for row in group)
        out["ssai_fraction"] = _mean_or_nan(row.get("is_ssai", 0) for row in group)
        out["synchronous_fraction"] = _mean_or_nan(row.get("is_synchronous", 0) for row in group)
        out["active_fraction"] = _mean_or_nan(row.get("is_active", 0) for row in group)
        for metric_name in (
            "rate_post_Hz",
            "rate_late_Hz",
            "rate_post_Hz_E",
            "rate_post_Hz_I",
            "rate_late_Hz_E",
            "rate_late_Hz_I",
            "ISI_CV_mean",
            "Fano_median_300ms",
            "mean_noise_corr_50ms",
            "peak_ratio",
            "oscillation_freq_hz",
            "score_persistence_proxy",
            "score_asynchrony_proxy",
        ):
            out[f"{metric_name}_mean"] = _mean_or_nan(row.get(metric_name, float("nan")) for row in group)
        summary_rows.append(out)
    return summary_rows


def _parse_scalar(value: str) -> object:
    text = value.strip()
    lowered = text.lower()
    if lowered in set(TOPOLOGY_CHOICES):
        return lowered
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _parse_sweep_spec(spec: str) -> SweepDefinition:
    if "=" in spec and spec.count(":") == 0:
        name, raw_values = spec.split("=", 1)
        values = [_parse_scalar(piece) for piece in raw_values.split(",") if piece.strip()]
        if not values:
            raise ValueError(f"Sweep spec '{spec}' did not contain any values.")
        return SweepDefinition(name=name.strip(), values=list(values), scale="discrete", mode="absolute", spec=spec)

    parts = [piece.strip() for piece in spec.split(":")]
    if len(parts) not in (4, 5, 6):
        raise ValueError(
            f"Invalid sweep spec '{spec}'. Use name:start:stop:num[:linear|log][:absolute|relative] or name=v1,v2,v3."
        )
    name = parts[0]
    start = float(parts[1])
    stop = float(parts[2])
    n_points = int(parts[3])
    scale = "linear"
    mode = "absolute"
    for token in parts[4:]:
        token = token.lower()
        if token in {"linear", "log"}:
            scale = token
            continue
        if token in {"absolute", "relative"}:
            mode = token
            continue
        raise ValueError(f"Sweep spec '{spec}' has unsupported option '{token}'.")
    if n_points < 2:
        raise ValueError(f"Sweep spec '{spec}' must request at least two points.")
    if scale == "log":
        if start <= 0 or stop <= 0:
            raise ValueError(f"Log sweep '{spec}' requires positive bounds.")
        values = np.geomspace(start, stop, n_points)
    else:
        values = np.linspace(start, stop, n_points)
    return SweepDefinition(name=name, values=[float(v) for v in values.tolist()], scale=scale, mode=mode, spec=spec)


def _format_value_for_row(value: object) -> object:
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _family_parameter_permutation(
    member_trials: Sequence[Mapping[str, object]],
    *,
    permutation_idx: int,
    mode: str,
    rng: np.random.Generator,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    if not member_trials:
        raise ValueError("Cannot permute an empty family.")
    member_params = [dict(trial["params"]) for trial in member_trials]
    param_names = sorted(member_params[0].keys())
    family_size = len(member_params)

    params: Dict[str, float] = {}
    sources: Dict[str, int] = {}
    if mode == "cyclic":
        source_orders = {
            name: np.roll(np.arange(family_size), permutation_idx + idx).tolist()
            for idx, name in enumerate(param_names)
        }
        row_idx = permutation_idx % family_size
        for name in param_names:
            source_idx = int(source_orders[name][row_idx])
            params[name] = float(member_params[source_idx][name])
            sources[name] = int(source_idx)
    else:
        for name in param_names:
            source_idx = int(rng.integers(0, family_size))
            params[name] = float(member_params[source_idx][name])
            sources[name] = int(source_idx)
    return params, sources


def _representative_parameter_perturbation(
    base_params: Mapping[str, float],
    *,
    perturbation_pct: float,
    rng: np.random.Generator,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    bounds = {str(name): (float(low), float(high)) for name, low, high, _ in PARAM_SPECS}
    scale = abs(float(perturbation_pct)) / 100.0
    params: Dict[str, float] = {}
    deltas: Dict[str, float] = {}

    for name, base_value in dict(base_params).items():
        base_value = float(base_value)
        rel_delta = float(rng.uniform(-scale, scale))
        perturbed = float(base_value * (1.0 + rel_delta))
        if name in bounds:
            low, high = bounds[name]
            perturbed = float(np.clip(perturbed, low, high))
        params[name] = perturbed
        deltas[name] = rel_delta

    return params, deltas


def _command_output_dir(run_dir: Path, command: str, out_dir: Optional[str]) -> Path:
    if out_dir:
        path = Path(out_dir).resolve()
    else:
        path = (run_dir / "post_nsga_analysis" / command).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_followup_analysis(script_name: str, arg_name: str, target_dir: Path) -> None:
    script_path = Path(__file__).resolve().with_name(script_name)
    subprocess.run(
        [
            sys.executable,
            str(script_path),
            arg_name,
            str(target_dir),
        ],
        check=True,
    )


def _run_robustness(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve()
    family_analysis_dir = _default_family_analysis_dir(run_dir, args.family_analysis_dir)
    out_dir = _command_output_dir(run_dir, "robustness", args.out_dir)
    classifier_cfg = _build_classification_cfg(args)
    cfg = _search_config_from_run(run_dir)
    trials = _load_pareto_trials(run_dir)
    family_items = _resolve_family_items(
        family_analysis_dir=family_analysis_dir,
        trials=trials,
        family_ids=args.family_ids,
        max_families=args.max_families,
    )

    detail_rows: List[Dict[str, object]] = []
    family_summary: List[Dict[str, object]] = []

    total_runs = 0
    for family in family_items:
        n_baseline_trials = 1 if args.representative_only else len(family["member_trials"])
        total_runs += n_baseline_trials * int(args.n_seeds)
        if args.representative_only and args.representative_perturbation_pct is not None:
            total_runs += int(args.n_permutations) * int(args.n_seeds)
        elif len(family["member_trials"]) >= int(args.min_family_size):
            total_runs += int(args.n_permutations) * int(args.n_seeds)

    progress = tqdm(total=total_runs, desc="Running family robustness", unit="sim")
    for family in family_items:
        family_id = int(family["cluster_id"])
        member_trials = list(family["member_trials"])
        baseline_trials = [family["representative_trial"]] if args.representative_only else member_trials

        for member_idx, trial in enumerate(baseline_trials):
            base_seed = int(trial.get("user_attrs", {}).get("trial_seed", cfg.base_seed + 1009 * (int(trial["number"]) + 1)))
            for repeat_idx in range(int(args.n_seeds)):
                run_seed = int(base_seed + 7919 * repeat_idx)
                metrics = _run_single_variant(
                    base_cfg=cfg,
                    base_params=trial["params"],
                    run_seed=run_seed,
                    classifier_cfg=classifier_cfg,
                    delay_std_frac_exc=args.delay_std_frac_exc,
                    delay_std_frac_inh=args.delay_std_frac_inh,
                    knob_values=[("topology", args.topology)],
                )
                row = {
                    "analysis_type": "family_member",
                    "family_id": family_id,
                    "family_size": int(len(member_trials)),
                    "trial_number": int(trial["number"]),
                    "member_index": int(member_idx if not args.representative_only else -1),
                    "representative_only": int(bool(args.representative_only)),
                    "repeat_idx": int(repeat_idx),
                    "run_seed": int(run_seed),
                }
                row.update({f"objective.{name}": float(trial["values"][i]) for i, name in enumerate(OBJECTIVE_NAMES)})
                row.update(metrics)
                detail_rows.append(row)
                progress.update(1)

        if args.representative_only and args.representative_perturbation_pct is not None:
            rep_trial = family["representative_trial"]
            pert_rng = np.random.default_rng(int(cfg.base_seed + 200000 + family_id))
            for permutation_idx in range(int(args.n_permutations)):
                pert_params, rel_deltas = _representative_parameter_perturbation(
                    rep_trial["params"],
                    perturbation_pct=float(args.representative_perturbation_pct),
                    rng=pert_rng,
                )
                for repeat_idx in range(int(args.n_seeds)):
                    run_seed = int(cfg.base_seed + 700000 + 1000 * family_id + 100 * permutation_idx + repeat_idx)
                    metrics = _run_single_variant(
                        base_cfg=cfg,
                        base_params=pert_params,
                        run_seed=run_seed,
                        classifier_cfg=classifier_cfg,
                        delay_std_frac_exc=args.delay_std_frac_exc,
                        delay_std_frac_inh=args.delay_std_frac_inh,
                        knob_values=[("topology", args.topology)],
                    )
                    row = {
                        "analysis_type": "parameter_perturbation",
                        "family_id": family_id,
                        "family_size": int(len(member_trials)),
                        "trial_number": int(family["representative_trial_number"]),
                        "member_index": -1,
                        "representative_only": 1,
                        "repeat_idx": int(repeat_idx),
                        "permutation_idx": int(permutation_idx),
                        "run_seed": int(run_seed),
                        "representative_perturbation_pct": float(args.representative_perturbation_pct),
                    }
                    row.update({f"perturb_rel.{name}": float(delta) for name, delta in rel_deltas.items()})
                    row.update(metrics)
                    detail_rows.append(row)
                    progress.update(1)
        else:
            if len(member_trials) < int(args.min_family_size):
                continue

            perm_rng = np.random.default_rng(int(cfg.base_seed + 100000 + family_id))
            for permutation_idx in range(int(args.n_permutations)):
                perm_params, perm_sources = _family_parameter_permutation(
                    member_trials,
                    permutation_idx=permutation_idx,
                    mode=str(args.permutation_mode),
                    rng=perm_rng,
                )
                for repeat_idx in range(int(args.n_seeds)):
                    run_seed = int(cfg.base_seed + 500000 + 1000 * family_id + 100 * permutation_idx + repeat_idx)
                    metrics = _run_single_variant(
                        base_cfg=cfg,
                        base_params=perm_params,
                        run_seed=run_seed,
                        classifier_cfg=classifier_cfg,
                        delay_std_frac_exc=args.delay_std_frac_exc,
                        delay_std_frac_inh=args.delay_std_frac_inh,
                        knob_values=[("topology", args.topology)],
                    )
                    row = {
                        "analysis_type": "parameter_permutation",
                        "family_id": family_id,
                        "family_size": int(len(member_trials)),
                        "trial_number": int(family["representative_trial_number"]),
                        "member_index": -1,
                        "representative_only": int(bool(args.representative_only)),
                        "repeat_idx": int(repeat_idx),
                        "permutation_idx": int(permutation_idx),
                        "run_seed": int(run_seed),
                    }
                    row.update({f"perm_source.{name}": int(idx) for name, idx in perm_sources.items()})
                    row.update(metrics)
                    detail_rows.append(row)
                    progress.update(1)
    progress.close()

    summary_rows = _summarize_rows(detail_rows, group_keys=("analysis_type", "family_id"))
    summary_lookup = {
        (str(row["analysis_type"]), int(row["family_id"])): dict(row)
        for row in summary_rows
    }
    for family in family_items:
        family_id = int(family["cluster_id"])
        row = {
            "family_id": family_id,
            "family_size": int(len(family["member_trials"])),
            "representative_trial_number": int(family["representative_trial_number"]),
            "member_trial_numbers": [int(x["number"]) for x in family["member_trials"]],
            "representative_only": int(bool(args.representative_only)),
            "member_replay_summary": summary_lookup.get(("family_member", family_id), {}),
            "permutation_summary": summary_lookup.get(("parameter_permutation", family_id), {}),
            "perturbation_summary": summary_lookup.get(("parameter_perturbation", family_id), {}),
        }
        family_summary.append(row)

    _write_rows_csv(out_dir / "robustness_detail.csv", detail_rows)
    _write_rows_csv(out_dir / "robustness_summary.csv", summary_rows)
    _write_json(
        out_dir / "robustness_summary.json",
        {
            "command": "robustness",
            "run_dir": str(run_dir),
            "family_analysis_dir": str(family_analysis_dir),
            "classification": asdict(classifier_cfg),
            "topology": str(args.topology),
            "representative_only": int(bool(args.representative_only)),
            "representative_perturbation_pct": args.representative_perturbation_pct,
            "n_families": int(len(family_items)),
            "n_rows": int(len(detail_rows)),
            "families": family_summary,
        },
    )
    _run_followup_analysis("analyze_robustness_results.py", "--robustness-dir", out_dir)


def _run_sweep1d(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve()
    family_analysis_dir = _default_family_analysis_dir(run_dir, args.family_analysis_dir)
    out_dir = _command_output_dir(run_dir, "sweep1d", args.out_dir)
    classifier_cfg = _build_classification_cfg(args)
    cfg = _search_config_from_run(run_dir)
    trials = _load_pareto_trials(run_dir)
    target_trials = _resolve_target_trials(
        run_dir=run_dir,
        family_analysis_dir=family_analysis_dir,
        trials=trials,
        trial_numbers=args.trial_numbers,
        family_ids=args.family_ids,
        max_families=args.max_families,
    )
    sweeps = [_parse_sweep_spec(spec) for spec in args.sweep]

    detail_rows: List[Dict[str, object]] = []
    total_runs = len(target_trials) * sum(len(sweep.values) for sweep in sweeps) * int(args.n_seeds)
    progress = tqdm(total=total_runs, desc="Running 1D sweeps", unit="sim")

    for trial in target_trials:
        trial_number = int(trial["number"])
        base_seed = int(trial.get("user_attrs", {}).get("trial_seed", cfg.base_seed + 1009 * (trial_number + 1)))
        for sweep in sweeps:
            for value_idx, value in enumerate(sweep.values):
                for repeat_idx in range(int(args.n_seeds)):
                    run_seed = int(base_seed + 100000 * (value_idx + 1) + 7919 * repeat_idx + 37 * len(detail_rows))
                    applied_value, base_value = _resolve_sweep_value(
                        cfg,
                        trial["params"],
                        sweep,
                        value,
                        topology=str(args.topology),
                        run_seed=run_seed,
                    )
                    metrics = _run_single_variant(
                        base_cfg=cfg,
                        base_params=trial["params"],
                        run_seed=run_seed,
                        classifier_cfg=classifier_cfg,
                        delay_std_frac_exc=args.delay_std_frac_exc,
                        delay_std_frac_inh=args.delay_std_frac_inh,
                        knob_values=[("topology", args.topology), (sweep.name, applied_value)],
                    )
                    row = {
                        "trial_number": trial_number,
                        "repeat_idx": int(repeat_idx),
                        "run_seed": int(run_seed),
                        "sweep_name": sweep.name,
                        "sweep_scale": sweep.scale,
                        "sweep_mode": sweep.mode,
                        "sweep_spec": sweep.spec,
                        "sweep_axis_value": _format_value_for_row(value if sweep.mode == "relative" else applied_value),
                        "sweep_value": _format_value_for_row(applied_value),
                        "sweep_factor": float(value) if sweep.mode == "relative" else float("nan"),
                        "sweep_base_value": _format_value_for_row(base_value),
                        "value_index": int(value_idx),
                    }
                    row.update({f"objective.{name}": float(trial["values"][i]) for i, name in enumerate(OBJECTIVE_NAMES)})
                    row.update(metrics)
                    detail_rows.append(row)
                    progress.update(1)
    progress.close()

    summary_rows = _summarize_rows(detail_rows, group_keys=("trial_number", "sweep_name", "sweep_axis_value"))
    _write_rows_csv(out_dir / "sweep1d_detail.csv", detail_rows)
    _write_rows_csv(out_dir / "sweep1d_summary.csv", summary_rows)
    _write_json(
        out_dir / "sweep1d_summary.json",
        {
            "command": "sweep1d",
            "run_dir": str(run_dir),
            "classification": asdict(classifier_cfg),
            "topology": str(args.topology),
            "target_trials": [int(trial["number"]) for trial in target_trials],
            "sweeps": [asdict(sweep) for sweep in sweeps],
            "n_rows": int(len(detail_rows)),
        },
    )


def _run_sweep2d(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve()
    family_analysis_dir = _default_family_analysis_dir(run_dir, args.family_analysis_dir)
    out_dir = _command_output_dir(run_dir, "sweep2d", args.out_dir)
    classifier_cfg = _build_classification_cfg(args)
    cfg = _search_config_from_run(run_dir)
    trials = _load_pareto_trials(run_dir)
    target_trials = _resolve_target_trials(
        run_dir=run_dir,
        family_analysis_dir=family_analysis_dir,
        trials=trials,
        trial_numbers=args.trial_numbers,
        family_ids=args.family_ids,
        max_families=args.max_families,
    )
    x_sweep = _parse_sweep_spec(args.x_sweep)
    y_sweep = _parse_sweep_spec(args.y_sweep)

    detail_rows: List[Dict[str, object]] = []
    total_runs = len(target_trials) * len(x_sweep.values) * len(y_sweep.values) * int(args.n_seeds)
    progress = tqdm(total=total_runs, desc="Running 2D sweeps", unit="sim")

    for trial in target_trials:
        trial_number = int(trial["number"])
        base_seed = int(trial.get("user_attrs", {}).get("trial_seed", cfg.base_seed + 1009 * (trial_number + 1)))
        for x_idx, x_value in enumerate(x_sweep.values):
            for y_idx, y_value in enumerate(y_sweep.values):
                for repeat_idx in range(int(args.n_seeds)):
                    run_seed = int(base_seed + 500000 + 10000 * x_idx + 1000 * y_idx + 7919 * repeat_idx)
                    applied_x_value, x_base_value = _resolve_sweep_value(
                        cfg,
                        trial["params"],
                        x_sweep,
                        x_value,
                        topology=str(args.topology),
                        run_seed=run_seed,
                    )
                    applied_y_value, y_base_value = _resolve_sweep_value(
                        cfg,
                        trial["params"],
                        y_sweep,
                        y_value,
                        topology=str(args.topology),
                        run_seed=run_seed,
                    )
                    metrics = _run_single_variant(
                        base_cfg=cfg,
                        base_params=trial["params"],
                        run_seed=run_seed,
                        classifier_cfg=classifier_cfg,
                        delay_std_frac_exc=args.delay_std_frac_exc,
                        delay_std_frac_inh=args.delay_std_frac_inh,
                        knob_values=[("topology", args.topology), (x_sweep.name, applied_x_value), (y_sweep.name, applied_y_value)],
                    )
                    row = {
                        "trial_number": trial_number,
                        "repeat_idx": int(repeat_idx),
                        "run_seed": int(run_seed),
                        "x_name": x_sweep.name,
                        "x_axis_value": _format_value_for_row(x_value if x_sweep.mode == "relative" else applied_x_value),
                        "x_value": _format_value_for_row(applied_x_value),
                        "x_factor": float(x_value) if x_sweep.mode == "relative" else float("nan"),
                        "x_base_value": _format_value_for_row(x_base_value),
                        "x_index": int(x_idx),
                        "x_mode": x_sweep.mode,
                        "y_name": y_sweep.name,
                        "y_axis_value": _format_value_for_row(y_value if y_sweep.mode == "relative" else applied_y_value),
                        "y_value": _format_value_for_row(applied_y_value),
                        "y_factor": float(y_value) if y_sweep.mode == "relative" else float("nan"),
                        "y_base_value": _format_value_for_row(y_base_value),
                        "y_index": int(y_idx),
                        "y_mode": y_sweep.mode,
                        "x_spec": x_sweep.spec,
                        "y_spec": y_sweep.spec,
                    }
                    row.update({f"objective.{name}": float(trial["values"][i]) for i, name in enumerate(OBJECTIVE_NAMES)})
                    row.update(metrics)
                    detail_rows.append(row)
                    progress.update(1)
    progress.close()

    summary_rows = _summarize_rows(
        detail_rows,
        group_keys=("trial_number", "x_name", "x_axis_value", "y_name", "y_axis_value"),
    )
    _write_rows_csv(out_dir / "sweep2d_detail.csv", detail_rows)
    _write_rows_csv(out_dir / "sweep2d_summary.csv", summary_rows)
    _write_json(
        out_dir / "sweep2d_summary.json",
        {
            "command": "sweep2d",
            "run_dir": str(run_dir),
            "classification": asdict(classifier_cfg),
            "topology": str(args.topology),
            "target_trials": [int(trial["number"]) for trial in target_trials],
            "x_sweep": asdict(x_sweep),
            "y_sweep": asdict(y_sweep),
            "n_rows": int(len(detail_rows)),
        },
    )
    _run_followup_analysis("analyze_sweep2d_results.py", "--sweep2d-dir", out_dir)


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "robustness":
        _run_robustness(args)
        return
    if args.command == "sweep1d":
        _run_sweep1d(args)
        return
    if args.command == "sweep2d":
        _run_sweep2d(args)
        return
    raise KeyError(f"Unhandled command '{args.command}'.")


if __name__ == "__main__":
    main()
