import numpy as np
import networkx as nx
from numpy.random import default_rng
from typing import Any, Mapping, Sequence

try:
    from src.network_weight_distributor import (
        assign_biological_weights,
        assign_lognormal_weights_for_ntype,
    )
except Exception:  # pragma: no cover - fallback when running inside src/
    from network_weight_distributor import (
        assign_biological_weights,
        assign_lognormal_weights_for_ntype,
    )

rng = default_rng()

def spatial_pa_directed(n=2000, m=3, box_dim=2, alpha=2.0, 
                        pa_gamma=1.0, local_frac=0.7, seed=0):
    """
    n: nodes; m: out-edges per new node
    box_dim: 2 or 3 (layout space)
    alpha: distance decay exponent in [1, 4] (higher = more local)
    pa_gamma: PA strength on target in-degree (>=0)
    local_frac: fraction of edges drawn with distance bias vs global PA
    """
    rng = default_rng(seed)
    pos = rng.random((n, box_dim))  # in [0,1]^d
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(i, pos=tuple(pos[i]))

    # start with a small seed (complete digraph without self-loops)
    seed_nodes = max(5, m+1)
    for i in range(seed_nodes):
        for j in range(seed_nodes):
            if i != j:
                G.add_edge(i, j)

    # helper to compute weights for choosing targets
    def choose_targets(u, m):
        targets = set()
        x = pos[u]
        # current in-degrees with PA smoothing
        indeg = np.array([G.in_degree(v) for v in range(u)]) + 1e-6
        if u <= 0:
            return []
        # distances from u to existing nodes
        X = pos[:u]
        d = np.linalg.norm(X - x, axis=1) + 1e-6

        # distance-biased weights (local)
        w_local = d**(-alpha)
        # preferential attachment weights (global)
        w_pa = indeg**pa_gamma

        # mixture
        w_mix = local_frac * w_local + (1 - local_frac) * w_pa
        w_mix = np.maximum(w_mix, 0)
        w_mix[np.isinf(w_mix) | np.isnan(w_mix)] = 0
        if w_mix.sum() == 0:
            return rng.choice(u, size=m, replace=False).tolist()

        # sample without replacement according to weights
        cand = np.arange(u)
        w = w_mix / w_mix.sum()
        k = min(m, u)
        targets = rng.choice(cand, size=k, replace=False, p=w).tolist()
        return targets

    for u in range(seed_nodes, n):
        T = choose_targets(u, m)
        for v in T:
            if u != v and not G.has_edge(u, v):
                G.add_edge(u, v)

    # Edge property distance as Euclidean distance
    for u, v in G.edges():
        G.edges[u, v]['distance'] = np.linalg.norm(pos[u] - pos[v])

    return G

def directed_triad_closure(G, p=0.1):
    nodes = list(G.nodes())
    for v in nodes:
        # for each path u -> v -> w, add u -> w with prob p
        preds = list(G.predecessors(v))
        succs = list(G.successors(v))
        for u in preds:
            for w in succs:
                if u != w and not G.has_edge(u, w) and rng.random() < p:
                    G.add_edge(u, w)



def spatial_pa_directed_var_out(n=100, box_dim=2, 
                                alpha=2.0, pa_gamma=1.0, local_frac=0.7,
                                kout_dist="lognormal", kout_params=(0.7, 0.9),
                                kmin=1, seed=0):
    """
    Differential out-degree version of spatial_pa_directed.
    n: nodes
    box_dim: 2 or 3 (layout space)
    alpha: distance decay exponent in [1, 4] (higher = more local)
    pa_gamma: PA strength on target in-degree (>=0)
    local_frac: fraction of edges drawn with distance bias vs global PA
    seed: random seed for reproducibility
    kout_dist: 'lognormal' or 'neg-bin'
      - lognormal params: (mu, sigma)
      - neg-bin params: (mean, dispersion>0)
    kmin: lower bound on out-degree per new node

    """
    rng = default_rng(seed)
    pos = rng.random((n, box_dim))
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(i, pos=tuple(pos[i]))

    # seed core to avoid early isolation
    s = 6
    for i in range(s):
        for j in range(s):
            if i != j:
                G.add_edge(i, j)

    def sample_kout(size=1):
        if kout_dist == "lognormal":
            mu, sigma = kout_params
            k = np.floor(np.exp(rng.normal(mu, sigma, size))).astype(int)
        else:  # neg-bin
            mean, phi = kout_params
            p = phi/(phi+mean)
            r = phi
            k = rng.negative_binomial(r, p, size)
        k = np.maximum(k, kmin)
        return k

    def choose_targets(u, k):
        if u == 0:
            return []
        x = pos[u]
        X = pos[:u]
        d = np.linalg.norm(X - x, axis=1) + 1e-9
        indeg = np.array([G.in_degree(v) for v in range(u)], dtype=float) + 1e-6
        w_local = d**(-alpha)
        w_pa = indeg**pa_gamma
        w = local_frac*w_local + (1-local_frac)*w_pa
        w = np.clip(w, 0, None)
        if w.sum() == 0:
            return rng.choice(u, size=min(k,u), replace=False).tolist()
        w = w/w.sum()
        return rng.choice(np.arange(u), size=min(k,u), replace=False, p=w).tolist()

    for u in range(s, n):
        k = int(sample_kout()[0])
        for v in choose_targets(u, k):
            if u != v and not G.has_edge(u, v):
                G.add_edge(u, v)

    # Edge property distance as Euclidean distance
    for u, v in G.edges():
        G.edges[u, v]['distance'] = np.linalg.norm(pos[u] - pos[v])

    return G


def ensure_min_in_out(G: nx.DiGraph, pos_attr: str = "pos"):
    """
    Ensure every node has at least one outgoing and one incoming edge.
    If a node is missing an out-edge (or in-edge), add an edge to (or from)
    its nearest neighbor by Euclidean distance using node attribute `pos`.

    Assumes:
      - G is a nx.DiGraph
      - Each node u has G.nodes[u][pos_attr] as a 2D/3D coordinate (array-like)

    Returns:
      added_edges: list[(u, v)] of edges added
    """
    if not isinstance(G, nx.DiGraph):
        raise TypeError("G must be a networkx.DiGraph")

    nodes = list(G.nodes())
    n = len(nodes)
    if n < 2:
        return []

    # Collect and validate positions
    try:
        P = np.array([np.asarray(G.nodes[u][pos_attr], dtype=float) for u in nodes], dtype=float)
    except KeyError as e:
        raise ValueError(f"Node {e} lacks '{pos_attr}' attribute.") from None

    if P.ndim != 2 or P.shape[0] != n:
        raise ValueError(f"Positions must be shape (N, d); got {P.shape}.")

    # KDTree if available; otherwise fall back to full pairwise distances
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(P)
        def nearest_order(i, k=None):
            # Return node indices sorted by distance from i (excluding i itself)
            k = n if k is None else min(k, n)
            d, idxs = tree.query(P[i], k=k)
            idxs = np.atleast_1d(idxs)
            return [j for j in idxs if j != i]
    except Exception:
        # O(n^2) fallback – fine up to a few 1e3 nodes
        D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
        np.fill_diagonal(D, np.inf)
        def nearest_order(i, k=None):
            return list(np.argsort(D[i]))

    added = []

    # Helper: find closest target v such that edge u->v does not exist
    def closest_missing_out(u_idx):
        u = nodes[u_idx]
        for v_idx in nearest_order(u_idx):
            v = nodes[v_idx]
            if not G.has_edge(u, v) and u != v:
                return v
        return None  # fully connected out-neighborhood

    # Helper: find closest source w such that edge w->u does not exist
    def closest_missing_in(u_idx):
        u = nodes[u_idx]
        for w_idx in nearest_order(u_idx):
            w = nodes[w_idx]
            if not G.has_edge(w, u) and w != u:
                return w
        return None  # fully connected in-neighborhood

    # Pass 1: ensure out-degree >= 1
    for i, u in enumerate(nodes):
        if G.out_degree(u) == 0:
            v = closest_missing_out(i)
            if v is not None:
                G.add_edge(u, v)
                added.append((u, v))

    # Pass 2: ensure in-degree >= 1 (note: degrees changed after pass 1)
    for i, u in enumerate(nodes):
        if G.in_degree(u) == 0:
            w = closest_missing_in(i)
            if w is not None:
                G.add_edge(w, u)
                added.append((w, u))

    for u, v in added:
        if not G.has_edge(u, v):
            G.add_edge(u, v)

    # Edge property distance as Euclidean distance
    for u, v in G.edges():
        G.edges[u, v]['distance'] = np.linalg.norm(P[nodes.index(u)] - P[nodes.index(v)])

def ensure_min_in_out_advanced(G: nx.DiGraph, pos, min_in=1, min_out=1, alpha_local=10.0, rng=None):
    """Guarantee each node has ≥min_in and ≥min_out by adding short edges (with distance attribute)."""
    if rng is None:
        rng = np.random.default_rng()
    nodes = list(G.nodes())
    coords = np.array([pos[u] for u in nodes])
    for u in nodes:
        # add out-edges if needed
        need_out = max(0, min_out - G.out_degree(u))
        if need_out > 0:
            dists = np.linalg.norm(coords - coords[u], axis=1)
            order = np.argsort(dists)
            for v in order:
                if need_out == 0:
                    break
                v = int(v)
                if v == u or G.has_edge(u, v):
                    continue
                # distance-biased acceptance
                p = np.exp(-alpha_local * dists[v])
                if rng.random() < p:
                    G.add_edge(u, v, distance=float(dists[v]))
                    need_out -= 1

        # add in-edges if needed
        need_in = max(0, min_in - G.in_degree(u))
        if need_in > 0:
            dists = np.linalg.norm(coords - coords[u], axis=1)
            order = np.argsort(dists)
            for v in order:
                if need_in == 0:
                    break
                v = int(v)
                if v == u or G.has_edge(v, u):
                    continue
                p = np.exp(-alpha_local * dists[v])
                if rng.random() < p:
                    G.add_edge(v, u, distance=float(dists[v]))
                    need_in -= 1

def _euclid_dist(a, b):
    return float(np.linalg.norm(a - b))

def _sample_outdeg(n, k_out, rng):
    """
    k_out can be:
      - int  -> constant out-degree
      - tuple (low, high) -> uniform integer range inclusive
      - callable i->int (receives node index, should return int)
    """
    if callable(k_out):
        return np.array([int(max(0, k_out(i))) for i in range(n)], dtype=int)
    if isinstance(k_out, (list, np.ndarray)):
        return np.asarray(k_out, dtype=int)
    if isinstance(k_out, tuple):
        lo, hi = map(int, k_out)
        return rng.integers(lo, hi + 1, size=n)
    return np.full(n, int(k_out), dtype=int)

def spatial_pa_directed_var_out_reciprocal(
    n: int,
    box_dim: int = 2,
    k_out = (4, 8),              # variable out-degree per new node
    m0: int = 8,                 # size of initial seed
    alpha_dist: float = 8.0,     # distance kernel; larger -> more local
    attractiveness: float = 1.0, # additive in-degree bias (a.k.a. "k_in + c")
    reciprocity: float = 0.3,    # probability to add v->u when u->v exists
    reciprocity_local: float = 1.5,  # multiplicative bias for short-range reciprocity
    seed: int | None = None,
    enforce_min_deg: bool = True,
    pos_attr: str = "pos",
):
    """
    Build a spatial, directed PA graph with variable out-degree and tunable reciprocity.

    Nodes are placed uniformly in [0,1]^box_dim.
    On adding node i, we pick k_out[i] targets among existing nodes j with prob ∝
        (k_in(j) + attractiveness) * exp(-alpha_dist * ||x_i - x_j||).

    After the forward edges u->v are placed, each such edge spawns a reciprocal
    edge v->u with probability:
        p_recip(u,v) = sigmoid( logit(reciprocity) + reciprocity_local * exp(-alpha_dist * d(u,v)) )
    which favors local mutual connections when reciprocity_local > 0.

    Edge attribute:
      - 'distance' : Euclidean distance between endpoints (for your connectome pipeline)

    Returns:
      G  (and pos dict if return_pos=True)
    """
    rng = np.random.default_rng(seed)
    G = nx.DiGraph()

    # --- positions ---
    coords = rng.random((n, box_dim))
    pos = {i: coords[i] for i in range(n)}

    # --- seed graph (weakly connected, local) ---
    m0 = max(2, min(m0, n))
    for i in range(m0):
        G.add_node(i)
    # connect seed with local distance-biased edges both directions sparsely
    for u in range(m0):
        for v in range(u + 1, m0):
            d = _euclid_dist(coords[u], coords[v])
            p = np.exp(-alpha_dist * d)
            if rng.random() < p:
                G.add_edge(u, v, distance=d)
            if rng.random() < p:
                G.add_edge(v, u, distance=d)

    # --- variable out-degree for each node ---
    kout = _sample_outdeg(n, k_out, rng)
    kout[:m0] = np.minimum(kout[:m0], max(0, m0 - 1))  # keep early nodes reasonable

    # --- PA growth ---
    for i in range(m0, n):
        G.add_node(i)
        # compute target distribution over existing nodes (0..i-1)
        existing = np.arange(i, dtype=int)
        if existing.size == 0:
            continue
        # in-degree term
        kin = np.array([G.in_degree(j) for j in existing], dtype=float)
        deg_term = kin + float(attractiveness)

        # distance term
        dists = np.linalg.norm(coords[existing] - coords[i], axis=1)
        dist_term = np.exp(-alpha_dist * dists)

        scores = deg_term * dist_term
        if np.all(scores <= 0) or not np.isfinite(scores).any():
            # fallback: pure distance
            scores = dist_term + 1e-12

        probs = scores / scores.sum()
        k = int(max(0, kout[i]))
        if k == 0:
            continue

        # sample without replacement, respecting no self-loops / no multiedges
        targets = []
        # choose up to min(k, i) unique targets
        choose = min(k, i)
        # handle corner case numeric issues
        choose = max(0, choose)
        if choose > 0:
            # draw indices by weighted sampling without replacement
            # (approximate by iterative draws to keep dependencies simple)
            available = existing.tolist()
            p_vec = probs.copy()
            for _ in range(choose):
                # re-normalize
                if p_vec.sum() <= 0:
                    break
                p_vec = p_vec / p_vec.sum()
                j_idx = rng.choice(len(available), p=p_vec)
                v = int(available[j_idx])
                # add edge i->v if not already present
                if i != v and not G.has_edge(i, v):
                    d = float(np.linalg.norm(coords[i] - coords[v]))
                    G.add_edge(i, v, distance=d)
                    targets.append(v)
                # remove chosen from pool
                del available[j_idx]
                p_vec = np.delete(p_vec, j_idx)

        # --- reciprocity step for edges i->v ---
        if reciprocity and targets:
            # convert 'reciprocity' into a base logit; add a local-distance bias
            base = np.log(reciprocity / (1.0 - reciprocity + 1e-12) + 1e-12)
            for v in targets:
                if G.has_edge(v, i):
                    continue
                d = float(np.linalg.norm(coords[i] - coords[v]))
                local_push = reciprocity_local * np.exp(-alpha_dist * d)
                p_back = 1.0 / (1.0 + np.exp(-(base + local_push)))
                if rng.random() < p_back:
                    G.add_edge(v, i, distance=d)

    # --- optional: ensure every node has at least 1 in and 1 out ---
    if enforce_min_deg:
        ensure_min_in_out_advanced(G, pos, min_in=1, min_out=1, alpha_local=alpha_dist, rng=rng)

    # Assign positions as node attributes
    nx.set_node_attributes(G, pos, name=pos_attr)

    return G


def _largest_remainder_counts(n_total: int, fractions: Mapping[str, float]) -> dict[str, int]:
    keys = list(fractions.keys())
    vals = np.array([float(fractions[k]) for k in keys], dtype=float)
    if vals.size == 0:
        raise ValueError("fractions must contain at least one type.")
    if np.any(vals < 0):
        raise ValueError("fractions must be non-negative.")
    if np.all(vals == 0):
        vals = np.ones_like(vals, dtype=float)
    vals = vals / vals.sum()
    raw = vals * int(n_total)
    base = np.floor(raw).astype(int)
    rem = int(n_total - base.sum())
    if rem > 0:
        order = np.argsort(-(raw - base))
        base[order[:rem]] += 1
    return {k: int(v) for k, v in zip(keys, base)}


def _pair_key(pre_inhibitory: bool, post_inhibitory: bool) -> str:
    if not pre_inhibitory and not post_inhibitory:
        return "EE"
    if not pre_inhibitory and post_inhibitory:
        return "EI"
    if pre_inhibitory and not post_inhibitory:
        return "IE"
    return "II"


def _sample_kout_heavy_tailed(
    ntype: str,
    rng: np.random.Generator,
    outdegree_config_by_type: Mapping[str, Mapping[str, float]] | None,
    *,
    default_dist: str = "lognormal",
    default_params: tuple[float, ...] = (2.2, 0.85),
    min_k: int = 1,
    max_k: int | None = None,
) -> int:
    cfg = {} if outdegree_config_by_type is None else dict(outdegree_config_by_type.get(ntype, {}))
    dist = str(cfg.get("dist", default_dist)).lower()
    params = cfg.get("params", default_params)
    if not isinstance(params, Sequence):
        raise ValueError(f"outdegree params for type '{ntype}' must be a tuple/list.")

    if dist == "lognormal":
        if len(params) != 2:
            raise ValueError("lognormal outdegree params must be (mu, sigma).")
        mu, sigma = float(params[0]), float(params[1])
        k = int(np.floor(np.exp(rng.normal(mu, sigma))))
    elif dist == "pareto":
        if len(params) != 2:
            raise ValueError("pareto outdegree params must be (shape, scale).")
        shape, scale = float(params[0]), float(params[1])
        k = int(np.floor((rng.pareto(shape) + 1.0) * scale))
    elif dist in ("neg-bin", "negative_binomial", "nb"):
        if len(params) != 2:
            raise ValueError("negative binomial outdegree params must be (mean, dispersion).")
        mean, phi = float(params[0]), float(params[1])
        p = phi / (phi + mean)
        k = int(rng.negative_binomial(phi, p))
    elif dist == "constant":
        if len(params) != 1:
            raise ValueError("constant outdegree params must be (k,).")
        k = int(params[0])
    else:
        raise ValueError(f"Unsupported out-degree distribution '{dist}'.")

    k = max(int(min_k), k)
    if max_k is not None:
        k = min(k, int(max_k))
    return int(k)


def _normalize_weights_total(
    G: nx.DiGraph,
    mode: str,
    target: float | Mapping[int, float],
    *,
    weight_attr: str = "weight",
):
    if mode not in ("in", "out"):
        raise ValueError("normalize_mode must be None, 'in', or 'out'.")

    def _target_for(node: int) -> float:
        if isinstance(target, Mapping):
            return float(target.get(node, 0.0))
        return float(target)

    for node in G.nodes():
        tgt = _target_for(node)
        if tgt <= 0:
            continue
        if mode == "in":
            edges = [(u, node) for u in G.predecessors(node)]
        else:
            edges = [(node, v) for v in G.successors(node)]
        if not edges:
            continue
        s = float(sum(max(0.0, G[u][v].get(weight_attr, 0.0)) for u, v in edges))
        if s <= 0:
            continue
        scale = tgt / s
        for u, v in edges:
            G[u][v][weight_attr] = float(max(0.0, G[u][v].get(weight_attr, 0.0) * scale))


def _normalize_out_weights_by_preclass(
    G: nx.DiGraph,
    *,
    target_E: float | Mapping[int, float] | None,
    target_I: float | Mapping[int, float] | None,
    inhib_attr: str = "inhibitory",
    weight_attr: str = "weight",
):
    def _target_for(node: int, target_val):
        if target_val is None:
            return None
        if isinstance(target_val, Mapping):
            return float(target_val.get(node, 0.0))
        return float(target_val)

    for node in G.nodes():
        is_inh = bool(G.nodes[node].get(inhib_attr, False))
        tgt = _target_for(node, target_I if is_inh else target_E)
        if tgt is None or tgt <= 0:
            continue
        edges = [(node, v) for v in G.successors(node)]
        if not edges:
            continue
        s = float(sum(max(0.0, G[u][v].get(weight_attr, 0.0)) for u, v in edges))
        if s <= 0:
            continue
        scale = tgt / s
        for u, v in edges:
            G[u][v][weight_attr] = float(max(0.0, G[u][v].get(weight_attr, 0.0) * scale))


def _assign_normal_weights_for_ntype(
    G: nx.DiGraph,
    ntype: str,
    *,
    mu: float,
    sigma: float,
    w_min: float,
    w_max: float,
    rng: np.random.Generator,
    ntype_attr: str = "ntype",
    weight_attr: str = "weight",
    apply_to: str = "pre",
):
    if apply_to not in ("pre", "post"):
        raise ValueError("apply_to must be 'pre' or 'post'.")

    for u, v, data in G.edges(data=True):
        node = u if apply_to == "pre" else v
        if G.nodes[node].get(ntype_attr, None) != ntype:
            continue
        w = float(rng.normal(loc=mu, scale=sigma))
        data[weight_attr] = float(np.clip(w, w_min, w_max))


def _reindex_graph_grouped_by_ntype(
    G: nx.DiGraph,
    *,
    ntype_attr: str = "ntype",
    ntype_order: Sequence[str] | None = None,
) -> nx.DiGraph:
    nodes = list(G.nodes())
    if not nodes:
        return G

    if ntype_order is None:
        # Stable deterministic default: alphabetical type order.
        ordered_types = sorted({str(G.nodes[u].get(ntype_attr, "")) for u in nodes})
    else:
        ordered_types = [str(t) for t in ntype_order]

    known_type_set = set(ordered_types)
    extra_types = sorted(
        {str(G.nodes[u].get(ntype_attr, "")) for u in nodes if str(G.nodes[u].get(ntype_attr, "")) not in known_type_set}
    )
    full_type_order = ordered_types + extra_types
    rank = {t: i for i, t in enumerate(full_type_order)}

    nodes_sorted = sorted(nodes, key=lambda u: (rank.get(str(G.nodes[u].get(ntype_attr, "")), len(rank)), int(u)))
    mapping = {old: new for new, old in enumerate(nodes_sorted)}
    return nx.relabel_nodes(G, mapping, copy=True)


def generate_spatial_ei_network(
    n_neurons: int = 1000,
    *,
    space_dim: int = 2,
    distance_scale: float = 1.0,
    seed: int | None = None,
    type_fractions: Mapping[str, float] | None = None,
    inhibitory_types: Sequence[str] = ("b",),
    layer: int = 0,
    p0_by_pair: Mapping[str, float] | None = None,
    lambda_by_preclass: Mapping[str, float] | None = None,
    outdegree_config_by_type: Mapping[str, Mapping[str, float]] | None = None,
    outdegree_min: int = 1,
    outdegree_max: int | None = None,
    weight_pair_scale: Mapping[str, float] | None = None,
    use_weight_distributor: bool = True,
    mu_E: float = -2.0,
    sigma_E: float = 1.0,
    mu_I: float = -1.6,
    sigma_I: float = 0.6,
    weight_dist_by_ntype: Mapping[str, str] | None = None,
    lognormal_by_ntype: Mapping[str, tuple[float, float]] | None = None,
    weight_clip: tuple[float, float] = (1e-4, 100.0),
    normalize_mode: str | None = None,
    normalize_target: float | Mapping[int, float] | None = None,
    normalize_target_out_E: float | Mapping[int, float] | None = None,
    normalize_target_out_I: float | Mapping[int, float] | None = None,
    pos_attr: str = "pos",
    distance_attr: str = "distance",
    weight_attr: str = "weight",
    multiplicity_attr: str = "multiplicity",
):
    """
    Generate a directed E/I spatial network using:
      1) k_out drawn from heavy-tailed distributions per type,
      2) target sampling with replacement from p0 * exp(-d/lambda),
      3) lognormal weights by neuron type, then optional in/out normalization.
    `distance_scale` only affects the stored edge `distance_attr` values
    (e.g., for delays). Connectivity uses unscaled geometric distances.
    Use `weight_dist_by_ntype` to pick per-type distribution:
      - "lognormal" (default)
      - "normal" (Gaussian)
    Per-type `(mu, sigma)` are supplied via `lognormal_by_ntype`
    (legacy name retained for compatibility).
    If `group_indices_by_ntype=True`, node IDs are relabeled so indices are
    grouped contiguously by `ntype` in `group_ntype_order` (or alphabetical).
    For `normalize_mode="out"`, you can set separate targets with
    `normalize_target_out_E` and `normalize_target_out_I`.

    Node attributes:
      - inhibitory: bool
      - ntype: str
      - layer: int
      - pos: tuple

    Edge attributes:
      - distance: float
      - multiplicity: int
      - weight: float
    """
    if n_neurons <= 0:
        raise ValueError("n_neurons must be > 0.")
    if space_dim not in (2, 3):
        raise ValueError("space_dim must be 2 or 3.")
    if distance_scale <= 0:
        raise ValueError("distance_scale must be > 0.")

    rng = np.random.default_rng(seed)

    if type_fractions is None:
        type_fractions = {"ss4": 0.8, "b": 0.2}
    inhibitory_types = set(inhibitory_types)
    ntype_counts = _largest_remainder_counts(n_neurons, type_fractions)

    node_types = []
    for ntype, count in ntype_counts.items():
        node_types.extend([ntype] * int(count))
    rng.shuffle(node_types)

    p0 = {"EE": 0.15, "EI": 0.15, "IE": 0.15, "II": 0.15}
    if p0_by_pair is not None:
        p0.update({k: float(v) for k, v in p0_by_pair.items()})

    lam = {"E": 0.2, "I": 0.2}
    if lambda_by_preclass is not None:
        lam.update({k: float(v) for k, v in lambda_by_preclass.items()})

    w_pair_scale = {"EE": 1.0, "EI": 1.0, "IE": 1.0, "II": 1.0}
    if weight_pair_scale is not None:
        w_pair_scale.update({k: float(v) for k, v in weight_pair_scale.items()})

    coords = rng.random((n_neurons, space_dim))
    G = nx.DiGraph()
    for i in range(n_neurons):
        ntype = str(node_types[i])
        is_inh = ntype in inhibitory_types
        G.add_node(
            i,
            inhibitory=bool(is_inh),
            ntype=ntype,
            layer=int(layer),
            **{pos_attr: tuple(coords[i])},
        )

    # Draw k_out and connect with replacement. Multiplicity stores repeated draws.
    for u in range(n_neurons):
        pre_inh = bool(G.nodes[u]["inhibitory"])
        pre_class = "I" if pre_inh else "E"
        k_out = _sample_kout_heavy_tailed(
            G.nodes[u]["ntype"],
            rng,
            outdegree_config_by_type,
            min_k=outdegree_min,
            max_k=outdegree_max,
        )
        if k_out <= 0 or n_neurons <= 1:
            continue

        d_raw = np.linalg.norm(coords - coords[u], axis=1)
        d_prob = d_raw.copy()
        d_prob[u] = np.inf

        probs = np.zeros(n_neurons, dtype=float)
        for v in range(n_neurons):
            if v == u:
                continue
            post_inh = bool(G.nodes[v]["inhibitory"])
            key = _pair_key(pre_inh, post_inh)
            lambda_pre = max(1e-12, lam[pre_class])
            probs[v] = max(0.0, p0[key]) * np.exp(-d_prob[v] / lambda_pre)

        total = float(probs.sum())
        if total <= 0 or not np.isfinite(total):
            probs = np.ones(n_neurons, dtype=float)
            probs[u] = 0.0
            total = float(probs.sum())
        probs /= total

        drawn = rng.choice(np.arange(n_neurons), size=int(k_out), replace=True, p=probs)
        counts = np.bincount(drawn, minlength=n_neurons)
        counts[u] = 0

        targets = np.where(counts > 0)[0]
        for v in targets:
            mult = int(counts[v])
            dist_uv = float(d_raw[v] * float(distance_scale))
            G.add_edge(
                u,
                int(v),
                **{
                    distance_attr: dist_uv,
                    multiplicity_attr: mult,
                },
            )

    # Ensure all edges carry distance even if altered externally later.
    for u, v in G.edges():
        if distance_attr not in G[u][v]:
            G[u][v][distance_attr] = float(np.linalg.norm(coords[u] - coords[v]) * float(distance_scale))
        if multiplicity_attr not in G[u][v]:
            G[u][v][multiplicity_attr] = 1

    # Base lognormal draw from the shared weight distributor utility.
    if use_weight_distributor:
        assign_biological_weights(
            G,
            rng=rng,
            mu_E=mu_E,
            sigma_E=sigma_E,
            mu_I=mu_I,
            sigma_I=sigma_I,
            use_distance=False,
            w_min=float(weight_clip[0]),
            w_max=float(weight_clip[1]),
            weight_attr=weight_attr,
        )
    else:
        for u, v in G.edges():
            G[u][v][weight_attr] = 1.0

    # Optional per-ntype overrides for finer control:
    # - choose distribution with weight_dist_by_ntype[ntype] in {"lognormal","normal"}.
    # - choose (mu, sigma) with lognormal_by_ntype[ntype] (name kept for compatibility).
    if lognormal_by_ntype or weight_dist_by_ntype:
        override_types = set()
        if lognormal_by_ntype:
            override_types.update(str(k) for k in lognormal_by_ntype.keys())
        if weight_dist_by_ntype:
            override_types.update(str(k) for k in weight_dist_by_ntype.keys())

        for ntype in override_types:
            dist_name = "lognormal"
            if weight_dist_by_ntype is not None:
                dist_name = str(weight_dist_by_ntype.get(str(ntype), "lognormal")).lower()

            if lognormal_by_ntype is not None and ntype in lognormal_by_ntype:
                mu, sigma = lognormal_by_ntype[ntype]
            else:
                is_inh_type = str(ntype) in inhibitory_types
                mu = mu_I if is_inh_type else mu_E
                sigma = sigma_I if is_inh_type else sigma_E

            if dist_name in ("lognormal", "log_norm", "log-normal"):
                assign_lognormal_weights_for_ntype(
                    G,
                    ntype=str(ntype),
                    mu=float(mu),
                    sigma=float(sigma),
                    w_min=float(weight_clip[0]),
                    w_max=float(weight_clip[1]),
                    rng=rng,
                    ntype_attr="ntype",
                    weight_attr=weight_attr,
                    apply_to="pre",
                )
            elif dist_name in ("normal", "gaussian", "gaussian_normal"):
                _assign_normal_weights_for_ntype(
                    G,
                    ntype=str(ntype),
                    mu=float(mu),
                    sigma=float(sigma),
                    w_min=float(weight_clip[0]),
                    w_max=float(weight_clip[1]),
                    rng=rng,
                    ntype_attr="ntype",
                    weight_attr=weight_attr,
                    apply_to="pre",
                )
            else:
                raise ValueError(
                    f"Unsupported weight distribution '{dist_name}' for type '{ntype}'. "
                    "Use 'lognormal' or 'normal'."
                )

    # Apply multiplicity and E/I pair scaling.
    for u, v in G.edges():
        pre_inh = bool(G.nodes[u]["inhibitory"])
        post_inh = bool(G.nodes[v]["inhibitory"])
        key = _pair_key(pre_inh, post_inh)
        mult = int(max(1, G[u][v].get(multiplicity_attr, 1)))
        scale = float(w_pair_scale.get(key, 1.0))
        w = float(G[u][v].get(weight_attr, 1.0))
        G[u][v][weight_attr] = float(np.clip(w * mult * scale, weight_clip[0], weight_clip[1]))

    if normalize_mode is not None:
        if normalize_mode == "out" and (normalize_target_out_E is not None or normalize_target_out_I is not None):
            default_target = 1.0 if normalize_target is None else normalize_target
            target_E = normalize_target_out_E if normalize_target_out_E is not None else default_target
            target_I = normalize_target_out_I if normalize_target_out_I is not None else default_target
            _normalize_out_weights_by_preclass(
                G,
                target_E=target_E,
                target_I=target_I,
                inhib_attr="inhibitory",
                weight_attr=weight_attr,
            )
        else:
            if normalize_target is None:
                normalize_target = 1.0
            _normalize_weights_total(G, mode=normalize_mode, target=normalize_target, weight_attr=weight_attr)

    return G


def _counts_from_distribution(
    n_total: int,
    names: Sequence[str],
    distribution: Mapping[str, float] | Sequence[float] | None,
) -> dict[str, int]:
    names = [str(x) for x in names]
    if n_total <= 0:
        raise ValueError("n_total must be > 0.")
    if len(names) == 0:
        raise ValueError("Need at least one compartment name.")

    if distribution is None:
        frac = {name: 1.0 / len(names) for name in names}
        return _largest_remainder_counts(n_total, frac)

    if isinstance(distribution, Mapping):
        vals = np.array([float(distribution.get(name, 0.0)) for name in names], dtype=float)
    else:
        vals = np.asarray(distribution, dtype=float)
        if vals.shape[0] != len(names):
            raise ValueError("distribution length must match number of compartments.")

    if np.any(vals < 0):
        raise ValueError("distribution values must be non-negative.")

    # Treat as counts only when all entries are whole numbers and sum to total.
    if np.all(np.floor(vals) == vals) and int(vals.sum()) == int(n_total):
        return {name: int(v) for name, v in zip(names, vals)}

    if vals.sum() <= 0:
        vals = np.ones_like(vals, dtype=float)
    vals = vals / vals.sum()
    return _largest_remainder_counts(n_total, {name: float(v) for name, v in zip(names, vals)})


def _value_for_compartment(
    spec: Any,
    names: Sequence[str],
    idx: int,
    name: str,
    default: Any,
) -> Any:
    if spec is None:
        return default
    if isinstance(spec, Mapping):
        return spec.get(name, default)
    if isinstance(spec, (list, tuple, np.ndarray)):
        if idx >= len(spec):
            return default
        return spec[idx]
    return spec


def _matrix_from_compartment_spec(
    names: Sequence[str],
    matrix_spec: np.ndarray | Mapping[tuple[str, str], float] | None,
    *,
    default_value: float = 0.0,
    zero_diagonal: bool = False,
) -> np.ndarray:
    names = [str(x) for x in names]
    n = len(names)
    out = np.full((n, n), float(default_value), dtype=float)

    if matrix_spec is None:
        if zero_diagonal:
            np.fill_diagonal(out, 0.0)
        return out

    if isinstance(matrix_spec, Mapping):
        idx = {name: i for i, name in enumerate(names)}
        for (a, b), val in matrix_spec.items():
            if a not in idx or b not in idx:
                raise ValueError(f"Unknown compartment in matrix key ({a}, {b}).")
            out[idx[a], idx[b]] = float(val)
    else:
        arr = np.asarray(matrix_spec, dtype=float)
        if arr.shape != (n, n):
            raise ValueError(f"Matrix must have shape ({n}, {n}), got {arr.shape}.")
        out = arr.copy()

    if zero_diagonal:
        np.fill_diagonal(out, 0.0)
    return out


def _classical_mds(distance_matrix: np.ndarray, dim: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n == 0:
        return np.zeros((0, dim), dtype=float)
    D = np.asarray(distance_matrix, dtype=float)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    keep = np.maximum(eigvals[:dim], 0.0)
    X = eigvecs[:, :dim] * np.sqrt(keep)[None, :]
    if X.shape[1] < dim:
        pad = np.zeros((n, dim - X.shape[1]), dtype=float)
        X = np.hstack([X, pad])
    return X


def generate_multi_compartment_spatial_ei_network(
    n_total_neurons: int = 2000,
    *,
    compartment_names: Sequence[str] | None = None,
    compartment_distribution: Mapping[str, float] | Sequence[float] | None = None,
    inhibitory_fraction_by_compartment: Mapping[str, float] | Sequence[float] | float = 0.2,
    excitatory_type_by_compartment: Mapping[str, str] | Sequence[str] | str = "ss4",
    inhibitory_type_by_compartment: Mapping[str, str] | Sequence[str] | str = "b",
    compartment_params: Mapping[str, Mapping[str, Any]] | None = None,
    inter_compartment_matrix: np.ndarray | Mapping[tuple[str, str], float] | None = None,
    compartment_distance_matrix: np.ndarray | Mapping[tuple[str, str], float] | None = None,
    compartment_centers: Mapping[str, Sequence[float]] | None = None,
    space_dim: int = 2,
    inter_pa_gamma_pre: float = 1.5,
    inter_pa_gamma_post: float = 1.5,
    inter_lambda_distance: float | None = None,
    inter_distance_scale: float = 1.0,
    inter_weight_dist: str = "lognormal",
    inter_mu_E: float = -2.0,
    inter_sigma_E: float = 1.0,
    inter_mu_I: float = -1.6,
    inter_sigma_I: float = 0.6,
    inter_weight_pair_scale: Mapping[str, float] | None = None,
    inter_weight_clip: tuple[float, float] = (1e-4, 100.0),
    seed: int | None = None,
    pos_attr: str = "pos",
    distance_attr: str = "distance",
    weight_attr: str = "weight",
    multiplicity_attr: str = "multiplicity",
    compartment_attr: str = "compartment",
) -> nx.DiGraph:
    """
    Build a directed multi-compartment network:
      - Each compartment is generated by `generate_spatial_ei_network`.
      - Compartments are then sparsely connected using a directed compartment->compartment matrix.
      - Inter-compartment edges are sampled with preferential attachment (degree-biased).

    `inter_compartment_matrix[i,j]` controls directed sparsity from compartment i to j.
    Inter-compartment delay/distance comes from `compartment_distance_matrix` (or center distances).
    """
    if n_total_neurons <= 0:
        raise ValueError("n_total_neurons must be > 0.")
    if space_dim not in (2, 3):
        raise ValueError("space_dim must be 2 or 3.")
    if inter_distance_scale <= 0:
        raise ValueError("inter_distance_scale must be > 0.")

    rng = np.random.default_rng(seed)

    if compartment_names is None:
        if isinstance(compartment_distribution, Mapping) and len(compartment_distribution) > 0:
            compartment_names = [str(k) for k in compartment_distribution.keys()]
        else:
            compartment_names = ["comp0", "comp1"]
    compartment_names = [str(x) for x in compartment_names]
    n_comp = len(compartment_names)
    if n_comp == 0:
        raise ValueError("compartment_names cannot be empty.")

    counts = _counts_from_distribution(n_total_neurons, compartment_names, compartment_distribution)

    inter_mat = _matrix_from_compartment_spec(
        compartment_names,
        inter_compartment_matrix,
        default_value=0.0,
        zero_diagonal=True,
    )

    if compartment_centers is not None:
        centers = np.zeros((n_comp, space_dim), dtype=float)
        for i, name in enumerate(compartment_names):
            c = np.asarray(compartment_centers.get(name, np.zeros(space_dim)), dtype=float).reshape(-1)
            if c.size < space_dim:
                c = np.pad(c, (0, space_dim - c.size))
            centers[i] = c[:space_dim]
    else:
        dist_mat = _matrix_from_compartment_spec(
            compartment_names,
            compartment_distance_matrix,
            default_value=0.0,
            zero_diagonal=True,
        )
        if np.any(dist_mat > 0):
            centers = _classical_mds(dist_mat, dim=space_dim)
        else:
            centers = np.zeros((n_comp, space_dim), dtype=float)
            for i in range(n_comp):
                centers[i, 0] = float(i) * 2.0

    if compartment_distance_matrix is None:
        comp_dist = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
        np.fill_diagonal(comp_dist, 0.0)
    else:
        comp_dist = _matrix_from_compartment_spec(
            compartment_names,
            compartment_distance_matrix,
            default_value=0.0,
            zero_diagonal=True,
        )

    inter_pair_scale = {"EE": 1.0, "EI": 1.0, "IE": 1.0, "II": 1.0}
    if inter_weight_pair_scale is not None:
        inter_pair_scale.update({k: float(v) for k, v in inter_weight_pair_scale.items()})

    G = nx.DiGraph()
    nodes_by_comp: dict[str, list[int]] = {name: [] for name in compartment_names}
    comp_idx = {name: i for i, name in enumerate(compartment_names)}
    offset = 0

    # Build each compartment independently, then merge with index offset.
    for ci, cname in enumerate(compartment_names):
        n_comp_nodes = int(counts[cname])
        if n_comp_nodes <= 0:
            continue

        inh_frac = float(_value_for_compartment(inhibitory_fraction_by_compartment, compartment_names, ci, cname, 0.2))
        inh_frac = float(np.clip(inh_frac, 0.0, 1.0))
        exc_type = str(_value_for_compartment(excitatory_type_by_compartment, compartment_names, ci, cname, "ss4"))
        inh_type = str(_value_for_compartment(inhibitory_type_by_compartment, compartment_names, ci, cname, "b"))
        type_fractions = {exc_type: float(1.0 - inh_frac), inh_type: float(inh_frac)}

        params = {} if compartment_params is None else dict(compartment_params.get(cname, {}))
        # Provide defaults from compartment-level requirements; compartment params can override.
        params.setdefault("space_dim", space_dim)
        params.setdefault("type_fractions", type_fractions)
        params.setdefault("inhibitory_types", (inh_type,))
        params.setdefault("seed", int(rng.integers(0, 2**31 - 1)))

        Gc = generate_spatial_ei_network(
            n_neurons=n_comp_nodes,
            pos_attr=pos_attr,
            distance_attr=distance_attr,
            weight_attr=weight_attr,
            multiplicity_attr=multiplicity_attr,
            **params,
        )

        center = centers[ci]
        mapping = {u: (offset + int(u)) for u in Gc.nodes()}

        for u, attrs in Gc.nodes(data=True):
            gu = mapping[u]
            node_attrs = dict(attrs)
            p_local = np.asarray(node_attrs.get(pos_attr, np.zeros(space_dim)), dtype=float).reshape(-1)
            if p_local.size < space_dim:
                p_local = np.pad(p_local, (0, space_dim - p_local.size))
            p_global = p_local[:space_dim] + center
            node_attrs[pos_attr] = tuple(p_global)
            node_attrs["local_pos"] = tuple(p_local[:space_dim])
            node_attrs[compartment_attr] = cname
            node_attrs["compartment_index"] = ci
            node_attrs["compartment_local_index"] = int(u)
            G.add_node(gu, **node_attrs)
            nodes_by_comp[cname].append(gu)

        for u, v, attrs in Gc.edges(data=True):
            gu, gv = mapping[u], mapping[v]
            edge_attrs = dict(attrs)
            edge_attrs["inter_compartment"] = False
            G.add_edge(gu, gv, **edge_attrs)

        offset += n_comp_nodes

    # Sparse directed inter-compartment wiring with PA.
    for i, pre_comp in enumerate(compartment_names):
        pre_nodes = nodes_by_comp[pre_comp]
        if len(pre_nodes) == 0:
            continue
        for j, post_comp in enumerate(compartment_names):
            if i == j:
                continue
            post_nodes = nodes_by_comp[post_comp]
            if len(post_nodes) == 0:
                continue

            p_comp = float(max(0.0, inter_mat[i, j]))
            if p_comp <= 0:
                continue

            d_comp = float(max(0.0, comp_dist[i, j])) * float(inter_distance_scale)
            dist_factor = 1.0
            if inter_lambda_distance is not None and inter_lambda_distance > 0:
                dist_factor = float(np.exp(-d_comp / float(inter_lambda_distance)))

            n_expected = p_comp * len(pre_nodes) * len(post_nodes) * dist_factor
            n_draws = int(rng.poisson(max(0.0, n_expected)))
            if n_draws <= 0:
                continue

            for _ in range(n_draws):
                pre_w = np.array([(G.out_degree(u) + 1.0) ** inter_pa_gamma_pre for u in pre_nodes], dtype=float)
                post_w = np.array([(G.in_degree(v) + 1.0) ** inter_pa_gamma_post for v in post_nodes], dtype=float)
                if pre_w.sum() <= 0:
                    pre_w = np.ones_like(pre_w, dtype=float)
                if post_w.sum() <= 0:
                    post_w = np.ones_like(post_w, dtype=float)
                pre_w /= pre_w.sum()
                post_w /= post_w.sum()

                u = int(rng.choice(pre_nodes, p=pre_w))
                v = int(rng.choice(post_nodes, p=post_w))

                pre_inh = bool(G.nodes[u].get("inhibitory", False))
                post_inh = bool(G.nodes[v].get("inhibitory", False))
                pair_key = _pair_key(pre_inh, post_inh)

                if inter_weight_dist.lower() in ("normal", "gaussian", "gaussian_normal"):
                    mu, sigma = (inter_mu_I, inter_sigma_I) if pre_inh else (inter_mu_E, inter_sigma_E)
                    w_draw = float(rng.normal(mu, sigma))
                else:
                    mu, sigma = (inter_mu_I, inter_sigma_I) if pre_inh else (inter_mu_E, inter_sigma_E)
                    w_draw = float(rng.lognormal(mu, sigma))
                w_draw *= float(inter_pair_scale.get(pair_key, 1.0))
                w_draw = float(np.clip(w_draw, inter_weight_clip[0], inter_weight_clip[1]))

                if G.has_edge(u, v):
                    G[u][v][multiplicity_attr] = int(G[u][v].get(multiplicity_attr, 1) + 1)
                    G[u][v][weight_attr] = float(
                        np.clip(G[u][v].get(weight_attr, 0.0) + w_draw, inter_weight_clip[0], inter_weight_clip[1])
                    )
                    if bool(G[u][v].get("inter_compartment", False)):
                        G[u][v][distance_attr] = float(d_comp)
                else:
                    G.add_edge(
                        u,
                        v,
                        **{
                            distance_attr: float(d_comp),
                            weight_attr: float(w_draw),
                            multiplicity_attr: 1,
                            "inter_compartment": True,
                        },
                    )

    return G
