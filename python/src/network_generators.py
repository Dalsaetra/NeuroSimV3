import numpy as np
import networkx as nx
from numpy.random import default_rng
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