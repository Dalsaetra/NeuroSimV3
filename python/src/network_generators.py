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