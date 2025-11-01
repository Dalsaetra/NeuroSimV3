import math
import numpy as np
import networkx as nx
from collections import defaultdict

def assign_biological_weights(
    G: nx.DiGraph,
    *,
    rng: np.random.Generator | None = None,
    # --- base distribution (log-normal) ---
    mu_E= -2.0,  sigma_E= 1.0,    # log-space params for E→*
    mu_I= -1.6,  sigma_I= 0.6,    # log-space params for I→*
    # --- distance kernel ---
    use_distance=True, lambda_mm=2.0, kernel="exp", alpha=1.0,
    distance_attr="distance",
    # --- degree modulation (compensation <0, hub-boost >0) ---
    beta_out=0.0, beta_in=0.0, eps_deg=1.0,
    # --- optional type / layer scaling ---
    layer_scale: dict[tuple[int,int], float] | None = None,   # {(pre_layer, post_layer): scale}
    type_scale:  dict[tuple[str,str], float] | None = None,   # {(pre_ntype, post_ntype): scale}
    # --- clipping ---
    w_min=1e-4, w_max=10.0,
    # --- homeostatic post normalization (separate E and I budgets) ---
    target_E_in=None,   # float or dict{post_node: float}; if None, skip E normalization
    target_I_in=None,   # same for I
    # --- attribute names on nodes ---
    inhib_attr="inhibitory", layer_attr="layer", ntype_attr="ntype",
    # --- write-back ---
    weight_attr="weight",
):
    """
    Assigns positive 'weight' to each edge (u->v) in G using:
      lognormal base  × distance kernel × degree modulation × (layer/type scale),
    then optionally normalizes incoming E and I weights per post neuron.

    NOTE: Inhibition sign is NOT set here; weights remain positive.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Precompute degrees
    outdeg = dict(G.out_degree())
    indeg  = dict(G.in_degree())

    # Helpers to pull node attrs quickly
    inhib   = nx.get_node_attributes(G, inhib_attr)
    layers  = nx.get_node_attributes(G, layer_attr)
    ntypes  = nx.get_node_attributes(G, ntype_attr)

    def dist_kernel(d):
        if not use_distance or d is None or not np.isfinite(d) or d <= 0:
            return 1.0
        if kernel == "exp":
            return math.exp(-d / lambda_mm)
        elif kernel == "power":
            return 1.0 / (1.0 + (d / lambda_mm)**alpha)
        else:
            return 1.0

    # First pass: sample raw weights
    for u, v, data in G.edges(data=True):
        pre_is_inhib = bool(inhib.get(u, False))
        # base lognormal
        mu, sigma = (mu_I, sigma_I) if pre_is_inhib else (mu_E, sigma_E)
        base = float(rng.lognormal(mean=mu, sigma=sigma))

        # distance factor
        d = data.get(distance_attr, None)
        # Raise warning if distance is missing
        if d is None:
            import warnings
            warnings.warn(f"Edge ({u}->{v}) missing distance attribute '{distance_attr}'; assuming no distance effect.")
        f_d = dist_kernel(d)

        # degree modulation
        f_k = ((outdeg.get(u,0)+eps_deg)/eps_deg)**beta_out * ((indeg.get(v,0)+eps_deg)/eps_deg)**beta_in

        # layer / type scaling
        f_lt = 1.0
        if layer_scale is not None and u in layers and v in layers:
            f_lt *= layer_scale.get((layers[u], layers[v]), 1.0)
        if type_scale is not None and u in ntypes and v in ntypes:
            f_lt *= type_scale.get((ntypes[u], ntypes[v]), 1.0)

        w = base * f_d * f_k * f_lt
        w = float(np.clip(w, w_min, w_max))
        data[weight_attr] = w

    # Optional homeostatic normalization per post neuron
    # We normalize E-in and I-in separately if targets are provided.
    if target_E_in is not None or target_I_in is not None:
        # collect incoming edges per post, split by E/I
        incoming_E = defaultdict(list)
        incoming_I = defaultdict(list)
        for u, v, data in G.edges(data=True):
            (incoming_I if inhib.get(u, False) else incoming_E)[v].append((u, data))

        def _get_target(val, node):
            if val is None:
                return None
            if isinstance(val, dict):
                return float(val.get(node, np.nan))
            return float(val)

        # scale group so that sum(weights) == target (if target given and current sum > 0)
        for v, edges in incoming_E.items():
            tgt = _get_target(target_E_in, v)
            if tgt is not None and edges:
                s = sum(d[weight_attr] for _, d in edges)
                if s > 0:
                    scale = tgt / s
                    for _, d in edges:
                        d[weight_attr] = float(np.clip(d[weight_attr] * scale, w_min, w_max))
        for v, edges in incoming_I.items():
            tgt = _get_target(target_I_in, v)
            if tgt is not None and edges:
                s = sum(d[weight_attr] for _, d in edges)
                if s > 0:
                    scale = tgt / s
                    for _, d in edges:
                        d[weight_attr] = float(np.clip(d[weight_attr] * scale, w_min, w_max))

    return G
