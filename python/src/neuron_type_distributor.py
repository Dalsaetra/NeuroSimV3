import math
import numpy as np
import networkx as nx
from numpy.random import default_rng

def _zscore_dict(d):
    """Return z-scored copy of dict node->value (handles constants)."""
    vals = np.array(list(d.values()), dtype=float)
    mu, sd = float(vals.mean()), float(vals.std())
    if sd == 0 or not np.isfinite(sd):
        return {k: 0.0 for k in d}
    return {k: (float(v) - mu) / sd for k, v in d.items()}

def _sigmoid(x): 
    return 1.0 / (1.0 + np.exp(-x))

def _softmax(scores):
    a = np.array(scores, dtype=float)
    a = a - np.max(a)  # stability
    ea = np.exp(a)
    Z = ea.sum()
    if Z == 0 or not np.isfinite(Z):
        return np.full_like(a, 1/len(a))
    return ea / Z

def _get_metric(G, name, default=0.0):
    return {u: G.nodes[u].get(name, default) for u in G.nodes()}

def compute_node_metrics(G: nx.DiGraph):
    """
    Compute and attach useful metrics as node attributes:
      k_in, k_out, k_total, kout_kin_ratio, clustering, betweenness
    Returns dict of metric dicts for convenience.
    """
    k_in  = dict(G.in_degree())
    k_out = dict(G.out_degree())
    k_tot = {u: k_in[u] + k_out[u] for u in G.nodes()}
    # ratio: stabilize with +1
    kout_kin_ratio = {u: (k_out[u] + 1) / (k_in[u] + 1) for u in G.nodes()}

    # clustering: use undirected projection (common in directed brain graphs)
    clu = nx.clustering(G.to_undirected())

    # betweenness on directed graph (can be expensive; use normalized)
    # For big graphs, consider k-sampling: nx.betweenness_centrality(G, k=1000)
    btw = nx.betweenness_centrality(G, normalized=True, endpoints=False)

    # attach to graph
    for u in G.nodes():
        G.nodes[u]["k_in"] = k_in[u]
        G.nodes[u]["k_out"] = k_out[u]
        G.nodes[u]["k_total"] = k_tot[u]
        G.nodes[u]["kout_kin_ratio"] = kout_kin_ratio[u]
        G.nodes[u]["clustering"] = clu[u]
        G.nodes[u]["betweenness"] = btw[u]

    return dict(k_in=k_in, k_out=k_out, k_total=k_tot, kout_kin_ratio=kout_kin_ratio,
                clustering=clu, betweenness=btw)

def assign_EI_and_subtypes(
    G: nx.DiGraph,
    target_frac_exc=0.8,
    # Weights for the E/I logistic: score = b0 + Σ w_f * z(f); p(E) = sigmoid(score + bias_to_hit_frac)
    EI_weights=None,
    # Subtype weight dicts: name -> feature weights (bias + Σ w_f * z(f)), softmax within group
    EXC_subtypes=None,
    INH_subtypes=None,
    rng_seed=0,
    mode="sample"  # "sample" or "argmax"
):
    """
    Assigns:
      - G.nodes[u]['polarity'] in {'E','I'}
      - G.nodes[u]['subtype']  (e.g., 'p23','TC','TI','TRN','nb','nb1','b')
    Based on z-scored node metrics and configurable weights.

    EI_weights: dict of feature weights for E probability (logistic). Default biases E with higher out-degree/centrality.
    EXC_subtypes / INH_subtypes: dict: subtype -> {'bias':..., 'k_out':..., 'k_in':..., 'k_total':..., 'kout_kin_ratio':..., 'clustering':..., 'betweenness':...}
    """
    rng = default_rng(rng_seed)

    # 1) Ensure metrics present
    metrics = compute_node_metrics(G)

    # 2) Build z-scored feature tables
    z_k_in   = _zscore_dict(metrics["k_in"])
    z_k_out  = _zscore_dict(metrics["k_out"])
    z_k_tot  = _zscore_dict(metrics["k_total"])
    z_ratio  = _zscore_dict(metrics["kout_kin_ratio"])
    z_clu    = _zscore_dict(metrics["clustering"])
    z_btw    = _zscore_dict(metrics["betweenness"])

    features = {
        "k_in": z_k_in, "k_out": z_k_out, "k_total": z_k_tot,
        "kout_kin_ratio": z_ratio, "clustering": z_clu, "betweenness": z_btw
    }

    # 3) Defaults: E enriched for high out-degree/centrality; I enriched for high clustering & in-degree
    if EI_weights is None:
        EI_weights = {
            "bias": 0.0,
            "k_out": 0.9,
            "k_in": -0.2,
            "k_total": 0.3,
            "kout_kin_ratio": 0.6,
            "clustering": -0.2,
            "betweenness": 0.5,
        }

    # Subtype defaults — tweak to taste
    if EXC_subtypes is None:
        EXC_subtypes = {
            # Layer 2/3 pyramidal-ish: moderate degree, high clustering
            "p23": {"bias": 0.2, "k_total": 0.2, "clustering": 0.8, "k_out": 0.1, "betweenness": 0.1},
            # Thalamocortical-like projector: high out-degree & centrality, less clustering
            "TC":  {"bias": 0.1, "k_out": 0.8, "betweenness": 0.5, "kout_kin_ratio": 0.3, "clustering": -0.2},
        }

    if INH_subtypes is None:
        INH_subtypes = {
            # Thalamic inhibitory interneuron-ish: higher in-degree / hub receivers
            "TI":  {"bias": 0.2, "k_in": 0.7, "k_total": 0.2, "clustering": 0.2},
            # Reticular nucleus-like: very central, gatekeeper; high btw
            "TRN": {"bias": 0.1, "betweenness": 0.8, "k_in": 0.2, "clustering": 0.1},
            # Basket cells: local dense wiring (high clustering), decent out
            "b":   {"bias": 0.0, "clustering": 0.8, "k_out": 0.2},
            # Neurogliaform (nb/nb1): extremely local, low degree but high clustering
            "nb":  {"bias": 0.0, "clustering": 0.9, "k_total": -0.2},
            "nb1": {"bias": 0.0, "clustering": 0.9, "k_total": -0.3},
        }

    nodes = list(G.nodes())

    # 4) Calibrate global bias to hit target E fraction on average
    #    We approximate by matching mean of feature score to sigmoid^-1(target_frac_exc).
    def score_E(u):
        s = EI_weights.get("bias", 0.0)
        for f, ztab in features.items():
            w = EI_weights.get(f, 0.0)
            s += w * ztab[u]
        return s

    raw_scores = np.array([score_E(u) for u in nodes], dtype=float)
    # Compute additional bias so that mean(sigmoid(raw + bias)) ~= target_frac_exc
    # Simple one-step via inverse-logit on mean raw score:
    target_logit = math.log(target_frac_exc/(1-target_frac_exc))
    bias_correction = target_logit - float(raw_scores.mean())

    # 5) Assign E/I
    polarities = {}
    for u, s in zip(nodes, raw_scores):
        pE = _sigmoid(s + bias_correction)
        if mode == "argmax":
            pol = "E" if pE >= 0.5 else "I"
        else:
            pol = "E" if (default_rng().random() < pE) else "I"
        polarities[u] = pol
        G.nodes[u]["polarity"] = pol
        G.nodes[u]["pE"] = pE  # optional: keep the probability

    # 6) Subtypes via softmax inside each polarity group
    def subtype_probs(u, subtype_weights):
        names = list(subtype_weights.keys())
        scores = []
        for name in names:
            w = subtype_weights[name]
            s = w.get("bias", 0.0)
            for f, ztab in features.items():
                s += w.get(f, 0.0) * ztab[u]
            scores.append(s)
        probs = _softmax(scores)
        return names, probs

    for u in nodes:
        if polarities[u] == "E":
            names, probs = subtype_probs(u, EXC_subtypes)
        else:
            names, probs = subtype_probs(u, INH_subtypes)
        if mode == "argmax":
            idx = int(np.argmax(probs))
        else:
            idx = int(default_rng().choice(len(names), p=probs))
        G.nodes[u]["subtype"] = names[idx]
        # optional diagnostics
        G.nodes[u]["subtype_probs"] = dict(zip(names, map(float, probs)))

    return {
        "EI_bias_used": float(EI_weights.get("bias", 0.0) + bias_correction),
        "target_frac_exc": target_frac_exc,
        "achieved_frac_exc": float(np.mean([1.0 if G.nodes[u]["polarity"]=="E" else 0.0 for u in nodes])),
        "EXC_subtypes": list(EXC_subtypes.keys()),
        "INH_subtypes": list(INH_subtypes.keys()),
    }

# ----------- Example usage -----------
# G = spatial_pa_directed_var_out(n=2000, box_dim=2, seed=7)  # from earlier
# ensure_min_in_out(G)                                        # from earlier
# info = assign_EI_and_subtypes(G, target_frac_exc=0.8, rng_seed=42, mode="sample")
# print(info)
# # Access attributes:
# # G.nodes[u]["polarity"] -> "E" or "I"
# # G.nodes[u]["subtype"]  -> e.g., "p23","TC","TI","TRN","nb","nb1","b"

# ---------- quota utilities ----------
def _normalize_quota(quota_dict, group_size):
    """
    Accepts counts or fractions; returns integer counts per key
    summing exactly to group_size using largest-remainder rounding.
    """
    keys = list(quota_dict.keys())
    vals = np.array([quota_dict[k] for k in keys], dtype=float)
    if np.all(vals >= 1.0) and np.all(np.floor(vals) == vals):
        # counts
        counts = vals.astype(int)
    else:
        # fractions -> counts
        frac = vals / vals.sum()
        raw = frac * group_size
        flo = np.floor(raw).astype(int)
        rem = group_size - flo.sum()
        if rem > 0:
            order = np.argsort(-(raw - flo))  # largest remainders first
            flo[order[:rem]] += 1
        counts = flo
    # Cap any overflows (shouldn't happen after LR, but guard)
    total = counts.sum()
    if total > group_size:
        # trim from largest counts
        order = np.argsort(-counts)
        i = 0
        while counts.sum() > group_size and i < len(order):
            if counts[order[i]] > 0:
                counts[order[i]] -= 1
            i += 1
    # If under-filled due to zero quotas, fill the remainder by largest current scores later.
    return dict(zip(keys, counts))

def _scores_for_subtypes(u, features, subtype_weights):
    s = {}
    for name, w in subtype_weights.items():
        v = w.get("bias", 0.0)
        for fname, ztab in features.items():
            if fname in w:
                v += w[fname] * ztab[u]
        s[name] = float(v)
    return s

def _greedy_assign_with_quotas(nodes, scores_by_node, quotas):
    """
    Greedy assignment: sort nodes by confidence margin (best - second best),
    assign each to best available subtype with remaining quota.
    If quotas don't fill (some zero), assign leftover to best scoring subtype available.
    """
    # Build per-node sorted subtype prefs and confidence margins
    pref = {}
    margins = []
    for u in nodes:
        items = sorted(scores_by_node[u].items(), key=lambda kv: kv[1], reverse=True)
        pref[u] = [name for name, _ in items]
        if len(items) >= 2:
            margins.append((u, items[0][1] - items[1][1]))
        else:
            margins.append((u, 1e9))
    order = [u for u,_ in sorted(margins, key=lambda t: t[1], reverse=True)]

    remaining = quotas.copy()
    assign = {}
    # First pass: try to give everyone their top feasible choice
    for u in order:
        for name in pref[u]:
            if remaining.get(name, 0) > 0:
                assign[u] = name
                remaining[name] -= 1
                break
    # Second pass (if some nodes unassigned due to zero quotas): fill to any subtype still available
    if len(assign) < len(nodes):
        leftover_nodes = [u for u in nodes if u not in assign]
        for u in leftover_nodes:
            # pick best among those with remaining > 0
            options = [(name, scores_by_node[u][name]) for name, cap in remaining.items() if cap > 0]
            if options:
                name = max(options, key=lambda kv: kv[1])[0]
                assign[u] = name
                remaining[name] -= 1
            else:
                # No quota left anywhere: assign to global best (won't meet quota but keeps label)
                assign[u] = pref[u][0]
    return assign

def _flow_assign_with_quotas(nodes, scores_by_node, quotas, scale=1000):
    """
    Min-cost flow assignment maximizing total score under quotas.
    Converts to integer costs = -score*scale.
    """
    import networkx as nx
    H = nx.DiGraph()
    s, t = "_S", "_T"
    H.add_node(s, demand=-len(nodes))
    H.add_node(t, demand=+len(nodes))

    # Subtype supply nodes
    for name, cap in quotas.items():
        if cap <= 0: 
            continue
        H.add_edge(s, f"sub::{name}", capacity=cap, weight=0)

    # Subtype -> node edges with costs
    for name, cap in quotas.items():
        if cap <= 0: 
            continue
        sub = f"sub::{name}"
        for u in nodes:
            sc = scores_by_node[u][name]
            cost = int(round(-sc * scale))
            H.add_edge(sub, f"n::{u}", capacity=1, weight=cost)

    # Node -> sink edges
    for u in nodes:
        H.add_edge(f"n::{u}", t, capacity=1, weight=0)

    flow = nx.min_cost_flow(H)
    assign = {}
    for name in quotas.keys():
        sub = f"sub::{name}"
        if sub not in flow:
            continue
        for node_label, f in flow[sub].items():
            if f > 0 and node_label.startswith("n::"):
                u = node_label[3:]
                assign[u] = name
    # Any missing (due to zero quotas everywhere) -> pick best
    for u in nodes:
        if u not in assign:
            assign[u] = max(scores_by_node[u].items(), key=lambda kv: kv[1])[0]
    return assign

# ---------- main: E/I + quotas ----------
def assign_EI_and_subtypes_with_quotas(
    G: nx.DiGraph,
    target_frac_exc=0.8,
    EI_weights=None,
    EXC_subtypes=None, # dict name->feature weights (weights are weakly typed)
    INH_subtypes=None, # dict name->feature weights
    EXC_quota=None,   # dict name->fraction or name->count
    INH_quota=None,   # dict name->fraction or name->count
    rng_seed=None,
    solver="greedy",  # "greedy" or "flow"
):
    """
    Writes:
      - G.nodes[u]['polarity'] in {'E','I'}
      - G.nodes[u]['subtype']  among EXC_subtypes or INH_subtypes keys
    Enforces per-polarity subtype quotas exactly when feasible.
    Prefer solver="greedy" for big graphs; use "flow" if you need the strict maximum-score assignment under quotas.
    """
    rng = default_rng(rng_seed)
    # --- metrics
    M = compute_node_metrics(G)
    z = {
        "k_in": _zscore_dict(M["k_in"]),
        "k_out": _zscore_dict(M["k_out"]),
        "k_total": _zscore_dict(M["k_total"]),
        "kout_kin_ratio": _zscore_dict(M["kout_kin_ratio"]),
        "clustering": _zscore_dict(M["clustering"]),
        "betweenness": _zscore_dict(M["betweenness"]),
    }

    # defaults
    if EI_weights is None:
        EI_weights = {
            "bias": 0.0,
            "k_out": 0.9, "k_in": -0.2, "k_total": 0.3,
            "kout_kin_ratio": 0.6, "clustering": -0.2, "betweenness": 0.5,
        }
    if EXC_subtypes is None:
        EXC_subtypes = {
            "p23": {"bias": 0.2, "k_total": 0.2, "clustering": 0.8, "k_out": 0.1, "betweenness": 0.1},
            "TC":  {"bias": 0.1, "k_out": 0.8, "betweenness": 0.5, "kout_kin_ratio": 0.3, "clustering": -0.2},
        }
    if INH_subtypes is None:
        INH_subtypes = {
            "TI":  {"bias": 0.2, "k_in": 0.7, "k_total": 0.2, "clustering": 0.2},
            "TRN": {"bias": 0.1, "betweenness": 0.8, "k_in": 0.2, "clustering": 0.1},
            "b":   {"bias": 0.0, "clustering": 0.8, "k_out": 0.2},
            "nb":  {"bias": 0.0, "clustering": 0.9, "k_total": -0.2},
            "nb1": {"bias": 0.0, "clustering": 0.9, "k_total": -0.3},
        }

    nodes = list(G.nodes())

    # --- E/I probability & calibration
    def score_E(u):
        s = EI_weights.get("bias", 0.0)
        for f, ztab in z.items():
            s += EI_weights.get(f, 0.0) * ztab[u]
        return s

    raw = np.array([score_E(u) for u in nodes])
    target_logit = math.log(target_frac_exc/(1-target_frac_exc))
    bias_corr = target_logit - float(raw.mean())

    polarities = {}
    for u, r in zip(nodes, raw):
        pE = _sigmoid(r + bias_corr)
        polarities[u] = "E" if rng.random() < pE else "I"
        G.nodes[u]["polarity"] = polarities[u]
        G.nodes[u]["pE"] = float(pE)

    # --- Build subtype scores per node for both groups
    def scores_table(group_nodes, subtype_weights):
        out = {}
        for u in group_nodes:
            out[u] = _scores_for_subtypes(u, z, subtype_weights)
        return out

    exc_nodes = [u for u in nodes if polarities[u] == "E"]
    inh_nodes = [u for u in nodes if polarities[u] == "I"]

    exc_scores = scores_table(exc_nodes, EXC_subtypes)
    inh_scores = scores_table(inh_nodes, INH_subtypes)

    # --- Normalize quotas to counts
    if EXC_quota is None:
        # default: proportional to softmaxed mean scores across nodes (data-driven)
        avg = {name: np.mean([exc_scores[u][name] for u in exc_nodes]) if exc_nodes else 0.0
               for name in EXC_subtypes.keys()}
        # convert to fractions via softmax of averages, then to counts
        vals = np.array(list(avg.values()))
        if vals.sum() == 0:
            fracs = np.ones_like(vals) / len(vals)
        else:
            fracs = _softmax(vals)
        EXC_quota = dict(zip(EXC_subtypes.keys(), fracs))
    if INH_quota is None:
        avg = {name: np.mean([inh_scores[u][name] for u in inh_nodes]) if inh_nodes else 0.0
               for name in INH_subtypes.keys()}
        vals = np.array(list(avg.values()))
        if vals.sum() == 0:
            fracs = np.ones_like(vals) / len(vals)
        else:
            fracs = _softmax(vals)
        INH_quota = dict(zip(INH_subtypes.keys(), fracs))

    exc_counts = _normalize_quota(EXC_quota, len(exc_nodes))
    inh_counts = _normalize_quota(INH_quota, len(inh_nodes))

    # Guard: if some subtype quota exceeds number of nodes (impossible), cap.
    def _cap(qdict, size):
        total = sum(qdict.values())
        if total <= size: 
            return qdict
        # scale down proportionally then largest-remainder
        frac = {k: v/total for k,v in qdict.items()}
        return _normalize_quota(frac, size)

    exc_counts = _cap(exc_counts, len(exc_nodes))
    inh_counts = _cap(inh_counts, len(inh_nodes))

    # --- Assign with quotas
    if solver == "flow":
        try:
            exc_assign = _flow_assign_with_quotas(exc_nodes, exc_scores, exc_counts)
            inh_assign = _flow_assign_with_quotas(inh_nodes, inh_scores, inh_counts)
        except Exception:
            # fallback to greedy if flow not available
            exc_assign = _greedy_assign_with_quotas(exc_nodes, exc_scores, exc_counts)
            inh_assign = _greedy_assign_with_quotas(inh_nodes, inh_scores, inh_counts)
    else:
        exc_assign = _greedy_assign_with_quotas(exc_nodes, exc_scores, exc_counts)
        inh_assign = _greedy_assign_with_quotas(inh_nodes, inh_scores, inh_counts)

    for u, name in exc_assign.items():
        G.nodes[u]["subtype"] = name
    for u, name in inh_assign.items():
        G.nodes[u]["subtype"] = name

    # Rename subtype -> ntype, polarity -> inhibitory (bool)
    for u in nodes:
        G.nodes[u]["ntype"] = f"{G.nodes[u]['subtype']}"
        G.nodes[u]["inhibitory"] = (G.nodes[u]["polarity"] == "I")

    # Remove subtype and pE if you want a cleaner graph
    for u in nodes:
        del G.nodes[u]["subtype"]
        del G.nodes[u]["pE"]

    return {
        "EI_bias_used": float(EI_weights.get("bias", 0.0) + bias_corr),
        "target_frac_exc": target_frac_exc,
        "achieved_frac_exc": float(np.mean([1.0 if G.nodes[u]["polarity"]=="E" else 0.0 for u in nodes])),
        "EXC_counts": exc_counts,
        "INH_counts": inh_counts,
        "n_exc": len(exc_nodes),
        "n_inh": len(inh_nodes),
        "solver": solver,
    }

# -------- Example ----------
# info = assign_EI_and_subtypes_with_quotas(
#     G,
#     target_frac_exc=0.8,
#     EXC_quota={"p23": 0.6, "TC": 0.4},       # 60/40 split among excitatory
#     INH_quota={"TI": 0.25, "TRN": 0.15, "b": 0.25, "nb": 0.2, "nb1": 0.15},
#     solver="greedy"  # or "flow"
# )
# print(info)
# # Node attributes set:
# #   G.nodes[u]["polarity"] in {"E","I"}
# #   G.nodes[u]["subtype"] among your subtype keys