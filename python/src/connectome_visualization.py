# connectome_visualization.py
import numpy as np
import networkx as nx

######################################################################
# 1.  Convert the matrices (M, NC, W, distances) in a Connectome into #
#     a DiGraph, adding 'distance' as an edge attribute.            #
######################################################################

def connectome_to_nx(connectome,
                     include_weights: bool = True,
                     include_self_loops: bool = False) -> nx.DiGraph:
    """
    Parameters
    ----------
    connectome :   your populated Connectome instance
    include_weights : copy W[i,j] onto the 'weight' edge attribute
    include_self_loops : keep autaptic edges if True

    Returns
    -------
    G : networkx.DiGraph  with node and edge attributes set, including:
          - 'weight' (if include_weights is True)
          - 'synapse_index'
          - 'distance'  (pulled from connectome.distances)
    """
    pop        = connectome.neuron_population
    M, NC, W   = connectome.M, connectome.NC, connectome.W
    D          = connectome.distances  # distances[i,j] was built in build_distances() :contentReference[oaicite:0]{index=0}

    # -------- nodes -------------------------------------------------
    G = nx.DiGraph()
    for i in range(pop.n_neurons):
        G.add_node(
            i,
            layer       = pop.get_layer(i),                # cortical/laminar layer
            inhibitory  = bool(pop.inhibitory_mask[i]),    # True = GABAergic
            ntype       = pop.type_from_neuron_index(i)    # original template name
        )

    # -------- edges -------------------------------------------------
    rows, cols = np.where(~NC)  # wherever NC[i,j] is False, there is a real synapse
    for i, j in zip(rows, cols):
        k = int(M[i, j])         # postsynaptic neuron index
        if (not include_self_loops) and (k == i):
            continue

        # Gather base attributes
        attrs = {
            'synapse_index': int(j),
            'distance': float(D[i, j])  # pull from connectome.distances :contentReference[oaicite:1]{index=1}
        }
        if include_weights:
            attrs['weight'] = float(W[i, j])
        G.add_edge(i, k, **attrs)

    return G


####################################
# 2a. Static spring-layout preview #
####################################

def quickplot(G,
              base_node_size: int = 50,
              seed: int = 42,
              figsize=(8, 8)):
    """
    Tiny helper for very small graphs (<~200 nodes).

    Now node sizes ∝ (in_degree + out_degree) of each node.
    The parameter base_node_size is multiplied by each node’s degree (plus one),
    so that an isolated node is still visible.
    """
    import matplotlib.pyplot as plt

    # Compute total degree for each node (in + out)
    deg_dict = dict(G.in_degree())  # in-degree
    for n, outdeg in G.out_degree():
        deg_dict[n] = deg_dict.get(n, 0) + outdeg

    # Build a list of node sizes: (degree + 1) * base_node_size
    node_sizes = [ (deg_dict.get(n, 0) + 1) * base_node_size for n in G.nodes ]

    # Color nodes by layer (fallback to single colour)
    layers = nx.get_node_attributes(G, 'layer')
    colours = [layers.get(n, 0) for n in G.nodes]

    pos = nx.spring_layout(G, seed=seed)
    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos,
                           node_size=node_sizes,
                           node_color=colours,
                           cmap=plt.cm.tab20,
                           linewidths=0.1)
    nx.draw_networkx_edges(G, pos,
                           arrows=True,
                           width=0.5,
                           alpha=0.7,
                           arrowsize=4)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


#########################################
# 2b. Interactive HTML with PyVis show() #
#########################################

def interactive(G,
                html_file: str = "connectome.html",
                notebook: bool = True,
                colour_scheme: str = "layer",
                node_size_scale: float = 2.0):
    """
    Write an HTML file you can open in any browser.

    colour_scheme = "layer"  |  "inhibitory"

    Nodes sizes are proportional to (in_degree + out_degree) * node_size_scale.

    Edges carry a 'distance' attribute from connectome.distances; we pass this
    as the 'length' parameter to PyVis so that each spring’s rest-length
    ≈ the anatomical distance between i→j.
    """
    try:
        from pyvis.network import Network
    except ImportError:
        raise ImportError("pyvis not found – install with `pip install pyvis`")

    net = Network(height="750px", width="100%", directed=True,
                  notebook=notebook, bgcolor="#ffffff")

    # Compute each node's total degree
    deg_in  = dict(G.in_degree())
    deg_out = dict(G.out_degree())
    total_deg = {n: deg_in.get(n, 0) + deg_out.get(n, 0) for n in G.nodes}

    # -------- nodes (with colours, tooltips, and dynamic sizes) ----------
    import matplotlib.colors as mcolors
    palette = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    layers = nx.get_node_attributes(G, 'layer')
    inhib  = nx.get_node_attributes(G, 'inhibitory')

    for n in G.nodes:
        # Determine color
        if colour_scheme == "inhibitory":
            color = "#d62728" if inhib[n] else "#1f77b4"
        else:  # by layer (default)
            color = palette[layers[n] % len(palette)]

        # Build hover title
        title = (
            f"Neuron {n}<br/>"
            f"Layer: {layers[n]}<br/>"
            f"{'Inhibitory' if inhib[n] else 'Excitatory'}<br/>"
            f"Template: {G.nodes[n]['ntype']}<br/>"
            f"InDegree: {deg_in.get(n, 0)}<br/>OutDegree: {deg_out.get(n, 0)}"
        )

        # Size = (total_degree + 1) * node_size_scale
        size = (total_deg.get(n, 0) + 1) * node_size_scale

        net.add_node(n,
                     label=str(n),
                     title=title,
                     color=color,
                     size=size)

    # -------- edges ------------
    for u, v, data in G.edges(data=True):
        w = data.get("weight", None)
        syn_idx = data.get("synapse_index", None)
        dist = data.get("distance", None)  # anatomical distance
        # If distance is missing or zero, fallback to a small positive default
        length = float(dist) if (dist is not None and dist > 0.0) else 50.0

        # Edge thickness ∝ |weight|
        thickness = abs(w) if w is not None else 1.0

        hover_txt = []
        if syn_idx is not None:
            hover_txt.append(f"synapse_index = {syn_idx}")
        if w is not None:
            hover_txt.append(f"weight = {w:.3f}")
        if dist is not None:
            hover_txt.append(f"distance = {dist:.2f}")

        net.add_edge(
            u,
            v,
            value   = thickness * 0.1,         # edge thickness ∝ |weight|
            title   = "<br/>".join(hover_txt),
            length  = length              # rest‐length ≈ anatomical distance
        )

    net.show(html_file)
    print(f"✅  Saved interactive graph to {html_file}")


############################
# 3. Minimal usage example #
############################
if __name__ == "__main__":
    # Dummy demo – remove once you wire this up to your real connectome
    from neuron_population import NeuronPopulation
    from connectome import Connectome

    neurons_per_layer  = [5, 5]
    neuron_types       = ["nb1", "p23"]
    inhibitory         = [False, True]
    neuron_distribution = [np.array([0.8, 0.2]),
                           np.array([0.2, 0.8])]
    layer_distances    = np.zeros((2, 2))
    pop  = NeuronPopulation(neurons_per_layer,
                             neuron_distribution,
                             layer_distances,
                             neuron_types,
                             inhibitory,
                             threshold_decay=0.99)

    # toy, fully-connected probability tensor:
    conn_prob = np.ones((2, 2, len(neuron_types), len(neuron_types)+2)) * 0.5
    syn_str   = np.ones((2, 2))

    con = Connectome(max_synapses=3,
                     neuron_population=pop,
                     connectivity_probability=conn_prob,
                     synapse_strengths=syn_str)
    # At this point, con.distances was populated via build_distances() :contentReference[oaicite:2]{index=2}

    G = connectome_to_nx(con)            # build the graph (with distances & weights)
    quickplot(G)                         # static preview (node sizes ∝ degree)
    interactive(G, "my_net.html")        # interactive HTML:
                                          #   - edge lengths ≈ con.distances[i,j]
                                          #   - node sizes ∝ (in_degree + out_degree)
