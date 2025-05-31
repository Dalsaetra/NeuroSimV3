import numpy as np
import networkx as nx

######################################################################
# 1.  Convert the matrices (M, NC, W) in a Connectome into a DiGraph #
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
    G : networkx.DiGraph  with useful node/edge attributes set
    """
    pop      = connectome.neuron_population
    M, NC, W = connectome.M, connectome.NC, connectome.W
    distances = connectome.distances

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
    rows, cols = np.where(~NC)     # only where a connection exists
    for i, j in zip(rows, cols):
        k = int(M[i, j])           # postsynaptic neuron
        if not include_self_loops and k == i:
            continue
        attrs = {'synapse_index': int(j)}
        if include_weights:
            attrs['weight'] = float(W[i, j])
        G.add_edge(i, k, **attrs)

    return G


####################################
# 2a. Static spring-layout preview #
####################################

def quickplot(G,
              node_size: int = 80,
              seed: int = 42,
              figsize=(8, 8)):
    """
    Tiny helper for very small graphs (<~200 nodes).
    """
    import matplotlib.pyplot as plt
    layers = nx.get_node_attributes(G, 'layer')
    # colour nodes by layer (fallback to single colour)
    colours = [layers.get(n, 0) for n in G.nodes]
    pos = nx.spring_layout(G, seed=seed)
    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos,
                           node_size=node_size,
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
                colour_scheme: str = "layer"):
    """
    Write an HTML file you can open in any browser.

    colour_scheme = "layer"  |  "inhibitory"
    """
    try:
        from pyvis.network import Network
    except ImportError:
        raise ImportError("pyvis not found – install with `pip install pyvis`")

    net = Network(height="750px", width="100%", directed=True,
                  notebook=notebook, bgcolor="#ffffff")

    # -------- nodes (with colours & tooltips) -----------------------
    import matplotlib.colors as mcolors
    palette = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    layers = nx.get_node_attributes(G, 'layer')
    inhib  = nx.get_node_attributes(G, 'inhibitory')

    for n in G.nodes:
        if colour_scheme == "inhibitory":
            color = "#d62728" if inhib[n] else "#1f77b4"
        else:                           # by layer (default)
            color = palette[layers[n] % len(palette)]
        title = (f"Neuron {n}<br/>Layer: {layers[n]}<br/>"
                 f"{'Inhibitory' if inhib[n] else 'Excitatory'}<br/>"
                 f"Template: {G.nodes[n]['ntype']}")
        net.add_node(n, label=str(n), title=title, color=color)

    # -------- edges -------------------------------------------------
    for u, v, data in G.edges(data=True):
        weight = abs(data.get("weight", 1.0))
        net.add_edge(u, v,
                     value=weight,         # edge thickness
                     title=f"w = {weight:.3f}")

    net.show(html_file)
    print(f"Saved interactive graph to {html_file}")


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

    G = connectome_to_nx(con)
    quickplot(G)              # small static preview
    interactive(G)            # full interactive HTML
