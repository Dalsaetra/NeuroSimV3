import numpy as np
from neuron_population import NeuronPopulation
import networkx as nx
import matplotlib.pyplot as plt

class Connectome:
    def __init__(self, max_synapses, neuron_population: NeuronPopulation, connectivity_probability, synapse_strengths=None,
                 autobuild=True):
        """
        Connectome class to represent the connectivity between neurons in a population.
        
        Parameters:
        max_synapses: int, maximum number of downstream synapses per neuron
        neuron_population: NeuronPopulation object, the neuron population to connect
        connectivity_probability: array of global connectivity probability, shape (n_layers, n_layers, n_neuron_types, n_neuron_types+2), -2 dimension is for autaptic connections, -1 for no connection
        synapse_strengths: array of synapse weight scale, shape (n_layers, n_layers)
        """
        assert max_synapses <= neuron_population.n_neurons, "max_synapses must be less than or equal to the number of neurons in the population."

        self.max_synapses = max_synapses
        self.neuron_population = neuron_population
        self.connectivity_probability = connectivity_probability
        self.synapse_strengths = synapse_strengths

        if self.synapse_strengths is None:
            # If no synapse strengths are provided, use a default value of 1.0
            self.synapse_strengths = np.ones((self.neuron_population.n_layers, self.neuron_population.n_layers))

        # Connectivity matrix
        # M[i, j] = k, k is the neuron index where the jth axon of the ith neuron ends up
        self.M = np.zeros((self.neuron_population.n_neurons, max_synapses), dtype=int)
        
        # Weight matrix
        self.W = np.zeros((self.neuron_population.n_neurons, max_synapses), dtype=float)

        # No-connection matrix
        self.NC = np.zeros((self.neuron_population.n_neurons, max_synapses), dtype=bool)

        # Dendritic markers
        self.dendritic = np.zeros((self.neuron_population.n_neurons, max_synapses), dtype=bool)

        if autobuild:
            assert len(connectivity_probability.shape) == 4, "connectivity_probability must be a 4D array."
            assert len(synapse_strengths.shape) == 2, "synapse_strengths must be a 2D array."
            self.build_connectome()
            self.build_distances()
            self.build_nx()

        # Invert NC
        # self.NC_invert = ~self.NC

    def set_connection(self, i, j, k, w = None):
        """
        Set the connection from neuron i to neuron j.
        
        Parameters:
        i: int, index of the presynaptic neuron
        j: int, index of the postsynaptic neuron
        """
        if j < self.max_synapses:
            self.M[i, j] = k
            if w is not None:
                self.W[i, j] = w
        else:
            raise ValueError("j must be less than max_synapses.")
        
    def get_random_weight(self, layer_from, layer_to):
        """
        Get a random weight for the connection between two layers.
        
        Parameters:
        layer_from: int, index of the presynaptic layer
        layer_to: int, index of the postsynaptic layer
        
        Returns:
        float, random weight for the connection
        """
        weight = np.random.normal(loc=self.synapse_strengths[layer_from, layer_to], scale=0.5)
        # Ensure the weight is positive
        weight = max(weight, 0.0)
        return weight
    
    def set_random_weights(self):
        """
        Set random weights for all connections in the connectome.
        """
        for i in range(self.neuron_population.n_neurons):
            layer_from = self.neuron_population.get_layer(i)
            for j in range(self.max_synapses):
                if not self.NC[i, j]:
                    layer_to = self.neuron_population.get_layer(self.M[i, j])
                    self.W[i, j] = self.get_random_weight(layer_from, layer_to)


    def build_connectome(self):
        """
        Build the connectome by generating the connectivity matrix and weight matrix.
        """
        # Generate random connectivity based on the connectivity probability
        for i in range(self.neuron_population.n_neurons):
            neuron_type = self.neuron_population.type_index_from_neuron_index(i)
            layer = self.neuron_population.get_layer(i)
            # connectivity_probability: shape (n_layers, n_layers, n_neuron_types, n_neuron_types+2)
            # Relevant connectivity
            # NOTE that ideally this should be scaled by the number of each neuron type in the layer
            connectivity_layer = self.connectivity_probability[layer, :, neuron_type, :]
            # Create probability distribution for the downstream neurons
            donwstream_neuron_probs = np.zeros(self.neuron_population.n_neurons, dtype=float)
            for k in range(self.neuron_population.n_neurons):
                if k != i:
                    # Get the layer and neuron type of the downstream neuron
                    downstream_layer = self.neuron_population.get_layer(k)
                    downstream_neuron_type = self.neuron_population.type_index_from_neuron_index(k)
                    # Get the connectivity probability
                    prob = connectivity_layer[downstream_layer, downstream_neuron_type]
                    donwstream_neuron_probs[k] = prob

            normalizer = np.sum(donwstream_neuron_probs)

            # Normalize the probabilities
            if np.abs(normalizer) < 1e-10:
                # If no connections are possible, draw autaptic connection only
                autaptic_prob = self.connectivity_probability[layer, layer, neuron_type, -2]
                if np.random.rand() < autaptic_prob:
                    # Autaptic connection
                    target_neuron = i
                    self.set_connection(i, 0, i, self.get_random_weight(layer, layer))
                else:
                    # No connection
                    self.NC[i, 0] = True

                # And set rest of connections to no connection
                for j in range(1, self.max_synapses):
                    self.NC[i, j] = True
                continue


            donwstream_neuron_probs /= normalizer

            autaptic_drawn = False
            for j in range(self.max_synapses):
                # First draw if we have an autaptic connection
                if not autaptic_drawn:
                    autaptic_drawn = True
                    # Autaptic connection probability
                    autaptic_prob = self.connectivity_probability[layer, layer, neuron_type, -2]
                    if np.random.rand() < autaptic_prob:
                        # Autaptic connection
                        target_neuron = i
                        self.set_connection(i, j, i, self.get_random_weight(layer, layer))
                        continue

                # Then draw if we have a downstream connection
                no_connection = self.connectivity_probability[layer, layer, neuron_type, -1]
                if np.random.rand() < no_connection:
                    # No connection
                    self.NC[i, j] = True
                else:
                    # Draw a downstream neuron, autaptic cant be drawn again since prob is 0
                    target_neuron = np.random.choice(np.arange(self.neuron_population.n_neurons), p=donwstream_neuron_probs)
                    # Set the connection
                    # NOTE that this allows for multiple connections to the same neuron
                    self.set_connection(i, j, target_neuron, self.get_random_weight(layer, self.neuron_population.get_layer(target_neuron)))

    
    def build_distances(self):
        """
        Build the distances between neurons in the connectome.
        """
        # Build the distances between neurons in the connectome
        self.distances = np.zeros_like(self.M, dtype=float)
        for i in range(self.neuron_population.n_neurons):
            layer_i = self.neuron_population.get_layer(i)
            for j in range(self.max_synapses):
                if not self.NC[i, j]:
                    layer_j = self.neuron_population.get_layer(self.M[i, j])
                    # Get the distance from distance matrix, in mm
                    self.distances[i, j] = self.neuron_population.layer_distances[layer_i, layer_j] + np.random.normal(0, 0.2)


    def connections_per_neuron(self):
        """
        Get the number of connections per neuron in the connectome.
        
        Returns:
        array, shape (n_neurons,), number of connections per neuron
        """
        # Count the number of connections per neuron
        # Outward connections are the number of downstream synapses
        outward_connections = np.sum(~self.NC, axis=1)
        # Inward connections are the number of upstream synapses, must use M to count them
        inward_connections = np.zeros(self.neuron_population.n_neurons, dtype=int)
        for i in range(self.neuron_population.n_neurons):
            inward_connections[self.M[i, :]] += 1
            # Minus the NC
            inward_connections[self.M[i, self.NC[i, :]]] -= 1

        total_connections = outward_connections + inward_connections

        return outward_connections, inward_connections, total_connections

    def plot_connections_histogram(self, figsize=(8,5)):
        """
        Plot histogram of the number of connections per neuron in the connectome.
        
        Parameters:
        bins: int, number of bins for the histogram
        
        Returns:
        tuple, (histogram, bin_edges)
        """
        outward_connections, inward_connections, total_connections = self.connections_per_neuron()
        bins = np.max(total_connections) - np.min(total_connections)
        # Plot histogram of the number of connections per neuron
        histogram, bin_edges = np.histogram(total_connections, bins=bins)
        plt.figure(figsize=figsize)
        plt.bar(bin_edges[:-1], histogram, width=np.diff(bin_edges), edgecolor='black', align='edge')
        plt.xlabel('Number of Connections per Neuron')
        plt.ylabel('Number of Neurons')
        plt.title('Histogram of Connections per Neuron')
        plt.xticks(bin_edges, rotation=45)
        plt.tight_layout()
        plt.show()

        return histogram, bin_edges


    def build_nx(self, include_weights: bool = True, include_self_loops: bool = True):
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
        pop        = self.neuron_population
        M, NC, W   = self.M, self.NC, self.W
        D          = self.distances  # distances[i,j] was built in build_distances() :contentReference[oaicite:0]{index=0}

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

        self.G = G

    def evaluate_small_world(self, legend=True):
        """
        Evaluate the small-world properties of the connectome.
        
        Returns:
        dict, small-world properties including clustering coefficient and average path length
        """
        if not hasattr(self, 'G'):
            self.build_nx()

        Gu = self.G.to_undirected()
        largest_cc = max(nx.connected_components(Gu), key=len)
        self.G_cc = Gu.subgraph(largest_cc).copy()# --- 5. Clustering & transitivity (undirected)

        self.C_G = nx.average_clustering(self.G_cc)
        self.T_G = nx.transitivity(self.G_cc)

        self.L_G = nx.average_shortest_path_length(self.G_cc)
        self.diam_G = nx.diameter(self.G_cc)


        n_cc, m_cc = self.G_cc.number_of_nodes(), self.G_cc.number_of_edges()
        R = nx.gnm_random_graph(n = n_cc, m = m_cc)

        C_R = nx.average_clustering(R)
        L_R = nx.average_shortest_path_length(R)
        self.sigma  = (self.C_G / C_R) / (self.L_G / L_R)

        # Find small-world measure omega as well
        # Need to find C_lattice, which is the clustering coefficient of a regular lattice of the same size
        # Use a ring lattice (watts_strogatz_graph with p=0) with average degree similar to our graph
        k = int(2 * m_cc / n_cc)  # Average degree
        k = max(k, 4)  # Ensure k is at least 4 for meaningful clustering
        k = min(k, n_cc - 1)  # Ensure k doesn't exceed n-1
        lattic_graph = nx.watts_strogatz_graph(n=n_cc, k=k, p=0)  # p=0 makes it a pure ring lattice
        C_lattice = nx.average_clustering(lattic_graph)

        self.omega = L_R / self.L_G - self.C_G / C_lattice 

        if legend:
            print(f"Sigma (small-world if > 1): {self.sigma:.4f}")
            print(f"Omega (small-world if close to 0): {self.omega:.4f}")

        return self.sigma, self.omega

    def quickplot(self, base_node_size: int = 50, figsize=(8, 8)):
        """
        Tiny helper for very small graphs (<~200 nodes).

        Now node sizes ∝ (in_degree + out_degree) of each node.
        The parameter base_node_size is multiplied by each node's degree (plus one),
        so that an isolated node is still visible.
        """
        import matplotlib.pyplot as plt

        # Compute total degree for each node (in + out)
        deg_dict = dict(self.G.in_degree())  # in-degree
        for n, outdeg in self.G.out_degree():
            deg_dict[n] = deg_dict.get(n, 0) + outdeg

        # Build a list of node sizes: (degree + 1) * base_node_size
        node_sizes = [ (deg_dict.get(n, 0) + 1) * base_node_size for n in self.G.nodes ]

        # Color nodes by layer (fallback to single colour)
        layers = nx.get_node_attributes(self.G, 'layer')
        colours = [layers.get(n, 0) for n in self.G.nodes]

        pos = nx.spring_layout(self.G)
        plt.figure(figsize=figsize)
        nx.draw_networkx_nodes(self.G, pos,
                            node_size=node_sizes,
                            node_color=colours,
                            cmap=plt.cm.tab20,
                            linewidths=0.1)
        nx.draw_networkx_edges(self.G, pos,
                            arrows=True,
                            width=0.5,
                            alpha=0.7,
                            arrowsize=4)
        plt.axis('off')
        plt.tight_layout()
        plt.show()


    def interactive(self, html_file: str = "connectome.html", notebook: bool = True,
                    colour_scheme: str = "layer", node_size_scale: float = 2.0):
        """
        Write an HTML file you can open in any browser.

        colour_scheme = "layer"  |  "inhibitory"

        Nodes sizes are proportional to (in_degree + out_degree) * node_size_scale.

        Edges carry a 'distance' attribute from connectome.distances; we pass this
        as the 'length' parameter to PyVis so that each spring's rest-length
        ≈ the anatomical distance between i→j.
        """
        try:
            from pyvis.network import Network
        except ImportError:
            raise ImportError("pyvis not found - install with `pip install pyvis`")

        net = Network(height="750px", width="100%", directed=True,
                    notebook=notebook, bgcolor="#ffffff")
        net.repulsion()

        # Compute each node's total degree
        deg_in  = dict(self.G.in_degree())
        deg_out = dict(self.G.out_degree())
        total_deg = {n: deg_in.get(n, 0) + deg_out.get(n, 0) for n in self.G.nodes}

        # -------- nodes (with colours, tooltips, and dynamic sizes) ----------
        import matplotlib.colors as mcolors
        palette = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
        layers = nx.get_node_attributes(self.G, 'layer')
        inhib  = nx.get_node_attributes(self.G, 'inhibitory')

        for n in self.G.nodes:
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
                f"Template: {self.G.nodes[n]['ntype']}<br/>"
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
        for u, v, data in self.G.edges(data=True):
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
                length  = length * 20              # rest‐length ≈ anatomical distance
            )

        net.show(html_file)
        print(f"Saved interactive graph to {html_file}")