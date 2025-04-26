import numpy as np
from neuron_population import NueronPopulation

class Connectome:
    def __init__(self, max_synapses, neuron_population: NueronPopulation, connectivity_probability, synapse_strengths):
        """
        Connectome class to represent the connectivity between neurons in a population.
        
        Parameters:
        max_synapses: int, maximum number of downstream synapses per neuron
        neuron_population: NueronPopulation object, the neuron population to connect
        connectivity_probability: array of global connectivity probability, shape (n_layers, n_layers, n_neuron_types, n_neuron_types+2), -2 dimension is for autaptic connections, -1 for no connection
        synapse_strengths: array of synapse weight scale, shape (n_layers, n_layers)
        """
        assert max_synapses <= neuron_population.n_neurons, "max_synapses must be less than or equal to the number of neurons in the population."
        assert len(connectivity_probability.shape) == 4, "connectivity_probability must be a 4D array."
        assert len(synapse_strengths.shape) == 2, "synapse_strengths must be a 2D array."

        self.max_synapses = max_synapses
        self.neuron_population = neuron_population
        self.connectivity_probability = connectivity_probability
        self.synapse_strengths = synapse_strengths

        # Connectivity matrix
        # M[i, j] = k, k is the neuron index where the jth axon of the ith neuron ends up
        self.M = np.zeros((self.neuron_population.n_neurons, max_synapses), dtype=int)
        
        # Weight matrix
        self.W = np.zeros((self.neuron_population.n_neurons, max_synapses), dtype=float)

        # No-connection matrix
        self.NC = np.zeros((self.neuron_population.n_neurons, max_synapses), dtype=bool)

        self.build_connectome()
        self.build_receivers()
        self.build_distances()

    def set_connection(self, i, j, w = None):
        """
        Set the connection from neuron i to neuron j.
        
        Parameters:
        i: int, index of the presynaptic neuron
        j: int, index of the postsynaptic neuron
        """
        if j < self.max_synapses:
            self.M[i, j] = j
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
        return np.random.normal(loc=self.synapse_strengths[layer_from, layer_to], scale=0.1)

    def build_connectome(self):
        """
        Build the connectome by generating the connectivity matrix and weight matrix.
        """
        # Generate random connectivity based on the connectivity probability
        for i in range(self.neuron_population.n_neurons):
            neuron_type = self.neuron_population.get_neuron_type(i)
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
                    downstream_neuron_type = self.neuron_population.get_neuron_type_index(k)
                    # Get the connectivity probability
                    prob = connectivity_layer[downstream_layer, downstream_neuron_type]
                    donwstream_neuron_probs[k] = prob

            autaptic_drawn = False
            for j in range(self.max_synapses):
                # First draw if we have an autoptic connection
                if not autaptic_drawn:
                    autaptic_drawn = True
                    # Autaptic connection probability
                    autaptic_prob = self.connectivity_probability[layer, layer, neuron_type, -2]
                    if np.random.rand() < autaptic_prob:
                        # Autaptic connection
                        target_neuron = i
                        self.set_connection(i, j, self.get_random_weight(layer, layer))
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
                    self.set_connection(i, j, self.get_random_weight(layer, self.neuron_population.get_layer(target_neuron)))

    def build_receivers(self):
        """
        Build the receivers for the connectome.
        """
        # Build the receivers for the connectome
        self.receivers = np.zeros((self.neuron_population.n_neurons, self.neuron_population.n_neurons, self.max_synapses), dtype=bool)
        for i in range(self.neuron_population.n_neurons):
            # Get where neuron i is downstream
            self.receivers[i][self.M == i] = True

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
                    # Get the distance from distance matrix
                    self.distances[i, j] = self.neuron_population.layer_distances[layer_i, layer_j]