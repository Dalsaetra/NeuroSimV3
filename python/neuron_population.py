import numpy as np
from neuron_templates import neuron_type_IZ

class NueronPopulation:
    def __init__(self, layer_distribution, neuron_distribution, layer_distances, neuron_types, inhibitory, n_params=13):
        """
        layer_distribution: list of integers, number of neurons in each layer
        neuron_distribution: list with length layer_distribution of np.arrays of length neuron_types, probabilities of each neuron type in each layer
        layer_distances: matrix of distances between layers, shape (n_layers, n_layers)
        neuron_types: list of neuron types (string), keys for types un neuron_templates.py
        inhibitory: list of booleans, True if the neuron is inhibitory, False if it is excitatory, shape as neuron_types
        n_params: number of parameters for each neuron, default is 13 (k, a, b, d, C, Vr, Vt, Vpeak, c, delta_V, bias, threshold_mult, threshold_decay)
        """

        self.layer_distribution = layer_distribution
        self.neuron_distribution = neuron_distribution
        self.layer_distances = layer_distances
        self.neuron_types = neuron_types
        self.inhibitory = inhibitory
        self.n_params = n_params
        self.n_neuron_types = len(neuron_types)

        # Initialize the neuron population
        self.n_layers = len(layer_distribution)
        self.n_neurons = sum(layer_distribution)

        self.neuron_population = np.zeros((self.n_neurons, self.n_params))
        self.inhibitory_mask = np.zeros((self.n_neurons), dtype=bool)
        self.neuron_population_types = []
        self.layer_indices = []

    def populate(self, threshold_decay, delta_V=2.5, bias=0.0, threshold_mult=1.05):
        """
        Populate the neuron population with random neurons from the neuron templates
        """
        n_template_params = len(neuron_type_IZ["nb1"]) # Usually 9

        # Populate the neuron population with random neurons from the neuron templates
        for i in range(self.n_layers):
            layer_indices_layer = []
            for j in range(self.layer_distribution[i]):
                index = sum(self.layer_distribution[:i]) + j
                layer_indices_layer.append(index)
                # Get the neuron type for this neuron
                neuron_type = np.random.choice(self.neuron_types, p=self.neuron_distribution[i])
                # Get the neuron type index
                neuron_type_index = self.get_neuron_type_index(neuron_type)
                # Set the inhibitory mask for this neuron
                self.inhibitory_mask[index] = self.inhibitory[neuron_type_index]
                # Store the neuron type in the neuron population
                self.neuron_population_types.append(neuron_type)
                # Get the parameters for this neuron type
                params = neuron_type_IZ[neuron_type]
                # Set the parameters for this neuron
                self.neuron_population[index][:n_template_params] = params

            self.layer_indices.append(np.array(layer_indices_layer))

        self.neuron_population[:, 9] = delta_V
        self.neuron_population[:, 10] = bias
        self.neuron_population[:, 11] = threshold_mult
        self.neuron_population[:, 12] = threshold_decay

    def get_layer(self, neuron_index):
        """
        Get the layer of a neuron given its index
        """
        for i in range(self.n_layers):
            if neuron_index in self.layer_indices[i]:
                return i
        return None
    
    def get_neuron_type(self, neuron_index):
        """
        Get the neuron type of a neuron given its index
        """
        return self.neuron_population_types[neuron_index]
    
    def get_neuron_type_index(self, neuron_type):
        """
        Get the index of a neuron type given its name
        """
        return self.neuron_types.index(neuron_type)