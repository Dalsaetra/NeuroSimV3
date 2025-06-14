import numpy as np
from neuron_templates import neuron_type_IZ

class NeuronPopulation:
    def __init__(self, neurons_per_layer, neuron_distribution, neuron_types, 
                 inhibitory, threshold_decay, layer_distances=None, delta_V=2.5, 
                 bias=0.0, threshold_mult=1.05, n_params=13, auto_populate=True):
        """
        neurons_per_layer: list of integers, number of neurons in each layer
        neuron_distribution: list with length neurons_per_layer of np.arrays of length neuron_types, probabilities of each neuron type in each layer
        layer_distances: matrix of distances between layers, shape (n_layers, n_layers)
        neuron_types: list of neuron types (string), keys for types in neuron_templates.py
        inhibitory: list of booleans, True if the neuron type is inhibitory, False if it is excitatory, shape as neuron_types
        n_params: number of parameters for each neuron, default is 13 (k, a, b, d, C, Vr, Vt, Vpeak, c, delta_V, bias, threshold_mult, threshold_decay)
        """

        self.neurons_per_layer = neurons_per_layer
        self.neuron_distribution = neuron_distribution
        self.layer_distances = layer_distances
        self.neuron_types = neuron_types
        self.inhibitory = inhibitory
        self.n_params = n_params

        # Initialize the neuron population
        self.n_layers = len(neurons_per_layer)
        self.n_neurons = sum(neurons_per_layer)

        self.neuron_population = np.zeros((self.n_neurons, self.n_params))
        self.inhibitory_mask = np.zeros((self.n_neurons), dtype=bool)
        self.neuron_population_types = []
        self.layer_indices = []

        if auto_populate:
            self.populate(threshold_decay, delta_V, bias, threshold_mult)
            self.n_neuron_types = len(neuron_types)
            if self.layer_distances is None:
                # If no layer distances are provided, build a linear distance matrix
                self.build_linear_layerdistance_matrix()


        # Weight inhibitory mask, 1 for excitatory, -1 for inhibitory
        self.weight_inhibitory_mask = np.where(self.inhibitory_mask, -1, 1)
        

    def populate(self, threshold_decay, delta_V, bias, threshold_mult):
        """
        Populate the neuron population with random neurons from the neuron templates
        """
        n_template_params = len(neuron_type_IZ["nb1"]) # Usually 9

        # Populate the neuron population with random neurons from the neuron templates
        for i in range(self.n_layers):
            layer_indices_layer = []
            self.dist_norm = self.neuron_distribution[i] / np.sum(self.neuron_distribution[i])
            for j in range(self.neurons_per_layer[i]):
                index = sum(self.neurons_per_layer[:i]) + j
                layer_indices_layer.append(index)
                # Get the neuron type for this neuron
                # Normalize distribution to sum to 1
                neuron_type = np.random.choice(self.neuron_types, p=self.dist_norm)
                # Get the neuron type index
                neuron_type_index = self.type_index_from_neuron_type(neuron_type)
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
    
    def type_from_neuron_index(self, neuron_index):
        """
        Get the neuron type of a neuron given its index
        """
        return self.neuron_population_types[neuron_index]
    
    def type_index_from_neuron_type(self, neuron_type):
        """
        Get the index of a neuron type given its name
        """
        return self.neuron_types.index(neuron_type)
    
    def type_index_from_neuron_index(self, neuron_index):
        """
        Get the index of a neuron type given its index
        """
        neuron_type = self.type_from_neuron_index(neuron_index)
        return self.type_index_from_neuron_type(neuron_type)
    
    def get_neurons_from_layer(self, layer):
        """
        Get the neurons from a layer given its index
        """
        return self.layer_indices[layer]
    
    def get_types_from_layer(self, layer):
        """
        Get the neuron types from a layer given its index
        """
        return np.array(self.neuron_population_types)[self.get_neurons_from_layer(layer)]


    def build_linear_layerdistance_matrix(self, inter_distance=0.6, layer_distance=5.0):
        """
        Build a linear distance matrix for the neuron population.
        inter_distance: distance between neurons in the same layer
        layer_distance: distance between layers
        """
        self.layer_distances = np.zeros((self.n_layers, self.n_layers))

        for i in range(self.n_layers):
            for j in range(self.n_layers):
                if i == j:
                    self.layer_distances[i, j] = inter_distance 
                else:
                    self.layer_distances[i, j] = layer_distance * abs(i - j)


