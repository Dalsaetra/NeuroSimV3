import numpy as np
from neuron_population import NueronPopulation

class Connectome:
    def __init__(self, max_synapses, neuron_population, connectivity_probability, synapse_strengths):
        """
        Connectome class to represent the connectivity between neurons in a population.
        
        Parameters:
        max_synapses: int, maximum number of downstream synapses per neuron
        neuron_population: NueronPopulation object, the neuron population to connect
        connectivity_probability: array of global connectivity probability, shape (n_layers, n_layers, n_neuron_types, n_neuron_types)
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