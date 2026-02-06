import numpy as np
from src.synapse_dynamics import SynapseDynamics

class InputIntegration:
    def __init__(self, synapse_dynamics: SynapseDynamics, noise=200.0):
        """
        InputIntegration class to represent the input integration of a neuron population.
        """
        self.synapse_dynamics = synapse_dynamics
        self.noise = noise

    def __call__(self, neurons_V, I_ext=None):
        """
        neurons_V: n_neurons x 1
        Returns: n_neurons x 1
        """
        # Get the synaptic input from the synapse dynamics
        I_syn = self.synapse_dynamics(neurons_V)
        # Add the external input if provided
        if I_ext is not None:
            I_syn += I_ext
        if self.noise > 0:
            # Add noise to the synaptic input
            I_syn += np.random.normal(0, self.noise, size=I_syn.shape)
        return I_syn