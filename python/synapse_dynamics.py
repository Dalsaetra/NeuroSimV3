import numpy as np
from connectome import Connectome

class SynapseDynamics:
    def __init__(self, connectome: Connectome, dt, tau_ST=5, tau_LT=150, 
                 E_AMPA=0, E_NMDA=0, E_GABA_A=-70, E_GABA_B=-90, NMDA_scale = 0.1, weight_mult = 1.0):
        """
        SynapseDynamics class to represent the synaptic dynamics of a neuron population.
        """

        self.connectome = connectome

        self.inhibitory_mask = connectome.neuron_population.inhibitory_mask
        self.excitatory_mask = ~self.inhibitory_mask

        self.dt = dt

        self.tau_ST = tau_ST
        self.tau_LT = tau_LT
        self.E_ST = np.where(self.inhibitory_mask, E_GABA_A, E_AMPA)
        self.E_LT = np.where(self.inhibitory_mask, E_GABA_B, E_NMDA)

        self.g_ST = np.zeros(self.connectome.neuron_population.n_neurons, dtype=float)
        self.g_LT = np.zeros(self.connectome.neuron_population.n_neurons, dtype=float)

        self.ST_decay = np.exp(-dt / tau_ST)
        self.LT_decay = np.exp(-dt / tau_LT)

        self.weight_mult = weight_mult
        self.NMDA_scale = NMDA_scale


    def decay(self):
        # Decay the synaptic conductances
        self.g_ST *= self.ST_decay
        self.g_LT *= self.LT_decay


    def spike_input(self, spikes):
        # spikes: n_neurons x max_synapses

        # Element-wise multiplication of spikes and weights, keeping the shape of spikes
        synaptic_input = np.multiply(spikes, self.connectome.W)
        # Apply the synaptic input to the presynaptic neurons
        synaptic_input = (self.connectome.receivers * synaptic_input).sum(axis=(1, 2))

        self.g_ST += synaptic_input
        self.g_LT += synaptic_input

    def __call__(self, neurons_V):
        # neurons_V: n_neurons x 1
        # I_ext: n_neurons x 1
        # Returns: n_neurons x 1

        V_shifted = (neurons_V + 80) / 60
        NMDA_factor = V_shifted**2 / (1 + V_shifted**2) * self.NMDA_scale
        I_ST = self.g_ST * (self.E_ST - neurons_V)
        # Combine NMDA and GABA_B calculations
        V_diff = self.E_LT - neurons_V
        I_LT = self.g_LT * V_diff * (NMDA_factor * self.excitatory_mask + self.inhibitory_mask)

        return (I_ST + I_LT) * self.weight_mult