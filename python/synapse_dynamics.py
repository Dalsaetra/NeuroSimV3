import numpy as np
from connectome import Connectome

import numba as nb

@nb.njit(parallel=True)
def masked_sum(mask, input_):
    n = mask.shape[0]
    out = np.empty(n, dtype=input_.dtype)
    for i in nb.prange(n):
        acc = 0.0
        for j in range(mask.shape[1]):
            for k in range(mask.shape[2]):
                if mask[i, j, k]:
                    acc += input_[j, k]
        out[i] = acc
    return out

@nb.njit(parallel=True)
def scatter_sum(M, WS, n_neurons):
    out = np.zeros(n_neurons, dtype=WS.dtype)
    for p in nb.prange(M.size):
        k = M.ravel()[p]
        if k >= 0:
            out[k] += WS.ravel()[p]
    return out

class SynapseDynamics:
    def __init__(self, connectome: Connectome, dt, tau_ST=5, tau_LT=150, 
                 E_AMPA=0, E_NMDA=0, E_GABA_A=-70, E_GABA_B=-90, NMDA_scale = 1.0, weight_mult = 0.002):
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

        # If spikes are all zeross then return
        if np.all(spikes == 0):
            return

        # Element-wise multiplication of spikes and weights, keeping the shape of spikes
        WS = np.multiply(spikes, self.connectome.W) # shape (n_neurons x max_synapses)
        # synaptic_input = scatter_sum(self.connectome.M, WS, self.connectome.neuron_population.n_neurons) # shape (n_neurons x 1)
        synaptic_input = np.bincount(self.connectome.M.ravel(), weights=WS.ravel(), minlength=self.connectome.M.shape[0])
        # Apply the synaptic input to the presynaptic neurons
        # Connectome.receivers: n_neurons x n_neurons x max_synapses
        # synaptic_input = np.einsum('ijk,jk->i', self.connectome.receivers, synaptic_input)
        # synaptic_input = masked_sum(self.connectome.receivers, synaptic_input)
        # synaptic_flat = synaptic_input.ravel()
        # synaptic_input = self.connectome.receivers2d @ synaptic_flat
        # synaptic_input = (self.connectome.receivers * synaptic_input).sum(axis=(1, 2))

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