import numpy as np
from connectome import Connectome

import numba as nb

@nb.njit(parallel=False, fastmath=True, cache=True)
def _syn_current_numba(neurons_V,
                       g_AMPA, E_AMPA,
                       g_NMDA, E_NMDA,
                       g_GABA_A, E_GABA_A,
                       g_GABA_B, E_GABA_B,
                       LT_scale: float,
                       weight_mult: float):          # <── scalar
    """
    Compute total synaptic current for every neuron.

    All arrays are 1-D with length n_neurons; NMDA_scale and weight_mult are scalars.
    Returns a 1-D float array of length n_neurons.
    """
    n = neurons_V.size
    out = np.empty(n, dtype=neurons_V.dtype)

    for i in nb.prange(n):          # multi-threaded loop over neurons
        V = neurons_V[i]

        # NMDA voltage dependence
        V_shift = (V + 80.0) / 60.0
        v2      = V_shift * V_shift
        nmda    = (v2 / (1.0 + v2))

        # Short-term current (AMPA / GABA-A, etc.)
        I_AMPA = g_AMPA[i] * (E_AMPA - V)
        I_NMDA = g_NMDA[i] * (E_NMDA - V) * nmda * LT_scale
        I_GABA_A = g_GABA_A[i] * (E_GABA_A - V)
        I_GABA_B = g_GABA_B[i] * (E_GABA_B - V) * LT_scale

        # Apply global weight multiplier (scalar)
        out[i] = (I_AMPA + I_NMDA + I_GABA_A + I_GABA_B) * weight_mult

    return out

class SynapseDynamics:
    def __init__(self, connectome: Connectome, dt, tau_AMPA=5, tau_NMDA=150, tau_GABA_A=6, tau_GABA_B=150,
                 E_AMPA=0, E_NMDA=0, E_GABA_A=-70, E_GABA_B=-90, LT_scale = 1.0, weight_mult = 0.5):
        """
        SynapseDynamics class to represent the synaptic dynamics of a neuron population.
        """

        self.connectome = connectome

        self.inhibitory_mask = connectome.neuron_population.inhibitory_mask
        self.excitatory_mask = ~self.inhibitory_mask

        self.dt = dt

        self.tau_AMPA = tau_AMPA
        self.tau_NMDA = tau_NMDA
        self.tau_GABA_A = tau_GABA_A
        self.tau_GABA_B = tau_GABA_B

        self.E_AMPA = E_AMPA
        self.E_NMDA = E_NMDA
        self.E_GABA_A = E_GABA_A
        self.E_GABA_B = E_GABA_B


        self.g_AMPA = np.zeros(self.connectome.neuron_population.n_neurons, dtype=float)
        self.g_NMDA = np.zeros(self.connectome.neuron_population.n_neurons, dtype=float)
        self.g_GABA_A = np.zeros(self.connectome.neuron_population.n_neurons, dtype=float)
        self.g_GABA_B = np.zeros(self.connectome.neuron_population.n_neurons, dtype=float)

        self.AMPA_decay = np.exp(-dt / tau_AMPA)
        self.NMDA_decay = np.exp(-dt / tau_NMDA)
        self.GABA_A_decay = np.exp(-dt / tau_GABA_A)
        self.GABA_B_decay = np.exp(-dt / tau_GABA_B)

        self.weight_mult = weight_mult
        self.LT_scale = LT_scale


    def decay(self):
        # Decay the synaptic conductances
        self.g_AMPA *= self.AMPA_decay
        self.g_NMDA *= self.NMDA_decay
        self.g_GABA_A *= self.GABA_A_decay
        self.g_GABA_B *= self.GABA_B_decay


    def spike_input(self, spikes):
        # spikes: n_neurons x max_synapses

        if not np.all(spikes[self.inhibitory_mask] == 0):
            # Element-wise multiplication of spikes and weights, keeping the shape of spikes
            WS = np.multiply(spikes[self.inhibitory_mask], self.connectome.W[self.inhibitory_mask]) # shape (n_neurons x max_synapses)
            inhib_input = np.bincount(self.connectome.M[self.inhibitory_mask].ravel(), weights=WS.ravel(), minlength=self.connectome.M.shape[0])
            self.g_GABA_A += inhib_input
            self.g_GABA_B += inhib_input 


        if not np.all(spikes[self.excitatory_mask] == 0):
            # Element-wise multiplication of spikes and weights, keeping the shape of spikes
            WS = np.multiply(spikes[self.excitatory_mask], self.connectome.W[self.excitatory_mask]) # shape (n_neurons x max_synapses)
            excit_input = np.bincount(self.connectome.M[self.excitatory_mask].ravel(), weights=WS.ravel(), minlength=self.connectome.M.shape[0])
            self.g_AMPA += excit_input
            self.g_NMDA += excit_input 

    # def __call__(self, neurons_V):
    #     # neurons_V: n_neurons x 1
    #     # Returns: n_neurons x 1

    #     V_shifted = (neurons_V + 80) / 60
    #     NMDA_factor = V_shifted**2 / (1 + V_shifted**2) * self.NMDA_scale
    #     I_ST = self.g_ST * (self.E_ST - neurons_V)
    #     # Combine NMDA and GABA_B calculations
    #     V_diff = self.E_LT - neurons_V
    #     I_LT = self.g_LT * V_diff * (NMDA_factor * self.excitatory_mask + self.inhibitory_mask)

    #     return (I_ST + I_LT) * self.weight_mult

    def __call__(self, neurons_V):
        """
        Calculate the synaptic current based on the neuron's membrane potential.

        Parameters
        ----------
        neurons_V : 1-D float array (n_neurons,)

        Returns
        -------
        out : 1-D float array (n_neurons,)
        """
        return _syn_current_numba(neurons_V,
                                self.g_AMPA, self.E_AMPA,
                                self.g_NMDA, self.E_NMDA,
                                self.g_GABA_A, self.E_GABA_A,
                                self.g_GABA_B, self.E_GABA_B,
                                self.LT_scale,
                                self.weight_mult)