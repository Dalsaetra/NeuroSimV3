import numpy as np
from connectome import Connectome

import numba as nb

@nb.njit(parallel=False, fastmath=True, cache=True)
def _syn_current_numba(neurons_V,
                       g_ST, E_ST,
                       g_LT, E_LT,
                       excit_mask, inhib_mask,
                       NMDA_scale: float,
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
        nmda    = (v2 / (1.0 + v2)) * NMDA_scale

        # Short-term current (AMPA / GABA-A, etc.)
        I_ST = g_ST[i] * (E_ST[i] - V)

        # Long-term current (NMDA & GABA_B)
        V_diff = E_LT[i] - V
        I_LT   = g_LT[i] * V_diff * (nmda * excit_mask[i] + inhib_mask[i])

        # Apply global weight multiplier (scalar)
        out[i] = (I_ST + I_LT) * weight_mult

    return out

class SynapseDynamics:
    def __init__(self, connectome: Connectome, dt, tau_ST=5, tau_LT=150, 
                 E_AMPA=0, E_NMDA=0, E_GABA_A=-70, E_GABA_B=-90, NMDA_scale = 1.0, weight_mult = 0.0005):
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
        synaptic_input = np.bincount(self.connectome.M.ravel(), weights=WS.ravel(), minlength=self.connectome.M.shape[0])

        self.g_ST += synaptic_input
        self.g_LT += synaptic_input

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
                                  self.g_ST, self.E_ST,
                                  self.g_LT, self.E_LT,
                                  self.excitatory_mask, self.inhibitory_mask,
                                  self.NMDA_scale,
                                  self.weight_mult)