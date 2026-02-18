import numpy as np
from src.connectome import Connectome

import numba as nb

@nb.njit(parallel=False, fastmath=True, cache=True)
def _syn_current_numba(neurons_V,
                       g_AMPA, E_AMPA,
                       g_NMDA, E_NMDA,
                       g_GABA_A, E_GABA_A,
                       g_GABA_B, E_GABA_B,
                       LT_scale: float,
                       weight_mult: float,
                       NMDA_weight: float):          # <── scalar
    """
    Compute total synaptic current for every neuron.

    All arrays are 1-D with length n_neurons; NMDA_scale and weight_mult are scalars.
    Returns a 1-D float array of length n_neurons.
    """
    n = neurons_V.size
    out = np.empty(n, dtype=neurons_V.dtype)

    for i in nb.prange(n):          # multi-threaded loop over neurons
        V = neurons_V[i]

        # NMDA voltage dependence (Izhikevich)
        # V_shift = (V + 80) / 60.0
        # v2      = V_shift * V_shift
        # nmda    = (v2 / (1.0 + v2))

        # NMDA voltage dependence (Jahr & Stevens 1990)
        Mg = 1.0  # mM, extracellular magnesium concentration
        nmda = 1.0 / (1.0 + 0.28 * Mg * np.exp(-0.062 * V))

        # Short-term current (AMPA / GABA-A, etc.)
        I_AMPA = g_AMPA[i] * (E_AMPA - V)
        I_NMDA = g_NMDA[i] * (E_NMDA - V) * nmda * LT_scale * NMDA_weight[i]
        I_GABA_A = g_GABA_A[i] * (E_GABA_A - V)
        I_GABA_B = g_GABA_B[i] * (E_GABA_B - V) * LT_scale

        # Apply global weight multiplier (scalar)
        out[i] = (I_AMPA + I_NMDA + I_GABA_A + I_GABA_B) * weight_mult

    return out

class SynapseDynamics:
    def __init__(self, connectome: Connectome, dt, tau_AMPA=5, tau_NMDA=150, tau_GABA_A=6, tau_GABA_B=150,
                 E_AMPA=0, E_NMDA=0, E_GABA_A=-70, E_GABA_B=-90, LT_scale = 1.0, weight_mult = 1.0, NMDA_weight=None):
        """
        SynapseDynamics class to represent the synaptic dynamics of a neuron population.
        """

        self.connectome = connectome

        self.inhibitory_mask = connectome.neuron_population.inhibitory_mask
        self.excitatory_mask = ~self.inhibitory_mask
        self.n_neurons = self.connectome.neuron_population.n_neurons
        self._inh_src_idx = np.flatnonzero(self.inhibitory_mask)
        self._exc_src_idx = np.flatnonzero(self.excitatory_mask)
        self._M_inh = self.connectome.M[self.inhibitory_mask]
        self._M_exc = self.connectome.M[self.excitatory_mask]

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

        self.g_AMPA_max = 943.8767019547389 * 0.5
        self.g_NMDA_max = 1132.4613811288566 * 0.5
        self.g_GABA_A_max = 477.1128477169971 * 2
        self.g_GABA_B_max = 978.2386456218012 * 0.5

        self.A_AMPA = 0.1965658831686625 * 0.1
        self.A_NMDA = 0.011641539779921259 * 0.05
        self.A_GABA_A = 0.00010575509751513417 * 10
        self.A_GABA_B = 0.00010290509538049661

        self.AMPA_decay = np.exp(-dt / tau_AMPA)
        self.NMDA_decay = np.exp(-dt / tau_NMDA)
        self.GABA_A_decay = np.exp(-dt / tau_GABA_A)
        self.GABA_B_decay = np.exp(-dt / tau_GABA_B)

        self.weight_mult = weight_mult
        self.LT_scale = LT_scale
        
        if NMDA_weight is None:
            self.NMDA_weight = np.ones(self.connectome.neuron_population.n_neurons, dtype=float)
        else:
            self.NMDA_weight = NMDA_weight


    def decay(self):
        # Decay the synaptic conductances
        self.g_AMPA *= self.AMPA_decay
        self.g_NMDA *= self.NMDA_decay
        self.g_GABA_A *= self.GABA_A_decay
        self.g_GABA_B *= self.GABA_B_decay


    def spike_input(self, spikes):
        # spikes: n_neurons x max_synapses

        inh_rows, inh_cols = np.nonzero(spikes[self.inhibitory_mask])
        if inh_rows.size:
            src_rows = self._inh_src_idx[inh_rows]
            weights = self.connectome.W[src_rows, inh_cols]
            targets = self._M_inh[inh_rows, inh_cols]
            inhib_input = np.bincount(targets, weights=weights, minlength=self.n_neurons)
            self.g_GABA_A += inhib_input * (1 - self.g_GABA_A) * self.A_GABA_A
            self.g_GABA_B += inhib_input * (1 - self.g_GABA_B) * self.A_GABA_B
            np.clip(self.g_GABA_A, 0, 1, out=self.g_GABA_A)
            np.clip(self.g_GABA_B, 0, 1, out=self.g_GABA_B)
        

        exc_rows, exc_cols = np.nonzero(spikes[self.excitatory_mask])
        if exc_rows.size:
            src_rows = self._exc_src_idx[exc_rows]
            weights = self.connectome.W[src_rows, exc_cols]
            targets = self._M_exc[exc_rows, exc_cols]
            excit_input = np.bincount(targets, weights=weights, minlength=self.n_neurons)
            self.g_AMPA += excit_input * (1 - self.g_AMPA) * self.A_AMPA
            self.g_NMDA += excit_input * (1 - self.g_NMDA) * self.A_NMDA
            np.clip(self.g_AMPA, 0, 1, out=self.g_AMPA)
            np.clip(self.g_NMDA, 0, 1, out=self.g_NMDA)


    def sensory_spike_input(self, weighted_spikes):
        # weighted_spikes: n_neurons x 1

        self.g_AMPA += weighted_spikes * (1 - self.g_AMPA) * self.A_AMPA
        np.clip(self.g_AMPA, 0, 1, out=self.g_AMPA)
        self.g_NMDA += weighted_spikes * (1 - self.g_NMDA) * self.A_NMDA
        np.clip(self.g_NMDA, 0, 1, out=self.g_NMDA)


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
                                self.g_AMPA * self.g_AMPA_max, self.E_AMPA,
                                self.g_NMDA * self.g_NMDA_max, self.E_NMDA,
                                self.g_GABA_A * self.g_GABA_A_max, self.E_GABA_A,
                                self.g_GABA_B * self.g_GABA_B_max, self.E_GABA_B,
                                self.LT_scale,
                                self.weight_mult,
                                self.NMDA_weight)
    
class SynapseDynamics_Rise(SynapseDynamics):
    def __init__(self, connectome: Connectome, dt, tau_AMPA=5, tau_NMDA=150, tau_GABA_A=6, tau_GABA_B=150,
                 E_AMPA=0, E_NMDA=0, E_GABA_A=-70, E_GABA_B=-90, LT_scale = 1.0, weight_mult = 1.0, NMDA_weight=None):
        """
        SynapseDynamics class to represent the synaptic dynamics of a neuron population.
        """
        super().__init__(connectome, dt, tau_AMPA, tau_NMDA, tau_GABA_A, tau_GABA_B,
                 E_AMPA, E_NMDA, E_GABA_A, E_GABA_B, LT_scale, weight_mult, NMDA_weight)

        self.x_rise_NMDA = np.zeros(self.connectome.neuron_population.n_neurons, dtype=float)
        self.x_rise_GABA_B = np.zeros(self.connectome.neuron_population.n_neurons, dtype=float)

        self.x_rise_time_NMDA = 30
        self.x_rise_factor = np.exp(-dt / self.x_rise_time_NMDA)
        self.x_rise_time_GABA_B = 150
        self.x_rise_factor_GABA_B = np.exp(-dt / self.x_rise_time_GABA_B)


    def decay(self):
        # Decay the synaptic conductances
        self.g_AMPA *= self.AMPA_decay
        self.g_NMDA *= self.NMDA_decay
        self.g_GABA_A *= self.GABA_A_decay
        self.g_GABA_B *= self.GABA_B_decay

        self.x_rise_NMDA *= self.x_rise_factor
        self.x_rise_GABA_B *= self.x_rise_factor_GABA_B


    def spike_input(self, spikes):
        # spikes: n_neurons x max_synapses

        inh_rows, inh_cols = np.nonzero(spikes[self.inhibitory_mask])
        if inh_rows.size:
            src_rows = self._inh_src_idx[inh_rows]
            weights = self.connectome.W[src_rows, inh_cols]
            targets = self._M_inh[inh_rows, inh_cols]
            inhib_input = np.bincount(targets, weights=weights, minlength=self.n_neurons)
            self.g_GABA_A += inhib_input * (1 - self.g_GABA_A) * self.A_GABA_A
            # self.g_GABA_B += inhib_input
            self.g_GABA_B += self.x_rise_GABA_B * self.dt * (1 - self.g_GABA_B) * self.A_GABA_B
            np.clip(self.g_GABA_A, 0, 1, out=self.g_GABA_A)
            np.clip(self.g_GABA_B, 0, 1, out=self.g_GABA_B)
            # Divided by normalization factor to adjust such that it rises to 1 for +1 input
            self.x_rise_GABA_B += inhib_input / 55.0
            
        exc_rows, exc_cols = np.nonzero(spikes[self.excitatory_mask])
        if exc_rows.size:
            src_rows = self._exc_src_idx[exc_rows]
            weights = self.connectome.W[src_rows, exc_cols]
            targets = self._M_exc[exc_rows, exc_cols]
            excit_input = np.bincount(targets, weights=weights, minlength=self.n_neurons)
            self.g_AMPA += excit_input * (1 - self.g_AMPA) * self.A_AMPA
            # self.g_NMDA += excit_input
            self.g_NMDA += self.x_rise_NMDA * self.dt * (1 - self.g_NMDA) * self.A_NMDA
            # Divided by normalization factor to adjust such that it rises to 1 for +1 input
            self.x_rise_NMDA += excit_input / 20.0
            np.clip(self.g_AMPA, 0, 1, out=self.g_AMPA)
            np.clip(self.g_NMDA, 0, 1, out=self.g_NMDA)


class SynapseDynamics_Uncapped:
    def __init__(self, connectome: Connectome, dt, tau_AMPA=5, tau_NMDA=150, tau_GABA_A=6, tau_GABA_B=150,
                 E_AMPA=0, E_NMDA=0, E_GABA_A=-70, E_GABA_B=-90, LT_scale = 1.0, weight_mult = 1.0, NMDA_weight=None):
        """
        SynapseDynamics class to represent the synaptic dynamics of a neuron population.
        """

        self.connectome = connectome

        self.inhibitory_mask = connectome.neuron_population.inhibitory_mask
        self.excitatory_mask = ~self.inhibitory_mask
        self.n_neurons = self.connectome.neuron_population.n_neurons
        self._inh_src_idx = np.flatnonzero(self.inhibitory_mask)
        self._exc_src_idx = np.flatnonzero(self.excitatory_mask)
        self._M_inh = self.connectome.M[self.inhibitory_mask]
        self._M_exc = self.connectome.M[self.excitatory_mask]

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
        self.NMDA_weight = NMDA_weight

        # Just for logging purposes, to know the scale of the conductances
        self.g_AMPA_max = 1.0
        self.g_NMDA_max = 1.0
        self.g_GABA_A_max = 1.0
        self.g_GABA_B_max = 1.0


    def decay(self):
        # Decay the synaptic conductances
        self.g_AMPA *= self.AMPA_decay
        self.g_NMDA *= self.NMDA_decay
        self.g_GABA_A *= self.GABA_A_decay
        self.g_GABA_B *= self.GABA_B_decay


    def spike_input(self, spikes):
        # spikes: n_neurons x max_synapses

        inh_rows, inh_cols = np.nonzero(spikes[self.inhibitory_mask])
        if inh_rows.size:
            src_rows = self._inh_src_idx[inh_rows]
            weights = self.connectome.W[src_rows, inh_cols]
            targets = self._M_inh[inh_rows, inh_cols]
            inhib_input = np.bincount(targets, weights=weights, minlength=self.n_neurons)
            self.g_GABA_A += inhib_input
            self.g_GABA_B += inhib_input 


        exc_rows, exc_cols = np.nonzero(spikes[self.excitatory_mask])
        if exc_rows.size:
            src_rows = self._exc_src_idx[exc_rows]
            weights = self.connectome.W[src_rows, exc_cols]
            targets = self._M_exc[exc_rows, exc_cols]
            excit_input = np.bincount(targets, weights=weights, minlength=self.n_neurons)
            self.g_AMPA += excit_input
            self.g_NMDA += excit_input

    def sensory_spike_input(self, weighted_spikes):
        # weighted_spikes: n_neurons x 1

        self.g_AMPA += weighted_spikes
        self.g_NMDA += weighted_spikes

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
                                self.weight_mult,
                                self.NMDA_weight)
