import numpy as np

from connectome import Connectome

class STDP:
    def __init__(self, connectome: Connectome, dt, tau_plus=16.8, tau_minus=33.7, A_plus=1.0, A_minus=2.0):
        """
        Spike-Timing-Dependent Plasticity (STDP) class to represent the STDP mechanism.
        
        Parameters:
        tau_plus: Time constant for potentiation (ms)
        tau_minus: Time constant for depression (ms)
        A_plus: Maximum change in weight for potentiation
        A_minus: Maximum change in weight for depression
        """
        self.connectome = connectome

        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.A_plus = A_plus
        self.A_minus = A_minus

        self.dt = dt

        self.decay_factor_plus = np.exp(-self.dt / self.tau_plus)
        self.decay_factor_minus = np.exp(-self.dt / self.tau_minus)

        self.pre_traces = np.zeros_like(connectome.M, dtype=np.float32)  # Pre-synaptic spike traces
        self.post_traces = np.zeros_like(connectome.M, dtype=np.float32)  # Post-synaptic spike traces

    def spikes_in(self, pre_spikes, post_spikes):
        """
        Update the spike traces based on pre-synaptic and post-synaptic spikes.
        
        Parameters:
        pre_spikes: n_neurons x max_synapses array of pre-synaptic spikes
        post_spikes: n_neurons x 1 array of post-synaptic spikes
        """
        if np.all(pre_spikes == 0) and np.all(post_spikes == 0):
            return
        elif np.all(post_spikes == 0):
            # Update pre-synaptic traces
            self.pre_traces[self.connectome.NC_invert] += pre_spikes.astype(np.float32)[self.connectome.NC_invert]
        elif np.all(pre_spikes == 0):
            # Update post-synaptic traces
            self.post_traces[self.connectome.NC_invert] += post_spikes[self.connectome.M[self.connectome.NC_invert]]
        else:
            # Update both pre-synaptic and post-synaptic traces
            self.pre_traces[self.connectome.NC_invert] += pre_spikes.astype(np.float32)[self.connectome.NC_invert]
            self.post_traces[self.connectome.NC_invert] += post_spikes[self.connectome.M[self.connectome.NC_invert]]

    def decay_traces(self):
        """
        Decay the spike traces over time.
        
        Parameters:
        dt: Time step (ms)
        """

        self.pre_traces *= self.decay_factor_plus
        self.post_traces *= self.decay_factor_minus

    def apply_weight_changes(self, gaba_plasticity, reward=1):
        """
        Calculate the weight changes based on the spike traces.
        
        Returns:
        weight_changes: n_neurons x max_synapses array of weight changes
        """
        weight_changes = np.zeros_like(self.connectome.W, dtype=np.float32)

        # Calculate potentiation (pre spikes before post spikes)
        pre_effect = self.pre_traces * self.A_plus * reward * self.dt
        post_effect = self.post_traces * self.A_minus * reward * self.dt
        dw = pre_effect - post_effect
        if not gaba_plasticity:
            # No weight changes for inhibitory neurons
            dw[self.connectome.neuron_population.inhibitory_mask] = 0  
        else:
            # Inhibitory synapses has opposite weight changes
            dw[self.connectome.neuron_population.inhibitory_mask] *= -1


        # Update weights based on pre and post spike traces
        weight_changes += dw

        # Apply weight changes to the connectome's weight matrix
        self.connectome.W += weight_changes