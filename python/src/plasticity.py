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
        # weight_changes = np.zeros_like(self.connectome.W, dtype=np.float32)

        # Weight stabilizer
        weight_stab = self.connectome.W * (1 - self.connectome.W)
        weight_stab = np.clip(weight_stab, 0.001, 1)
        # weight_stab = 1

        # Calculate potentiation (pre spikes before post spikes)
        pre_effect = self.pre_traces * self.A_plus * reward * self.dt
        post_effect = self.post_traces * self.A_minus * reward * self.dt
        dw = weight_stab * (pre_effect - post_effect)
        if not gaba_plasticity:
            # No weight changes for inhibitory synapses
            dw[self.connectome.neuron_population.inhibitory_mask] = 0  
        else:
            # Inhibitory synapses has opposite weight changes
            dw[self.connectome.neuron_population.inhibitory_mask] *= -1


        # Update weights based on pre and post spike traces
        # weight_changes += dw

        # Apply weight changes to the connectome's weight matrix
        self.connectome.W += dw

    def step(self, pre_spikes, post_spikes, gaba_plasticity=True, reward=1):
        # Update the synapse weights based on the traces from last step
        self.apply_weight_changes(gaba_plasticity=True, reward=1)
        # Perform plasticity time step (like trace decay)
        self.decay_traces()
        # Update plasticity based on new spikes
        self.spikes_in(pre_spikes, post_spikes)

# A2_plus, A3_plus, A2_minus, A3_minus, tau_x, tau_y
t_stdp_params = {
    # Visual Cortex Data Set
    "AtA_C_full": (5e-10, 6.2e-3, 7e-3, 2.3e-4, 101, 125),
    "AtA_C_min": (0, 6.5e-3, 7.1e-3, 0, 0, 114),
    "NS_C_full": (8.8e-11, 5.3e-2, 6.6e-3, 3.1e-3, 714, 40),
    "NS_C_min": (0, 5e-2, 8e-3, 0, 0, 40),

    # Hippocampal Culture Data Set
    "AtA_H_full": (6.1e-3, 6.7e-3, 1.6e-3, 1.4e-3, 946, 27),
    "AtA_H_min": (5.3e-3, 8e-3, 3.5e-3, 0, 0, 40),
    "NS_H_full": (4.6e-3, 9.1e-3, 3e-3, 7.5e-9, 575, 47),
    "NS_H_min": (4.6e-3, 9.1e-3, 3e-3, 0, 0, 48),
}
    
class T_STDP:
    def __init__(self, connectome: Connectome, dt, tau_plus=16.8, tau_minus=33.7, 
                 A_plus=1.0, A_minus=2.0, mode="AtA_H_min"):
        """
        Triplet Spike-Timing-Dependent Plasticity (T-STDP) class to represent the T-STDP mechanism.
        
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
        self.A2_plus, self.A3_plus, self.A2_minus, self.A3_minus, self.tau_x, self.tau_y = t_stdp_params[mode]

        self.dt = dt

        self.decay_factor_plus = np.exp(-self.dt / self.tau_plus)
        self.decay_factor_minus = np.exp(-self.dt / self.tau_minus)

        self.decay_factor_x = np.exp(-self.dt / self.tau_x)
        self.decay_factor_y = np.exp(-self.dt / self.tau_y)

        self.pre_traces = np.zeros_like(connectome.M, dtype=np.float32)  # Pre-synaptic spike traces
        self.post_traces = np.zeros_like(connectome.M, dtype=np.float32)  # Post-synaptic spike traces

        self.pre_x_traces = np.zeros_like(connectome.M, dtype=np.float32)  # Pre-synaptic x traces
        self.post_y_traces = np.zeros_like(connectome.M, dtype=np.float32)  # Post-synaptic y traces

    def spikes_in_main(self, pre_spikes, post_spikes):
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

    def spikes_in_sub(self, pre_spikes, post_spikes):
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
            self.pre_x_traces[self.connectome.NC_invert] += pre_spikes.astype(np.float32)[self.connectome.NC_invert]
        elif np.all(pre_spikes == 0):
            # Update post-synaptic traces
            self.post_y_traces[self.connectome.NC_invert] += post_spikes[self.connectome.M[self.connectome.NC_invert]]
        else:
            # Update both pre-synaptic and post-synaptic traces
            self.pre_x_traces[self.connectome.NC_invert] += pre_spikes.astype(np.float32)[self.connectome.NC_invert]
            self.post_y_traces[self.connectome.NC_invert] += post_spikes[self.connectome.M[self.connectome.NC_invert]]

    def decay_traces_main(self):
        """
        Decay the spike traces over time.
        
        Parameters:
        dt: Time step (ms)
        """

        self.pre_traces *= self.decay_factor_plus
        self.post_traces *= self.decay_factor_minus

    def decay_traces_sub(self):
        """
        Decay the spike traces over time.
        
        Parameters:
        dt: Time step (ms)
        """

        self.pre_x_traces *= self.decay_factor_x
        self.post_y_traces *= self.decay_factor_y

    def apply_weight_changes(self, gaba_plasticity, reward=1):
        """
        Calculate the weight changes based on the spike traces.
        
        Returns:
        weight_changes: n_neurons x max_synapses array of weight changes
        """
        weight_changes = np.zeros_like(self.connectome.W, dtype=np.float32)

        # Weight stabilizer
        weight_stab = self.connectome.W * (1 - self.connectome.W)
        weight_stab = np.clip(weight_stab, 0.001, 1)

        # Calculate potentiation (pre spikes before post spikes)
        pre_effect = (self.A2_plus + self.A3_plus * self.post_y_traces) * self.pre_traces * reward * self.dt
        post_effect = (self.A2_minus + self.A3_minus * self.pre_x_traces) * self.post_traces * reward * self.dt
        dw = (pre_effect - post_effect) * weight_stab
        if not gaba_plasticity:
            # No weight changes for inhibitory synapses
            dw[self.connectome.neuron_population.inhibitory_mask] = 0  
        else:
            # Inhibitory synapses has opposite weight changes
            dw[self.connectome.neuron_population.inhibitory_mask] *= -1


        # Update weights based on pre and post spike traces
        weight_changes += dw

        # Apply weight changes to the connectome's weight matrix
        self.connectome.W += weight_changes

    def step(self, pre_spikes, post_spikes, gaba_plasticity=True, reward=1):
        # Update the synapse weights based on the traces from last step
        self.apply_weight_changes(gaba_plasticity=gaba_plasticity, reward=reward)
        # Perform plasticity time step (like trace decay)
        self.decay_traces_main()
        self.decay_traces_sub()
        # Update plasticity based on new spikes
        self.spikes_in_main(pre_spikes, post_spikes)
        self.spikes_in_sub(pre_spikes, post_spikes)


class PredictiveCoding:
    def __init__(self, connectome: Connectome, dt, A=0.001, tau_activity=100.0):
        """
        Predictive Coding class to represent the predictive coding mechanism.

        """
        self.connectome = connectome

        self.tau_activity = tau_activity 
        self.A = A

        self.dt = dt

        self.activity_trace = np.zeros(self.connectome.neuron_population.n_neurons, dtype=np.float32)  # Pre-synaptic spike traces
        self.decay_pre = np.exp(-dt / tau_activity)

    def step(self, pre_spikes, post_spikes, gaba_plasticity=True, reward=1):
        """
        Step the simulation forward in time.
        """
        self.activity_trace *= self.decay_pre
        self.activity_trace[post_spikes] += 1.0

        # Weight times pre-synaptic activity
        wx = np.multiply(self.activity_trace[:, np.newaxis], self.connectome.W)  # shape (n_neurons x max_synapses)

        if gaba_plasticity:
            wx[self.connectome.neuron_population.inhibitory_mask] *= -1 
        else:
            wx[self.connectome.neuron_population.inhibitory_mask] = 0

        # Expected activity
        mu = np.bincount(self.connectome.M.ravel(), weights=wx.ravel(), minlength=self.connectome.M.shape[0])

        # Calculate error
        error = self.activity_trace - mu

        # delta w_ij = A * error_j * activity_i
        dw = self.A * error[self.connectome.M] * self.activity_trace[:, np.newaxis]

        # Set weight changes to zero where no connection is marked
        dw[self.connectome.NC] = 0
 
        self.connectome.W += dw