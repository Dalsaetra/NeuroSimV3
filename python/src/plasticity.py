import numpy as np

from src.connectome import Connectome

class SmoothActivity:
    def __init__(self, n_neurons, tau, dt, A=0.0001):
        self.n = n_neurons
        self.dt = dt
        self.tau = tau
        self.A = A
        # two state vectors per neuron
        self.x = np.ones(n_neurons, dtype=float)
        self.y = np.zeros(n_neurons, dtype=float)
        # precompute constants
        self.decay1 = 1 - (2*dt/tau)
        self.decay2 = 1 - (dt/(tau*tau))

    def step(self, spikes):
        # spikes: bool or float array length n_neurons
        # update y with incoming spikes
        self.y += self.A * spikes
        # integrate x and y (Euler)
        x_new = self.x + self.dt * self.y
        y_new = self.y + self.dt * (
            - (2/self.tau)*self.y
            - (1/self.tau**2)*self.x
        )
        self.x, self.y = x_new, y_new
        # self.x is your smooth activity
        return self.x



class STDP:
    def __init__(self, connectome: Connectome, dt, tau_plus=20.0, tau_minus=20.0, A_plus=0.1, A_minus=0.12, gaba_factor=0.0, plastic_source_mask=None):
        """
        Spike-Timing-Dependent Plasticity (STDP) class to represent the STDP mechanism.
        
        Parameters:
        tau_plus: Time constant for potentiation (ms)
        tau_minus: Time constant for depression (ms)
        A_plus: Maximum change in weight for potentiation
        A_minus: Maximum change in weight for depression
        plastic_source_mask: Optional bool mask over source neurons (length n_neurons);
            only True rows have plastic outgoing synapses.
        """
        self.connectome = connectome

        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.gaba_factor = gaba_factor
        if plastic_source_mask is None:
            self.plastic_source_mask = np.ones(connectome.M.shape[0], dtype=bool)
        else:
            self.plastic_source_mask = np.asarray(plastic_source_mask, dtype=bool)
            if self.plastic_source_mask.ndim != 1 or self.plastic_source_mask.shape[0] != connectome.M.shape[0]:
                raise ValueError(
                    f"plastic_source_mask must be a 1D boolean array of length {connectome.M.shape[0]}."
                )

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

    def apply_weight_changes(self, reward=1):
        """
        Calculate the weight changes based on the spike traces.
        
        Returns:
        weight_changes: n_neurons x max_synapses array of weight changes
        """
        # weight_changes = np.zeros_like(self.connectome.W, dtype=np.float32)

        max_weight = 2.1 / 100

        # Weight stabilizer
        # weight_stab = self.connectome.W * (max_weight - self.connectome.W)
        # weight_stab = np.clip(weight_stab, 0.001, max_weight)
        weight_stab = 1.0

        # Calculate potentiation (pre spikes before post spikes)
        pre_effect = self.pre_traces * self.A_plus * reward * self.dt
        post_effect = self.post_traces * self.A_minus * reward * self.dt
        dw = weight_stab * (pre_effect - post_effect)
        dw[self.connectome.neuron_population.inhibitory_mask] *= self.gaba_factor
        dw[~self.plastic_source_mask] = 0.0


        # Update weights based on pre and post spike traces
        # weight_changes += dw

        # Apply weight changes to the connectome's weight matrix
        self.connectome.W += dw

        # Cap weights to [0, max_weight]
        np.clip(self.connectome.W, 0.0, max_weight, out=self.connectome.W)

    def step(self, pre_spikes, post_spikes, reward=1):
        # Update the synapse weights based on the traces from last step
        self.apply_weight_changes(reward=reward)
        # Perform plasticity time step (like trace decay)
        self.decay_traces()
        # Update plasticity based on new spikes
        self.spikes_in(pre_spikes, post_spikes)


class STDPMasked:
    def __init__(self, connectome: Connectome, dt, tau_plus=20.0, tau_minus=20.0, A_plus=0.1, A_minus=0.12, gaba_factor=0.0, plastic_source_mask=None):
        """
        STDP variant optimized for partial plasticity over source neurons.

        Only rows selected by `plastic_source_mask` are tracked and updated.
        This reduces both memory and per-step compute when the mask is sparse.
        """
        self.connectome = connectome

        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.gaba_factor = gaba_factor

        n_neurons = connectome.M.shape[0]
        if plastic_source_mask is None:
            plastic_source_mask = np.ones(n_neurons, dtype=bool)
        else:
            plastic_source_mask = np.asarray(plastic_source_mask, dtype=bool)
            if plastic_source_mask.ndim != 1 or plastic_source_mask.shape[0] != n_neurons:
                raise ValueError(
                    f"plastic_source_mask must be a 1D boolean array of length {n_neurons}."
                )
        self.plastic_source_mask = plastic_source_mask
        self.src_idx = np.flatnonzero(self.plastic_source_mask)

        # Pre-sliced connectivity for selected source rows.
        self.M_sel = self.connectome.M[self.src_idx]
        self.NC_invert_sel = self.connectome.NC_invert[self.src_idx]
        self.inhibitory_sel = self.connectome.neuron_population.inhibitory_mask[self.src_idx]

        self.dt = dt

        self.decay_factor_plus = np.exp(-self.dt / self.tau_plus)
        self.decay_factor_minus = np.exp(-self.dt / self.tau_minus)

        self.pre_traces = np.zeros_like(self.M_sel, dtype=np.float32)
        self.post_traces = np.zeros_like(self.M_sel, dtype=np.float32)

    def spikes_in(self, pre_spikes, post_spikes):
        if self.src_idx.size == 0:
            return
        pre_sel = pre_spikes[self.src_idx].astype(np.float32, copy=False)
        self.pre_traces[self.NC_invert_sel] += pre_sel[self.NC_invert_sel]
        self.post_traces[self.NC_invert_sel] += post_spikes[self.M_sel[self.NC_invert_sel]]

    def decay_traces(self):
        if self.src_idx.size == 0:
            return
        self.pre_traces *= self.decay_factor_plus
        self.post_traces *= self.decay_factor_minus

    def apply_weight_changes(self, reward=1):
        if self.src_idx.size == 0:
            return

        max_weight = 2.1 / 100
        weight_stab = 1.0

        pre_effect = self.pre_traces * self.A_plus * reward * self.dt
        post_effect = self.post_traces * self.A_minus * reward * self.dt
        dw = weight_stab * (pre_effect - post_effect)
        dw[self.inhibitory_sel] *= self.gaba_factor

        self.connectome.W[self.src_idx] += dw
        self.connectome.W[self.src_idx] = np.clip(self.connectome.W[self.src_idx], 0.0, max_weight)

    def step(self, pre_spikes, post_spikes, reward=1):
        self.apply_weight_changes(reward=reward)
        self.decay_traces()
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
                 A_plus=1.0, A_minus=2.0, mode="AtA_H_full", gaba_factor=-1.0):
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
        self.gaba_factor = gaba_factor

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

    def apply_weight_changes(self, reward=1):
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
        dw[self.connectome.neuron_population.inhibitory_mask] *= self.gaba_factor


        # Update weights based on pre and post spike traces
        weight_changes += dw

        # Apply weight changes to the connectome's weight matrix
        self.connectome.W += weight_changes

    def step(self, pre_spikes, post_spikes, reward=1):
        # Update the synapse weights based on the traces from last step
        self.apply_weight_changes(reward=reward)
        # Perform plasticity time step (like trace decay)
        self.decay_traces_main()
        self.decay_traces_sub()
        # Update plasticity based on new spikes
        self.spikes_in_main(pre_spikes, post_spikes)
        self.spikes_in_sub(pre_spikes, post_spikes)


class PredictiveCoding:
    def __init__(self, connectome: Connectome, dt, A=0.00001, tau_activity=1000.0, gaba_factor=1.0, mirror_neurons=[]):
        """
        Predictive Coding class to represent the predictive coding mechanism.
        mirror_neurons: Optional list of tuples (i,j) that note that the expected activity of neuron j is the activity of neuron i
        """
        self.connectome = connectome

        self.tau_activity = tau_activity 
        self.A = A
        self.gaba_factor = gaba_factor

        self.dt = dt

        self.smoothact = SmoothActivity(self.connectome.neuron_population.n_neurons, tau_activity, dt)

        self.mirror_neurons = mirror_neurons

        # self.activity_trace = np.zeros(self.connectome.neuron_population.n_neurons, dtype=np.float32)  # Pre-synaptic spike traces
        # self.decay_pre = np.exp(-dt / tau_activity)

    def step(self, pre_spikes, post_spikes, reward=1):
        """
        Step the simulation forward in time.
        """
        # self.activity_trace *= self.decay_pre
        # self.activity_trace[post_spikes] += 1.0

        # Update the activity trace using the smooth activity
        self.activity_trace = self.smoothact.step(post_spikes)

        # Weight times pre-synaptic activity
        wx = np.multiply(self.activity_trace[:, np.newaxis], self.connectome.W)  # shape (n_neurons x max_synapses)

        wx[self.connectome.neuron_population.inhibitory_mask] *= self.gaba_factor

        # Expected activity
        mu = np.bincount(self.connectome.M.ravel(), weights=wx.ravel(), minlength=self.connectome.M.shape[0])

        for i, j in self.mirror_neurons:
            # mu[j] = self.activity_trace[i]
            self.activity_trace[j] = self.activity_trace[i]  # Set the expected activity of neuron j to the activity of neuron i

        # Calculate error
        error = self.activity_trace - mu

        # delta w_ij = A * error_j * activity_i
        dw = reward * self.A * (error[self.connectome.M] * self.activity_trace[:, np.newaxis]) * self.connectome.W

        # Set weight changes to zero where no connection is marked
        dw[self.connectome.NC] = 0

 
        self.connectome.W += dw


class PredictiveCodingSaponati:
    def __init__(self, connectome: Connectome, dt, A=0.001, tau_activity=10.0, gaba_factor=1.0):
        """
        Predictive Coding class to represent the predictive coding mechanism.

        """
        self.connectome = connectome

        self.tau_activity = tau_activity 
        self.A = A
        self.gaba_factor = gaba_factor

        # Get the resting potentials of the neurons
        self.Vrs = connectome.neuron_population.neuron_population[:, 5]  

        self.dt = dt

        # self.smoothact = SmoothActivity(self.connectome.neuron_population.n_neurons, tau_activity, dt)

        self.p_t = np.zeros_like(self.connectome.M, dtype=np.float32)  # Pre-synaptic spike traces
        self.decay_pre = np.exp(-dt / tau_activity)

    def step(self, pre_spikes, post_spikes, Vs, reward=1):
        """
        Step the simulation forward in time.
        """

        self.p_t *= self.decay_pre
        self.p_t += pre_spikes

        # Get relative potentials
        relative_potentials = Vs - self.Vrs
        # print(relative_potentials)

        error = pre_spikes - relative_potentials[:, np.newaxis] * self.connectome.W

        # eps_t
        eps_t = (error * self.connectome.W).sum(axis=1)  # shape (n_neurons,)
        # print(eps_t.shape, self.connectome.W.shape, self.p_t.shape, eps_t[: np.newaxis].shape, relative_potentials[:, np.newaxis].shape)
        # print((eps_t[: np.newaxis] * self.p_t).shape)

        # delta w_ij = A * error_j * activity_i
        dw = reward * self.A * (error * relative_potentials[:, np.newaxis] + eps_t[:, np.newaxis] * self.p_t) * self.connectome.W

        dw[self.connectome.neuron_population.inhibitory_mask] *= self.gaba_factor

        # Set weight changes to zero where no connection is marked
        dw[self.connectome.NC] = 0
 
        self.connectome.W += dw
