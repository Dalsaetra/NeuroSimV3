import numpy as np

from src.connectome import Connectome


def _get_weight_multiplier(W, weight_multiplicity, max_weight):
    if weight_multiplicity is None:
        return 1.0
    if weight_multiplicity == "weight_stab":
        if max_weight is None:
            raise ValueError("max_weight must be provided when weight_multiplicity='weight_stab'.")
        weight_mult = W * (max_weight - W)
        return np.clip(weight_mult, 0.001, max_weight)
    if weight_multiplicity == "multiplicative_factor":
        return W
    raise ValueError(
        "weight_multiplicity must be either 'weight_stab' or 'multiplicative_factor'."
    )


def _clip_weights_inplace(W, max_weight):
    if max_weight is not None:
        np.clip(W, 0.0, max_weight, out=W)


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
    def __init__(self, connectome: Connectome, dt, tau_plus=20.0, tau_minus=20.0, A_plus=0.1, A_minus=0.12, gaba_factor=0.0, plastic_source_mask=None,
                 max_weight=300, weight_update_scale=1.0, weight_multiplicity="weight_stab"):
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
        self.max_weight = max_weight
        self.weight_update_scale = weight_update_scale
        self.weight_multiplicity = weight_multiplicity
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

        # Calculate potentiation (pre spikes before post spikes)
        pre_effect = self.pre_traces * self.A_plus * reward * self.dt
        post_effect = self.post_traces * self.A_minus * reward * self.dt
        weight_mult = _get_weight_multiplier(self.connectome.W, self.weight_multiplicity, self.max_weight)
        dw = (pre_effect - post_effect) * self.weight_update_scale * weight_mult
        dw[self.connectome.neuron_population.inhibitory_mask] *= self.gaba_factor
        dw[~self.plastic_source_mask] = 0.0


        # Update weights based on pre and post spike traces
        # weight_changes += dw

        # Apply weight changes to the connectome's weight matrix
        self.connectome.W += dw

        # Cap weights to [0, max_weight]
        _clip_weights_inplace(self.connectome.W, self.max_weight)

    def step(self, pre_spikes, post_spikes, reward=1):
        # Update the synapse weights based on the traces from last step
        self.apply_weight_changes(reward=reward)
        # Perform plasticity time step (like trace decay)
        self.decay_traces()
        # Update plasticity based on new spikes
        self.spikes_in(pre_spikes, post_spikes)

    def step_no_weight_changes(self, pre_spikes, post_spikes, reward=1):
        self.decay_traces()
        self.spikes_in(pre_spikes, post_spikes)

    def reset_traces(self):
        self.pre_traces.fill(0)
        self.post_traces.fill(0)


class STDPMasked:
    def __init__(self, connectome: Connectome, dt, tau_plus=250.0, tau_minus=500.0, A_plus=0.01, A_minus=0.012, gaba_factor=0.0, plastic_source_mask=None,
                 max_weight=300, weight_update_scale=1.0, weight_multiplicity="weight_stab"):
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
        self.max_weight = max_weight
        self.weight_update_scale = weight_update_scale
        self.weight_multiplicity = weight_multiplicity

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

        pre_effect = self.pre_traces * self.A_plus * reward * self.dt
        post_effect = self.post_traces * self.A_minus * reward * self.dt
        W_sel = self.connectome.W[self.src_idx]
        weight_mult = _get_weight_multiplier(W_sel, self.weight_multiplicity, self.max_weight)
        dw = (pre_effect - post_effect) * self.weight_update_scale * weight_mult
        dw[self.inhibitory_sel] *= self.gaba_factor

        self.connectome.W[self.src_idx] += dw
        if self.max_weight is not None:
            self.connectome.W[self.src_idx] = np.clip(self.connectome.W[self.src_idx], 0.0, self.max_weight)

    def step(self, pre_spikes, post_spikes, reward=1):
        self.apply_weight_changes(reward=reward)
        self.decay_traces()
        self.spikes_in(pre_spikes, post_spikes)

    def step_no_weight_changes(self, pre_spikes, post_spikes, reward=1):
        self.decay_traces()
        self.spikes_in(pre_spikes, post_spikes)

    def reset_traces(self):
        self.pre_traces.fill(0)
        self.post_traces.fill(0)


class DA_BCM:
    def __init__(
        self,
        connectome: Connectome,
        dt,
        tau_pre=20.0,
        tau_post=40.0,
        tau_theta=20.0,
        A=0.02,
        weight_decay=0.01,
        epsilon=None,
        gaba_factor=0.0,
        plastic_source_mask=None,
        max_weight=300,
        weight_update_scale=1.0,
        weight_multiplicity="weight_stab",
        theta_init=0.0,
    ):
        """
        Dopamine-modulated BCM plasticity with sparse source masking.

        Implementation details:
        - Uses STDP-style exponential decay for eligibility traces.
        - Uses an exponential moving average for BCM threshold theta.
        - Uses reward as the global dopamine modulation signal.
        """
        self.connectome = connectome

        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.tau_theta = tau_theta
        self.A = A
        if epsilon is not None:
            weight_decay = epsilon
        self.weight_decay = weight_decay
        self.gaba_factor = gaba_factor
        self.max_weight = max_weight
        self.weight_update_scale = weight_update_scale
        self.weight_multiplicity = weight_multiplicity
        self.theta_init = theta_init

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
        self.decay_factor_pre = np.exp(-self.dt / self.tau_pre)
        self.decay_factor_post = np.exp(-self.dt / self.tau_post)
        self.theta_alpha = 1.0 - np.exp(-self.dt / self.tau_theta)

        self.pre_traces = np.zeros_like(self.M_sel, dtype=np.float32)
        self.post_traces = np.zeros_like(self.M_sel, dtype=np.float32)

        # Per-neuron post activity trace is used to define a neuron-level BCM threshold.
        self.post_activity_trace = np.zeros(n_neurons, dtype=np.float32)
        self.theta = np.full(n_neurons, theta_init, dtype=np.float32)

    def spikes_in(self, pre_spikes, post_spikes):
        post_flat = np.asarray(post_spikes, dtype=np.float32).reshape(-1)
        if self.src_idx.size > 0:
            pre_sel = pre_spikes[self.src_idx].astype(np.float32, copy=False)
            self.pre_traces[self.NC_invert_sel] += pre_sel[self.NC_invert_sel]
            self.post_traces[self.NC_invert_sel] += post_flat[self.M_sel[self.NC_invert_sel]]
        self.post_activity_trace += post_flat

    def decay_traces(self):
        if self.src_idx.size > 0:
            self.pre_traces *= self.decay_factor_pre
            self.post_traces *= self.decay_factor_post
        self.post_activity_trace *= self.decay_factor_post

    def update_theta(self):
        self.theta += self.theta_alpha * (self.post_activity_trace - self.theta)

    def apply_weight_changes(self, reward=1):
        if self.src_idx.size == 0:
            return

        W_sel = self.connectome.W[self.src_idx]
        theta_syn = self.theta[self.M_sel]
        post_drive = self.post_traces * (self.post_traces - theta_syn)
        eligibility = self.pre_traces * post_drive
        if self.weight_decay != 0.0:
            eligibility -= self.weight_decay * W_sel

        weight_mult = _get_weight_multiplier(W_sel, self.weight_multiplicity, self.max_weight)
        dw = reward * self.A * eligibility * self.dt
        dw = dw * self.weight_update_scale * weight_mult
        dw[self.inhibitory_sel] *= self.gaba_factor

        self.connectome.W[self.src_idx] += dw
        if self.max_weight is not None:
            self.connectome.W[self.src_idx] = np.clip(self.connectome.W[self.src_idx], 0.0, self.max_weight)

    def step(self, pre_spikes, post_spikes, reward=1):
        self.apply_weight_changes(reward=reward)
        self.decay_traces()
        self.spikes_in(pre_spikes, post_spikes)
        self.update_theta()

    def step_no_weight_changes(self, pre_spikes, post_spikes, reward=1):
        self.decay_traces()
        self.spikes_in(pre_spikes, post_spikes)
        self.update_theta()

    def reset_traces(self):
        self.pre_traces.fill(0)
        self.post_traces.fill(0)
        self.post_activity_trace.fill(0)
        self.theta.fill(self.theta_init)


class ClopathMasked:
    def __init__(
        self,
        connectome: Connectome,
        dt,
        tau_x=20.0,
        tau_v=10.0,
        A_plus=0.001*2,
        A_minus=0.0012*2,
        theta_plus=-45.0,
        theta_minus=-55.0,
        gaba_factor=0.0,
        plastic_source_mask=None,
        min_weight=0.0,
        max_weight=300,
        weight_update_scale=1.0,
        weight_multiplicity=None,
        enable_debug_logger=False,
    ):
        """
        Clopath-style voltage-based plasticity with sparse source masking.

        Update equations (Euler form):
            x += dt * (-x / tau_x + s_pre)
            v_bar += dt * (-v_bar / tau_v + V)
            dw = dt * (A_plus * x * relu(V - theta_plus) * relu(v_bar - theta_minus)
                       - A_minus * x * relu(v_bar - theta_minus))

        Only rows selected by `plastic_source_mask` are tracked/updated.
        """
        self.connectome = connectome
        self.dt = dt
        self.tau_x = tau_x
        self.tau_v = tau_v
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.theta_plus = theta_plus
        self.theta_minus = theta_minus
        self.gaba_factor = gaba_factor
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.weight_update_scale = weight_update_scale
        self.weight_multiplicity = weight_multiplicity
        self.enable_debug_logger = bool(enable_debug_logger)

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
        self.src_idx = np.flatnonzero(self.plastic_source_mask).astype(np.intp, copy=False)

        # Store only valid plastic synapses to avoid dense masked work every step.
        self.M_sel = self.connectome.M[self.src_idx]
        self.NC_invert_sel = self.connectome.NC_invert[self.src_idx]
        self.valid_src_local, self.valid_cols = np.nonzero(self.NC_invert_sel)
        self.valid_src_local = self.valid_src_local.astype(np.intp, copy=False)
        self.valid_cols = self.valid_cols.astype(np.intp, copy=False)
        self.valid_src = self.src_idx[self.valid_src_local]
        self.valid_post = self.M_sel[self.valid_src_local, self.valid_cols].astype(np.intp, copy=False)
        self.inhibitory_valid = self.connectome.neuron_population.inhibitory_mask[self.valid_src]
        self.weight_flat_idx = self.valid_src * self.connectome.max_synapses + self.valid_cols
        self.n_valid = self.valid_post.size
        self.supports_sparse_pre_spikes = True
        self._synapse_lookup = np.full(
            connectome.neuron_population.n_neurons * self.connectome.max_synapses,
            -1,
            dtype=np.int32,
        )
        self._synapse_lookup[self.weight_flat_idx] = np.arange(self.n_valid, dtype=np.int32)

        # If theta_plus and theta_minus are per-neuron, project them to valid synapses once.
        self.theta_plus_syn = self._prepare_synapse_param(
            self.theta_plus,
            n_neurons,
            self.M_sel.shape,
            self.valid_src_local,
            self.valid_cols,
            self.valid_post,
        )
        self.theta_minus_syn = self._prepare_synapse_param(
            self.theta_minus,
            n_neurons,
            self.M_sel.shape,
            self.valid_src_local,
            self.valid_cols,
            self.valid_post,
        )

        # Per-synapse presynaptic trace for selected valid synapses only.
        self.pre_traces = np.zeros(self.n_valid, dtype=np.float32)
        # Per-neuron low-pass membrane potential.
        self.v_bar = np.zeros(n_neurons, dtype=np.float32)
        self.pre_trace_decay = np.float32(1.0 - (self.dt / self.tau_x))
        self.pre_spike_scale = np.float32(self.dt)
        self.v_bar_alpha = np.float32(self.dt / self.tau_v)
        self.v_bar_decay = np.float32(1.0 - self.v_bar_alpha)
        self.dw_scale = np.float32(self.dt * self.weight_update_scale)
        self._W_flat = self.connectome.W.reshape(-1)
        self._post_voltage = np.empty(self.n_valid, dtype=np.float32)
        self._post_v_bar = np.empty(self.n_valid, dtype=np.float32)
        self._gate_plus = np.empty(self.n_valid, dtype=np.float32)
        self._gate_minus = np.empty(self.n_valid, dtype=np.float32)
        self._dw = np.empty(self.n_valid, dtype=np.float32)
        self._weight_buf = np.empty(self.n_valid, dtype=self.connectome.W.dtype)
        if self.enable_debug_logger:
            self._ltp_buf = np.empty(self.n_valid, dtype=np.float32)
            self._ltd_buf = np.empty(self.n_valid, dtype=np.float32)
        self.debug_logger = {
            "mean_ltp": [],
            "mean_ltd": [],
            "mean_dw": [],
            "active_ltp_gate_frac": [],
            "active_ltd_gate_frac": [],
            "active_synapse_frac": [],
            "mean_pre_trace": [],
            "mean_v_bar_post": [],
        }

    @staticmethod
    def _as_voltage_vector(Vs):
        return np.asarray(Vs, dtype=np.float32).reshape(-1)

    @staticmethod
    def _prepare_synapse_param(param, n_neurons, synapse_shape, valid_rows, valid_cols, valid_post):
        param_arr = np.asarray(param)
        if param_arr.ndim == 0:
            return np.float32(param_arr)
        if param_arr.ndim == 1:
            if param_arr.shape[0] != n_neurons:
                raise ValueError(
                    f"Per-neuron threshold arrays must have length {n_neurons}."
                )
            return np.asarray(param_arr[valid_post], dtype=np.float32)
        param_full = np.broadcast_to(param_arr, synapse_shape)
        return np.asarray(param_full[valid_rows, valid_cols], dtype=np.float32)

    def _update_pre_trace(self, pre_spikes):
        if self.n_valid == 0:
            return
        self.pre_traces *= self.pre_trace_decay
        self.pre_traces += self.pre_spike_scale * pre_spikes[self.valid_src, self.valid_cols]

    def _update_pre_trace_sparse(self, pre_spike_rows, pre_spike_cols):
        if self.n_valid == 0:
            return
        self.pre_traces *= self.pre_trace_decay
        if pre_spike_rows.size == 0:
            return
        flat_idx = (
            pre_spike_rows.astype(np.intp, copy=False) * self.connectome.max_synapses
            + pre_spike_cols.astype(np.intp, copy=False)
        )
        trace_idx = self._synapse_lookup[flat_idx]
        valid = trace_idx >= 0
        if np.any(valid):
            self.pre_traces[trace_idx[valid]] += self.pre_spike_scale

    def _update_voltage_trace(self, V):
        # Low-pass filter with steady state equal to V.
        self.v_bar += self.v_bar_alpha * (V - self.v_bar)

    def decay_traces(self):
        if self.n_valid > 0:
            self.pre_traces *= self.pre_trace_decay
        self.v_bar *= self.v_bar_decay

    def spikes_in(self, pre_spikes, post_spikes=None, Vs=None):
        self._update_pre_trace(pre_spikes)
        if Vs is not None:
            self._update_voltage_trace(self._as_voltage_vector(Vs))

    def apply_weight_changes(self, Vs, reward=1):
        if self.n_valid == 0:
            if self.enable_debug_logger:
                self._log_debug(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            return

        V = self._as_voltage_vector(Vs)
        np.take(V, self.valid_post, out=self._post_voltage)
        np.take(self.v_bar, self.valid_post, out=self._post_v_bar)

        np.subtract(self._post_voltage, self.theta_plus_syn, out=self._gate_plus)
        np.maximum(self._gate_plus, 0.0, out=self._gate_plus)
        np.subtract(self._post_v_bar, self.theta_minus_syn, out=self._gate_minus)
        np.maximum(self._gate_minus, 0.0, out=self._gate_minus)

        np.copyto(self._dw, self._gate_plus)
        self._dw *= self.A_plus
        self._dw -= self.A_minus
        self._dw *= self.pre_traces
        self._dw *= self._gate_minus

        np.take(self._W_flat, self.weight_flat_idx, out=self._weight_buf)
        if self.weight_multiplicity is not None:
            weight_mult = _get_weight_multiplier(self._weight_buf, self.weight_multiplicity, self.max_weight)
            self._dw *= weight_mult
        self._dw *= (reward * self.dw_scale)
        self._dw[self.inhibitory_valid] *= self.gaba_factor

        if self.enable_debug_logger:
            np.multiply(self.pre_traces, self._gate_minus, out=self._ltd_buf)
            np.copyto(self._ltp_buf, self._ltd_buf)
            self._ltp_buf *= self._gate_plus
            self._ltp_buf *= self.A_plus
            self._ltd_buf *= self.A_minus
            active_synapse_frac = float(np.mean(np.abs(self._dw) > 0.0))
            mean_ltp = float(np.mean(self._ltp_buf))
            mean_ltd = float(np.mean(self._ltd_buf))
            mean_dw = float(np.mean(self._dw))
            active_ltp_gate_frac = float(np.mean((self._gate_plus > 0.0) & (self._gate_minus > 0.0)))
            active_ltd_gate_frac = float(np.mean(self._gate_minus > 0.0))
            mean_pre_trace = float(np.mean(self.pre_traces))
            mean_v_bar_post = float(np.mean(self._post_v_bar))
            self._log_debug(
                mean_ltp,
                mean_ltd,
                mean_dw,
                active_ltp_gate_frac,
                active_ltd_gate_frac,
                active_synapse_frac,
                mean_pre_trace,
                mean_v_bar_post,
            )

        self._weight_buf += self._dw
        if self.max_weight is not None:
            np.clip(self._weight_buf, self.min_weight, self.max_weight, out=self._weight_buf)
        else:
            np.maximum(self._weight_buf, self.min_weight, out=self._weight_buf)
        self._W_flat[self.weight_flat_idx] = self._weight_buf

    def step(self, pre_spikes, post_spikes, Vs, reward=1):
        V = self._as_voltage_vector(Vs)
        self._update_pre_trace(pre_spikes)
        self._update_voltage_trace(V)
        self.apply_weight_changes(V, reward=reward)

    def step_sparse(self, pre_spike_rows, pre_spike_cols, post_spikes, Vs, reward=1):
        V = self._as_voltage_vector(Vs)
        self._update_pre_trace_sparse(pre_spike_rows, pre_spike_cols)
        self._update_voltage_trace(V)
        self.apply_weight_changes(V, reward=reward)

    def step_no_weight_changes(self, pre_spikes, post_spikes, Vs=None, reward=1):
        self._update_pre_trace(pre_spikes)
        if Vs is not None:
            self._update_voltage_trace(self._as_voltage_vector(Vs))
        else:
            # If V is not provided, decay toward 0 as a neutral fallback.
            self.v_bar *= self.v_bar_decay

    def step_no_weight_changes_sparse(self, pre_spike_rows, pre_spike_cols, post_spikes, Vs=None, reward=1):
        self._update_pre_trace_sparse(pre_spike_rows, pre_spike_cols)
        if Vs is not None:
            self._update_voltage_trace(self._as_voltage_vector(Vs))
        else:
            self.v_bar *= self.v_bar_decay

    def reset_traces(self):
        self.pre_traces.fill(0)
        self.v_bar.fill(0)
        self.reset_debug_logger()

    def _log_debug(
        self,
        mean_ltp,
        mean_ltd,
        mean_dw,
        active_ltp_gate_frac,
        active_ltd_gate_frac,
        active_synapse_frac,
        mean_pre_trace,
        mean_v_bar_post,
    ):
        self.debug_logger["mean_ltp"].append(float(mean_ltp))
        self.debug_logger["mean_ltd"].append(float(mean_ltd))
        self.debug_logger["mean_dw"].append(float(mean_dw))
        self.debug_logger["active_ltp_gate_frac"].append(float(active_ltp_gate_frac))
        self.debug_logger["active_ltd_gate_frac"].append(float(active_ltd_gate_frac))
        self.debug_logger["active_synapse_frac"].append(float(active_synapse_frac))
        self.debug_logger["mean_pre_trace"].append(float(mean_pre_trace))
        self.debug_logger["mean_v_bar_post"].append(float(mean_v_bar_post))

    def reset_debug_logger(self):
        for k in self.debug_logger:
            self.debug_logger[k] = []

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
                 A_plus=1.0, A_minus=2.0, mode="AtA_H_full", gaba_factor=-1.0,
                 max_weight=300, weight_update_scale=1.0, weight_multiplicity="weight_stab"):
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
        self.max_weight = max_weight
        self.weight_update_scale = weight_update_scale
        self.weight_multiplicity = weight_multiplicity

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
        # Calculate potentiation (pre spikes before post spikes)
        pre_effect = (self.A2_plus + self.A3_plus * self.post_y_traces) * self.pre_traces * reward * self.dt
        post_effect = (self.A2_minus + self.A3_minus * self.pre_x_traces) * self.post_traces * reward * self.dt
        weight_mult = _get_weight_multiplier(self.connectome.W, self.weight_multiplicity, self.max_weight)
        dw = (pre_effect - post_effect) * self.weight_update_scale * weight_mult
        dw[self.connectome.neuron_population.inhibitory_mask] *= self.gaba_factor

        # Apply weight changes to the connectome's weight matrix
        self.connectome.W += dw
        _clip_weights_inplace(self.connectome.W, self.max_weight)

    def step(self, pre_spikes, post_spikes, reward=1):
        # Update the synapse weights based on the traces from last step
        self.apply_weight_changes(reward=reward)
        # Perform plasticity time step (like trace decay)
        self.decay_traces_main()
        self.decay_traces_sub()
        # Update plasticity based on new spikes
        self.spikes_in_main(pre_spikes, post_spikes)
        self.spikes_in_sub(pre_spikes, post_spikes)

    def step_no_weight_changes(self, pre_spikes, post_spikes, reward=1):
        self.decay_traces_main()
        self.decay_traces_sub()
        self.spikes_in_main(pre_spikes, post_spikes)
        self.spikes_in_sub(pre_spikes, post_spikes)

    def reset_traces(self):
        self.pre_traces.fill(0)
        self.post_traces.fill(0)
        self.pre_x_traces.fill(0)
        self.post_y_traces.fill(0)


class PredictiveCoding:
    def __init__(self, connectome: Connectome, dt, A=0.00001, tau_activity=1000.0, gaba_factor=1.0, mirror_neurons=[],
                 max_weight=None, weight_update_scale=1.0, weight_multiplicity="multiplicative_factor"):
        """
        Predictive Coding class to represent the predictive coding mechanism.
        mirror_neurons: Optional list of tuples (i,j) that note that the expected activity of neuron j is the activity of neuron i
        """
        self.connectome = connectome

        self.tau_activity = tau_activity 
        self.A = A
        self.gaba_factor = gaba_factor
        self.max_weight = max_weight
        self.weight_update_scale = weight_update_scale
        self.weight_multiplicity = weight_multiplicity

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
        dw = reward * self.A * (error[self.connectome.M] * self.activity_trace[:, np.newaxis])
        weight_mult = _get_weight_multiplier(self.connectome.W, self.weight_multiplicity, self.max_weight)
        dw = dw * self.weight_update_scale * weight_mult

        # Set weight changes to zero where no connection is marked
        dw[self.connectome.NC] = 0

 
        self.connectome.W += dw
        _clip_weights_inplace(self.connectome.W, self.max_weight)

    def step_no_weight_changes(self, pre_spikes, post_spikes, reward=1):
        self.activity_trace = self.smoothact.step(post_spikes)
        for i, j in self.mirror_neurons:
            self.activity_trace[j] = self.activity_trace[i]


class PredictiveCodingSaponati:
    def __init__(self, connectome: Connectome, dt, A=0.001, tau_activity=10.0, gaba_factor=1.0,
                 max_weight=None, weight_update_scale=1.0, weight_multiplicity="multiplicative_factor"):
        """
        Predictive Coding class to represent the predictive coding mechanism.

        """
        self.connectome = connectome

        self.tau_activity = tau_activity 
        self.A = A
        self.gaba_factor = gaba_factor
        self.max_weight = max_weight
        self.weight_update_scale = weight_update_scale
        self.weight_multiplicity = weight_multiplicity

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
        dw = reward * self.A * (error * relative_potentials[:, np.newaxis] + eps_t[:, np.newaxis] * self.p_t)
        weight_mult = _get_weight_multiplier(self.connectome.W, self.weight_multiplicity, self.max_weight)
        dw = dw * self.weight_update_scale * weight_mult

        dw[self.connectome.neuron_population.inhibitory_mask] *= self.gaba_factor

        # Set weight changes to zero where no connection is marked
        dw[self.connectome.NC] = 0
 
        self.connectome.W += dw
        _clip_weights_inplace(self.connectome.W, self.max_weight)

    def step_no_weight_changes(self, pre_spikes, post_spikes, Vs=None, reward=1):
        self.p_t *= self.decay_pre
        self.p_t += pre_spikes
