from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from src.connectome import Connectome


def _resolve_target_array(
    connectome: Connectome,
    target: float | Mapping[int | str, float],
) -> np.ndarray:
    n = connectome.neuron_population.n_neurons
    if not isinstance(target, Mapping):
        return np.full(n, float(target), dtype=float)

    values = np.zeros(n, dtype=float)
    for node in range(n):
        if node in target:
            values[node] = float(target[node])
            continue
        ntype = str(connectome.neuron_population.type_from_neuron_index(node))
        values[node] = float(target.get(ntype, 0.0))
    return values


class FiringRateHomeostaticNormalizer:
    """
    Rate-dependent homeostatic normalization of incoming E and I synapses.

    For each postsynaptic neuron with estimated firing rate r and target rate
    r_target, the update is:

      w_E *= (1 + eta_E * (r_target - r))
      w_I *= (1 - eta_I * (r_target - r))

    Incoming excitatory and inhibitory synapses are updated separately while
    keeping all weights nonnegative.
    """

    def __init__(
        self,
        connectome: Connectome,
        *,
        dt_ms: float,
        target_rate_hz: float | Mapping[int | str, float],
        eta_E: float,
        eta_I: float,
        rate_method: str = "window",
        rate_window_ms: float = 100.0,
        rate_tau_ms: float = 100.0,
        apply_every_ms: float | None = None,
        min_weight: float = 0.0,
        max_weight: float | None = None,
    ):
        if dt_ms <= 0:
            raise ValueError("dt_ms must be > 0.")
        if eta_E < 0 or eta_I < 0:
            raise ValueError("eta_E and eta_I must be >= 0.")
        if rate_method not in ("window", "ema"):
            raise ValueError("rate_method must be 'window' or 'ema'.")
        if rate_method == "window" and rate_window_ms <= 0:
            raise ValueError("rate_window_ms must be > 0 for window rate estimation.")
        if rate_method == "ema" and rate_tau_ms <= 0:
            raise ValueError("rate_tau_ms must be > 0 for EMA rate estimation.")
        if apply_every_ms is not None and apply_every_ms <= 0:
            raise ValueError("apply_every_ms must be > 0 when provided.")
        if max_weight is not None and max_weight < min_weight:
            raise ValueError("max_weight must be >= min_weight.")

        self.connectome = connectome
        self.dt_ms = float(dt_ms)
        self.n_neurons = connectome.neuron_population.n_neurons
        self.target_rate_hz = _resolve_target_array(connectome, target_rate_hz)
        self.eta_E = float(eta_E)
        self.eta_I = float(eta_I)
        self.rate_method = str(rate_method)
        self.rate_window_ms = float(rate_window_ms)
        self.rate_tau_ms = float(rate_tau_ms)
        self.apply_every_ms = apply_every_ms
        self.min_weight = float(min_weight)
        self.max_weight = None if max_weight is None else float(max_weight)

        self.apply_every_steps = 1
        if apply_every_ms is not None:
            self.apply_every_steps = max(1, int(round(float(apply_every_ms) / self.dt_ms)))

        self.step_count = 0
        self.current_rates_hz = np.zeros(self.n_neurons, dtype=float)
        self._instantaneous_scale_hz = 1000.0 / self.dt_ms

        if self.rate_method == "window":
            self.window_steps = max(1, int(round(self.rate_window_ms / self.dt_ms)))
            self.spike_buffer = np.zeros((self.window_steps, self.n_neurons), dtype=np.uint8)
            self.spike_counts = np.zeros(self.n_neurons, dtype=float)
            self.buffer_cursor = 0
            self.buffer_filled = 0
            self.ema_alpha = None
        else:
            self.window_steps = None
            self.spike_buffer = None
            self.spike_counts = None
            self.buffer_cursor = None
            self.buffer_filled = None
            self.ema_alpha = 1.0 - np.exp(-self.dt_ms / self.rate_tau_ms)

        self._build_incoming_index_cache()

    def _build_incoming_index_cache(self):
        valid_rows, valid_cols = np.where(~self.connectome.NC)
        posts = self.connectome.M[valid_rows, valid_cols]
        src_is_inh = np.asarray(self.connectome.neuron_population.inhibitory_mask, dtype=bool)[valid_rows]

        self.incoming_exc_rows: list[np.ndarray] = []
        self.incoming_exc_cols: list[np.ndarray] = []
        self.incoming_inh_rows: list[np.ndarray] = []
        self.incoming_inh_cols: list[np.ndarray] = []

        for node in range(self.n_neurons):
            post_mask = posts == node
            exc_mask = post_mask & (~src_is_inh)
            inh_mask = post_mask & src_is_inh
            self.incoming_exc_rows.append(valid_rows[exc_mask])
            self.incoming_exc_cols.append(valid_cols[exc_mask])
            self.incoming_inh_rows.append(valid_rows[inh_mask])
            self.incoming_inh_cols.append(valid_cols[inh_mask])

    def reset(self):
        self.step_count = 0
        self.current_rates_hz.fill(0.0)
        if self.rate_method == "window":
            self.spike_buffer.fill(0)
            self.spike_counts.fill(0.0)
            self.buffer_cursor = 0
            self.buffer_filled = 0

    def update(self, post_spikes) -> bool:
        spikes = np.asarray(post_spikes, dtype=bool).reshape(-1)
        if spikes.size != self.n_neurons:
            raise ValueError("post_spikes has wrong size for rate normalizer.")

        self.step_count += 1
        if self.rate_method == "window":
            old = self.spike_buffer[self.buffer_cursor].astype(float, copy=False)
            new = spikes.astype(np.uint8, copy=False)
            self.spike_counts += new - old
            self.spike_buffer[self.buffer_cursor] = new
            self.buffer_cursor = (self.buffer_cursor + 1) % self.window_steps
            self.buffer_filled = min(self.buffer_filled + 1, self.window_steps)
            elapsed_ms = self.buffer_filled * self.dt_ms
            if elapsed_ms > 0:
                self.current_rates_hz = self.spike_counts * (1000.0 / elapsed_ms)
            else:
                self.current_rates_hz.fill(0.0)
        else:
            inst_rate_hz = spikes.astype(float, copy=False) * self._instantaneous_scale_hz
            self.current_rates_hz += self.ema_alpha * (inst_rate_hz - self.current_rates_hz)

        if self.step_count % self.apply_every_steps != 0:
            return False

        self.apply()
        return True

    def apply(self):
        rate_error = self.target_rate_hz - self.current_rates_hz
        scale_exc = np.maximum(0.0, 1.0 + self.eta_E * rate_error)
        scale_inh = np.maximum(0.0, 1.0 - self.eta_I * rate_error)

        W = self.connectome.W
        for node in range(self.n_neurons):
            exc_rows = self.incoming_exc_rows[node]
            if exc_rows.size > 0:
                exc_cols = self.incoming_exc_cols[node]
                W[exc_rows, exc_cols] *= scale_exc[node]
            inh_rows = self.incoming_inh_rows[node]
            if inh_rows.size > 0:
                inh_cols = self.incoming_inh_cols[node]
                W[inh_rows, inh_cols] *= scale_inh[node]

        valid = ~self.connectome.NC
        W[valid] = np.maximum(W[valid], self.min_weight)
        if self.max_weight is not None:
            W[valid] = np.minimum(W[valid], self.max_weight)
        self.connectome.mark_graph_weights_stale()


def build_firing_rate_normalizer(
    connectome: Connectome,
    *,
    dt_ms: float,
    config: Mapping,
) -> FiringRateHomeostaticNormalizer:
    return FiringRateHomeostaticNormalizer(
        connectome,
        dt_ms=dt_ms,
        target_rate_hz=config["target_rate_hz"],
        eta_E=config["eta_E"],
        eta_I=config["eta_I"],
        rate_method=config.get("rate_method", "window"),
        rate_window_ms=config.get("rate_window_ms", 100.0),
        rate_tau_ms=config.get("rate_tau_ms", 100.0),
        apply_every_ms=config.get("apply_every_ms"),
        min_weight=config.get("min_weight", 0.0),
        max_weight=config.get("max_weight"),
    )
