import numpy as np

from src.connectome import Connectome

class AxonalDynamics:
    def __init__(self, connectome: Connectome, dt, velocity = 1.0, dendritic_factor = 10.0):
        """
        AxonalDynamics class to represent the axonal dynamics of a neuron population.
        """
        self.connectome = connectome

        self.dt = dt

        self.L = self.connectome.distances
        self.v = velocity
        self.delays = self.L / self.v
        self.delays[self.connectome.dendritic] *= dendritic_factor  # Dendritic delays are longer

        # Simulation uses fixed-step time, so convert delays to discrete arrival steps.
        self.delay_steps = np.maximum(1, np.ceil(self.delays / self.dt).astype(np.int32))

        n_neurons = self.connectome.neuron_population.n_neurons
        self._syn_cols_by_pre = []
        self._delay_steps_by_pre = []
        for i in range(n_neurons):
            cols = np.flatnonzero(~self.connectome.NC[i]).astype(np.int32)
            self._syn_cols_by_pre.append(cols)
            self._delay_steps_by_pre.append(self.delay_steps[i, cols])

        # Ring buffer keyed by absolute simulation step; avoids O(log n) heap ops.
        self._max_delay_steps = int(self.delay_steps.max())
        self._ring_len = self._max_delay_steps + 1
        self._ring_events = [[] for _ in range(self._ring_len)]
        self._ring_epoch = np.full(self._ring_len, -1, dtype=np.int64)
        self._inflight = 0
        self._event_stride = self.connectome.max_synapses
        self._empty_rows = np.empty(0, dtype=np.int32)
        self._empty_cols = np.empty(0, dtype=np.int32)

        self._spike_buf = np.zeros(
            (self.connectome.neuron_population.n_neurons, self.connectome.max_synapses),
            dtype=bool
        )


    def push_many(self, spikes, t_now):
        """Schedule arrivals for all synapses of spiking neurons."""
        spike_rows = np.flatnonzero(spikes)
        if spike_rows.size == 0:
            return
        emit_step = int(np.rint(t_now / self.dt))
        ring_len = self._ring_len
        ring_events = self._ring_events
        ring_epoch = self._ring_epoch
        event_stride = self._event_stride
        inflight = self._inflight

        for i in spike_rows:
            cols = self._syn_cols_by_pre[i]
            if cols.size == 0:
                continue
            arrival_steps = emit_step + self._delay_steps_by_pre[i]
            for j, arrival_step in zip(cols, arrival_steps):
                slot = int(arrival_step % ring_len)
                if ring_epoch[slot] != arrival_step:
                    ring_events[slot].clear()
                    ring_epoch[slot] = arrival_step
                ring_events[slot].append(int(i) * event_stride + int(j))
                inflight += 1
        self._inflight = inflight

    def _pop_arrived_events(self, arrival_step):
        slot = int(arrival_step % self._ring_len)
        if self._ring_epoch[slot] != arrival_step:
            return None
        arrived = self._ring_events[slot]
        if not arrived:
            return None
        self._inflight -= len(arrived)
        return arrived

    def _unpack_arrived_events(self, arrived):
        flat = np.fromiter(arrived, dtype=np.int64, count=len(arrived))
        rows = (flat // self._event_stride).astype(np.int32, copy=False)
        cols = (flat % self._event_stride).astype(np.int32, copy=False)
        arrived.clear()
        return rows, cols

    def check_sparse(self, t_now):
        arrival_step = int(np.rint(t_now / self.dt))
        arrived = self._pop_arrived_events(arrival_step)
        if arrived is None:
            return self._empty_rows, self._empty_cols
        return self._unpack_arrived_events(arrived)

    def check(self, t_now):
        rows, cols = self.check_sparse(t_now)
        return self.materialize_sparse(rows, cols)

    def materialize_sparse(self, rows, cols):
        spikes = self._spike_buf
        spikes.fill(False)
        if rows.size:
            spikes[rows, cols] = True
        return spikes

    def __len__(self):
        """Number of spikes still in flight."""
        return self._inflight

    def next_arrival_time(self):
        """Unsupported for ring-buffer scheduler."""
        return None
