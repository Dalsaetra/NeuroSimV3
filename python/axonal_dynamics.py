import numpy as np
import heapq

from neuron_population import NueronPopulation
from connectome import Connectome

class AxonalDynamics:
    def __init__(self, connectome: Connectome, dt, velocity = 1.0):
        """
        AxonalDynamics class to represent the axonal dynamics of a neuron population.
        """
        self.connectome = connectome

        self.dt = dt

        self.L = self.connectome.distances
        self.v = velocity

        self._heap: list[tuple[float, int, int]] = []      # (arrival_time, i, j)


    def push_many(self, spikes, t_now):
        """Vectorised insertion of many spikes in one call."""
        # Spikes shape: n_neurons x 1
        if spikes.sum() == 0:
            return
        ii = np.where(spikes)[0]
        v_vals = self.v
        delays = t_now + self.L[ii] / v_vals
        for i in range(len(ii)):
            for j in range(self.connectome.max_synapses):
                heapq.heappush(self._heap, (float(delays[i,j]), int(ii[i]), int(j)))

    def check(self, t_now):
        arrived = []
        heap = self._heap
        while heap and heap[0][0] <= t_now:
            _, i, j = heapq.heappop(heap)
            arrived.append((i, j))
        spikes = np.zeros((self.connectome.neuron_population.n_neurons, self.connectome.max_synapses), dtype=bool)
        spikes[arrived] = True
        return spikes

    def __len__(self):
        """Number of spikes still in flight."""
        return len(self._heap)

    def next_arrival_time(self):
        """Peek at the earliest scheduled arrival, or None if queue empty."""
        return self._heap[0][0] if self._heap else None