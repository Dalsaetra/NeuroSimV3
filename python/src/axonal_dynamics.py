import numpy as np
import heapq

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

        self._heap: list[tuple[float, int, int]] = []      # (arrival_time, i, j)


    def push_many(self, spikes, t_now):
        """Vectorised insertion of many spikes in one call."""
        # Spikes shape: n_neurons x 1
        if spikes.sum() == 0:
            return
        ii = np.where(spikes)[0]
        v_vals = self.v
        delays = t_now + self.delays[ii, :]  # Delays for the neurons that spiked
        for i in range(len(ii)):
            # NOTE: j is the index of the synapse, not the neuron
            for j in range(self.connectome.max_synapses):
                if not self.connectome.NC[ii[i], j]:
                    heapq.heappush(self._heap, (float(delays[i,j]), int(ii[i]), int(j)))
                    # print("delay: ", delays[i,j] - t_now, "i:", ii[i], "j:", self.connectome.M[ii[i], j], "weight:", self.connectome.W[ii[i], j])

    def check(self, t_now):
        arrived = []
        heap = self._heap
        while heap and heap[0][0] <= t_now:
            _, i, j = heapq.heappop(heap)
            arrived.append((i, j))
        spikes = np.zeros((self.connectome.neuron_population.n_neurons, self.connectome.max_synapses), dtype=bool)
        if arrived:
            rows, cols = zip(*arrived)    
            spikes[rows, cols] = True
        return spikes

    def __len__(self):
        """Number of spikes still in flight."""
        return len(self._heap)

    def next_arrival_time(self):
        """Peek at the earliest scheduled arrival, or None if queue empty."""
        return self._heap[0][0] if self._heap else None