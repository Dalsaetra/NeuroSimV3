import numpy as np

class ExternalInput:
    def __call__(self, t, dt):
        raise NotImplementedError

class PoissonInput:
    """
    Generate Poisson-distributed spike inputs per neuron.

    Each neuron i receives spikes according to its firing rate rate[i] (Hz),
    and each spike contributes amplitude[i] to the input (current or conductance).

    Returns a 1D array (n_neurons,) of weighted spike amplitudes at time t.
    """

    def __init__(self, n_neurons, rate, amplitude=1.0, rng=None):
        """
        Parameters
        ----------
        n_neurons : int
            Number of target neurons to stimulate.
        rate : float or np.ndarray
            Instantaneous firing rate (Hz) per neuron. Can be scalar or shape (n_neurons,).
        amplitude : float or np.ndarray
            Spike weight per neuron (e.g., synaptic conductance increment or injected current).
        rng : np.random.Generator, optional
            Numpy random generator for reproducibility.
        """
        self.n_neurons = n_neurons
        self.rate = np.broadcast_to(np.asarray(rate, dtype=float), (n_neurons,))
        self.amplitude = np.broadcast_to(np.asarray(amplitude, dtype=float), (n_neurons,))
        self.rng = rng or np.random.default_rng()

    def __call__(self, dt):
        """
        Generate Poisson spikes at the current timestep.

        Parameters
        ----------
        dt : float
            Simulation timestep (ms).

        Returns
        -------
        spikes : np.ndarray, shape (n_neurons,)
            Weighted spikes: amplitude[i] if a spike occurred, else 0.
        """
        # Spike probability per neuron per time step
        p_spike = self.rate * dt / 1000.0  # Convert rate from Hz to per ms
        rand = self.rng.random(self.n_neurons)
        spikes = (rand < p_spike).astype(float) * self.amplitude
        return spikes

class SinusoidalInput:
    """
    Generate sinusoidal current inputs per neuron.

    Each neuron i receives a current:
        I_i(t) = amplitude[i] * sin(2π * freq[i] * t + phase[i])
    """

    def __init__(self, n_neurons, freq, amplitude=1.0, phase=0.0):
        """
        Parameters
        ----------
        n_neurons : int
            Number of neurons.
        freq : float or np.ndarray
            Frequency (Hz) per neuron.
        amplitude : float or np.ndarray
            Current amplitude per neuron.
        phase : float or np.ndarray
            Initial phase offset per neuron (radians).
        """
        self.n_neurons = n_neurons
        self.freq = np.broadcast_to(np.asarray(freq, dtype=float), (n_neurons,))
        self.amplitude = np.broadcast_to(np.asarray(amplitude, dtype=float), (n_neurons,))
        self.phase = np.broadcast_to(np.asarray(phase, dtype=float), (n_neurons,))
        self.t = 0.0  # internal time tracker

    def __call__(self, dt):
        """
        Advance internal time and return the current vector.

        Parameters
        ----------
        dt : float
            Simulation timestep (ms).

        Returns
        -------
        I_ext : np.ndarray, shape (n_neurons,)
            Sinusoidal current input.
        """
        self.t += dt / 1000.0  # Convert dt from ms to s
        I = self.amplitude * np.sin(2 * np.pi * self.freq * self.t + self.phase)
        return I
