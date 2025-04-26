import numpy as np

class NeuronState:
    def __init__(self, V, u, spike, params, T=None):
        self.V = V
        self.u = u
        self.spike = spike
        # T is the threshold delta to add after each spike
        if T is None:
            self.T = np.zeros_like(V)
        else:
            self.T = T

        self.params = params
        self.k = params[0]
        self.a = params[1]
        self.b = params[2]
        self.d = params[3]
        self.C = params[4]
        self.Vr = params[5]
        self.Vt = params[6]
        self.Vpeak = params[7]
        self.c = params[8]
        self.delta_V = params[9]
        self.bias = params[10]
        self.threshold_mult = params[11]
        self.threshold_decay = params[12]

        self.n_neurons = V.shape[0]

    def __call__(self):
        return np.array([self.V, self.u, self.spike, self.T])
    
    def __getitem__(self, key):
        return np.array([self.V[key], self.u[key], self.spike[key], self.T[key]])
    
    def __setitem__(self, key, value):
        self.V[key] = value[0]
        self.u[key] = value[1]
        self.spike[key] = value[2]
        self.T[key] = value[3]

    def IZ_Neuron_stepper_euler(self, I, dt):
        # states: states_per_neuron x n_neurons, params: params_per_neuron x n_neurons
        # states = [V, u]
        # params = [k, a, b, d, C, Vr, Vt, Vpeak, c, delta_V]
        # Vectorized version of IZ_Neuron.step_euler

        spike = np.zeros(self.n_neurons, dtype=bool)

        dV = (self.k * (self.V - self.Vr) * (self.V - self.Vt) - self.u + I + self.bias)/self.C
        du = self.a * (self.b * (self.V - self.Vr) - self.u)

        self.V += dt * dV
        self.u += dt * du

        spike_prob = dt * np.exp((self.V - self.Vpeak) / self.delta_V)
        spike_rand = np.random.rand(self.n_neurons)
        spike = spike_rand < spike_prob

        self.V = np.where(spike, self.c, self.V)
        self.u = np.where(spike, self.u + self.d, self.u)
        self.spike = spike


    def IZ_Neuron_stepper_euler_deterministic(self, I, dt):
        # states: states_per_neuron x n_neurons, params: params_per_neuron x n_neurons
        # states = [V, u]
        # params = [k, a, b, d, C, Vr, Vt, Vpeak, c, delta_V]
        # Vectorized version of IZ_Neuron.step_euler

        spike = np.zeros(self.n_neurons, dtype=bool)

        dV = (self.k * (self.V - self.Vr) * (self.V - self.Vt) - self.u + I + self.bias)/self.C
        du = self.a * (self.b * (self.V - self.Vr) - self.u)

        self.V += dt * dV
        self.u += dt * du

        spike = self.V >= self.Vpeak

        self.V = np.where(spike, self.c, self.V)
        self.u = np.where(spike, self.u + self.d, self.u)
        self.spike = spike

    def IZ_Neuron_stepper_adapt(self, I, dt):
            # states: states_per_neuron x n_neurons, params: params_per_neuron x n_neurons
        # states = [V, u]
        # params = [k, a, b, d, C, Vr, Vt, Vpeak, c, delta_V]

        spike = np.zeros(self.n_neurons, dtype=bool)

        dV = (self.k * (self.V - self.Vr) * (self.V - self.Vt) - self.u + I + self.bias)/self.C
        du = self.a * (self.b * (self.V - self.Vr) - self.u)

        self.V += dt * dV
        self.u += dt * du

        eff_threshold = self.Vpeak + self.T

        self.T *= self.threshold_decay

        spike_prob = dt * np.exp((self.V - eff_threshold) / self.delta_V)
        spike_rand = np.random.rand(self.n_neurons)
        spike = spike_rand < spike_prob

        if spike.any():
            self.T = np.where(spike, self.T * self.threshold_mult, self.T)
            self.V = np.where(spike, self.c, self.V)
            self.u = np.where(spike, self.u + self.d, self.u)

        self.spike = spike

    def IZ_Neuron_stepper_adapt_deterministic(self, I, dt):
        # states: states_per_neuron x n_neurons, params: params_per_neuron x n_neurons
        # states = [V, u]
        # params = [k, a, b, d, C, Vr, Vt, Vpeak, c, delta_V]
        # Vectorized version of IZ_Neuron.step_euler

        spike = np.zeros(self.n_neurons, dtype=bool)

        dV = (self.k * (self.V - self.Vr) * (self.V - self.Vt) - self.u + I + self.bias)/self.C
        du = self.a * (self.b * (self.V - self.Vr) - self.u)

        self.V += dt * dV
        self.u += dt * du

        eff_threshold = self.Vpeak + self.T

        self.T *= self.threshold_decay

        spike = self.V >= eff_threshold

        if spike.any():
            self.T = np.where(spike, eff_threshold * self.threshold_mult - self.Vpeak, self.T)
            self.V = np.where(spike, self.c, self.V)
            self.u = np.where(spike, self.u + self.d, self.u)

        self.spike = spike