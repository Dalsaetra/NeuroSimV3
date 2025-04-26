import numpy as np

class Neuron_State:
    def __init__(self, V, u, spike, T=None):
        self.V = V
        self.u = u
        self.spike = spike
        # T is the threshold delta to add after each spike
        if T is None:
            self.T = np.zeros_like(V)
        else:
            self.T = T

    def __call__(self):
        return np.array([self.V, self.u, self.spike, self.T])
    
    def __getitem__(self, key):
        return np.array([self.V[key], self.u[key], self.spike[key], self.T[key]])
    
    def __setitem__(self, key, value):
        self.V[key] = value[0]
        self.u[key] = value[1]
        self.spike[key] = value[2]
        self.T[key] = value[3]

def IZ_Neuron_stepper_euler(states, params, I, dt):
    # states: states_per_neuron x n_neurons, params: params_per_neuron x n_neurons
    # states = [V, u]
    # params = [k, a, b, d, C, Vr, Vt, Vpeak, c, delta_V]
    # Vectorized version of IZ_Neuron.step_euler
    n_neurons = states.shape[1]
    k = params[0]
    a = params[1]
    b = params[2]
    d = params[3]
    C = params[4]
    Vr = params[5]
    Vt = params[6]
    Vpeak = params[7]
    c = params[8]
    delta_V = params[9]
    bias = params[10]
    V = states[0]
    u = states[1]

    spike = np.zeros(n_neurons, dtype=bool)

    dV = (k * (V - Vr) * (V - Vt) - u + I + bias)/C
    du = a * (b * (V - Vr) - u)

    V += dt * dV
    u += dt * du

    spike_prob = dt * np.exp((V - Vpeak) / delta_V)
    spike_rand = np.random.rand(n_neurons)
    spike = spike_rand < spike_prob

    V = np.where(spike, c, V)
    u = np.where(spike, u + d, u)

    return Neuron_State(V, u, spike)


def IZ_Neuron_stepper_euler_deterministic(states, params, I, dt):
    # states: states_per_neuron x n_neurons, params: params_per_neuron x n_neurons
    # states = [V, u]
    # params = [k, a, b, d, C, Vr, Vt, Vpeak, c, delta_V]
    # Vectorized version of IZ_Neuron.step_euler
    n_neurons = states.shape[1]
    # delta_V not used
    k = params[0]
    a = params[1]
    b = params[2]
    d = params[3]
    C = params[4]
    Vr = params[5]
    Vt = params[6]
    Vpeak = params[7]
    c = params[8]
    # delta_V = params[9]
    bias = params[10]
    V = states[0]
    u = states[1]

    spike = np.zeros(n_neurons, dtype=bool)

    dV = (k * (V - Vr) * (V - Vt) - u + I + bias)/C
    du = a * (b * (V - Vr) - u)

    V += dt * dV
    u += dt * du

    spike = V >= Vpeak

    V = np.where(spike, c, V)
    u = np.where(spike, u + d, u)

    return Neuron_State(V, u, spike)

def IZ_Neuron_stepper_adapt(states, params, I, dt):
        # states: states_per_neuron x n_neurons, params: params_per_neuron x n_neurons
    # states = [V, u]
    # params = [k, a, b, d, C, Vr, Vt, Vpeak, c, delta_V]
    # Vectorized version of IZ_Neuron.step_euler
    n_neurons = states.shape[1]
    # delta_V not used
    k = params[0]
    a = params[1]
    b = params[2]
    d = params[3]
    C = params[4]
    Vr = params[5]
    Vt = params[6]
    Vpeak = params[7]
    c = params[8]
    delta_V = params[9]
    bias = params[10]
    threshold_mult = params[11]
    threshold_decay = params[12]
    V = states[0]
    u = states[1]
    T = states[3]

    spike = np.zeros(n_neurons, dtype=bool)

    dV = (k * (V - Vr) * (V - Vt) - u + I + bias)/C
    du = a * (b * (V - Vr) - u)

    V += dt * dV
    u += dt * du

    eff_threshold = Vpeak + T

    T *= threshold_decay

    spike_prob = dt * np.exp((V - eff_threshold) / delta_V)
    spike_rand = np.random.rand(n_neurons)
    spike = spike_rand < spike_prob

    if spike.any():
        T = np.where(spike, T * threshold_mult, T)
        V = np.where(spike, c, V)
        u = np.where(spike, u + d, u)

    return Neuron_State(V, u, spike, T)

def IZ_Neuron_stepper_adapt_deterministic(states, params, I, dt):
    # states: states_per_neuron x n_neurons, params: params_per_neuron x n_neurons
    # states = [V, u]
    # params = [k, a, b, d, C, Vr, Vt, Vpeak, c, delta_V]
    # Vectorized version of IZ_Neuron.step_euler
    n_neurons = states.shape[1]
    # delta_V not used
    k = params[0]
    a = params[1]
    b = params[2]
    d = params[3]
    C = params[4]
    Vr = params[5]
    Vt = params[6]
    Vpeak = params[7]
    c = params[8]
    # delta_V = params[9]
    bias = params[10]
    threshold_mult = params[11]
    threshold_decay = params[12]
    V = states[0]
    u = states[1]
    T = states[3]

    spike = np.zeros(n_neurons, dtype=bool)

    dV = (k * (V - Vr) * (V - Vt) - u + I + bias)/C
    du = a * (b * (V - Vr) - u)

    V += dt * dV
    u += dt * du

    eff_threshold = Vpeak + T

    T *= threshold_decay

    spike = V >= eff_threshold

    if spike.any():
        T = np.where(spike, eff_threshold * threshold_mult - Vpeak, T)
        V = np.where(spike, c, V)
        u = np.where(spike, u + d, u)

    return Neuron_State(V, u, spike, T)