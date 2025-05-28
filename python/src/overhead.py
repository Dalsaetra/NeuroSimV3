import numpy as np
from izhikevich import NeuronState
from connectome import Connectome
from axonal_dynamics import AxonalDynamics
from synapse_dynamics import SynapseDynamics
from neuron_templates import neuron_type_IZ
from input_integration import InputIntegration
from plasticity import STDP, T_STDP, PredictiveCoding, PredictiveCodingSaponati

class SimulationStats:
    def __init__(self):
        self.Vs = []
        self.us = []
        self.spikes = []
        self.ts = []

class Simulation:
    def __init__(self, connectome: Connectome, dt, stepper_type="adapt", state0=None):
        """
        Simulation class to represent the simulation of a neuron population.
        """
        self.dt = dt
        self.connectome = connectome
        self.axonal_dynamics = AxonalDynamics(connectome, self.dt)
        self.synapse_dynamics = SynapseDynamics(connectome, self.dt)
        self.neuron_states = NeuronState(connectome.neuron_population.neuron_population.T, stepper_type=stepper_type, state0=state0)
        self.integrator = InputIntegration(self.synapse_dynamics)
        # self.plasticity = STDP(connectome, self.dt)
        self.plasticity = T_STDP(connectome, self.dt)
        # self.plasticity = PredictiveCoding(connectome, self.dt)
        # self.plasticity = PredictiveCodingSaponati(connectome, self.dt)

        self.stats = SimulationStats()
        self.stats.Vs.append(self.neuron_states.V.copy())
        self.stats.us.append(self.neuron_states.u.copy())
        self.stats.spikes.append(self.neuron_states.spike.copy())

        self.t_now = 0.0
        self.stats.ts.append(self.t_now)


    def step(self, I_ext=None):
        """
        Step the simulation forward in time.
        """
        # Get the synaptic input from the synapse dynamics
        I_syn = self.integrator(self.neuron_states.V, I_ext=I_ext)
        # Update the neuron states
        self.neuron_states.step(I_syn, self.dt)
        post_spikes = self.neuron_states.spike # shape n_neurons x 1
        # Update the axonal dynamics
        pre_spikes = self.axonal_dynamics.check(self.t_now + self.dt) # shape n_neurons x max_synapses
        self.pre_spikes = pre_spikes.copy()  # Store the pre_spikes for plasticity
        # Push the spikes to the axonal dynamics, do it after the pre_spikes are checked,
        # as the spikes comes from the end of the current step
        self.axonal_dynamics.push_many(post_spikes, self.t_now + self.dt)
        # Time step for synapse dynamics (only decay)
        self.synapse_dynamics.decay()
        # Update the synapse weights based on the traces from last step
        self.plasticity.step(pre_spikes, post_spikes, reward=1)
        # self.plasticity.step(pre_spikes, post_spikes, self.neuron_states.V, reward=1)
        # self.plasticity.step(post_spikes, I_syn, reward=1) 
        # Update synapse reaction class from the pre_spikes
        self.synapse_dynamics.spike_input(pre_spikes)
        # Update the current time
        self.t_now += self.dt
        # Store the current state
        self.stats.Vs.append(self.neuron_states.V.copy())
        self.stats.us.append(self.neuron_states.u.copy())
        self.stats.spikes.append(self.neuron_states.spike.copy())
        self.stats.ts.append(self.t_now)
