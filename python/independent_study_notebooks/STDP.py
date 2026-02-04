# %%
import numpy as np
import matplotlib.pyplot as plt

import networkx as nx
from networkx.algorithms import smallworld
import random
from collections import Counter
from scipy.spatial import cKDTree

# %%
from src.neuron_population import NeuronPopulation
from src.connectome import Connectome
from src.overhead import Simulation
from src.neuron_templates import neuron_type_IZ
from src.network_grower import *
from src.network_generators import *
from src.neuron_type_distributor import *
from src.network_weight_distributor import *
from src.external_inputs import *

# %% [markdown]
# ## Constants

# %%
weight_scale = 1.2
g = 0.85

J_I = weight_scale * g
J_E = weight_scale
delay_mean = 10.0
delay_std = delay_mean * 0.5
v_ext = 0.10

excitatory_type = "ss4"
inhibitory_type = "b"

# %% [markdown]
# ## Generate network

# %%
G = nx.DiGraph()

# Add 1000 nodes
for i in range(1000):
    G.add_node(i)

# Assign 800 nodes as excitatory and 200 as inhibitory
for i in range(800):
        G.nodes[i]['inhibitory'] = False
        G.nodes[i]['ntype'] = excitatory_type
        G.nodes[i]['layer'] = 0

for i in range(800, 1000):
        G.nodes[i]['inhibitory'] = True
        G.nodes[i]['ntype'] = inhibitory_type
        G.nodes[i]['layer'] = 0

# For each node, draw 100 outgoing edges to random nodes
for i in range(1000):
    targets = random.sample(range(1000), 100)
    for target in targets:
        if G.nodes[i]['inhibitory']:
            weight = J_I
        else:
            weight = J_E
        delay = max(0.1, np.random.normal(delay_mean, delay_std))
        G.add_edge(i, target, weight=weight, distance=delay)

# %% [markdown]
# ## Simulation setup

# %%
dt = 0.5

# %%
# Neuron population parameters
n_neurons = G.number_of_nodes()
neuron_types = [excitatory_type, inhibitory_type]
n_neuron_types = len(neuron_types)
inhibitory = [False, True]
threshold_decay = np.exp(-dt / 5)

pop = NeuronPopulation(n_neurons, neuron_types, inhibitory, threshold_decay)

# %%
# Connectome
# Max number of outgoing synapses per neuron
max_synapses = max(dict(G.out_degree()).values())

connectome = Connectome(max_synapses, pop)

connectome.nx_to_connectome(G)

# %% [markdown]
# ## Simulation

# %%

# Randomize initial voltages
Vs = np.random.uniform(-100, -70, size=n_neurons)
us = np.random.uniform(0, 400, size=n_neurons)
spikes = np.zeros(n_neurons, dtype=bool)
Ts = np.zeros_like(spikes)

state0 = (Vs,
          us,
          spikes.copy(),
          Ts.copy())

sim = Simulation(connectome, dt, stepper_type="euler_det", state0=state0)

poisson = PoissonInput(n_neurons, rate=v_ext, amplitude=1.0)

from tqdm import tqdm

times_to_plot = [1000, 10000, 30000, 60000, 100000, 150000, 200000, 250000, 299000] # ms
seconds_to_run = 3600

for i in tqdm(range(int(seconds_to_run * 1000 // dt))):
    sensory_spikes = poisson(dt)
    sim.step(spike_ext=sensory_spikes)
    if sim.t_now in times_to_plot:
        sim.plot_spike_raster(figsize=(10, 6), title=f"Spike raster at t={sim.t_now} ms", save_path=f"stdp_long_nogaba_noweightstab/spike_raster_ws_{weight_scale}_g{g}_d{delay_mean}_std{delay_std}_t{sim.t_now}.png")
        sim.reset_stats()
    elif int(sim.t_now) % 10000 == 0:
        sim.reset_stats()
    

# import time
# time_start = time.time()

# Save sim.stats.Vs, sim.stats.spikes, sim.stats.ts for analysis
# np.savez(f"sim_stats_{time_start}.npz", Vs=sim.stats.Vs, spikes=sim.stats.spikes, ts=sim.stats.ts, us=sim.stats.us)
