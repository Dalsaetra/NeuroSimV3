
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

from tqdm import tqdm

# %% [markdown]
# ## Constants

# %%
weight_scale = 0.2

g = 10000.0

J_I = weight_scale * g
J_E = weight_scale
delay_mean = 1.5
delay_std = 0.2
v_ext = 1.7

excitatory_type = "ss4"
inhibitory_type = "ss4_I"

# %% [markdown]
# ## Generate network

# %%
G = nx.DiGraph()

# Add 1000 nodes
for i in range(1000):
    G.add_node(i)

# Assign 800 nodes as excitatory and 200 as inhibitory
excitatory_nodes = random.sample(range(1000), 800)
for i in range(1000):
    if i in excitatory_nodes:
        G.nodes[i]['inhibitory'] = False
        G.nodes[i]['ntype'] = excitatory_type
        G.nodes[i]['layer'] = 0
    else:
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
dt = 0.1

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

# Gridsearch

weight_scales = [1.0]

gs = [2.0, 5.0, 20.0]

delay_mean = 30.0
delay_std_prcents = [0.01, 0.2, 0.5]
v_ext = [0.1, 0.15, 0.2]

for weight_scale in weight_scales:
    for g in gs:
        J_I = weight_scale * g
        J_E = weight_scale
        for delay_std_prcent in delay_std_prcents:
            delay_std = delay_mean * delay_std_prcent
            for v_e in v_ext:
                # Update weights and delays
                for i in range(len(connectome.W)):
                    if connectome.neuron_population.inhibitory_mask[i]:
                        connectome.W[i][:] = J_I
                    else:
                        connectome.W[i][:] = J_E
                    connectome.distances[i][:] = np.maximum(0.1, np.random.normal(delay_mean, delay_std, size=connectome.max_synapses))
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

                # rate = np.zeros(n_neurons)
                poisson = PoissonInput(n_neurons, rate=v_e, amplitude=1.0)

                for i in tqdm(range(100000)):
                    sensory_spikes = poisson(dt)
                    sim.step(spike_ext=sensory_spikes)
                    # sim.step(I_ext=I_ext)

                stats = sim.stats.compute_metrics(dt, bin_ms_participation=300)
                isi_mean = stats['ISI_CV_mean']
                isi_top = stats["ISI_CV_mean_top10pct"]

                sim.plot_spike_raster(figsize=(6, 6), title=f"ISI_CV_Mean: {isi_mean}, ISI_CV_top10pct: {isi_top}" ,save_path=f"gs5/spike_raster_ws{weight_scale}_g{g}_d{delay_mean}_std{delay_std}_ve{v_e}.png")