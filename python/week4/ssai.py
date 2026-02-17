
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

# %%
weight_scale = 1.0
g = 1.2984752590298583

J_I = weight_scale * g
J_E = weight_scale
delay_mean_E = 1.5
delay_std_E = delay_mean_E * 0.2
delay_mean_I = 1.5
delay_std_I = delay_mean_I * 0.2
v_ext = 0.45527031369449

excitatory_type = "ss4"
inhibitory_type = "b"

seed = 1234

# %% [markdown]
# ## Generate network

# %%
G = nx.DiGraph()

rng = np.random.default_rng(seed)


n_neurons = 1000

I_percent = 0.2

n_excitatory = int(n_neurons * (1 - I_percent))

density = 0.15

# Add 1000 nodes
for i in range(n_neurons):
    G.add_node(i)

# Assign 800 nodes as excitatory and 200 as inhibitory
# excitatory_nodes = random.sample(range(1000), 800)

for i in range(n_excitatory):
        G.nodes[i]['inhibitory'] = False
        G.nodes[i]['ntype'] = excitatory_type
        G.nodes[i]['layer'] = 0

for i in range(n_excitatory, n_neurons):
        G.nodes[i]['inhibitory'] = True
        G.nodes[i]['ntype'] = inhibitory_type
        G.nodes[i]['layer'] = 0

# For each node, draw m outgoing edges to random nodes
n_out = int(n_neurons * density)
for i in range(n_neurons):
    targets = rng.choice(range(n_neurons), n_out, replace=False)
    for target in targets:
        if G.nodes[i]['inhibitory']:
            weight = J_I
            delay = max(0.1, rng.normal(delay_mean_I, delay_std_I))
        else:
            weight = J_E
            delay = max(0.1, rng.normal(delay_mean_E, delay_std_E))
        G.add_edge(i, target, weight=weight, distance=delay)

# %%
# Redistribute lognormally
G = assign_lognormal_weights_for_ntype(G, "ss4", mu=0.0, sigma=1.643570, w_max=100.0)

# %%
# Plot ss4 weight distribution
# weights = [G[u][v]['weight'] for u, v in G.edges() if G.nodes[u]['ntype'] == 'ss4']
# plt.figure(figsize=(6,4))
# plt.hist(weights, bins=50, log=True)
# plt.title("ss4 Weight Distribution after Lognormal Redistribution")

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

# %%
nmda_weight = np.ones(connectome.neuron_population.n_neurons, dtype=float)
nmda_weight[pop.inhibitory_mask.astype(bool)] = 0.959685703507305
# Invert to make excitatory neurons have NMDA weight 1, inhibitory 0
# nmda_weight

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

# sim = Simulation(connectome, dt, stepper_type="simple", state0=state0,
#                  enable_plasticity=False)
sim = Simulation(connectome, dt, stepper_type="euler_det", state0=state0,
                 enable_plasticity=False, synapse_kwargs={"LT_scale": 1.0, "NMDA_weight": nmda_weight}, synapse_type="standard",
                 enable_debug_logger=True)

# rate = np.zeros(n_neurons)
poisson = PoissonInput(n_neurons, rate=v_ext, amplitude=2.44625509556019)

from tqdm import tqdm

for i in tqdm(range(5000)):
    sensory_spikes = poisson(dt)
    sensory_spikes[pop.inhibitory_mask.astype(bool)] = False
    sim.step(spike_ext=sensory_spikes)

for i in tqdm(range(15000)):
    sim.step()

# sim.plot_voltage_per_type(figsize=(6, 6))

stats = sim.stats.compute_metrics(dt, bin_ms_participation=300, t_start_ms=750.0, t_stop_ms=1750.0)

isi_mean = stats['ISI_CV_mean']
# isi_top = stats["ISI_CV_mean_top10pct"]
isi_E = stats['ISI_CV_mean_E']
isi_I = stats['ISI_CV_mean_I']

sim.plot_spike_raster(figsize=(10, 6), title=f"ISI_CV_Mean: {isi_mean:.3f}, ISI_CV_E: {isi_E:.3f}, ISI_CV_I: {isi_I:.3f}", t_start_ms=0.0, t_stop_ms=3000.0)


# sim.stats.compute_metrics(dt, bin_ms_participation=300, t_start_ms=750.0, t_stop_ms=2250.0)

# %%
# Plot spikes for one neuron
n_idx = 707
t_first = 0
t_last = -1
# plt.plot(np.array(sim.stats.Vs)[t_first:t_last,n_idx])
# plt.plot(np.array(sim.stats.spikes)[t_first:t_last,n_idx] * 10)
# plt.show()
# plt.plot(np.array(sim.stats.us)[t_first:t_last,n_idx])
# plt.show()
plt.plot(np.array(sim.debug_logger.s_ampa)[t_first:t_last,n_idx], label="AMPA")
plt.plot(np.array(sim.debug_logger.s_nmda)[t_first:t_last,n_idx], label="NMDA")
plt.plot(np.array(sim.debug_logger.s_gaba_a)[t_first:t_last,n_idx], label="GABA_A")
plt.plot(np.array(sim.debug_logger.s_gaba_b)[t_first:t_last,n_idx], label="GABA_B")
plt.legend()
plt.show()


# %%
# connectome.compute_metrics(small_world=False)


