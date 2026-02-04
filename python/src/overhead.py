import numpy as np
import matplotlib.pyplot as plt

from src.izhikevich import NeuronState
from src.connectome import Connectome
from src.axonal_dynamics import AxonalDynamics
from src.synapse_dynamics import SynapseDynamics
from src.neuron_templates import neuron_type_IZ
from src.input_integration import InputIntegration
from src.plasticity import STDP, T_STDP, PredictiveCoding, PredictiveCodingSaponati
from src.utilities import bin_counts, power_spectrum_fft, spectral_entropy

class SimulationStats:
    def __init__(self):
        self.Vs = []
        self.us = []
        self.spikes = []
        self.ts = []

    # --- consolidate views ---
    def spikes_bool(self):
        """Return a (N, T) boolean array from the stepwise list."""
        cols = [np.asarray(s).reshape(-1, 1).astype(bool) for s in self.spikes]
        if not cols:
            return np.zeros((0, 0), dtype=bool)
        return np.hstack(cols)

    def voltages(self):
        """Return a (N, T) float array of membrane potentials (if stored)."""
        cols = [np.asarray(v).reshape(-1, 1) for v in self.Vs]
        return np.hstack(cols) if cols else np.zeros((0, 0), dtype=float)

    def times_ms(self, dt_ms=None):
        """
        Return a (T,) array of times in ms.
        If self.ts already in ms, we trust it; otherwise set dt_ms to build a grid.
        """
        if len(self.ts) > 1:
            # assume user stored actual times; convert to np.array
            t = np.array(self.ts, dtype=float)
            # If they are seconds, convert to ms by heuristic (optional):
            # if t[-1] < 50: t = 1000.0 * t
            return t
        elif dt_ms is not None:
            T = self.spikes_bool().shape[1]
            return np.arange(T, dtype=float) * dt_ms
        else:
            raise ValueError("times_ms: need dt_ms if stats.ts is empty or length 1.")

    def spike_times_list(self, dt_ms=None):
        """
        Return list of length N; each entry is an array of spike times (ms).
        Uses self.ts if available, otherwise uses dt_ms.
        """
        S = self.spikes_bool()
        N, T = S.shape
        if len(self.ts) == T:
            t = np.array(self.ts, dtype=float)
        else:
            if dt_ms is None:
                raise ValueError("Provide dt_ms if self.ts not aligned with spikes.")
            t = np.arange(T, dtype=float) * dt_ms
        times_list = [t[S[i]] for i in range(N)]
        return times_list

    # --- core metrics, including CV_ISI ---
    def compute_metrics(
        self,
        dt_ms,
        bin_ms_fano=300.0,
        bin_ms_corr=50.0,
        refractory_ms=1.5,
        spectrum_from="population",   # "population" or "mean_neuron"
        pop_smooth_ms=0.0,
        bin_ms_participation=200.0     # <--- NEW: window for activity sparsity
    ):
        """
        Returns a dict with:
          - rate_mean_Hz / rate_median_Hz / rate_p95_Hz
          - ISI_CV_median (want around 1)
          - refractory_violations_per_neuron (want 0)
          - Fano_median (bin_ms) (want around 1)
          - mean_noise_corr (bin_ms) (want 0-0.1)
          - pop_spec_entropy (higher is richer spectrum)
          - participation_frac_mean_(bin_ms) / participation_frac_median_(bin_ms) / participation_frac_p95_(bin_ms)
        """
        out = {}
        S = self.spikes_bool()              # (N, T) bool
        if S.size == 0:
            return out
        N, T = S.shape
        T_ms = T * dt_ms
        dt_s = dt_ms / 1000.0
        fs_hz = 1.0 / dt_s

        # --- firing rates (Hz) ---
        spike_counts_total = S.sum(axis=1)
        rates = spike_counts_total / (T_ms / 1000.0)
        out["rate_mean_Hz"] = float(np.nanmean(rates))
        out["rate_median_Hz"] = float(np.nanmedian(rates))
        out["rate_p95_Hz"] = float(np.nanpercentile(rates, 95))
        active_mask = spike_counts_total > 0

        # --- ISI CV per neuron ---
        t = self.times_ms(dt_ms=dt_ms) if len(self.ts) != T else np.array(self.ts, float)
        cvs = np.full(N, np.nan, dtype=float)
        refrac_viol = np.zeros(N, dtype=int)
        for i in range(N):
            ts_i = t[S[i]]
            if ts_i.size >= 3:
                isi = np.diff(ts_i)
                m = isi.mean()
                if m > 0:
                    cvs[i] = isi.std(ddof=1) / m
                # refractory violations
                refrac_viol[i] = int((isi < refractory_ms).sum())
        valid_cvs = np.isfinite(cvs) & active_mask
        out["ISI_CV_median"] = float(np.median(cvs[valid_cvs])) if np.any(valid_cvs) else 0.0
        out["ISI_CV_mean"] = float(np.mean(cvs[valid_cvs])) if np.any(valid_cvs) else 0.0
        inhib_mask = getattr(self, "inhibitory_mask", None)
        if inhib_mask is not None and len(inhib_mask) == N:
            inhib_mask = np.asarray(inhib_mask, dtype=bool)
            exc_mask = ~inhib_mask
            valid_exc = valid_cvs & exc_mask
            valid_inh = valid_cvs & inhib_mask
            out["ISI_CV_mean_E"] = float(np.mean(cvs[valid_exc])) if np.any(valid_exc) else 0.0
            out["ISI_CV_mean_I"] = float(np.mean(cvs[valid_inh])) if np.any(valid_inh) else 0.0
        if np.any(valid_cvs):
            cv_vals = cvs[valid_cvs]
            p90 = np.percentile(cv_vals, 90)
            top_mask = cv_vals >= p90
            out["ISI_CV_mean_top10pct"] = float(np.mean(cv_vals[top_mask])) if np.any(top_mask) else 0.0
        else:
            out["ISI_CV_mean_top10pct"] = 0.0
        out["refractory_violations_per_neuron"] = float(np.nanmean(refrac_viol))

        # --- spike count stats in bins ---
        bin_steps = max(1, int(round(bin_ms_fano / dt_ms)))
        counts = bin_counts(S, bin_steps=bin_steps)   # (N, n_bins)
        if counts.shape[1] >= 2:
            # Fano factor per neuron
            mu = counts.mean(axis=1)
            var = counts.var(axis=1, ddof=1)
            fanos = np.where(mu > 0, var / mu, np.nan)
            valid_fanos = np.isfinite(fanos) & active_mask
            out["Fano_median_%dms" % int(bin_ms_fano)] = float(np.median(fanos[valid_fanos])) if np.any(valid_fanos) else 0.0

            # noise correlation (mean of off-diagonals of correlation matrix)
            X = counts - counts.mean(axis=1, keepdims=True)
            X /= (counts.std(axis=1, keepdims=True) + 1e-9)
            C = (X @ X.T) / X.shape[1]
            iu = np.triu_indices(N, k=1)
            corr_vals = C[iu]
            valid_corr = np.isfinite(corr_vals)
            out["mean_noise_corr_%dms" % int(bin_ms_corr)] = float(np.mean(corr_vals[valid_corr])) if np.any(valid_corr) else 0.0
        else:
            out["Fano_median_%dms" % int(bin_ms_fano)] = 0.0
            out["mean_noise_corr_%dms" % int(bin_ms_corr)] = 0.0

        # --- participation sparsity ---
        part_steps = max(1, int(round(bin_ms_participation / dt_ms)))
        part_counts = bin_counts(S, bin_steps=part_steps)  # (N, n_part_bins)
        if part_counts.shape[1] >= 1:
            # For each bin: fraction of neurons with ≥1 spike
            active_mask = part_counts > 0
            frac_active_per_bin = active_mask.sum(axis=0) / float(N)
            out["participation_frac_mean_%dms"   % int(bin_ms_participation)] = float(np.mean(frac_active_per_bin))
            out["participation_frac_median_%dms" % int(bin_ms_participation)] = float(np.median(frac_active_per_bin))
            out["participation_frac_p95_%dms"    % int(bin_ms_participation)] = float(np.percentile(frac_active_per_bin, 95))
        else:
            out["participation_frac_mean_%dms"   % int(bin_ms_participation)] = np.nan
            out["participation_frac_median_%dms" % int(bin_ms_participation)] = np.nan
            out["participation_frac_p95_%dms"    % int(bin_ms_participation)] = np.nan

        # --- global participation over full simulation ---
        total_active = (S.sum(axis=1) > 0).sum()
        out["participation_frac_total"] = float(total_active / float(N))

        # --- population rate & spectrum entropy ---
        pop_rate = S.sum(axis=0) / dt_s  # spikes/s across population
        if pop_smooth_ms and pop_smooth_ms > 0:
            sig = max(1, int(round(pop_smooth_ms / dt_ms)))
            k = int(6 * sig)
            w = np.arange(-k, k + 1)
            g = np.exp(-0.5 * (w / sig) ** 2); g /= g.sum()
            pop_rate = np.convolve(pop_rate, g, mode="same")

        if spectrum_from == "population":
            f, Pxx = power_spectrum_fft(pop_rate, fs_hz=fs_hz)
        else:
            # mean of individual PSDs (slower, but can be informative)
            Pxx_accum = None
            for i in range(N):
                xi = S[i].astype(float) / dt_s
                fi, Pxx_i = power_spectrum_fft(xi, fs_hz=fs_hz)
                if Pxx_accum is None:
                    f, Pxx_accum = fi, Pxx_i.copy()
                else:
                    Pxx_accum += Pxx_i
            Pxx = Pxx_accum / N

        out["pop_spec_entropy"] = spectral_entropy(Pxx)
        out["pop_psd_freq_hz"] = f
        out["pop_psd"] = Pxx

        # (Optional) add more: branching ratio, participation ratio, etc.
        return out

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
        self.plasticity = STDP(connectome, self.dt)
        # self.plasticity = T_STDP(connectome, self.dt)
        # self.plasticity = PredictiveCoding(connectome, self.dt)
        # self.plasticity = PredictiveCodingSaponati(connectome, self.dt)

        self.stats = SimulationStats()
        self.stats.inhibitory_mask = connectome.neuron_population.inhibitory_mask.copy()
        self.stats.Vs.append(self.neuron_states.V.copy())
        self.stats.us.append(self.neuron_states.u.copy())
        self.stats.spikes.append(self.neuron_states.spike.copy())

        self.t_now = 0.0
        self.stats.ts.append(self.t_now)


    def step(self, I_ext=None, spike_ext=None):
        """
        Step the simulation forward in time.
        """
        # Get the synaptic input from the synapse dynamics (calls synapse_dynamics)
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
        self.plasticity.step(pre_spikes, post_spikes, reward=1.0)
        # self.plasticity.step(pre_spikes, post_spikes, self.neuron_states.V, reward=1)
        # self.plasticity.step(post_spikes, I_syn, reward=1) 
        # Update synapse reaction class from the pre_spikes
        self.synapse_dynamics.spike_input(pre_spikes)
        if spike_ext is not None:
            self.synapse_dynamics.sensory_spike_input(spike_ext)
        # Update the current time
        self.t_now += self.dt
        # Store the current state
        self.stats.Vs.append(self.neuron_states.V.copy())
        self.stats.us.append(self.neuron_states.u.copy())
        self.stats.spikes.append(self.neuron_states.spike.copy())
        self.stats.ts.append(self.t_now)

    def reset_stats(self):
        self.stats = SimulationStats()
        self.stats.inhibitory_mask = self.connectome.neuron_population.inhibitory_mask.copy()
        self.stats.Vs.append(self.neuron_states.V.copy())
        self.stats.us.append(self.neuron_states.u.copy())
        self.stats.spikes.append(self.neuron_states.spike.copy())
        self.stats.ts.append(self.t_now)

    def plot_voltage_per_type(self, dt_ms=None, t_start_ms=None, t_stop_ms=None, figsize=(10, 6)):
        # Example voltage plot: plt.plot(np.array(sim.stats.Vs)[:, pop.get_neurons_from_type("b")])
        # Plot one voltage trace per neuron type, in same figure, with mean
        plt.figure(figsize=figsize)
        Vs = np.array(self.stats.Vs)  # shape (T, N)
        if Vs.size == 0:
            return
        T = Vs.shape[0]
        if len(self.stats.ts) == T:
            t = np.array(self.stats.ts, dtype=float)
        else:
            t = self.stats.times_ms(dt_ms=dt_ms)
        mask = np.ones_like(t, dtype=bool)
        if t_start_ms is not None:
            mask &= t >= t_start_ms
        if t_stop_ms is not None:
            mask &= t <= t_stop_ms
        t_plot = t[mask]
        Vs = Vs[mask, :]
        if t_plot.size == 0:
            return
        types = self.connectome.neuron_population.neuron_types
        for t in range(len(types)):
            type_name = types[t]
            indices = self.connectome.neuron_population.get_neurons_from_type(type_name)
            if len(indices) == 0:
                continue
            plt.subplot(len(types), 1, t + 1)
            for i in indices:
                plt.plot(t_plot, Vs[:, i], alpha=0.3)
            plt.plot(t_plot, Vs[:, indices].mean(axis=1), color='black', linewidth=2)
            plt.title(f'Neuron type: {type_name}')
            plt.ylabel('V (mV)')
            plt.xlabel('Time (ms)')
        plt.tight_layout()
        plt.show()

    def plot_spike_raster(self, dt_ms=None, t_start_ms=None, t_stop_ms=None, figsize=(10, 6), s=8, alpha=0.7, legend=True, title=None, save_path=None):
        """
        Raster plot of spikes across all neurons.
        Colors indicate inhibitory/excitatory; marker shape indicates neuron type.
        """
        S = self.stats.spikes_bool()
        if S.size == 0:
            return

        N, T = S.shape
        if len(self.stats.ts) == T:
            t = np.array(self.stats.ts, dtype=float)
        else:
            t = self.stats.times_ms(dt_ms=dt_ms)
        mask = np.ones_like(t, dtype=bool)
        if t_start_ms is not None:
            mask &= t >= t_start_ms
        if t_stop_ms is not None:
            mask &= t <= t_stop_ms
        t = t[mask]
        S = S[:, mask]
        if t.size == 0:
            return

        pop = self.connectome.neuron_population
        type_names = list(pop.neuron_types)
        markers = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">", "h", "H", "d", "p"]
        type_to_marker = {name: markers[i % len(markers)] for i, name in enumerate(type_names)}

        excit_color = "#1f77b4"
        inhib_color = "#d62728"

        plt.figure(figsize=figsize)
        ax = plt.gca()

        for type_name in type_names:
            indices = pop.get_neurons_from_type(type_name)
            if len(indices) == 0:
                continue

            xs = []
            ys = []
            for i in indices:
                spike_idx = np.flatnonzero(S[i])
                if spike_idx.size == 0:
                    continue
                xs.append(t[spike_idx])
                ys.append(np.full(spike_idx.size, i, dtype=float))

            if not xs:
                continue

            xs = np.concatenate(xs)
            ys = np.concatenate(ys)

            type_idx = pop.type_index_from_neuron_type(type_name)
            is_inhib = bool(pop.inhibitory[type_idx])
            color = inhib_color if is_inhib else excit_color
            label = f"{type_name} ({'I' if is_inhib else 'E'})"

            ax.scatter(xs, ys, s=s, alpha=alpha, marker=type_to_marker[type_name],
                       color=color, edgecolors="none", label=label)

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Neuron index")
        ax.set_ylim(-0.5, N - 0.5)
        if legend:
            ax.legend(loc="upper right", fontsize=8, ncol=2, frameon=False)
        plt.tight_layout()
        if title is not None:
            plt.title(title)
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
