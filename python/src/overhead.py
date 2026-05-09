import inspect
from collections.abc import Mapping

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from src.izhikevich import NeuronState
from src.connectome import Connectome
from src.axonal_dynamics import AxonalDynamics
from src.synapse_dynamics import SynapseDynamics, SynapseDynamics_Rise, SynapseDynamics_Uncapped
from src.neuron_templates import neuron_type_IZ
from src.input_integration import InputIntegration
from src.normalization import build_firing_rate_normalizer
from src.plasticity import STDP, STDPMasked, DA_BCM, ClopathMasked, T_STDP, PredictiveCoding, PredictiveCodingSaponati
from src.utilities import bin_counts, power_spectrum_fft, spectral_entropy
from src.external_inputs import PoissonInput

class SimulationStats:
    def __init__(self):
        self.Vs = []
        self.us = []
        self.spikes = []
        self.ts = []
        self.separation_property = None
        self.separation_property_history = []
        self.memory_capacity = None
        self.memory_capacity_history = []
        self.generalization_property = None
        self.generalization_property_history = []
        self.regime_persistence_regeneration = None
        self.regime_persistence_regeneration_history = []
        self.input_amplitude_bifurcation = None
        self.input_amplitude_bifurcation_history = []

    @staticmethod
    def effective_rank(matrix, *, center=True, normalize=True, eps=1e-12):
        """
        Effective rank from the entropy of singular values.

        Rows are treated as features and columns as samples. With the default
        normalization, each feature is centered and z-scored across samples
        before the SVD so high-rate neurons do not dominate the rank estimate.
        """
        X = np.asarray(matrix, dtype=float)
        if X.ndim != 2 or X.size == 0:
            return 0.0
        if center:
            X = X - np.mean(X, axis=1, keepdims=True)
        if normalize:
            scale = np.std(X, axis=1, keepdims=True)
            scale[scale < eps] = 1.0
            X = X / scale
        if not np.any(np.isfinite(X)):
            return 0.0
        X = np.nan_to_num(X, copy=False)
        svals = np.linalg.svd(X, compute_uv=False, full_matrices=False)
        svals = svals[svals > eps]
        if svals.size == 0:
            return 0.0
        probs = svals / np.sum(svals)
        entropy = -float(np.sum(probs * np.log(probs + eps)))
        return float(np.exp(entropy))

    @classmethod
    def separation_effective_rank_metrics(
        cls,
        states,
        *,
        center=True,
        normalize=True,
        eps=1e-12,
    ):
        """
        Compute time-resolved and pooled effective-rank separation metrics.

        Parameters
        ----------
        states : array, shape (n_trials, n_time_bins, n_features)
            Filtered reservoir states. Each time slice is ranked as
            features x trials; the pooled matrix is features x trial-time samples.
        """
        S = np.asarray(states, dtype=float)
        if S.ndim != 3:
            raise ValueError("states must have shape (n_trials, n_time_bins, n_features).")
        n_trials, n_time_bins, n_features = S.shape
        if n_trials == 0 or n_time_bins == 0 or n_features == 0:
            return {
                "effective_rank_by_time": np.zeros(0, dtype=float),
                "effective_rank_norm_by_time": np.zeros(0, dtype=float),
                "effective_rank_mean": 0.0,
                "effective_rank_max": 0.0,
                "effective_rank_final": 0.0,
                "effective_rank_norm_mean": 0.0,
                "effective_rank_norm_max": 0.0,
                "effective_rank_norm_final": 0.0,
                "effective_rank_pooled": 0.0,
                "effective_rank_norm_pooled": 0.0,
                "rank_max_by_time": float(min(n_features, n_trials)),
                "rank_max_pooled": float(min(n_features, n_trials * n_time_bins)),
            }

        eranks = np.zeros(n_time_bins, dtype=float)
        rank_max_time = float(min(n_features, n_trials))
        for t_idx in range(n_time_bins):
            eranks[t_idx] = cls.effective_rank(
                S[:, t_idx, :].T,
                center=center,
                normalize=normalize,
                eps=eps,
            )
        eranks_norm = eranks / rank_max_time if rank_max_time > 0 else np.zeros_like(eranks)

        pooled = S.reshape(n_trials * n_time_bins, n_features).T
        pooled_rank_max = float(min(n_features, n_trials * n_time_bins))
        pooled_erank = cls.effective_rank(
            pooled,
            center=center,
            normalize=normalize,
            eps=eps,
        )
        pooled_norm = pooled_erank / pooled_rank_max if pooled_rank_max > 0 else 0.0

        return {
            "effective_rank_by_time": eranks,
            "effective_rank_norm_by_time": eranks_norm,
            "effective_rank_mean": float(np.mean(eranks)),
            "effective_rank_max": float(np.max(eranks)),
            "effective_rank_final": float(eranks[-1]),
            "effective_rank_norm_mean": float(np.mean(eranks_norm)),
            "effective_rank_norm_max": float(np.max(eranks_norm)),
            "effective_rank_norm_final": float(eranks_norm[-1]),
            "effective_rank_pooled": float(pooled_erank),
            "effective_rank_norm_pooled": float(pooled_norm),
            "rank_max_by_time": rank_max_time,
            "rank_max_pooled": pooled_rank_max,
        }

    def save_separation_property(self, result):
        self.separation_property = result
        self.separation_property_history.append(result)
        return result

    @staticmethod
    def _r2_score(y_true, y_pred, eps=1e-12):
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        if y_true.size == 0 or y_true.shape != y_pred.shape:
            return 0.0
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        if ss_tot <= eps:
            return 0.0
        return float(1.0 - ss_res / ss_tot)

    @staticmethod
    def _fit_ridge_readout(X_train, y_train, *, alpha=1.0, standardize=True, eps=1e-12):
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float).reshape(-1)
        if X_train.ndim != 2:
            raise ValueError("X_train must be 2D.")
        if X_train.shape[0] != y_train.size:
            raise ValueError("X_train rows must match y_train length.")

        if standardize:
            mu = np.mean(X_train, axis=0, keepdims=True)
            sigma = np.std(X_train, axis=0, keepdims=True)
            sigma[sigma < eps] = 1.0
            Xs = (X_train - mu) / sigma
        else:
            mu = np.zeros((1, X_train.shape[1]), dtype=float)
            sigma = np.ones((1, X_train.shape[1]), dtype=float)
            Xs = X_train

        X_aug = np.hstack([np.ones((Xs.shape[0], 1), dtype=float), Xs])
        reg = np.eye(X_aug.shape[1], dtype=float) * float(alpha)
        reg[0, 0] = 0.0
        coef = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ y_train)
        return coef, mu.reshape(-1), sigma.reshape(-1)

    @staticmethod
    def _predict_ridge_readout(X, coef, mu, sigma):
        X = np.asarray(X, dtype=float)
        Xs = (X - mu.reshape(1, -1)) / sigma.reshape(1, -1)
        X_aug = np.hstack([np.ones((Xs.shape[0], 1), dtype=float), Xs])
        return X_aug @ np.asarray(coef, dtype=float)

    @classmethod
    def memory_capacity_metrics(
        cls,
        states,
        input_values,
        *,
        delays_bins,
        train_fraction=0.7,
        ridge_alpha=1.0,
        standardize=True,
        clip_negative_r2=True,
        shuffle_split=False,
        delay_bin_ms=None,
        rng=None,
        show_progress=False,
    ):
        """
        Train one ridge readout per delay to reconstruct u(t-k) from x(t).

        `states` is shape (n_time_bins, n_features), and `input_values` is a
        scalar input stream of length n_time_bins sampled at the same bins.
        """
        X_all = np.asarray(states, dtype=float)
        u = np.asarray(input_values, dtype=float).reshape(-1)
        delays = np.asarray(delays_bins, dtype=int).reshape(-1)
        if X_all.ndim != 2:
            raise ValueError("states must have shape (n_time_bins, n_features).")
        if X_all.shape[0] != u.size:
            raise ValueError("states and input_values must have the same time-bin length.")
        if np.any(delays <= 0):
            raise ValueError("delays_bins must contain positive integer delays.")
        if train_fraction <= 0 or train_fraction >= 1:
            raise ValueError("train_fraction must be in (0, 1).")
        if rng is None:
            rng = np.random.default_rng()

        r2_train = np.full(delays.size, np.nan, dtype=float)
        r2_test = np.full(delays.size, np.nan, dtype=float)
        n_samples_by_delay = np.zeros(delays.size, dtype=int)
        coefficients = []

        delay_iter = enumerate(delays)
        if show_progress:
            try:
                from tqdm.auto import tqdm
                delay_iter = enumerate(tqdm(delays, desc="Memory capacity readouts", unit="delay"))
            except Exception:
                pass

        for i, delay in delay_iter:
            if delay >= u.size:
                coefficients.append(None)
                continue
            X = X_all[delay:, :]
            y = u[:-delay]
            n_samples = y.size
            n_samples_by_delay[i] = int(n_samples)
            if n_samples < 3:
                coefficients.append(None)
                continue

            indices = np.arange(n_samples, dtype=int)
            if shuffle_split:
                indices = rng.permutation(indices)
            n_train = int(np.floor(float(train_fraction) * n_samples))
            n_train = max(1, min(n_train, n_samples - 1))
            train_idx = indices[:n_train]
            test_idx = indices[n_train:]
            if test_idx.size == 0:
                coefficients.append(None)
                continue

            coef, mu, sigma = cls._fit_ridge_readout(
                X[train_idx],
                y[train_idx],
                alpha=ridge_alpha,
                standardize=standardize,
            )
            yhat_train = cls._predict_ridge_readout(X[train_idx], coef, mu, sigma)
            yhat_test = cls._predict_ridge_readout(X[test_idx], coef, mu, sigma)
            r2_train[i] = cls._r2_score(y[train_idx], yhat_train)
            r2_test[i] = cls._r2_score(y[test_idx], yhat_test)
            coefficients.append({"coef": coef, "mu": mu, "sigma": sigma})

        valid = np.isfinite(r2_test)
        r2_for_capacity = np.where(valid, r2_test, 0.0)
        if clip_negative_r2:
            r2_for_capacity = np.maximum(0.0, r2_for_capacity)
        memory_capacity = float(np.sum(r2_for_capacity))
        n_valid_delays = int(np.sum(valid))
        n_features = int(X_all.shape[1])
        delay_bound = float(max(1, n_valid_delays))
        rank_bound = float(max(1, min(n_features, n_valid_delays)))
        delay_unit_ms = 1.0 if delay_bin_ms is None else float(delay_bin_ms)
        delays_ms = delays.astype(float) * delay_unit_ms

        first_negative_delay_bins = np.nan
        first_negative_delay_ms = np.nan
        valid_order = np.argsort(delays[valid])
        valid_delays = delays[valid][valid_order]
        valid_delays_ms = delays_ms[valid][valid_order]
        valid_r2 = r2_test[valid][valid_order]
        if valid_r2.size:
            if valid_r2[0] < 0.0:
                first_negative_delay_bins = 0.0
                first_negative_delay_ms = 0.0
            else:
                negative_idx = np.flatnonzero(valid_r2 < 0.0)
                if negative_idx.size:
                    first_idx = int(negative_idx[0])
                    first_negative_delay_bins = float(valid_delays[first_idx])
                    first_negative_delay_ms = float(valid_delays_ms[first_idx])

        return {
            "delays_bins": delays,
            "delays_ms": delays_ms,
            "r2_train_by_delay": r2_train,
            "r2_test_by_delay": r2_test,
            "r2_capacity_by_delay": r2_for_capacity,
            "n_samples_by_delay": n_samples_by_delay,
            "memory_capacity": memory_capacity,
            "memory_capacity_mean_r2": float(memory_capacity / delay_bound),
            "memory_capacity_norm_delay_bound": float(memory_capacity / delay_bound),
            "memory_capacity_norm_rank_bound": float(memory_capacity / rank_bound),
            "memory_first_negative_delay_bins": first_negative_delay_bins,
            "memory_first_negative_delay_ms": first_negative_delay_ms,
            "n_valid_delays": n_valid_delays,
            "n_features": n_features,
            "clip_negative_r2": bool(clip_negative_r2),
            "coefficients": coefficients,
        }

    def save_memory_capacity(self, result):
        self.memory_capacity = result
        self.memory_capacity_history.append(result)
        return result

    @staticmethod
    def _stratified_train_test_indices(labels, *, train_fraction=0.7, rng=None):
        labels = np.asarray(labels)
        if labels.ndim != 1:
            labels = labels.reshape(-1)
        if rng is None:
            rng = np.random.default_rng()

        train_parts = []
        test_parts = []
        for label in np.unique(labels):
            idx = np.flatnonzero(labels == label)
            idx = rng.permutation(idx)
            if idx.size < 2:
                raise ValueError("Each class needs at least two samples for stratified train/test splitting.")
            n_train = int(np.floor(float(train_fraction) * idx.size))
            n_train = max(1, min(n_train, idx.size - 1))
            train_parts.append(idx[:n_train])
            test_parts.append(idx[n_train:])

        train_idx = rng.permutation(np.concatenate(train_parts))
        test_idx = rng.permutation(np.concatenate(test_parts))
        return train_idx.astype(int), test_idx.astype(int)

    @staticmethod
    def _fit_ridge_classifier(X_train, y_train, *, classes, alpha=1.0, standardize=True, eps=1e-12):
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train).reshape(-1)
        classes = np.asarray(classes)
        if X_train.ndim != 2:
            raise ValueError("X_train must be 2D.")
        if X_train.shape[0] != y_train.size:
            raise ValueError("X_train rows must match y_train length.")

        if standardize:
            mu = np.mean(X_train, axis=0, keepdims=True)
            sigma = np.std(X_train, axis=0, keepdims=True)
            sigma[sigma < eps] = 1.0
            Xs = (X_train - mu) / sigma
        else:
            mu = np.zeros((1, X_train.shape[1]), dtype=float)
            sigma = np.ones((1, X_train.shape[1]), dtype=float)
            Xs = X_train

        Y = np.zeros((y_train.size, classes.size), dtype=float)
        class_to_col = {label: i for i, label in enumerate(classes.tolist())}
        for row_idx, label in enumerate(y_train.tolist()):
            Y[row_idx, class_to_col[label]] = 1.0

        X_aug = np.hstack([np.ones((Xs.shape[0], 1), dtype=float), Xs])
        reg = np.eye(X_aug.shape[1], dtype=float) * float(alpha)
        reg[0, 0] = 0.0
        weights = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ Y)
        return weights, mu.reshape(-1), sigma.reshape(-1)

    @staticmethod
    def _predict_ridge_classifier(X, weights, mu, sigma, *, classes):
        X = np.asarray(X, dtype=float)
        Xs = (X - mu.reshape(1, -1)) / sigma.reshape(1, -1)
        X_aug = np.hstack([np.ones((Xs.shape[0], 1), dtype=float), Xs])
        scores = X_aug @ np.asarray(weights, dtype=float)
        pred_cols = np.argmax(scores, axis=1)
        return np.asarray(classes)[pred_cols], scores

    @classmethod
    def generalization_property_metrics(
        cls,
        state_vectors,
        labels,
        *,
        train_fraction=0.7,
        ridge_alpha=1.0,
        lda_regularization=1e-6,
        standardize=True,
        rng=None,
        eps=1e-12,
    ):
        """
        Fisher/LDA-style class consistency and linear-readout metrics.

        `state_vectors` has shape (n_samples, n_features). `labels` gives the
        input class for each sample. Features are z-scored by default before
        scatter metrics and readout training.
        """
        X = np.asarray(state_vectors, dtype=float)
        y = np.asarray(labels).reshape(-1)
        if X.ndim != 2:
            raise ValueError("state_vectors must have shape (n_samples, n_features).")
        if X.shape[0] != y.size:
            raise ValueError("state_vectors rows must match labels length.")
        if X.shape[0] < 2 or X.shape[1] == 0:
            raise ValueError("state_vectors must contain at least two samples and one feature.")
        if train_fraction <= 0 or train_fraction >= 1:
            raise ValueError("train_fraction must be in (0, 1).")
        if rng is None:
            rng = np.random.default_rng()

        classes = np.unique(y)
        if classes.size < 2:
            raise ValueError("At least two classes are required.")

        if standardize:
            feature_mean = np.mean(X, axis=0, keepdims=True)
            feature_std = np.std(X, axis=0, keepdims=True)
            feature_std[feature_std < eps] = 1.0
            X_metric = (X - feature_mean) / feature_std
        else:
            X_metric = X.copy()

        overall_mean = np.mean(X_metric, axis=0)
        n_features = int(X_metric.shape[1])
        Sw = np.zeros((n_features, n_features), dtype=float)
        Sb = np.zeros((n_features, n_features), dtype=float)
        class_counts = np.zeros(classes.size, dtype=int)
        class_means = np.zeros((classes.size, n_features), dtype=float)
        within_trace_by_class = np.zeros(classes.size, dtype=float)

        for class_idx, label in enumerate(classes):
            Xc = X_metric[y == label]
            class_counts[class_idx] = int(Xc.shape[0])
            mu_c = np.mean(Xc, axis=0)
            class_means[class_idx, :] = mu_c
            centered = Xc - mu_c
            Sw += centered.T @ centered
            mean_delta = (mu_c - overall_mean).reshape(-1, 1)
            Sb += Xc.shape[0] * (mean_delta @ mean_delta.T)
            within_trace_by_class[class_idx] = float(np.sum(centered * centered))

        sw_trace = float(np.trace(Sw))
        sb_trace = float(np.trace(Sb))
        total_trace = float(sw_trace + sb_trace)
        scatter_ratio = float(sb_trace / (sw_trace + eps))
        separability_index = float(sb_trace / (total_trace + eps))
        mean_pairwise_class_distance = 0.0
        if classes.size > 1:
            distances = []
            for i in range(classes.size):
                for j in range(i + 1, classes.size):
                    distances.append(float(np.linalg.norm(class_means[i] - class_means[j])))
            mean_pairwise_class_distance = float(np.mean(distances)) if distances else 0.0

        reg_scale = float(lda_regularization) * max(1.0, sw_trace / max(1, n_features))
        fisher_matrix_trace = np.nan
        fisher_eigenvalues = np.zeros(0, dtype=float)
        try:
            A = Sw + np.eye(n_features, dtype=float) * reg_scale
            fisher_matrix = np.linalg.solve(A, Sb)
            fisher_matrix_trace = float(np.trace(fisher_matrix))
            fisher_eigenvalues = np.real(np.linalg.eigvals(fisher_matrix))
            fisher_eigenvalues = np.sort(fisher_eigenvalues[fisher_eigenvalues > eps])[::-1]
        except np.linalg.LinAlgError:
            fisher_matrix_trace = np.nan
            fisher_eigenvalues = np.zeros(0, dtype=float)

        class_mean_erank = cls.effective_rank(
            class_means.T,
            center=True,
            normalize=True,
            eps=eps,
        )
        class_mean_erank_norm = float(class_mean_erank / max(1, min(classes.size, n_features)))

        train_idx, test_idx = cls._stratified_train_test_indices(
            y,
            train_fraction=train_fraction,
            rng=rng,
        )
        weights, mu, sigma = cls._fit_ridge_classifier(
            X[train_idx],
            y[train_idx],
            classes=classes,
            alpha=ridge_alpha,
            standardize=standardize,
            eps=eps,
        )
        pred_train, scores_train = cls._predict_ridge_classifier(
            X[train_idx],
            weights,
            mu,
            sigma,
            classes=classes,
        )
        pred_test, scores_test = cls._predict_ridge_classifier(
            X[test_idx],
            weights,
            mu,
            sigma,
            classes=classes,
        )
        train_accuracy = float(np.mean(pred_train == y[train_idx]))
        test_accuracy = float(np.mean(pred_test == y[test_idx]))

        confusion = np.zeros((classes.size, classes.size), dtype=int)
        class_to_idx = {label: i for i, label in enumerate(classes.tolist())}
        for true_label, pred_label in zip(y[test_idx].tolist(), pred_test.tolist()):
            confusion[class_to_idx[true_label], class_to_idx[pred_label]] += 1
        with np.errstate(divide="ignore", invalid="ignore"):
            per_class_accuracy = np.diag(confusion) / np.maximum(1, np.sum(confusion, axis=1))
        balanced_accuracy = float(np.mean(per_class_accuracy))

        return {
            "within_scatter_trace": sw_trace,
            "between_scatter_trace": sb_trace,
            "total_scatter_trace": total_trace,
            "between_within_scatter_ratio": scatter_ratio,
            "separability_index": separability_index,
            "regularized_fisher_trace": fisher_matrix_trace,
            "fisher_eigenvalues": fisher_eigenvalues,
            "class_mean_effective_rank": float(class_mean_erank),
            "class_mean_effective_rank_norm": class_mean_erank_norm,
            "mean_pairwise_class_distance": mean_pairwise_class_distance,
            "within_scatter_trace_by_class": within_trace_by_class,
            "class_counts": class_counts,
            "classes": classes,
            "linear_readout_train_accuracy": train_accuracy,
            "linear_readout_test_accuracy": test_accuracy,
            "linear_readout_balanced_accuracy": balanced_accuracy,
            "linear_readout_chance_accuracy": float(1.0 / classes.size),
            "confusion_matrix": confusion,
            "per_class_accuracy": per_class_accuracy,
            "train_indices": train_idx,
            "test_indices": test_idx,
            "pred_train": pred_train,
            "pred_test": pred_test,
            "scores_train": scores_train,
            "scores_test": scores_test,
            "classifier_weights": weights,
            "classifier_feature_mean": mu,
            "classifier_feature_std": sigma,
        }

    def save_generalization_property(self, result):
        self.generalization_property = result
        self.generalization_property_history.append(result)
        return result

    def save_regime_persistence_regeneration(self, result):
        self.regime_persistence_regeneration = result
        self.regime_persistence_regeneration_history.append(result)
        return result

    def save_input_amplitude_bifurcation(self, result):
        self.input_amplitude_bifurcation = result
        self.input_amplitude_bifurcation_history.append(result)
        return result

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
        refractory_ms=0.5,
        spectrum_from="population",   # "population" or "mean_neuron"
        pop_smooth_ms=0.0,
        bin_ms_participation=200.0,     # <--- NEW: window for activity sparsity
        bin_ms_synchrony=3.0,
        t_start_ms=None,
        t_stop_ms=None,
    ):
        """
        Returns a dict with:
          - rate_mean_Hz / rate_std_Hz / rate_median_Hz / rate_p95_Hz
          - mean_voltage_mV / mean_voltage_mV_E / mean_voltage_mV_I
          - ISI_CV_median (want around 1)
          - refractory_violations_per_neuron (want 0)
          - Fano_median (bin_ms) (want around 1)
          - mean_noise_corr (bin_ms) (want 0-0.1)
          - CV_R (same bin sizes as mean_noise_corr; population-rate CV over time)
          - potjans_diesmann_synchrony_(bin_ms): variance / mean of population spike counts (authors define <8 as asynchronous)
          - pop_spec_entropy (higher is richer spectrum)
          - psd_peak_freq_hz / psd_peak_amplitude / psd_peak_ratio (dominant oscillation in 2-120 Hz band)
          - participation_frac_mean_(bin_ms) / participation_frac_median_(bin_ms) / participation_frac_p95_(bin_ms)
        """
        out = {}
        S = self.spikes_bool()              # (N, T) bool
        if S.size == 0:
            return out
        N, T = S.shape
        t_full = self.times_ms(dt_ms=dt_ms) if len(self.ts) != T else np.array(self.ts, float)
        mask = np.ones_like(t_full, dtype=bool)
        if t_start_ms is not None:
            mask &= t_full >= t_start_ms
        if t_stop_ms is not None:
            mask &= t_full <= t_stop_ms
        if not np.all(mask):
            S = S[:, mask]
            t_full = t_full[mask]
        N, T = S.shape
        if T == 0:
            return out
        T_ms = T * dt_ms
        dt_s = dt_ms / 1000.0
        fs_hz = 1.0 / dt_s
        inhib_mask = getattr(self, "inhibitory_mask", None)
        if inhib_mask is not None and len(inhib_mask) == N:
            inhib_mask = np.asarray(inhib_mask, dtype=bool)
            exc_mask = ~inhib_mask
        else:
            inhib_mask = None
            exc_mask = None

        # --- membrane voltage stats (mV) ---
        V = self.voltages()
        if V.size > 0 and V.shape[1] == len(mask):
            V = V[:, mask]
            if V.shape[1] > 0:
                out["mean_voltage_mV"] = float(np.mean(V))
                # Ignore the first 500 timesteps (50 ms at dt=0.1 ms) for min-voltage stats.
                V_min = V[:, 500:] if V.shape[1] > 500 else V
                out["min_voltage_mV"] = float(np.min(V_min))
                if inhib_mask is not None:
                    out["mean_voltage_mV_E"] = float(np.mean(V[exc_mask])) if np.any(exc_mask) else np.nan
                    out["mean_voltage_mV_I"] = float(np.mean(V[inhib_mask])) if np.any(inhib_mask) else np.nan
                    out["min_voltage_mV_E"] = float(np.min(V_min[exc_mask])) if np.any(exc_mask) else np.nan
                    out["min_voltage_mV_I"] = float(np.min(V_min[inhib_mask])) if np.any(inhib_mask) else np.nan

        # --- firing rates (Hz) ---
        spike_counts_total = S.sum(axis=1)
        rates = spike_counts_total / (T_ms / 1000.0)
        out["rate_mean_Hz"] = float(np.nanmean(rates))
        out["rate_std_Hz"] = float(np.nanstd(rates))
        out["rate_median_Hz"] = float(np.nanmedian(rates))
        out["rate_p95_Hz"] = float(np.nanpercentile(rates, 95))
        active2_mask = spike_counts_total >= 2
        out["rate_mean_Hz_active2spk"] = float(np.nanmean(rates[active2_mask])) if np.any(active2_mask) else 0.0
        out["rate_std_Hz_active2spk"] = float(np.nanstd(rates[active2_mask])) if np.any(active2_mask) else 0.0
        if inhib_mask is not None:
            out["rate_mean_Hz_E"] = float(np.nanmean(rates[exc_mask])) if np.any(exc_mask) else 0.0
            out["rate_std_Hz_E"] = float(np.nanstd(rates[exc_mask])) if np.any(exc_mask) else 0.0
            out["rate_mean_Hz_I"] = float(np.nanmean(rates[inhib_mask])) if np.any(inhib_mask) else 0.0
            out["rate_std_Hz_I"] = float(np.nanstd(rates[inhib_mask])) if np.any(inhib_mask) else 0.0
            exc_active2 = exc_mask & active2_mask
            inh_active2 = inhib_mask & active2_mask
            out["rate_mean_Hz_E_active2spk"] = float(np.nanmean(rates[exc_active2])) if np.any(exc_active2) else 0.0
            out["rate_std_Hz_E_active2spk"] = float(np.nanstd(rates[exc_active2])) if np.any(exc_active2) else 0.0
            out["rate_mean_Hz_I_active2spk"] = float(np.nanmean(rates[inh_active2])) if np.any(inh_active2) else 0.0
            out["rate_std_Hz_I_active2spk"] = float(np.nanstd(rates[inh_active2])) if np.any(inh_active2) else 0.0
        active_mask = spike_counts_total > 0

        # --- ISI CV per neuron ---
        t = t_full
        cvs = np.full(N, np.nan, dtype=float)
        refrac_viol = np.zeros(N, dtype=int)
        for i in range(N):
            ts_i = t[S[i]]
            if ts_i.size >= 2:
                isi = np.diff(ts_i)
                m = isi.mean()
                if m > 0:
                    cvs[i] = isi.std(ddof=1) / m
                # refractory violations
                refrac_viol[i] = int((isi < refractory_ms).sum())
        # ISI CV is only defined for neurons with at least 2 spikes in the analysis window.
        cv_eligible_mask = spike_counts_total >= 2
        valid_cvs = np.isfinite(cvs) & cv_eligible_mask
        out["ISI_CV_median"] = float(np.median(cvs[valid_cvs])) if np.any(valid_cvs) else 0.0
        out["ISI_CV_mean"] = float(np.mean(cvs[valid_cvs])) if np.any(valid_cvs) else 0.0
        if inhib_mask is not None:
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
        fano_bins = [2, 10, 50, 100, 300, 500, 1000]
        for bin_ms_fano in fano_bins:
            bin_steps = max(1, int(round(bin_ms_fano / dt_ms)))
            counts_fano = bin_counts(S, bin_steps=bin_steps)  # (N, n_fano_bins)
            if counts_fano.shape[1] >= 2:
                # Fano factor per neuron
                mu = counts_fano.mean(axis=1)
                var = counts_fano.var(axis=1, ddof=1)
                fanos = np.where(mu > 0, var / mu, np.nan)
                valid_fanos = np.isfinite(fanos) & active_mask
                out["Fano_median_%dms" % int(bin_ms_fano)] = float(np.median(fanos[valid_fanos])) if np.any(valid_fanos) else 0.0
            else:
                out["Fano_median_%dms" % int(bin_ms_fano)] = 0.0

        # --- Potjans & Diesmann-style synchrony from population spike-count histogram ---
        sync_steps = max(1, int(round(float(bin_ms_synchrony) / dt_ms)))
        sync_counts = bin_counts(S, bin_steps=sync_steps).sum(axis=0).astype(float)
        sync_bin_label = ("%g" % float(bin_ms_synchrony)).replace(".", "p")
        if sync_counts.size >= 1:
            sync_mean = float(np.mean(sync_counts))
            out[f"potjans_diesmann_synchrony_{sync_bin_label}ms"] = (
                float(np.var(sync_counts) / (sync_mean + 1e-12))
                if sync_mean > 0.0 else 0.0
            )
        else:
            out[f"potjans_diesmann_synchrony_{sync_bin_label}ms"] = 0.0

        corr_bins = [2, 10, 50, 100, 300, 500, 1000]


        for bin_ms_corr in corr_bins:
            # choose bin so ~2.5 spikes/bin
            # mean_rate = rates[active_mask].mean()  # Hz
            # h_ms = max(10.0, 2500.0 / mean_rate)   # 2.5 spikes ≈ 2500 ms·Hz
            h_ms = bin_ms_corr
            bin_steps = max(1, int(round(h_ms / dt_ms)))
            counts_corr = bin_counts(S, bin_steps=bin_steps)   # (N, n_bins)

            if counts_corr.shape[1] >= 2:
                pop_counts = counts_corr.sum(axis=0).astype(float)
                pop_rate_mean = float(np.mean(pop_counts))
                if pop_rate_mean > 0.0:
                    out["CV_R_%dms" % int(h_ms)] = float(np.std(pop_counts, ddof=1) / pop_rate_mean)
                else:
                    out["CV_R_%dms" % int(h_ms)] = 0.0

                # noise correlation (mean of off-diagonals of correlation matrix)
                X = counts_corr - counts_corr.mean(axis=1, keepdims=True)
                X /= (counts_corr.std(axis=1, keepdims=True) + 1e-9)
                C = (X @ X.T) / X.shape[1]
                iu = np.triu_indices(N, k=1)
                corr_vals = C[iu]
                valid_corr = np.isfinite(corr_vals)
                out["mean_noise_corr_%dms" % int(h_ms)] = float(np.mean(corr_vals[valid_corr])) if np.any(valid_corr) else 0.0
            else:
                out["CV_R_%dms" % int(h_ms)] = 0.0
                out["mean_noise_corr_%dms" % int(h_ms)] = 0.0

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
        if inhib_mask is not None:
            total_active_mask = S.sum(axis=1) > 0
            out["participation_frac_total_E"] = (
                float(total_active_mask[exc_mask].sum() / float(np.sum(exc_mask)))
                if np.any(exc_mask) else np.nan
            )
            out["participation_frac_total_I"] = (
                float(total_active_mask[inhib_mask].sum() / float(np.sum(inhib_mask)))
                if np.any(inhib_mask) else np.nan
            )

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

        band_mask = (f >= 2.0) & (f <= 120.0)
        if np.any(band_mask):
            band = np.asarray(Pxx[band_mask], dtype=float)
            band_freqs = np.asarray(f[band_mask], dtype=float)
            band_median = float(np.median(band))
            peak_idx = int(np.argmax(band))
            out["psd_peak_freq_hz"] = float(band_freqs[peak_idx])
            out["psd_peak_amplitude"] = float(band[peak_idx])
            out["psd_peak_ratio"] = float(np.max(band) / (band_median + 1e-12)) if band_median > 0.0 else 0.0
        else:
            out["psd_peak_freq_hz"] = 0.0
            out["psd_peak_amplitude"] = 0.0
            out["psd_peak_ratio"] = 0.0
        out["pop_spec_entropy"] = spectral_entropy(Pxx)
        n_entropy_bins = int(np.sum(np.asarray(Pxx, dtype=float) > 0.0))
        if n_entropy_bins > 1:
            out["pop_spec_entropy_norm"] = float(out["pop_spec_entropy"] / np.log(n_entropy_bins))
        else:
            out["pop_spec_entropy_norm"] = 0.0
        out["pop_psd_freq_hz"] = f
        out["pop_psd"] = Pxx

        # (Optional) add more: branching ratio, participation ratio, etc.
        return out

class DebugLogger:
    def __init__(self):
        self.s_ampa = []
        self.s_nmda = []
        self.s_gaba_a = []
        self.s_gaba_b = []

class Simulation:
    _PLASTICITY_REGISTRY = {
        "stdp": (STDP, "pre_post"),
        "stpd": (STDP, "pre_post"),
        "stdp_masked": (STDPMasked, "pre_post"),
        "stdp_sparse": (STDPMasked, "pre_post"),
        "masked_stdp": (STDPMasked, "pre_post"),
        "sparse_stdp": (STDPMasked, "pre_post"),
        "da_bcm": (DA_BCM, "pre_post"),
        "da-bcm": (DA_BCM, "pre_post"),
        "dabcm": (DA_BCM, "pre_post"),
        "bcm_da": (DA_BCM, "pre_post"),
        "bcm-da": (DA_BCM, "pre_post"),
        "clopath": (ClopathMasked, "pre_post_v"),
        "clopath_masked": (ClopathMasked, "pre_post_v"),
        "clopath_sparse": (ClopathMasked, "pre_post_v"),
        "masked_clopath": (ClopathMasked, "pre_post_v"),
        "sparse_clopath": (ClopathMasked, "pre_post_v"),
        "t_stdp": (T_STDP, "pre_post"),
        "t-stdp": (T_STDP, "pre_post"),
        "tstdp": (T_STDP, "pre_post"),
        "predictive": (PredictiveCoding, "pre_post"),
        "predictivecoding": (PredictiveCoding, "pre_post"),
        "saponati": (PredictiveCodingSaponati, "pre_post_v"),
        "predictivecodingsaponati": (PredictiveCodingSaponati, "pre_post_v"),
    }

    @staticmethod
    def _normalize_plasticity_reward_type(plasticity_reward_type):
        key = str(plasticity_reward_type).strip().lower()
        if key in ("online", "dense", "step"):
            return "online"
        if key in ("sparse", "episodic", "delayed"):
            return "sparse"
        raise ValueError(
            f"Unknown plasticity_reward_type '{plasticity_reward_type}'. "
            "Expected 'online' or 'sparse'."
        )

    @classmethod
    def _resolve_plasticity_factory(cls, plasticity_key):
        if plasticity_key not in cls._PLASTICITY_REGISTRY:
            raise ValueError(f"Unknown plasticity '{plasticity_key}'.")
        return cls._PLASTICITY_REGISTRY[plasticity_key]

    @staticmethod
    def _infer_base_plasticity_step(plasticity_obj):
        step_fn = getattr(plasticity_obj, "step", None)
        if not callable(step_fn):
            raise ValueError("Plasticity object must implement a callable step(...) method.")

        sig = inspect.signature(step_fn)
        param_names = [
            p.name
            for p in sig.parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if param_names and param_names[0] == "self":
            param_names = param_names[1:]
        names = set(param_names)

        if {"pre_spikes", "post_spikes"}.issubset(names):
            if "Vs" in names or "V" in names:
                return "pre_post_v"
            return "pre_post"
        if {"post_spikes", "I_syn"}.issubset(names):
            return "post_isyn"
        raise ValueError(
            "Could not infer plasticity step mode from step(...) signature. "
            "Expected (pre_spikes, post_spikes[, Vs], reward=...) or (post_spikes, I_syn, reward=...)."
        )

    @staticmethod
    def _resolve_runtime_plasticity_step(base_step, plasticity_reward_type):
        if plasticity_reward_type == "online":
            return base_step
        if plasticity_reward_type != "sparse":
            raise ValueError(f"Unknown plasticity_reward_type '{plasticity_reward_type}'.")
        if base_step == "pre_post":
            return "sparse_pre_post"
        if base_step == "pre_post_v":
            return "sparse_pre_post_v"
        raise ValueError(
            f"Sparse reward mode is not supported for plasticity step '{base_step}'."
        )

    def __init__(self, connectome: Connectome, dt, stepper_type="adapt", state0=None,
                 enable_plasticity=True, plasticity="stdp", plasticity_reward_type="online", plasticity_kwargs=None,
                 synapse_type="standard", synapse_kwargs=None, enable_state_logger=True, enable_debug_logger=False,
                 rate_normalization=None):
        """
        Simulation class to represent the simulation of a neuron population.
        """
        self.dt = dt
        self.connectome = connectome
        self.stepper_type = stepper_type
        self.synapse_type = synapse_type
        self.synapse_kwargs = dict(synapse_kwargs or {})
        self.plasticity_key = plasticity
        self.plasticity_kwargs = dict(plasticity_kwargs or {})
        self.rate_normalization_config = rate_normalization
        self.axonal_dynamics = AxonalDynamics(connectome, self.dt)
        synapse_kwargs = self.synapse_kwargs
        if synapse_type == "rise":
            self.synapse_dynamics = SynapseDynamics_Rise(connectome, self.dt, **synapse_kwargs)
        elif synapse_type == "standard":
            self.synapse_dynamics = SynapseDynamics(connectome, self.dt, **synapse_kwargs)
        elif synapse_type == "uncapped":
            self.synapse_dynamics = SynapseDynamics_Uncapped(connectome, self.dt, **synapse_kwargs)
        else:
            raise ValueError(f"Unknown synapse_type '{synapse_type}'.")
        self.neuron_states = NeuronState(connectome.neuron_population.neuron_population.T, stepper_type=stepper_type, state0=state0)
        Cs = connectome.neuron_population.neuron_population[:, 4]
        self.integrator = InputIntegration(self.synapse_dynamics, Cs)
        plasticity_kwargs = plasticity_kwargs or {}
        self.plasticity_reward_type = self._normalize_plasticity_reward_type(plasticity_reward_type)
        self.plasticity = None
        self.plasticity_step = None
        if enable_plasticity:
            if isinstance(plasticity, str):
                key = plasticity.lower()
                factory, base_step = self._resolve_plasticity_factory(key)
                self.plasticity = factory(connectome, self.dt, **plasticity_kwargs)
            elif callable(plasticity):
                self.plasticity = plasticity(connectome, self.dt, **plasticity_kwargs)
                base_step = self._infer_base_plasticity_step(self.plasticity)
            else:
                raise ValueError("plasticity must be a string key or a callable factory.")
            self.plasticity_step = self._resolve_runtime_plasticity_step(
                base_step,
                self.plasticity_reward_type,
            )

        self.enable_state_logger = enable_state_logger
        self.stats = SimulationStats()
        self.stats.inhibitory_mask = connectome.neuron_population.inhibitory_mask.copy()
        if self.enable_state_logger:
            self.stats.Vs.append(self.neuron_states.V.copy())
            self.stats.us.append(self.neuron_states.u.copy())
            self.stats.spikes.append(self.neuron_states.spike.copy())


        self.enable_debug_logger = enable_debug_logger
        if self.enable_debug_logger:
            self.debug_logger = DebugLogger()
            self.debug_logger.s_ampa.append(self.synapse_dynamics.g_AMPA.copy())
            self.debug_logger.s_nmda.append(self.synapse_dynamics.g_NMDA.copy())
            self.debug_logger.s_gaba_a.append(self.synapse_dynamics.g_GABA_A.copy())
            self.debug_logger.s_gaba_b.append(self.synapse_dynamics.g_GABA_B.copy())

        self.t_now = 0.0
        if self.enable_state_logger:
            self.stats.ts.append(self.t_now)
        self.output_readout = None
        self.output_vector = None
        self.rate_normalizer = None
        if rate_normalization is not None:
            if not isinstance(rate_normalization, Mapping):
                raise ValueError("rate_normalization must be a mapping or None.")
            self.rate_normalizer = build_firing_rate_normalizer(
                self.connectome,
                dt_ms=self.dt,
                config=rate_normalization,
            )


    def step(self, I_ext=None, spike_ext=None, reward=1.0):
        """
        Step the simulation forward in time.
        """
        # Get the synaptic input from the synapse dynamics (calls synapse_dynamics)
        I_syn = self.integrator(self.neuron_states.V, I_ext=I_ext)
        # Update the neuron states
        self.neuron_states.step(I_syn, self.dt)
        post_spikes = self.neuron_states.spike # shape n_neurons x 1
        # Update the axonal dynamics
        pre_spike_rows, pre_spike_cols = self.axonal_dynamics.check_sparse(self.t_now + self.dt)
        pre_spikes = None
        if self.plasticity is not None and self.plasticity_step:
            if not hasattr(self.plasticity, "step_sparse"):
                pre_spikes = self.axonal_dynamics.materialize_sparse(pre_spike_rows, pre_spike_cols)
                self.pre_spikes = pre_spikes.copy()  # Store only when plasticity uses it
            else:
                self.pre_spikes = None
        else:
            self.pre_spikes = None
        # Push the spikes to the axonal dynamics, do it after the pre_spikes are checked,
        # as the spikes comes from the end of the current step
        self.axonal_dynamics.push_many(post_spikes, self.t_now + self.dt)
        # Time step for synapse dynamics (only decay)
        self.synapse_dynamics.decay()
        # Update the synapse weights based on the traces from last step
        if self.plasticity is not None and self.plasticity_step:
            if self.plasticity_step == "pre_post":
                if hasattr(self.plasticity, "step_sparse"):
                    self.plasticity.step_sparse(pre_spike_rows, pre_spike_cols, post_spikes, reward=reward)
                else:
                    self.plasticity.step(pre_spikes, post_spikes, reward=reward)
            elif self.plasticity_step == "sparse_pre_post":
                if hasattr(self.plasticity, "step_no_weight_changes_sparse"):
                    self.plasticity.step_no_weight_changes_sparse(pre_spike_rows, pre_spike_cols, post_spikes, reward=reward)
                elif hasattr(self.plasticity, "step_no_weight_changes"):
                    self.plasticity.step_no_weight_changes(pre_spikes, post_spikes, reward=reward)
                else:
                    self.plasticity.decay_traces()
                    self.plasticity.spikes_in(pre_spikes, post_spikes)
            elif self.plasticity_step == "pre_post_v":
                if hasattr(self.plasticity, "step_sparse"):
                    self.plasticity.step_sparse(pre_spike_rows, pre_spike_cols, post_spikes, self.neuron_states.V, reward=reward)
                else:
                    self.plasticity.step(pre_spikes, post_spikes, self.neuron_states.V, reward=reward)
            elif self.plasticity_step == "sparse_pre_post_v":
                if hasattr(self.plasticity, "step_no_weight_changes_sparse"):
                    self.plasticity.step_no_weight_changes_sparse(pre_spike_rows, pre_spike_cols, post_spikes, self.neuron_states.V, reward=reward)
                else:
                    self.plasticity.step_no_weight_changes(pre_spikes, post_spikes, self.neuron_states.V, reward=reward)
            elif self.plasticity_step == "post_isyn":
                self.plasticity.step(post_spikes, I_syn, reward=reward)
            elif callable(self.plasticity_step):
                self.plasticity_step(self.plasticity, pre_spikes, post_spikes, I_syn, self.neuron_states.V, self)
            else:
                raise ValueError(f"Unknown plasticity_step '{self.plasticity_step}'.")
        # Update synapse reaction class from the pre_spikes
        self.synapse_dynamics.spike_input_sparse(pre_spike_rows, pre_spike_cols)
        if spike_ext is not None:
            self.synapse_dynamics.sensory_spike_input(spike_ext)
        # Update the current time
        self.t_now += self.dt
        # Store the current state
        if self.enable_state_logger:
            self.stats.Vs.append(self.neuron_states.V.copy())
            self.stats.us.append(self.neuron_states.u.copy())
            self.stats.spikes.append(self.neuron_states.spike.copy())
            self.stats.ts.append(self.t_now)
        if self.rate_normalizer is not None:
            self.rate_normalizer.update(post_spikes)
        self._update_output_readout(post_spikes)

        if self.enable_debug_logger:
            self.debug_logger.s_ampa.append(self.synapse_dynamics.g_AMPA.copy() * self.synapse_dynamics.g_AMPA_max)
            self.debug_logger.s_nmda.append(self.synapse_dynamics.g_NMDA.copy() * self.synapse_dynamics.g_NMDA_max)
            self.debug_logger.s_gaba_a.append(self.synapse_dynamics.g_GABA_A.copy() * self.synapse_dynamics.g_GABA_A_max)
            self.debug_logger.s_gaba_b.append(self.synapse_dynamics.g_GABA_B.copy() * self.synapse_dynamics.g_GABA_B_max)

    def _resolve_neuron_subset(
        self,
        *,
        neurons=None,
        fraction=None,
        pool="all",
        rng=None,
        name="neuron subset",
    ):
        n_neurons = int(self.connectome.neuron_population.n_neurons)
        if rng is None:
            rng = np.random.default_rng()

        if neurons is not None:
            idx = np.asarray(neurons, dtype=int).reshape(-1)
            if idx.size == 0:
                raise ValueError(f"{name} cannot be empty.")
            if np.any(idx < 0) or np.any(idx >= n_neurons):
                raise ValueError(f"{name} contains out-of-range neuron indices.")
            return np.unique(idx)

        pool_key = "all" if pool is None else str(pool)
        pop = self.connectome.neuron_population
        inhib_mask = np.asarray(pop.inhibitory_mask, dtype=bool)
        if pool_key.lower() in ("all", "*"):
            candidates = np.arange(n_neurons, dtype=int)
        elif pool_key.lower() in ("e", "exc", "excit", "excitatory"):
            candidates = np.flatnonzero(~inhib_mask)
        elif pool_key.lower() in ("i", "inh", "inhib", "inhibitory"):
            candidates = np.flatnonzero(inhib_mask)
        else:
            try:
                candidates = np.asarray(pop.get_neurons_from_type(pool_key), dtype=int).reshape(-1)
            except Exception as exc:
                raise ValueError(
                    f"Unknown {name} pool '{pool_key}'. Use 'all', 'E', 'I', or a neuron type name."
                ) from exc

        if candidates.size == 0:
            raise ValueError(f"{name} pool '{pool_key}' contains no neurons.")

        if fraction is None:
            return candidates.copy()

        frac = float(fraction)
        if frac <= 0:
            raise ValueError(f"{name} fraction must be > 0.")
        if frac > 1:
            raise ValueError(f"{name} fraction must be <= 1.")
        n_select = int(round(frac * candidates.size))
        n_select = max(1, min(n_select, candidates.size))
        return np.asarray(rng.choice(candidates, size=n_select, replace=False), dtype=int)

    @staticmethod
    def _random_state0(n_neurons, rng, *, v_range=(-100.0, -70.0), u_range=(0.0, 400.0)):
        Vs = rng.uniform(float(v_range[0]), float(v_range[1]), size=int(n_neurons))
        us = rng.uniform(float(u_range[0]), float(u_range[1]), size=int(n_neurons))
        spikes = np.zeros(int(n_neurons), dtype=bool)
        Ts = np.zeros(int(n_neurons), dtype=float)
        return Vs, us, spikes, Ts

    @staticmethod
    def _rate_hz_for_indices(spikes, times_ms, indices, t_start_ms, t_stop_ms):
        idx = np.asarray(indices, dtype=int).reshape(-1)
        if idx.size == 0:
            return 0.0
        S = np.asarray(spikes, dtype=bool)
        times = np.asarray(times_ms, dtype=float)
        mask = (times >= float(t_start_ms)) & (times < float(t_stop_ms))
        if S.size == 0 or not np.any(mask):
            return 0.0
        duration_s = max(1e-12, (float(t_stop_ms) - float(t_start_ms)) / 1000.0)
        return float(np.sum(S[idx][:, mask]) / idx.size / duration_s)

    def _candidate_indices_for_pool(self, pool="E"):
        n_neurons = int(self.connectome.neuron_population.n_neurons)
        pool_key = "all" if pool is None else str(pool)
        pop = self.connectome.neuron_population
        inhib_mask = np.asarray(pop.inhibitory_mask, dtype=bool)
        if pool_key.lower() in ("all", "*"):
            return np.arange(n_neurons, dtype=int)
        if pool_key.lower() in ("e", "exc", "excit", "excitatory"):
            return np.flatnonzero(~inhib_mask)
        if pool_key.lower() in ("i", "inh", "inhib", "inhibitory"):
            return np.flatnonzero(inhib_mask)
        return np.asarray(pop.get_neurons_from_type(pool_key), dtype=int).reshape(-1)

    def _select_spatial_source_target(
        self,
        *,
        fraction,
        pool="E",
        axis=0,
        pos_attr="pos",
    ):
        candidates = self._candidate_indices_for_pool(pool)
        G = getattr(self.connectome, "G", None)
        if G is None:
            raise ValueError("Spatial source/target selection requires connectome.G.")
        coords = []
        keep = []
        for node in candidates:
            if node not in G.nodes or pos_attr not in G.nodes[node]:
                continue
            p = np.asarray(G.nodes[node][pos_attr], dtype=float).reshape(-1)
            if p.size <= int(axis):
                continue
            coords.append(float(p[int(axis)]))
            keep.append(int(node))
        if not keep:
            raise ValueError(f"No candidate neurons contain spatial node attribute '{pos_attr}'.")
        keep = np.asarray(keep, dtype=int)
        coords = np.asarray(coords, dtype=float)
        n_select = max(1, min(int(round(float(fraction) * keep.size)), keep.size))
        order = np.argsort(coords)
        source = keep[order[:n_select]]
        target = keep[order[-n_select:]]
        target = np.setdiff1d(target, source, assume_unique=False)
        if target.size == 0:
            raise ValueError("Spatial source and target sets overlap completely; use a smaller fraction.")
        return source, target

    def _select_graph_distant_target(
        self,
        *,
        input_indices,
        fraction,
        pool="E",
    ):
        input_indices = np.asarray(input_indices, dtype=int).reshape(-1)
        candidates = self._candidate_indices_for_pool(pool)
        candidates = np.setdiff1d(candidates, input_indices, assume_unique=False)
        if candidates.size == 0:
            raise ValueError("No non-input target candidates are available.")
        n_select = max(1, min(int(round(float(fraction) * candidates.size)), candidates.size))
        G = getattr(self.connectome, "G", None)
        if G is None:
            self.connectome.build_nx()
            G = self.connectome.G
        Gu = G.to_undirected()
        distances = {}
        for source in input_indices:
            for node, dist in nx.single_source_shortest_path_length(Gu, int(source)).items():
                node = int(node)
                if node not in distances or int(dist) < distances[node]:
                    distances[node] = int(dist)
        finite = [int(x) for x in candidates if int(x) in distances]
        missing = [int(x) for x in candidates if int(x) not in distances]
        finite_sorted = sorted(finite, key=lambda node: distances[node], reverse=True)
        ordered = finite_sorted + missing
        return np.asarray(ordered[:n_select], dtype=int)

    @staticmethod
    def _classify_brunel_from_metrics(
        metrics,
        *,
        cv_threshold=0.99,
        fano_threshold=0.99,
        corr_threshold=0.05,
        peak_ratio_threshold=250.0,
        oscillatory_entropy_norm_threshold=0.85,
        bursty_fano_threshold=3.0,
    ):
        irregular = (
            float(metrics.get("Fano_median_300ms", 0.0)) > float(fano_threshold)
            and float(metrics.get("ISI_CV_mean_E", metrics.get("ISI_CV_mean", 0.0))) > float(cv_threshold)
        )
        oscillatory = (
            float(metrics.get("psd_peak_ratio", 0.0)) > float(peak_ratio_threshold)
            and float(metrics.get("pop_spec_entropy_norm", np.inf)) < float(oscillatory_entropy_norm_threshold)
        )
        oscillatory = oscillatory or float(metrics.get("psd_peak_ratio", 0.0)) > float(peak_ratio_threshold + 100.0)
        oscillatory = oscillatory or float(metrics.get("pop_spec_entropy_norm", np.inf)) < float(oscillatory_entropy_norm_threshold - 0.05)
        bursty_individual = float(metrics.get("Fano_median_300ms", 0.0)) > float(bursty_fano_threshold)
        synchronous = (
            float(metrics.get("mean_noise_corr_50ms", 0.0)) > float(corr_threshold)
            or oscillatory
        )
        if not synchronous and irregular:
            label = "AI"
        elif synchronous and irregular:
            label = "SI"
        elif not synchronous and not irregular:
            label = "AR"
        else:
            label = "SR"
        return {
            "brunel_class": label,
            "brunel_irregular": bool(irregular),
            "brunel_synchronous": bool(synchronous),
            "oscillatory": bool(oscillatory),
            "bursty_individual": bool(bursty_individual),
        }

    @staticmethod
    def _detect_global_bursts(spikes, times_ms, *, t_start_ms, t_stop_ms, bin_ms=10.0, rate_threshold_hz=10.0, participation_threshold=0.3):
        S = np.asarray(spikes, dtype=bool)
        times = np.asarray(times_ms, dtype=float)
        if S.size == 0:
            return {
                "n_global_bursts": 0,
                "max_burst_participation": 0.0,
                "mean_burst_participation": 0.0,
                "burst_times_ms": np.zeros(0, dtype=float),
            }
        n_neurons = S.shape[0]
        starts = np.arange(float(t_start_ms), float(t_stop_ms), float(bin_ms))
        burst_times = []
        participations = []
        for start in starts:
            stop = min(float(t_stop_ms), start + float(bin_ms))
            mask = (times >= start) & (times < stop)
            if not np.any(mask):
                continue
            S_bin = S[:, mask]
            active = np.any(S_bin, axis=1)
            participation = float(np.sum(active) / max(1, n_neurons))
            rate = float(np.sum(S_bin) / max(1, n_neurons) / max(1e-12, (stop - start) / 1000.0))
            if rate >= float(rate_threshold_hz) and participation >= float(participation_threshold):
                burst_times.append(float(0.5 * (start + stop)))
                participations.append(participation)
        participations_arr = np.asarray(participations, dtype=float)
        return {
            "n_global_bursts": int(participations_arr.size),
            "max_burst_participation": float(np.max(participations_arr)) if participations_arr.size else 0.0,
            "mean_burst_participation": float(np.mean(participations_arr)) if participations_arr.size else 0.0,
            "burst_times_ms": np.asarray(burst_times, dtype=float),
        }

    @staticmethod
    def _scalar_metric_items(metrics):
        out = {}
        for key, value in dict(metrics).items():
            if isinstance(value, (int, float, bool, np.integer, np.floating, np.bool_)):
                out[key] = value.item() if isinstance(value, np.generic) else value
        return out

    @staticmethod
    def _threshold_first_true(rows, condition_key):
        vals = [float(row["input_amplitude"]) for row in rows if bool(row.get(condition_key, False))]
        return float(vals[0]) if vals else None

    @staticmethod
    def _threshold_min_true(rows, condition_key):
        vals = [float(row["input_amplitude"]) for row in rows if bool(row.get(condition_key, False))]
        return float(min(vals)) if vals else None

    @staticmethod
    def _hysteresis_curve_summary(rows, *, metric_key="rate_mean_Hz"):
        up = {float(row["input_amplitude"]): row for row in rows if row["sweep_direction"] == "up"}
        down = {float(row["input_amplitude"]): row for row in rows if row["sweep_direction"] == "down"}
        amps = sorted(set(up).intersection(down))
        curve = []
        for amp in amps:
            up_val = float(up[amp].get(metric_key, np.nan))
            down_val = float(down[amp].get(metric_key, np.nan))
            active_up = bool(up[amp].get("is_active", False))
            active_down = bool(down[amp].get("is_active", False))
            strong_up = bool(up[amp].get("is_strongly_oscillatory", False))
            strong_down = bool(down[amp].get("is_strongly_oscillatory", False))
            curve.append(
                {
                    "input_amplitude": float(amp),
                    f"up_{metric_key}": up_val,
                    f"down_{metric_key}": down_val,
                    f"delta_{metric_key}": float(down_val - up_val) if np.isfinite(up_val) and np.isfinite(down_val) else np.nan,
                    "active_up": active_up,
                    "active_down": active_down,
                    "active_hysteretic": bool(active_up != active_down),
                    "strong_oscillatory_up": strong_up,
                    "strong_oscillatory_down": strong_down,
                    "strong_oscillatory_hysteretic": bool(strong_up != strong_down),
                }
            )
        if len(amps) >= 2:
            diffs = np.array([abs(row[f"delta_{metric_key}"]) for row in curve], dtype=float)
            if np.all(np.isfinite(diffs)):
                area = float(np.trapz(diffs, np.asarray(amps, dtype=float)))
            else:
                area = np.nan
        else:
            area = 0.0
        return curve, area

    def analyze_separation_property(
        self,
        *,
        n_trials=50,
        input_neurons=None,
        input_fraction=0.2,
        input_pool="E",
        state_neurons=None,
        state_fraction=None,
        state_pool="all",
        input_active_probability=1.0,
        input_rate_hz=50.0,
        input_amplitude=1.0,
        warmup_ms=500.0,
        measurement_ms=1000.0,
        tau_ms=30.0,
        bin_ms=20.0,
        seed=None,
        randomize_initial_state=True,
        state_v_range=(-100.0, -70.0),
        state_u_range=(0.0, 400.0),
        enable_plasticity=False,
        center=True,
        normalize=True,
        store_states=True,
        ensure_nonempty_input=True,
        show_progress=True,
        disjoint_state_pool=False,
    ):
        """
        Estimate input separation using exponentially filtered spike states.

        Each trial samples an active subset from the input population, injects
        Poisson spike input during warmup and measurement, and records filtered
        spike traces from the requested state population every `bin_ms`.

        Returns and stores on `self.stats.separation_property` a result dict
        containing the state tensor, input masks, sampled times, and effective
        rank metrics. The state tensor has shape
        `(n_trials, n_time_bins, n_state_neurons)`.
        """
        if int(n_trials) <= 0:
            raise ValueError("n_trials must be > 0.")
        if warmup_ms < 0 or measurement_ms <= 0:
            raise ValueError("warmup_ms must be >= 0 and measurement_ms must be > 0.")
        if tau_ms <= 0 or bin_ms <= 0:
            raise ValueError("tau_ms and bin_ms must be > 0.")
        if input_rate_hz < 0 or input_amplitude < 0:
            raise ValueError("input_rate_hz and input_amplitude must be >= 0.")
        active_prob = float(input_active_probability)
        if active_prob < 0.0 or active_prob > 1.0:
            raise ValueError("input_active_probability must be in [0, 1].")

        rng = np.random.default_rng(seed)
        n_neurons = int(self.connectome.neuron_population.n_neurons)
        input_indices = self._resolve_neuron_subset(
            neurons=input_neurons,
            fraction=input_fraction,
            pool=input_pool,
            rng=rng,
            name="input neuron subset",
        )
        state_indices = self._resolve_neuron_subset(
            neurons=state_neurons,
            fraction=state_fraction,
            pool=state_pool,
            rng=rng,
            name="state neuron subset",
        )
        if disjoint_state_pool:
            state_indices = np.setdiff1d(state_indices, input_indices, assume_unique=False)
            if state_indices.size == 0:
                raise ValueError(
                    "disjoint_state_pool=True removed all state neurons. "
                    "Choose a smaller input pool, a broader state pool, or pass explicit disjoint state_neurons."
                )

        warmup_steps = int(round(float(warmup_ms) / float(self.dt)))
        measurement_steps = int(round(float(measurement_ms) / float(self.dt)))
        bin_steps = max(1, int(round(float(bin_ms) / float(self.dt))))
        if measurement_steps <= 0:
            raise ValueError("measurement_ms is too short for the current dt.")

        sample_step_numbers = np.arange(bin_steps, measurement_steps + 1, bin_steps, dtype=int)
        if sample_step_numbers.size == 0 or sample_step_numbers[-1] != measurement_steps:
            sample_step_numbers = np.unique(np.append(sample_step_numbers, measurement_steps)).astype(int)
        sample_step_set = set(int(x) for x in sample_step_numbers.tolist())
        sample_times_ms = float(warmup_ms) + sample_step_numbers.astype(float) * float(self.dt)

        states = np.zeros((int(n_trials), sample_step_numbers.size, state_indices.size), dtype=float)
        input_active_masks = np.zeros((int(n_trials), n_neurons), dtype=bool)
        active_input_counts = np.zeros(int(n_trials), dtype=int)
        trial_mean_rates = np.zeros(int(n_trials), dtype=float)
        decay = float(np.exp(-float(self.dt) / float(tau_ms)))

        base_state0 = None
        if not randomize_initial_state:
            base_state0 = (
                self.neuron_states.V.copy(),
                self.neuron_states.u.copy(),
                self.neuron_states.spike.copy(),
                self.neuron_states.T.copy(),
            )

        trial_iter = range(int(n_trials))
        progress_bar = None
        if show_progress:
            try:
                from tqdm.auto import tqdm
                progress_bar = tqdm(trial_iter, desc="Separation trials", unit="trial")
                trial_iter = progress_bar
            except Exception:
                progress_bar = None

        for trial_idx in trial_iter:
            if randomize_initial_state:
                state0 = self._random_state0(
                    n_neurons,
                    rng,
                    v_range=state_v_range,
                    u_range=state_u_range,
                )
            else:
                state0 = tuple(x.copy() for x in base_state0)

            active_mask = np.zeros(n_neurons, dtype=bool)
            if input_indices.size > 0 and active_prob > 0.0:
                selected = rng.random(input_indices.size) < active_prob
                if ensure_nonempty_input and not np.any(selected):
                    selected[int(rng.integers(0, input_indices.size))] = True
                active_mask[input_indices[selected]] = True
            input_active_masks[trial_idx] = active_mask
            active_input_counts[trial_idx] = int(np.sum(active_mask))

            rates = np.zeros(n_neurons, dtype=float)
            amps = np.zeros(n_neurons, dtype=float)
            rates[active_mask] = float(input_rate_hz)
            amps[active_mask] = float(input_amplitude)
            poisson = PoissonInput(n_neurons, rate=rates, amplitude=amps, rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))))

            trial_sim = Simulation(
                self.connectome,
                self.dt,
                stepper_type=self.stepper_type,
                state0=state0,
                enable_plasticity=enable_plasticity,
                plasticity=self.plasticity_key,
                plasticity_reward_type=self.plasticity_reward_type,
                plasticity_kwargs=self.plasticity_kwargs,
                synapse_type=self.synapse_type,
                synapse_kwargs=self.synapse_kwargs,
                enable_state_logger=False,
                enable_debug_logger=False,
                rate_normalization=None,
            )

            filtered = np.zeros(n_neurons, dtype=float)
            measurement_spike_count = 0.0
            for _ in range(warmup_steps):
                trial_sim.step(spike_ext=poisson(self.dt))
                filtered *= decay
                filtered += trial_sim.neuron_states.spike.astype(float)

            sample_idx = 0
            for measurement_step in range(1, measurement_steps + 1):
                trial_sim.step(spike_ext=poisson(self.dt))
                spikes_now = trial_sim.neuron_states.spike.astype(float)
                filtered *= decay
                filtered += spikes_now
                measurement_spike_count += float(np.sum(spikes_now[state_indices]))
                if measurement_step in sample_step_set:
                    states[trial_idx, sample_idx, :] = filtered[state_indices]
                    sample_idx += 1

            trial_mean_rates[trial_idx] = float(
                measurement_spike_count / max(1, state_indices.size) / (measurement_steps * self.dt / 1000.0)
            )

        metrics = SimulationStats.separation_effective_rank_metrics(
            states,
            center=center,
            normalize=normalize,
        )
        pop = self.connectome.neuron_population
        neuron_type_by_index = np.asarray(pop.neuron_population_types)
        inhibitory_mask = np.asarray(pop.inhibitory_mask, dtype=bool)
        result = {
            "metrics": metrics,
            "states": states if store_states else None,
            "state_times_ms": sample_times_ms,
            "input_indices": input_indices,
            "state_indices": state_indices,
            "input_neuron_types": neuron_type_by_index[input_indices].copy(),
            "state_neuron_types": neuron_type_by_index[state_indices].copy(),
            "input_inhibitory_mask": inhibitory_mask[input_indices].copy(),
            "state_inhibitory_mask": inhibitory_mask[state_indices].copy(),
            "input_active_masks": input_active_masks,
            "active_input_counts": active_input_counts,
            "trial_mean_rates_Hz": trial_mean_rates,
            "config": {
                "n_trials": int(n_trials),
                "input_fraction": None if input_fraction is None else float(input_fraction),
                "input_pool": input_pool,
                "state_fraction": None if state_fraction is None else float(state_fraction),
                "state_pool": state_pool,
                "input_active_probability": active_prob,
                "input_rate_hz": float(input_rate_hz),
                "input_amplitude": float(input_amplitude),
                "warmup_ms": float(warmup_ms),
                "measurement_ms": float(measurement_ms),
                "tau_ms": float(tau_ms),
                "bin_ms": float(bin_ms),
                "dt_ms": float(self.dt),
                "center": bool(center),
                "normalize": bool(normalize),
                "randomize_initial_state": bool(randomize_initial_state),
                "show_progress": bool(show_progress),
                "disjoint_state_pool": bool(disjoint_state_pool),
            },
        }
        return self.stats.save_separation_property(result)

    def analyze_generalization_property(
        self,
        *,
        n_classes=5,
        trials_per_class=10,
        input_rate_vectors=None,
        input_neurons=None,
        input_fraction=0.3,
        input_pool="E",
        state_neurons=None,
        state_fraction=None,
        state_pool="E",
        disjoint_state_pool=True,
        input_rate_min_hz=5.0,
        input_rate_max_hz=50.0,
        input_rate_noise_std_hz=None,
        input_rate_noise_fraction=0.1,
        input_rate_clip=(0.0, None),
        input_amplitude=0.5,
        warmup_ms=250.0,
        measurement_ms=500.0,
        tau_ms=30.0,
        bin_ms=20.0,
        state_summary="mean",
        seed=None,
        randomize_initial_state=True,
        state_v_range=(-100.0, -70.0),
        state_u_range=(0.0, 400.0),
        enable_plasticity=False,
        train_fraction=0.7,
        ridge_alpha=1.0,
        lda_regularization=1e-6,
        standardize=True,
        store_states=True,
        store_rate_vectors=True,
        show_progress=True,
    ):
        """
        Estimate input-class generalization from noisy Poisson-rate patterns.

        Each input class is a base vector of Poisson rates over a fixed input
        neuron subset. Each trial adds Gaussian noise to that class vector,
        runs the network, records exponentially filtered spike states, and
        summarizes one state vector per trial. Metrics include Fisher/LDA-style
        within-class and between-class scatter plus a linear ridge readout
        trained to classify input class from reservoir state.
        """
        if int(n_classes) <= 1:
            raise ValueError("n_classes must be > 1.")
        if int(trials_per_class) <= 1:
            raise ValueError("trials_per_class must be > 1.")
        if warmup_ms < 0 or measurement_ms <= 0:
            raise ValueError("warmup_ms must be >= 0 and measurement_ms must be > 0.")
        if tau_ms <= 0 or bin_ms <= 0:
            raise ValueError("tau_ms and bin_ms must be > 0.")
        if input_rate_min_hz < 0 or input_rate_max_hz < 0:
            raise ValueError("input rates must be >= 0.")
        if input_rate_min_hz > input_rate_max_hz:
            raise ValueError("input_rate_min_hz cannot exceed input_rate_max_hz.")
        if input_amplitude < 0:
            raise ValueError("input_amplitude must be >= 0.")
        if train_fraction <= 0 or train_fraction >= 1:
            raise ValueError("train_fraction must be in (0, 1).")

        summary_key = str(state_summary).strip().lower()
        if summary_key not in ("mean", "final", "max", "mean_final"):
            raise ValueError("state_summary must be 'mean', 'final', 'max', or 'mean_final'.")

        rng = np.random.default_rng(seed)
        n_neurons = int(self.connectome.neuron_population.n_neurons)
        input_indices = self._resolve_neuron_subset(
            neurons=input_neurons,
            fraction=input_fraction,
            pool=input_pool,
            rng=rng,
            name="input neuron subset",
        )
        state_indices = self._resolve_neuron_subset(
            neurons=state_neurons,
            fraction=state_fraction,
            pool=state_pool,
            rng=rng,
            name="state neuron subset",
        )
        if disjoint_state_pool:
            state_indices = np.setdiff1d(state_indices, input_indices, assume_unique=False)
            if state_indices.size == 0:
                raise ValueError(
                    "disjoint_state_pool=True removed all state neurons. "
                    "Choose a smaller input pool, a broader state pool, or pass explicit disjoint state_neurons."
                )

        if input_rate_vectors is None:
            base_rate_vectors = rng.uniform(
                float(input_rate_min_hz),
                float(input_rate_max_hz),
                size=(int(n_classes), input_indices.size),
            )
        else:
            base_rate_vectors = np.asarray(input_rate_vectors, dtype=float)
            if base_rate_vectors.ndim != 2:
                raise ValueError("input_rate_vectors must have shape (n_classes, n_input_neurons).")
            if base_rate_vectors.shape[0] != int(n_classes):
                raise ValueError("input_rate_vectors first dimension must match n_classes.")
            if base_rate_vectors.shape[1] != input_indices.size:
                raise ValueError(
                    "input_rate_vectors second dimension must match the resolved input neuron subset size. "
                    "Pass explicit input_neurons if you need a fixed ordering."
                )
            if np.any(base_rate_vectors < 0):
                raise ValueError("input_rate_vectors cannot contain negative rates.")

        if input_rate_noise_std_hz is None:
            noise_std_hz = float(input_rate_noise_fraction) * float(input_rate_max_hz - input_rate_min_hz)
        else:
            noise_std_hz = float(input_rate_noise_std_hz)
        if noise_std_hz < 0:
            raise ValueError("input_rate_noise_std_hz/input_rate_noise_fraction implies a negative noise scale.")

        clip_low = None
        clip_high = None
        if input_rate_clip is not None:
            clip_low = input_rate_clip[0]
            clip_high = input_rate_clip[1]
            clip_low = None if clip_low is None else float(clip_low)
            clip_high = None if clip_high is None else float(clip_high)
            if clip_low is not None and clip_high is not None and clip_high < clip_low:
                raise ValueError("input_rate_clip upper bound cannot be smaller than lower bound.")

        warmup_steps = int(round(float(warmup_ms) / float(self.dt)))
        measurement_steps = int(round(float(measurement_ms) / float(self.dt)))
        bin_steps = max(1, int(round(float(bin_ms) / float(self.dt))))
        if measurement_steps <= 0:
            raise ValueError("measurement_ms is too short for the current dt.")
        sample_step_numbers = np.arange(bin_steps, measurement_steps + 1, bin_steps, dtype=int)
        if sample_step_numbers.size == 0 or sample_step_numbers[-1] != measurement_steps:
            sample_step_numbers = np.unique(np.append(sample_step_numbers, measurement_steps)).astype(int)
        sample_step_set = set(int(x) for x in sample_step_numbers.tolist())
        sample_times_ms = float(warmup_ms) + sample_step_numbers.astype(float) * float(self.dt)

        n_samples = int(n_classes) * int(trials_per_class)
        labels = np.repeat(np.arange(int(n_classes), dtype=int), int(trials_per_class))
        trial_order = rng.permutation(n_samples)
        labels = labels[trial_order]
        states = np.zeros((n_samples, sample_step_numbers.size, state_indices.size), dtype=float)
        if summary_key == "mean_final":
            state_vectors = np.zeros((n_samples, state_indices.size * 2), dtype=float)
        else:
            state_vectors = np.zeros((n_samples, state_indices.size), dtype=float)
        trial_rate_vectors = np.zeros((n_samples, input_indices.size), dtype=float)
        trial_mean_rates = np.zeros(n_samples, dtype=float)
        decay = float(np.exp(-float(self.dt) / float(tau_ms)))

        base_state0 = None
        if not randomize_initial_state:
            base_state0 = (
                self.neuron_states.V.copy(),
                self.neuron_states.u.copy(),
                self.neuron_states.spike.copy(),
                self.neuron_states.T.copy(),
            )

        sample_iter = range(n_samples)
        if show_progress:
            try:
                from tqdm.auto import tqdm
                sample_iter = tqdm(sample_iter, desc="Generalization trials", unit="trial")
            except Exception:
                pass

        for sample_idx in sample_iter:
            class_idx = int(labels[sample_idx])
            if randomize_initial_state:
                state0 = self._random_state0(
                    n_neurons,
                    rng,
                    v_range=state_v_range,
                    u_range=state_u_range,
                )
            else:
                state0 = tuple(x.copy() for x in base_state0)

            rate_vector = base_rate_vectors[class_idx].copy()
            if noise_std_hz > 0:
                rate_vector = rate_vector + rng.normal(0.0, noise_std_hz, size=rate_vector.shape)
            if clip_low is not None or clip_high is not None:
                low = -np.inf if clip_low is None else clip_low
                high = np.inf if clip_high is None else clip_high
                rate_vector = np.clip(rate_vector, low, high)
            trial_rate_vectors[sample_idx, :] = rate_vector

            rates = np.zeros(n_neurons, dtype=float)
            amps = np.zeros(n_neurons, dtype=float)
            rates[input_indices] = rate_vector
            amps[input_indices] = float(input_amplitude)
            poisson = PoissonInput(
                n_neurons,
                rate=rates,
                amplitude=amps,
                rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
            )

            trial_sim = Simulation(
                self.connectome,
                self.dt,
                stepper_type=self.stepper_type,
                state0=state0,
                enable_plasticity=enable_plasticity,
                plasticity=self.plasticity_key,
                plasticity_reward_type=self.plasticity_reward_type,
                plasticity_kwargs=self.plasticity_kwargs,
                synapse_type=self.synapse_type,
                synapse_kwargs=self.synapse_kwargs,
                enable_state_logger=False,
                enable_debug_logger=False,
                rate_normalization=None,
            )

            filtered = np.zeros(n_neurons, dtype=float)
            measurement_spike_count = 0.0
            for _ in range(warmup_steps):
                trial_sim.step(spike_ext=poisson(self.dt))
                filtered *= decay
                filtered += trial_sim.neuron_states.spike.astype(float)

            out_idx = 0
            for measurement_step in range(1, measurement_steps + 1):
                trial_sim.step(spike_ext=poisson(self.dt))
                spikes_now = trial_sim.neuron_states.spike.astype(float)
                filtered *= decay
                filtered += spikes_now
                measurement_spike_count += float(np.sum(spikes_now[state_indices]))
                if measurement_step in sample_step_set:
                    states[sample_idx, out_idx, :] = filtered[state_indices]
                    out_idx += 1

            if summary_key == "mean":
                state_vectors[sample_idx, :] = np.mean(states[sample_idx], axis=0)
            elif summary_key == "final":
                state_vectors[sample_idx, :] = states[sample_idx, -1, :]
            elif summary_key == "max":
                state_vectors[sample_idx, :] = np.max(states[sample_idx], axis=0)
            else:
                state_vectors[sample_idx, :] = np.concatenate(
                    [np.mean(states[sample_idx], axis=0), states[sample_idx, -1, :]]
                )
            trial_mean_rates[sample_idx] = float(
                measurement_spike_count / max(1, state_indices.size) / (measurement_steps * self.dt / 1000.0)
            )

        metric_iter = None
        if show_progress:
            try:
                from tqdm.auto import tqdm
                metric_iter = tqdm(total=1, desc="Generalization metrics", unit="fit")
            except Exception:
                metric_iter = None
        metrics = SimulationStats.generalization_property_metrics(
            state_vectors,
            labels,
            train_fraction=train_fraction,
            ridge_alpha=ridge_alpha,
            lda_regularization=lda_regularization,
            standardize=standardize,
            rng=rng,
        )
        if metric_iter is not None:
            metric_iter.update(1)
            metric_iter.close()

        pop = self.connectome.neuron_population
        neuron_type_by_index = np.asarray(pop.neuron_population_types)
        inhibitory_mask = np.asarray(pop.inhibitory_mask, dtype=bool)
        result = {
            "metrics": metrics,
            "state_vectors": state_vectors,
            "states": states if store_states else None,
            "state_times_ms": sample_times_ms,
            "labels": labels,
            "input_rate_vectors_Hz": base_rate_vectors if store_rate_vectors else None,
            "trial_rate_vectors_Hz": trial_rate_vectors if store_rate_vectors else None,
            "input_indices": input_indices,
            "state_indices": state_indices,
            "input_neuron_types": neuron_type_by_index[input_indices].copy(),
            "state_neuron_types": neuron_type_by_index[state_indices].copy(),
            "input_inhibitory_mask": inhibitory_mask[input_indices].copy(),
            "state_inhibitory_mask": inhibitory_mask[state_indices].copy(),
            "trial_mean_rates_Hz": trial_mean_rates,
            "config": {
                "n_classes": int(n_classes),
                "trials_per_class": int(trials_per_class),
                "input_fraction": None if input_fraction is None else float(input_fraction),
                "input_pool": input_pool,
                "state_fraction": None if state_fraction is None else float(state_fraction),
                "state_pool": state_pool,
                "disjoint_state_pool": bool(disjoint_state_pool),
                "input_rate_min_hz": float(input_rate_min_hz),
                "input_rate_max_hz": float(input_rate_max_hz),
                "input_rate_noise_std_hz": float(noise_std_hz),
                "input_rate_noise_fraction": float(input_rate_noise_fraction),
                "input_rate_clip": input_rate_clip,
                "input_amplitude": float(input_amplitude),
                "warmup_ms": float(warmup_ms),
                "measurement_ms": float(measurement_ms),
                "tau_ms": float(tau_ms),
                "bin_ms": float(bin_ms),
                "dt_ms": float(self.dt),
                "state_summary": summary_key,
                "train_fraction": float(train_fraction),
                "ridge_alpha": float(ridge_alpha),
                "lda_regularization": float(lda_regularization),
                "standardize": bool(standardize),
                "randomize_initial_state": bool(randomize_initial_state),
                "show_progress": bool(show_progress),
            },
        }
        return self.stats.save_generalization_property(result)

    def analyze_memory_capacity(
        self,
        *,
        input_neurons=None,
        input_fraction=0.3,
        input_pool="E",
        state_neurons=None,
        state_fraction=None,
        state_pool="all",
        disjoint_state_pool=True,
        input_rate_min_hz=5.0,
        input_rate_max_hz=50.0,
        input_amplitude=0.5,
        input_value_range=(0.0, 1.0),
        washout_ms=1000.0,
        measurement_ms=30000.0,
        tau_ms=30.0,
        bin_ms=20.0,
        max_delay_bins=50,
        delays_bins=None,
        train_fraction=0.7,
        ridge_alpha=1.0,
        standardize=True,
        clip_negative_r2=True,
        shuffle_split=False,
        seed=None,
        randomize_initial_state=True,
        state_v_range=(-100.0, -70.0),
        state_u_range=(0.0, 400.0),
        enable_plasticity=False,
        store_states=True,
        store_readouts=False,
        show_progress=True,
    ):
        """
        Estimate linear memory capacity from a continuous random input stream.

        A scalar input value u(t) is sampled uniformly once per `bin_ms`, mapped
        to a Poisson input rate between `input_rate_min_hz` and
        `input_rate_max_hz`, and broadcast to the fixed input neuron population.
        Filtered spike states are sampled once per bin after the washout period.
        Ridge readouts then reconstruct u(t-k) from x(t) for each delay k.
        """
        if washout_ms < 0 or measurement_ms <= 0:
            raise ValueError("washout_ms must be >= 0 and measurement_ms must be > 0.")
        if tau_ms <= 0 or bin_ms <= 0:
            raise ValueError("tau_ms and bin_ms must be > 0.")
        if input_rate_min_hz < 0 or input_rate_max_hz < 0:
            raise ValueError("input rates must be >= 0.")
        if input_rate_min_hz > input_rate_max_hz:
            raise ValueError("input_rate_min_hz cannot exceed input_rate_max_hz.")
        if input_amplitude < 0:
            raise ValueError("input_amplitude must be >= 0.")
        if int(max_delay_bins) <= 0 and delays_bins is None:
            raise ValueError("max_delay_bins must be > 0 when delays_bins is not provided.")
        value_low, value_high = float(input_value_range[0]), float(input_value_range[1])
        if value_high <= value_low:
            raise ValueError("input_value_range must be increasing.")

        rng = np.random.default_rng(seed)
        n_neurons = int(self.connectome.neuron_population.n_neurons)
        input_indices = self._resolve_neuron_subset(
            neurons=input_neurons,
            fraction=input_fraction,
            pool=input_pool,
            rng=rng,
            name="input neuron subset",
        )
        state_indices = self._resolve_neuron_subset(
            neurons=state_neurons,
            fraction=state_fraction,
            pool=state_pool,
            rng=rng,
            name="state neuron subset",
        )
        if disjoint_state_pool:
            state_indices = np.setdiff1d(state_indices, input_indices, assume_unique=False)
            if state_indices.size == 0:
                raise ValueError(
                    "disjoint_state_pool=True removed all state neurons. "
                    "Choose a smaller input pool, a broader state pool, or pass explicit disjoint state_neurons."
                )

        bin_steps = max(1, int(round(float(bin_ms) / float(self.dt))))
        realized_bin_ms = float(bin_steps * self.dt)
        washout_bins = max(0, int(round(float(washout_ms) / realized_bin_ms)))
        measurement_bins = max(1, int(round(float(measurement_ms) / realized_bin_ms)))
        if delays_bins is None:
            delays = np.arange(1, int(max_delay_bins) + 1, dtype=int)
        else:
            delays = np.asarray(delays_bins, dtype=int).reshape(-1)
        delays = delays[(delays > 0) & (delays < measurement_bins)]
        if delays.size == 0:
            raise ValueError("No valid positive delays shorter than the measurement window.")

        if randomize_initial_state:
            state0 = self._random_state0(
                n_neurons,
                rng,
                v_range=state_v_range,
                u_range=state_u_range,
            )
        else:
            state0 = (
                self.neuron_states.V.copy(),
                self.neuron_states.u.copy(),
                self.neuron_states.spike.copy(),
                self.neuron_states.T.copy(),
            )

        trial_sim = Simulation(
            self.connectome,
            self.dt,
            stepper_type=self.stepper_type,
            state0=state0,
            enable_plasticity=enable_plasticity,
            plasticity=self.plasticity_key,
            plasticity_reward_type=self.plasticity_reward_type,
            plasticity_kwargs=self.plasticity_kwargs,
            synapse_type=self.synapse_type,
            synapse_kwargs=self.synapse_kwargs,
            enable_state_logger=False,
            enable_debug_logger=False,
            rate_normalization=None,
        )

        total_bins = washout_bins + measurement_bins
        all_input_values = rng.uniform(value_low, value_high, size=total_bins)
        input_norm = (all_input_values - value_low) / (value_high - value_low)
        all_input_rates = float(input_rate_min_hz) + input_norm * float(input_rate_max_hz - input_rate_min_hz)

        states = np.zeros((measurement_bins, state_indices.size), dtype=float)
        input_values = all_input_values[washout_bins:].copy()
        input_rates = all_input_rates[washout_bins:].copy()
        state_times_ms = (np.arange(measurement_bins, dtype=float) + 1.0) * realized_bin_ms
        state_times_ms += float(washout_bins * realized_bin_ms)
        decay = float(np.exp(-float(self.dt) / float(tau_ms)))
        filtered = np.zeros(n_neurons, dtype=float)
        measurement_spike_count = 0.0

        bin_iter = range(total_bins)
        if show_progress:
            try:
                from tqdm.auto import tqdm
                bin_iter = tqdm(bin_iter, desc="Memory capacity bins", unit="bin")
            except Exception:
                pass

        for bin_idx in bin_iter:
            rates = np.zeros(n_neurons, dtype=float)
            amps = np.zeros(n_neurons, dtype=float)
            rates[input_indices] = float(all_input_rates[bin_idx])
            amps[input_indices] = float(input_amplitude)
            poisson = PoissonInput(
                n_neurons,
                rate=rates,
                amplitude=amps,
                rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
            )

            for _ in range(bin_steps):
                trial_sim.step(spike_ext=poisson(self.dt))
                spikes_now = trial_sim.neuron_states.spike.astype(float)
                filtered *= decay
                filtered += spikes_now
                if bin_idx >= washout_bins:
                    measurement_spike_count += float(np.sum(spikes_now[state_indices]))

            if bin_idx >= washout_bins:
                states[bin_idx - washout_bins, :] = filtered[state_indices]

        metrics = SimulationStats.memory_capacity_metrics(
            states,
            input_values,
            delays_bins=delays,
            train_fraction=train_fraction,
            ridge_alpha=ridge_alpha,
            standardize=standardize,
            clip_negative_r2=clip_negative_r2,
            shuffle_split=shuffle_split,
            delay_bin_ms=realized_bin_ms,
            rng=rng,
            show_progress=show_progress,
        )
        coefficients = metrics.get("coefficients")
        if not store_readouts:
            metrics = dict(metrics)
            metrics["coefficients"] = None

        pop = self.connectome.neuron_population
        neuron_type_by_index = np.asarray(pop.neuron_population_types)
        inhibitory_mask = np.asarray(pop.inhibitory_mask, dtype=bool)
        mean_state_rate = float(
            measurement_spike_count / max(1, state_indices.size) / (measurement_bins * realized_bin_ms / 1000.0)
        )
        result = {
            "metrics": metrics,
            "states": states if store_states else None,
            "input_values": input_values,
            "input_rates_Hz": input_rates,
            "state_times_ms": state_times_ms,
            "input_indices": input_indices,
            "state_indices": state_indices,
            "input_neuron_types": neuron_type_by_index[input_indices].copy(),
            "state_neuron_types": neuron_type_by_index[state_indices].copy(),
            "input_inhibitory_mask": inhibitory_mask[input_indices].copy(),
            "state_inhibitory_mask": inhibitory_mask[state_indices].copy(),
            "mean_state_rate_Hz": mean_state_rate,
            "readouts": coefficients if store_readouts else None,
            "config": {
                "input_fraction": None if input_fraction is None else float(input_fraction),
                "input_pool": input_pool,
                "state_fraction": None if state_fraction is None else float(state_fraction),
                "state_pool": state_pool,
                "disjoint_state_pool": bool(disjoint_state_pool),
                "input_rate_min_hz": float(input_rate_min_hz),
                "input_rate_max_hz": float(input_rate_max_hz),
                "input_amplitude": float(input_amplitude),
                "input_value_range": (value_low, value_high),
                "washout_ms": float(washout_ms),
                "measurement_ms": float(measurement_ms),
                "realized_washout_ms": float(washout_bins * realized_bin_ms),
                "realized_measurement_ms": float(measurement_bins * realized_bin_ms),
                "tau_ms": float(tau_ms),
                "bin_ms": float(bin_ms),
                "realized_bin_ms": float(realized_bin_ms),
                "dt_ms": float(self.dt),
                "max_delay_bins": int(max_delay_bins),
                "delays_bins": delays.copy(),
                "train_fraction": float(train_fraction),
                "ridge_alpha": float(ridge_alpha),
                "standardize": bool(standardize),
                "clip_negative_r2": bool(clip_negative_r2),
                "shuffle_split": bool(shuffle_split),
                "randomize_initial_state": bool(randomize_initial_state),
                "show_progress": bool(show_progress),
            },
        }
        return self.stats.save_memory_capacity(result)

    def analyze_regime_persistence_regeneration(
        self,
        *,
        warmup_ms=500.0,
        observation_ms=1000.0,
        transient_ms=3000.0,
        input_neurons=None,
        input_fraction=0.3,
        input_pool="E",
        target_fraction=None,
        topology_type="auto",
        spatial_axis=0,
        spatial_pos_attr="pos",
        input_rate_hz=50.0,
        input_amplitude=0.5,
        seed=None,
        randomize_initial_state=True,
        state_v_range=(-100.0, -70.0),
        state_u_range=(0.0, 400.0),
        enable_plasticity=False,
        transient_start_window_ms=500.0,
        transient_end_window_ms=500.0,
        evoked_min_rate_hz=1.0,
        spread_min_rate_hz=1.0,
        spread_ratio_threshold=0.7,
        persistent_min_rate_hz=1.0,
        persistent_ratio_threshold=0.2,
        runaway_rate_hz=100.0,
        brunel_cv_threshold=0.99,
        brunel_fano_threshold=0.99,
        brunel_corr_threshold=0.05,
        brunel_peak_ratio_threshold=250.0,
        oscillatory_entropy_norm_threshold=0.85,
        bursty_fano_threshold=3.0,
        burst_bin_ms=20.0,
        burst_rate_threshold_hz=10.0,
        burst_participation_threshold=0.3,
        show_progress=True,
        plot_raster=False,
        raster_t_start_ms=None,
        raster_t_stop_ms=None,
        raster_figsize=(12, 6),
        raster_save_path=None,
        store_simulation_stats=True,
    ):
        """
        Classify Brunel regime, persistence, and regenerative activity.

        Periods:
          1. warmup: external input on, no classification
          2. observation: external input on, Brunel + spread metrics
          3. transient: input off, persistence + burst regeneration metrics
        """
        if warmup_ms < 0 or observation_ms <= 0 or transient_ms <= 0:
            raise ValueError("warmup_ms must be >= 0 and observation_ms/transient_ms must be > 0.")
        if input_rate_hz < 0 or input_amplitude < 0:
            raise ValueError("input_rate_hz and input_amplitude must be >= 0.")
        if input_fraction <= 0 or input_fraction > 1:
            raise ValueError("input_fraction must be in (0, 1].")
        if target_fraction is None:
            target_fraction = input_fraction
        if target_fraction <= 0 or target_fraction > 1:
            raise ValueError("target_fraction must be in (0, 1].")

        rng = np.random.default_rng(seed)
        n_neurons = int(self.connectome.neuron_population.n_neurons)
        pop = self.connectome.neuron_population
        inhib_mask = np.asarray(pop.inhibitory_mask, dtype=bool)
        exc_indices = np.flatnonzero(~inhib_mask)

        topology_key = str(topology_type).lower()
        if topology_key == "auto":
            G = getattr(self.connectome, "G", None)
            has_pos = (
                G is not None
                and exc_indices.size > 0
                and all((int(i) in G.nodes and spatial_pos_attr in G.nodes[int(i)]) for i in exc_indices[: min(20, exc_indices.size)])
            )
            topology_key = "spatial" if has_pos else "nonspatial"

        if topology_key == "spatial":
            if input_neurons is not None:
                input_indices = np.asarray(input_neurons, dtype=int).reshape(-1)
                if input_indices.size == 0:
                    raise ValueError("input_neurons cannot be empty.")
                _, target_indices = self._select_spatial_source_target(
                    fraction=target_fraction,
                    pool=input_pool,
                    axis=spatial_axis,
                    pos_attr=spatial_pos_attr,
                )
                target_indices = np.setdiff1d(target_indices, input_indices, assume_unique=False)
            else:
                input_indices, target_indices = self._select_spatial_source_target(
                    fraction=input_fraction,
                    pool=input_pool,
                    axis=spatial_axis,
                    pos_attr=spatial_pos_attr,
                )
        elif topology_key in ("nonspatial", "fixed", "random", "graph"):
            input_indices = self._resolve_neuron_subset(
                neurons=input_neurons,
                fraction=input_fraction,
                pool=input_pool,
                rng=rng,
                name="input neuron subset",
            )
            target_indices = self._select_graph_distant_target(
                input_indices=input_indices,
                fraction=target_fraction,
                pool=input_pool,
            )
        else:
            raise ValueError("topology_type must be 'auto', 'spatial', or 'nonspatial'.")

        input_indices = np.asarray(input_indices, dtype=int).reshape(-1)
        target_indices = np.asarray(target_indices, dtype=int).reshape(-1)
        if np.any(input_indices < 0) or np.any(input_indices >= n_neurons):
            raise ValueError("input_indices contain out-of-range neurons.")
        if target_indices.size == 0:
            raise ValueError("target selection produced no neurons.")

        if randomize_initial_state:
            state0 = self._random_state0(
                n_neurons,
                rng,
                v_range=state_v_range,
                u_range=state_u_range,
            )
        else:
            state0 = (
                self.neuron_states.V.copy(),
                self.neuron_states.u.copy(),
                self.neuron_states.spike.copy(),
                self.neuron_states.T.copy(),
            )

        trial_sim = Simulation(
            self.connectome,
            self.dt,
            stepper_type=self.stepper_type,
            state0=state0,
            enable_plasticity=enable_plasticity,
            plasticity=self.plasticity_key,
            plasticity_reward_type=self.plasticity_reward_type,
            plasticity_kwargs=self.plasticity_kwargs,
            synapse_type=self.synapse_type,
            synapse_kwargs=self.synapse_kwargs,
            enable_state_logger=True,
            enable_debug_logger=False,
            rate_normalization=None,
        )

        rates = np.zeros(n_neurons, dtype=float)
        amps = np.zeros(n_neurons, dtype=float)
        rates[input_indices] = float(input_rate_hz)
        amps[input_indices] = float(input_amplitude)
        poisson = PoissonInput(
            n_neurons,
            rate=rates,
            amplitude=amps,
            rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
        )

        def _steps(duration_ms):
            return max(0, int(round(float(duration_ms) / float(self.dt))))

        periods = [
            ("warmup", _steps(warmup_ms), True),
            ("observation", _steps(observation_ms), True),
            ("transient", _steps(transient_ms), False),
        ]
        period_iter = periods
        if show_progress:
            try:
                from tqdm.auto import tqdm
                period_iter = tqdm(periods, desc="Regime periods", unit="period")
            except Exception:
                pass

        period_start_times = {}
        period_stop_times = {}
        for period_name, n_steps, input_on in period_iter:
            period_start_times[period_name] = float(trial_sim.t_now)
            for _ in range(n_steps):
                if input_on:
                    trial_sim.step(spike_ext=poisson(self.dt))
                else:
                    trial_sim.step()
            period_stop_times[period_name] = float(trial_sim.t_now)

        obs_start = period_start_times["observation"]
        obs_stop = period_stop_times["observation"]
        trans_start = period_start_times["transient"]
        trans_stop = period_stop_times["transient"]
        trans_start_stop = min(trans_stop, trans_start + float(transient_start_window_ms))
        trans_end_start = max(trans_start, trans_stop - float(transient_end_window_ms))

        obs_metrics = trial_sim.stats.compute_metrics(self.dt, t_start_ms=obs_start, t_stop_ms=obs_stop)
        transient_metrics = trial_sim.stats.compute_metrics(self.dt, t_start_ms=trans_start, t_stop_ms=trans_stop)
        transient_start_metrics = trial_sim.stats.compute_metrics(self.dt, t_start_ms=trans_start, t_stop_ms=trans_start_stop)
        transient_end_metrics = trial_sim.stats.compute_metrics(self.dt, t_start_ms=trans_end_start, t_stop_ms=trans_stop)
        brunel = self._classify_brunel_from_metrics(
            obs_metrics,
            cv_threshold=brunel_cv_threshold,
            fano_threshold=brunel_fano_threshold,
            corr_threshold=brunel_corr_threshold,
            peak_ratio_threshold=brunel_peak_ratio_threshold,
            oscillatory_entropy_norm_threshold=oscillatory_entropy_norm_threshold,
            bursty_fano_threshold=bursty_fano_threshold,
        )

        S = trial_sim.stats.spikes_bool()
        times = trial_sim.stats.times_ms(dt_ms=self.dt)
        source_rate_obs = self._rate_hz_for_indices(S, times, input_indices, obs_start, obs_stop)
        target_rate_obs = self._rate_hz_for_indices(S, times, target_indices, obs_start, obs_stop)
        spread_ratio = float(target_rate_obs / source_rate_obs) if source_rate_obs > 1e-12 else 0.0
        is_spread_regenerative = bool(
            source_rate_obs >= float(evoked_min_rate_hz)
            and target_rate_obs >= float(spread_min_rate_hz)
            and spread_ratio >= float(spread_ratio_threshold)
        )

        obs_rate = float(obs_metrics.get("rate_mean_Hz", 0.0))
        transient_start_rate = float(transient_start_metrics.get("rate_mean_Hz", 0.0))
        transient_mean_rate = float(transient_metrics.get("rate_mean_Hz", 0.0))
        transient_end_rate = float(transient_end_metrics.get("rate_mean_Hz", 0.0))
        persistent_score = float(transient_end_rate / obs_rate) if obs_rate > 1e-12 else 0.0
        is_persistent = bool(
            transient_end_rate >= float(persistent_min_rate_hz)
            and persistent_score >= float(persistent_ratio_threshold)
        )
        is_runaway = bool(transient_end_rate >= float(runaway_rate_hz))
        burst_metrics = self._detect_global_bursts(
            S,
            times,
            t_start_ms=trans_start,
            t_stop_ms=trans_stop,
            bin_ms=burst_bin_ms,
            rate_threshold_hz=burst_rate_threshold_hz,
            participation_threshold=burst_participation_threshold,
        )
        is_persistent_regenerative = bool(is_persistent and burst_metrics["n_global_bursts"] > 0)

        if obs_rate < float(evoked_min_rate_hz) and source_rate_obs < float(evoked_min_rate_hz):
            classification = "inactive"
        elif is_runaway:
            classification = "runaway"
        elif is_persistent_regenerative:
            classification = "persistent_regenerative_bursting"
        elif is_persistent:
            classification = "persistent_nonregenerative"
        elif is_spread_regenerative:
            classification = "transient_regenerative_spread"
        else:
            classification = "transient_local"

        if plot_raster:
            raster_start = 0.0 if raster_t_start_ms is None else float(raster_t_start_ms)
            raster_stop = trans_stop if raster_t_stop_ms is None else float(raster_t_stop_ms)
            title = (
                f"{classification} | Brunel {brunel['brunel_class']} | "
                f"persistent={is_persistent} | spread={is_spread_regenerative}"
            )
            trial_sim.plot_spike_raster(
                dt_ms=self.dt,
                t_start_ms=raster_start,
                t_stop_ms=raster_stop,
                figsize=raster_figsize,
                title=title,
                save_path=raster_save_path,
            )

        result = {
            "classification": classification,
            "brunel": brunel,
            "persistence": {
                "is_persistent": is_persistent,
                "persistent_score_end_over_observation": persistent_score,
                "observation_rate_Hz": obs_rate,
                "transient_start_rate_Hz": transient_start_rate,
                "transient_mean_rate_Hz": transient_mean_rate,
                "transient_end_rate_Hz": transient_end_rate,
                "is_runaway": is_runaway,
            },
            "regeneration": {
                "topology_type": topology_key,
                "source_rate_observation_Hz": source_rate_obs,
                "target_rate_observation_Hz": target_rate_obs,
                "spread_ratio": spread_ratio,
                "is_spread_regenerative": is_spread_regenerative,
                "is_persistent_regenerative": is_persistent_regenerative,
                **burst_metrics,
            },
            "metrics": {
                "observation": obs_metrics,
                "transient": transient_metrics,
                "transient_start": transient_start_metrics,
                "transient_end": transient_end_metrics,
            },
            "input_indices": input_indices,
            "target_indices": target_indices,
            "input_neuron_types": np.asarray(pop.neuron_population_types)[input_indices].copy(),
            "target_neuron_types": np.asarray(pop.neuron_population_types)[target_indices].copy(),
            "periods_ms": {
                "warmup": (period_start_times["warmup"], period_stop_times["warmup"]),
                "observation": (obs_start, obs_stop),
                "transient": (trans_start, trans_stop),
                "transient_start": (trans_start, trans_start_stop),
                "transient_end": (trans_end_start, trans_stop),
            },
            "simulation_stats": trial_sim.stats if store_simulation_stats else None,
            "config": {
                "warmup_ms": float(warmup_ms),
                "observation_ms": float(observation_ms),
                "transient_ms": float(transient_ms),
                "input_fraction": float(input_fraction),
                "input_pool": input_pool,
                "target_fraction": float(target_fraction),
                "topology_type": topology_type,
                "resolved_topology_type": topology_key,
                "spatial_axis": int(spatial_axis),
                "spatial_pos_attr": spatial_pos_attr,
                "input_rate_hz": float(input_rate_hz),
                "input_amplitude": float(input_amplitude),
                "transient_start_window_ms": float(transient_start_window_ms),
                "transient_end_window_ms": float(transient_end_window_ms),
                "evoked_min_rate_hz": float(evoked_min_rate_hz),
                "spread_min_rate_hz": float(spread_min_rate_hz),
                "spread_ratio_threshold": float(spread_ratio_threshold),
                "persistent_min_rate_hz": float(persistent_min_rate_hz),
                "persistent_ratio_threshold": float(persistent_ratio_threshold),
                "runaway_rate_hz": float(runaway_rate_hz),
                "burst_bin_ms": float(burst_bin_ms),
                "burst_rate_threshold_hz": float(burst_rate_threshold_hz),
                "burst_participation_threshold": float(burst_participation_threshold),
                "oscillatory_entropy_norm_threshold": float(oscillatory_entropy_norm_threshold),
                "bursty_fano_threshold": float(bursty_fano_threshold),
                "plot_raster": bool(plot_raster),
                "raster_t_start_ms": raster_t_start_ms,
                "raster_t_stop_ms": raster_t_stop_ms,
                "raster_save_path": raster_save_path,
                "randomize_initial_state": bool(randomize_initial_state),
            },
        }
        return self.stats.save_regime_persistence_regeneration(result)

    def analyze_input_amplitude_bifurcation(
        self,
        input_amplitudes,
        *,
        input_rate_hz=50.0,
        warmup_ms=500.0,
        measurement_ms=1000.0,
        input_neurons=None,
        input_fraction=0.3,
        input_pool="E",
        seed=None,
        randomize_initial_state=True,
        state_v_range=(-100.0, -70.0),
        state_u_range=(0.0, 400.0),
        enable_plasticity=False,
        active_rate_threshold_hz=1.0,
        strong_peak_ratio_threshold=250.0,
        strong_entropy_norm_threshold=0.85,
        metadata=None,
        show_progress=True,
    ):
        """
        Run a ramp-up/ramp-down input-amplitude sweep and summarize hysteresis.

        The simulation state is carried continuously across amplitude blocks.
        Within each block, input is on during both warmup and measurement, but
        metrics/classifications are computed only over the measurement window.
        Returned rows are scalar-only dictionaries intended for direct
        concatenation across networks/topologies/parameter settings.
        """
        amplitudes = [float(a) for a in input_amplitudes]
        if len(amplitudes) == 0:
            raise ValueError("input_amplitudes must contain at least one value.")
        if warmup_ms < 0 or measurement_ms <= 0:
            raise ValueError("warmup_ms must be >= 0 and measurement_ms must be > 0.")
        if input_rate_hz < 0:
            raise ValueError("input_rate_hz must be >= 0.")
        metadata = {} if metadata is None else dict(metadata)

        rng = np.random.default_rng(seed)
        n_neurons = int(self.connectome.neuron_population.n_neurons)
        input_indices = self._resolve_neuron_subset(
            neurons=input_neurons,
            fraction=input_fraction,
            pool=input_pool,
            rng=rng,
            name="input neuron subset",
        )

        if randomize_initial_state:
            state0 = self._random_state0(
                n_neurons,
                rng,
                v_range=state_v_range,
                u_range=state_u_range,
            )
        else:
            state0 = (
                self.neuron_states.V.copy(),
                self.neuron_states.u.copy(),
                self.neuron_states.spike.copy(),
                self.neuron_states.T.copy(),
            )

        ramp_sim = Simulation(
            self.connectome,
            self.dt,
            stepper_type=self.stepper_type,
            state0=state0,
            enable_plasticity=enable_plasticity,
            plasticity=self.plasticity_key,
            plasticity_reward_type=self.plasticity_reward_type,
            plasticity_kwargs=self.plasticity_kwargs,
            synapse_type=self.synapse_type,
            synapse_kwargs=self.synapse_kwargs,
            enable_state_logger=True,
            enable_debug_logger=False,
            rate_normalization=None,
        )

        schedule = []
        for step_idx, amp in enumerate(amplitudes):
            schedule.append({"sweep_direction": "up", "direction_step_index": step_idx, "input_amplitude": float(amp)})
        for step_idx, amp in enumerate(reversed(amplitudes)):
            schedule.append({"sweep_direction": "down", "direction_step_index": step_idx, "input_amplitude": float(amp)})

        schedule_iter = schedule
        if show_progress:
            try:
                from tqdm.auto import tqdm
                schedule_iter = tqdm(schedule, desc="Bifurcation amplitude blocks", unit="block")
            except Exception:
                pass

        warmup_steps = int(round(float(warmup_ms) / float(self.dt)))
        measurement_steps = int(round(float(measurement_ms) / float(self.dt)))
        rows = []

        for block_index, block in enumerate(schedule_iter):
            amp = float(block["input_amplitude"])
            ramp_sim.reset_stats()

            rates = np.zeros(n_neurons, dtype=float)
            amps = np.zeros(n_neurons, dtype=float)
            rates[input_indices] = float(input_rate_hz)
            amps[input_indices] = amp
            poisson = PoissonInput(
                n_neurons,
                rate=rates,
                amplitude=amps,
                rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
            )

            for _ in range(warmup_steps):
                ramp_sim.step(spike_ext=poisson(self.dt))
            measurement_start_ms = float(ramp_sim.t_now)
            for _ in range(measurement_steps):
                ramp_sim.step(spike_ext=poisson(self.dt))
            measurement_stop_ms = float(ramp_sim.t_now)

            metrics = ramp_sim.stats.compute_metrics(
                self.dt,
                t_start_ms=measurement_start_ms,
                t_stop_ms=measurement_stop_ms,
            )
            scalar_metrics = self._scalar_metric_items(metrics)
            is_active = float(scalar_metrics.get("rate_mean_Hz", 0.0)) >= float(active_rate_threshold_hz)
            is_strong = (
                float(scalar_metrics.get("psd_peak_ratio", 0.0)) > float(strong_peak_ratio_threshold)
                and float(scalar_metrics.get("pop_spec_entropy_norm", np.inf)) < float(strong_entropy_norm_threshold)
            )
            is_strong = is_strong or float(scalar_metrics.get("psd_peak_ratio", 0.0)) > float(strong_peak_ratio_threshold + 100.0)
            is_strong = is_strong or float(scalar_metrics.get("pop_spec_entropy_norm", np.inf)) < float(strong_entropy_norm_threshold - 0.05)

            row = {
                **metadata,
                "block_index": int(block_index),
                "sweep_direction": block["sweep_direction"],
                "direction_step_index": int(block["direction_step_index"]),
                "input_amplitude": amp,
                "input_rate_hz": float(input_rate_hz),
                "warmup_ms": float(warmup_ms),
                "measurement_ms": float(measurement_ms),
                "measurement_start_ms": measurement_start_ms,
                "measurement_stop_ms": measurement_stop_ms,
                "n_input_neurons": int(input_indices.size),
                "is_active": bool(is_active),
                "is_strongly_oscillatory": bool(is_strong),
                "strong_peak_ratio_threshold": float(strong_peak_ratio_threshold),
                "strong_entropy_norm_threshold": float(strong_entropy_norm_threshold),
            }
            row.update(scalar_metrics)
            rows.append(row)

        up_rows = [row for row in rows if row["sweep_direction"] == "up"]
        down_rows = [row for row in rows if row["sweep_direction"] == "down"]
        active_up = self._threshold_first_true(up_rows, "is_active")
        active_down = self._threshold_min_true(down_rows, "is_active")
        strong_up = self._threshold_first_true(up_rows, "is_strongly_oscillatory")
        strong_down = self._threshold_min_true(down_rows, "is_strongly_oscillatory")
        hysteresis_curve, rate_area = self._hysteresis_curve_summary(rows, metric_key="rate_mean_Hz")

        def _width(up, down):
            return None if up is None or down is None else float(up - down)

        summary = {
            **metadata,
            "active_up_threshold": active_up,
            "active_down_threshold": active_down,
            "active_hysteresis_width": _width(active_up, active_down),
            "strong_oscillatory_up_threshold": strong_up,
            "strong_oscillatory_down_threshold": strong_down,
            "strong_oscillatory_hysteresis_width": _width(strong_up, strong_down),
            "rate_hysteresis_area": float(rate_area) if np.isfinite(rate_area) else np.nan,
            "n_blocks": int(len(rows)),
            "n_input_neurons": int(input_indices.size),
            "active_rate_threshold_hz": float(active_rate_threshold_hz),
            "strong_peak_ratio_threshold": float(strong_peak_ratio_threshold),
            "strong_entropy_norm_threshold": float(strong_entropy_norm_threshold),
        }

        result = {
            "rows": rows,
            "summary": summary,
            "hysteresis_curve": hysteresis_curve,
            "input_indices": input_indices,
            "input_neuron_types": np.asarray(self.connectome.neuron_population.neuron_population_types)[input_indices].copy(),
            "config": {
                "input_amplitudes": amplitudes,
                "input_rate_hz": float(input_rate_hz),
                "warmup_ms": float(warmup_ms),
                "measurement_ms": float(measurement_ms),
                "input_fraction": None if input_fraction is None else float(input_fraction),
                "input_pool": input_pool,
                "active_rate_threshold_hz": float(active_rate_threshold_hz),
                "strong_peak_ratio_threshold": float(strong_peak_ratio_threshold),
                "strong_entropy_norm_threshold": float(strong_entropy_norm_threshold),
                "randomize_initial_state": bool(randomize_initial_state),
                "metadata": metadata,
            },
        }
        return self.stats.save_input_amplitude_bifurcation(result)

    def apply_reward(self, reward):
        """
        Apply a reward signal to the plasticity mechanism (if it uses it).
        """
        if self.plasticity is None:
            return

        apply_fn = getattr(self.plasticity, "apply_weight_changes", None)
        if not callable(apply_fn):
            raise RuntimeError(
                f"Plasticity '{type(self.plasticity).__name__}' does not implement apply_weight_changes(...)."
            )

        sig = inspect.signature(apply_fn)
        param_names = set(sig.parameters.keys())
        if "Vs" in param_names or "V" in param_names:
            apply_fn(self.neuron_states.V, reward=reward)
        else:
            apply_fn(reward=reward)

    def configure_output_readout(self, output_neuron_indices, output_dim, enable_logger=False, rate_window_ms=100.0):
        """
        Configure an online output decoder from grouped output neurons.

        Args:
            output_neuron_indices: 1D iterable of neuron indices defining the super group.
            output_dim: Size of the decoded output vector (y). The super group is split
                        into `output_dim` contiguous groups. If the neuron count is not
                        divisible by `output_dim`, the first group gets the remainder.
            rate_window_ms: Sliding window (ms) used for online firing-rate estimation.
                            Output units are mean subgroup firing rates in Hz/neuron.
        """
        n_neurons = int(self.neuron_states.n_neurons)
        raw_indices = np.asarray(output_neuron_indices)
        if raw_indices.dtype == bool:
            if raw_indices.ndim != 1 or raw_indices.size != n_neurons:
                raise ValueError(
                    "Boolean output_neuron_indices must be a 1D mask of length n_neurons."
                )
            indices = np.flatnonzero(raw_indices)
        else:
            indices = np.asarray(output_neuron_indices, dtype=int).reshape(-1)
        if output_dim <= 0:
            raise ValueError("output_dim must be a positive integer.")
        if indices.size == 0:
            raise ValueError("output_neuron_indices must be non-empty.")
        if int(output_dim) > indices.size:
            raise ValueError(
                "output_dim cannot exceed number of output neurons "
                f"(got {output_dim} and {indices.size})."
            )
        if np.any(indices < 0) or np.any(indices >= n_neurons):
            raise ValueError("output_neuron_indices contains out-of-range neuron indices.")
        if rate_window_ms <= 0:
            raise ValueError("rate_window_ms must be > 0.")

        n_groups = int(output_dim)
        base = indices.size // n_groups
        rem = indices.size % n_groups
        group_sizes_list = [base] * n_groups
        group_sizes_list[0] += rem

        groups = []
        start = 0
        for size in group_sizes_list:
            end = start + size
            groups.append(indices[start:end])
            start = end

        group_sizes = np.array([len(g) for g in groups], dtype=float)
        window_steps = max(1, int(round(rate_window_ms / self.dt)))

        self.output_readout = {
            "groups": groups,
            "group_sizes": group_sizes,
            "window_steps": window_steps,
            "buffer": np.zeros((window_steps, int(output_dim)), dtype=float),
            "rolling_counts": np.zeros(int(output_dim), dtype=float),
            "cursor": 0,
            "filled": 0,
        }
        self.output_vector = np.zeros(int(output_dim), dtype=float)

        self.output_logger_enabled = enable_logger
        if self.output_logger_enabled:
            self.output_logger = []

    def _update_output_readout(self, post_spikes):
        if self.output_readout is None:
            return

        s = np.asarray(post_spikes).reshape(-1).astype(float)
        group_counts = np.array([s[g].sum() for g in self.output_readout["groups"]], dtype=float)
        group_counts = group_counts / self.output_readout["group_sizes"]

        cursor = self.output_readout["cursor"]
        old = self.output_readout["buffer"][cursor]
        self.output_readout["rolling_counts"] += group_counts - old
        self.output_readout["buffer"][cursor] = group_counts
        self.output_readout["cursor"] = (cursor + 1) % self.output_readout["window_steps"]
        self.output_readout["filled"] = min(self.output_readout["filled"] + 1, self.output_readout["window_steps"])

        elapsed_s = self.output_readout["filled"] * self.dt / 1000.0
        if elapsed_s > 0:
            self.output_vector = self.output_readout["rolling_counts"] / elapsed_s
        else:
            self.output_vector = np.zeros_like(self.output_vector)

        if self.output_logger_enabled:
            self.output_logger.append(self.output_vector.copy())

    def read_output_vector(self):
        """
        Return current decoded output vector (Hz/neuron), one value per output group.
        """
        if self.output_vector is None:
            raise RuntimeError("Output readout not configured. Call configure_output_readout(...) first.")
        return self.output_vector.copy()

    def reset_stats(self):
        self.stats = SimulationStats()
        self.stats.inhibitory_mask = self.connectome.neuron_population.inhibitory_mask.copy()
        if self.enable_state_logger:
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

    def plot_voltage_per_neuron(self, dt_ms=None, t_start_ms=None, t_stop_ms=None, figsize=(10, 2), sharex=True):
        """
        Plot one voltage trace per neuron, each in its own subplot.
        """
        Vs = np.array(self.stats.Vs)  # shape (T, N)
        if Vs.size == 0:
            return

        T, N = Vs.shape
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

        fig_h = max(figsize[1] * N, 2.5)
        fig, axes = plt.subplots(N, 1, figsize=(figsize[0], fig_h), sharex=sharex)
        if N == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            ax.plot(t_plot, Vs[:, i], linewidth=1.0)
            ax.set_ylabel(f"n{i}\nV (mV)")
            ax.grid(alpha=0.2, linewidth=0.5)

        axes[-1].set_xlabel("Time (ms)")
        fig.tight_layout()
        plt.show()

    def plot_spike_raster(self, dt_ms=None, t_start_ms=None, t_stop_ms=None, figsize=(10, 6), s=8, alpha=0.7, legend=True, title=None, save_path=None):
        """
        Raster plot of spikes across all neurons.
        Neurons are displayed in contiguous blocks ordered by neuron type.
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

        # Build a stable index remap so rows are grouped by neuron type.
        ordered_indices = []
        for type_name in type_names:
            idx = sorted(pop.get_neurons_from_type(type_name))
            ordered_indices.extend(idx)
        ordered_set = set(ordered_indices)
        remainder = [i for i in range(N) if i not in ordered_set]
        ordered_indices.extend(remainder)
        row_of_neuron = {old_idx: new_row for new_row, old_idx in enumerate(ordered_indices)}

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
                ys.append(np.full(spike_idx.size, row_of_neuron[i], dtype=float))

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
        ax.set_ylabel("Neuron index (grouped by type)")
        ax.set_ylim(-0.5, N - 0.5)
        if title is not None:
            ax.set_title(title, pad=10)
        if legend:
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, ncol=1, frameon=False, borderaxespad=0.0)
        # Leave room on the right for the external legend.
        plt.tight_layout(rect=(0.0, 0.0, 0.84, 0.96))
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
