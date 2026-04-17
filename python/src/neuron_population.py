import numpy as np

from src.neuron_templates import neuron_type_IZ


class NeuronPopulation:
    PARAMETER_NAMES = (
        "k",
        "a",
        "b",
        "d",
        "C",
        "Vr",
        "Vt",
        "Vpeak",
        "c",
        "delta_V",
        "bias",
        "threshold_mult",
        "threshold_decay",
    )

    def __init__(
        self,
        n_neurons,
        neuron_types,
        inhibitory,
        threshold_decay,
        delta_V=2.5,
        bias=0.0,
        threshold_mult=1.05,
        n_template_params=9,
        heterogeneity=None,
        rng=None,
    ):
        """
        neuron_types: list of neuron type names from neuron_templates.py
        inhibitory: list of booleans aligned with neuron_types
        heterogeneity: optional parameter sampling rules. Supported forms:
            {"*": {0: ("gaussian", None, 0.1, 2.0)}}
            {"ss4": {"params": [0, 5], "distributions": [("gaussian", None, 0.1), ("gaussian", None, 4.0, 2.0)]}}
            [{"param": 0, "dist": "gaussian", "variance": 0.1, "truncate_std": 2.0}]
        Gaussian tuples are interpreted as:
            (distribution, mean, variance, truncate_std)
        If mean is None, the template value for that neuron type is used.
        """
        if heterogeneity is None and isinstance(delta_V, (dict, list, tuple)):
            heterogeneity = delta_V
            delta_V = 2.5

        self.neuron_types = list(neuron_types)
        self.n_neurons = int(n_neurons)
        self.n_neuron_types = len(self.neuron_types)
        self.inhibitory = list(inhibitory)
        self.n_template_params = int(n_template_params)
        self.n_params = self.n_template_params + 4

        self.threshold_decay = threshold_decay
        self.delta_V = delta_V
        self.bias = bias
        self.threshold_mult = threshold_mult

        self.rng = rng if rng is not None else np.random
        self.heterogeneity = self._normalize_heterogeneity(heterogeneity)

        self.layer_indices = []
        self._initialize_population_arrays(self.n_neurons)

    def _initialize_population_arrays(self, n_neurons):
        self.n_neurons = int(n_neurons)
        self.neuron_population = np.zeros((self.n_neurons, self.n_params), dtype=float)
        self.inhibitory_mask = np.zeros(self.n_neurons, dtype=bool)
        self.neuron_population_types = [None] * self.n_neurons
        self.weight_inhibitory_mask = np.ones(self.n_neurons, dtype=int)

    def _is_distribution_spec(self, spec):
        if isinstance(spec, str):
            return True
        if isinstance(spec, dict):
            return any(
                key in spec
                for key in (
                    "dist",
                    "distribution",
                    "mean",
                    "variance",
                    "std",
                    "sigma",
                    "truncate_std",
                    "truncate",
                )
            )
        if isinstance(spec, (list, tuple)) and spec:
            return isinstance(spec[0], str)
        return False

    def _looks_like_global_heterogeneity(self, heterogeneity):
        if not isinstance(heterogeneity, dict) or not heterogeneity:
            return False
        if "params" in heterogeneity and "distributions" in heterogeneity:
            return True
        if "param" in heterogeneity and self._is_distribution_spec(heterogeneity):
            return True
        if self._is_distribution_spec(heterogeneity):
            return True

        for key in heterogeneity:
            if isinstance(key, int):
                return True
            if isinstance(key, str) and key.isdigit():
                return True
        return False

    def _normalize_heterogeneity(self, heterogeneity):
        if heterogeneity is None:
            return {}

        if isinstance(heterogeneity, (list, tuple)):
            heterogeneity = {"*": heterogeneity}
        elif self._looks_like_global_heterogeneity(heterogeneity):
            heterogeneity = {"*": heterogeneity}
        elif not isinstance(heterogeneity, dict):
            raise TypeError("heterogeneity must be None, a dict, or a list of parameter specifications.")

        normalized = {}
        for neuron_type, target_spec in heterogeneity.items():
            if neuron_type != "*" and neuron_type not in self.neuron_types:
                raise ValueError(f"Unknown neuron type in heterogeneity spec: {neuron_type}")
            normalized[neuron_type] = self._normalize_target_spec(target_spec)
        return normalized

    def _normalize_target_spec(self, target_spec):
        if target_spec is None:
            return {}

        if (
            isinstance(target_spec, (list, tuple))
            and len(target_spec) == 2
            and not isinstance(target_spec[0], (dict, list, tuple))
        ):
            target_spec = [target_spec]

        if isinstance(target_spec, dict) and "params" in target_spec:
            params = list(target_spec["params"])
            distributions = list(target_spec["distributions"])
            if len(params) != len(distributions):
                raise ValueError("heterogeneity params and distributions must have the same length.")
            return {
                self._normalize_param_index(param): self._normalize_distribution_spec(param, dist_spec)
                for param, dist_spec in zip(params, distributions)
            }

        if isinstance(target_spec, dict) and "param" in target_spec:
            param_idx = self._normalize_param_index(target_spec["param"])
            return {param_idx: self._normalize_distribution_spec(param_idx, target_spec)}

        if isinstance(target_spec, dict):
            return {
                self._normalize_param_index(param_idx): self._normalize_distribution_spec(param_idx, dist_spec)
                for param_idx, dist_spec in target_spec.items()
            }

        if isinstance(target_spec, (list, tuple)):
            normalized = {}
            for entry in target_spec:
                if isinstance(entry, dict) and "param" in entry:
                    param_idx = self._normalize_param_index(entry["param"])
                    normalized[param_idx] = self._normalize_distribution_spec(param_idx, entry)
                elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                    param_idx = self._normalize_param_index(entry[0])
                    normalized[param_idx] = self._normalize_distribution_spec(param_idx, entry[1])
                else:
                    raise ValueError(
                        "Sequence heterogeneity entries must be {'param': ..., ...} dicts "
                        "or (param_index, distribution_spec) pairs."
                    )
            return normalized

        raise TypeError("Invalid heterogeneity target specification.")

    def _normalize_param_index(self, param_idx):
        param_idx = int(param_idx)
        if not 0 <= param_idx < self.n_params:
            raise ValueError(
                f"Parameter index {param_idx} is out of bounds for {self.n_params} neuron parameters."
            )
        return param_idx

    def _normalize_distribution_spec(self, param_idx, spec):
        param_idx = self._normalize_param_index(param_idx)

        if isinstance(spec, str):
            spec = {"dist": spec}
        elif isinstance(spec, (list, tuple)):
            if not spec:
                raise ValueError("Distribution specification tuples cannot be empty.")
            spec = {
                "dist": spec[0],
                "mean": spec[1] if len(spec) > 1 else None,
                "variance": spec[2] if len(spec) > 2 else None,
                "truncate_std": spec[3] if len(spec) > 3 else None,
            }
        elif not isinstance(spec, dict):
            raise TypeError("Distribution specifications must be strings, tuples/lists, or dicts.")

        dist = spec.get("dist", spec.get("distribution"))
        if dist is None:
            raise ValueError(f"Missing distribution type for parameter index {param_idx}.")

        dist = str(dist).lower()
        if dist == "normal":
            dist = "gaussian"

        truncate_std = spec.get("truncate_std")
        truncate = spec.get("truncate")
        if truncate_std is None and truncate is not None:
            if isinstance(truncate, dict):
                truncate_std = truncate.get("std")
            else:
                truncate_std = truncate

        normalized = {
            "dist": dist,
            "mean": spec.get("mean"),
            "variance": spec.get("variance"),
            "std": spec.get("std", spec.get("sigma")),
            "truncate_std": truncate_std,
        }

        if normalized["dist"] not in ("gaussian", "fixed", "constant", "uniform"):
            raise ValueError(
                f"Unsupported heterogeneity distribution '{dist}' for parameter index {param_idx}."
            )

        return normalized

    def _full_params_from_type(self, neuron_type):
        if neuron_type not in neuron_type_IZ:
            raise ValueError(f"Neuron type {neuron_type} not found in neuron templates.")

        template = np.asarray(neuron_type_IZ[neuron_type], dtype=float)
        params = np.zeros(self.n_params, dtype=float)

        if len(template) == 4:
            params[:9] = [0.04, template[0], template[1], template[3], 1.0, 0.0, -125.0, 30.0, template[2]]
            params[10] = 140.0
        elif len(template) == self.n_template_params:
            params[: self.n_template_params] = template
            params[10] = self.bias
        else:
            raise ValueError(
                f"Neuron template '{neuron_type}' has {len(template)} parameters, expected 4 or {self.n_template_params}."
            )

        params[9] = self.delta_V
        params[11] = self.threshold_mult
        params[12] = self.threshold_decay
        return params

    def _draw_from_spec(self, template_value, spec):
        mean = template_value if spec["mean"] is None else float(spec["mean"])
        dist = spec["dist"]

        if dist in ("fixed", "constant"):
            return mean

        if dist == "gaussian":
            std = spec["std"]
            variance = spec["variance"]
            if std is None:
                if variance is None:
                    raise ValueError("Gaussian heterogeneity requires either std or variance.")
                variance = float(variance)
                if variance < 0:
                    raise ValueError("Gaussian variance must be non-negative.")
                std = np.sqrt(variance)
            else:
                std = float(std)
                if std < 0:
                    raise ValueError("Gaussian std must be non-negative.")

            if std == 0:
                return mean

            truncate_std = spec["truncate_std"]
            if truncate_std is None:
                return float(self.rng.normal(loc=mean, scale=std))

            truncate_std = float(truncate_std)
            if truncate_std < 0:
                raise ValueError("truncate_std must be non-negative.")

            lower = mean - truncate_std * std
            upper = mean + truncate_std * std

            sample = mean
            for _ in range(1000):
                sample = float(self.rng.normal(loc=mean, scale=std))
                if lower <= sample <= upper:
                    return sample
            return float(np.clip(sample, lower, upper))

        if dist == "uniform":
            variance = spec["variance"]
            std = spec["std"]
            if std is None:
                if variance is None:
                    raise ValueError("Uniform heterogeneity requires either std or variance.")
                variance = float(variance)
                if variance < 0:
                    raise ValueError("Uniform variance must be non-negative.")
                std = np.sqrt(variance)
            else:
                std = float(std)
                if std < 0:
                    raise ValueError("Uniform std must be non-negative.")

            half_width = np.sqrt(3.0) * std
            return float(self.rng.uniform(mean - half_width, mean + half_width))

        raise ValueError(f"Unsupported heterogeneity distribution '{dist}'.")

    def _apply_heterogeneity(self, neuron_type, params):
        template_params = params.copy()
        for target in ("*", neuron_type):
            for param_idx, spec in self.heterogeneity.get(target, {}).items():
                params[param_idx] = self._draw_from_spec(template_params[param_idx], spec)
        return params

    def _build_params_from_type(self, neuron_type):
        params = self._full_params_from_type(neuron_type)
        if self.heterogeneity:
            params = self._apply_heterogeneity(neuron_type, params)
        return params

    def _set_neuron_type_metadata(self, idx, neuron_type):
        neuron_type_index = self.type_index_from_neuron_type(neuron_type)
        is_inhibitory = self.inhibitory[neuron_type_index]
        self.inhibitory_mask[idx] = is_inhibitory
        self.weight_inhibitory_mask[idx] = -1 if is_inhibitory else 1
        self.neuron_population_types[idx] = neuron_type

    def set_neuron_params_from_type(self, idx, neuron_type):
        """
        Set the parameters of a neuron given its type.
        """
        if neuron_type not in self.neuron_types:
            raise ValueError(f"Neuron type {neuron_type} not found in neuron types.")

        self.neuron_population[idx] = self._build_params_from_type(neuron_type)
        self._set_neuron_type_metadata(idx, neuron_type)

    def populate_from_probability(self, neurons_per_layer, neuron_distribution, layer_distances=None):
        """
        neurons_per_layer: list of integers, number of neurons in each layer
        neuron_distribution: list with length neurons_per_layer of arrays of length neuron_types,
            probabilities of each neuron type in each layer
        layer_distances: matrix of distances between layers, shape (n_layers, n_layers)
        """
        self.neurons_per_layer = list(neurons_per_layer)
        self.neuron_distribution = neuron_distribution
        self.layer_distances = layer_distances

        self.n_layers = len(self.neurons_per_layer)
        self._initialize_population_arrays(sum(self.neurons_per_layer))

        self.populate(self.threshold_decay, self.delta_V, self.bias, self.threshold_mult)
        if self.layer_distances is None:
            self.build_linear_layerdistance_matrix()

    def populate(self, threshold_decay, delta_V, bias, threshold_mult):
        """
        Populate the neuron population with random neurons from the neuron templates.
        """
        self.threshold_decay = threshold_decay
        self.delta_V = delta_V
        self.bias = bias
        self.threshold_mult = threshold_mult

        self.layer_indices = []
        self.neuron_population[:] = 0.0
        self.inhibitory_mask[:] = False
        self.weight_inhibitory_mask[:] = 1
        self.neuron_population_types = [None] * self.n_neurons

        for i in range(self.n_layers):
            layer_indices_layer = []
            dist_norm = np.asarray(self.neuron_distribution[i], dtype=float)
            dist_norm /= np.sum(dist_norm)

            for j in range(self.neurons_per_layer[i]):
                index = sum(self.neurons_per_layer[:i]) + j
                layer_indices_layer.append(index)
                neuron_type = self.rng.choice(self.neuron_types, p=dist_norm)
                self.neuron_population[index] = self._build_params_from_type(neuron_type)
                self._set_neuron_type_metadata(index, neuron_type)

            self.layer_indices.append(np.array(layer_indices_layer))

    def get_layer(self, neuron_index):
        """
        Get the layer of a neuron given its index.
        """
        for i in range(self.n_layers):
            if neuron_index in self.layer_indices[i]:
                return i
        return None

    def type_from_neuron_index(self, neuron_index):
        """
        Get the neuron type of a neuron given its index.
        """
        return self.neuron_population_types[neuron_index]

    def type_index_from_neuron_type(self, neuron_type):
        """
        Get the index of a neuron type given its name.
        """
        return self.neuron_types.index(neuron_type)

    def type_index_from_neuron_index(self, neuron_index):
        """
        Get the index of a neuron type given its index.
        """
        neuron_type = self.type_from_neuron_index(neuron_index)
        return self.type_index_from_neuron_type(neuron_type)

    def get_neurons_from_layer(self, layer):
        """
        Get the neurons from a layer given its index.
        """
        return self.layer_indices[layer]

    def get_neurons_from_type(self, neuron_type):
        """
        Get the neurons from a neuron type given its name.
        """
        return [i for i, ntype in enumerate(self.neuron_population_types) if ntype == neuron_type]

    def get_types_from_layer(self, layer):
        """
        Get the neuron types from a layer given its index.
        """
        return np.array(self.neuron_population_types)[self.get_neurons_from_layer(layer)]

    def build_linear_layerdistance_matrix(self, inter_distance=0.6, layer_distance=5.0):
        """
        Build a linear distance matrix for the neuron population.
        inter_distance: distance between neurons in the same layer
        layer_distance: distance between layers
        """
        self.layer_distances = np.zeros((self.n_layers, self.n_layers))

        for i in range(self.n_layers):
            for j in range(self.n_layers):
                if i == j:
                    self.layer_distances[i, j] = inter_distance
                else:
                    self.layer_distances[i, j] = layer_distance * abs(i - j)
