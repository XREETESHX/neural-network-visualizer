"""A fully connected neural network whose calculations can be inspected."""
import numpy as np
from .activations import activate, derivative, vjp
from .losses import accuracy, loss_value
from .optimizers import Optimizer


class NeuralNetwork:
    """A dense network; batches are rows and weights have shape (inputs, outputs)."""

    def __init__(self, architecture=None, activation="tanh", learning_rate=0.04,
                 optimizer="Adam", seed=7, output_activation=None, loss=None,
                 initialization="auto", dropout=0.0, l2=0.0):
        self.architecture = list(architecture or [2, 5, 4, 1])
        self.activation = activation
        self.output_activation = output_activation or ("sigmoid" if self.architecture[-1] == 1 else "softmax")
        self.loss_name = loss or ("bce" if self.output_activation == "sigmoid" else "categorical_cross_entropy")
        self.learning_rate = float(learning_rate)
        self.optimizer_name = optimizer
        self.initialization = initialization
        self.dropout = float(dropout)
        self.l2 = float(l2)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self._validate()
        self.weights, self.biases = [], []
        self.reset()

    def _validate(self):
        if len(self.architecture) < 2 or len(self.architecture) > 6 or any(not isinstance(n, int) or n < 1 for n in self.architecture):
            raise ValueError("Use an input layer, up to four hidden layers, and a positive output width")
        if self.architecture[0] > 16 or self.architecture[-1] > 10 or any(width > 128 for width in self.architecture[1:-1]):
            raise ValueError("Input width is limited to 16, hidden layers to 128, and output width to 10")
        if self.activation not in {"relu", "leaky_relu", "tanh", "sigmoid"}:
            raise ValueError("Unsupported hidden activation")
        if self.output_activation not in {"sigmoid", "softmax"}:
            raise ValueError("Output activation must be sigmoid or softmax")
        if self.loss_name not in {"bce", "mse", "categorical_cross_entropy"}:
            raise ValueError("Unsupported loss")
        if self.optimizer_name not in {"Vanilla GD", "SGD", "Momentum", "Adam"}:
            raise ValueError("Unsupported optimizer")
        if self.initialization not in {"auto", "normal", "xavier", "he"}:
            raise ValueError("Initialization must be auto, normal, xavier, or he")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Learning rate must be a finite positive number")
        if self.loss_name == "bce" and self.output_activation != "sigmoid":
            raise ValueError("Binary cross-entropy requires sigmoid output")
        if self.loss_name == "categorical_cross_entropy" and self.output_activation != "softmax":
            raise ValueError("Categorical cross-entropy requires softmax output")
        if self.loss_name == "categorical_cross_entropy" and self.architecture[-1] < 2:
            raise ValueError("Categorical output requires at least two neurons")
        if not np.isfinite(self.dropout) or not 0 <= self.dropout < 0.8 or not np.isfinite(self.l2) or self.l2 < 0:
            raise ValueError("Dropout must be in [0, 0.8) and L2 must be non-negative")

    def reset(self):
        self.rng = np.random.default_rng(self.seed)
        self.weights, self.biases = [], []
        self.last_update = None
        method = self.initialization
        if method == "auto":
            method = "he" if self.activation in {"relu", "leaky_relu"} else "xavier"
        if method not in {"normal", "xavier", "he"}:
            raise ValueError("Initialization must be normal, xavier, he, or auto")
        for fan_in, fan_out in zip(self.architecture[:-1], self.architecture[1:]):
            if method == "he":
                scale = np.sqrt(2.0 / fan_in)
            elif method == "xavier":
                scale = np.sqrt(2.0 / (fan_in + fan_out))
            else:
                scale = 0.1
            self.weights.append(self.rng.normal(0, scale, (fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out), dtype=float))
        self.optimizer = Optimizer(self.optimizer_name, self.learning_rate)

    def configure(self, architecture=None, activation=None, learning_rate=None, optimizer=None,
                  output_activation=None, loss=None, initialization=None, dropout=None, l2=None, seed=None):
        old_shape = self.architecture[:]
        old_config = (self.architecture[:], self.activation, self.output_activation, self.loss_name,
                      self.initialization, self.dropout, self.l2, self.seed, self.learning_rate,
                      self.optimizer_name)
        old_activation = (self.activation, self.output_activation, self.loss_name, self.initialization, self.seed)
        if architecture is not None:
            self.architecture = list(architecture)
        if activation is not None:
            self.activation = activation
        if output_activation is not None:
            self.output_activation = output_activation
        elif architecture is not None and self.architecture[-1] != old_shape[-1]:
            self.output_activation = "sigmoid" if self.architecture[-1] == 1 else "softmax"
        if loss is not None:
            self.loss_name = loss
        elif output_activation is not None:
            self.loss_name = "bce" if output_activation == "sigmoid" else "categorical_cross_entropy"
        if initialization is not None:
            self.initialization = initialization
        if dropout is not None:
            self.dropout = float(dropout)
        if l2 is not None:
            self.l2 = float(l2)
        if seed is not None:
            self.seed = int(seed)
        if learning_rate is not None:
            self.learning_rate = float(learning_rate)
        if optimizer is not None:
            self.optimizer_name = optimizer
        try:
            self._validate()
        except Exception:
            (self.architecture, self.activation, self.output_activation, self.loss_name,
             self.initialization, self.dropout, self.l2, self.seed, self.learning_rate,
             self.optimizer_name) = old_config
            raise
        if self.architecture != old_shape or (self.activation, self.output_activation, self.loss_name, self.initialization, self.seed) != old_activation:
            self.reset()
        else:
            self.optimizer = Optimizer(self.optimizer_name, self.learning_rate)
            self.last_update = None

    def _forward(self, x, training=False):
        x = np.asarray(x, dtype=float)
        activations, preactivations, masks = [x], [], []
        current = x
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = current @ weight + bias
            preactivations.append(z)
            name = self.output_activation if index == len(self.weights) - 1 else self.activation
            current = activate(z, name)
            mask = None
            if training and index < len(self.weights) - 1 and self.dropout:
                mask = (self.rng.random(current.shape) >= self.dropout).astype(float) / (1.0 - self.dropout)
                current = current * mask
            masks.append(mask)
            activations.append(current)
        return current, activations, preactivations, masks

    def forward(self, x, keep_cache=False, training=False):
        prediction, activations, preactivations, _ = self._forward(x, training)
        if keep_cache:
            return prediction, activations, preactivations
        return prediction

    def loss(self, x, y):
        prediction = self.forward(x)
        return loss_value(y, prediction, self.loss_name) + 0.5 * self.l2 * sum(float(np.sum(w ** 2)) for w in self.weights)

    def gradients(self, x, y, training=False):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        prediction, acts, zs, masks = self._forward(x, training=training)
        batch_size = len(x)
        deltas = [None] * len(self.weights)
        if self.loss_name in {"bce", "categorical_cross_entropy"}:
            # For sigmoid with binary cross-entropy, or softmax with categorical
            # cross-entropy, the output gradient simplifies to prediction minus target.
            deltas[-1] = prediction - y
        else:
            upstream = 2.0 * (prediction - y) / prediction.shape[1]
            deltas[-1] = vjp(zs[-1], prediction, upstream, self.output_activation)
        for layer in range(len(self.weights) - 2, -1, -1):
            propagated = deltas[layer + 1] @ self.weights[layer + 1].T
            deltas[layer] = vjp(zs[layer], acts[layer + 1], propagated, self.activation)
            if masks[layer] is not None:
                deltas[layer] *= masks[layer]
        grad_w = [acts[i].T @ deltas[i] / batch_size + self.l2 * self.weights[i] for i in range(len(self.weights))]
        grad_b = [np.mean(deltas[i], axis=0, keepdims=True) for i in range(len(self.weights))]
        return prediction, acts, zs, deltas, grad_w, grad_b

    def train_batch(self, x, y, training=True):
        prediction, _, _, _, grad_w, grad_b = self.gradients(x, y, training=training)
        value = loss_value(y, prediction, self.loss_name) + 0.5 * self.l2 * sum(float(np.sum(w ** 2)) for w in self.weights)
        score = accuracy(y, prediction)
        before_w = [weight.copy() for weight in self.weights]
        before_b = [bias.copy() for bias in self.biases]
        self.optimizer.update(self.weights, self.biases, grad_w, grad_b)
        self.last_update = {
            "optimizer": self.optimizer_name,
            "learning_rate": self.optimizer.learning_rate,
            "step": self.optimizer.step_count,
            "layers": [
                {
                    "weights_before": before_w[i].tolist(),
                    "weight_gradients": grad_w[i].tolist(),
                    "weights_after": self.weights[i].tolist(),
                    "biases_before": before_b[i].reshape(-1).tolist(),
                    "bias_gradients": grad_b[i].reshape(-1).tolist(),
                    "biases_after": self.biases[i].reshape(-1).tolist(),
                }
                for i in range(len(self.weights))
            ],
        }
        return value, score

    def snapshot(self):
        return {
            "format": "neural-network-lab/model-v1",
            "architecture": self.architecture,
            "activation": self.activation,
            "output_activation": self.output_activation,
            "loss": self.loss_name,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer_name,
            "initialization": self.initialization,
            "dropout": self.dropout,
            "l2": self.l2,
            "seed": self.seed,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.reshape(-1).tolist() for b in self.biases],
        }

    @classmethod
    def from_snapshot(cls, data):
        if data.get("format") != "neural-network-lab/model-v1":
            raise ValueError("Unsupported model file format")
        model = cls(
            architecture=data["architecture"], activation=data["activation"],
            learning_rate=data["learning_rate"], optimizer=data["optimizer"], seed=data.get("seed", 7),
            output_activation=data["output_activation"], loss=data["loss"],
            initialization=data.get("initialization", "auto"), dropout=data.get("dropout", 0), l2=data.get("l2", 0),
        )
        weights = [np.asarray(w, dtype=float) for w in data["weights"]]
        biases = [np.asarray(b, dtype=float).reshape(1, -1) for b in data["biases"]]
        if len(weights) != len(model.weights) or len(biases) != len(model.biases):
            raise ValueError("Model parameter count does not match architecture")
        for i, (weight, bias) in enumerate(zip(weights, biases)):
            if weight.shape != model.weights[i].shape or bias.shape != model.biases[i].shape:
                raise ValueError("Model parameter shape does not match architecture")
            if not np.all(np.isfinite(weight)) or not np.all(np.isfinite(bias)):
                raise ValueError("Model parameters must be finite numbers")
        model.weights, model.biases = weights, biases
        return model

    def forward_trace(self, x):
        prediction, activations, preactivations, _ = self._forward(np.asarray(x, dtype=float), training=False)
        layers = []
        for i, z in enumerate(preactivations):
            layers.append({
                "layer": i + 1,
                "input": activations[i].tolist(),
                "weights": self.weights[i].tolist(),
                "biases": self.biases[i].reshape(-1).tolist(),
                "preactivation": z.tolist(),
                "activation": self.output_activation if i == len(preactivations) - 1 else self.activation,
                "output": activations[i + 1].tolist(),
            })
        return {"input": np.asarray(x, dtype=float).tolist(), "prediction": prediction.tolist(), "layers": layers}

    def backprop_detail(self, x, y):
        prediction, acts, zs, deltas, grad_w, grad_b = self.gradients(x, y)
        rows = []
        for layer in range(len(self.weights) - 1, -1, -1):
            label = "output" if layer == len(self.weights) - 1 else f"hidden {layer + 1}"
            activation_name = self.output_activation if layer == len(self.weights) - 1 else self.activation
            if layer == len(self.weights) - 1:
                rule = "δ = ŷ − y" if self.loss_name != "mse" else f"δ = (ŷ − y) ⊙ f′(z{layer + 1})"
                if self.loss_name in {"bce", "categorical_cross_entropy"}:
                    upstream_error = prediction
                    error = prediction - y
                    activation_slope = None
                    calculation = "Fused output/loss derivative: δ = ŷ − y"
                else:
                    upstream_error = 2.0 * (prediction - y) / prediction.shape[1]
                    error = upstream_error
                    activation_slope = None if activation_name == "softmax" else derivative(zs[layer], acts[layer + 1], activation_name)
                    calculation = "Softmax Jacobian-vector product" if activation_name == "softmax" else "δ = upstream error × output activation slope"
            else:
                rule = f"δ{layer + 1} = (W{layer + 2} · δ{layer + 2}) ⊙ f′(z{layer + 1})"
                upstream_error = deltas[layer + 1] @ self.weights[layer + 1].T
                error = upstream_error
                activation_slope = derivative(zs[layer], acts[layer + 1], self.activation)
                calculation = "δ = (next-layer δ × outgoing weights) × activation slope"
            rows.append({
                "layer": label, "rule": rule,
                "delta": np.round(deltas[layer].reshape(-1), 5).tolist(),
                "upstream_error": np.round(np.asarray(error).reshape(-1), 5).tolist(),
                "activation_slope": None if activation_slope is None else np.round(activation_slope.reshape(-1), 5).tolist(),
                "calculation": calculation,
                "gradient_weights": np.round(grad_w[layer], 5).tolist(),
                "gradient_biases": np.round(grad_b[layer].reshape(-1), 5).tolist(),
                "activation": np.round(acts[layer + 1].reshape(-1), 5).tolist(),
                "preactivation": np.round(zs[layer].reshape(-1), 5).tolist(),
                "weights": np.round(self.weights[layer], 5).tolist(),
                "biases": np.round(self.biases[layer].reshape(-1), 5).tolist(),
            })
        return {
            "x": np.asarray(x).reshape(-1).tolist(), "y": np.asarray(y).reshape(-1).tolist(),
            "prediction": prediction.reshape(-1).tolist(), "loss": self.loss(x, y),
            "forward": [np.round(a.reshape(-1), 5).tolist() for a in acts],
            "backward": rows,
            "chain_rule": "dW = A_prev.T @ dZ / batch_size + l2 * W; db = mean(dZ, axis=0)",
        }

    def gradient_check(self, x, y, epsilon=1e-5):
        """Check each weight and bias gradient using small numerical changes."""
        if self.dropout:
            raise ValueError("Disable dropout before gradient checking")
        _, _, _, _, grad_w, grad_b = self.gradients(x, y, training=False)
        _, _, base_zs, _ = self._forward(x, training=False)
        report, max_abs, max_rel, skipped = [], 0.0, 0.0, 0
        for group_name, parameters, analytic in (("weights", self.weights, grad_w), ("biases", self.biases, grad_b)):
            for layer, parameter in enumerate(parameters):
                for index in np.ndindex(parameter.shape):
                    original = parameter[index]
                    step = epsilon
                    stable = False
                    for _ in range(12):
                        parameter[index] = original + step
                        plus, _, plus_zs, _ = self._forward(x, training=False)
                        plus = loss_value(y, plus, self.loss_name) + 0.5 * self.l2 * sum(float(np.sum(w ** 2)) for w in self.weights)
                        parameter[index] = original - step
                        minus, _, minus_zs, _ = self._forward(x, training=False)
                        minus = loss_value(y, minus, self.loss_name) + 0.5 * self.l2 * sum(float(np.sum(w ** 2)) for w in self.weights)
                        parameter[index] = original
                        if self.activation not in {"relu", "leaky_relu"} or all(
                            np.all(np.sign(base_zs[layer]) == np.sign(plus_zs[layer]))
                            and np.all(np.sign(base_zs[layer]) == np.sign(minus_zs[layer]))
                            and not np.any(base_zs[layer] == 0)
                            for layer in range(len(base_zs) - 1)
                        ):
                            stable = True
                            break
                        step *= 0.5
                    if not stable:
                        skipped += 1
                        continue
                    numeric = (plus - minus) / (2 * step)
                    exact = float(analytic[layer][index])
                    absolute = abs(exact - numeric)
                    relative = absolute / max(1e-8, abs(exact) + abs(numeric))
                    max_abs, max_rel = max(max_abs, absolute), max(max_rel, relative)
                    report.append({"parameter": group_name, "layer": layer, "index": list(index), "analytical": exact, "numerical": numeric, "absolute_error": absolute, "relative_error": relative})
        return {"passed": bool(report) and (max_abs < 1e-5 or max_rel < 1e-4), "max_absolute_error": max_abs, "max_relative_error": max_rel, "skipped_nondifferentiable": skipped, "checks": report}
