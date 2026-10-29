"""Parameter updates for gradient descent, momentum SGD, and Adam."""
import numpy as np


class Optimizer:
    def __init__(self, name="Adam", learning_rate=0.04, momentum=0.9):
        self.name = name
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.step_count = 0
        self.m = None
        self.v = None
        self.velocity = None

    def update(self, weights, biases, grad_w, grad_b):
        self.step_count += 1
        if self.name in {"SGD", "Vanilla GD"}:
            for i in range(len(weights)):
                weights[i] -= self.learning_rate * grad_w[i]
                biases[i] -= self.learning_rate * grad_b[i]
            return
        if self.name == "Momentum":
            if self.velocity is None:
                self.velocity = ([np.zeros_like(x) for x in weights], [np.zeros_like(x) for x in biases])
            for group, grads in ((0, grad_w), (1, grad_b)):
                params = weights if group == 0 else biases
                for i, grad in enumerate(grads):
                    self.velocity[group][i] = self.momentum * self.velocity[group][i] + grad
                    params[i] -= self.learning_rate * self.velocity[group][i]
            return
        if self.name != "Adam":
            raise ValueError(f"Unsupported optimizer: {self.name}")
        if self.m is None:
            self.m = ([np.zeros_like(x) for x in weights], [np.zeros_like(x) for x in biases])
            self.v = ([np.zeros_like(x) for x in weights], [np.zeros_like(x) for x in biases])
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for group, grads in ((0, grad_w), (1, grad_b)):
            params = weights if group == 0 else biases
            for i, grad in enumerate(grads):
                self.m[group][i] = beta1 * self.m[group][i] + (1 - beta1) * grad
                self.v[group][i] = beta2 * self.v[group][i] + (1 - beta2) * grad ** 2
                m_hat = self.m[group][i] / (1 - beta1 ** self.step_count)
                v_hat = self.v[group][i] / (1 - beta2 ** self.step_count)
                params[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    def state_dict(self):
        def encode(pair):
            return None if pair is None else [[value.tolist() for value in group] for group in pair]
        return {
            "name": self.name,
            "learning_rate": self.learning_rate,
            "step_count": self.step_count,
            "momentum": self.momentum,
            "m": encode(self.m),
            "v": encode(self.v),
            "velocity": encode(self.velocity),
        }
