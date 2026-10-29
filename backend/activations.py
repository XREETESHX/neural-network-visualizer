"""Activation functions and derivatives that work on batches of values."""
import numpy as np


def activate(z, name):
    if name == "relu":
        return np.maximum(0.0, z)
    if name == "leaky_relu":
        return np.where(z > 0, z, 0.01 * z)
    if name == "tanh":
        return np.tanh(z)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
    if name == "softmax":
        shifted = z - np.max(z, axis=1, keepdims=True)
        ex = np.exp(shifted)
        return ex / np.sum(ex, axis=1, keepdims=True)
    raise ValueError(f"Unsupported activation: {name}")


def derivative(z, a, name):
    if name == "relu":
        return (z > 0).astype(float)
    if name == "leaky_relu":
        return np.where(z > 0, 1.0, 0.01)
    if name == "tanh":
        return 1.0 - a ** 2
    if name == "sigmoid":
        return a * (1.0 - a)
    if name == "softmax":
        raise ValueError("Softmax has a Jacobian; use vjp() to multiply it by an upstream gradient")
    raise ValueError(f"Unsupported activation: {name}")


def vjp(z, a, upstream, name):
    """Calculate dL/dZ from the incoming gradient dL/dA."""
    if name == "softmax":
        return a * (upstream - np.sum(upstream * a, axis=1, keepdims=True))
    return upstream * derivative(z, a, name)
