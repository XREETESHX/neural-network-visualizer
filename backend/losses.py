"""Loss functions that average across examples and output values."""
import numpy as np


def binary_cross_entropy(y, probabilities):
    p = np.clip(probabilities, 1e-8, 1.0 - 1e-8)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def accuracy(y, probabilities):
    if probabilities.shape[1] == 1:
        return float(np.mean((probabilities >= 0.5) == (y >= 0.5)))
    return float(np.mean(np.argmax(probabilities, axis=1) == np.argmax(y, axis=1)))


def mean_squared_error(y, prediction):
    return float(np.mean((prediction - y) ** 2))


def categorical_cross_entropy(y, probabilities):
    p = np.clip(probabilities, 1e-8, 1.0)
    return float(-np.mean(np.sum(y * np.log(p), axis=1)))


def loss_value(y, prediction, name):
    if name == "bce":
        return binary_cross_entropy(y, prediction)
    if name == "mse":
        return mean_squared_error(y, prediction)
    if name == "categorical_cross_entropy":
        return categorical_cross_entropy(y, prediction)
    raise ValueError(f"Unsupported loss: {name}")
