import numpy as np
import pytest

from backend.activations import activate, derivative, vjp
from backend.losses import binary_cross_entropy, categorical_cross_entropy, mean_squared_error
from backend.neural_network import NeuralNetwork
from backend.optimizers import Optimizer
from backend.trainer import DATASETS, TrainingSession, make_dataset


def test_activation_values_and_derivatives():
    z = np.array([[-1.0, 0.0, 2.0]])
    assert np.allclose(activate(z, "relu"), [[0, 0, 2]])
    assert np.allclose(activate(z, "leaky_relu"), [[-0.01, 0, 2]])
    assert np.allclose(derivative(z, np.tanh(z), "tanh"), 1 - np.tanh(z) ** 2)
    softmax = activate(np.array([[1.0, 2.0, 3.0]]), "softmax")
    assert np.allclose(softmax.sum(axis=1), 1.0)
    assert np.allclose(vjp(np.zeros_like(softmax), softmax, np.ones_like(softmax), "softmax"), 0)


def test_losses_match_known_values():
    y = np.array([[1.0], [0.0]])
    p = np.array([[0.8], [0.25]])
    assert binary_cross_entropy(y, p) == pytest.approx((-np.log(.8) - np.log(.75)) / 2)
    assert mean_squared_error(y, p) == pytest.approx((.2 ** 2 + .25 ** 2) / 2)
    labels = np.eye(3)[[0, 2]]
    probabilities = np.array([[.7, .2, .1], [.05, .15, .8]])
    assert categorical_cross_entropy(labels, probabilities) == pytest.approx((-np.log(.7) - np.log(.8)) / 2)


@pytest.mark.parametrize("optimizer", ["Vanilla GD", "SGD", "Momentum", "Adam"])
def test_optimizers_update_weights(optimizer):
    weights, biases = [np.array([[1.0]])], [np.array([[0.5]])]
    updater = Optimizer(optimizer, learning_rate=0.1)
    updater.update(weights, biases, [np.array([[2.0]])], [np.array([[1.0]])])
    assert weights[0][0, 0] < 1.0
    assert biases[0][0, 0] < 0.5


@pytest.mark.parametrize("optimizer", ["Vanilla GD", "SGD", "Momentum", "Adam"])
def test_training_exposes_the_actual_optimizer_parameter_change(optimizer):
    model = NeuralNetwork([2, 3, 1], optimizer=optimizer, learning_rate=0.05, seed=17)
    x = np.array([[0.3, -0.6], [0.8, 0.2]])
    y = np.array([[1.0], [0.0]])
    model.train_batch(x, y)

    update = model.last_update
    layer = update["layers"][0]
    before = np.asarray(layer["weights_before"])
    after = np.asarray(layer["weights_after"])
    gradient = np.asarray(layer["weight_gradients"])
    assert update["optimizer"] == optimizer
    assert update["step"] == 1
    assert np.allclose(after, model.weights[0])
    assert np.any(after != before)
    assert np.allclose(np.asarray(layer["biases_after"]), model.biases[0].reshape(-1))
    assert gradient.shape == before.shape


def test_forward_shapes_and_fused_multiclass_gradient():
    model = NeuralNetwork([2, 4, 3], activation="tanh", seed=3)
    x = np.array([[.1, -.2], [.4, .8]])
    y = np.eye(3)[[1, 2]]
    prediction = model.forward(x)
    assert prediction.shape == (2, 3)
    assert np.allclose(prediction.sum(axis=1), 1)
    _, _, _, delta, grad_w, grad_b = model.gradients(x, y)
    assert np.allclose(delta[-1], prediction - y)
    assert grad_w[0].shape == (2, 4) and grad_b[-1].shape == (1, 3)


def test_configurable_input_and_multiclass_output_widths_train():
    model = NeuralNetwork([16, 32, 10], activation="relu", output_activation="softmax", seed=13)
    rng = np.random.default_rng(21)
    x = rng.random((12, 16))
    labels = rng.integers(0, 10, size=12)
    y = np.eye(10)[labels]
    model.train_batch(x, y)

    assert model.forward(x).shape == (12, 10)
    assert all(np.isfinite(weight).all() for weight in model.weights)
    assert all(np.isfinite(bias).all() for bias in model.biases)


def test_failed_model_reconfiguration_does_not_corrupt_live_parameters():
    model = NeuralNetwork([2, 4, 1], seed=9)
    before = model.snapshot()

    with pytest.raises(ValueError, match="Input width"):
        model.configure([785, 4, 1])

    after = model.snapshot()
    assert after["architecture"] == before["architecture"]
    assert after["weights"] == before["weights"]
    assert after["biases"] == before["biases"]


@pytest.mark.parametrize("loss,output", [("bce", "sigmoid"), ("mse", "sigmoid"), ("categorical_cross_entropy", "softmax"), ("mse", "softmax")])
def test_finite_difference_gradient_check(loss, output):
    width = 1 if output == "sigmoid" else 3
    model = NeuralNetwork([2, 3, width], activation="tanh", output_activation=output,
                          loss=loss, learning_rate=.02, seed=11, l2=.015)
    x = np.array([[.2, -.4], [.7, .3]])
    y = np.array([[1.0], [0.0]]) if width == 1 else np.eye(3)[[0, 2]]
    result = model.gradient_check(x, y)
    assert result["passed"], result


def test_dropout_only_applies_during_training_and_roundtrip_is_safe():
    model = NeuralNetwork([2, 12, 1], dropout=.5, seed=5)
    x = np.ones((8, 2))
    evaluation_a, evaluation_b = model.forward(x), model.forward(x)
    assert np.array_equal(evaluation_a, evaluation_b)
    training_a, training_b = model._forward(x, training=True)[0], model._forward(x, training=True)[0]
    assert not np.array_equal(training_a, training_b)
    restored = NeuralNetwork.from_snapshot(model.snapshot())
    assert np.allclose(restored.forward(x), model.forward(x))
    damaged = model.snapshot()
    damaged["weights"][0][0][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        NeuralNetwork.from_snapshot(damaged)


def test_datasets_are_deterministic_and_include_required_shapes():
    assert {"XOR", "Moons"}.issubset(DATASETS)
    for name in DATASETS:
        x1, y1 = make_dataset(name, n=120, seed=42)
        x2, y2 = make_dataset(name, n=120, seed=42)
        assert x1.shape == (120, 2)
        assert y1.shape == (120, 1)
        assert np.array_equal(x1, x2) and np.array_equal(y1, y2)
    for name in ("Moons", "Spiral"):
        assert make_dataset(name, n=121, seed=8)[0].shape == (121, 2)


def test_training_records_validation_and_fixed_seed_is_reproducible():
    def run():
        model = NeuralNetwork([2, 6, 1], learning_rate=.04, seed=9)
        session = TrainingSession(model, "Linear", samples=120, seed=6, early_stopping_patience=0)
        for _ in range(8):
            session.train_one_epoch()
        return session

    first, second = run(), run()
    assert len(first.history["validation_loss"]) == 8
    assert first.train_indices.size + first.validation_indices.size == 120
    assert first.history == second.history
    assert np.array_equal(first.model.weights[0], second.model.weights[0])


def test_xor_learns_above_chance():
    model = NeuralNetwork([2, 8, 6, 1], learning_rate=.04, seed=7)
    session = TrainingSession(model, "XOR", samples=400, noise=0, seed=13, early_stopping_patience=0)
    for _ in range(240):
        session.train_one_epoch()
    assert session.history["accuracy"][-1] > .9


def test_reset_reproduces_the_seeded_initial_parameters():
    model = NeuralNetwork([2, 5, 1], seed=44)
    initial = [weight.copy() for weight in model.weights]
    model.weights[0] += 1
    model.reset()
    assert all(np.array_equal(before, after) for before, after in zip(initial, model.weights))


def test_gradient_checker_reports_relu_kink_skips():
    model = NeuralNetwork([2, 3, 1], activation="relu", seed=2)
    result = model.gradient_check(np.zeros((1, 2)), np.ones((1, 1)))
    assert "skipped_nondifferentiable" in result

