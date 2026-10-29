"""Toy dataset generation and training or validation session management."""
import math
import numpy as np
import threading
import time
from .losses import accuracy, loss_value


DATASETS = ("XOR", "Linear", "Moons", "Circles", "Spiral", "Clusters")


def class_count(name):
    return 3 if name == "Clusters" else 2


def output_width(name):
    return 1 if class_count(name) == 2 else class_count(name)


def expand_inputs(points, input_size):
    """Expand each toy data point with repeatable polynomial features."""
    points = np.asarray(points, dtype=float)
    if input_size == 1:
        return points[:, :1]
    if input_size == 2:
        return points
    x, y = points[:, 0], points[:, 1]
    features = [x, y, x * y, x ** 2, y ** 2, np.sin(np.pi * x), np.sin(np.pi * y), x ** 2 * y,
                x * y ** 2, np.cos(np.pi * x), np.cos(np.pi * y), x ** 3, y ** 3, np.sin(np.pi * (x + y)),
                np.cos(np.pi * (x - y)), x ** 2 * y ** 2]
    return np.column_stack(features[:input_size])


def make_dataset(name="XOR", n=360, seed=13, noise=0.08):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    n = max(20, min(int(n), 2000))
    rng = np.random.default_rng(seed)
    if name == "Circles":
        angles = rng.uniform(0, 2 * np.pi, n)
        labels = rng.integers(0, 2, n)
        radii = np.where(labels == 0, 0.34, 0.78) + rng.normal(0, noise, n)
        x = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    elif name == "Spiral":
        sizes = (n // 2, n - n // 2)
        parts, target = [], []
        for cls, size in enumerate(sizes):
            t = np.sqrt(rng.random(size)) * 4 * np.pi + cls * np.pi
            radius = t / (4 * np.pi)
            parts.append(np.column_stack([radius * np.cos(t), radius * np.sin(t)]))
            target.append(np.full(size, cls))
        x, labels = np.vstack(parts), np.concatenate(target)
        x += rng.normal(0, noise * 0.5, x.shape)
    elif name == "Moons":
        sizes = (n // 2, n - n // 2)
        angle_a = rng.uniform(0, np.pi, sizes[0])
        angle_b = rng.uniform(0, np.pi, sizes[1])
        upper = np.column_stack([np.cos(angle_a), np.sin(angle_a)])
        lower = np.column_stack([1 - np.cos(angle_b), 0.5 - np.sin(angle_b)])
        x = np.vstack([upper, lower]) + rng.normal(0, noise, (n, 2))
        labels = np.concatenate([np.zeros(sizes[0], dtype=int), np.ones(sizes[1], dtype=int)])
    elif name == "Linear":
        x = rng.uniform(-1.0, 1.0, (n, 2))
        labels = (x[:, 1] > 0.65 * x[:, 0] + 0.05).astype(int)
        x += rng.normal(0, noise * 0.15, x.shape)
    elif name == "Clusters":
        labels = rng.integers(0, 3, n)
        centers = np.array([[-0.65, -0.45], [0.62, -0.42], [0.02, 0.68]])
        x = centers[labels] + rng.normal(0, noise * 1.4, (n, 2))
    else:
        x = rng.uniform(-1.0, 1.0, (n, 2))
        labels = ((x[:, 0] * x[:, 1]) > 0).astype(int)
        labels = np.logical_xor(labels, rng.random(n) < min(0.25, noise * 0.5)).astype(int)
    return x.astype(float), labels.reshape(-1, 1).astype(float)


class TrainingSession:
    def __init__(self, model, dataset="XOR", batch_size=24, samples=360, noise=0.08,
                 validation_split=0.2, seed=13, early_stopping_patience=30,
                 scheduler_patience=12, scheduler_factor=0.5, input_size=2, output_size=None):
        self.model = model
        self.dataset_name = dataset
        self.samples, self.noise, self.seed = samples, noise, seed
        self.validation_split = validation_split
        self.input_size = int(input_size)
        self.output_size = int(output_size or output_width(dataset))
        self.base_points, self.y = make_dataset(dataset, samples, seed, noise)
        self.x = expand_inputs(self.base_points, self.input_size)
        self.batch_size = int(batch_size)
        self.early_stopping_patience = int(early_stopping_patience)
        self.scheduler_patience = int(scheduler_patience)
        self.scheduler_factor = float(scheduler_factor)
        self.status = "ready"
        self.epoch = 0
        self.lock = threading.RLock()
        self.worker = None
        self.max_epochs = 1000
        self.started_at = None
        self.elapsed_seconds = 0.0
        self.error = None
        self.rng = np.random.default_rng(seed + 31)
        split = max(1, min(len(self.x) - 1, int(len(self.x) * (1 - validation_split))))
        split_rng = np.random.default_rng(seed + 1)
        order = split_rng.permutation(len(self.x))
        self.train_indices, self.validation_indices = order[:split], order[split:]
        self.best_loss = float("inf")
        self.best_weights = None
        self.stale_epochs = 0
        self.scheduler_best = float("inf")
        self.scheduler_stale_epochs = 0
        self.history = {"loss": [], "accuracy": [], "validation_loss": [], "validation_accuracy": [], "learning_rate": []}
        parameter_count = sum(weight.size + bias.size for weight, bias in zip(self.model.weights, self.model.biases))
        self.replay_limit = max(2, min(61, int(6_000_000 / max(4, parameter_count * 4))) + 1)
        self.replay_stride = 1
        self.replay_frames = []
        self._record_replay_frame(force=True)

    def prepare_replay(self, max_epochs):
        self.max_epochs = int(max_epochs)
        self.replay_stride = max(1, math.ceil(max_epochs / (self.replay_limit - 1)))
        if self.epoch == 0 and self.replay_frames:
            self.replay_frames = [self.replay_frames[0]]

    def _record_replay_frame(self, force=False):
        if not force and self.epoch % self.replay_stride:
            return
        frame = {
            "epoch": self.epoch,
            "weights": [weight.astype(np.float32, copy=True) for weight in self.model.weights],
            "biases": [bias.astype(np.float32, copy=True) for bias in self.model.biases],
        }
        if self.replay_frames and self.replay_frames[-1]["epoch"] == self.epoch:
            self.replay_frames[-1] = frame
        elif len(self.replay_frames) < self.replay_limit:
            self.replay_frames.append(frame)
        elif force:
            self.replay_frames[-1] = frame

    def replay_manifest(self):
        return [{"index": index, "epoch": frame["epoch"],
                 "loss": self.history["loss"][frame["epoch"] - 1] if frame["epoch"] else None,
                 "accuracy": self.history["accuracy"][frame["epoch"] - 1] if frame["epoch"] else None}
                for index, frame in enumerate(self.replay_frames)]

    def replay_frame(self, index, sample=0):
        if not self.replay_frames:
            raise ValueError("No training replay is available")
        index = max(0, min(int(index), len(self.replay_frames) - 1))
        sample = max(0, min(int(sample), len(self.x) - 1))
        frame = self.replay_frames[index]
        current_weights, current_biases = self.model.weights, self.model.biases
        try:
            self.model.weights = [weight.astype(float) for weight in frame["weights"]]
            self.model.biases = [bias.astype(float) for bias in frame["biases"]]
            _, activations, _ = self.model.forward(self.x[sample:sample + 1], keep_cache=True)
            predictions = self.model.forward(self.x)
            replay_model = self.model.snapshot()
            grid = self.decision_grid()
            return {"index": index, "epoch": frame["epoch"], "model": replay_model,
                    "sample_activations": [layer.reshape(-1).tolist() for layer in activations],
                    "predictions": np.round(predictions[:, 0], 4).tolist() if predictions.shape[1] == 1 else np.round(predictions, 4).tolist(),
                    "predicted_classes": (predictions[:, 0] >= 0.5).astype(int).tolist() if predictions.shape[1] == 1 else np.argmax(predictions, axis=1).tolist(),
                    "decision_boundary": grid, "history": {key: values[:frame["epoch"]] for key, values in self.history.items()},
                    "last_update": None}
        finally:
            self.model.weights, self.model.biases = current_weights, current_biases

    def targets(self, indices):
        labels = self.y[indices].astype(int).reshape(-1)
        if self.model.architecture[-1] == 1:
            return labels.reshape(-1, 1).astype(float)
        result = np.zeros((len(labels), self.model.architecture[-1]), dtype=float)
        result[np.arange(len(labels)), labels] = 1.0
        return result

    def train_one_epoch(self):
        with self.lock:
            order = self.rng.permutation(self.train_indices)
            for start in range(0, len(order), self.batch_size):
                indices = order[start:start + self.batch_size]
                self.model.train_batch(self.x[indices], self.targets(indices))
            train_prediction = self.model.forward(self.x[self.train_indices])
            validation_prediction = self.model.forward(self.x[self.validation_indices])
            train_target, validation_target = self.targets(self.train_indices), self.targets(self.validation_indices)
            self.epoch += 1
            train_loss = loss_value(train_target, train_prediction, self.model.loss_name)
            val_loss = loss_value(validation_target, validation_prediction, self.model.loss_name)
            self.history["loss"].append(train_loss)
            self.history["accuracy"].append(accuracy(train_target, train_prediction))
            self.history["validation_loss"].append(val_loss)
            self.history["validation_accuracy"].append(accuracy(validation_target, validation_prediction))
            if val_loss < self.scheduler_best:
                self.scheduler_best, self.scheduler_stale_epochs = val_loss, 0
            else:
                self.scheduler_stale_epochs += 1
            if self.scheduler_patience and self.scheduler_stale_epochs >= self.scheduler_patience:
                reduced = max(1e-4, self.model.learning_rate * self.scheduler_factor)
                self.model.learning_rate = reduced
                self.model.optimizer.learning_rate = reduced
                self.scheduler_stale_epochs = 0
            self.history["learning_rate"].append(self.model.learning_rate)
            if val_loss < self.best_loss:
                self.best_loss, self.stale_epochs = val_loss, 0
                self.best_weights = ([w.copy() for w in self.model.weights], [b.copy() for b in self.model.biases])
            else:
                self.stale_epochs += 1
            if self.early_stopping_patience and self.stale_epochs >= self.early_stopping_patience:
                if self.best_weights:
                    self.model.weights, self.model.biases = self.best_weights
                    self.model.last_update = None
                    train_prediction = self.model.forward(self.x[self.train_indices])
                    validation_prediction = self.model.forward(self.x[self.validation_indices])
                    self.history["loss"][-1] = loss_value(train_target, train_prediction, self.model.loss_name)
                    self.history["accuracy"][-1] = accuracy(train_target, train_prediction)
                    self.history["validation_loss"][-1] = loss_value(validation_target, validation_prediction, self.model.loss_name)
                    self.history["validation_accuracy"][-1] = accuracy(validation_target, validation_prediction)
                self.status = "complete"
            self._record_replay_frame(force=self.status == "complete" or self.epoch >= self.max_epochs)
            return self.history["loss"][-1], self.history["accuracy"][-1]

    def reset(self):
        with self.lock:
            self.model.reset()
            self.epoch = 0
            self.rng = np.random.default_rng(self.seed + 31)
            split_rng = np.random.default_rng(self.seed + 1)
            order = split_rng.permutation(len(self.x))
            split = max(1, min(len(self.x) - 1, int(len(self.x) * (1 - self.validation_split))))
            self.train_indices, self.validation_indices = order[:split], order[split:]
            self.history = {"loss": [], "accuracy": [], "validation_loss": [], "validation_accuracy": [], "learning_rate": []}
            self.best_loss, self.best_weights, self.stale_epochs = float("inf"), None, 0
            self.scheduler_best, self.scheduler_stale_epochs = float("inf"), 0
            self.status, self.error = "ready", None
            self.started_at, self.elapsed_seconds = None, 0.0
            self.replay_frames = []
            self.replay_stride = 1
            self._record_replay_frame(force=True)

    def evaluation(self, index):
        target = self.targets(index)
        prediction = self.model.forward(self.x[index])
        return {
            "loss": loss_value(target, prediction, self.model.loss_name),
            "accuracy": accuracy(target, prediction),
            "predictions": prediction,
            "targets": target,
        }

    def decision_grid(self, resolution=38):
        xs = np.linspace(-1.2, 1.2, resolution)
        xx, yy = np.meshgrid(xs, xs)
        points = expand_inputs(np.column_stack([xx.ravel(), yy.ravel()]), self.input_size)
        probabilities = self.model.forward(points)
        if probabilities.shape[1] == 1:
            z = probabilities[:, 0].reshape(resolution, resolution)
            classes = (z >= 0.5).astype(int)
        else:
            z = probabilities[:, 1].reshape(resolution, resolution)
            classes = np.argmax(probabilities, axis=1).reshape(resolution, resolution)
        return {"x": xs.tolist(), "y": xs.tolist(), "z": np.round(z, 4).tolist(), "classes": classes.tolist()}
