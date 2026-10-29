"""HTTP routes for configuring, training, and inspecting neural networks."""
from pathlib import Path
from typing import Literal
from uuid import uuid4
import asyncio
import logging
import os
import threading
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .neural_network import NeuralNetwork
from .trainer import DATASETS, TrainingSession, class_count, output_width


logger = logging.getLogger(__name__)
sessions: dict[str, TrainingSession] = {}
MAX_SESSIONS = 32


@asynccontextmanager
async def app_lifespan(_app):
    yield
    workers = []
    for session in sessions.values():
        with session.lock:
            if session.status in {"running", "paused"}:
                session.status = "stopped"
            if session.worker is not None:
                workers.append(session.worker)
    for worker in workers:
        if worker.is_alive():
            worker.join(timeout=2)


app = FastAPI(title="Neural Network Lab API", version="1.0.0", lifespan=app_lifespan)
allowed_origins = [origin.strip() for origin in os.getenv(
    "CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
).split(",") if origin.strip()]
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_credentials=False,
                   allow_methods=["GET", "POST", "PUT", "PATCH", "OPTIONS"], allow_headers=["Content-Type"])


@app.middleware("http")
async def enforce_request_size(request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > 2_000_000:
        return JSONResponse(status_code=413, content={"detail": "Request body must be smaller than 2 MB"})
    return await call_next(request)


class SessionConfig(BaseModel):
    hidden_layers: list[int] = Field(default_factory=lambda: [5, 4], min_length=0, max_length=4)
    input_size: int = Field(default=2, ge=1, le=16)
    output_size: int = Field(default=1, ge=1, le=10)
    activation: Literal["relu", "leaky_relu", "tanh", "sigmoid"] = "tanh"
    learning_rate: float = Field(default=0.04, gt=0.0001, le=1.0)
    optimizer: Literal["Adam", "SGD", "Momentum", "Vanilla GD"] = "Adam"
    batch_size: int = Field(default=24, ge=4, le=128)
    dataset: Literal["XOR", "Linear", "Moons", "Circles", "Spiral", "Clusters"] = "XOR"
    samples: int = Field(default=360, ge=20, le=2000)
    noise: float = Field(default=0.08, ge=0, le=0.5)
    validation_split: float = Field(default=0.2, ge=0.05, le=0.4)
    seed: int = Field(default=13, ge=0, le=1000000)
    initialization: Literal["auto", "normal", "xavier", "he"] = "auto"
    dropout: float = Field(default=0, ge=0, lt=0.8)
    l2: float = Field(default=0, ge=0, le=1)
    early_stopping_patience: int = Field(default=30, ge=0, le=500)
    scheduler_patience: int = Field(default=12, ge=0, le=500)
    scheduler_factor: float = Field(default=0.5, ge=0.1, lt=1)


class UpdateConfig(BaseModel):
    hidden_layers: list[int] = Field(min_length=0, max_length=4)
    input_size: int = Field(default=2, ge=1, le=16)
    output_size: int = Field(default=1, ge=1, le=10)
    activation: Literal["relu", "leaky_relu", "tanh", "sigmoid"]
    learning_rate: float = Field(gt=0.0001, le=1.0)
    optimizer: Literal["Adam", "SGD", "Momentum", "Vanilla GD"]
    batch_size: int = Field(ge=4, le=128)
    dataset: Literal["XOR", "Linear", "Moons", "Circles", "Spiral", "Clusters"]
    samples: int = Field(default=360, ge=20, le=2000)
    noise: float = Field(default=0.08, ge=0, le=0.5)
    validation_split: float = Field(default=0.2, ge=0.05, le=0.4)
    seed: int = Field(default=13, ge=0, le=1000000)
    initialization: Literal["auto", "normal", "xavier", "he"] = "auto"
    dropout: float = Field(default=0, ge=0, lt=0.8)
    l2: float = Field(default=0, ge=0, le=1)
    early_stopping_patience: int = Field(default=30, ge=0, le=500)
    scheduler_patience: int = Field(default=12, ge=0, le=500)
    scheduler_factor: float = Field(default=0.5, ge=0.1, lt=1)


class TrainRequest(BaseModel):
    max_epochs: int = Field(default=1000, ge=1, le=10000)


class PredictionRequest(BaseModel):
    inputs: list[list[float]] = Field(min_length=1, max_length=500)


class ModelImportRequest(BaseModel):
    model: dict
    training: dict = Field(default_factory=dict)


def get_session(session_id: str) -> TrainingSession:
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Training session not found")
    return session


def make_session(config: SessionConfig) -> TrainingSession:
    validate_architecture(config.dataset, config.input_size, config.output_size)
    architecture = [config.input_size, *config.hidden_layers, config.output_size]
    if any(size < 2 or size > 128 for size in config.hidden_layers):
        raise HTTPException(status_code=422, detail="Hidden layer widths must be between 2 and 128")
    output_activation = "sigmoid" if architecture[-1] == 1 else "softmax"
    model = NeuralNetwork(architecture, config.activation, config.learning_rate, config.optimizer,
                          seed=config.seed, output_activation=output_activation,
                          initialization=config.initialization, dropout=config.dropout, l2=config.l2)
    return TrainingSession(model, config.dataset, config.batch_size, config.samples, config.noise,
                           config.validation_split, config.seed, config.early_stopping_patience,
                           config.scheduler_patience, config.scheduler_factor,
                           config.input_size, config.output_size)


def validate_architecture(dataset, input_size, output_size):
    classes = class_count(dataset)
    if input_size > 16:
        raise HTTPException(status_code=422, detail="Toy datasets support 1–16 input features")
    if output_size < classes and not (classes == 2 and output_size == 1):
        raise HTTPException(status_code=422, detail=f"{dataset} needs at least {classes} outputs for its classes (binary datasets also support one sigmoid output)")


def session_payload(session_id: str, session: TrainingSession):
    with session.lock:
        return _session_payload_locked(session_id, session)


def _session_payload_locked(session_id: str, session: TrainingSession):
    probabilities = session.model.forward(session.x)
    _, sample_activations, _ = session.model.forward(session.x[:1], keep_cache=True)
    return {
        "session_id": session_id,
        "status": session.status,
        "epoch": session.epoch,
        "elapsed_seconds": round(session.elapsed_seconds + (time.monotonic() - session.started_at if session.started_at and session.status == "running" else 0), 3),
        "error": session.error,
        "batch_size": session.batch_size,
        "dataset": session.dataset_name,
        "input_size": session.input_size,
        "output_size": session.output_size,
        "samples": session.samples,
        "noise": session.noise,
        "validation_split": session.validation_split,
        "seed": session.seed,
        "history": session.history,
        "model": session.model.snapshot(),
        "metrics": {
            "loss": session.history["loss"][-1] if session.history["loss"] else None,
            "accuracy": session.history["accuracy"][-1] if session.history["accuracy"] else None,
            "validation_loss": session.history["validation_loss"][-1] if session.history["validation_loss"] else None,
            "validation_accuracy": session.history["validation_accuracy"][-1] if session.history["validation_accuracy"] else None,
        },
        "points": session.base_points.tolist(),
        "labels": session.y.reshape(-1).astype(int).tolist(),
        "predictions": np_round(probabilities[:, 0]) if probabilities.shape[1] == 1 else np.round(probabilities, 4).tolist(),
        "predicted_classes": (probabilities[:, 0] >= 0.5).astype(int).tolist() if probabilities.shape[1] == 1 else np.argmax(probabilities, axis=1).tolist(),
        "train_indices": session.train_indices.tolist(),
        "validation_indices": session.validation_indices.tolist(),
        "sample_activations": [np_round(layer.reshape(-1)) for layer in sample_activations],
        "last_update": session.model.last_update,
        "decision_boundary": session.decision_grid(),
        "sample_count": len(session.x),
    }


def np_round(values):
    return [round(float(value), 4) for value in values]


@app.get("/api/health")
def health():
    return {"status": "ok", "engine": "NumPy", "sessions": len(sessions)}


@app.get("/api/datasets")
def list_datasets():
    return {"datasets": [{"name": name, "classes": class_count(name), "dimensions": 2} for name in DATASETS]}


@app.post("/api/sessions")
def create_session(config: SessionConfig):
    if len(sessions) >= MAX_SESSIONS:
        raise HTTPException(status_code=429, detail="Session limit reached; restart the server to clear sessions")
    session_id = str(uuid4())
    try:
        session = make_session(config)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    sessions[session_id] = session
    return session_payload(session_id, session)


@app.get("/api/sessions/{session_id}")
def get_state(session_id: str):
    return session_payload(session_id, get_session(session_id))


@app.post("/api/sessions/{session_id}/predict")
def predict(session_id: str, request: PredictionRequest):
    session = get_session(session_id)
    expected_inputs = session.model.architecture[0]
    if any(len(row) != expected_inputs for row in request.inputs):
        raise HTTPException(status_code=422, detail=f"Each prediction input must contain exactly {expected_inputs} values")
    inputs = np.array(request.inputs, dtype=float)
    if not np.all(np.isfinite(inputs)):
        raise HTTPException(status_code=422, detail="Prediction inputs must be finite numbers")
    with session.lock:
        predictions = session.model.forward(inputs)
    return {"predictions": np_round(predictions[:, 0]) if predictions.shape[1] == 1 else np.round(predictions, 5).tolist(),
            "classes": (predictions[:, 0] >= 0.5).astype(int).tolist() if predictions.shape[1] == 1 else np.argmax(predictions, axis=1).tolist()}


@app.put("/api/sessions/{session_id}/config")
def update_config(session_id: str, config: UpdateConfig):
    session = get_session(session_id)
    validate_architecture(config.dataset, config.input_size, config.output_size)
    if any(size < 2 or size > 128 for size in config.hidden_layers):
        raise HTTPException(status_code=422, detail="Hidden layer widths must be between 2 and 128")
    architecture = [config.input_size, *config.hidden_layers, config.output_size]
    output_activation = "sigmoid" if architecture[-1] == 1 else "softmax"
    with session.lock:
        if session.status == "running":
            raise HTTPException(status_code=409, detail="Pause training before changing configuration")
        # A paused worker is still waiting for a resume signal. Stop and wait for
        # it before replacing the session, so it cannot update stale state.
        if session.status == "paused":
            session.status = "stopped"
            session.elapsed_seconds += time.monotonic() - session.started_at if session.started_at else 0
            session.started_at = None
        worker = session.worker
    if worker is not None and worker.is_alive():
        worker.join(timeout=2)
        if worker.is_alive():
            raise HTTPException(status_code=409, detail="Training worker is still stopping; retry configuration")
    try:
        candidate_model = NeuralNetwork.from_snapshot(session.model.snapshot())
        candidate_model.configure(architecture, config.activation, config.learning_rate, config.optimizer,
                                  output_activation=output_activation, initialization=config.initialization,
                                  dropout=config.dropout, l2=config.l2, seed=config.seed)
        candidate_session = TrainingSession(candidate_model, config.dataset, config.batch_size, config.samples,
                                            config.noise, config.validation_split, config.seed,
                                            config.early_stopping_patience, config.scheduler_patience,
                                            config.scheduler_factor, config.input_size, config.output_size)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    sessions[session_id] = candidate_session
    session = candidate_session
    return session_payload(session_id, session)


@app.post("/api/sessions/{session_id}/run")
def run(session_id: str, request: TrainRequest):
    session = get_session(session_id)
    with session.lock:
        if session.status == "complete":
            raise HTTPException(status_code=409, detail="Reset the model before starting a new run")
        if session.status == "error":
            raise HTTPException(status_code=409, detail="Reset the model after a training error before retrying")
        session.status = "running"
        session.max_epochs = request.max_epochs
        session.prepare_replay(request.max_epochs)
        if session.started_at is None:
            session.started_at = time.monotonic()
        if session.worker is None or not session.worker.is_alive():
            session.worker = threading.Thread(target=_training_worker, args=(session_id, session), daemon=True,
                                              name=f"nn-training-{session_id[:8]}")
            session.worker.start()
    return {"status": session.status, "max_epochs": request.max_epochs}


@app.post("/api/sessions/{session_id}/pause")
def pause(session_id: str):
    session = get_session(session_id)
    with session.lock:
        if session.status == "running":
            session.elapsed_seconds += time.monotonic() - session.started_at if session.started_at else 0
            session.started_at = None
            session.status = "paused"
    return {"status": session.status}


@app.post("/api/sessions/{session_id}/resume")
def resume(session_id: str, request: TrainRequest):
    session = get_session(session_id)
    with session.lock:
        if session.status == "complete":
            raise HTTPException(status_code=409, detail="Reset the model before training again")
        session.status = "running"
        session.max_epochs = request.max_epochs
        session.prepare_replay(request.max_epochs)
        session.started_at = time.monotonic()
        if session.worker is None or not session.worker.is_alive():
            session.worker = threading.Thread(target=_training_worker, args=(session_id, session), daemon=True,
                                              name=f"nn-training-{session_id[:8]}")
            session.worker.start()
    return {"status": session.status}


@app.post("/api/sessions/{session_id}/stop")
def stop(session_id: str):
    session = get_session(session_id)
    with session.lock:
        if session.status in {"running", "paused"}:
            session.elapsed_seconds += time.monotonic() - session.started_at if session.started_at else 0
            session.started_at = None
            session.status = "stopped"
    return {"status": session.status, "epoch": session.epoch}


@app.get("/api/sessions/{session_id}/status")
def training_status(session_id: str):
    session = get_session(session_id)
    with session.lock:
        return {"status": session.status, "epoch": session.epoch,
                "elapsed_seconds": round(session.elapsed_seconds + (time.monotonic() - session.started_at if session.started_at else 0), 3),
                "error": session.error}


def _training_worker(session_id, session):
    while True:
        with session.lock:
            status = session.status
            if status in {"complete", "stopped", "error", "ready"}:
                return
            if status == "running" and session.epoch >= session.max_epochs:
                session.status = "complete"
                session.elapsed_seconds += time.monotonic() - session.started_at if session.started_at else 0
                session.started_at = None
                return
        if status == "paused":
            time.sleep(0.08)
            continue
        try:
            session.train_one_epoch()
            with session.lock:
                if session.status == "running" and (session.epoch >= session.max_epochs or session.status == "complete"):
                    session.status = "complete"
                if session.status == "complete":
                    session.elapsed_seconds += time.monotonic() - session.started_at if session.started_at else 0
                    session.started_at = None
                    return
                if session.status == "stopped":
                    session._record_replay_frame(force=True)
                    return
            time.sleep(0.04)
        except Exception as exc:
            logger.exception("Training failed for session %s", session_id)
            with session.lock:
                session.error = str(exc)
                session.status = "error"
                session.elapsed_seconds += time.monotonic() - session.started_at if session.started_at else 0
                session.started_at = None
            return


@app.post("/api/sessions/{session_id}/train")
def train_one_epoch(session_id: str):
    session = get_session(session_id)
    with session.lock:
        if session.status != "running":
            return {"status": session.status, "epoch": session.epoch}
        if session.worker is not None and session.worker.is_alive():
            raise HTTPException(status_code=409, detail="A background trainer is active; read progress over WebSocket")
        loss, score = session.train_one_epoch()
        if session.epoch >= getattr(session, "max_epochs", 1000):
            session.status = "complete"
        return {"status": session.status, "epoch": session.epoch, "loss": loss, "accuracy": score}


@app.post("/api/sessions/{session_id}/reset")
def reset(session_id: str):
    session = get_session(session_id)
    with session.lock:
        session.status = "stopped"
    if session.worker is not None and session.worker.is_alive():
        session.worker.join(timeout=2)
    session.reset()
    return session_payload(session_id, session)


@app.get("/api/sessions/{session_id}/backprop")
def backprop(session_id: str, sample: int = 0):
    session = get_session(session_id)
    sample = max(0, min(sample, len(session.x) - 1))
    with session.lock:
        detail = session.model.backprop_detail(session.x[sample:sample + 1], session.targets([sample]))
    detail["y"] = int(session.y[sample, 0])
    detail["sample_index"] = sample
    return detail


@app.post("/api/sessions/{session_id}/backprop/update")
def apply_backprop_update(session_id: str, sample: int = 0):
    """Update the model once using gradients from the selected example."""
    session = get_session(session_id)
    with session.lock:
        if session.status == "running":
            raise HTTPException(status_code=409, detail="Pause training before applying a manual backpropagation update")
        sample = max(0, min(sample, len(session.x) - 1))
        # Use the same gradients as the backpropagation display: calculate this
        # one example in inference mode, with dropout turned off.
        session.model.train_batch(session.x[sample:sample + 1], session.targets([sample]), training=False)
        session.model.last_update["source"] = "manual_backprop"
        session.model.last_update["sample_index"] = sample
        session._record_replay_frame(force=True)
    return session_payload(session_id, session)


@app.get("/api/sessions/{session_id}/forward-trace")
def forward_trace(session_id: str, sample: int = 0):
    session = get_session(session_id)
    sample = max(0, min(sample, len(session.x) - 1))
    with session.lock:
        return session.model.forward_trace(session.x[sample:sample + 1])


@app.get("/api/sessions/{session_id}/replay")
def replay_manifest(session_id: str):
    session = get_session(session_id)
    with session.lock:
        if session.status == "running":
            raise HTTPException(status_code=409, detail="Pause or finish training before replaying it")
        return {"frames": session.replay_manifest(), "stride": session.replay_stride}


@app.get("/api/sessions/{session_id}/replay/{frame_index}")
def replay_frame(session_id: str, frame_index: int, sample: int = 0):
    session = get_session(session_id)
    with session.lock:
        if session.status == "running":
            raise HTTPException(status_code=409, detail="Pause or finish training before replaying it")
        try:
            return session.replay_frame(frame_index, sample)
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/api/sessions/{session_id}/gradient-check")
def gradient_check(session_id: str, sample: int = 0):
    session = get_session(session_id)
    with session.lock:
        if session.status == "running":
            raise HTTPException(status_code=409, detail="Pause training before gradient checking")
        if session.model.dropout:
            raise HTTPException(status_code=409, detail="Set dropout to zero before gradient checking")
        sample = max(0, min(sample, len(session.x) - 1))
        return session.model.gradient_check(session.x[sample:sample + 1], session.targets([sample]))


@app.get("/api/sessions/{session_id}/export")
def export(session_id: str):
    session = get_session(session_id)
    with session.lock:
        document = {
        "format": "neural-network-lab/experiment-v1",
        "model": session.model.snapshot(),
        "training": {
            "dataset": session.dataset_name,
            "input_size": session.input_size,
            "output_size": session.output_size,
            "epochs": session.epoch,
            "status": session.status,
            "history": session.history,
            "samples": session.samples,
            "noise": session.noise,
            "validation_split": session.validation_split,
            "seed": session.seed,
            "batch_size": session.batch_size,
        },
        }
    return JSONResponse(document)


@app.post("/api/sessions/import")
def import_model(request: ModelImportRequest):
    try:
        if len(sessions) >= MAX_SESSIONS:
            raise HTTPException(status_code=429, detail="Session limit reached; restart the server to clear sessions")
        model = NeuralNetwork.from_snapshot(request.model)
        training = request.training
        dataset = training.get("dataset", "XOR")
        if dataset not in DATASETS:
            raise ValueError("Unknown dataset in imported experiment")
        validate_architecture(dataset, model.architecture[0], model.architecture[-1])
        config = SessionConfig(
            hidden_layers=model.architecture[1:-1], activation=model.activation,
            input_size=model.architecture[0], output_size=model.architecture[-1],
            learning_rate=model.learning_rate, optimizer=model.optimizer_name, dataset=dataset,
            samples=training.get("samples", 360), noise=training.get("noise", 0.08),
            validation_split=training.get("validation_split", 0.2), seed=training.get("seed", 13),
            initialization=model.initialization, dropout=model.dropout, l2=model.l2,
            batch_size=training.get("batch_size", 24),
        )
        session = TrainingSession(model, dataset, config.batch_size, config.samples, config.noise,
                                  config.validation_split, config.seed, config.early_stopping_patience,
                                  config.scheduler_patience, config.scheduler_factor,
                                  config.input_size, config.output_size)
        history = training.get("history", {})
        if history:
            keys = ("loss", "accuracy", "validation_loss", "validation_accuracy", "learning_rate")
            lengths = {len(history.get(key, [])) for key in keys}
            if len(lengths) != 1 or max(lengths, default=0) > 10000:
                raise ValueError("Training history arrays must be aligned and contain at most 10,000 epochs")
            for key in keys:
                values = [float(value) for value in history.get(key, [])]
                if not np.all(np.isfinite(values)):
                    raise ValueError("Training history must contain only finite numbers")
                session.history[key] = values
            session.epoch = len(session.history["loss"])
        session_id = str(uuid4())
        sessions[session_id] = session
        return session_payload(session_id, session)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.websocket("/ws/training/{session_id}")
async def training_updates(websocket: WebSocket, session_id: str):
    session = sessions.get(session_id)
    if session is None:
        await websocket.close(code=4404, reason="Training session not found")
        return
    origin = websocket.headers.get("origin")
    if origin and origin not in allowed_origins:
        await websocket.close(code=1008, reason="Origin is not allowed")
        return
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(session_payload(session_id, session))
            if session.status in {"complete", "stopped", "error"}:
                await asyncio.sleep(0.8)
                return
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        return
    except Exception:
        logger.exception("WebSocket update failed for session %s", session_id)


FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
