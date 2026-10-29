from backend.main import (
    PredictionRequest, SessionConfig, TrainRequest, UpdateConfig, apply_backprop_update, backprop, create_session,
    export, forward_trace, get_state, gradient_check, import_model, list_datasets,
    pause, predict, reset, resume, run, sessions, stop, training_status, update_config,
)


def test_model_prediction_traces_export_import_and_reset():
    created = create_session(SessionConfig(hidden_layers=[4], dataset="Moons", samples=80))
    session_id = created["session_id"]
    assert created["model"]["architecture"] == [2, 4, 1]
    assert predict(session_id, PredictionRequest(inputs=[[0.1, 0.2]]))["classes"]
    assert forward_trace(session_id, sample=0)["layers"]
    backward = backprop(session_id, sample=0)["backward"]
    assert backward
    hidden_step = backward[1]
    assert len(hidden_step["upstream_error"]) == len(hidden_step["activation_slope"]) == len(hidden_step["delta"])
    assert all(abs(error * slope - delta) < 2e-5 for error, slope, delta in zip(
        hidden_step["upstream_error"], hidden_step["activation_slope"], hidden_step["delta"]
    ))
    assert gradient_check(session_id, sample=0)["passed"]
    exported = export(session_id).body
    import json
    restored = import_model(type("Request", (), {"model": json.loads(exported)["model"], "training": json.loads(exported)["training"]})())
    assert restored["model"]["architecture"] == [2, 4, 1]
    assert reset(session_id)["epoch"] == 0
    assert list_datasets()["datasets"]


def test_state_contains_actual_last_optimizer_update():
    created = create_session(SessionConfig(hidden_layers=[3], samples=80))
    session_id = created["session_id"]
    sessions[session_id].train_one_epoch()
    state = get_state(session_id)
    update = state["last_update"]
    assert update["optimizer"] == "Adam"
    assert update["step"] > 0
    assert len(update["layers"]) == len(state["model"]["weights"])
    layer = update["layers"][0]
    assert layer["weights_before"] != layer["weights_after"]


def test_training_controls_report_state():
    created = create_session(SessionConfig(hidden_layers=[3], samples=2000, early_stopping_patience=0))
    session_id = created["session_id"]
    assert run(session_id, TrainRequest(max_epochs=10000))["status"] == "running"
    assert pause(session_id)["status"] in {"paused", "running"}
    assert training_status(session_id)["status"] in {"paused", "running"}
    assert resume(session_id, TrainRequest(max_epochs=10000))["status"] == "running"
    assert stop(session_id)["status"] in {"stopped", "running"}
    assert get_state(session_id)["status"] in {"stopped", "running"}


def test_applying_config_while_paused_stops_old_worker():
    created = create_session(SessionConfig(samples=2000, early_stopping_patience=0))
    session_id = created["session_id"]
    run(session_id, TrainRequest(max_epochs=10000))
    pause(session_id)
    old_session = sessions[session_id]

    updated = update_config(session_id, UpdateConfig(
        hidden_layers=[4], activation="tanh", learning_rate=0.03,
        optimizer="Adam", batch_size=24, dataset="XOR", samples=80,
        noise=0.08, validation_split=0.2, seed=13,
    ))

    assert updated["model"]["architecture"] == [2, 4, 1]
    assert old_session.status == "stopped"
    assert old_session.worker is None or not old_session.worker.is_alive()


def test_configurable_input_and_output_widths_reach_the_backend_model():
    created = create_session(SessionConfig(hidden_layers=[4], samples=80))
    session_id = created["session_id"]
    updated = update_config(session_id, UpdateConfig(
        hidden_layers=[4], input_size=4, output_size=2, activation="tanh",
        learning_rate=0.04, optimizer="Adam", batch_size=24, dataset="Moons",
        samples=80, noise=0.08, validation_split=0.2, seed=13,
    ))

    assert updated["model"]["architecture"] == [4, 4, 2]
    assert sessions[session_id].x.shape == (80, 4)
    assert sessions[session_id].targets([0, 1]).shape == (2, 2)
    sessions[session_id].train_one_epoch()
    prediction = predict(session_id, PredictionRequest(inputs=[[0.1, 0.2, 0.03, 0.04]]))
    assert len(prediction["predictions"][0]) == 2


def test_manual_backprop_update_changes_weights_and_biases():
    created = create_session(SessionConfig(hidden_layers=[3], samples=80))
    session_id = created["session_id"]
    before = sessions[session_id].model.snapshot()

    detail = backprop(session_id, sample=0)
    assert detail["backward"]
    updated = apply_backprop_update(session_id, sample=0)

    assert updated["model"]["weights"] != before["weights"]
    assert updated["model"]["biases"] != before["biases"]
    assert updated["last_update"]["step"] == 1
    assert updated["last_update"]["layers"][0]["weights_before"] != updated["last_update"]["layers"][0]["weights_after"]
