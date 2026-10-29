import { NetworkDiagram } from "./network.js?v=12";
import { drawHistory, drawBoundary } from "./charts.js?v=2";

(() => {
  const $ = id => document.getElementById(id);
  const API_BASE = String(window.NN_LAB_CONFIG?.apiBase || "").replace(/\/$/, "");
  const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
  const lessons = [
    { title: "Start with one point", description: "A neural network learns from examples. Each example has two input coordinates and a class label.", action: "Choose XOR and find one colored point in the decision-boundary panel.", target: "dataset" },
    { title: "Build the network", description: "Each column is a layer. Connections carry weights; each neuron also has a bias.", action: "Try adding a hidden layer, then remove it to see the architecture change.", target: "layer-stack" },
    { title: "Meet one neuron", description: "A neuron multiplies inputs by weights, adds a bias, and applies an activation function.", action: "Open the single-neuron sandbox and change one input or weight.", target: "neuron-sandbox", open: true },
    { title: "Follow a prediction", description: "Forward propagation carries values from the inputs through each layer to a prediction.", action: "Switch to Visualize, then use Next in the forward-pass panel to inspect each layer.", target: "nav-visualize" },
    { title: "Let it learn", description: "Training compares predictions with the correct labels, then changes weights and biases to reduce the error.", action: "Train briefly. Watch the highlighted connections, bias labels, live update values, and decision boundary change.", target: "train-button" },
    { title: "Send the error backward", description: "Backpropagation uses the chain rule to work out how each layer and weight contributed to the error.", action: "Switch to Visualize, choose Backpropagation, then watch δ values travel backward and inspect the latest weight and bias update.", target: "nav-visualize" },
    { title: "Change one setting", description: "Learning rate controls update size. Activation and architecture change how the network can represent patterns.", action: "Change one setting, apply it, then train again and compare the result.", target: "learning-rate" },
    { title: "Keep and compare experiments", description: "A useful experiment records the settings as well as its results.", action: "Switch to Visualize, save two runs, and compare their settings and metrics side by side.", target: "nav-visualize" },
  ];
  const challenges = [
    { title: "Teach the network XOR", description: "Train the default network, then watch its boundary separate the four XOR regions.", dataset: "XOR", activation: "tanh", hidden: [5, 4], samples: 360, noise: 0.02, rate: 0.04 },
    { title: "Fit two moons", description: "Try a compact network on curved classes. Compare the boundary before and after training.", dataset: "Moons", activation: "tanh", hidden: [5, 4], samples: 360, noise: 0.05, rate: 0.04 },
    { title: "Separate concentric circles", description: "Try changing the activation while keeping the data fixed. Notice the shape of the boundary.", dataset: "Circles", activation: "relu", hidden: [6, 4], samples: 360, noise: 0.04, rate: 0.03 },
    { title: "Tackle the spiral", description: "Use more examples and an extra hidden layer. Watch how the boundary forms as training progresses.", dataset: "Spiral", activation: "tanh", hidden: [8, 6, 4], samples: 600, noise: 0.05, rate: 0.025 },
  ];
  let sessionId = null;
  let state = null;
  let progressSocket = null;
  let forwardTrace = null;
  let forwardStep = 0;
  let selectedForwardNeuron = 0;
  let forwardTimer = null;
  let backwardDetail = null;
  let backwardStep = 0;
  let backwardTimer = null;
  let networkPass = { mode: "forward", step: 0 };
  let lessonStep = 0;
  let workspaceView = (() => { try { return localStorage.getItem("nnLabWorkspaceView") === "guide" ? "guide" : "visualize"; } catch { return "visualize"; } })();
  let challengeIndex = 0;
  let lastUpdateLayer = 0;
  let selectedSample = 0;
  let lastNetworkPaint = 0;
  let hiddenLayers = [5, 4];
  let configDirty = false;
  let epochLimit = 1000;
  let signalsOn = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let trainingVizSpeed = (() => {
    try {
      const saved = Number(localStorage.getItem("nnLabTrainingVisualizationSpeed"));
      return [0.5, 1, 2].includes(saved) ? saved : 1;
    } catch { return 1; }
  })();
  let replayState = null;
  let replayFrames = [];
  let replayTimer = null;
  let replayPlaying = false;
  let replayManifestEpoch = null;
  let replayIndex = 0;
  let toastTimer;

  async function api(path, options = {}) {
    const response = await fetch(`${API_BASE}${path}`, {
      ...options,
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
    return data;
  }

  function notify(message) {
    const toast = $("toast");
    toast.textContent = message;
    toast.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove("show"), 2300);
  }

  function renderLesson(scrollToTarget = false) {
    const lesson = lessons[lessonStep];
    $("lesson-counter").textContent = `STEP ${lessonStep + 1} OF ${lessons.length}`;
    $("lesson-position").textContent = `${lessonStep + 1} / ${lessons.length}`;
    $("lesson-title").textContent = lesson.title;
    $("lesson-description").textContent = lesson.description;
    $("lesson-action").textContent = lesson.action;
    $("lesson-progress-fill").style.width = `${(lessonStep + 1) / lessons.length * 100}%`;
    $("lesson-prev").disabled = lessonStep === 0;
    $("lesson-next").textContent = lessonStep === lessons.length - 1 ? "Finish guide ✓" : "Next idea →";
    document.querySelectorAll(".lesson-focus").forEach(element => element.classList.remove("lesson-focus"));
    const target = document.getElementById(lesson.target);
    target?.classList.add("lesson-focus");
    if (lesson.open) $("neuron-sandbox").open = true;
    if (scrollToTarget && target && !target.closest("[hidden]")) {
      if (target.matches(".top-nav-tab")) target.focus({ preventScroll: true });
      else target.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "center", inline: "nearest" });
    }
  }

  function setWorkspaceView(view) {
    workspaceView = view === "guide" ? "guide" : "visualize";
    const isGuide = workspaceView === "guide";
    $("guide-view").hidden = !isGuide;
    $("visualizer-view").hidden = isGuide;
    $("nav-guide").classList.toggle("is-active", isGuide);
    $("nav-guide").setAttribute("aria-selected", String(isGuide));
    $("nav-guide").tabIndex = isGuide ? 0 : -1;
    $("nav-visualize").classList.toggle("is-active", !isGuide);
    $("nav-visualize").setAttribute("aria-selected", String(!isGuide));
    $("nav-visualize").tabIndex = isGuide ? -1 : 0;
    try { localStorage.setItem("nnLabWorkspaceView", workspaceView); } catch {}
    if (isGuide) renderLesson();
    else requestAnimationFrame(() => { if (state) renderNetwork(); });
  }

  function renderChallenge() {
    const challenge = challenges[challengeIndex % challenges.length];
    $("challenge-title").textContent = challenge.title;
    $("challenge-description").textContent = challenge.description;
  }

  function markConfigDirty() {
    configDirty = true;
    $("config-note").textContent = "Settings changed. Click Apply configuration to rebuild the network and use these neuron counts.";
    if ($("apply-dimensions")) $("apply-dimensions").disabled = false;
    if ($("train-button")) $("train-button").disabled = true;
    if ($("train-hint")) $("train-hint").textContent = "Apply the changed dataset or neuron counts before training.";
  }

  function togglePanel(button) {
    const content = document.getElementById(button.dataset.minimize);
    if (!content) return;
    const expanded = button.getAttribute("aria-expanded") === "true";
    content.hidden = expanded;
    button.setAttribute("aria-expanded", String(!expanded));
    const panelName = ({ "network-content": "network graph", "history-content": "learning curves", "boundary-content": "decision boundary", "forward-content": "forward pass", "backward-content": "backpropagation" })[button.dataset.minimize] || "panel";
    button.setAttribute("aria-label", `${expanded ? "Expand" : "Minimize"} ${panelName}`);
    button.title = expanded ? "Expand panel" : "Minimize panel";
    button.textContent = expanded ? "+" : "−";
    if (!expanded) requestAnimationFrame(() => {
      if (content.id === "network-content" && state) renderNetwork();
      if (content.id === "history-content" && state) drawHistory($("history-chart"), replayState?.history || state.history);
      if (content.id === "boundary-content" && state) {
        const boundaryState = replayState?.decision_boundary ? replayState : state;
        if (boundaryState.decision_boundary) drawBoundary($("boundary-chart"), { ...boundaryState, selectedSample });
      }
      if (content.id === "forward-content" && forwardTrace) renderForwardTrace();
    });
  }

  function loadChallenge() {
    if (!sessionId) { notify("Wait for the Python API to connect before loading a challenge."); return; }
    const challenge = challenges[challengeIndex % challenges.length];
    paintLayerEditor(challenge.hidden);
    $("dataset").value = challenge.dataset;
    $("input-size").value = "2";
    $("output-size").value = challenge.dataset === "Clusters" ? "3" : "1";
    markConfigDirty();
    $("dataset-name").textContent = challenge.dataset;
    $("activation").value = challenge.activation;
    $("sample-count").value = String(challenge.samples);
    $("noise").value = String(challenge.noise);
    $("noise-value").textContent = challenge.noise.toFixed(2);
    $("learning-rate").value = String(challenge.rate);
    $("lr-value").textContent = challenge.rate.toFixed(3);
    $("apply-config").click();
  }

  function renderSingleNeuron() {
    const x1 = Number($("neuron-x1").value);
    const w1 = Number($("neuron-w1").value);
    const x2 = Number($("neuron-x2").value);
    const w2 = Number($("neuron-w2").value);
    const bias = Number($("neuron-bias").value);
    const activation = $("neuron-activation").value;
    const z = x1 * w1 + x2 * w2 + bias;
    const output = activation === "relu" ? Math.max(0, z) : activation === "sigmoid" ? 1 / (1 + Math.exp(-z)) : Math.tanh(z);
    const fmt = value => (value < 0 ? `−${Math.abs(value).toFixed(2)}` : value.toFixed(2));
    $("neuron-sum").textContent = z.toFixed(3);
    $("neuron-equation").textContent = `(${fmt(x1)} × ${fmt(w1)}) + (${fmt(x2)} × ${fmt(w2)}) + ${fmt(bias)}`;
    $("neuron-output").textContent = output.toFixed(3);
    $("neuron-output-rule").textContent = `${activation}(z)`;
    $("neuron-explanation").textContent = `${activation} receives z = ${z.toFixed(3)} and produces ${output.toFixed(3)}.`;
  }

  function readExperiments() {
    try {
      const value = JSON.parse(localStorage.getItem("nnLabExperiments") || "[]");
      return Array.isArray(value) ? value : [];
    }
    catch { return []; }
  }

  function populateComparison() {
    const saved = readExperiments();
    for (const id of ["comparison-a", "comparison-b"]) {
      const selector = $(id);
      const oldValue = selector.value;
      selector.replaceChildren(...saved.map((item, index) => {
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = item.name || `Run ${index + 1}`;
        return option;
      }));
      if (saved[Number(oldValue)]) selector.value = oldValue;
    }
    if (saved.length > 1) {
      $("comparison-a").value = String(saved.length - 2);
      $("comparison-b").value = String(saved.length - 1);
    }
    renderComparison();
  }

  function renderComparison() {
    const saved = readExperiments();
    const a = saved[Number($("comparison-a").value)];
    const b = saved[Number($("comparison-b").value)];
    const container = $("comparison-table");
    container.replaceChildren();
    if (saved.length < 2) {
      $("comparison-message").textContent = "Save at least two experiments with Save experiment before comparing runs.";
      return;
    }
    if (!a || !b) return;
    if (a === b) {
      $("comparison-message").textContent = "Choose two different saved experiments to compare them.";
      return;
    }
    $("comparison-message").textContent = "Compare the settings and final recorded metrics for these saved runs.";
    const modelA = a.experiment?.model || {};
    const modelB = b.experiment?.model || {};
    const trainA = a.experiment?.training || {};
    const trainB = b.experiment?.training || {};
    const metric = (training, key, digits = 4, percent = false) => {
      const values = training.history?.[key] || [];
      if (!values.length) return "—";
      const value = Number(values.at(-1));
      return percent ? `${(value * 100).toFixed(1)}%` : value.toFixed(digits);
    };
    const rows = [
      ["Dataset", trainA.dataset, trainB.dataset],
      ["Architecture", (modelA.architecture || []).join(" — "), (modelB.architecture || []).join(" — ")],
      ["Activation", modelA.activation, modelB.activation],
      ["Optimizer", modelA.optimizer, modelB.optimizer],
      ["Learning rate", Number(modelA.learning_rate || 0).toPrecision(3), Number(modelB.learning_rate || 0).toPrecision(3)],
      ["Epochs", trainA.epochs ?? 0, trainB.epochs ?? 0],
      ["Training loss", metric(trainA, "loss"), metric(trainB, "loss")],
      ["Training accuracy", metric(trainA, "accuracy", 1, true), metric(trainB, "accuracy", 1, true)],
      ["Validation loss", metric(trainA, "validation_loss"), metric(trainB, "validation_loss")],
      ["Validation accuracy", metric(trainA, "validation_accuracy", 1, true), metric(trainB, "validation_accuracy", 1, true)],
    ];
    const table = document.createElement("table");
    const head = document.createElement("thead");
    const headRow = document.createElement("tr");
    ["Measure", a.name || "Run A", b.name || "Run B"].forEach(value => { const cell = document.createElement("th"); cell.textContent = value; headRow.append(cell); });
    head.append(headRow); table.append(head);
    const body = document.createElement("tbody");
    rows.forEach(row => { const tr = document.createElement("tr"); row.forEach((value, index) => { const cell = document.createElement(index === 0 ? "th" : "td"); cell.textContent = value ?? "—"; tr.append(cell); }); body.append(tr); });
    table.append(body); container.append(table);
  }

  function renderUpdateExplanation(update = state?.last_update) {
    const selector = $("update-layer");
    const container = $("update-values");
    const matrixWasOpen = container.querySelector(".update-matrix")?.open || false;
    container.replaceChildren();
    if (!update?.layers?.length) {
      selector.disabled = true;
      $("update-summary").textContent = "Train the model to see the actual parameter change from its latest mini-batch.";
      return;
    }
    if (selector.options.length !== update.layers.length) {
      selector.replaceChildren(...update.layers.map((_, index) => {
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = String(index + 1);
        return option;
      }));
    }
    selector.disabled = false;
    lastUpdateLayer = Math.max(0, Math.min(lastUpdateLayer, update.layers.length - 1));
    selector.value = String(lastUpdateLayer);
    const updateSource = update.source === "manual_backprop"
      ? `Values are from selected example #${Number(update.sample_index) + 1}.`
      : "Values are from the latest training mini-batch.";
    $("update-summary").textContent = `Actual ${update.optimizer} update · learning rate ${Number(update.learning_rate).toPrecision(3)} · optimizer step ${update.step}. ${updateSource}`;
    const layer = update.layers[lastUpdateLayer];
    const oldWeight = Number(layer.weights_before[0][0]);
    const gradient = Number(layer.weight_gradients[0][0]);
    const newWeight = Number(layer.weights_after[0][0]);
    const oldBias = Number(layer.biases_before[0]);
    const biasGradient = Number(layer.bias_gradients[0]);
    const newBias = Number(layer.biases_after[0]);
    const stat = (label, before, derivative, after, suffix) => {
      const card = document.createElement("div"); card.className = "update-stat";
      const name = document.createElement("span"); name.textContent = label;
      const values = document.createElement("strong"); values.textContent = `${before.toFixed(5)} → ${after.toFixed(5)}`;
      const note = document.createElement("small"); note.textContent = `${suffix} ${derivative.toFixed(5)} · change ${(after - before).toFixed(5)}`;
      card.append(name, values, note); return card;
    };
    const cards = document.createElement("div"); cards.className = "update-stat-grid";
    cards.append(stat(`W${lastUpdateLayer + 1}[0,0]`, oldWeight, gradient, newWeight, "gradient"));
    cards.append(stat(`b${lastUpdateLayer + 1}[0]`, oldBias, biasGradient, newBias, "gradient"));
    container.append(cards);
    const disclosure = document.createElement("details"); disclosure.className = "math-disclosure update-matrix";
    const summary = document.createElement("summary"); summary.textContent = "Inspect the complete parameter update for this layer";
    const matrix = document.createElement("pre");
    matrix.textContent = `W before = ${JSON.stringify(layer.weights_before, null, 2)}\n∂L/∂W = ${JSON.stringify(layer.weight_gradients, null, 2)}\nW after = ${JSON.stringify(layer.weights_after, null, 2)}\nb before = ${JSON.stringify(layer.biases_before)}\n∂L/∂b = ${JSON.stringify(layer.bias_gradients)}\nb after = ${JSON.stringify(layer.biases_after)}`;
    disclosure.open = matrixWasOpen;
    disclosure.append(summary, matrix); container.append(disclosure);
  }

  function setRunning(running) {
    $("model-status").textContent = state?.status?.toUpperCase() || "READY";
    $("top-status").textContent = running ? "TRAINING" : state?.status === "error" ? "TRAINING ERROR" : "SESSION READY";
    $("top-status").style.color = running ? "#c7f36b" : "";
    $("train-button").disabled = running || configDirty || state?.status === "complete" || state?.status === "error";
    $("apply-backprop-update").disabled = running;
    $("apply-dimensions").disabled = running || !configDirty;
    $("pause-button").disabled = !running;
    $("stop-button").disabled = !running && state?.status !== "paused";
    $("train-button").innerHTML = state?.status === "paused" ? "<span>▶</span> Resume" : state?.status === "complete" ? "Complete" : "<span>▶</span> Train";
    $("apply-config").disabled = running;
    $("dataset").disabled = running;
    $("input-size").disabled = running;
    $("output-size").disabled = running;
    $("replay-button").disabled = running || replayFrames.length < 2;
    $("live-tag").classList.toggle("is-running", running);
    $("train-hint").textContent = state?.status === "error" ? `Training stopped: ${state.error || "unknown error"}` : running ? "Training in a Python worker. Progress streams over WebSocket." : configDirty ? "Apply the changed dataset or neuron counts before training." : state?.status === "paused" ? "Training paused. Resume to continue from the current weights." : state?.status === "stopped" ? "Training stopped. Resume or reset the current experiment." : "Train, pause, resume, or stop at any time.";
  }

  function paintLayerEditor(values = hiddenLayers) {
    hiddenLayers = [...values];
    const stack = $("layer-stack");
    stack.replaceChildren();
    hiddenLayers.forEach((width, index) => {
      const row = document.createElement("div");
      row.className = "layer-row";
      const bubble = document.createElement("span");
      bubble.className = "layer-node";
      bubble.textContent = String(index + 1);
      const caption = document.createElement("span");
      caption.className = "layer-caption";
      caption.textContent = `Hidden layer ${index + 1}`;
      const input = document.createElement("input");
      input.className = "layer-input";
      input.type = "number";
      input.min = "2";
      input.max = "128";
      input.value = String(width);
      input.setAttribute("aria-label", `Neurons in hidden layer ${index + 1}`);
      input.addEventListener("input", () => { hiddenLayers[index] = Number(input.value); });
      row.append(bubble, caption, input);
      stack.append(row);
    });
    $("add-layer").disabled = hiddenLayers.length >= 4;
    $("remove-layer").disabled = hiddenLayers.length === 0;
  }

  function syncDimensionFields(dataset, inputSize, outputSize) {
    $("input-size").value = String(inputSize);
    $("output-size").value = String(outputSize);
    $("input-size").max = "16";
    const classMinimum = dataset === "Clusters" ? 3 : 1;
    $("output-size").min = String(classMinimum);
    $("dimension-hint").innerHTML = `${inputSize} input feature${inputSize === 1 ? "" : "s"} <span>·</span> ${outputSize} output neuron${outputSize === 1 ? "" : "s"} <span>·</span> ${dataset === "Clusters" ? "3" : "2"} classes`;
  }

  function refreshDimensionHint() {
    const dataset = $("dataset").value;
    const inputSize = Math.max(1, Number($("input-size").value) || 1);
    const outputSize = Math.max(1, Number($("output-size").value) || 1);
    const classMinimum = dataset === "Clusters" ? 3 : 1;
    $("output-size").min = String(classMinimum);
    $("dimension-hint").innerHTML = `${inputSize} input feature${inputSize === 1 ? "" : "s"} <span>·</span> ${outputSize} output neuron${outputSize === 1 ? "" : "s"} <span>·</span> ${dataset === "Clusters" ? "3" : "2"} classes`;
  }

  function reflectConfig(model, dataset, batchSize) {
    configDirty = false;
    $("apply-dimensions").disabled = true;
    $("config-note").textContent = "Applying a new shape or activation reinitializes weights.";
    hiddenLayers = model.architecture.slice(1, -1);
    paintLayerEditor(hiddenLayers);
    $("activation").value = model.activation;
    $("learning-rate").value = model.learning_rate;
    $("lr-value").textContent = Number(model.learning_rate).toFixed(3);
    $("optimizer").value = model.optimizer;
    $("batch-size").value = String(batchSize);
    $("dataset").value = dataset;
    syncDimensionFields(dataset, model.architecture[0], model.architecture.at(-1));
    $("sample-count").value = String(state?.samples || 360);
    $("seed").value = String(state?.seed || 13);
    $("noise").value = String(state?.noise || 0);
    $("noise-value").textContent = Number($("noise").value).toFixed(2);
    $("validation-split").value = String(state?.validation_split || 0.2);
    $("split-value").textContent = `${Math.round(Number($("validation-split").value) * 100)}%`;
    $("initialization").value = model.initialization || "auto";
    $("dropout").value = String(model.dropout || 0);
    $("l2").value = String(model.l2 || 0);
  }

  function renderNetwork(pass = networkPass) {
    if (!state) return;
    const displayState = replayState || state;
    networkPass = pass;
    $("show-forward-view").classList.toggle("is-active", networkPass.mode === "forward");
    $("show-forward-view").setAttribute("aria-pressed", String(networkPass.mode === "forward"));
    $("show-backward-view").classList.toggle("is-active", networkPass.mode === "backward");
    $("show-backward-view").setAttribute("aria-pressed", String(networkPass.mode === "backward"));
    if (networkPass.mode === "forward") {
      $("pass-indicator").textContent = networkPass.step === 0 ? "FORWARD · INPUT" : `FORWARD · LAYER ${networkPass.step}`;
    } else if (networkPass.mode === "backward" && backwardDetail?.backward?.length) {
      const row = backwardDetail.backward[backwardDetail.backward.length - 1 - networkPass.layer];
      if (row) $("pass-indicator").textContent = `BACKWARD · ${row.layer.toUpperCase()}`;
    }
    const update = replayState ? null : state.last_update;
    const updatedLayer = update?.layers?.[lastUpdateLayer];
    const updateIndicator = $("weight-update-indicator");
    if (replayState) {
      updateIndicator.textContent = `TRAINING REPLAY · EPOCH ${replayState.epoch} · actual recorded network parameters`;
      updateIndicator.classList.add("has-update");
    } else if (updatedLayer) {
      const signed = value => `${value >= 0 ? "+" : ""}${Number(value).toFixed(4)}`;
      const largestChange = (before, after) => {
        let largest = { index: 0, before: Number(before[0]), after: Number(after[0]), magnitude: -1 };
        before.forEach((row, index) => {
          const values = Array.isArray(row) ? row : [row];
          const nextValues = Array.isArray(after[index]) ? after[index] : [after[index]];
          values.forEach((value, column) => {
            const next = Number(nextValues[column]);
            const magnitude = Math.abs(next - Number(value));
            if (magnitude > largest.magnitude) largest = { index: Array.isArray(row) ? [index, column] : index, before: Number(value), after: next, magnitude };
          });
        });
        return largest;
      };
      const weight = largestChange(updatedLayer.weights_before, updatedLayer.weights_after);
      const bias = largestChange(updatedLayer.biases_before, updatedLayer.biases_after);
      const weightIndex = Array.isArray(weight.index) ? `[${weight.index.join(",")}]` : `[${weight.index}]`;
      updateIndicator.textContent = `${state.status === "running" ? "LIVE" : "LAST"} ${update.optimizer} UPDATE ${update.step} · LAYER ${lastUpdateLayer + 1} · W${lastUpdateLayer + 1}${weightIndex} ${weight.before.toFixed(3)} → ${weight.after.toFixed(3)} (Δ ${signed(weight.after - weight.before)}) · b${lastUpdateLayer + 1}[${bias.index}] ${bias.before.toFixed(3)} → ${bias.after.toFixed(3)} (Δ ${signed(bias.after - bias.before)})`;
      updateIndicator.classList.add("has-update");
    } else {
      updateIndicator.textContent = "Train to see real weight and bias updates";
      updateIndicator.classList.remove("has-update");
    }
    try {
      const updateDurationMs = Math.round(1000 / trainingVizSpeed);
      NetworkDiagram.render($("network-view"), displayState, signalsOn, {
        ...networkPass,
        update,
        updateLayer: lastUpdateLayer,
        updateDurationMs,
        durationMs: state.status === "running" ? updateDurationMs : networkPass.durationMs,
      });
    } catch (error) {
      console.error("Could not render the network diagram", error);
      notify(`Network diagram error: ${error.message}`);
    }
  }

  function render(nextState) {
    state = nextState;
    $("top-status").textContent = state.status === "running" ? "TRAINING" : "SESSION READY";
    $("model-status").textContent = state.status.toUpperCase();
    $("live-tag").classList.toggle("is-running", state.status === "running");
    // Training updates can arrive while someone is editing a layer size. Leave
    // the focused field alone so we do not overwrite a partial edit or spinner
    // change before the browser reports it.
    const editingDimensions = ["input-size", "output-size"].includes(document.activeElement?.id);
    const preserveDraft = configDirty || editingDimensions;
    const shownDataset = preserveDraft ? $("dataset").value : state.dataset;
    $("dataset-name").textContent = shownDataset;
    if (preserveDraft) refreshDimensionHint();
    else syncDimensionFields(state.dataset, state.model.architecture[0], state.model.architecture.at(-1));
    if (replayState) {
      $("epoch-count").textContent = replayState.epoch.toLocaleString();
      return;
    }
    $("epoch-count").textContent = state.epoch.toLocaleString();
    $("epoch-total").textContent = epochLimit.toLocaleString();
    $("progress-fill").style.width = `${Math.min(100, state.epoch / epochLimit * 100)}%`;
    $("current-lr").textContent = Number(state.model.learning_rate).toPrecision(3);
    $("elapsed-time").textContent = `${Number(state.elapsed_seconds || 0).toFixed(1)}s`;
    $("loss-metric").textContent = state.metrics.loss == null ? "—" : Number(state.metrics.loss).toFixed(4);
    $("accuracy-metric").textContent = state.metrics.accuracy == null ? "—" : `${(100 * state.metrics.accuracy).toFixed(1)}%`;
    $("validation-loss-metric").textContent = state.metrics.validation_loss == null ? "—" : Number(state.metrics.validation_loss).toFixed(4);
    $("validation-accuracy-metric").textContent = state.metrics.validation_accuracy == null ? "—" : `${(100 * state.metrics.validation_accuracy).toFixed(1)}%`;
    $("sample-metric").textContent = String(state.sample_count ?? state.points.length);
    $("architecture-label").textContent = state.model.architecture.join(" — ");
    renderUpdateExplanation(state.last_update);
    const networkNow = performance.now();
    if (state.status !== "running" || networkNow - lastNetworkPaint > Math.round(1000 / trainingVizSpeed)) {
      renderNetwork();
      lastNetworkPaint = networkNow;
    }
    drawHistory($("history-chart"), state.history);
    if (state.decision_boundary) drawBoundary($("boundary-chart"), { ...state, selectedSample });
    setRunning(state.status === "running");
    if (state.status !== "running" && state.history.loss.length && replayManifestEpoch !== state.epoch) loadReplayManifest();
  }

  function getConfig() {
    const widths = [...document.querySelectorAll(".layer-input")].map(field => Number(field.value));
    if (widths.some(width => !Number.isInteger(width) || width < 2 || width > 128)) {
      throw new Error("Each hidden layer must contain 2–128 neurons.");
    }
    const inputSize = Number($("input-size").value);
    const outputSize = Number($("output-size").value);
    if (!Number.isInteger(inputSize) || inputSize < 1 || inputSize > 16 || !Number.isInteger(outputSize) || outputSize < 1 || outputSize > 10) {
      throw new Error("Choose a valid number of input and output neurons.");
    }
    if ($("dataset").value === "Clusters" && outputSize < 3) throw new Error("Clusters has three classes, so it needs at least three output neurons.");
    return {
      hidden_layers: widths,
      input_size: inputSize,
      output_size: outputSize,
      activation: $("activation").value,
      learning_rate: Number($("learning-rate").value),
      optimizer: $("optimizer").value,
      batch_size: Number($("batch-size").value),
      dataset: $("dataset").value,
      samples: Number($("sample-count").value),
      noise: Number($("noise").value),
      validation_split: Number($("validation-split").value),
      seed: Number($("seed").value),
      initialization: $("initialization").value,
      dropout: Number($("dropout").value),
      l2: Number($("l2").value),
      early_stopping_patience: 30,
    };
  }

  function refreshSampleOptions(sampleCount) {
    const selector = $("sample-select");
    selector.replaceChildren(...Array.from({ length: Number(sampleCount) || 0 }, (_, index) => {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = `#${String(index + 1).padStart(3, "0")}`;
      return option;
    }));
  }

  function formatVector(values, digits = 3, limit = 12) {
    const shown = values.slice(0, limit).map(value => Number(value).toFixed(digits));
    if (values.length > limit) shown.push(`… ${values.length} total`);
    return `[${shown.join(", ")}]`;
  }

  function renderForwardTrace(durationMs = forwardPlaybackDelay()) {
    if (!forwardTrace) return;
    const total = forwardTrace.layers.length;
    $("forward-step-count").textContent = `Step ${forwardStep} / ${total}`;
    $("forward-step-count").setAttribute("aria-live", "polite");
    $("pass-indicator").textContent = forwardStep === 0 ? "FORWARD · INPUT" : `FORWARD · LAYER ${forwardStep}`;
    const container = $("forward-inspector");
    container.replaceChildren();
    const inputCard = document.createElement("button");
    inputCard.type = "button";
    inputCard.className = `flow-layer flow-step-button${forwardStep === 0 ? " selected-step" : ""}`;
    inputCard.setAttribute("aria-pressed", String(forwardStep === 0));
    inputCard.innerHTML = `<span>INPUT · x</span><strong>${formatVector(forwardTrace.input[0])}</strong>`;
    inputCard.addEventListener("click", () => {
      clearInterval(forwardTimer); forwardTimer = null; $("forward-play").textContent = "Play";
      forwardStep = 0; selectedForwardNeuron = 0; renderForwardTrace();
    });
    container.append(inputCard);
    forwardTrace.layers.forEach((layer, index) => {
      const arrow = document.createElement("span"); arrow.className = "flow-arrow"; arrow.textContent = "→"; container.append(arrow);
      const card = document.createElement("button");
      card.type = "button";
      card.className = `flow-layer flow-step-button${forwardStep === index + 1 ? " selected-step" : ""}`;
      card.setAttribute("aria-pressed", String(forwardStep === index + 1));
      card.innerHTML = `<span>LAYER ${index + 1} · ${layer.activation.toUpperCase()}</span><strong>${formatVector(layer.output[0])}</strong>`;
      card.addEventListener("click", () => {
        clearInterval(forwardTimer); forwardTimer = null; $("forward-play").textContent = "Play";
        forwardStep = index + 1; selectedForwardNeuron = 0; renderForwardTrace();
      });
      container.append(card);
    });
    const detail = $("forward-detail");
    detail.replaceChildren();
    detail.className = "trace-detail";
    const heading = document.createElement("div");
    heading.className = "trace-detail-heading";
    const symbol = document.createElement("span");
    symbol.className = "rule-symbol";
    const text = document.createElement("div");
    const title = document.createElement("p");
    const caption = document.createElement("span");
    if (forwardStep === 0) {
      symbol.textContent = "x";
      title.textContent = `Input vector x · example #${String(selectedSample + 1).padStart(2, "0")}`;
      caption.textContent = `${forwardTrace.input[0].length} input value${forwardTrace.input[0].length === 1 ? "" : "s"} enter the first dense layer.`;
      text.append(title, caption);
      heading.append(symbol, text);
      detail.append(heading);
      const input = document.createElement("pre");
      input.className = "trace-vector";
      input.textContent = formatVector(forwardTrace.input[0], 4, 24);
      detail.append(input);
    } else {
      const layerIndex = forwardStep - 1;
      const layer = forwardTrace.layers[layerIndex];
      symbol.textContent = "z";
      title.textContent = `Layer ${forwardStep} · ${layer.activation}`;
      caption.textContent = `z${forwardStep} = a${layerIndex}W${forwardStep} + b${forwardStep};  a${forwardStep} = ${layer.activation}(z${forwardStep})`;
      text.append(title, caption);
      heading.append(symbol, text);
      detail.append(heading);
      const readouts = document.createElement("div");
      readouts.className = "trace-readouts";
      [["INPUT a", layer.input[0]], ["PRE-ACTIVATION z", layer.preactivation[0]], ["OUTPUT a", layer.output[0]]].forEach(([label, values]) => {
        const tile = document.createElement("div");
        tile.className = "trace-readout";
        const name = document.createElement("span"); name.textContent = label;
        const value = document.createElement("strong"); value.textContent = formatVector(values);
        tile.append(name, value); readouts.append(tile);
      });
      detail.append(readouts);
      const activations = layer.output[0].map(Number);
      selectedForwardNeuron = Math.max(0, Math.min(selectedForwardNeuron, activations.length - 1));
      const activationScale = Math.max(1e-8, ...activations.map(Math.abs));
      const neuronSection = document.createElement("section");
      neuronSection.className = "forward-neuron-section";
      const neuronHeading = document.createElement("div");
      neuronHeading.className = "forward-neuron-heading";
      const neuronTitle = document.createElement("div");
      const neuronKicker = document.createElement("span"); neuronKicker.textContent = "LAYER OUTPUTS";
      const neuronLabel = document.createElement("strong"); neuronLabel.textContent = "Choose a neuron to see how its inputs combine";
      neuronTitle.append(neuronKicker, neuronLabel);
      const neuronHint = document.createElement("span"); neuronHint.textContent = `${activations.length} neuron${activations.length === 1 ? "" : "s"}`;
      neuronHeading.append(neuronTitle, neuronHint);
      neuronSection.append(neuronHeading);
      const neuronGrid = document.createElement("div");
      neuronGrid.className = "forward-neuron-grid";
      const shownActivations = Math.min(activations.length, 24);
      activations.slice(0, shownActivations).forEach((value, neuronIndex) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = `forward-neuron-card${neuronIndex === selectedForwardNeuron ? " is-selected" : ""}${value < 0 ? " is-negative" : ""}`;
        button.setAttribute("aria-pressed", String(neuronIndex === selectedForwardNeuron));
        button.setAttribute("aria-label", `Inspect neuron ${neuronIndex + 1}, activation ${value.toFixed(4)}`);
        const name = document.createElement("span"); name.textContent = `NEURON ${String(neuronIndex + 1).padStart(2, "0")}`;
        const number = document.createElement("strong"); number.textContent = value.toFixed(4);
        const meter = document.createElement("span"); meter.className = "forward-neuron-meter";
        const fill = document.createElement("i");
        const normalized = layer.activation === "sigmoid" ? Math.max(0, Math.min(1, value)) : Math.min(1, Math.abs(value) / activationScale);
        fill.style.width = `${normalized * (layer.activation === "tanh" ? 48 : 100)}%`;
        fill.style.left = layer.activation === "tanh" && value < 0 ? `${50 - normalized * 48}%` : layer.activation === "tanh" ? "50%" : "0%";
        meter.append(fill);
        button.append(name, number, meter);
        button.addEventListener("click", () => {
          clearInterval(forwardTimer); forwardTimer = null; $("forward-play").textContent = "Play";
          selectedForwardNeuron = neuronIndex; renderForwardTrace();
        });
        neuronGrid.append(button);
      });
      neuronSection.append(neuronGrid);
      detail.append(neuronSection);

      const selectedNeuron = selectedForwardNeuron;
      const contributions = layer.input[0].map((inputValue, source) => ({
        source, input: Number(inputValue), weight: Number(layer.weights[source][selectedNeuron]),
        product: Number(inputValue) * Number(layer.weights[source][selectedNeuron]),
      }));
      const contributionScale = Math.max(1e-8, ...contributions.map(item => Math.abs(item.product)));
      const contributionBox = document.createElement("section");
      contributionBox.className = "forward-contributions";
      const contributionTitle = document.createElement("div");
      contributionTitle.className = "forward-contribution-heading";
      contributionTitle.innerHTML = `<span>NEURON ${String(selectedNeuron + 1).padStart(2, "0")} · WEIGHTED SUM</span><strong>Σ(a × w) + b = z</strong>`;
      contributionBox.append(contributionTitle);
      const shownContributions = Math.min(contributions.length, 12);
      contributions.slice(0, shownContributions).forEach(item => {
        const row = document.createElement("div");
        row.className = `forward-contribution${item.product < 0 ? " is-negative" : ""}`;
        const source = document.createElement("span"); source.textContent = `a${item.source + 1}`;
        const equation = document.createElement("strong"); equation.textContent = `${item.input.toFixed(3)} × ${item.weight.toFixed(3)}`;
        const meter = document.createElement("span"); meter.className = "contribution-meter";
        const fill = document.createElement("i");
        const fillWidth = Math.min(1, Math.abs(item.product) / contributionScale) * 48;
        fill.style.width = `${fillWidth}%`;
        fill.style.left = item.product < 0 ? `${50 - fillWidth}%` : "50%";
        meter.append(fill);
        const result = document.createElement("b"); result.textContent = `${item.product >= 0 ? "+" : "−"}${Math.abs(item.product).toFixed(4)}`;
        row.append(source, equation, meter, result);
        contributionBox.append(row);
      });
      const biasValue = Number(layer.biases[selectedNeuron]);
      const preactivationValue = Number(layer.preactivation[0][selectedNeuron]);
      const sumLine = document.createElement("div");
      sumLine.className = "forward-sum-line";
      sumLine.innerHTML = `<span>BIAS b${forwardStep}[${selectedNeuron + 1}] <strong>${biasValue >= 0 ? "+" : "−"}${Math.abs(biasValue).toFixed(4)}</strong></span><span>z = <strong>${preactivationValue.toFixed(4)}</strong></span><span>${layer.activation}(z) = <strong>${activations[selectedNeuron].toFixed(4)}</strong></span>`;
      contributionBox.append(sumLine);
      if (contributions.length > shownContributions) {
        const more = document.createElement("p"); more.className = "forward-contribution-note";
        more.textContent = `Showing ${shownContributions} of ${contributions.length} inputs. Open the matrix below for all weights.`;
        contributionBox.append(more);
      }
      detail.append(contributionBox);
      const disclosure = document.createElement("details");
      disclosure.className = "math-disclosure";
      const summary = document.createElement("summary");
      summary.textContent = `Weights and biases · ${layer.weights.length} × ${layer.weights[0]?.length || 0}`;
      const matrix = document.createElement("pre");
      matrix.textContent = `W${forwardStep} = ${JSON.stringify(layer.weights.map(row => row.map(v => Number(v).toFixed(4))), null, 2)}\nb${forwardStep} = ${JSON.stringify(layer.biases.map(v => Number(v).toFixed(4)))}`;
      disclosure.append(summary, matrix); detail.append(disclosure);
    }
    $("forward-prev").disabled = forwardStep === 0;
    $("forward-next").disabled = forwardStep >= total;
    renderNetwork({ mode: "forward", step: forwardStep, selectedLayer: forwardStep, selectedNeuron: selectedForwardNeuron, durationMs });
  }

  function forwardPlaybackDelay() {
    return Math.round(1200 / Number($("forward-speed").value || 1));
  }

  function backwardPlaybackDelay() {
    return Math.round(1200 / Number($("backward-speed").value || 1));
  }

  function startForwardPlayback(advanceImmediately = false) {
    clearInterval(forwardTimer);
    if (!forwardTrace) return;
    if (advanceImmediately) forwardStep = forwardStep >= forwardTrace.layers.length ? 0 : forwardStep + 1;
    renderForwardTrace(forwardPlaybackDelay());
    forwardTimer = setInterval(() => {
      forwardStep = forwardStep >= forwardTrace.layers.length ? 0 : forwardStep + 1;
      renderForwardTrace(forwardPlaybackDelay());
    }, forwardPlaybackDelay());
    $("forward-play").textContent = "Pause";
  }

  function startBackwardPlayback() {
    clearInterval(backwardTimer);
    if (!backwardDetail?.backward?.length) return;
    selectBackwardStep(backwardStep, backwardPlaybackDelay());
    backwardTimer = setInterval(() => {
      selectBackwardStep(backwardStep >= backwardDetail.backward.length - 1 ? 0 : backwardStep + 1, backwardPlaybackDelay());
    }, backwardPlaybackDelay());
    $("backward-play").textContent = "Pause";
  }

  function selectBackwardStep(step = backwardStep, durationMs = backwardPlaybackDelay()) {
    if (!backwardDetail?.backward?.length) return;
    backwardStep = Math.max(0, Math.min(step, backwardDetail.backward.length - 1));
    const current = backwardDetail.backward[backwardStep];
    const layerIndex = backwardDetail.backward.length - 1 - backwardStep;
    $("pass-indicator").textContent = `BACKWARD · ${current.layer.toUpperCase()}`;
    $("backward-step-count").textContent = `Step ${backwardStep + 1} / ${backwardDetail.backward.length}`;
    $("backward-prev").disabled = backwardStep === 0;
    $("backward-next").disabled = backwardStep === backwardDetail.backward.length - 1;
    $("backward-steps").querySelectorAll(".gradient-step").forEach((card, index) => {
      card.classList.toggle("gradient-step-current", index === backwardStep);
      card.setAttribute("aria-pressed", String(index === backwardStep));
    });
    const mechanism = $("backprop-mechanism");
    mechanism.replaceChildren();
    const mechanismHeader = document.createElement("header");
    const formula = document.createElement("strong");
    formula.textContent = current.calculation;
    const direction = document.createElement("span");
    direction.textContent = current.layer === "output" ? "error starts at the prediction" : "follow one neuron at a time";
    mechanismHeader.append(formula, direction);
    mechanism.append(mechanismHeader);
    const errors = current.upstream_error || current.delta;
    const slopes = current.activation_slope;
    const maxError = Math.max(1e-8, ...errors.map(value => Math.abs(Number(value))));
    const maxDelta = Math.max(1e-8, ...current.delta.map(value => Math.abs(Number(value))));
    const table = document.createElement("div");
    table.className = "bp-flow-table";
    const columns = document.createElement("div");
    columns.className = "bp-flow-columns";
    ["NEURON", current.layer === "output" ? "OUTPUT ERROR" : "WEIGHTED ERROR  Σ(Wδ)", slopes ? "ACTIVATION SLOPE  f′(z)" : "LOSS DERIVATIVE", "RESULT  δ"].forEach((text, index) => {
      const label = document.createElement("span");
      label.textContent = text;
      if (index === 0) label.className = "bp-columns-neuron";
      if (index === 2) label.className = "bp-columns-slope";
      columns.append(label);
    });
    table.append(columns);
    const makeSignal = (rawValue, maxValue, isSlope = false) => {
      const cell = document.createElement("div");
      cell.className = "bp-signal";
      const meter = document.createElement("div");
      meter.className = `bp-meter${isSlope ? " slope" : ""}`;
      const fill = document.createElement("i");
      const value = Number(rawValue);
      const fraction = Math.min(1, Math.abs(value) / maxValue);
      const percent = fraction * (isSlope ? 100 : 48);
      fill.className = `bp-meter-fill${!isSlope && value < 0 ? " negative" : ""}`;
      fill.style.width = `${percent}%`;
      fill.style.left = isSlope ? "0%" : value < 0 ? `${50 - percent}%` : "50%";
      meter.append(fill);
      const number = document.createElement("strong");
      number.className = "bp-signal-value";
      number.textContent = value.toFixed(4);
      cell.setAttribute("aria-label", Number(value).toFixed(4));
      cell.append(meter, number);
      return cell;
    };
    const shownNeurons = Math.min(errors.length, 12);
    for (let index = 0; index < shownNeurons; index += 1) {
      const row = document.createElement("div");
      row.className = "bp-neuron-row";
      const neuron = document.createElement("span");
      neuron.className = "bp-neuron-name";
      neuron.textContent = `N${String(index + 1).padStart(2, "0")}`;
      row.append(neuron);
      row.append(makeSignal(errors[index], maxError));
      const operation = document.createElement("span");
      operation.className = "bp-operation";
      operation.textContent = slopes ? "×" : "→";
      row.append(operation);
      if (slopes) row.append(makeSignal(slopes[index], 1, true));
      else {
        const direct = document.createElement("span");
        direct.className = "bp-direct-derivative";
        direct.textContent = current.calculation.includes("Fused") ? "fused" : "Jacobian";
        row.append(direct);
      }
      const resultArrow = document.createElement("span");
      resultArrow.className = "bp-operation";
      resultArrow.textContent = "→";
      row.append(resultArrow, makeSignal(current.delta[index], maxDelta));
      table.append(row);
    }
    mechanism.append(table);
    if (errors.length > shownNeurons) {
      const omitted = document.createElement("p");
      omitted.className = "bp-omitted-note";
      omitted.textContent = `Showing ${shownNeurons} of ${errors.length} neurons.`;
      mechanism.append(omitted);
    }
    const gradientDisclosure = document.createElement("details");
    gradientDisclosure.className = "bp-matrix-disclosure";
    const gradientSummary = document.createElement("summary");
    gradientSummary.textContent = `Inspect exact gradients · W ${current.gradient_weights.length} × ${current.gradient_weights[0]?.length || 0} · b ${current.gradient_biases.length}`;
    const gradientMatrix = document.createElement("pre");
    gradientMatrix.textContent = `∂L/∂W = ${JSON.stringify(current.gradient_weights.map(row => row.map(value => Number(value).toFixed(4))), null, 2)}\n∂L/∂b = ${JSON.stringify(current.gradient_biases.map(value => Number(value).toFixed(4)))}`;
    gradientDisclosure.append(gradientSummary, gradientMatrix);
    mechanism.append(gradientDisclosure);
    const focus = $("backward-current-step");
    focus.replaceChildren();
    focus.dataset.layer = current.layer;
    const stepLabel = document.createElement("span");
    stepLabel.textContent = `BACKWARD STEP ${String(backwardStep + 1).padStart(2, "0")} / ${String(backwardDetail.backward.length).padStart(2, "0")}`;
    const stepTitle = document.createElement("strong");
    stepTitle.textContent = `${current.layer.toUpperCase()} LAYER`;
    const stepExplain = document.createElement("p");
    stepExplain.textContent = current.calculation;
    const stepProgress = document.createElement("div");
    stepProgress.className = "bp-progress-track";
    const progressFill = document.createElement("i");
    progressFill.style.width = `${((backwardStep + 1) / backwardDetail.backward.length) * 100}%`;
    stepProgress.append(progressFill);
    focus.append(stepLabel, stepTitle, stepExplain, stepProgress);
    renderNetwork({
      mode: "backward", layer: layerIndex, deltas: current.delta,
      weightGradients: current.gradient_weights, biasGradients: current.gradient_biases,
      durationMs,
    });
  }

  async function loadInspectors(sample) {
    if (!sessionId) return;
    selectedSample = sample;
    try {
      const [trace] = await Promise.all([
        api(`/api/sessions/${sessionId}/forward-trace?sample=${sample}`),
        loadBackprop(sample),
      ]);
      forwardTrace = trace;
      forwardStep = 0;
      selectedForwardNeuron = 0;
      renderForwardTrace();
      if (state?.decision_boundary) drawBoundary($("boundary-chart"), { ...state, selectedSample });
      if (replayState) await showReplayFrame(replayIndex);
    } catch (error) { notify(error.message); }
  }

  async function loadReplayManifest() {
    if (!sessionId || !state || state.status === "running" || !state.history.loss.length) return;
    try {
      const manifest = await api(`/api/sessions/${sessionId}/replay`);
      replayFrames = manifest.frames || [];
      replayManifestEpoch = state.epoch;
      $("replay-slider").max = String(Math.max(0, replayFrames.length - 1));
      $("replay-slider").disabled = replayFrames.length < 2;
      $("replay-button").disabled = replayFrames.length < 2;
      $("replay-status").textContent = replayFrames.length > 1
        ? `${replayFrames.length} replay checkpoints · epochs 0–${state.epoch}`
        : "Training is too short to replay yet.";
    } catch (error) { $("replay-status").textContent = error.message; }
  }

  async function showReplayFrame(index) {
    if (!replayFrames.length) return;
    const frame = await api(`/api/sessions/${sessionId}/replay/${index}?sample=${selectedSample}`);
    replayIndex = frame.index;
    replayState = { ...state, ...frame, last_update: null };
    networkPass = { mode: "idle" };
    $("replay-exit").hidden = false;
    $("replay-slider").value = String(replayIndex);
    $("epoch-count").textContent = String(frame.epoch);
    $("progress-fill").style.width = `${Math.min(100, frame.epoch / Math.max(1, epochLimit) * 100)}%`;
    $("replay-status").textContent = `Replay · epoch ${frame.epoch} of ${state.epoch}`;
    const history = frame.history;
    const metricIndex = Math.max(0, frame.epoch - 1);
    $("loss-metric").textContent = frame.epoch ? Number(history.loss[metricIndex]).toFixed(4) : "—";
    $("accuracy-metric").textContent = frame.epoch ? `${(100 * Number(history.accuracy[metricIndex])).toFixed(1)}%` : "—";
    $("validation-loss-metric").textContent = frame.epoch ? Number(history.validation_loss[metricIndex]).toFixed(4) : "—";
    $("validation-accuracy-metric").textContent = frame.epoch ? `${(100 * Number(history.validation_accuracy[metricIndex])).toFixed(1)}%` : "—";
    renderNetwork();
    drawHistory($("history-chart"), history);
    if (frame.decision_boundary) drawBoundary($("boundary-chart"), { ...state, ...frame, selectedSample });
  }

  function stopReplay() {
    clearTimeout(replayTimer);
    replayTimer = null;
    replayPlaying = false;
    $("replay-button").textContent = "↺ Replay training";
  }

  async function startReplay() {
    if (state?.status === "running") { notify("Pause or finish training before replaying it."); return; }
    if (!replayFrames.length) await loadReplayManifest();
    if (replayFrames.length < 2) return;
    stopReplay();
    replayPlaying = true;
    $("replay-button").textContent = "Ⅱ Pause replay";
    await showReplayFrame(0);
    if (replayPlaying) replayTimer = setTimeout(replayNextFrame, Math.round(1300 / trainingVizSpeed));
  }

  async function replayNextFrame() {
    if (!replayPlaying) return;
    if (replayIndex >= replayFrames.length - 1) { stopReplay(); return; }
    try {
      await showReplayFrame(replayIndex + 1);
      if (replayPlaying) replayTimer = setTimeout(replayNextFrame, Math.round(1300 / trainingVizSpeed));
    } catch (error) {
      stopReplay();
      notify(error.message);
      return;
    }
  }

  function exitReplay() {
    stopReplay();
    replayState = null;
    $("replay-exit").hidden = true;
    if (state) render(state);
  }

  function connectProgress() {
    if (!sessionId || progressSocket?.readyState === WebSocket.OPEN) return;
    const apiOrigin = new URL(API_BASE || window.location.origin);
    const protocol = apiOrigin.protocol === "https:" ? "wss:" : "ws:";
    progressSocket = new WebSocket(`${protocol}//${apiOrigin.host}/ws/training/${sessionId}`);
    progressSocket.onopen = () => { $("top-status").textContent = "CONNECTED"; };
    progressSocket.onmessage = event => {
      try { render(JSON.parse(event.data)); }
      catch (error) { console.error("Invalid training update", error); }
    };
    progressSocket.onerror = () => { $("top-status").textContent = "RECONNECTING"; };
    progressSocket.onclose = () => {
      if (!new Set(["complete", "stopped", "error"]).has(state?.status)) {
        $("top-status").textContent = "RECONNECTING";
        setTimeout(connectProgress, 800);
      }
    };
  }

  function reconnectProgress() {
    // A new configuration replaces the training session. Reconnect only after
    // the server has installed it, or this socket will keep reading the old one.
    if (progressSocket) {
      progressSocket.onclose = null;
      progressSocket.close();
    }
    progressSocket = null;
    connectProgress();
  }

  async function loadBackprop(sample) {
    if (!sessionId) return;
    try {
      const detail = await api(`/api/sessions/${sessionId}/backprop?sample=${sample}`);
      backwardDetail = detail;
      backwardStep = 0;
      clearInterval(backwardTimer);
      backwardTimer = null;
      $("backward-play").textContent = "Play";
      const flow = $("forward-flow");
      flow.replaceChildren();
      detail.forward.forEach((values, index) => {
        const layer = document.createElement("div");
        layer.className = "flow-layer";
        const title = document.createElement("span");
        title.textContent = index === 0 ? "INPUT x" : index === detail.forward.length - 1 ? `OUTPUT ŷ` : `HIDDEN ${index}`;
        const data = document.createElement("strong");
        data.textContent = formatVector(values, 2);
        layer.append(title, data);
        flow.append(layer);
        if (index < detail.forward.length - 1) {
          const arrow = document.createElement("span");
          arrow.className = "flow-arrow";
          arrow.textContent = "→";
          flow.append(arrow);
        }
      });
      const predictionText = (Array.isArray(detail.prediction) ? detail.prediction : [detail.prediction]).map(value => Number(value).toFixed(4)).join(", ");
      $("chain-rule").innerHTML = `<span class="rule-symbol">∂</span><div><p>${detail.chain_rule}</p><span>Example #${String(detail.sample_index + 1).padStart(2, "0")} · y = ${detail.y} · ŷ = [${predictionText}] · ${state.model.loss.toUpperCase()} = ${detail.loss.toFixed(4)}</span></div>`;
      const steps = $("backward-steps");
      steps.replaceChildren();
      detail.backward.forEach((step, index) => {
        const card = document.createElement("button");
        card.type = "button";
        card.className = `gradient-step gradient-step-nav${index === 0 ? " gradient-step-current" : ""}`;
        card.setAttribute("aria-pressed", String(index === 0));
        card.setAttribute("aria-label", `Show backward step ${index + 1}, ${step.layer}`);
        const title = document.createElement("span");
        title.textContent = `STEP ${String(index + 1).padStart(2, "0")} · ${step.layer.toUpperCase()}`;
        const rule = document.createElement("strong");
        rule.textContent = step.rule;
        const delta = document.createElement("em");
        delta.textContent = `δ = [${step.delta.map(value => Number(value).toFixed(4)).join(", ")}]`;
        card.append(title, rule, delta);
        card.addEventListener("click", () => selectBackwardStep(index));
        steps.append(card);
      });
      selectBackwardStep(0);
    } catch (error) {
      notify(error.message);
    }
  }

  async function initialize() {
    try {
      const initial = await api("/api/sessions", { method: "POST", body: JSON.stringify({ hidden_layers: hiddenLayers, input_size: 2, output_size: 1, activation: "tanh", learning_rate: .04, optimizer: "Adam", batch_size: 24, dataset: "XOR" }) });
      sessionId = initial.session_id;
      render(initial);
      reflectConfig(initial.model, initial.dataset, initial.batch_size);
      $("signal-button").setAttribute("aria-pressed", String(signalsOn));
      $("signal-button").innerHTML = `Signals ${signalsOn ? "on" : "off"} <span>${signalsOn ? "◉" : "○"}</span>`;
      connectProgress();
      refreshSampleOptions(initial.sample_count ?? initial.points.length);
      await loadInspectors(0);
      $("top-status").textContent = "CONNECTED";
    } catch (error) {
      $("top-status").textContent = "API OFFLINE";
      $("model-status").textContent = "START THE PYTHON API";
      $("train-hint").textContent = error.message;
      notify("Could not connect. Start the FastAPI server and refresh this page.");
    }
  }

  $("add-layer").addEventListener("click", () => paintLayerEditor([...hiddenLayers, 5]));
  document.querySelectorAll("[data-minimize]").forEach(button => button.addEventListener("click", () => togglePanel(button)));
  $("remove-layer").addEventListener("click", () => paintLayerEditor(hiddenLayers.slice(0, -1)));
  $("nav-visualize").addEventListener("click", () => setWorkspaceView("visualize"));
  $("nav-guide").addEventListener("click", () => setWorkspaceView("guide"));
  $("training-viz-speed").value = String(trainingVizSpeed);
  $("training-viz-speed").addEventListener("change", event => {
    trainingVizSpeed = Number(event.target.value);
    try { localStorage.setItem("nnLabTrainingVisualizationSpeed", String(trainingVizSpeed)); } catch {}
    lastNetworkPaint = 0;
    if (replayTimer) { clearTimeout(replayTimer); replayTimer = setTimeout(replayNextFrame, Math.round(1300 / trainingVizSpeed)); }
    if (state) renderNetwork();
  });
  [$("nav-visualize"), $("nav-guide")].forEach((tab, index, tabs) => tab.addEventListener("keydown", event => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 : (index + (event.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length;
    tabs[next].focus();
    setWorkspaceView(next === 0 ? "visualize" : "guide");
  }));
  setWorkspaceView(workspaceView);
  $("lesson-prev").addEventListener("click", () => { lessonStep = Math.max(0, lessonStep - 1); renderLesson(true); });
  $("lesson-next").addEventListener("click", () => {
    if (lessonStep >= lessons.length - 1) { setWorkspaceView("visualize"); notify("Guide complete. Explore the network freely."); return; }
    lessonStep += 1;
    renderLesson(true);
  });
  $("challenge-next").addEventListener("click", () => { challengeIndex = (challengeIndex + 1) % challenges.length; renderChallenge(); });
  $("challenge-apply").addEventListener("click", loadChallenge);
  ["neuron-x1", "neuron-w1", "neuron-x2", "neuron-w2", "neuron-bias", "neuron-activation"].forEach(id => $(id).addEventListener("input", renderSingleNeuron));
  $("update-layer").addEventListener("change", event => { lastUpdateLayer = Number(event.target.value); renderUpdateExplanation(); renderNetwork(); });
  $("compare-experiments").addEventListener("click", () => { populateComparison(); $("comparison-dialog").showModal(); });
  $("comparison-a").addEventListener("change", renderComparison);
  $("comparison-b").addEventListener("change", renderComparison);
  $("comparison-close").addEventListener("click", () => $("comparison-dialog").close());
  $("learning-rate").addEventListener("input", event => { $("lr-value").textContent = Number(event.target.value).toFixed(3); });
  $("noise").addEventListener("input", event => { $("noise-value").textContent = Number(event.target.value).toFixed(2); });
  $("validation-split").addEventListener("input", event => { $("split-value").textContent = `${Math.round(Number(event.target.value) * 100)}%`; });
  $("epoch-edit").addEventListener("click", () => {
    const value = window.prompt("Stop training at which epoch?", String(epochLimit));
    if (value === null) return;
    const parsed = Math.floor(Number(value));
    if (!Number.isFinite(parsed) || parsed < 1 || parsed > 10000) return notify("Choose an epoch limit from 1 to 10,000.");
    epochLimit = parsed;
    if (state) render(state);
  });
  $("apply-config").addEventListener("click", async () => {
    try {
      if (!sessionId) throw new Error("Wait for the Python API to connect before applying settings.");
      const config = getConfig();
      $("apply-config").disabled = true;
      exitReplay();
      replayFrames = [];
      replayManifestEpoch = null;
      $("replay-button").disabled = true;
      $("replay-status").textContent = "Preparing training replay…";
      const updated = await api(`/api/sessions/${sessionId}/config`, { method: "PUT", body: JSON.stringify(config) });
      configDirty = false;
      render(updated);
      reconnectProgress();
      reflectConfig(updated.model, updated.dataset, updated.batch_size);
      refreshSampleOptions(updated.sample_count ?? updated.points.length);
      $("sample-select").value = "0";
      await loadInspectors(0);
      notify("Configuration applied. The network is ready to train.");
    } catch (error) { $("apply-config").disabled = false; notify(error.message); }
  });
  $("apply-dimensions").addEventListener("click", () => $("apply-config").click());
  $("train-button").addEventListener("click", async () => {
    if (state?.status === "running") return;
    try {
      exitReplay();
      if (state?.status === "paused") await api(`/api/sessions/${sessionId}/resume`, { method: "POST", body: JSON.stringify({ max_epochs: epochLimit }) });
      else await api(`/api/sessions/${sessionId}/run`, { method: "POST", body: JSON.stringify({ max_epochs: epochLimit }) });
      state.status = "running";
      setRunning(true);
      connectProgress();
    } catch (error) { notify(error.message); }
  });
  $("pause-button").addEventListener("click", async () => {
    if (!sessionId) return;
    try {
      await api(`/api/sessions/${sessionId}/pause`, { method: "POST" });
      state.status = "paused";
      setRunning(false);
    } catch (error) { notify(error.message); }
  });
  $("stop-button").addEventListener("click", async () => {
    try {
      const result = await api(`/api/sessions/${sessionId}/stop`, { method: "POST" });
      state.status = result.status;
      setRunning(false);
    } catch (error) { notify(error.message); }
  });
  $("reset-button").addEventListener("click", async () => {
    try {
      exitReplay();
      replayFrames = [];
      replayManifestEpoch = null;
      const fresh = await api(`/api/sessions/${sessionId}/reset`, { method: "POST" });
      render(fresh);
      $("sample-select").value = "0";
      await loadInspectors(0);
      notify("Weights and training history reset.");
    } catch (error) { notify(error.message); }
  });
  $("replay-button").addEventListener("click", () => {
    if (replayPlaying) { stopReplay(); return; }
    startReplay().catch(error => { stopReplay(); notify(error.message); });
  });
  $("replay-exit").addEventListener("click", exitReplay);
  $("replay-slider").addEventListener("input", event => {
    stopReplay();
    showReplayFrame(Number(event.target.value)).catch(error => notify(error.message));
  });
  $("dataset").addEventListener("change", event => {
    markConfigDirty();
    if (event.target.value === "Clusters" && Number($("output-size").value) < 3) $("output-size").value = "3";
    $("dataset-name").textContent = event.target.value;
    refreshDimensionHint();
  });
  ["input-size", "output-size"].forEach(id => {
    const onDimensionEdit = () => {
      markConfigDirty();
      refreshDimensionHint();
    };
    // Listen for typing and spinner clicks. Some browsers report a number-field
    // change only after the field loses focus, so handle that event too.
    $(id).addEventListener("input", onDimensionEdit);
    $(id).addEventListener("change", onDimensionEdit);
  });
  $("sample-select").addEventListener("change", event => loadInspectors(Number(event.target.value)));
  function inspectNetworkTarget(target) {
    if (!state) return;
    const node = target.closest("[data-inspect-neuron]");
    const edge = target.closest("[data-inspect-edge]");
    if (!node && !edge) return;
    const model = replayState?.model || state.model;
    const inspector = $("network-inspector");
    if (node) {
      const layer = Number(node.dataset.layer), neuron = Number(node.dataset.neuron);
      const layerName = layer === 0 ? "Input" : layer === model.architecture.length - 1 ? "Output" : `Hidden ${layer}`;
      const activation = (replayState?.sample_activations || state.sample_activations)?.[layer]?.[neuron];
      const bias = layer === 0 ? null : model.biases[layer - 1]?.[neuron];
      const backwardRow = !replayState && layer > 0 ? backwardDetail?.backward?.[backwardDetail.backward.length - layer] : null;
      const details = [`Activation: ${activation === undefined ? "—" : Number(activation).toFixed(5)}`];
      if (bias !== null) details.push(`Bias b: ${Number(bias).toFixed(5)}`);
      if (backwardRow?.delta?.[neuron] !== undefined) details.push(`Backprop delta δ: ${Number(backwardRow.delta[neuron]).toFixed(5)}`);
      if (backwardRow?.gradient_biases?.[neuron] !== undefined) details.push(`Bias gradient ∂L/∂b: ${Number(backwardRow.gradient_biases[neuron]).toFixed(5)}`);
      $("network-inspector-kind").textContent = `${layerName.toUpperCase()} NEURON ${neuron + 1}`;
      $("network-inspector-title").textContent = "Neuron values";
      $("network-inspector-value").textContent = details.join("   ·   ");
    } else {
      const layer = Number(edge.dataset.layer), source = Number(edge.dataset.source), destination = Number(edge.dataset.target);
      const weight = model.weights[layer]?.[source]?.[destination];
      const backwardRow = !replayState ? backwardDetail?.backward?.[backwardDetail.backward.length - 1 - layer] : null;
      const gradient = backwardRow?.gradient_weights?.[source]?.[destination];
      const update = !replayState ? state.last_update?.layers?.[layer] : null;
      const before = update?.weights_before?.[source]?.[destination];
      const after = update?.weights_after?.[source]?.[destination];
      const details = [`Weight: ${Number(weight).toFixed(5)}`];
      if (gradient !== undefined) details.push(`Gradient ∂L/∂W: ${Number(gradient).toFixed(5)}`);
      if (before !== undefined && after !== undefined) details.push(`Last update: ${Number(before).toFixed(4)} → ${Number(after).toFixed(4)}`);
      $("network-inspector-kind").textContent = `CONNECTION · LAYER ${layer + 1}`;
      $("network-inspector-title").textContent = `Neuron ${source + 1} → neuron ${destination + 1}`;
      $("network-inspector-value").textContent = details.join("   ·   ");
    }
    inspector.hidden = false;
  }
  $("network-view").addEventListener("click", event => inspectNetworkTarget(event.target));
  $("network-view").addEventListener("keydown", event => {
    if (event.key === "Enter" || event.key === " ") { event.preventDefault(); inspectNetworkTarget(event.target); }
  });
  $("network-inspector-close").addEventListener("click", () => { $("network-inspector").hidden = true; });

  function closestBoundarySample(event) {
    if (!state?.points?.length) return null;
    const rect = $("boundary-chart").getBoundingClientRect();
    const plot = { x: 28, y: 7, w: rect.width - 37, h: rect.height - 17 };
    const pointer = { x: event.clientX - rect.left, y: event.clientY - rect.top };
    let nearest = { index: -1, distance: Infinity };
    state.points.forEach((point, index) => {
      const x = plot.x + ((point[0] + 1.2) / 2.4) * plot.w;
      const y = plot.y + (1 - (point[1] + 1.2) / 2.4) * plot.h;
      const distance = Math.hypot(pointer.x - x, pointer.y - y);
      if (distance < nearest.distance) nearest = { index, distance };
    });
    return nearest;
  }
  $("boundary-chart").addEventListener("pointermove", event => {
    const nearest = closestBoundarySample(event);
    const canvas = $("boundary-chart");
    const hit = nearest && nearest.distance <= 12;
    canvas.style.cursor = hit ? "pointer" : "crosshair";
    canvas.title = hit ? `Example #${String(nearest.index + 1).padStart(2, "0")} · click to inspect this sample` : "Click near a sample point to inspect it in the network";
  });
  $("boundary-chart").addEventListener("click", event => {
    const nearest = closestBoundarySample(event);
    if (!nearest || nearest.distance > 14) return;
    $("sample-select").value = String(nearest.index);
    loadInspectors(nearest.index);
  });

  $("history-chart").addEventListener("pointermove", event => {
    const history = replayState?.history || state?.history;
    const count = history?.loss?.length || 0;
    if (!count) return;
    const rect = $("history-chart").getBoundingClientRect();
    const fraction = Math.max(0, Math.min(1, (event.clientX - rect.left - 37) / Math.max(1, rect.width - 45)));
    const epochIndex = Math.round(fraction * Math.max(0, count - 1));
    $("history-chart").title = `Epoch ${epochIndex + 1} · training loss ${Number(history.loss[epochIndex]).toFixed(4)} · click to replay this checkpoint`;
  });
  $("history-chart").addEventListener("click", async event => {
    if (!state) return;
    if (state.status === "running") { notify("Pause training before replaying a checkpoint."); return; }
    const count = state.history.loss.length;
    if (!count) { notify("Train the model first to create a learning curve."); return; }
    if (!replayFrames.length) await loadReplayManifest();
    if (!replayFrames.length) { notify("No replay checkpoints are available for this run."); return; }
    const rect = $("history-chart").getBoundingClientRect();
    const fraction = Math.max(0, Math.min(1, (event.clientX - rect.left - 37) / Math.max(1, rect.width - 45)));
    const epoch = Math.round(fraction * Math.max(0, count - 1)) + 1;
    const checkpoint = replayFrames.reduce((best, frame, index) => Math.abs(frame.epoch - epoch) < Math.abs(replayFrames[best].epoch - epoch) ? index : best, 0);
    showReplayFrame(checkpoint).catch(error => notify(error.message));
  });
  $("signal-button").addEventListener("click", () => {
    signalsOn = !signalsOn;
    $("signal-button").setAttribute("aria-pressed", String(signalsOn));
    $("signal-button").innerHTML = `Signals ${signalsOn ? "on" : "off"} <span>${signalsOn ? "◉" : "○"}</span>`;
    renderNetwork();
  });
  $("show-forward-view").addEventListener("click", () => renderNetwork({ mode: "forward", step: forwardStep }));
  $("show-backward-view").addEventListener("click", () => {
    if (backwardDetail?.backward?.length) selectBackwardStep(backwardStep);
    else loadBackprop(selectedSample);
  });
  window.matchMedia("(prefers-reduced-motion: reduce)").addEventListener("change", event => {
    if (event.matches) signalsOn = false;
    $("signal-button").setAttribute("aria-pressed", String(signalsOn));
    $("signal-button").innerHTML = `Signals ${signalsOn ? "on" : "off"} <span>${signalsOn ? "◉" : "○"}</span>`;
    renderNetwork();
  });
  $("export-json").addEventListener("click", async () => {
    try {
      const data = await api(`/api/sessions/${sessionId}/export`);
      download(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }), "neural-network-model.json");
      notify("Model weights, biases, and training history exported.");
    } catch (error) { notify(error.message); }
  });
  $("export-csv").addEventListener("click", () => {
    if (!state) return;
    const rows = ["epoch,train_loss,train_accuracy,validation_loss,validation_accuracy", ...state.history.loss.map((loss, i) => `${i + 1},${loss},${state.history.accuracy[i]},${state.history.validation_loss[i]},${state.history.validation_accuracy[i]}`)];
    download(new Blob([rows.join("\n")], { type: "text/csv" }), "training-history.csv");
  });
  $("forward-reset").addEventListener("click", () => { clearInterval(forwardTimer); forwardTimer = null; forwardStep = 0; renderForwardTrace(); });
  $("forward-prev").addEventListener("click", () => { forwardStep = Math.max(0, forwardStep - 1); renderForwardTrace(); });
  $("forward-next").addEventListener("click", () => { forwardStep = Math.min(forwardTrace?.layers.length || 0, forwardStep + 1); renderForwardTrace(); });
  $("forward-play").addEventListener("click", () => {
    if (forwardTimer) { clearInterval(forwardTimer); forwardTimer = null; $("forward-play").textContent = "Play"; return; }
    startForwardPlayback(true);
  });
  $("forward-speed").addEventListener("change", () => { if (forwardTimer) startForwardPlayback(); });
  $("backward-reset").addEventListener("click", () => selectBackwardStep(0));
  $("backward-prev").addEventListener("click", () => selectBackwardStep(backwardStep - 1));
  $("backward-next").addEventListener("click", () => selectBackwardStep(backwardStep + 1));
  $("backward-play").addEventListener("click", () => {
    if (backwardTimer) { clearInterval(backwardTimer); backwardTimer = null; $("backward-play").textContent = "Play"; return; }
    startBackwardPlayback();
  });
  $("backward-speed").addEventListener("change", () => { if (backwardTimer) startBackwardPlayback(); });
  $("gradient-check-button").addEventListener("click", async () => {
    try {
      $("gradient-result").textContent = "Checking finite differences…";
      const result = await api(`/api/sessions/${sessionId}/gradient-check?sample=${selectedSample}`);
      $("gradient-result").textContent = `${result.passed ? "PASS" : "CHECK"} · ${result.checks.length} parameters · max abs error ${result.max_absolute_error.toExponential(2)} · max relative error ${result.max_relative_error.toExponential(2)}`;
      $("gradient-result").className = result.passed ? "check-pass" : "check-fail";
    } catch (error) { $("gradient-result").textContent = error.message; }
  });
  $("apply-backprop-update").addEventListener("click", async () => {
    const button = $("apply-backprop-update");
    const status = $("apply-update-status");
    button.disabled = true;
    status.textContent = "Applying the selected example's gradients…";
    try {
      const updated = await api(`/api/sessions/${sessionId}/backprop/update?sample=${selectedSample}`, { method: "POST" });
      replayFrames = [];
      replayManifestEpoch = null;
      render(updated);
      await loadInspectors(selectedSample);
      status.textContent = `One ${updated.model.optimizer} update applied from example #${selectedSample + 1}. See W before/after and bias changes above.`;
      notify("One backpropagation update applied to the weights and biases.");
    } catch (error) {
      status.textContent = error.message;
    } finally {
      button.disabled = state?.status === "running";
    }
  });
  $("save-experiment").addEventListener("click", async () => {
    try {
      const name = window.prompt("Name this experiment", `${state.dataset} · ${state.model.optimizer}`);
      if (!name) return;
      const experiment = await api(`/api/sessions/${sessionId}/export`);
      const saved = JSON.parse(localStorage.getItem("nnLabExperiments") || "[]");
      saved.push({ name: name.slice(0, 80), savedAt: new Date().toISOString(), experiment });
      localStorage.setItem("nnLabExperiments", JSON.stringify(saved.slice(-30)));
      notify(`Saved locally. ${Math.min(saved.length, 30)} experiment(s) on this browser.`);
    } catch (error) { notify(error.message); }
  });
  $("import-json").addEventListener("click", () => $("import-file").click());
  $("import-file").addEventListener("change", async event => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      if (file.size > 2_000_000) throw new Error("Model file must be smaller than 2 MB.");
      const document = JSON.parse(await file.text());
      const imported = await api("/api/sessions/import", { method: "POST", body: JSON.stringify(document) });
      progressSocket?.close();
      sessionId = imported.session_id;
      configDirty = false;
      render(imported);
      reflectConfig(imported.model, imported.dataset, imported.batch_size);
      replayFrames = [];
      replayManifestEpoch = null;
      refreshSampleOptions(imported.sample_count ?? imported.points.length);
      $("sample-select").value = "0";
      await loadInspectors(0);
      connectProgress();
      notify("Model imported. Optimizer state starts fresh; exported parameters and history were loaded.");
    } catch (error) { notify(error.message); }
    event.target.value = "";
  });
  window.addEventListener("resize", () => { if (state) { drawHistory($("history-chart"), replayState?.history || state.history); const boundaryState = replayState?.decision_boundary ? replayState : state; if (boundaryState.decision_boundary) drawBoundary($("boundary-chart"), { ...boundaryState, selectedSample }); } });

  function download(blob, filename) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url; link.download = filename; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  renderChallenge();
  renderSingleNeuron();
  initialize();
})();
