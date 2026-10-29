# Neural Network Lab — From-Scratch ML Visualizer

An educational neural-network playground that exposes the calculations of a NumPy model. FastAPI owns model state and trains in a background worker. A vanilla JavaScript frontend renders REST and WebSocket results; it does not fabricate model parameters or predictions.

For setup and a complete walkthrough, see the [User Guide](USER_GUIDE.md).

## Features

- Up to four hidden layers, 2–128 neurons per hidden layer; configurable input and output sizes for supported datasets.
- ReLU, leaky ReLU, tanh, or sigmoid hidden activation.
- Binary sigmoid/BCE and multiclass softmax/categorical-cross-entropy output.
- Vanilla gradient descent, SGD, momentum, and Adam optimizers.
- Auto, Xavier, He, or normal weight initialization.
- Learning rate, batch size, seed, samples, data noise, validation split, dropout, L2, early stopping, and plateau learning-rate scheduling.
- Deterministic XOR, linear, two-moons, circles, spiral, and three-class cluster datasets.
- Background training with pause, resume, stop, reset, error status, and WebSocket updates.
- Live weights, biases, activations, decision regions, and separate train/validation loss and accuracy plots.
- Forward trace, backpropagation gradients, and central finite-difference gradient checks.
- Separate Visualize and Learning guide tabs, an eight-step beginner walkthrough, a single-neuron sandbox, and mini challenges.
- Adjustable independent playback speeds for forward and backward propagation, plus a separate live training visualization speed.
- Actual before/after parameter values and gradients from the latest optimizer update.
- Bounded training replay with recorded network parameters, learning curves, and decision-boundary checkpoints.
- Versioned JSON model/history export and validated import; CSV history export.
- Named experiment snapshots in this browser's localStorage.

## Architecture

~~~mermaid
flowchart LR
  UI[Browser: HTML, CSS, JS modules] -->|REST JSON| API[FastAPI]
  UI <-->|WebSocket progress| API
  API --> S[Session state and controls]
  S --> T[Background trainer]
  T --> E[NumPy engine: forward, gradients, optimizer]
  E --> D[Deterministic local datasets]
  S -->|weights, metrics, traces| UI
~~~

~~~mermaid
stateDiagram-v2
  [*] --> ready
  ready --> running: start
  running --> paused: pause
  paused --> running: resume
  running --> stopped: stop
  paused --> stopped: stop
  running --> complete: epoch limit or early stopping
  running --> error: worker exception
  stopped --> running: resume
  complete --> ready: reset
  stopped --> ready: reset
  error --> ready: reset
~~~

## Neural-network math

Batches are row-major: X has shape (batch, input_features), W has shape (input_features, output_features), and b has shape (1, output_features). A dense layer computes Z = X @ W + b. Backpropagation averages once across the batch: dW = A_prev.T @ dZ / batch_size + l2 * W; db = mean(dZ, axis=0). The L2 term in the objective is 0.5 * l2 * sum(W²).

The engine uses fused sigmoid/BCE and softmax/categorical-cross-entropy gradients (dZ = prediction - target). Softmax subtracts the largest logit before exponentiation; its general derivative is implemented as a vector-Jacobian product. MSE supports sigmoid or softmax outputs. The gradient checker uses central finite differences for every weight and bias and reports absolute and relative errors.

Auto initialization chooses He for ReLU-family activations and Xavier otherwise. Dropout uses inverted masks only during training and is disabled for inference/validation. The plateau scheduler reduces the learning rate after a validation-loss plateau; early stopping restores the best validation weights.

## Project layout

~~~text
backend/
  activations.py       stable activations and VJPs
  losses.py            BCE, categorical cross-entropy, MSE, accuracy
  neural_network.py    NumPy forward/backward, serialization, gradient check
  optimizers.py        GD, SGD, momentum, Adam
  trainer.py           datasets, splits, scheduling, early stopping
  main.py              REST/WebSocket API, background workers, static serving
  tests/               math, training, serialization, API lifecycle tests
frontend/
  index.html
  style.css
  app.js               ES module app, REST/WebSocket integration
  network.js           SVG network diagram
  charts.js            Canvas charts and decision boundary
  config.js            Optional separately hosted API origin
  e2e/                 Playwright browser-level learning-flow checks
  nginx.conf           static server and API/WebSocket reverse proxy
Dockerfile
docker-compose.yml
package.json
package-lock.json
playwright.config.js
requirements.txt
requirements-dev.txt
~~~

## Local development

Python 3.10 or newer is supported. From the project root:

~~~powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m uvicorn backend.main:app --reload
~~~

Open http://127.0.0.1:8000. Interactive API documentation is at http://127.0.0.1:8000/docs.

Run the tests:

~~~powershell
python -m pytest backend/tests -q
~~~

Run the browser flow tests (requires Node.js and a Playwright Chromium install):

~~~powershell
npm ci
npx playwright install chromium
npm run test:e2e
~~~

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | /api/health | Health and active-session count |
| GET | /api/datasets | Dataset catalog |
| POST | /api/sessions | Create model, dataset, and split |
| GET | /api/sessions/{id} | Model state, samples, metrics, and decision grid |
| PUT | /api/sessions/{id}/config | Apply architecture/training settings |
| POST | /api/sessions/{id}/predict | Predict probabilities for 2D input |
| GET | /api/sessions/{id}/forward-trace?sample=0 | Layer-by-layer forward calculations |
| GET | /api/sessions/{id}/backprop?sample=0 | Per-layer deltas and gradients |
| GET | /api/sessions/{id}/gradient-check?sample=0 | Analytic versus numeric gradients |
| POST | /api/sessions/{id}/run | Start background training |
| POST | /api/sessions/{id}/pause | Pause after the current epoch |
| POST | /api/sessions/{id}/resume | Continue a paused/stopped run |
| POST | /api/sessions/{id}/stop | Stop after the current epoch |
| POST | /api/sessions/{id}/reset | Reinitialize using the configured seed |
| GET | /api/sessions/{id}/export | Export model parameters and history |
| POST | /api/sessions/import | Validate and import a JSON experiment |
| WebSocket | /ws/training/{id} | Live backend state and metrics |

## Model files and experiments

Exports use the JSON envelope neural-network-lab/experiment-v1. The model object uses neural-network-lab/model-v1 and contains architecture, hyperparameters, weights, and biases. The training object contains dataset settings and metric history. Import validates dimensions, finite values, and output/dataset compatibility. Optimizer moments are not exported; an imported model starts a fresh optimizer and cannot exactly resume an interrupted Adam trajectory.

Named snapshots are stored in this browser's localStorage, up to 30 entries. They do not sync across browsers or devices. JSON export is the portable format.

## Docker

To set non-default CORS origins, copy .env.example to .env and edit CORS_ORIGINS. Then run:

~~~powershell
docker compose up --build
~~~

Open http://localhost:8080 for the app and http://localhost:8000/docs for API docs. Stop the containers with docker compose down.

## Free demo deployment

For a small portfolio demo, deploy the frontend as a static site and the API as a single Web Service on a free container host. For example, Render documents free static sites and Python web services, supports inbound WebSockets, and spins free services down after 15 minutes without inbound HTTP requests or WebSocket messages. A cold start can take about a minute, and the filesystem is ephemeral, so in-memory sessions and saved service files do not survive restarts or spin-down. See [Render's free instance limits](https://render.com/docs/free) and [WebSocket guidance](https://render.com/docs/websocket).

For a split deployment, set frontend/config.js apiBase to the deployed API origin (for example, https://your-api.example.com), set the API's CORS_ORIGINS to the static site's origin, and use a start command that binds the platform-provided port, such as uvicorn backend.main:app --host 0.0.0.0 --port $PORT --workers 1. The frontend converts HTTPS API origins to secure WebSockets automatically. Keep one API worker because sessions currently live in process memory.

## CI and deployment

GitHub Actions runs the Python suite and Playwright browser flows on pushes and pull requests. Docker Compose runs a single backend worker because session state is in process memory. A small container or VM host with WebSocket support can run the app. Free tiers may sleep, impose CPU/memory caps, or discard ephemeral storage. Sleeping interrupts active experiments. Add persistent storage and shared worker coordination before scaling to multiple replicas or hosting multi-user sessions. No deployment URL is configured.

## Limitations and future work

- This is an educational prototype, not a hardened multi-tenant service. Sessions are in memory, unauthenticated, and lost on restart.
- The browser is a 2D classification UI. The NumPy engine accepts general network widths, but regression and arbitrary-dimensional data editing are not provided.
- Hidden activation, dropout, and L2 are global settings rather than per-layer settings.
- SVG pan/zoom, selectable gradient visualization, progressive edge filtering, and parameter-by-parameter optimizer replay remain future work.
- Browser-local snapshots can be compared on this device; use JSON export to move them to another browser or computer.
- CORS is restricted to configured origins. Set CORS_ORIGINS to a comma-separated allowlist when frontend/API origins differ. Do not use a wildcard in a public deployment.

## Screenshots

Add a current screenshot after capturing it from a running instance. No screenshot or benchmark is included yet.
