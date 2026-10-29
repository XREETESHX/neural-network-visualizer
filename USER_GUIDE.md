# Neural Network Lab — User Guide

This guide is for someone opening Neural Network Lab for the first time. It explains how to start the app, run a first experiment, use each panel, and understand the results.

## 1. What this project is

Neural Network Lab is an interactive teaching tool for small neural networks. You can change a network and its training settings, then watch a NumPy model learn to classify points. The browser displays values returned by the Python backend, including predictions, weights, biases, activations, gradients, training metrics, and decision regions.

Use it to explore questions such as:

- How does adding a hidden layer change a model?
- What happens when the learning rate or optimizer changes?
- How do weights and biases affect a prediction?
- How does backpropagation calculate gradients?
- How do loss, accuracy, and the decision boundary change during training?

The built-in datasets are small, generated, two-dimensional classification examples. This is an educational visualizer, not a tool for training large production models or uploading arbitrary datasets.

## 2. Start the application

The app has a Python API and a browser interface. Start the API first; it serves the interface and performs the model calculations.

### Windows: run locally

1. Install Python 3.10 or newer.
2. Open PowerShell in the project folder.
3. Create an environment and install the app dependencies:

   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

4. Start the server:

   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
   ```

5. Open **http://127.0.0.1:8000/** in your browser.

Keep the PowerShell window open while using the app. Press **Ctrl+C** in that window to stop the server.

The API documentation is available at **http://127.0.0.1:8000/docs**.

Use the header tabs to move between **Visualize** and **Learning guide**. Visualize keeps the model graph, learning curves, decision boundary, forward pass, and backpropagation together. Learning guide has the eight-step walkthrough, mini challenges, and interactive single-neuron sandbox. Pressing **Next idea** scrolls the relevant control into view; steps that need the graph highlight the **Visualize** tab. Network settings remain available in the left sidebar in either view, and the selected tab is remembered in this browser.

### macOS or Linux: run locally

From the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --reload
```

Open **http://127.0.0.1:8000/**. Keep the terminal open while using the app; press **Ctrl+C** to stop it.

### Run with Docker

If Docker Desktop and Docker Compose are installed, run this in the project folder:

```powershell
docker compose up --build
```

Open **http://localhost:8080/**. The API documentation is at **http://localhost:8000/docs**. Stop the containers with **Ctrl+C**, then run `docker compose down` if you want to remove them.

## 3. First experiment: train XOR

The initial setup is ready for a simple XOR classification experiment.

1. Leave the dataset set to **XOR**.
2. Keep the default architecture, **2 — 5 — 4 — 1**. This means two input values, two hidden layers with five and four neurons, and one output.
3. Keep the default **tanh** activation, **Adam** optimizer, and learning rate of **0.04**.
4. Click **Apply configuration**.
5. Click **Train**. Watch the epoch count, learning curves, network values, and decision boundary update.
6. Click **Pause** to pause after the current training step. Click **Resume** to continue. Click **Stop** to stop and keep the current model.
7. Use **Reset** to initialize the model again and clear its training history.

The default epoch limit is 1,000. Click the pencil beside the epoch limit to set a different limit, from 1 to 10,000. Training can also stop early if the validation loss stops improving.

Open **Play with one neuron** to change inputs, weights, bias, and activation while the weighted sum and output update immediately. The challenge card can load suggested settings for a small task.

## 4. Configure a network

### Architecture

- Toy datasets use two input coordinates by default. You can set 1–16 input features; sizes above two are deterministic features derived from those coordinates.
- Add up to four hidden layers with **Add layer**.
- Remove the last hidden layer with **Remove**.
- Each hidden layer can have 2–128 neurons.
- Choose the output width too. Binary datasets support one sigmoid output or a softmax output width of at least two; Clusters needs at least three. When you choose extra outputs, the extra classes are left unused by the dataset labels.

### Training setup

- **Dataset:** Choose XOR, Linear, Moons, Circles, Spiral, or Clusters.
- **Samples:** Set how many generated points to use. More points usually make each epoch take longer.
- **Random seed:** Reuse the same seed to regenerate the same initial model and dataset settings.
- **Dataset noise:** Add variation to the generated examples. Higher noise can make classification harder.
- **Validation split:** Reserve part of the data to evaluate the model while it trains. Validation examples are not used for weight updates.
- **Hidden activation:** Choose tanh, ReLU, leaky ReLU, or sigmoid.
- **Learning rate:** Set the size of the optimizer's weight updates. If loss jumps around or becomes very large, try lowering it.
- **Optimizer:** Choose Adam, SGD, Momentum, or Vanilla GD.
- **Batch size:** Choose how many training examples are processed before each weight update.
- **Initialization:** Choose Auto, Xavier, He, or Normal to initialize the weights.
- **Dropout:** Randomly disables hidden activations during training to reduce overfitting. It is disabled when the model makes predictions.
- **L2 regularization:** Penalizes large weights during training. It is off by default.

Click **Apply configuration** after changing settings. This clears the previous training history. A change to the architecture, activation, output shape, initialization, or seed reinitializes weights; changes to dataset, learning rate, optimizer, batch size, dropout, or L2 keep weights when the network shape and activation remain compatible. If training is running, stop or pause it before applying settings.

After a run has paused or completed, use **Replay training** and the slider to inspect recorded checkpoints. Replay restores the actual weights and biases from each checkpoint and updates the graph, learning curves, and toy-data decision boundary. Checkpoints are sampled across the run to keep memory bounded; this is an epoch-level replay, not a replay of every individual optimizer update.

## 5. Read the visualizations

### Network graph

- Each column represents a layer; each circle represents a neuron.
- The value in a neuron is its activation for the currently selected example.
- The `b:` value is that neuron's bias.
- Each connecting line is a weight. Blue lines are positive; red lines are negative. Thicker lines have larger absolute weights.
- The current forward or backward step is highlighted on the graph. During the forward pass, signal moves toward the output; during backpropagation, the active layer and its delta values are highlighted as the view steps backward.
- While training, the latest mini-batch update highlights changed weights in the selected layer and shows each weight's delta. The selected neurons show bias deltas; hover a connection or neuron for the exact before/after values and gradient. The graph status line shows a representative weight and bias update, while the **last batch update** panel shows full detail. Expand its matrix to see every updated value for that layer.
- Use **Live updates** beside the graph controls to slow down or speed up how often the training graph refreshes and how long parameter changes are highlighted. This changes only the visualization; it does not change the optimizer or training rate.
- Use the **Forward pass** and **Backpropagation** buttons above the graph to choose which direction it displays. Detailed backward steps are controlled in the Backpropagation panel below.
- Hover a line or neuron to see its exact weight, activation, or bias.
- Use **Signals on/off** to toggle animated signals. On narrow screens, scroll the network panel horizontally to see the full graph.

### Learning curves and metrics

- **Training loss** shows the model's error on examples used for weight updates. Lower is generally better.
- **Validation loss** shows error on held-out examples. It helps reveal whether the model is generalizing beyond its training points.
- **Training accuracy** and **validation accuracy** show the share of correctly classified examples.
- The sample count and current loss/accuracy are shown above the chart.

Accuracy can change in steps, and small fluctuations are normal. A model can have low training loss but noticeably worse validation performance if it is overfitting.

### Decision boundary

The background colors show which class the current model predicts at each location. Dots are the generated examples. Watch this panel during training to see the boundary move. The plotted range is limited to the displayed two-dimensional input area.

## 6. Inspect a prediction and its gradients

### Forward pass

The **Step through a prediction** panel shows one example moving through each layer. Use the **Speed** menu to slow playback to 0.5× or speed it up to 1.5× or 2×. Changing speed while playback is running takes effect immediately.

- **Reset** returns to the input step.
- **Prev** and **Next** move one step at a time.
- **Play** advances automatically; click it again to pause.
- The selected layer shows its input, pre-activation `z`, and output activation `a`.
- Expand **Weights and biases** to inspect the complete parameter values used by that layer.

### Backpropagation and chain rule

Choose an example from the **Example** menu. The panel shows its input, activations, prediction, target, loss, and layer-by-layer gradients. Expand a gradient matrix to inspect every value of `∂L/∂W`.

Use **Start**, **Prev**, **Next**, and **Play** in the **Backward flow** controls to step through the output layer's error and then each hidden layer's gradients. The network graph follows the selected backward step and displays the active layer's `δ` values beside its neurons. Its **Speed** menu controls backward playback independently from the forward pass.

**Run gradient check** compares the model's analytic gradients with central finite-difference estimates. A passing result means the two calculations agree within the check's tolerance. Gradient checking is unavailable while training is running and when dropout is enabled. Set dropout to **Off**, apply the configuration, then run the check.

## 7. Save, import, and export

- **Save experiment** stores a named snapshot in this browser's local storage. It is local to this browser and does not sync to another computer.
- **Compare runs** shows saved settings, architecture, epoch count, loss, and accuracy side by side. Save at least two experiments first.
- **Export model + history** downloads a JSON file containing model parameters, settings, and training history.
- **Export CSV** downloads the recorded loss, accuracy, and learning-rate history.
- **Import model** loads a previously exported JSON experiment into a new session.

Imported weights and history are restored, but optimizer moments such as Adam's internal state are not included. Therefore, continuing after import will not exactly reproduce an interrupted training run.

## 8. Common problems

### “API offline” or “Could not connect”

- Confirm the FastAPI server is running in a terminal.
- Open the exact address served by the backend: **http://127.0.0.1:8000/** for local mode or **http://localhost:8080/** for Docker mode.
- Do not open `frontend/index.html` directly as a file.
- If port 8000 is already in use, stop the other process or start this app on another port and open that port in the browser.

### A page looks old or its styles are missing

Hard-refresh the browser with **Ctrl+F5**. For Docker, rebuild the frontend image with `docker compose up --build`.

### Training appears to stop

Check the status shown in the app and the server terminal. Training can finish when it reaches the epoch limit or early-stopping condition. **Reset** starts a fresh run. If the status reports an error, read the message in the training hint and the backend terminal.

### Network graph is cut off on a small screen

Scroll horizontally inside the network panel. The diagram keeps a readable minimum width so its neurons and connections remain visible.

### Gradient check is unavailable

Pause or stop training, set Dropout to Off, and apply the configuration. Applying a configuration starts a fresh model, so run gradient checking before or after training as needed.

## 9. What this project does not do

- It does not load arbitrary CSV datasets or images; it uses the included generated datasets.
- The visualizer focuses on classification, not general regression or large-scale deep learning.
- Sessions live in server memory and are lost when the server restarts.
- Saved experiment names live in the browser that saved them. Use JSON export to move a model to another browser or computer.
- It has no user accounts or multi-user access controls; run it locally for learning and demos.

## 10. Project components

- `backend/neural_network.py` implements the NumPy neural network and gradient calculations.
- `backend/trainer.py` creates datasets and runs training sessions.
- `backend/main.py` exposes the backend through FastAPI and serves the frontend.
- `frontend/index.html` and `frontend/style.css` define the interface.
- `frontend/app.js` connects the interface to the API.
- `frontend/network.js` draws the network graph.
- `frontend/charts.js` draws learning curves and the decision boundary.
