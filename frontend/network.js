export const NetworkDiagram = (() => {
  const NS = "http://www.w3.org/2000/svg";
  const svgEl = (tag, attrs = {}) => {
    const element = document.createElementNS(NS, tag);
    Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
    return element;
  };

  function render(container, state, animate = true, pass = { mode: "idle" }) {
    const { model } = state;
    const layers = model.architecture;
    if (!Array.isArray(model.weights) || !Array.isArray(model.biases) || model.weights.length !== layers.length - 1 || model.biases.length !== layers.length - 1) {
      throw new Error("Model parameter count does not match the selected architecture. Reapply the configuration or reset the session.");
    }
    model.weights.forEach((matrix, layer) => {
      if (!Array.isArray(matrix) || matrix.length !== layers[layer] || matrix.some(row => !Array.isArray(row) || row.length !== layers[layer + 1] || row.some(value => !Number.isFinite(Number(value))))) {
        throw new Error(`Weights for layer ${layer + 1} do not match the architecture or contain invalid values. Reapply the configuration or reset the session.`);
      }
      const bias = model.biases[layer];
      if (!Array.isArray(bias) || bias.length !== layers[layer + 1] || bias.some(value => !Number.isFinite(Number(value)))) {
        throw new Error(`Biases for layer ${layer + 1} do not match the architecture or contain invalid values. Reapply the configuration or reset the session.`);
      }
    });
    if (!Array.isArray(state.sample_activations) || state.sample_activations.length !== layers.length ||
        state.sample_activations.some((values, layer) => !Array.isArray(values) || values.length !== layers[layer] || values.some(value => !Number.isFinite(Number(value))))) {
      throw new Error("Sample activations do not match the selected architecture. Refresh the sample or reapply the configuration.");
    }
    const svg = svgEl("svg", { viewBox: "0 0 1000 320", preserveAspectRatio: "xMidYMid meet", class: "network-svg", style: `--training-update-duration:${Math.max(250, Number(pass.updateDurationMs) || 1000)}ms`, "data-pass-mode": pass.mode || "idle", role: "img", "aria-label": "Live neural network showing the selected forward or backward pass, weights, and neuron biases" });
    const title = svgEl("title");
    title.textContent = "Neural network: weights are printed beside connections; biases are shown below neurons.";
    svg.append(title);
    if (pass.mode === "backward") {
      const defs = svgEl("defs");
      const marker = svgEl("marker", { id: "backward-arrow", viewBox: "0 0 10 10", refX: "5", refY: "5", markerWidth: "5", markerHeight: "5", orient: "auto-start-reverse" });
      marker.append(svgEl("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: "#c4a7ff" }));
      defs.append(marker);
      svg.append(defs);
    }
    const visibleIndices = layers.map(count => count <= 14
      ? Array.from({ length: count }, (_, index) => index)
      : Array.from({ length: 14 }, (_, index) => Math.round(index * (count - 1) / 13)));
    const positions = layers.map((count, layerIndex) => {
      const x = 90 + layerIndex * (820 / Math.max(1, layers.length - 1));
      const shown = visibleIndices[layerIndex].length;
      const gap = Math.min(42, 250 / Math.max(1, shown - 1));
      const center = 155;
      return visibleIndices[layerIndex].map((node, visibleIndex) => ({
        x, y: center + (visibleIndex - (shown - 1) / 2) * gap, index: node,
      }));
    });
    const positionByIndex = positions.map(nodes => new Map(nodes.map(point => [point.index, point])));
    const activations = state.sample_activations || [];
    const activeEdgeLayer = pass.mode === "forward" ? pass.step - 1 : pass.mode === "backward" ? pass.layer : -1;
    const activeNodeLayer = pass.mode === "forward" ? pass.step : pass.mode === "backward" ? pass.layer + 1 : -1;
    const updateLayerIndex = Number(pass.updateLayer);
    const formatChange = value => `${value >= 0 ? "+" : ""}${Number(value).toFixed(4)}`;
    const selectedUpdate = pass.update?.layers?.[updateLayerIndex];
    const topWeightUpdates = new Set();
    if (selectedUpdate) {
      const ranked = [];
      selectedUpdate.weights_before.forEach((row, source) => row.forEach((before, target) => {
        const after = Number(selectedUpdate.weights_after[source][target]);
        ranked.push({ key: `${source}-${target}`, magnitude: Math.abs(after - Number(before)) });
      }));
      ranked.sort((a, b) => b.magnitude - a.magnitude).slice(0, 3).forEach(item => topWeightUpdates.add(item.key));
    }

    layers.forEach((count, layer) => {
      const name = layer === 0 ? "INPUT" : layer === layers.length - 1 ? "OUTPUT" : `HIDDEN ${layer}`;
      const caption = svgEl("text", { x: positions[layer][0].x, y: 24, class: `layer-title${layer === activeNodeLayer ? " layer-current" : ""}` });
      caption.textContent = `${name} · ${count}${count > 14 ? ` · ${positions[layer].length} shown` : ""}`;
      svg.append(caption);
    });

    const rankedConnections = [];
    model.weights.forEach((matrix, layer) => matrix.forEach((row, source) => row.forEach((weight, target) => {
      rankedConnections.push({ key: `${layer}-${source}-${target}`, magnitude: Math.abs(weight) });
    })));
    rankedConnections.sort((a, b) => b.magnitude - a.magnitude);
    const animatedConnections = new Set(rankedConnections.slice(0, 80).map(edge => edge.key));
    const visibleGradientKeys = new Set();
    if (pass.mode === "backward" && Array.isArray(pass.weightGradients)) {
      const rankedGradients = [];
      pass.weightGradients.forEach((row, source) => row.forEach((gradient, target) => {
        if (positionByIndex[activeEdgeLayer]?.has(source) && positionByIndex[activeEdgeLayer + 1]?.has(target)) {
          rankedGradients.push({ key: `${source}-${target}`, magnitude: Math.abs(Number(gradient)) });
        }
      }));
      rankedGradients.sort((a, b) => b.magnitude - a.magnitude);
      rankedGradients.slice(0, 8).forEach(item => visibleGradientKeys.add(item.key));
    }

    model.weights.forEach((matrix, layer) => {
      matrix.forEach((row, source) => row.forEach((weight, target) => {
        const from = positionByIndex[layer].get(source);
        const to = positionByIndex[layer + 1].get(target);
        if (!from || !to) return;
        const path = `M ${from.x + 17} ${from.y} L ${to.x - 17} ${to.y}`;
        const color = weight >= 0 ? "#8ea7ff" : "#f27f81";
        const active = layer === activeEdgeLayer;
        const selectedConnection = pass.mode === "forward" && (
          (pass.selectedLayer > 0 && layer === pass.selectedLayer - 1 && target === pass.selectedNeuron) ||
          (pass.selectedLayer === 0 && layer === 0 && source === pass.selectedNeuron)
        );
        const update = pass.update?.layers?.[layer];
        const showUpdate = layer === updateLayerIndex && update;
        const weightBefore = showUpdate ? Number(update.weights_before[source]?.[target]) : weight;
        const weightAfter = showUpdate ? Number(update.weights_after[source]?.[target]) : weight;
        const weightChange = weightAfter - weightBefore;
        const changed = showUpdate && Math.abs(weightChange) > 1e-10;
        const updateClass = changed ? ` weight-updated ${weightChange > 0 ? "weight-increased" : "weight-decreased"}` : "";
        const edgeClass = `net-edge${active ? ` ${pass.mode}-active` : pass.mode !== "idle" ? " pass-dimmed" : ""}${selectedConnection ? " connection-selected" : ""}${updateClass}`;
        const line = svgEl("path", { d: path, class: edgeClass, stroke: selectedConnection ? "#dfff91" : active ? (pass.mode === "backward" ? "#c4a7ff" : "#c7f36b") : color, "stroke-width": selectedConnection ? "3.2" : active ? "2.8" : (0.65 + Math.min(1.8, Math.abs(weight) * 0.6)).toFixed(2), "data-inspect-edge": "true", "data-layer": layer, "data-source": source, "data-target": target, tabindex: "0", role: "button", "aria-label": `Inspect weight from neuron ${source + 1} in layer ${layer + 1} to neuron ${target + 1} in layer ${layer + 2}`, ...(active && pass.mode === "backward" ? { "marker-start": "url(#backward-arrow)" } : {}) });
        const label = svgEl("title");
        const gradient = showUpdate ? Number(update.weight_gradients[source]?.[target]) : 0;
        label.textContent = changed
          ? `Weight W${layer + 1}[${source}, ${target}]: ${weightBefore.toFixed(6)} → ${weightAfter.toFixed(6)}; change ${formatChange(weightChange)}; gradient ${gradient.toFixed(6)}`
          : `Weight W${layer + 1}[${source}, ${target}] = ${Number(weight).toFixed(5)}`;
        line.append(label);
        svg.append(line);
        const labelText = svgEl("text", { x: (from.x + to.x) / 2, y: (from.y + to.y) / 2 - 3, class: `edge-label${active ? " edge-active-label" : pass.mode !== "idle" ? " edge-dimmed-label" : ""}`, "text-anchor": "middle" });
        labelText.textContent = Number(weight).toFixed(2);
        labelText.setAttribute("pointer-events", "none");
        svg.append(labelText);
        if (active && pass.mode === "backward" && visibleGradientKeys.has(`${source}-${target}`)) {
          const gradientLabel = svgEl("text", { x: (from.x + to.x) / 2, y: (from.y + to.y) / 2 + 10, class: "gradient-edge-label", "text-anchor": "middle" });
          gradientLabel.textContent = `∂W ${Number(pass.weightGradients[source][target]).toFixed(3)}`;
          gradientLabel.setAttribute("pointer-events", "none");
          svg.append(gradientLabel);
        }
        if (changed && topWeightUpdates.has(`${source}-${target}`)) {
          const deltaLabel = svgEl("text", { x: (from.x + to.x) / 2, y: (from.y + to.y) / 2 + 9, class: `edge-change-label ${weightChange > 0 ? "weight-increased" : "weight-decreased"}`, "text-anchor": "middle" });
          deltaLabel.textContent = `Δ${formatChange(weightChange)}`;
          deltaLabel.setAttribute("pointer-events", "none");
          svg.append(deltaLabel);
        }
        if (animate && (active || (pass.mode === "idle" && animatedConnections.has(`${layer}-${source}-${target}`)))) {
          const motionPath = pass.mode === "backward" && active
            ? `M ${to.x - 17} ${to.y} L ${from.x + 17} ${from.y}`
            : path;
          const motionId = `edge-${pass.mode}-${layer}-${source}-${target}`;
          const motion = svgEl("path", { id: motionId, d: motionPath, fill: "none", stroke: "none" });
          svg.append(motion);
          const dot = svgEl("circle", { r: active ? "4" : "2.5", class: `signal-dot${pass.mode === "backward" && active ? " reverse-signal" : ""}` });
          const animation = svgEl("animateMotion", { dur: active ? `${Math.max(250, Number(pass.durationMs) || 1200)}ms` : `${1.5 + ((source + target) % 3) * 0.4}s`, repeatCount: active ? "1" : "indefinite", fill: active ? "freeze" : "remove", begin: active ? "0s" : `${((source * 3 + target) % 10) * 0.16}s`, rotate: "auto" });
          const mpath = svgEl("mpath", { href: `#${motionId}` });
          animation.append(mpath);
          dot.append(animation);
          svg.append(dot);
        }
      }));
    });

    positions.forEach((nodes, layer) => nodes.forEach(point => {
      const index = point.index;
      const bias = layer === 0 ? 0 : model.biases[layer - 1][index];
      const activation = activations[layer]?.[index];
      const biasUpdate = layer > 0 && layer - 1 === updateLayerIndex ? pass.update?.layers?.[layer - 1] : null;
      const biasBefore = biasUpdate ? Number(biasUpdate.biases_before[index]) : bias;
      const biasAfter = biasUpdate ? Number(biasUpdate.biases_after[index]) : bias;
      const biasChange = biasAfter - biasBefore;
      const biasChanged = Boolean(biasUpdate && Math.abs(biasChange) > 1e-10);
      const cls = layer === 0 ? "neuron input" : layer === layers.length - 1 ? "neuron output" : "neuron";
      const denseLayer = layers[layer] > 8;
      const currentLayer = layer === activeNodeLayer;
      const selected = pass.mode === "forward" && pass.selectedLayer === layer && pass.selectedNeuron === index;
      const node = svgEl("circle", { cx: point.x, cy: point.y, r: denseLayer ? 10 : 15, class: `${cls}${activation > 0.7 ? " active" : ""}${currentLayer ? ` ${pass.mode}-focus` : pass.mode !== "idle" ? " pass-dimmed" : ""}${selected ? " inspect-selected" : ""}`, "data-inspect-neuron": "true", "data-layer": layer, "data-neuron": index, tabindex: "0", role: "button", "aria-label": `Inspect neuron ${index + 1} in ${layer === 0 ? "input" : layer === layers.length - 1 ? "output" : `hidden layer ${layer}`}` });
      const nodeTitle = svgEl("title");
      nodeTitle.textContent = `Neuron ${layer === 0 ? "input" : layer === layers.length - 1 ? "output" : `hidden layer ${layer}`} ${index + 1}; bias = ${Number(bias).toFixed(5)}${biasChanged ? `; latest bias update ${biasBefore.toFixed(6)} → ${biasAfter.toFixed(6)}; change ${formatChange(biasChange)}; gradient ${Number(biasUpdate.bias_gradients[index]).toFixed(6)}` : ""}; activation = ${activation === undefined ? "—" : Number(activation).toFixed(5)}`;
      node.append(nodeTitle);
      svg.append(node);
      const value = svgEl("text", { x: point.x, y: point.y + 3, class: "neuron-label" });
      value.textContent = activation === undefined ? `x${index + 1}` : Number(activation).toFixed(2);
      value.setAttribute("pointer-events", "none");
      svg.append(value);
      const biasText = svgEl("text", { x: point.x, y: point.y + (denseLayer ? 18 : 27), class: "neuron-bias" });
      biasText.textContent = biasChanged ? `b:${Number(bias).toFixed(2)} Δ${formatChange(biasChange)}` : `b:${Number(bias).toFixed(2)}`;
      if (biasChanged) biasText.classList.add("bias-updated", biasChange > 0 ? "weight-increased" : "weight-decreased");
      biasText.setAttribute("pointer-events", "none");
      svg.append(biasText);
      if (pass.mode === "backward" && currentLayer && Array.isArray(pass.deltas) && pass.deltas[index] !== undefined) {
        const deltaText = svgEl("text", { x: point.x + 23, y: point.y + 3, class: "delta-label" });
        deltaText.textContent = `δ ${Number(pass.deltas[index]).toFixed(2)}`;
        deltaText.setAttribute("pointer-events", "none");
        svg.append(deltaText);
      }
      if (pass.mode === "backward" && currentLayer && Array.isArray(pass.biasGradients) && pass.biasGradients[index] !== undefined) {
        const biasGradient = svgEl("text", { x: point.x + 23, y: point.y + 15, class: "bias-gradient-label" });
        biasGradient.textContent = `∂b ${Number(pass.biasGradients[index]).toFixed(3)}`;
        biasGradient.setAttribute("pointer-events", "none");
        svg.append(biasGradient);
      }
    }));
    container.replaceChildren(svg);
  }

  return { render };
})();

export const renderNetwork = NetworkDiagram.render;
if (typeof window !== "undefined") window.NetworkDiagram = NetworkDiagram;
