const PlaygroundCharts = (() => {
  const palette = { grid: "#30323e", muted: "#858795", lime: "#c7f36b", violet: "#b1a1ff", coral: "#f27f81", mint: "#59d5a0" };
  function sizedContext(canvas) {
    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(rect.width * ratio));
    canvas.height = Math.max(1, Math.round(rect.height * ratio));
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { ctx, width: rect.width, height: rect.height };
  }

  function history(canvas, values) {
    const { ctx, width, height } = sizedContext(canvas);
    ctx.clearRect(0, 0, width, height);
    const left = 37, right = 8, top = 6, bottom = 18;
    const plotW = Math.max(1, width - left - right), panelH = Math.max(1, (height - top - bottom - 18) / 2);
    const count = values.loss.length;
    ctx.font = "9px 'DM Mono', monospace";
    ctx.fillStyle = palette.muted;
    const plotSeries = [
      { name: "LOSS", top, series: values.loss, validation: values.validation_loss || [], color: palette.lime, max: Math.max(1, ...values.loss, ...(values.validation_loss || [])) },
      { name: "ACCURACY", top: top + panelH + 18, series: values.accuracy, validation: values.validation_accuracy || [], color: palette.violet, max: 1 },
    ];
    plotSeries.forEach(panel => {
      ctx.fillStyle = palette.muted; ctx.fillText(panel.name, 3, panel.top + 9);
      [0, .5, 1].forEach(tick => {
        const y = panel.top + 15 + panelH * tick;
        ctx.strokeStyle = palette.grid; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(width - right, y); ctx.stroke();
        ctx.fillText((panel.max * (1 - tick)).toFixed(panel.max <= 1 ? 1 : 2), 2, y + 3);
      });
      const drawLine = (series, color, dashed) => {
        if (!series.length) return;
        ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.setLineDash(dashed ? [4, 3] : []); ctx.beginPath();
        series.forEach((value, index) => {
          const x = left + (count <= 1 ? plotW : (index / (count - 1)) * plotW);
          const y = panel.top + 15 + panelH * (1 - Math.max(0, Math.min(1, value / panel.max)));
          index ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
        });
        ctx.stroke(); ctx.setLineDash([]);
      };
      drawLine(panel.series, panel.color, false);
      drawLine(panel.validation, panel.name === "LOSS" ? palette.mint : palette.coral, true);
    });
    ctx.fillStyle = palette.muted; ctx.fillText("1", left, height - 4); ctx.textAlign = "right";
    ctx.fillText(String(Math.max(1, count)), width - right, height - 4); ctx.textAlign = "left";
  }

  function boundary(canvas, state) {
    const { ctx, width, height } = sizedContext(canvas);
    ctx.clearRect(0, 0, width, height);
    const plot = { x: 28, y: 7, w: Math.max(1, width - 37), h: Math.max(1, height - 17) };
    const grid = state.decision_boundary;
    const rows = grid.z.length, cols = grid.z[0].length;
    const cellW = plot.w / cols, cellH = plot.h / rows;
    const colors = [[242, 127, 129], [177, 161, 255], [89, 213, 160]];
    for (let row = 0; row < rows; row++) for (let col = 0; col < cols; col++) {
      const probability = grid.z[row][col];
      const t = Math.max(0, Math.min(1, probability));
      const cls = grid.classes?.[row]?.[col] || 0;
      const rgb = grid.classes ? colors[cls % colors.length] : colors[0].map((v, i) => Math.round(v + (colors[1][i] - v) * t));
      ctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${0.15 + 0.18 * Math.abs(t - 0.5) * 2})`;
      ctx.fillRect(plot.x + col * cellW, plot.y + (rows - 1 - row) * cellH, cellW + 0.5, cellH + 0.5);
    }
    ctx.strokeStyle = "#707282"; ctx.lineWidth = 1; ctx.setLineDash([3, 4]);
    const mid = plot.x + plot.w / 2;
    ctx.beginPath(); ctx.moveTo(mid, plot.y); ctx.lineTo(mid, plot.y + plot.h); ctx.stroke();
    ctx.setLineDash([]);
    const validation = new Set(state.validation_indices || []);
    state.points.forEach((point, i) => {
      const px = plot.x + ((point[0] + 1.2) / 2.4) * plot.w;
      const py = plot.y + (1 - (point[1] + 1.2) / 2.4) * plot.h;
      const label = state.labels[i];
      const predictedClass = state.predicted_classes?.[i] ?? (Array.isArray(state.predictions[i]) ? state.predictions[i].indexOf(Math.max(...state.predictions[i])) : (state.predictions[i] >= .5 ? 1 : 0));
      const color = colors[label % colors.length];
      ctx.beginPath(); ctx.arc(px, py, 2.4, 0, Math.PI * 2);
      ctx.fillStyle = `rgb(${color.join(",")})`;
      ctx.fill();
      ctx.strokeStyle = predictedClass === label ? (validation.has(i) ? "#f1f1f5" : "#171923") : "#fff";
      ctx.lineWidth = predictedClass === label ? (validation.has(i) ? 1.1 : .7) : 1.1;
      ctx.stroke();
      if (i === state.selectedSample) { ctx.beginPath(); ctx.arc(px, py, 5, 0, Math.PI * 2); ctx.strokeStyle = palette.lime; ctx.lineWidth = 1.5; ctx.stroke(); }
    });
    ctx.strokeStyle = palette.grid; ctx.lineWidth = 1; ctx.strokeRect(plot.x, plot.y, plot.w, plot.h);
  }
  return { history, boundary };
})();

export const drawHistory = PlaygroundCharts.history;
export const drawBoundary = PlaygroundCharts.boundary;
