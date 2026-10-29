import { expect, test } from "@playwright/test";

test("beginner mode explains a neuron and supports guided lessons", async ({ page }) => {
  await page.goto("/");
  await page.locator("#nav-guide").click();
  await expect(page.locator("#lesson-title")).toHaveText("Start with one point");
  await page.locator("#lesson-next").click();
  await expect(page.locator("#lesson-title")).toHaveText("Build the network");
  await page.locator("#lesson-next").click();
  await expect(page.locator("#neuron-sandbox")).toHaveAttribute("open", "");

  const initialOutput = await page.locator("#neuron-output").textContent();
  await page.locator("#neuron-w1").fill("1.50");
  await expect(page.locator("#neuron-output")).not.toHaveText(initialOutput);
  await expect(page.locator("#neuron-equation")).toContainText("1.50");
});

test("network graph can step through forward and backward propagation", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#network-view svg")).toBeVisible();
  await page.locator("#network-view [data-inspect-neuron]").first().click();
  await expect(page.locator("#network-inspector")).toBeVisible();
  await expect(page.locator("#network-inspector-value")).toContainText("Activation:");
  await page.locator("#network-inspector-close").click();
  await page.locator("#network-view [data-inspect-edge]").first().click();
  await expect(page.locator("#network-inspector-title")).toContainText("→");
  await expect(page.locator("#network-inspector-value")).toContainText("Weight:");
  await page.locator("#network-inspector-close").click();
  await page.locator("#show-backward-view").click();
  await expect(page.locator("#pass-indicator")).toContainText("BACKWARD");
  await expect(page.locator("#backward-step-count")).toHaveText("Step 1 / 3");
  await expect(page.locator("#network-view .delta-label").first()).toBeVisible();
  await expect(page.locator("#network-view .gradient-edge-label").first()).toBeVisible();
  await expect(page.locator("#network-view .bias-gradient-label").first()).toBeVisible();
  await expect(page.locator('#network-view .backward-active[marker-start="url(#backward-arrow)"]').first()).toBeVisible();
  const outputGradient = await page.locator("#backward-current-step").textContent();
  await page.locator("#backward-next").click();
  await expect(page.locator("#backward-step-count")).toHaveText("Step 2 / 3");
  await expect(page.locator("#backward-current-step")).not.toHaveText(outputGradient);
  await expect(page.locator("#backward-current-step")).toContainText("HIDDEN 2");
  await expect(page.locator("#backprop-mechanism .bp-neuron-row")).toHaveCount(4);
  await expect(page.locator(".gradient-step-current")).toHaveCount(1);
  await page.getByRole("button", { name: "Show backward step 3, hidden 1" }).click();
  await expect(page.locator("#backward-step-count")).toHaveText("Step 3 / 3");
  await expect(page.locator("#backward-current-step")).toContainText("HIDDEN 1");

  await page.locator("#show-forward-view").click();
  await expect(page.locator("#pass-indicator")).toContainText("FORWARD");
  await page.locator("#forward-next").click();
  await expect(page.locator("#forward-step-count")).toHaveText("Step 1 / 3");
  await expect(page.locator(".forward-neuron-card")).toHaveCount(5);
  await page.locator(".forward-neuron-card").nth(1).click();
  await expect(page.locator(".forward-neuron-card").nth(1)).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator('#network-view .inspect-selected[data-layer="1"][data-neuron="1"]')).toBeVisible();
  await page.locator("#forward-inspector .flow-step-button").last().click();
  await expect(page.locator("#forward-step-count")).toHaveText("Step 3 / 3");
});

test("backpropagation can apply one selected-example optimizer update", async ({ page }) => {
  await page.goto("/");
  await page.locator("#apply-backprop-update").click();
  await expect(page.locator("#apply-update-status")).toContainText("update applied");
  await expect(page.locator("#update-summary")).toContainText("Actual Adam update");
  await expect(page.locator(".update-stat")).toHaveCount(2);
});

test("input/output widths persist while choosing a dataset and apply to the model", async ({ page }) => {
  await page.goto("/");
  await page.locator("#input-size").fill("4");
  await page.locator("#output-size").fill("2");
  await page.locator("#dataset").selectOption("Moons");
  await expect(page.locator("#input-size")).toHaveValue("4");
  await expect(page.locator("#output-size")).toHaveValue("2");
  await expect(page.locator("#dataset-name")).toHaveText("Moons");
  await expect(page.locator("#apply-dimensions")).toBeEnabled();
  await page.locator("#apply-dimensions").click();
  await expect(page.locator("#architecture-label")).toHaveText("4 — 5 — 4 — 2");
  await expect(page.locator("#network-view .layer-title")).toHaveText([
    "INPUT · 4", "HIDDEN 1 · 5", "HIDDEN 2 · 4", "OUTPUT · 2",
  ]);
  await expect(page.locator("#network-view circle.neuron")).toHaveCount(15);
  await page.locator("#input-size").fill("1");
  await page.locator("#output-size").fill("1");
  await page.locator("#apply-dimensions").click();
  await expect(page.locator("#architecture-label")).toHaveText("1 — 5 — 4 — 1");
  await expect(page.locator("#network-view .layer-title")).toHaveText([
    "INPUT · 1", "HIDDEN 1 · 5", "HIDDEN 2 · 4", "OUTPUT · 1",
  ]);
  await expect(page.locator("#network-view circle.neuron")).toHaveCount(11);
});

test("forward and backward explanation panels can be minimized and expanded", async ({ page }) => {
  await page.goto("/");
  for (const panel of ["forward-content", "backward-content"]) {
    const content = page.locator(`#${panel}`);
    const button = page.locator(`[data-minimize="${panel}"]`);
    await button.click();
    await expect(content).toBeHidden();
    await expect(button).toHaveAttribute("aria-expanded", "false");
    await button.click();
    await expect(content).toBeVisible();
    await expect(button).toHaveAttribute("aria-expanded", "true");
  }
});

test("top navigation separates the visualizer and guide for keyboard users", async ({ page }) => {
  await page.goto("/");
  const guideTab = page.locator("#nav-guide");
  await guideTab.focus();
  await page.keyboard.press("Enter");
  await expect(page.locator("#guide-view")).toBeVisible();
  await expect(page.locator("#visualizer-view")).toBeHidden();
  await expect(guideTab).toHaveAttribute("aria-selected", "true");
  const visualTab = page.locator("#nav-visualize");
  await visualTab.focus();
  await page.keyboard.press("Enter");
  await expect(page.locator("#visualizer-view")).toBeVisible();
  await expect(page.locator("#guide-view")).toBeHidden();

  const backward = page.locator("#show-backward-view");
  await backward.focus();
  await page.keyboard.press("Space");
  await expect(backward).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator("#pass-indicator")).toContainText("BACKWARD");
});

test("training reports the real optimizer update", async ({ page }) => {
  await page.goto("/");
  page.once("dialog", dialog => dialog.accept("1"));
  await page.locator("#epoch-edit").click();
  await expect(page.locator("#epoch-total")).toHaveText("1");
  await page.locator("#train-button").click();
  await expect(page.locator("#model-status")).toHaveText("COMPLETE", { timeout: 20_000 });
  await expect(page.locator("#update-summary")).toContainText("Actual Adam update");
  await expect(page.locator(".update-stat")).toHaveCount(2);
});

test("saved experiments can be compared", async ({ page }) => {
  const makeSavedRun = (name, dataset, loss, accuracy) => ({
    name,
    experiment: {
      model: { architecture: [2, 4, 1], activation: "tanh", optimizer: "Adam", learning_rate: 0.04 },
      training: { dataset, epochs: 10, history: { loss: [loss], accuracy: [accuracy], validation_loss: [loss + 0.1], validation_accuracy: [accuracy - 0.05] } },
    },
  });
  await page.addInitScript(([a, b]) => localStorage.setItem("nnLabExperiments", JSON.stringify([a, b])), [
    makeSavedRun("Run A", "XOR", 0.4, 0.8),
    makeSavedRun("Run B", "Moons", 0.2, 0.9),
  ]);
  await page.goto("/");
  await page.locator("#compare-experiments").click();
  await expect(page.locator("#comparison-dialog")).toBeVisible();
  await expect(page.locator("#comparison-table")).toContainText("XOR");
  await expect(page.locator("#comparison-table")).toContainText("Moons");
});
