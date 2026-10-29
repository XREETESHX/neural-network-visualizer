import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./frontend/e2e",
  fullyParallel: true,
  workers: 1,
  reporter: "list",
  use: {
    baseURL: "http://127.0.0.1:8137",
    browserName: "chromium",
    headless: true,
  },
  webServer: {
    command: "python -m uvicorn backend.main:app --host 127.0.0.1 --port 8137",
    url: "http://127.0.0.1:8137/api/health",
    env: { CORS_ORIGINS: "http://127.0.0.1:8137" },
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
  },
});
