import { defineConfig, devices } from "@playwright/test";

const backendPort = process.env.CI ? "8000" : "8010";

export default defineConfig({
  testDir: "./tests/e2e",
  fullyParallel: false,
  retries: 0,
  reporter: "list",
  use: {
    baseURL: process.env.PLAYWRIGHT_BASE_URL ?? "http://localhost:3000",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: process.env.PLAYWRIGHT_EXTERNAL_SERVERS
    ? undefined
    : [
        {
          command: "python scripts/run_e2e_backend.py",
          cwd: "..",
          url: `http://127.0.0.1:${backendPort}/health`,
          timeout: 180_000,
          reuseExistingServer: false,
          env: {
            DATABASE_URL: "sqlite:///./.pytest_tmp/playwright.db",
            REQUIRE_API_KEY: "false",
            SIMULATION_AUTO_START: "true",
            SIMULATION_SLEEP_SECONDS: "0.5",
            GOOGLE_API_KEY: "test-key-not-real",
            E2E_BACKEND_PORT: backendPort,
          },
        },
        {
          command: "node node_modules/next/dist/bin/next dev -p 3000",
          cwd: ".",
          url: "http://localhost:3000/fleet",
          timeout: 120_000,
          reuseExistingServer: !process.env.CI,
        },
      ],
});
