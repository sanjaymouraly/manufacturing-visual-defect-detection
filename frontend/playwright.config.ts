import { defineConfig, devices } from "@playwright/test";

/**
 * End-to-end tests run against a real backend serving the built SPA (default http://127.0.0.1:8000).
 *
 *   cd backend && python -m app.cli bootstrap-demo          # once: demo datasets + models
 *   DL_ADMIN_PASSWORD=... uvicorn app.main:app --port 8000  # serves frontend/dist too
 *   cd frontend && npm run build && npx playwright test
 *
 * Credentials come from E2E_ADMIN_USER / E2E_ADMIN_PASSWORD (default admin / DefectLens-e2e-2026). Set them to
 * the admin account of the server under test (DL_ADMIN_USERNAME / DL_ADMIN_PASSWORD on first start).
 */
export default defineConfig({
  testDir: "./e2e",
  timeout: 90_000,
  expect: { timeout: 15_000 },
  fullyParallel: false, // tests share one server/database; keep ordering deterministic
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: [["list"], ["html", { open: "never", outputFolder: "playwright-report" }]],
  globalSetup: "./e2e/global-setup.ts",
  use: {
    baseURL: process.env.E2E_BASE_URL ?? "http://127.0.0.1:8000",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    viewport: { width: 1440, height: 900 },
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } } }],
});
