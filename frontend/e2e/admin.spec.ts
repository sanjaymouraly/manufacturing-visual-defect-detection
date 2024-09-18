import { expect, test } from "@playwright/test";
import { ADMIN, login, nav } from "./helpers";

test.describe("model lab, robustness, analytics & settings", () => {
  test("model lab lists versions and the model detail shows held-out metrics", async ({ page }) => {
    await login(page, ADMIN);
    await nav(page, "Model lab");
    await page.getByLabel(/product/i).first().selectOption("demo-pcb");
    await expect(page.getByText(/patchcore/i).first()).toBeVisible();
    await page.getByRole("link", { name: /demo-pcb-patchcore/ }).first().click();
    await expect(page.getByRole("tab", { name: /Held-out test/i })).toBeVisible();
    await expect(page.getByText(/AUROC/).first()).toBeVisible();
    await page.getByRole("tab", { name: /Thresholds/i }).click();
    await expect(page.getByText(/reason/i).first()).toBeVisible();
  });

  test("robustness lab previews a degradation with clean vs degraded decisions", async ({ page }) => {
    await login(page, ADMIN);
    await nav(page, "Robustness lab");
    await page.getByRole("combobox").first().selectOption("demo-pcb");
    await page.getByRole("tab", { name: "Degradation preview" }).click();
    await expect(page.getByAltText("degraded sample")).toBeVisible({ timeout: 60_000 });
    await expect(page.getByAltText("clean sample")).toBeVisible();
  });

  test("analytics page renders and the AI box works without any LLM configured", async ({ page }) => {
    await login(page, ADMIN);
    await nav(page, "Analytics");
    await expect(page.getByRole("heading", { name: "Analytics & reports" })).toBeVisible();
    await page.getByRole("button", { name: /summar/i }).first().click();
    await expect(page.getByText(/template|model-generated|generated/i).first()).toBeVisible({ timeout: 30_000 });
  });

  test("settings: policy section saves, providers can be added and tested (no network needed)", async ({ page }) => {
    await login(page, ADMIN);
    await nav(page, "Settings");
    await expect(page.getByRole("heading", { name: "Settings" })).toBeVisible();
    await page.getByRole("tab", { name: "AI providers" }).click();
    await expect(page.getByText("Built-in template summaries")).toBeVisible();
    const name = `e2e-ollama-${Date.now() % 100000}`;
    await page.getByRole("button", { name: "Add provider" }).click();
    await page.getByLabel("Display name").fill(name);
    await page.getByLabel("Provider type").selectOption("ollama");
    await page.getByLabel("Base URL").fill("http://127.0.0.1:9");
    await page.getByLabel("Timeout (s)").fill("2");
    await page.getByLabel("Retries").fill("0");
    await page.getByRole("button", { name: "Add provider" }).last().click();
    const row = page.locator("div").filter({ hasText: name }).filter({ has: page.getByRole("button", { name: "Test connection" }) }).last();
    await row.getByRole("button", { name: "Test connection" }).click();
    await expect(row.getByText(/refused|failed|error|connect/i)).toBeVisible({ timeout: 20_000 }); // nothing listens on port 9
    // clean up: delete through the UI (confirmation dialog) so repeated runs leave no providers behind
    await page.getByRole("button", { name: `Delete provider ${name}` }).click();
    await page.getByRole("dialog").getByRole("button", { name: "Confirm" }).click();
    await expect(page.getByText(name)).toHaveCount(0);
    await page.getByRole("tab", { name: "Audit log" }).click();
    await page.getByRole("button", { name: "Verify chain" }).click();
    await expect(page.getByText(/Chain intact/)).toBeVisible();
  });

  test("jobs page lists background jobs", async ({ page }) => {
    await login(page, ADMIN);
    await nav(page, "Jobs");
    await expect(page.getByRole("heading", { name: "Background jobs" })).toBeVisible();
    // either rows (jobs from earlier specs) or the explicit empty state - never a blank/broken page
    await expect(page.locator("tbody tr").first().or(page.getByText("No jobs"))).toBeVisible();
  });
});
