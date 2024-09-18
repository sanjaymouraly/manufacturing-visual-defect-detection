import { expect, test, type Page } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { ADMIN, apiAs, login } from "./helpers";

/**
 * Captures documentation screenshots of the real running application into docs/images/.
 * Run with:  npm run screenshots   (SCREENSHOTS=1; skipped in the normal e2e run)
 * Optional:  SHOT_PRODUCT=visa-pcb1 to feature a real-data product (default: first VisA product, else demo-pcb).
 */
const here = path.dirname(fileURLToPath(import.meta.url));
const OUT = path.resolve(here, "../../docs/images");
test.skip(!process.env.SCREENSHOTS, "set SCREENSHOTS=1 to capture documentation screenshots");
test.describe.configure({ mode: "serial" });

async function shot(page: Page, name: string, opts: { fullPage?: boolean } = {}) {
  fs.mkdirSync(OUT, { recursive: true });
  await page.waitForLoadState("networkidle");
  await page.waitForTimeout(600); // let charts finish their entry animation
  if (opts.fullPage) {
    // Grow the viewport to the document height instead of Playwright's fullPage stitching, so the fixed
    // sidebar spans the whole image rather than floating in the middle of it.
    const vp = page.viewportSize()!;
    const h = await page.evaluate(() => document.documentElement.scrollHeight);
    await page.setViewportSize({ width: vp.width, height: Math.min(Math.max(h, vp.height), 4000) });
    await page.waitForTimeout(400);
    await page.screenshot({ path: path.join(OUT, `${name}.png`) });
    await page.setViewportSize(vp);
  } else {
    await page.screenshot({ path: path.join(OUT, `${name}.png`) });
  }
}

test("capture documentation screenshots", async ({ page }) => {
  test.setTimeout(600_000);
  const api = await apiAs(ADMIN);
  const products: { key: string; active_model_id: number | null }[] = await (await api.get("/api/products")).json();
  const product = process.env.SHOT_PRODUCT ?? products.find((p) => p.key.startsWith("visa-") && p.active_model_id)?.key ?? "demo-pcb";
  // pick a real anomalous inspection for the result shots (highest-scoring FAIL of the featured product,
  // so the localisation boxes are visible)
  const fails = await (await api.get(`/api/inspections?product_key=${product}&decision=FAIL&page_size=1&sort=-score`)).json();
  const review = await (await api.get("/api/reviews/queue?limit=1")).json();
  const models: { id: number; name: string }[] = await (await api.get(`/api/models?product_key=${product}`)).json();
  const evals: { id: number; kind: string; model?: string | null }[] = await (await api.get(`/api/evaluations?product_key=${product}`)).json();
  const batchJobs: { id: string; params: { product_key?: string }; result?: { PASS?: number; FAIL?: number; REVIEW?: number } }[] =
    await (await api.get("/api/jobs?type=batch_inspection&status=succeeded&limit=10")).json();
  const batch = batchJobs.find((j) => (j.result?.PASS ?? 0) + (j.result?.FAIL ?? 0) + (j.result?.REVIEW ?? 0) > 0) ?? batchJobs[0];
  await api.dispose();

  await login(page, ADMIN);
  await page.goto("/");
  await shot(page, "dashboard");

  if (fails.items?.length) {
    const id = fails.items[0].id;
    await page.goto(`/inspections/${id}`);
    await expect(page.locator(".card.border-l-4").first()).toBeVisible();
    await shot(page, "inspect-result", { fullPage: true });
    const layers = page.getByRole("radiogroup", { name: "Visualization layer" });
    await layers.getByRole("radio", { name: /heatmap/i }).click();
    await shot(page, "heatmap");
    await layers.getByRole("radio", { name: /overlay/i }).click();
    await page.getByLabel("Overlay opacity").fill("0.6");
    await shot(page, "overlay");
  }

  await page.goto("/reviews");
  await shot(page, "review-queue");
  if (review.items?.length) {
    await page.goto(`/reviews/${review.items[0].id}`);
    await expect(page.getByRole("heading", { name: "Review inspection" })).toBeVisible();
    await shot(page, "review-detail");
  }

  await page.goto(batch ? `/batch?job=${batch.id}` : "/batch");
  if (batch?.params.product_key) {
    await page.getByLabel(/product/i).first().selectOption(batch.params.product_key).catch(() => undefined);
  }
  await shot(page, "batch");

  await page.goto("/models");
  await page.getByLabel(/product/i).first().selectOption(product).catch(() => undefined);
  await shot(page, "model-lab", { fullPage: true });
  const bench = evals.find((e) => e.kind === "benchmark");
  if (bench) {
    await page.getByText(new RegExp(`#${bench.id}\\b`)).first().click().catch(() => undefined);
    await shot(page, "model-comparison");
    await page.keyboard.press("Escape");
  }
  const pc = models.find((m) => m.name.includes("patchcore"));
  if (pc) {
    await page.goto(`/models/${pc.id}`);
    await shot(page, "model-detail", { fullPage: true });
  }

  await page.goto("/robustness");
  await page.getByRole("combobox").first().selectOption(product).catch(() => undefined);
  const rob = evals.find((e) => e.kind === "robustness");
  // show the model the featured run was made with in the model selector, so the page is unambiguous
  const robModel = models.find((m) => m.name === rob?.model);
  if (robModel) await page.getByLabel("Model", { exact: true }).selectOption(String(robModel.id)).catch(() => undefined);
  if (rob) await page.getByRole("button", { name: new RegExp(`#${rob.id}\\b`) }).first().click().catch(() => undefined);
  await shot(page, "robustness", { fullPage: true });

  await page.goto("/history");
  await shot(page, "history");
  await page.goto("/analytics");
  await shot(page, "analytics", { fullPage: true });
  await page.goto("/datasets");
  await shot(page, "datasets");
  await page.goto("/settings?tab=providers");
  await shot(page, "settings-providers");
  await page.goto("/settings?tab=policy");
  await shot(page, "settings-policy", { fullPage: true });

  // dark theme variant of the dashboard
  await page.getByRole("button", { name: /Switch to (light|dark) theme/ }).click();
  await page.goto("/");
  await shot(page, "dashboard-dark");
  await page.getByRole("button", { name: /Switch to (light|dark) theme/ }).click();
});
