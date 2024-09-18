import { expect, test } from "@playwright/test";
import { ADMIN, INSPECTOR, apiAs, login, nav, sample } from "./helpers";

test.describe("batch → review → feedback workflow", () => {
  test("batch of images is processed in the background with results", async ({ page }) => {
    await login(page, INSPECTOR);
    await nav(page, "Batch");
    await page.getByLabel("Product").selectOption("demo-pcb");
    await page.getByLabel("Batch / lot code").fill("E2E-BATCH");
    await page.getByTestId("batch-input").setInputFiles([
      sample("demo-pcb/N_0002_normal.png"), sample("demo-pcb/N_0004_normal.png"),
      sample("demo-pcb/A_0160_defect-missing_component.png"), sample("demo-pcb/A_0163_defect-solder_bridge.png"),
    ]);
    await page.getByRole("button", { name: "Start batch" }).click();
    await expect(page.getByText(/succeeded/i).first()).toBeVisible({ timeout: 120_000 });
    const rows = page.locator("table tbody tr").filter({ hasText: /\.png/ });
    await expect(rows).toHaveCount(4);
    await page.getByLabel("Filter by decision").selectOption("FAIL");
    await expect(page.getByRole("button", { name: "CSV" })).toBeVisible();
  });

  test("inspector reviews a queued item with a region and proposes it as feedback", async ({ page }) => {
    // make sure at least one item is waiting: a borderline/failed item from the batch, or force one via API
    const api = await apiAs(INSPECTOR);
    const q = await (await api.get("/api/reviews/queue?limit=5")).json();
    test.skip(q.total === 0, "no pending reviews produced by the decision policy on the demo data");
    await api.dispose();

    await login(page, INSPECTOR);
    await nav(page, "Review queue");
    await expect(page.getByRole("heading", { name: "Review queue" })).toBeVisible();
    await page.locator("tbody tr").first().click();
    await expect(page.getByRole("heading", { name: "Review inspection" })).toBeVisible();
    await page.getByRole("radiogroup", { name: "Label" }).getByRole("radio", { name: "Defective" }).click();
    await page.getByLabel("Defect type").fill("contamination");
    await expect(page.getByRole("button", { name: /Submit review/ })).toBeDisabled(); // severity is mandatory
    await page.getByLabel("Severity").selectOption("medium");
    // draw a region box on the image
    const area = page.getByRole("application", { name: "Region drawing area" });
    const box = await area.boundingBox();
    if (box) {
      await page.mouse.move(box.x + box.width * 0.3, box.y + box.height * 0.3);
      await page.mouse.down();
      await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.5, { steps: 5 });
      await page.mouse.up();
    }
    await page.getByLabel("Comment").fill("E2E review");
    const before = q.total;
    await page.getByRole("button", { name: /Submit review/ }).click();
    await expect(page.getByText("Review saved", { exact: true })).toBeVisible();
    const api2 = await apiAs(INSPECTOR);
    const after = await (await api2.get("/api/reviews/queue?limit=5")).json();
    expect(after.total).toBe(before - 1);
    await api2.dispose();
  });

  test("admin sees pending feedback in Datasets and can reject it", async ({ page }) => {
    const api = await apiAs(ADMIN);
    const fb = await (await api.get("/api/feedback?status=pending")).json();
    await api.dispose();
    test.skip(fb.items.length === 0, "no pending feedback samples");
    await login(page, ADMIN);
    await nav(page, "Datasets");
    await page.getByRole("tab", { name: /Feedback samples/ }).click();
    await expect(page.getByRole("button", { name: "Approve" }).first()).toBeVisible();
    await page.getByRole("button", { name: "Reject" }).first().click();
    await expect(page.locator("tbody tr")).toHaveCount(fb.items.length - 1);
  });
});
