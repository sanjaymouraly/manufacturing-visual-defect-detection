import { expect, test } from "@playwright/test";
import { INSPECTOR, PCB_DEFECT, PCB_NORMAL, login, nav } from "./helpers";

test.describe("single-image inspection", () => {
  test("defective sample is not auto-passed and shows explanation layers", async ({ page }) => {
    await login(page, INSPECTOR);
    await nav(page, "Inspect");
    await expect(page.getByRole("heading", { name: "Inspect" })).toBeVisible();
    await page.getByLabel("Product").selectOption("demo-pcb");
    await page.getByTestId("file-input").setInputFiles(PCB_DEFECT);
    await page.getByLabel("Batch / lot").fill("E2E-LOT-1");
    await page.getByRole("button", { name: "Inspect", exact: true }).click();

    const header = page.locator(".card.border-l-4").first();
    await expect(header).toBeVisible({ timeout: 60_000 });
    const decision = (await header.locator("span.font-bold").first().textContent())?.trim();
    expect(["FAIL", "REVIEW"]).toContain(decision); // a defective part must never be auto-passed

    // visual explanation: switch layers and change overlay opacity
    const layers = page.getByRole("radiogroup", { name: "Visualization layer" });
    await layers.getByRole("radio", { name: /heatmap/i }).click();
    await expect(layers.getByRole("radio", { name: /heatmap/i })).toHaveAttribute("aria-checked", "true");
    await layers.getByRole("radio", { name: /overlay/i }).click();
    const slider = page.getByLabel("Overlay opacity");
    await slider.fill("0.3");
    await expect(slider).toHaveValue("0.3");
    await expect(page.getByText(/raw anomaly score/i).first()).toBeVisible();
  });

  test("re-uploading the same image returns the existing result unless forced", async ({ page }) => {
    await login(page, INSPECTOR);
    await nav(page, "Inspect");
    await page.getByLabel("Product").selectOption("demo-pcb");
    await page.getByTestId("file-input").setInputFiles(PCB_NORMAL);
    await page.getByRole("button", { name: "Inspect", exact: true }).click();
    await expect(page.locator(".card.border-l-4").first()).toBeVisible({ timeout: 60_000 });
    await page.getByTestId("file-input").setInputFiles(PCB_NORMAL);
    await page.getByRole("button", { name: "Inspect", exact: true }).click();
    await expect(page.getByText(/duplicate image/i).first()).toBeVisible();
    await page.getByRole("button", { name: "Inspect again anyway" }).click();
    await expect(page.getByText(/duplicate image — existing result returned/i)).toHaveCount(0, { timeout: 60_000 });
  });

  test("history lists the inspection and the detail page shows the audit trail", async ({ page }) => {
    await login(page, INSPECTOR);
    await nav(page, "History");
    await page.getByPlaceholder(/search/i).fill("E2E-LOT-1");
    await page.getByPlaceholder(/search/i).press("Enter"); // search applies on submit
    const row = page.locator("tbody tr").filter({ hasText: "A_0162" }).first();
    await expect(row).toBeVisible();
    await row.click();
    await expect(page.getByText(/inspection\.created|Audit trail/i).first()).toBeVisible();
  });
});
