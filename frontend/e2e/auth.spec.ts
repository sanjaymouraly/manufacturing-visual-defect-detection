import { expect, test } from "@playwright/test";
import { ADMIN, BASE, INSPECTOR, LOGOUT_USER, VIEWER, apiAs, login, nav, uiLogin } from "./helpers";

test.describe("authentication & roles", () => {
  test("rejects a wrong password and shows an error", async ({ page }) => {
    await page.goto("/");
    await page.getByLabel("Username").fill(ADMIN.username);
    await page.getByLabel("Password").fill("definitely-wrong-password");
    await page.getByRole("button", { name: "Sign in" }).click();
    await expect(page.getByRole("alert")).toContainText(/invalid|incorrect/i);
    await expect(page.getByRole("navigation", { name: "Main" })).toHaveCount(0);
  });

  test("user signs in through the form, signs out, and the old session is revoked", async ({ page }) => {
    await uiLogin(page, LOGOUT_USER);
    const mainNav = page.getByRole("navigation", { name: "Main" });
    await expect(mainNav.getByRole("link", { name: "Inspect", exact: true })).toBeVisible();
    await expect(mainNav.getByRole("link", { name: "Settings" })).toHaveCount(0);
    await page.getByRole("button", { name: "Sign out" }).click();
    await expect(page.getByRole("button", { name: "Sign in" })).toBeVisible();
    // the old session cookie must no longer work
    const me = await page.request.get(`${BASE}/api/auth/me`);
    expect(me.status()).toBe(401);
  });

  test("a new account is prompted to replace its admin-set password and can change it in the UI", async ({ page }) => {
    const admin = await apiAs(ADMIN);
    const user = { username: `e2e-pw-${Date.now() % 100000}`, password: "Initial-e2e-2026" };
    expect((await admin.post("/api/users", { data: { ...user, role: "viewer" } })).status()).toBe(201);
    await uiLogin(page, user);
    const banner = page.getByRole("alert").filter({ hasText: /default or temporary password/ });
    await expect(banner).toBeVisible();
    await banner.getByRole("button", { name: "Change password" }).click();
    const dialog = page.getByRole("dialog", { name: "Change password" });
    await dialog.getByLabel("Current password").fill(user.password);
    await dialog.getByLabel("New password", { exact: true }).fill("Chosen-e2e-2026!");
    await dialog.getByLabel("Confirm new password").fill("Different-2026!");
    await dialog.getByRole("button", { name: "Change password" }).click();
    await expect(dialog.getByRole("alert")).toContainText(/do not match/);
    await dialog.getByLabel("Confirm new password").fill("Chosen-e2e-2026!");
    await dialog.getByRole("button", { name: "Change password" }).click();
    await expect(dialog.getByRole("status")).toContainText(/Password changed/);
    await dialog.getByRole("button", { name: "Done" }).click();
    await expect(banner).toHaveCount(0);
    // the old password no longer works, the new one does
    const fresh = await page.request.post(`${BASE}/api/auth/token`, { form: { username: user.username, password: user.password } });
    expect(fresh.status()).toBe(401);
    const ok = await page.request.post(`${BASE}/api/auth/token`, { form: { username: user.username, password: "Chosen-e2e-2026!" } });
    expect(ok.status()).toBe(200);
    await admin.dispose();
  });

  test("admin sees admin navigation", async ({ page }) => {
    await login(page, ADMIN);
    await expect(page.getByRole("navigation", { name: "Main" }).getByRole("link", { name: "Settings" })).toBeVisible();
  });

  test("viewer has read-only navigation and is blocked from operator pages", async ({ page }) => {
    await login(page, VIEWER);
    const navEl = page.getByRole("navigation", { name: "Main" });
    await expect(navEl.getByRole("link", { name: "History" })).toBeVisible();
    await expect(navEl.getByRole("link", { name: "Inspect", exact: true })).toHaveCount(0);
    await expect(navEl.getByRole("link", { name: "Settings" })).toHaveCount(0);
    await page.goto("/settings");
    await expect(page.getByText("Your role (viewer) does not have access to this page.")).toBeVisible();
  });

  test("server enforces roles and CSRF independently of the UI", async () => {
    const viewer = await apiAs(VIEWER);
    expect((await viewer.get("/api/users")).status()).toBe(403);
    expect((await viewer.put("/api/settings/general", { data: { site_name: "x" } })).status()).toBe(403);
    await viewer.dispose();
    const insp = await apiAs(INSPECTOR);
    expect((await insp.post("/api/models/train", { data: { dataset_id: 1, algorithm: "pixelstats" } })).status()).toBe(403);
    // unsafe request without the CSRF header is refused even with a valid cookie
    const r = await insp.post("/api/reviews/does-not-matter", { headers: { "X-Requested-With": "" }, data: { label: "normal" } });
    expect(r.status()).toBe(403);
    await insp.dispose();
  });

  test("theme toggle switches and persists", async ({ page }) => {
    await login(page, INSPECTOR);
    const html = page.locator("html");
    const wasDark = (await html.getAttribute("class"))?.includes("dark") ?? false;
    await page.getByRole("button", { name: /Switch to (light|dark) theme/ }).click();
    await expect(html).toHaveClass(wasDark ? /^(?!.*dark)/ : /dark/);
    await page.reload();
    await expect(html).toHaveClass(wasDark ? /^(?!.*dark)/ : /dark/);
    await page.getByRole("button", { name: /Switch to (light|dark) theme/ }).click(); // restore
    await nav(page, "Dashboard");
  });
});
