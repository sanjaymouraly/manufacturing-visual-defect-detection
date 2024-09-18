import { expect, request, type APIRequestContext, type Page } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const BASE = process.env.E2E_BASE_URL ?? "http://127.0.0.1:8000";
export const ADMIN = { username: process.env.E2E_ADMIN_USER ?? "admin", password: process.env.E2E_ADMIN_PASSWORD ?? "DefectLens-e2e-2026" };
export const INSPECTOR = { username: "e2e-inspector", password: "Inspector-e2e-2026" };
export const VIEWER = { username: "e2e-viewer", password: "Viewer-e2e-2026" };
/** Used only by the logout test: logging out revokes all of a user's sessions (token_version). */
export const LOGOUT_USER = { username: "e2e-logout", password: "Logout-e2e-2026" };

const here = path.dirname(fileURLToPath(import.meta.url));
export const SAMPLES = path.resolve(here, "../../samples");
export const sample = (rel: string) => path.join(SAMPLES, rel);
export const PCB_DEFECT = sample("demo-pcb/A_0162_defect-contamination.png");
export const PCB_NORMAL = sample("demo-pcb/N_0000_normal.png");

type Cred = { username: string; password: string };
/** Session cookies saved by global-setup (one real login per role keeps us under the login rate limit). */
export const authFile = (user: Cred) => path.join(here, ".auth", `${user.username}.json`);

/** API context reusing the saved session cookie (CSRF header included for unsafe methods). */
export async function apiAs(user: Cred): Promise<APIRequestContext> {
  return request.newContext({ baseURL: BASE, storageState: authFile(user), extraHTTPHeaders: { "X-Requested-With": "defectlens" } });
}

/** Fresh API login (counts against the login rate limit; used by global-setup only). */
export async function apiLogin(user: Cred): Promise<APIRequestContext> {
  const ctx = await request.newContext({ baseURL: BASE, extraHTTPHeaders: { "X-Requested-With": "defectlens" } });
  const r = await ctx.post("/api/auth/login", { data: user });
  expect(r.ok(), `login ${user.username}: ${r.status()} ${await r.text()}`).toBeTruthy();
  return ctx;
}

/** Open the app as `user` using the saved session. */
export async function login(page: Page, user: Cred) {
  const state = JSON.parse(fs.readFileSync(authFile(user), "utf-8"));
  await page.context().addCookies(state.cookies);
  await page.goto("/");
  await expect(page.getByRole("navigation", { name: "Main" })).toBeVisible();
}

/** Sign in through the login form (for tests of the login UI itself). */
export async function uiLogin(page: Page, user: Cred) {
  await page.goto("/");
  await page.getByLabel("Username").fill(user.username);
  await page.getByLabel("Password").fill(user.password);
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByRole("navigation", { name: "Main" })).toBeVisible();
}

export async function nav(page: Page, label: string) {
  await page.getByRole("navigation", { name: "Main" }).getByRole("link", { name: label, exact: false }).first().click();
}
