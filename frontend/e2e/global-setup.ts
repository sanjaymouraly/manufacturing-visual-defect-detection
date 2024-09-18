import { request } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
import { ADMIN, BASE, INSPECTOR, LOGOUT_USER, VIEWER, apiLogin, authFile } from "./helpers";

/** Ensure the server is up and the inspector/viewer test accounts exist with known passwords. */
export default async function globalSetup() {
  const ctx = await request.newContext({ baseURL: BASE, extraHTTPHeaders: { "X-Requested-With": "defectlens" } });
  const health = await ctx.get("/api/health/ready").catch(() => null);
  if (!health || !health.ok()) throw new Error(`Backend not ready at ${BASE} (GET /api/health/ready). Start it first, see playwright.config.ts.`);
  const r = await ctx.post("/api/auth/login", { data: ADMIN });
  if (!r.ok()) throw new Error(`Admin login failed (${r.status()}). Set E2E_ADMIN_USER / E2E_ADMIN_PASSWORD to the server's admin credentials.`);
  const users: { id: number; username: string }[] = await (await ctx.get("/api/users")).json();
  for (const [u, role] of [[INSPECTOR, "inspector"], [VIEWER, "viewer"], [LOGOUT_USER, "inspector"]] as const) {
    const existing = users.find((x) => x.username === u.username);
    const res = existing
      ? await ctx.patch(`/api/users/${existing.id}`, { data: { password: u.password, role, is_active: true } })
      : await ctx.post("/api/users", { data: { username: u.username, password: u.password, role, full_name: `E2E ${role}` } });
    if (!res.ok()) throw new Error(`Could not prepare ${u.username}: ${res.status()} ${await res.text()}`);
  }
  // one real login per role; tests reuse these cookies
  fs.mkdirSync(path.dirname(authFile(ADMIN)), { recursive: true });
  await ctx.storageState({ path: authFile(ADMIN) });
  await ctx.dispose();
  for (const u of [INSPECTOR, VIEWER]) {
    const c = await apiLogin(u);
    await c.storageState({ path: authFile(u) });
    await c.dispose();
  }
}
