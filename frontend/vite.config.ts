import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";

// The dev server proxies /api to the FastAPI backend so the browser only ever talks to one origin.
const backend = process.env.DL_BACKEND_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    allowedHosts: true,
    proxy: { "/api": { target: backend, changeOrigin: false } },
  },
  preview: { host: "0.0.0.0", port: 4173, allowedHosts: true, proxy: { "/api": backend } },
  build: { outDir: "dist", sourcemap: false, chunkSizeWarningLimit: 1200 },
});
