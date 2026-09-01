import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Every backend route the app calls is proxied here in dev, so the app code
// always uses relative paths ("/query", "/documents", ...) — identical to how
// nginx proxies them in the Docker build. No API base URL to configure.
const API_TARGET = "http://localhost:8000";
const PROXIED_PREFIXES = ["/documents", "/search", "/query", "/agent", "/conversations", "/health"];

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: Object.fromEntries(
      PROXIED_PREFIXES.map((prefix) => [prefix, { target: API_TARGET, changeOrigin: true }])
    ),
  },
});
