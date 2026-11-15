import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // The simulation API (backend/, FastAPI + WNTR) runs on 8000. Set WATERBOARD_API to point elsewhere.
    proxy: { "/api": process.env.WATERBOARD_API ?? "http://localhost:8000" },
  },
});
