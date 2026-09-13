import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    allowedHosts: ["plastics-cite-incidents-reason.trycloudflare.com"],
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
});
