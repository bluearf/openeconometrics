import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8765",
      "/chart-assets": "http://127.0.0.1:8765",
    },
  },
  build: {
    outDir: "../src/openecon/static",
    emptyOutDir: true,
    rollupOptions: {
      output: {
        manualChunks: {
          editor: [
            "codemirror",
            "@codemirror/lang-python",
            "@codemirror/view",
            "@codemirror/state",
            "@codemirror/commands",
          ],
        },
      },
    },
  },
});
