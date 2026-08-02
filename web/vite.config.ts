import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The frontend talks to the FastAPI backend on :8000. In dev we proxy /api to it so the
// browser makes same-origin requests (no CORS surprises) and nothing is hard-coded.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
  // three.js is large. Pre-bundling it keeps the first dev load from stalling for seconds,
  // and splitting it into its own chunk means the scoring tools are not blocked on the 3D
  // code downloading.
  optimizeDeps: {
    include: ["three", "@react-three/fiber", "@react-three/drei"],
  },
  build: {
    chunkSizeWarningLimit: 1200,
    rollupOptions: {
      output: {
        manualChunks: {
          three: ["three"],
          drei: ["@react-three/drei", "@react-three/postprocessing", "postprocessing"],
          motion: ["framer-motion"],
        },
      },
    },
  },
});
