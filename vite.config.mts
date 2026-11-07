// The apps: web/host.html (TV) and web/player.html (phones), built to dist/ with web/public
// (art, rule pages).
import { resolve } from 'node:path';
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  root: 'web',
  plugins: [react()],
  build: {
    outDir: '../dist',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        host: resolve(__dirname, 'web/host.html'),
        player: resolve(__dirname, 'web/player.html'),
      },
      output: {
        // Libraries change rarely: their own chunks stay cached across app deploys
        manualChunks(id) {
          if (/node_modules[\/](firebase|@firebase)[\/]/.test(id)) return 'firebase';
          if (/node_modules[\/](react|react-dom|scheduler)[\/]/.test(id)) return 'react';
        },
      },
    },
  },
  // Polling: on Windows the native watcher missed edits and kept serving stale modules (the host
  // and phones then ran different versions of the same rules)
  server: { port: 5173, strictPort: true, watch: { usePolling: true, interval: 300 } },
  preview: { port: 4173, strictPort: true },
  test: {
    root: '.',
    include: ['web/src/**/*.test.ts', 'web/src/**/*.test.tsx'],
    globals: true,
  },
});
