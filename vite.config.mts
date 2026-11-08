// The apps: web/host.html (the host's console), web/tv.html (the shared screen) and
// web/player.html (phones), built to dist/ with web/public
// (art, rule pages).
import { resolve } from 'node:path';
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

const target = process.env.ROOM_SERVER || 'http://127.0.0.1:8787';
// (not /debug: through the proxy every request looks local; the test helpers call :8787 directly)
const ROOM_SERVER = { '/api': target, '/healthz': target, '/ws': { target, ws: true } };

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
        tv: resolve(__dirname, 'web/tv.html'),
      },
      output: {
        // Libraries change rarely: their own chunks stay cached across app deploys
        manualChunks(id) {
          if (/node_modules[\/](react|react-dom|scheduler)[\/]/.test(id)) return 'react';
        },
      },
    },
  },
  // Polling: on Windows the native watcher missed edits and kept serving stale modules (the host
  // and phones then ran different versions of the same rules). The room server (npm run server,
  // :8787) sits behind the dev and preview servers, so the pages talk to it on their own origin.
  server: { port: 5173, strictPort: true, watch: { usePolling: true, interval: 300 }, proxy: ROOM_SERVER },
  preview: { port: 4173, strictPort: true, proxy: ROOM_SERVER },
  test: {
    root: '.',
    include: ['web/src/**/*.test.ts', 'web/src/**/*.test.tsx', 'server/src/**/*.test.ts'],
    globals: true,
  },
});
