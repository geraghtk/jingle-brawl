// @ts-check
const { defineConfig, devices } = require('@playwright/test');

/** @see https://playwright.dev/docs/test-configuration */
// The app (web/) is built fresh and served by `vite preview` on :4173. BASE_URL=… points the
// suite at something already running instead (e.g. the dev server on :5173).
const BASE = process.env.BASE_URL || 'http://localhost:4173';
// TRANSPORT=server (npm run test:e2e:server): the screens talk to the room server, which serves the
// build itself with its test-only debug API; no Firebase emulators.
const ON_SERVER = process.env.TRANSPORT === 'server';

module.exports = defineConfig({
  testDir: './e2e',
  globalSetup: require.resolve('./e2e/global-setup.js'),
  fullyParallel: false, // Run tests sequentially (multi-user flows share Firebase state)
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: 1,
  reporter: 'html',
  use: {
    baseURL: BASE,
    trace: 'retain-on-failure', // the intermittent join (≈1 in 50 joins under load) leaves a trace to read
    video: 'on-first-retry',
    screenshot: 'only-on-failure',
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
  ],
  webServer: process.env.CI ? undefined : ON_SERVER ? [{
    command: 'npm run build:server-mode && node dist-server/index.mjs --port 4173 --host 127.0.0.1 --db :memory: --debug-api',
    url: 'http://localhost:4173/healthz',
    reuseExistingServer: false,
    timeout: 120000,
  }] : [
    ...(process.env.BASE_URL ? [] : [{
      command: 'npm run build && npm run preview',
      url: 'http://localhost:4173/host.html',
      reuseExistingServer: false, // always test a fresh build
      timeout: 120000,
    }]),
    {
      // Local RTDB + Auth emulators — pages on localhost connect automatically
      // (see web/src/firebase/app.ts). Requires Java 21+.
      command: 'firebase emulators:start --only database,auth --project jingle-brawl-e8308',
      url: 'http://127.0.0.1:9000/.json?ns=jingle-brawl-e8308-default-rtdb',
      reuseExistingServer: !process.env.CI,
      timeout: 60000,
    },
  ],
});
