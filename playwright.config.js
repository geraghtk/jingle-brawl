// @ts-check
const { defineConfig, devices } = require('@playwright/test');

/** @see https://playwright.dev/docs/test-configuration */
// The app (web/) is built fresh and served by `vite preview` on :4173. BASE_URL=… points the
// suite at something already running instead (e.g. the dev server on :5173).
const BASE = process.env.BASE_URL || 'http://localhost:4173';
module.exports = defineConfig({
  testDir: './e2e',
  fullyParallel: false, // Run tests sequentially (multi-user flows share Firebase state)
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: 1,
  reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : 'html',
  use: {
    baseURL: BASE,
    trace: 'retain-on-failure', // the intermittent join (≈1 in 50 joins under load) leaves a trace to read
    video: 'on-first-retry',
    screenshot: 'only-on-failure',
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
  ],
  // The room server serves the fresh build itself, with its test-only debug API (this machine only).
  webServer: process.env.BASE_URL ? undefined : [{
    command: 'npm run build && node dist-server/index.mjs --port 4173 --host 127.0.0.1 --db :memory: --debug-api',
    env: { JB_FAKE_CRITIC: '1' },                   // AI Draw Off judged by the fake critic: no network, no key spent
    url: 'http://localhost:4173/healthz',
    reuseExistingServer: false, // always test a fresh build
    timeout: 120000,
  }],
});
