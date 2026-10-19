// @ts-check
const { defineConfig, devices } = require('@playwright/test');

/** @see https://playwright.dev/docs/test-configuration */
module.exports = defineConfig({
  testDir: './e2e',
  globalSetup: require.resolve('./e2e/global-setup.js'),
  fullyParallel: false, // Run tests sequentially (multi-user flows share Firebase state)
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: 1,
  reporter: 'html',
  use: {
    baseURL: process.env.BASE_URL || 'http://localhost:8080',
    trace: 'on-first-retry',
    video: 'on-first-retry',
    screenshot: 'only-on-failure',
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
  ],
  webServer: process.env.CI ? undefined : [
    {
      command: 'python -m http.server 8080',
      url: 'http://localhost:8080',
      reuseExistingServer: !process.env.CI,
    },
    {
      // Local RTDB + Auth emulators — pages on localhost connect automatically
      // (see connectGameDatabase in firebase-config.js). Requires Java 21+.
      command: 'firebase emulators:start --only database,auth --project jingle-brawl-e8308',
      url: 'http://127.0.0.1:9000/.json?ns=jingle-brawl-e8308-default-rtdb',
      reuseExistingServer: !process.env.CI,
      timeout: 60000,
    },
  ],
});
