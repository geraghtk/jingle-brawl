// The live table (web/src/sim/live.ts): simulated guests press real buttons on the real pages. Its
// checks live in web/src/sim/live-check.ts (TypeScript, run with vite-node against the app this
// suite serves); each prints a JSON line, and this test fails on any that didn't pass.
const { test, expect } = require('@playwright/test');
const { spawnSync } = require('child_process');
const H = require('./helpers');


test('the live table: guests play a whole game on the real pages; Whack-an-Elf is played once per guest; the TV picture is the whole page; guests pick avatars', async () => {
  test.setTimeout(15 * 60000);
  const run = spawnSync(process.platform === 'win32' ? 'npx.cmd' : 'npx', ['vite-node', 'web/src/sim/live-check.ts'], {
    cwd: `${__dirname}/..`, env: { ...process.env, BASE_URL: H.BASE }, encoding: 'utf-8', shell: process.platform === 'win32',
    timeout: 14 * 60000,
  });
  const checks = (run.stdout || '').split('\n').filter((l) => l.startsWith('{')).map((l) => JSON.parse(l));
  for (const c of checks) console.log(`${c.ok ? 'ok  ' : 'FAIL'} ${c.check}${c.ok ? '' : ` ${JSON.stringify(c.detail)}`}`);
  expect(checks.length, run.stderr).toBe(4);
  expect(checks.filter((c) => !c.ok)).toEqual([]);
});
