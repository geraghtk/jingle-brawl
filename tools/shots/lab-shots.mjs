// Pictures of a Game Lab game played by practice bots, against the dev server (npm run dev):
//   node tools/shots/lab-shots.mjs gift_conveyor out_dir [--bots 4] [--at 6,20,45] [--base http://localhost:5173] [--phone]
// Opens the host console, makes a room, adds bots, starts the game and saves the TV (1920 x 1080) at each
// of the `--at` seconds (and the console, and with --phone one joined phone that plays alongside nobody).
import { chromium } from '@playwright/test';
import { mkdirSync } from 'node:fs';
import { join } from 'node:path';

const args = process.argv.slice(2);
const opt = (k, d) => (args.includes(`--${k}`) ? args[args.indexOf(`--${k}`) + 1] : d);
const [gameId = 'gift_conveyor', out = 'shots'] = args;
const base = opt('base', 'http://localhost:5173');
const bots = Number(opt('bots', 4));
const at = String(opt('at', '6,20,45')).split(',').map(Number);
mkdirSync(out, { recursive: true });

const browser = await chromium.launch();
const host = await (await browser.newContext({ viewport: { width: 1440, height: 900 } })).newPage();
host.on('pageerror', (e) => console.log('HOST ERROR', e.message));
await host.goto(`${base}/host.html`);
await host.getByRole('button', { name: /Create New Game/i }).click();
await host.waitForSelector('#game-screen', { state: 'visible' });
const room = (await host.locator('#room-code-display').textContent()).trim();
console.log('room', room);

const tv = await (await browser.newContext({ viewport: { width: 1920, height: 1080 } })).newPage();
tv.on('pageerror', (e) => console.log('TV ERROR', e.message));
await tv.goto(`${base}/tv.html?room=${room}`);

let phone = null;
if (args.includes('--phone')) {
  phone = await (await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2 })).newPage();
  await phone.goto(`${base}/player.html?room=${room}&testId=1&name=Kevin`);
  await phone.getByRole('button', { name: /join/i }).first().click().catch(() => {});
  await host.locator('#lobby-roster-list .roster-row').first().waitFor();
}

await host.getByTestId('lab-open').click();
for (let i = 0; i < bots; i++) { await host.getByTestId('lab-add-bot').click(); await host.waitForTimeout(250); }
await host.screenshot({ path: join(out, `${gameId}-lab.png`) });
await host.getByTestId('lab-game').selectOption(gameId);
await host.getByTestId('lab-start').click();
const t0 = Date.now();
for (const s of at) {
  await tv.waitForTimeout(Math.max(0, t0 + s * 1000 - Date.now()));
  await tv.screenshot({ path: join(out, `${gameId}-tv-${s}.png`) });
  if (phone) await phone.screenshot({ path: join(out, `${gameId}-phone-${s}.png`) });
  console.log('shot', s);
}
await host.screenshot({ path: join(out, `${gameId}-console.png`) });
await browser.close();
