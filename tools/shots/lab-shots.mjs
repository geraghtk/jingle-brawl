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

const size = (k, d) => { const [width, height] = String(opt(k, d)).split('x').map(Number); return { width, height }; };   // --tv 1280x720, --phone-size 320x568
const tv = await (await browser.newContext({ viewport: size('tv', '1920x1080') })).newPage();
tv.on('pageerror', (e) => console.log('TV ERROR', e.message));
await tv.goto(`${base}/tv.html?room=${room}`);

let phone = null;
if (args.includes('--phone')) {
  phone = await (await browser.newContext({ viewport: size('phone-size', '390x844'), deviceScaleFactor: 2 })).newPage();
  await phone.goto(`${base}/player.html?room=${room}&testId=1&name=Kevin`);
  await phone.getByRole('button', { name: /join/i }).first().click().catch(() => {});
  await host.locator('#lobby-roster-list .roster-row').first().waitFor();
}

await host.getByTestId('lab-open').click();
for (let i = 0; i < bots; i++) { await host.getByTestId('lab-add-bot').click(); await host.waitForTimeout(250); }
await host.screenshot({ path: join(out, `${gameId}-lab.png`) });
await host.getByTestId('lab-game').selectOption(gameId);
await host.getByTestId('lab-start').click();
// the instructions come up first (unless the room has them off): a picture of them, then the phone says Ready
// (bots are ready by themselves) and the 3, 2, 1 runs; the `--at` seconds count from the game's own start
const brief = tv.getByTestId('arena-briefing').or(tv.getByTestId('party-ready'));
if (await brief.first().waitFor({ timeout: 4000 }).then(() => true, () => false)) {
  await tv.waitForTimeout(700);
  await tv.screenshot({ path: join(out, `${gameId}-tv-ready.png`) });
  await host.screenshot({ path: join(out, `${gameId}-console-ready.png`) });
  if (phone) {
    await phone.screenshot({ path: join(out, `${gameId}-phone-ready.png`) });
    await phone.getByTestId('briefing-ready').or(phone.getByTestId('party-ready-btn')).first().click().catch(() => {});
  }
  await brief.first().waitFor({ state: 'detached', timeout: 15000 }).catch(() => {});
}
const t0 = Date.now();
// Sketch-Off with a phone: it draws a little (a face), so the pad and the big screen's wall have something on them
if (phone && gameId === 'sketch_off') {
  phone.getByTestId('sketch-pad').waitFor({ timeout: 20000 }).then(async () => {
    const b = await phone.getByTestId('sketch-pad').boundingBox();
    const at = (x, y) => [b.x + b.width * x, b.y + b.height * y];
    const stroke = async (pts) => { await phone.mouse.move(...at(...pts[0])); await phone.mouse.down(); for (const p of pts.slice(1)) await phone.mouse.move(...at(...p), { steps: 6 }); await phone.mouse.up(); };
    const ring = (cx, cy, r) => Array.from({ length: 25 }, (_, i) => [cx + r * Math.cos(i / 24 * 6.283), cy + r * Math.sin(i / 24 * 6.283)]);
    await stroke(ring(0.5, 0.5, 0.33)); await phone.waitForTimeout(1500);
    await stroke(ring(0.38, 0.42, 0.04)); await stroke(ring(0.62, 0.42, 0.04)); await phone.waitForTimeout(1500);
    await stroke(Array.from({ length: 13 }, (_, i) => [0.34 + i * 0.027, 0.6 + 0.09 * Math.sin(i / 12 * 3.1416)]));
  }).catch(() => {});
}
for (const s of at) {
  await tv.waitForTimeout(Math.max(0, t0 + s * 1000 - Date.now()));
  await tv.screenshot({ path: join(out, `${gameId}-tv-${s}.png`) });
  if (phone) await phone.screenshot({ path: join(out, `${gameId}-phone-${s}.png`) });
  console.log('shot', s);
}
await host.screenshot({ path: join(out, `${gameId}-console.png`) });
await browser.close();
