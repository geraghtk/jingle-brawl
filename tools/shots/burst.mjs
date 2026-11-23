// A game in motion, for judging animation (a still cannot show a glitch between two poses):
//   node tools/shots/burst.mjs mecha_santa out_dir [--bots 4] [--from 0] [--to 30] [--every 120] [--tv 960x540] [--phone]
// Like lab-shots.mjs (a bots-only Game Lab game against `npm run dev`), but it photographs the TV every
// `--every` ms between `--from` and `--to` seconds of the game, as out_dir/tv-<ms>.jpg (and with --phone a
// joined phone as phone-<ms>.jpg), and writes out_dir/events.json: every new event a runtime game's state
// reported (which team or side, what, when) so the frames round a moment can be found. Lay frames out side
// by side with `python tools/shots/sheet.py out_dir <fromMs> <toMs>`.
import { chromium } from '@playwright/test';
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const args = process.argv.slice(2);
const opt = (k, d) => (args.includes(`--${k}`) ? args[args.indexOf(`--${k}`) + 1] : d);
const [gameId = 'mecha_santa', out = 'burst'] = args;
const base = opt('base', 'http://localhost:5173'), debug = opt('debug', 'http://127.0.0.1:8787');
const bots = Number(opt('bots', 4)), from = Number(opt('from', 0)) * 1000, to = Number(opt('to', 30)) * 1000, every = Number(opt('every', 120));
const size = (k, d) => { const [width, height] = String(opt(k, d)).split('x').map(Number); return { width, height }; };
mkdirSync(out, { recursive: true });

const browser = await chromium.launch();
const host = await (await browser.newContext({ viewport: { width: 1440, height: 900 } })).newPage();
await host.goto(`${base}/host.html`);
await host.getByRole('button', { name: /Create New Game/i }).click();
await host.waitForSelector('#game-screen', { state: 'visible' });
const room = (await host.locator('#room-code-display').textContent()).trim();
const tv = await (await browser.newContext({ viewport: size('tv', '960x540') })).newPage();
tv.on('pageerror', (e) => console.log('TV ERROR', e.message));
await tv.goto(`${base}/tv.html?room=${room}`);
let phone = null;
if (args.includes('--phone')) {
  phone = await (await browser.newContext({ viewport: size('phone-size', '390x844') })).newPage();
  phone.on('pageerror', (e) => console.log('PHONE ERROR', e.message));
  await phone.goto(`${base}/player.html?room=${room}&testId=1&name=Kevin`);
  await phone.getByRole('button', { name: /join/i }).first().click().catch(() => {});
  await host.locator('#lobby-roster-list .roster-row').first().waitFor();
}
await host.getByTestId('lab-open').click();
for (let i = 0; i < bots; i++) { await host.getByTestId('lab-add-bot').click(); await host.waitForTimeout(250); }
await host.getByTestId('lab-game').selectOption(gameId);
await host.getByTestId('lab-start').click();
const brief = tv.getByTestId('arena-briefing').or(tv.getByTestId('party-ready'));
if (await brief.first().waitFor({ timeout: 4000 }).then(() => true, () => false)) {
  if (phone) await phone.getByTestId('briefing-ready').or(phone.getByTestId('party-ready-btn')).first().click().catch(() => {});
  if (from < 0) { /* (a negative --from keeps the opening in the frames) */ } else await brief.first().waitFor({ state: 'detached', timeout: 15000 }).catch(() => {});
}
const t0 = Date.now();
const events = [], seen = new Set();
const watch = setInterval(async () => {
  try {
    const a = await (await fetch(`${debug}/debug/rooms/${room}/arena`)).json();
    const at = Date.now() - t0;
    const lists = [...(a?.state?.teams || []).map((t, i) => [`team${i}`, t.walker?.events || []]), ['field', a?.state?.field?.events || []]];
    for (const [who, list] of lists) for (const e of list) {
      const key = `${who}|${e.n}`;
      if (seen.has(key)) continue;
      seen.add(key);
      events.push({ at, who, kind: e.kind, ...(e.id ? { id: e.id } : {}), ...(e.what ? { what: e.what } : {}) });
    }
    if (a?.result && !seen.has('result')) { seen.add('result'); events.push({ at, who: 'game', kind: 'result' }); }
  } catch { /* the room is between states */ }
}, 150);
await tv.waitForTimeout(Math.max(0, from));
let n = 0;
while (Date.now() - t0 < to) {
  const ms = Date.now() - t0, name = String(ms).padStart(6, '0');
  await tv.screenshot({ path: join(out, `tv-${name}.jpg`), type: 'jpeg', quality: 62 });
  if (phone) await phone.screenshot({ path: join(out, `phone-${name}.jpg`), type: 'jpeg', quality: 62 });
  n++;
  await tv.waitForTimeout(Math.max(0, every - (Date.now() - t0 - ms)));
}
clearInterval(watch);
writeFileSync(join(out, 'events.json'), JSON.stringify(events, null, 1));
console.log(`room ${room}: ${n} frames, ${events.length} events`);
await browser.close();
