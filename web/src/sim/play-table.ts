// Watch a party at the table with simple guests (each presses one of their own buttons at random),
// on either backend: a quick check that a backend plays a whole game, and a demo.
//
//   npx vite-node web/src/sim/play-table.ts -- [--live] [--headed] [--players 4] [--seed 1] [--minutes 30]
import { HOST } from '../core/table';
import { rng } from '../core/rng';
import { LiveTable } from './live';
import { Table } from './table';

const arg = (k: string, d: string) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : d; };
const live = process.argv.includes('--live');
const n = Number(arg('players', '4'));
const seed = Number(arg('seed', '1'));
const minutes = Number(arg('minutes', live ? '30' : '240'));
const NAMES = ['Ada', 'Bo', 'Cy', 'Di', 'Ed', 'Flo', 'Gus', 'Hal'];
const people = NAMES.slice(0, n).map((name, i) => ({ id: name.toLowerCase(), name, skills: { duel: (i + 1) / (n + 1), whack: 1 - i / n, sketch: ((i * 3) % n) / n } }));

(async () => {
  const o = { seed, people, settings: {}, autoHost: live ? { paceMs: 1500, duelMs: 3000, patienceMs: 20000 } : true };
  const t = live ? await LiveTable.open({ ...o, headed: process.argv.includes('--headed') }) : new Table(o);
  if (t instanceof LiveTable) console.log(`room ${t.room}`);
  const r = rng(seed);
  const end = Date.now() + minutes * 60000;
  let clock = 0;
  while (!t.ended && (live ? Date.now() < end : clock < minutes * 60000)) {
    for (const p of people) {
      const os = await t.offers(p.id);
      if (!os.length || r() < 0.4) continue;
      const pick = os[Math.floor(r() * os.length)];
      const res = await t.act(p.id, pick);
      if (!res.ok) console.log(`  ✗ ${p.name} ${pick.type}: ${res.error}`);
    }
    const events = await t.advance(live ? 1000 : 3000);
    clock += 3000;
    for (const e of events) if (!e.private) console.log(`${(e.t / 1000).toFixed(0).padStart(5)}s  ${e.text}`);
  }
  const res = await t.result();
  console.log(JSON.stringify({ ...res, host: res.host }, null, 1));
  void HOST;
  if (t instanceof LiveTable) await t.close();
  process.exit(res.ended ? 0 : 1);
})();
