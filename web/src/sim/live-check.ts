// Checks for the live table (sim/live.ts): run by e2e/table-live.spec.js against the app it serves.
// Prints one JSON line per check ({check, ok, detail}) and exits 1 if any failed.
//
//   BASE_URL=http://localhost:4173 npx vite-node web/src/sim/live-check.ts
import { rng } from '../core/rng';
import { LiveTable } from './live';

const out = (check: string, ok: boolean, detail: unknown = '') => { console.log(JSON.stringify({ check, ok, detail })); return ok; };
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const brisk = { paceMs: 1000, duelMs: 1500, patienceMs: 15000 };

/** A JPEG's height in pixels (its start-of-frame marker). */
function jpegHeight(b64 = ''): number {
  const b = Buffer.from(b64, 'base64');
  for (let i = 2; i + 8 < b.length && b[i] === 0xff;) {
    if (b[i + 1] >= 0xc0 && b[i + 1] <= 0xc3) return b.readUInt16BE(i + 5);
    i += 2 + b.readUInt16BE(i + 2);
  }
  return 0;
}

async function wholeGame() {
  const people = ['Ada', 'Bo', 'Cy'].map((name, i) => ({ id: name.toLowerCase(), name, skills: { duel: 0.3 * i, whack: 0.5, sketch: 0.5 } }));
  const t = await LiveTable.open({ seed: 1, people, settings: { voteWindow: 0, duelPick: 'off' }, autoHost: brisk });
  try {
    const r = rng(1);
    const end = Date.now() + 6 * 60000;
    while (!t.ended && Date.now() < end) {
      await t.refresh();
      for (const p of people) {
        const os = t.offers(p.id);
        if (os.length && r() < 0.6) await t.act(p.id, os[Math.floor(r() * os.length)]);
      }
      await t.advance(500);
    }
    const res = t.result();
    // the TV's picture is the whole host page: the boards below the fold too (the Studio scrolls it)
    const [tall, seen] = await t['hostPage'].evaluate(() => [document.documentElement.scrollHeight, innerHeight]);
    const tv = jpegHeight((await t.frames()).tv);
    return [out('random guests and the scripted host play a whole game on the real pages, with no page errors or missing buttons',
      res.ended && !res.issues.length && res.seats.every((s) => s.gift), { minutes: res.minutes, issues: res.issues, room: res.room }),
    out("the TV's picture is the whole host page, below the fold too", tall > seen && tv >= tall, { picture: tv, page: tall, window: seen }),
    out('guests pick their avatars on the join screen; the app has each one they picked (or gave them)',
      Object.values(t.avatars).some((a) => a.picked) && people.every((p) => t.game.players?.[t['idOf'][p.id]]?.avatar === t.avatars[p.id].avatar),
      { picks: t.avatars, app: Object.fromEntries(people.map((p) => [p.id, t.game.players?.[t['idOf'][p.id]]?.avatar])) })];
  } finally { await t.close(); }
}

async function whackOnce() {
  // everyone bids the same on the first gift: a tie, so the whole table plays Whack-an-Elf
  const people = ['Ada', 'Bo', 'Cy', 'Di'].map((name) => ({ id: name.toLowerCase(), name }));
  const t = await LiveTable.open({ seed: 2, people, settings: { voteWindow: 0, duelPick: 'off', tiebreakerMethod: 'whack_elf' }, autoHost: brisk });
  try {
    const until = (ms: number, fn: () => boolean) => (async () => { const end = Date.now() + ms; while (Date.now() < end) { await t.advance(300); if (fn()) return true; } return false; })();
    await until(30000, () => t.game.currentTurn?.phase === 'bidding');
    for (const p of people) if (t.offers(p.id).some((o) => o.type === 'bid')) await t.act(p.id, { type: 'bid', amount: 2 });
    await until(30000, () => people.some((p) => t.offers(p.id).some((o) => o.type === 'fight')));
    const opener = people.find((p) => t.offers(p.id).some((o) => o.type === 'fight'))!;
    await t.act(opener.id, { type: 'fight' });
    if (!(await until(40000, () => t.offers('ada').some((o) => o.type === 'whack')))) return out('Whack-an-Elf is offered once a guest is playing', false, 'no whack offered');
    const r = await t.act('ada', { type: 'whack' });
    await sleep(300);
    const again = t.offers('ada').some((o) => o.type === 'whack');
    return out("once a guest is whacking, their phone doesn't offer Whack-an-Elf again until they're done", r.ok && !again, { pressed: r, again });
  } finally { await t.close(); }
}

(async () => {
  const results = [await whackOnce(), ...(await wholeGame())];
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch((e) => { out('crashed', false, String(e?.stack || e)); process.exit(1); });
