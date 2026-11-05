// The table over JSON lines (stdin → stdout), so a simulator in another language (showrunner's
// Simulation Lab, Python) can seat its people at a game of Yule Duel. One request per line, one
// reply per line. Two backends behind the same commands:
//   engine  the real rules engine on a virtual clock (sim/table.ts): fast, seeded, replayable
//   live    the real app in a browser against the emulators (sim/live.ts): real screens, real time
//
//   {"cmd": "new", "backend": "engine", "seed": 1, "settings": {}, "autoHost": true,
//    "people": [{"id": "ada", "name": "Ada", "age": 12, "skills": {"duel": 0.7, "whack": 0.4, "sketch": 0.9}}]}
//                                     → {"ok": true, "people": [...ids], "host": "host", "room"?}
//   {"cmd": "observe", "who": "ada"}  → {now, tv, phone, offers, buttons, you, focus, roles, gifts, values, drawings?}
//                                     ("host": the console; buttons: each offer in words, for whoever watches)
//   {"cmd": "act", "who": "ada", "offer": {"type": "bid", "amount": 2}}  → {ok, error?, ended}
//   {"cmd": "advance", "ms": 5000}    → {now, ended, events: [...]}  (events marked private are only the actor's)
//   {"cmd": "state"}                  → {ended, game}
//   {"cmd": "frames"}                 → {tv, <person id>: JPEG base64}  (live: the pages as pictures; engine: {})
//   {"cmd": "result"}                 → {ended, minutes, seats: [{id, chips, gift, value, partyPoints, did}], host, issues?}
//   {"cmd": "close"}
//
// An offer is anything from observe's `offers`, sent back as it came (Whack-an-Elf's score and the
// Sketch-Off drawing may be left out: the person's own hands fill them in from their skills).
//
//   npx vite-node web/src/sim/table-bridge.ts
import * as readline from 'node:readline';
import type { Offer, Who } from '../core/table';
import { HOST } from '../core/table';
import { LiveTable } from './live';
import { buttons, Table } from './table';

type Backend = Table | LiveTable;
let table: Backend | null = null;

async function handle(req: any): Promise<unknown> {
  if (req.cmd === 'new') {
    if (table instanceof LiveTable) await table.close();
    const o = { seed: Number(req.seed) || 0, people: req.people || [], settings: req.settings || {}, autoHost: req.autoHost ?? true };
    if (!o.people.length) return { error: 'seat some people' };
    table = req.backend === 'live' ? await LiveTable.open({ ...o, base: req.base, headed: !!req.headed, slowMo: req.slowMo }) : new Table(o);
    return { ok: true, people: o.people.map((p: { id: string }) => p.id), host: HOST, ...(table instanceof LiveTable ? { room: table.room } : {}) };
  }
  if (!table) return { error: 'no table yet: send "new" first' };
  const t = table;
  switch (req.cmd) {
    case 'observe': {
      const o = await t.observe(String(req.who) as Who);
      return { ...o, buttons: buttons(o.offers, t.game, t.people) };
    }
    case 'act': return { ...(await t.act(String(req.who) as Who, req.offer as Offer)), ended: t.ended };
    case 'advance': {
      const events = await t.advance(Number(req.ms) || 0);
      return { now: t instanceof Table ? t.now - 1_700_000_000_000 : Date.now(), ended: t.ended, events };
    }
    case 'state': return { ended: t.ended, game: t.game };
    case 'frames': return t instanceof LiveTable ? t.frames() : {};
    case 'result': return t.result();
    case 'close':
      if (t instanceof LiveTable) await t.close();
      table = null;
      return { ok: true };
    default: return { error: `unknown cmd ${req.cmd}` };
  }
}

// one request at a time, in order (a live press takes a while)
const rl = readline.createInterface({ input: process.stdin });
let queue = Promise.resolve();
rl.on('line', (line) => {
  queue = queue.then(async () => {
    let out: unknown;
    try { out = await handle(JSON.parse(line)); } catch (e) { out = { error: String((e as Error).message || e) }; }
    process.stdout.write(JSON.stringify(out) + '\n');
  });
});
rl.on('close', () => { queue.then(async () => { if (table instanceof LiveTable) await table.close(); process.exit(0); }); });
