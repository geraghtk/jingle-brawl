// The Simulation Lab's bridge: a game of Yule Duel driven one decision at a time over JSON lines
// (stdin → stdout), so a simulator in another language (showrunner's simlab, Python) can play it
// through the real rules engine. One request per line, one reply per line:
//
//   {"cmd": "new", "seed": 1, "players": 6, "settings": {}}   → the first decision
//   {"cmd": "observe", "player": "p2"}                         → what that player sees now
//   {"cmd": "legal"}                                           → the current decision's options
//   {"cmd": "apply", "action": 3}                              → the next decision, or the result
//   {"cmd": "snapshot"}                                        → the game state and the step count
//   {"cmd": "reset", "seed": 2}                                → a new game, same table and settings
//
// A decision is {kind: bid|gambit|fight|reprisal|steal, player, ...}. Options: a bid is a whole
// number 0..max; gambit, reprisal and steal are a gift id or null (keep / skip / auction); fight is
// true or false. An answer that isn't an option is refused ({"error"}) and nothing moves. The game
// itself is sim/simulate.ts steps(): the same code and random draws as the economics report.
//
//   npx vite-node web/src/sim/bridge.ts
import * as readline from 'node:readline';
import type { Settings } from '../core/types';
import { steps } from './simulate';
import type { Decision, GameResult } from './simulate';

let it: Generator<Decision, GameResult, unknown> | null = null;
let cur: IteratorResult<Decision, GameResult> | null = null;
let table = { players: 6, settings: {} as Settings };
let count = 0;

function options(d: Decision): (number | string | boolean | null)[] {
  switch (d.kind) {
    case 'bid': return Array.from({ length: d.max + 1 }, (_, i) => i);
    case 'fight': return [true, false];
    case 'gambit': case 'reprisal': case 'steal': return [null, ...d.targets.map((t) => t.id)];
  }
}

function describe(d: Decision) {
  const base = { kind: d.kind, player: d.player };
  switch (d.kind) {
    case 'bid': return { ...base, giftId: d.giftId, max: d.max };
    case 'fight': return { ...base, giftId: d.giftId };
    case 'gambit': case 'reprisal': return { ...base, targets: d.targets };
    case 'steal': return { ...base, targets: d.targets, misfitId: d.misfitId };
  }
}

/** What a player perceives: their own chips and holding, and what each gift in play is worth to them. */
function observe(player: string) {
  if (!cur || cur.done) return { error: 'no decision pending' };
  const d = cur.value;
  const g = d.game;
  const me = g.players?.[player];
  if (!me) return { error: `no player ${player}` };
  const v = d.view;
  const gifts = Object.keys(g.gifts || {});
  // Values are private: a player only ever sees their own (v belongs to the decider; others get theirs
  // through the same rule: the bridge asks the generator's view only for the decider).
  const values = d.player === player ? Object.fromEntries(gifts.map((id) => [id, Math.round(v.value(id) * 100) / 100])) : {};
  return {
    player, chips: me.chips, hasGift: !!me.giftId, holding: d.player === player ? v.holding : null,
    progress: v.progress, endgame: v.endgame, phase: g.phase, myTurn: d.player === player,
    decision: d.player === player ? describe(d) : null, values,
    others: Object.fromEntries(Object.entries(g.players || {}).filter(([id]) => id !== player).map(([id, p]) => [id, { chips: p.chips, hasGift: !!p.giftId }])),
  };
}

function result(r: GameResult) {
  return { done: true, result: r };
}

function start(seed: number) {
  it = steps({ settings: table.settings, names: Array.from({ length: table.players }, (_, i) => `p${i}`), seed });
  cur = it.next();
  count = 0;
  return cur.done ? result(cur.value) : { done: false, decision: describe(cur.value) };
}

function handle(req: any) {
  switch (req.cmd) {
    case 'new':
      table = { players: Number(req.players) || 6, settings: req.settings || {} };
      return start(Number(req.seed) || 0);
    case 'reset':
      return start(Number(req.seed) || 0);
    case 'observe':
      return observe(String(req.player));
    case 'legal':
      if (!cur || cur.done) return { options: [] };
      return { decision: describe(cur.value), options: options(cur.value) };
    case 'apply': {
      if (!it || !cur || cur.done) return { error: 'the game is over' };
      const opts = options(cur.value);
      const a = req.action === undefined ? null : req.action;
      if (!opts.some((o) => o === a)) return { error: `not an option for ${cur.value.kind}: ${JSON.stringify(a)}` };
      cur = it.next(a);
      count++;
      return cur.done ? result(cur.value) : { done: false, decision: describe(cur.value) };
    }
    case 'snapshot':
      return { steps: count, done: !!cur?.done, game: cur && !cur.done ? cur.value.game : null };
    default:
      return { error: `unknown cmd ${req.cmd}` };
  }
}

const rl = readline.createInterface({ input: process.stdin });
rl.on('line', (line) => {
  let out: unknown;
  try { out = handle(JSON.parse(line)); } catch (e) { out = { error: String((e as Error).message || e) }; }
  process.stdout.write(JSON.stringify(out) + '\n');
});
