// Mecha Santa Mayhem, hundreds of races through the real engine: people with different reaction
// times and slip rates do their jobs (through the game's own offers, late and sometimes wrong).
// Measures how long a race takes, how often it reaches the bell, whether the better team wins, and
// what each kind of mistake costs.
//   npm run analyze:mecha [-- --races 400]   (mecha-report.ts)
import { rng as makeRng } from '../core/rng';
import type { Rng } from '../core/rng';
import { apply, NOW, table } from '../core/testkit';
import { Tx } from '../core/tx';
import type { Game } from '../core/types';
import { assignTeams } from '../core/minigames/engines/teams';
import { mecha, onDuty } from '../core/minigames/mecha/mecha';
import type { MechaAction, MechaState } from '../core/minigames/mecha/mecha';
import { arenaAct, arenaDeadline, arenaOf, startArena } from '../core/minigames/runtime';

/** A person: how long they take to press once there is something to press (ms), and how often they press the wrong thing. */
export interface Hands { react: number; slip: number }
export interface Race { winner: 'A' | 'B' | null; reason: string; ms: number; finished: boolean; distance: [number, number]; slips: [number, number]; kinds: Record<string, number>[] }

const st = (g: Game) => arenaOf(g)!.state as MechaState;

/** One race on a virtual clock: `teams[0]` are team A's people in seat order (captain first), `teams[1]` team B's. */
export function race(teams: [Hands[], Hands[]], seed: number, config: Record<string, unknown> = {}): Race {
  const r: Rng = makeRng(seed);
  const n = teams[0].length + teams[1].length;
  let g = table(n);
  g.currentTurn = { phase: 'duel', challengerId: 'p1', defenderId: 'p2', pot: 2 };
  const ids = Object.keys(g.players!);
  // p1 and p2 captain; the rest fill team A then team B in order
  const rest = ids.slice(2);
  const a = ['p1', ...rest.slice(0, teams[0].length - 1)], b = ['p2', ...rest.slice(teams[0].length - 1)];
  const deal = { ...assignTeams(['p1', 'p2'], ids, makeRng(1)), teams: [{ teamId: 'A', captainId: 'p1', memberIds: a }, { teamId: 'B', captainId: 'p2', memberIds: b }] };
  const hands: Record<string, Hands> = {};
  a.forEach((id, i) => { hands[id] = teams[0][i]; });
  b.forEach((id, i) => { hands[id] = teams[1][i]; });
  const tx = new Tx(g);
  const started = startArena(tx, 'mecha_santa', { mode: 'captain_teams', participants: ids, duelists: ['p1', 'p2'], config: { teams: deal, ...config } }, { now: NOW, rng: r });
  if (started.error) throw new Error(started.error);
  g = apply(g, tx.plan());
  const ctx = (now: number) => ({ now, rng: r });
  const mctx = (now: number) => ({ ...ctx(now), instanceId: 'x', mode: 'captain_teams' as const, participants: ids, duelists: ['p1', 'p2'] as [string, string], connected: () => true });
  // each person: the moment they will next press, once they have seen something to press
  const pressAt: Record<string, number> = {};
  const kinds: Record<string, number>[] = [{}, {}];
  let now = NOW;
  for (; now < NOW + 200_000 && !arenaOf(g)!.result; now += 40) {
    for (let i = 0; i < 100 && !arenaOf(g)!.result; i++) {
      const d = [...arenaOf(g)!.deadlines].sort((x, y) => x.at - y.at)[0];
      if (!d || d.at > now) break;
      g = apply(g, arenaDeadline(g, d.id, ctx(d.at)));
    }
    if (arenaOf(g)!.result) break;
    for (const id of ids) {
      const os = mecha.offers(mctx(now), st(g), id) as MechaAction[];
      if (!os.length) { delete pressAt[id]; continue; }
      if (pressAt[id] == null) { pressAt[id] = now + hands[id].react * (0.5 + r()); continue; }
      if (now < pressAt[id]) continue;
      delete pressAt[id];
      let o = os[Math.floor(r() * os.length)];
      // a slip: the wrong way on the balance; a leg stepping where it should jump
      if (r() < hands[id].slip) {
        if (o.kind === 'lean') o = { kind: 'lean', dir: o.dir === 1 ? -1 : 1 };
        else if (o.kind === 'jump') o = { kind: 'step', leg: o.leg };
      }
      const plan = arenaAct(g, id, { action: o }, ctx(now));
      if (!plan.error) g = apply(g, plan);
    }
    // an eager leg: sometimes the leg that has just stepped goes again (a stumble)
    for (const [ti, t] of st(g).teams.entries()) {
      const w = t.walker;
      if (w.move || now < w.busyUntil || w.last < 0 || now - w.landedAt > 60) continue;
      const who = onDuty(t, w.last);
      if (r() < hands[who].slip * 0.5) { const plan = arenaAct(g, who, { action: { kind: 'step', leg: w.last } }, ctx(now)); if (!plan.error) g = apply(g, plan); }
      void ti;
    }
  }
  const res = arenaOf(g)!.result;
  const s = st(g);
  s.teams.forEach((t, i) => { for (const e of t.walker.events) kinds[i][e.kind] = (kinds[i][e.kind] || 0) + 1; });
  const m = (res?.metrics || {}) as { distance?: { A: number; B: number }; slips?: { A: number; B: number } };
  return {
    winner: res?.status === 'resolved' ? (res.duel?.winningDuelistId === 'p1' ? 'A' : 'B') : null, reason: res?.reason || 'never ended', ms: now - NOW,
    finished: res?.reason === 'first over the line', distance: [m.distance?.A ?? 0, m.distance?.B ?? 0], slips: [m.slips?.A ?? 0, m.slips?.B ?? 0], kinds,
  };
}

const team = (size: number, h: Hands): Hands[] => Array.from({ length: size }, () => ({ ...h }));
export const QUICK: Hands = { react: 300, slip: 0.03 }, AVERAGE: Hands = { react: 550, slip: 0.08 }, SLOW: Hands = { react: 900, slip: 0.15 };

export function report(races = 300, config: Record<string, unknown> = {}): string {
  const lines: string[] = ['# Mecha Santa Mayhem: simulated races', ''];
  const run = (label: string, a: Hands[], b: Hands[]) => {
    const out = Array.from({ length: races }, (_, i) => race([a, b], i + 1, config));
    const secs = out.map((o) => o.ms / 1000).sort((x, y) => x - y);
    const pct = (k: number) => `${Math.round((k / races) * 100)}%`;
    const avg = (f: (o: Race) => number) => (out.reduce((s, o) => s + f(o), 0) / races).toFixed(1);
    lines.push(`| ${label} | ${pct(out.filter((o) => o.winner === 'A').length)} | ${pct(out.filter((o) => o.winner === 'B').length)} | ${pct(out.filter((o) => !o.winner).length)} | ${pct(out.filter((o) => o.finished).length)} | ${secs[Math.floor(races / 2)].toFixed(0)} s | ${secs[Math.floor(races * 0.1)].toFixed(0)} to ${secs[Math.floor(races * 0.9)].toFixed(0)} s | ${avg((o) => o.slips[0])} / ${avg((o) => o.slips[1])} | ${avg((o) => Math.abs(o.distance[0] - o.distance[1]))} |`);
  };
  lines.push('| teams (A v B) | A wins | B wins | host decides | over the line | median length | 8 in 10 between | tumbles A / B | gap at the end |', '|---|---|---|---|---|---|---|---|---|');
  run('average 4 v average 4', team(4, AVERAGE), team(4, AVERAGE));
  run('quick 4 v average 4', team(4, QUICK), team(4, AVERAGE));
  run('average 4 v slow 4', team(4, AVERAGE), team(4, SLOW));
  run('slow 4 v slow 4', team(4, SLOW), team(4, SLOW));
  run('quick 4 v quick 4', team(4, QUICK), team(4, QUICK));
  run('average 2 v average 2', team(2, AVERAGE), team(2, AVERAGE));
  run('average 2 v average 3', team(2, AVERAGE), team(3, AVERAGE));
  run('average 4 v average 5', team(4, AVERAGE), team(5, AVERAGE));
  run('average 3 v average 4', team(3, AVERAGE), team(4, AVERAGE));
  lines.push('', 'One weak player (slow) on an otherwise average team of four, by job:', '', '| the slow one is | A (with them) wins | median length |', '|---|---|---|');
  // seat order: the captain is the left leg, then right leg, balance, arms
  ['the left leg', 'the right leg', 'the balance', 'the arms'].forEach((job, k) => {
    const a = team(4, AVERAGE); a[k] = { ...SLOW };
    const out = Array.from({ length: races }, (_, i) => race([a, team(4, AVERAGE)], i + 1, config));
    const secs = out.map((o) => o.ms / 1000).sort((x, y) => x - y);
    lines.push(`| ${job} | ${Math.round((out.filter((o) => o.winner === 'A').length / races) * 100)}% | ${secs[Math.floor(races / 2)].toFixed(0)} s |`);
  });
  return lines.join('\n');
}
