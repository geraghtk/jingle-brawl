// Pass the Gremlin, thousands of rounds through the real engine (the spec's bot simulation):
// people with different speeds and accuracy hold, solve and pass it; the server's deadline dooms
// someone. Measures who ends up doomed (is it fair by skill?), how often it's passed, how often a
// problem is solved first time, stalls, and how often a party round needs the sudden death.
//   npm run analyze:gremlin [-- --rounds 5000 --players 6]   (gremlin-report.ts)
import { rng as makeRng } from '../core/rng';
import type { Rng } from '../core/rng';
import { apply, NOW, table } from '../core/testkit';
import { Tx } from '../core/tx';
import type { Game } from '../core/types';
import type { MatchMode } from '../core/minigames/contracts';
import type { GremlinState } from '../core/minigames/gremlin/gremlin';
import { CHALLENGES } from '../core/minigames/gremlin/challenges';
import { arenaAct, arenaDeadline, arenaOf, startArena } from '../core/minigames/runtime';

export interface Hands { speed: number; accuracy: number }   // speed: 0 slow … 1 fast
export interface RoundStats { loser: string; passes: number; solves: number; misses: number; suddenDeath: boolean; ms: number; stalled: boolean; winner?: string }

const st = (g: Game) => arenaOf(g)!.state as GremlinState;

/** One round on a virtual clock. Each holder thinks (by their speed), answers (right by their accuracy), then passes. */
export function round(mode: MatchMode, hands: Hands[], seed: number): RoundStats {
  const r: Rng = makeRng(seed);
  let g = table(hands.length);
  g.currentTurn = { phase: 'duel', challengerId: 'p1', defenderId: 'p2', pot: 2 };
  const tx = new Tx(g);
  const started = startArena(tx, 'pass_the_gremlin', { mode, participants: Object.keys(g.players!), duelists: mode === 'free_for_all' ? undefined : ['p1', 'p2'] }, { now: NOW, rng: r });
  if (started.error) throw new Error(started.error);
  g = apply(g, tx.plan());
  let now = NOW, passes = 0, solves = 0, misses = 0, suddenDeath = false, idle = 0;
  const who = (pid: string) => hands[Number(pid.slice(1)) - 1];
  for (let guard = 0; guard < 5000 && !arenaOf(g)!.result; guard++) {
    const a = arenaOf(g)!;
    const doom = a.deadlines[0];
    const s = st(g);
    if (s.stage === 'sudden_death_intro' || s.stage === 'doomed') { now = doom.at; g = apply(g, arenaDeadline(g, doom.id, { now, rng: r })); suddenDeath = true; continue; }
    const h = who(s.holder);
    const think = 1200 + (1 - h.speed) * 4000 + r() * 1500;              // a few seconds to solve
    const next = Math.max(now + think, s.cooldownUntil || 0, s.stage === 'pass_ready' ? s.holdSince + s.cfg.minHoldMs : 0);
    if (next >= doom.at) { now = doom.at; g = apply(g, arenaDeadline(g, doom.id, { now, rng: r })); continue; }
    now = next;
    const kind = CHALLENGES[s.challenge.kind];
    const right = s.stage === 'holding' ? kind.solve(s.challenge, now) : null;
    if (s.stage === 'holding' && right === null) { now += 50; continue; }   // perfect_timing: wait for the marker to come round
    const hit = r() < h.accuracy;
    const action = s.stage === 'holding'
      ? { kind: 'answer', value: hit ? right : kind.miss(s.challenge, now) }
      : r() < 0.5 ? { kind: 'pass_random' } : { kind: 'pass', to: s.pool.filter((p) => p !== s.holder && p !== s.prevHolder)[0] ?? s.pool.find((p) => p !== s.holder) };
    const plan = arenaAct(g, s.holder, { action }, { now, rng: r });
    if (plan.error) { idle++; now += 500; continue; }
    if (action.kind === 'answer') { if (hit) solves++; else misses++; }
    else passes++;
    g = apply(g, plan);
  }
  const res = arenaOf(g)!.result;
  return { loser: res?.losers[0] ?? '?', passes, solves, misses, suddenDeath, ms: now - NOW, stalled: !res || idle > 50, winner: res?.duel?.winningDuelistId };
}

export function report(rounds = 3000, players = 6): string {
  const hands: Hands[] = Array.from({ length: players }, (_, i) => ({ speed: i / Math.max(1, players - 1), accuracy: 0.7 + 0.25 * (i / Math.max(1, players - 1)) }));
  const lines: string[] = ['# Pass the Gremlin: simulated rounds', ''];
  for (const mode of ['free_for_all', 'party_with_duelists', 'direct_duel'] as MatchMode[]) {
    const n = mode === 'direct_duel' ? 2 : players;
    const out = Array.from({ length: rounds }, (_, i) => round(mode, hands.slice(0, n), i + 1));
    const doomed: Record<string, number> = {};
    for (const o of out) doomed[o.loser] = (doomed[o.loser] || 0) + 1;
    const avg = (f: (o: RoundStats) => number) => (out.reduce((s, o) => s + f(o), 0) / out.length).toFixed(1);
    lines.push(`## ${mode.replace(/_/g, ' ')} (${n} players, ${rounds} rounds)`, '');
    lines.push('| player | speed | accuracy | doomed |', '|---|---|---|---|');
    for (let i = 1; i <= n; i++) lines.push(`| p${i} | ${hands[i - 1].speed.toFixed(2)} | ${hands[i - 1].accuracy.toFixed(2)} | ${(((doomed[`p${i}`] || 0) / rounds) * 100).toFixed(1)}% |`);
    lines.push('', `passes per round ${avg((o) => o.passes)} · problems solved first time ${((out.reduce((s, o) => s + o.solves, 0) / Math.max(1, out.reduce((s, o) => s + o.solves + o.misses, 0))) * 100).toFixed(0)}% · round ${avg((o) => o.ms / 1000)} s · stalls ${out.filter((o) => o.stalled).length}`);
    if (mode === 'party_with_duelists') lines.push(`sudden death ${((out.filter((o) => o.suddenDeath).length / rounds) * 100).toFixed(0)}% · a duelist always won: ${out.every((o) => o.winner === 'p1' || o.winner === 'p2')}`);
    lines.push('');
  }
  return lines.join('\n');
}

