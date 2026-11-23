// Elf Tower Wars, hundreds of battles through the real engine: people with different reaction times do
// their bit through the game's own offers (all-rounders, or hands with a habit: toys, towers, the workshop). Measures how long a battle takes, how often a workshop
// falls before the bell, whether the quicker team wins, and whether attack or defence is the better buy.
//   npm run analyze:towers [-- --battles 200]   (towers-report.ts)
import { rng as makeRng } from '../core/rng';
import type { Rng } from '../core/rng';
import { apply, NOW, table } from '../core/testkit';
import { Tx } from '../core/tx';
import type { Game } from '../core/types';
import { assignTeams } from '../core/minigames/engines/teams';
import { arenaAct, arenaDeadline, arenaOf, startArena } from '../core/minigames/runtime';
import { handOffers } from '../core/minigames/towers/towers';
import type { Leaning, TowerAction, TowerState } from '../core/minigames/towers/towers';

/** A person: how long they take to press once there is something to press (ms); `lean`: what they mostly spend their energy on (else their place on the team picks). */
export interface Hands { react: number; lean?: Leaning }
export interface Battle { winner: 'A' | 'B' | null; reason: string; ms: number; fell: boolean; health: [number, number]; sent: [number, number]; towers: [number, number] }

const st = (g: Game) => arenaOf(g)!.state as TowerState;

export function battle(teams: [Hands[], Hands[]], seed: number, config: Record<string, unknown> = {}): Battle {
  const r: Rng = makeRng(seed);
  const n = teams[0].length + teams[1].length;
  let g = table(n);
  g.currentTurn = { phase: 'duel', challengerId: 'p1', defenderId: 'p2', pot: 2 };
  const ids = Object.keys(g.players!), rest = ids.slice(2);
  const a = ['p1', ...rest.slice(0, teams[0].length - 1)], b = ['p2', ...rest.slice(teams[0].length - 1)];
  const deal = { ...assignTeams(['p1', 'p2'], ids, makeRng(1)), teams: [{ teamId: 'A', captainId: 'p1', memberIds: a }, { teamId: 'B', captainId: 'p2', memberIds: b }] };
  const hands: Record<string, Hands> = {};
  a.forEach((id, i) => { hands[id] = teams[0][i]; });
  b.forEach((id, i) => { hands[id] = teams[1][i]; });
  const tx = new Tx(g);
  const started = startArena(tx, 'elf_tower_wars', { mode: 'captain_teams', participants: ids, duelists: ['p1', 'p2'], config: { teams: deal, ...config } }, { now: NOW, rng: r });
  if (started.error) throw new Error(started.error);
  g = apply(g, tx.plan());
  const ctx = (now: number) => ({ now, rng: r });
  const mctx = (now: number) => ({ ...ctx(now), instanceId: 'x', mode: 'captain_teams' as const, participants: ids, duelists: ['p1', 'p2'] as [string, string], connected: () => true });
  const pressAt: Record<string, number> = {};
  let now = NOW;
  for (; now < NOW + 300_000 && !arenaOf(g)!.result; now += 100) {
    for (let i = 0; i < 50 && !arenaOf(g)!.result; i++) {
      const d = [...arenaOf(g)!.deadlines].sort((x, y) => x.at - y.at)[0];
      if (!d || d.at > now) break;
      g = apply(g, arenaDeadline(g, d.id, ctx(d.at)));
    }
    if (arenaOf(g)!.result) break;
    for (const id of ids) {
      const os: TowerAction[] = handOffers(mctx(now), st(g), id, hands[id].lean);
      if (!os.length) { delete pressAt[id]; continue; }
      if (pressAt[id] == null) { pressAt[id] = now + hands[id].react * (0.5 + r()); continue; }
      if (now < pressAt[id]) continue;
      delete pressAt[id];
      const plan = arenaAct(g, id, { action: os[Math.floor(r() * os.length)] }, ctx(now));
      if (!plan.error) g = apply(g, plan);
    }
  }
  const res = arenaOf(g)!.result, f = st(g).field;
  return {
    winner: res?.status === 'resolved' ? (res.duel?.winningDuelistId === 'p1' ? 'A' : 'B') : null, reason: res?.reason || 'never ended', ms: now - NOW, fell: res?.reason === 'the other workshop fell',
    health: [f.sides[0].hp, f.sides[1].hp], sent: [f.sides[0].sent, f.sides[1].sent], towers: [f.sides[0].towers.length, f.sides[1].towers.length],
  };
}

const team = (size: number, h: Hands): Hands[] => Array.from({ length: size }, () => ({ ...h }));
export const QUICK: Hands = { react: 350 }, AVERAGE: Hands = { react: 700 }, SLOW: Hands = { react: 1300 };

export function report(battles = 200, config: Record<string, unknown> = {}): string {
  const lines: string[] = ['# Elf Tower Wars: simulated battles', '',
    '| teams (A v B) | A wins | B wins | host decides | a workshop fell | median length | health left A / B | toys sent A / B | towers A / B |', '|---|---|---|---|---|---|---|---|---|'];
  const run = (label: string, a: Hands[], b: Hands[]) => {
    const out = Array.from({ length: battles }, (_, i) => battle([a, b], i + 1, config));
    const secs = out.map((o) => o.ms / 1000).sort((x, y) => x - y);
    const pct = (k: number) => `${Math.round((k / battles) * 100)}%`;
    const avg = (f: (o: Battle) => number) => (out.reduce((s, o) => s + f(o), 0) / battles).toFixed(1);
    lines.push(`| ${label} | ${pct(out.filter((o) => o.winner === 'A').length)} | ${pct(out.filter((o) => o.winner === 'B').length)} | ${pct(out.filter((o) => !o.winner).length)} | ${pct(out.filter((o) => o.fell).length)} | ${secs[Math.floor(battles / 2)].toFixed(0)} s | ${avg((o) => o.health[0])} / ${avg((o) => o.health[1])} | ${avg((o) => o.sent[0])} / ${avg((o) => o.sent[1])} | ${avg((o) => o.towers[0])} / ${avg((o) => o.towers[1])} |`);
  };
  run('average 3 v average 3', team(3, AVERAGE), team(3, AVERAGE));
  run('quick 3 v average 3', team(3, QUICK), team(3, AVERAGE));
  run('average 3 v slow 3', team(3, AVERAGE), team(3, SLOW));
  run('average 2 v average 2', team(2, AVERAGE), team(2, AVERAGE));
  // (uneven teams earn the same energy in all; the simulated hands split saving and sending by head count, so
  // these two rows show a different mix of habits as much as a different size)
  run('average 2 v average 3', team(2, AVERAGE), team(3, AVERAGE));
  run('average 3 v average 4', team(3, AVERAGE), team(4, AVERAGE));
  // habits: everyone can do everything, so is one way of spending the energy simply the best?
  const of = (...leans: Leaning[]) => leans.map((lean) => ({ ...AVERAGE, lean }));
  const mixed = of('send', 'build', 'defend');
  run('A all toys (a rush) v a mixed team', of('send', 'send', 'send'), mixed);
  run('A no toys (towers and defence) v mixed', of('build', 'defend', 'defend'), mixed);
  run('A toys and towers, no defender v mixed', of('send', 'build', 'send'), mixed);
  run('A toys and defence, no builder v mixed', of('send', 'defend', 'send'), mixed);
  run('A two builders v mixed', of('send', 'build', 'build'), mixed);
  run('2 (toys, towers) v 3 mixed', of('send', 'build'), mixed);
  run('4 (toys, towers, defence, toys) v 3 mixed', of('send', 'build', 'defend', 'send'), mixed);
  return lines.join('\n');
}
