// Pass the Gremlin on the runtime (the spec's test list): who may act, passing rules, fair random
// passes, secrets that stay secret, and results for every mode, incl. the duelist-only sudden death.
import { describe, expect, it } from 'vitest';
import { projectGame } from '../../projection';
import { rng } from '../../rng';
import { apply, NOW, table } from '../../testkit';
import { Tx } from '../../tx';
import type { Game } from '../../types';
import type { MatchMode } from '../contracts';
import { arenaAct, arenaDeadline, arenaOf, startArena } from '../runtime';
import type { GremlinState } from './gremlin';
import { eligible } from './gremlin';

function play(mode: MatchMode, n: number, seed = 1, duelists: [string, string] | undefined = ['p1', 'p2'], challenges = ['quick_math']): Game {
  let g = table(n);
  g.currentTurn = { phase: 'duel', challengerId: 'p1', defenderId: 'p2', pot: 2 };
  const tx = new Tx(g);
  const out = startArena(tx, 'pass_the_gremlin', { mode, participants: Object.keys(g.players!), duelists: mode === 'free_for_all' ? undefined : duelists, config: { challenges } }, { now: NOW, rng: rng(seed) });
  expect(out.error).toBeUndefined();
  g = apply(g, tx.plan());
  return g;
}
const st = (g: Game) => arenaOf(g)!.state as GremlinState;
const act = (g: Game, who: string, action: unknown, now = NOW + 5000, seed = 2) => arenaAct(g, who, { action }, { now, rng: rng(seed) });
/** The holder solves and passes (to someone, or at random). */
function solveAndPass(g: Game, to?: string, now = NOW + 5000): Game {
  const s = st(g);
  g = apply(g, act(g, s.holder, { kind: 'answer', value: s.challenge.answer }, now));
  return apply(g, act(g, s.holder, to ? { kind: 'pass', to } : { kind: 'pass_random' }, now));
}
const doom = (g: Game) => { const d = arenaOf(g)!.deadlines.find((x) => x.id === 'doom')!; return apply(g, arenaDeadline(g, 'doom', { now: d.at, rng: rng(9) })); };

describe('Pass the Gremlin: passing', () => {
  it('only the holder acts; no passing before the problem is solved, or before the minimum hold', () => {
    const g = play('free_for_all', 4);
    const s = st(g);
    const other = s.pool.find((p) => p !== s.holder)!;
    expect(act(g, other, { kind: 'answer', value: 1 }).error).toMatch(/not holding/);
    expect(act(g, s.holder, { kind: 'pass_random' }).error).toMatch(/Solve/);
    const solved = apply(g, act(g, s.holder, { kind: 'answer', value: s.challenge.answer }));
    expect(act(solved, s.holder, { kind: 'pass_random' }, NOW + 200).error).toMatch(/Hold on/);
    expect(act(solved, s.holder, { kind: 'pass_random' }, NOW + 1500).error).toBeUndefined();
  });

  it('a wrong answer gets a new problem after a short cooldown', () => {
    const g = play('free_for_all', 3);
    const s = st(g);
    const after = apply(g, act(g, s.holder, { kind: 'answer', value: (s.challenge.answer as number) + 1 }));
    expect(st(after).stage).toBe('holding');
    expect(act(after, s.holder, { kind: 'answer', value: st(after).challenge.answer }, NOW + 5100).error).toMatch(/next one/);
  });

  it("no passing straight back when anyone else can take it; back is fine when it's the only choice", () => {
    let g = play('free_for_all', 3);
    const first = st(g).holder;
    const next = st(g).pool.find((p) => p !== first)!;
    g = solveAndPass(g, next);
    const s = st(g);
    const later = NOW + 8000;                                         // past the new holder's minimum hold
    g = apply(g, act(g, s.holder, { kind: 'answer', value: s.challenge.answer }, later));
    expect(act(g, s.holder, { kind: 'pass', to: first }, later).error).toMatch(/straight back/);
    let duel = play('direct_duel', 2);
    duel = solveAndPass(duel);
    const d = st(duel);
    duel = apply(duel, act(duel, d.holder, { kind: 'answer', value: d.challenge.answer }, later));
    expect(act(duel, d.holder, { kind: 'pass', to: d.prevHolder! }, later).error).toBeUndefined();   // two players: back it goes
  });

  it('a random pass is fair: uniform over who may take it (seeded)', () => {
    const counts: Record<string, number> = {};
    const g0 = play('free_for_all', 4, 3);
    const s0 = st(g0);
    const solved = apply(g0, act(g0, s0.holder, { kind: 'answer', value: s0.challenge.answer }));
    const can = eligible({ connected: () => true }, st(solved));
    for (let seed = 0; seed < 3000; seed++) {
      const g = apply(solved, act(solved, s0.holder, { kind: 'pass_random' }, NOW + 5000, seed));
      counts[st(g).holder] = (counts[st(g).holder] || 0) + 1;
    }
    expect(Object.keys(counts).sort()).toEqual([...can].sort());
    for (const id of can) expect(counts[id] / 3000).toBeGreaterThan(1 / can.length - 0.04);
  });
});

describe('Pass the Gremlin: secrets', () => {
  it('no view (any phone, the TV, the host) carries the answer or the deadline', () => {
    const g = play('party_with_duelists', 4);
    const s = st(g);
    for (const v of [{ kind: 'tv' }, { kind: 'host' }, ...['p1', 'p2', 'p3', 'p4'].map((pid) => ({ kind: 'phone', pid }))] as Parameters<typeof projectGame>[1][]) {
      const out = JSON.stringify(projectGame(g, v, NOW + 1000));
      expect(out).not.toContain(`"answer"`);
      expect(out).not.toContain(String(s.doomAt));
      expect(out).not.toContain('doomAt');
    }
    const holderView = projectGame(g, { kind: 'phone', pid: s.holder }, NOW).arenaView!.view as { prompt?: string };
    expect(holderView.prompt).toBe(s.challenge.prompt);
    const otherView = projectGame(g, { kind: 'phone', pid: s.pool.find((p) => p !== s.holder)! }, NOW).arenaView!.view as { prompt?: string };
    expect(otherView.prompt).toBeUndefined();
  });

  it('the danger level comes from elapsed time and the public window only', () => {
    const g = play('free_for_all', 3);
    const view = (now: number) => (projectGame(g, { kind: 'tv' }, now).arenaView!.view as { danger: string }).danger;
    expect(view(NOW + 1000)).toBe('calm');
    expect(view(NOW + 45_000)).toBe('uneasy');
    expect(view(NOW + 61_000)).toBe('unstable');
  });

  it('the secret deadline lands inside the configured window', () => {
    for (let seed = 1; seed < 40; seed++) {
      const s = st(play('free_for_all', 3, seed));
      expect(s.doomAt - NOW).toBeGreaterThanOrEqual(60_000);
      expect(s.doomAt - NOW).toBeLessThanOrEqual(100_000);
    }
  });
});

describe('Pass the Gremlin: results', () => {
  it('a two-player duel: the doomed duelist loses, the other wins', () => {
    const g = doom(play('direct_duel', 2));
    const holder = st(g).doomed!;
    const r = arenaOf(g)!.result!;
    expect(r.status).toBe('resolved');
    expect(r.duel).toMatchObject({ losingDuelistId: holder, winningDuelistId: holder === 'p1' ? 'p2' : 'p1' });
  });

  it('a free-for-all: one loser, everyone else survives, no champion invented', () => {
    const g = doom(play('free_for_all', 5));
    const r = arenaOf(g)!.result!;
    expect(r).toMatchObject({ status: 'resolved', winners: [], losers: [st(g).doomed] });
    expect(r.duel).toBeUndefined();
    expect((r.metrics!.survivors as string[]).length).toBe(4);
  });

  it('party with duelists: a doomed duelist loses the duel to the other', () => {
    for (let seed = 1; seed < 60; seed++) {
      const g0 = play('party_with_duelists', 5, seed);
      if (!['p1', 'p2'].includes(st(g0).holder)) continue;
      const g = doom(g0);
      const r = arenaOf(g)!.result!;
      expect(r.status).toBe('resolved');
      expect(r.duel!.winningDuelistId).toBe(st(g0).holder === 'p1' ? 'p2' : 'p1');
      return;
    }
    throw new Error('no seed started with a duelist holding it');
  });

  it('party with duelists: a doomed bystander starts a duelist-only sudden death that always ends on one of them', () => {
    let tested = 0;
    for (let seed = 1; seed < 80 && tested < 5; seed++) {
      let g = play('party_with_duelists', 5, seed);
      if (['p1', 'p2'].includes(st(g).holder)) continue;
      const bystander = st(g).holder;
      g = doom(g);
      expect(arenaOf(g)!.result).toBeUndefined();                       // not over: no duel winner from two survivors
      expect(st(g)).toMatchObject({ stage: 'sudden_death_intro', bystanderDoomed: bystander });
      const intro = arenaOf(g)!.deadlines.find((d) => d.id === 'sudden_death')!;
      g = apply(g, arenaDeadline(g, 'sudden_death', { now: intro.at, rng: rng(seed) }));
      expect(st(g)).toMatchObject({ round: 'sudden_death', stage: 'holding' });
      expect(st(g).pool.sort()).toEqual(['p1', 'p2']);
      expect(st(g).doomAt).not.toBe(intro.at);
      expect(act(g, bystander, { kind: 'answer', value: 0 }, intro.at + 10).error).toBeTruthy();   // bystanders are out
      g = doom(g);
      const r = arenaOf(g)!.result!;
      expect(r.status).toBe('resolved');
      expect(['p1', 'p2']).toContain(r.duel!.winningDuelistId);
      expect(r.duel!.losingDuelistId).toBe(st(g).doomed);
      tested++;
    }
    expect(tested).toBeGreaterThan(0);
  });

  it('the same seed gives the same game', () => {
    const run = () => { let g = play('party_with_duelists', 4, 42); g = solveAndPass(g); return JSON.stringify(arenaOf(g)); };
    expect(run()).toBe(run());
  });
});
