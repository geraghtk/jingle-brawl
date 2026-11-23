// Pass the Gremlin on the runtime (the spec's test list): who may act, passing rules, fair random
// passes, secrets that stay secret, and results for every mode, incl. the duelist-only sudden death.
import { describe, expect, it } from 'vitest';
import { projectGame } from '../../projection';
import { rng } from '../../rng';
import { apply, NOW, table } from '../../testkit';
import { Tx } from '../../tx';
import type { Game } from '../../types';
import type { MatchMode } from '../contracts';
import { arenaAct, arenaDeadline, arenaOf, handoff, startArena } from '../runtime';
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
/** Put the Gremlin in someone's hands (a test's shortcut for a run of passes). */
const hold = (g: Game, pid: string): Game => { const c = structuredClone(g); (arenaOf(c)!.state as GremlinState).holder = pid; return c; };
const next = (g: Game, id: string) => { const d = arenaOf(g)!.deadlines.find((x) => x.id === id)!; return apply(g, arenaDeadline(g, id, { now: d.at, rng: rng(5) })); };
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

  it('the big screen can show the table: how long the holder has had it, a wrong answer, who has held it how often', () => {
    let g = play('free_for_all', 4);
    const s = st(g), first = s.holder;
    const tv = (x: Game, now = NOW + 5000) => projectGame(x, { kind: 'tv' }, now).arenaView!.view as { holdSince?: number; misses?: number; held?: Record<string, number> };
    expect(tv(g).holdSince).toBe(NOW);
    expect(tv(g).misses ?? 0).toBe(0);
    expect(tv(g).held).toEqual({ [first]: 1 });
    g = apply(g, act(g, first, { kind: 'answer', value: Number(s.challenge.answer) + 1 }));
    expect(tv(g).misses).toBe(1);
    g = solveAndPass(g, undefined, NOW + 8000);
    const next = st(g).holder;
    expect(tv(g, NOW + 9000).holdSince).toBe(NOW + 8000);
    expect(tv(g, NOW + 9000).misses ?? 0).toBe(0);
    expect(tv(g, NOW + 9000).held).toEqual({ [first]: 1, [next]: 1 });
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

  it('party with duelists: a caught bystander is out and the rest play on, on a shorter fuse', () => {
    let g = hold(play('party_with_duelists', 5), 'p3');
    g = doom(g);
    expect(arenaOf(g)!.result).toBeUndefined();
    expect(st(g)).toMatchObject({ stage: 'out_intro', bystanderDoomed: 'p3', out: ['p3'] });
    expect(act(g, 'p3', { kind: 'answer', value: 0 }).error).toBeTruthy();   // nobody moves during the beat
    g = next(g, 'next_round');
    expect(st(g)).toMatchObject({ round: 'main', roundNo: 2, stage: 'holding' });
    expect([...st(g).pool].sort()).toEqual(['p1', 'p2', 'p4', 'p5']);
    expect(st(g).window).toEqual([35_000, 55_000]);
    const fuse = st(g).doomAt - st(g).roundStartedAt;
    expect(fuse).toBeGreaterThanOrEqual(35_000);
    expect(fuse).toBeLessThanOrEqual(55_000);
    // the one who is out can neither hold it nor be passed it
    const s = st(g);
    const solved = apply(g, act(g, s.holder, { kind: 'answer', value: s.challenge.answer }, s.roundStartedAt + 5000));
    expect(act(solved, s.holder, { kind: 'pass', to: 'p3' }, s.roundStartedAt + 5000).error).toMatch(/can't pass/);
    const tv = projectGame(g, { kind: 'tv' }, s.roundStartedAt + 100).arenaView!.view as { out?: string[]; roundNo?: number; maxOut?: number };
    expect(tv).toMatchObject({ out: ['p3'], roundNo: 2, maxOut: 2 });
  });

  it('party with duelists: after two bystanders (or the last one) it is the duelists alone, and it always ends on one of them', () => {
    let g = doom(hold(play('party_with_duelists', 6), 'p3'));
    g = doom(hold(next(g, 'next_round'), 'p5'));
    expect(st(g)).toMatchObject({ stage: 'sudden_death_intro', out: ['p3', 'p5'] });   // p4 and p6 are still in, but two is the cap
    const intro = arenaOf(g)!.deadlines.find((d) => d.id === 'sudden_death')!;
    g = next(g, 'sudden_death');
    expect(st(g)).toMatchObject({ round: 'sudden_death', stage: 'holding' });
    expect([...st(g).pool].sort()).toEqual(['p1', 'p2']);
    expect(st(g).doomAt).not.toBe(intro.at);
    expect(act(g, 'p4', { kind: 'answer', value: 0 }, intro.at + 10).error).toBeTruthy();   // bystanders are out
    g = doom(g);
    const r = arenaOf(g)!.result!;
    expect(r.status).toBe('resolved');
    expect(['p1', 'p2']).toContain(r.duel!.winningDuelistId);
    expect(r.duel!.losingDuelistId).toBe(st(g).doomed);
    // a table with one bystander: caught, and there is nobody left to play on with
    const three = doom(hold(play('party_with_duelists', 3), 'p3'));
    expect(st(three)).toMatchObject({ stage: 'sudden_death_intro', out: ['p3'] });
  });

  it('party with duelists: a duelist caught in a later round loses the duel there and then', () => {
    let g = next(doom(hold(play('party_with_duelists', 5), 'p4')), 'next_round');
    g = doom(hold(g, 'p2'));
    expect(arenaOf(g)!.result).toMatchObject({ status: 'resolved', duel: { winningDuelistId: 'p1', losingDuelistId: 'p2' } });
  });

  it('bystanders earn a Party point for never being caught; a caught one and the duelists get none', () => {
    let g = doom(hold(play('party_with_duelists', 6), 'p3'));
    g = doom(hold(next(g, 'next_round'), 'p5'));
    g = doom(hold(next(g, 'sudden_death'), 'p1'));
    const points = (x: Game) => Object.fromEntries((arenaOf(x)!.result!.rewards || []).map((r) => [r.playerId, r.points]));
    expect(points(g)).toEqual({ p4: 1, p6: 1 });
    // nobody caught but a duelist: every bystander lasted; one whose phone has gone gets nothing for it
    let quick = hold(play('party_with_duelists', 5), 'p2');
    quick.players!.p5.connected = false;
    quick = doom(quick);
    expect(points(quick)).toEqual({ p3: 1, p4: 1 });
    // a two-player duel and a free-for-all hand out nothing
    expect(arenaOf(doom(play('direct_duel', 2)))!.result!.rewards).toBeUndefined();
    expect(arenaOf(doom(play('free_for_all', 4)))!.result!.rewards).toBeUndefined();
  });

  it('the hand-off pays those points once, and says who got what', () => {
    let g = hold(play('party_with_duelists', 4), 'p1');
    g.currentTurn = { ...g.currentTurn!, phase: 'minigame_arena' };
    g = doom(g);
    const before = { ...g.players!.p3 };
    const plan = handoff(g);
    const after = apply(g, plan);
    expect(after.players!.p3.partyPoints).toBe((before.partyPoints || 0) + 1);
    expect(after.players!.p1.partyPoints || 0).toBe(g.players!.p1.partyPoints || 0);
    expect(JSON.stringify(plan)).toContain('Party points: ');
    expect(JSON.stringify(plan)).toContain(`${g.players!.p3.name} +1`);
  });

  it('the same seed gives the same game', () => {
    const run = () => { let g = play('party_with_duelists', 4, 42); g = solveAndPass(g); return JSON.stringify(arenaOf(g)); };
    expect(run()).toBe(run());
  });
});
