// Reindeer Ring Toss on the runtime: practice doesn't score, the server decides every landing, every
// target is 1 and a full set +5 (again and again), rings come in batches with a reload until the clock
// runs out, a resent throw counts once, the duelists are compared (not the room), ties go to extra rings
// and then the closest ring, and the same engines run a different game unchanged.
import { beforeAll, describe, expect, it } from 'vitest';
import { startDuelApp } from '../../duelgame';
import { projectGame } from '../../projection';
import { rng } from '../../rng';
import { apply, NOW, table } from '../../testkit';
import { Tx } from '../../tx';
import type { Game } from '../../types';
import type { MatchMode } from '../contracts';
import { pathFor } from '../engines/gesture';
import type { Launch } from '../engines/gesture';
import { DEFAULT_FLIGHT } from '../engines/projectile';
import { getDef, register } from '../registry';
import { arenaAct, arenaDeadline, arenaOf, startArena } from '../runtime';
import { snowballs } from '../testgames/snowballs';
import { PEGS } from './ringtoss';
import type { RingState } from './ringtoss';

const st = (g: Game) => arenaOf(g)!.state as RingState;
const ctx = (now: number, seed = 3) => ({ now, rng: rng(seed) });

/** The launch that lands dead on a peg ('miss': well short of everything). */
function aimAt(pegId: string): Launch {
  if (pegId === 'miss') return { angle: 0, power: 0.1 };
  const p = PEGS.find((x) => x.id === pegId)!;
  const range = Math.hypot(p.x, p.z);
  return { angle: Math.atan2(p.x, p.z), power: (range - DEFAULT_FLIGHT.baseRange) / DEFAULT_FLIGHT.rangePerPower };
}
const toss = (pegId: string) => ({ kind: 'throw', path: pathFor(aimAt(pegId)) });
const ALL = ['left-outer', 'left-inner', 'golden', 'right-inner', 'right-outer'];

function play(mode: MatchMode, n = 3, config: Record<string, unknown> = {}): Game {
  let g = table(n);
  g.currentTurn = { phase: 'duel', challengerId: 'p1', defenderId: 'p2', pot: 2 };
  const tx = new Tx(g);
  const out = startArena(tx, 'reindeer_ring_toss', { mode, participants: Object.keys(g.players!), duelists: mode === 'free_for_all' ? undefined : ['p1', 'p2'], config }, ctx(NOW));
  expect(out.error).toBeUndefined();
  return apply(g, tx.plan());
}
/** Throw for `who` at `now`; returns the new game (fails the test if refused). */
function thrown(g: Game, who: string, pegId: string, now: number, aid?: string): Game {
  const plan = arenaAct(g, who, { action: toss(pegId), aid }, ctx(now));
  expect(plan.error, `${who} → ${pegId}`).toBeUndefined();
  return apply(g, plan);
}
const due = (g: Game, now?: number) => {
  const d = arenaOf(g)!.deadlines.find((x) => x.id === 'stage')!;
  return apply(g, arenaDeadline(g, 'stage', ctx(now ?? d.at)));
};

describe('Reindeer Ring Toss', () => {
  it('practice first: throws land and show, nothing scores; then everyone gets their rings', () => {
    let g = play('party_with_duelists', 3, { practiceThrows: 1, batch: 2 });
    expect(st(g).stage).toBe('practice');
    g = thrown(g, 'p1', 'golden', NOW + 100);
    expect(st(g).throws[0]).toMatchObject({ pid: 'p1', pegId: 'golden', points: 0, stage: 'practice' });
    expect(st(g).scores.p1).toBe(0);
    g = due(g);
    expect(st(g)).toMatchObject({ stage: 'throwing', left: { p1: 2, p2: 2, p3: 2 } });
  });

  it("the server decides where a ring lands from the swipe: a client's own score or hit is ignored", () => {
    let g = play('direct_duel', 2, { practiceThrows: 0, batch: 2 });
    const plan = arenaAct(g, 'p1', { action: { ...toss('miss'), points: 5, pegId: 'golden', x: 0, z: 1.78 } }, ctx(NOW + 100));
    g = apply(g, plan);
    expect(st(g).throws[0]).toMatchObject({ pegId: null, points: 0 });
    expect(arenaAct(g, 'p2', { action: { kind: 'throw', path: [[0.5, 0.2, 0], [0.5, 0.9, 100]] } }, ctx(NOW + 100)).error).toMatch(/Flick up/);
    expect(arenaAct(g, 'p2', { action: { kind: 'throw', score: 99 } }, ctx(NOW + 100)).error).toBeTruthy();
  });

  it('every target is 1 (the hat too); a quick second ring waits; a miss is 0', () => {
    let g = play('direct_duel', 2, { practiceThrows: 0, batch: 5 });
    g = thrown(g, 'p1', 'left-outer', NOW + 100);
    expect(arenaAct(g, 'p1', { action: toss('golden') }, ctx(NOW + 300)).error).toMatch(/Next ring/);
    g = thrown(g, 'p1', 'right-inner', NOW + 800);
    g = thrown(g, 'p1', 'golden', NOW + 1500);
    g = thrown(g, 'p1', 'miss', NOW + 2200);
    expect(st(g).scores.p1).toBe(3);
    expect(st(g).throws.map((t) => t.points)).toEqual([1, 1, 1, 0]);
  });

  it('a full set: a ring on all five since your last set is +5, and you can make another', () => {
    let g = play('direct_duel', 2, { practiceThrows: 0, batch: 20 });
    let at = NOW + 100;
    // the same target twice counts 1 each time but only once toward the set
    for (const peg of ['left-outer', 'left-outer', 'left-inner', 'golden', 'right-inner']) { g = thrown(g, 'p1', peg, at); at += 700; }
    expect(st(g).scores.p1).toBe(5);
    expect(st(g).got.p1).toHaveLength(4);
    g = thrown(g, 'p1', 'right-outer', at); at += 700;                // the last piece
    expect(st(g).scores.p1).toBe(6 + 5);
    expect(st(g).throws.at(-1)).toMatchObject({ pegId: 'right-outer', points: 1, bonus: 5 });
    expect(st(g).sets.p1).toBe(1);
    expect(st(g).got.p1 || []).toEqual([]);                         // a new set starts
    for (const peg of ALL) { g = thrown(g, 'p1', peg, at); at += 700; }
    expect(st(g).sets.p1).toBe(2);
    expect(st(g).scores.p1).toBe(11 + 5 + 5);
  });

  it("practice rings don't count toward a set", () => {
    let g = play('direct_duel', 2, { practiceThrows: 5, batch: 5 });
    let at = NOW + 100;
    for (const peg of ALL) { g = thrown(g, 'p1', peg, at); at += 700; }
    expect((st(g).sets || {}).p1 || 0).toBe(0);                     // (stored state drops an empty map)
    expect(st(g).throws.every((t) => !t.bonus)).toBe(true);
  });

  it('the scored round: rings in batches, a reload after each, as many as the clock allows', () => {
    let g = play('direct_duel', 2, { practiceThrows: 0, batch: 2, reloadMs: 3000, roundMs: 40_000 });
    g = thrown(g, 'p1', 'golden', NOW + 100);
    g = thrown(g, 'p1', 'miss', NOW + 800);                           // the batch is used up at NOW + 800
    expect(arenaAct(g, 'p1', { action: toss('golden') }, ctx(NOW + 2000)).error).toMatch(/Reloading/);
    const phone = (now: number) => projectGame(g, { kind: 'phone', pid: 'p1' }, now).arenaView!.view as { myLeft: number; reloadAt: Record<string, number> };
    expect(phone(NOW + 2000)).toMatchObject({ myLeft: 0, reloadAt: { p1: NOW + 3800 } });
    expect(phone(NOW + 3800).myLeft).toBe(2);                        // back in hand, no new move needed
    g = thrown(g, 'p1', 'golden', NOW + 3800);
    g = thrown(g, 'p1', 'golden', NOW + 4500);
    expect(st(g).scores.p1).toBe(3);
    // everyone empty-handed doesn't end the round: only the clock does
    g = thrown(g, 'p2', 'miss', NOW + 100);
    g = thrown(g, 'p2', 'miss', NOW + 800);
    expect(st(g).stage).toBe('throwing');
    expect(arenaOf(g)!.deadlines).toEqual([{ id: 'stage', at: NOW + 40_000 }]);
    g = due(g);
    expect(arenaOf(g)!.result).toMatchObject({ status: 'resolved', duel: { winningDuelistId: 'p1' } });
  });

  it('a resent throw (same action id, after a reconnect) counts once', () => {
    let g = play('direct_duel', 2, { practiceThrows: 0, batch: 3 });
    g = thrown(g, 'p1', 'golden', NOW + 100, 'a1');
    expect(arenaAct(g, 'p1', { action: toss('golden'), aid: 'a1' }, ctx(NOW + 900)).error).toMatch(/Already done/);
    expect(st(g).scores.p1).toBe(1);
    expect(st(g).throws).toHaveLength(1);
  });

  it('throws at the same moment from different phones all count', () => {
    let g = play('party_with_duelists', 3, { practiceThrows: 0, batch: 2 });
    for (const p of ['p1', 'p2', 'p3']) g = thrown(g, p, 'golden', NOW + 100);
    expect(Object.values(st(g).scores)).toEqual([1, 1, 1]);
  });

  it('the duelists are compared, not the room: a bystander topping the table changes nothing', () => {
    let g = play('party_with_duelists', 3, { practiceThrows: 0, batch: 1 });
    g = thrown(g, 'p3', 'golden', NOW + 100);
    g = thrown(g, 'p3', 'left-outer', NOW + 3200);   // after its reload
    g = thrown(g, 'p1', 'left-inner', NOW + 100);
    g = thrown(g, 'p2', 'miss', NOW + 100);
    g = due(g);                                               // time's up
    const r = arenaOf(g)!.result!;
    expect(r).toMatchObject({ status: 'resolved', duel: { winningDuelistId: 'p1', losingDuelistId: 'p2' } });
    expect(r.placements!.find((p) => p.playerId === 'p3')!.rank).toBe(1);   // the table's ranking is there, for show
  });

  it('level: one more ring each for the duelists only; still level: the ring closest to the golden peg', () => {
    let g = play('party_with_duelists', 3, { practiceThrows: 0, batch: 1, tiebreakRounds: 1 });
    g = thrown(g, 'p1', 'left-inner', NOW + 100);   // 1 each: level
    g = thrown(g, 'p2', 'right-inner', NOW + 100);
    g = thrown(g, 'p3', 'miss', NOW + 100);
    g = due(g);
    expect(st(g)).toMatchObject({ stage: 'tiebreak', pool: ['p1', 'p2'], tiebreakRound: 1 });
    expect(arenaAct(g, 'p3', { action: toss('golden') }, ctx(NOW + 5000)).error).toMatch(/Just the duelists/);
    // both miss the extra ring: p1's earlier ring (left inner) and p2's (right inner) are as close as each other…
    // so p2 throws a ring that lands nearer the golden peg without scoring
    const near = { kind: 'throw', path: pathFor({ angle: 0.03, power: aimAt('golden').power + 0.12 }) };
    g = apply(g, arenaAct(g, 'p2', { action: near }, ctx(NOW + 5000)));
    g = thrown(g, 'p1', 'miss', NOW + 5000);
    expect(st(g).scores.p1).toBe(st(g).scores.p2);
    g = due(g);
    expect(arenaOf(g)!.result).toMatchObject({ status: 'resolved', duel: { winningDuelistId: 'p2', resolutionReason: 'closest ring to the golden peg' } });
  });

  it('nobody threw at all: no winner is made up, the host decides', () => {
    let g = play('direct_duel', 2, { practiceThrows: 0, batch: 1, tiebreakRounds: 1 });
    g = due(g);
    g = due(g);
    expect(arenaOf(g)!.result).toMatchObject({ status: 'requires_tiebreaker' });
  });

  it('a free-for-all: placements by score, ties share a place', () => {
    let g = play('free_for_all', 3, { practiceThrows: 0, batch: 1 });
    g = thrown(g, 'p1', 'golden', NOW + 100);
    g = thrown(g, 'p2', 'left-outer', NOW + 100);
    g = thrown(g, 'p3', 'miss', NOW + 100);
    g = due(g);
    const r = arenaOf(g)!.result!;
    expect(r.winners.sort()).toEqual(['p1', 'p2']);
    expect(r.placements!.find((p) => p.playerId === 'p3')!.rank).toBe(3);
  });

  it('every screen sees the same throws (nothing is secret); a phone also gets its own rings left', () => {
    let g = play('direct_duel', 2, { practiceThrows: 0 });
    g = thrown(g, 'p1', 'golden', NOW + 100);
    const tv = projectGame(g, { kind: 'tv' }, NOW + 200).arenaView!.view as { throws: unknown[]; scores: Record<string, number> };
    const phone = projectGame(g, { kind: 'phone', pid: 'p2' }, NOW + 200).arenaView!.view as { throws: unknown[]; myLeft: number };
    expect(phone.throws).toEqual(tv.throws);
    expect(tv.scores.p1).toBe(1);
    expect(phone.myLeft).toBe(5);
  });

  it('starts as a duel game from the setup screen: a party with three at the table, a direct duel with two', () => {
    const duel = (n: number) => {
      const g = table(n, { duelPick: 'off', duelAppGames: ['reindeer_ring_toss'], voteWindow: 0 });
      g.currentTurn = { phase: 'duel', challengerId: 'p2', defenderId: 'p1', pot: 2, duelGame: { name: 'Reindeer Ring Toss', kind: 'app', app: 'reindeer_ring_toss', via: 'picker' } };
      return arenaOf(apply(g, startDuelApp(g, ctx(NOW))))!;
    };
    expect(duel(3).mode).toBe('party_with_duelists');
    expect(duel(2).mode).toBe('direct_duel');
  });

  it('the simulated hands throw through the same checks', () => {
    let g = play('direct_duel', 2, { practiceThrows: 0, batch: 2 });
    const mctx = { instanceId: 'x', mode: 'direct_duel' as const, participants: ['p1', 'p2'], duelists: ['p1', 'p2'] as [string, string], connected: () => true, now: NOW + 100, rng: rng(1) };
    const offer = getDef('reindeer_ring_toss')!.offers(mctx, st(g), 'p1');
    expect(offer).toHaveLength(1);
    g = apply(g, arenaAct(g, 'p1', { action: offer[0] }, ctx(NOW + 100)));
    expect(st(g).throws).toHaveLength(1);
    // and go for what their set still needs
    const need = { ...st(g), got: { p1: ['left-outer', 'left-inner', 'golden', 'right-inner'] }, lastAt: {} };
    for (let i = 0; i < 5; i++) {
      const o = getDef('reindeer_ring_toss')!.offers({ ...mctx, rng: rng(i) }, need, 'p1')[0] as { launch: Launch };
      expect(o.launch.angle).toBeGreaterThan(0.15);                 // the right far prong, the one missing
    }
  });
});

describe('the throw engines in another game', () => {
  beforeAll(() => register(snowballs));
  it('Snowball Splat registers and runs on the same gesture and flight, with no runtime change', () => {
    let g = table(2);
    g.currentTurn = { phase: 'duel', challengerId: 'p1', defenderId: 'p2', pot: 2 };
    const tx = new Tx(g);
    expect(startArena(tx, 'snowball_splat', { mode: 'direct_duel', participants: ['p1', 'p2'], duelists: ['p1', 'p2'] }, ctx(NOW)).error).toBeUndefined();
    g = apply(g, tx.plan());
    const bull = { path: pathFor({ angle: 0, power: (1.2 - DEFAULT_FLIGHT.baseRange) / DEFAULT_FLIGHT.rangePerPower }) };
    const wide = { path: pathFor({ angle: 0.4, power: 0.3 }) };
    for (let i = 0; i < 3; i++) {
      g = apply(g, arenaAct(g, 'p1', { action: bull }, ctx(NOW + i)));
      g = apply(g, arenaAct(g, 'p2', { action: wide }, ctx(NOW + i)));
    }
    expect(arenaOf(g)!.result).toMatchObject({ status: 'resolved', duel: { winningDuelistId: 'p1' } });
  });
});
