// The Gremlin's challenges (M3): memory_flash and perfect_timing next to quick_math. Answers are
// checked by the server; a phone's claimed tap time is believed only within the latency window; the
// challenge reaches the holder's phone and nobody else's.
import { describe, expect, it } from 'vitest';
import { projectGame } from '../../projection';
import { rng } from '../../rng';
import { apply, NOW, table } from '../../testkit';
import { Tx } from '../../tx';
import type { Game } from '../../types';
import { startDuelApp } from '../../duelgame';
import { arenaAct, arenaOf, startArena } from '../runtime';
import { gremlin } from './gremlin';
import type { GremlinState } from './gremlin';
import {
  CHALLENGES, MEMORY_LENGTH, SYMBOLS, TIMING_LATENCY_MS, markerAt, memoryFlash, nextChallenge, perfectTiming, tapTime,
} from './challenges';
import type { TimingParams } from './challenges';

function play(challenges: string[], seed = 1): Game {
  let g = table(3);
  g.currentTurn = { phase: 'duel', challengerId: 'p1', defenderId: 'p2', pot: 2 };
  const tx = new Tx(g);
  const out = startArena(tx, 'pass_the_gremlin', { mode: 'party_with_duelists', participants: ['p1', 'p2', 'p3'], duelists: ['p1', 'p2'], config: { challenges } }, { now: NOW, rng: rng(seed) });
  expect(out.error).toBeUndefined();
  return apply(g, tx.plan());
}
const st = (g: Game) => arenaOf(g)!.state as GremlinState;
const act = (g: Game, who: string, action: unknown, now: number) => arenaAct(g, who, { action }, { now, rng: rng(3) });

describe('memory_flash', () => {
  it('a row of symbols, never the same one twice in a row; only that exact row back is right', () => {
    for (let seed = 1; seed < 50; seed++) {
      const c = memoryFlash.make(rng(seed), NOW);
      const seq = c.answer as number[];
      expect(seq).toHaveLength(MEMORY_LENGTH);
      expect(seq.every((n, i) => n >= 0 && n < SYMBOLS.length && n !== seq[i - 1])).toBe(true);
      expect(memoryFlash.check(c, [...seq], NOW)).toBe(true);
      expect(memoryFlash.check(c, [...seq].reverse(), NOW)).toBe(seq.join() === [...seq].reverse().join());
      expect(memoryFlash.check(c, seq.slice(0, 3), NOW)).toBe(false);
      expect(memoryFlash.check(c, memoryFlash.miss(c, NOW), NOW)).toBe(false);
    }
    expect(memoryFlash.parse([0, 5, 2])).toEqual([0, 5, 2]);
    for (const bad of [[6], [-1], [1.5], 'abc', 3, null, Array(20).fill(0)]) expect(memoryFlash.parse(bad)).toBeNull();
  });

  it('in a game: the holder sees the row; the other phones and the TV never do; the server checks it', () => {
    const g = play(['memory_flash']);
    const s = st(g);
    const seq = s.challenge.answer as number[];
    const mine = projectGame(g, { kind: 'phone', pid: s.holder }, NOW).arenaView!.view as Record<string, unknown>;
    expect(mine.sequence).toEqual(seq);
    expect(mine.showMs).toBeGreaterThan(1000);
    const other = s.pool.find((p) => p !== s.holder)!;
    for (const v of [{ kind: 'phone' as const, pid: other }, { kind: 'tv' as const }, { kind: 'host' as const }]) {
      const view = projectGame(g, v, NOW).arenaView!.view as Record<string, unknown>;
      expect(view.sequence).toBeUndefined();
      expect(view.challengeKind).toBe('memory_flash');
      expect(JSON.stringify(view)).not.toContain(JSON.stringify(seq));
    }
    expect(act(g, s.holder, { kind: 'answer', value: 'nope' }, NOW + 4000).error).toMatch(/Not an answer/);
    const wrong = apply(g, act(g, s.holder, { kind: 'answer', value: memoryFlash.miss(s.challenge, NOW) }, NOW + 4000));
    expect(st(wrong).stage).toBe('holding');
    expect(st(wrong).challenge.kind).toBe('memory_flash');                      // a retry is the same kind
    const w = st(wrong);
    const right = apply(wrong, act(wrong, w.holder, { kind: 'answer', value: w.challenge.answer }, NOW + 6000));
    expect(st(right).stage).toBe('pass_ready');
  });
});

describe('perfect_timing', () => {
  const p: TimingParams = { startAt: 1000, periodMs: 1000, band: [0.4, 0.58] };

  it('the marker swings end to end and back at a steady speed, from its start', () => {
    expect(markerAt(p, 999)).toBeNull();
    expect(markerAt(p, 1000)).toBe(0);
    expect(markerAt(p, 1250)).toBeCloseTo(0.5);
    expect(markerAt(p, 1500)).toBeCloseTo(1);
    expect(markerAt(p, 1750)).toBeCloseTo(0.5);
    expect(markerAt(p, 2000)).toBeCloseTo(0);
  });

  it("believes a phone's tap time only within the latency window; otherwise judges the moment it arrived", () => {
    expect(tapTime(9_500, 10_000)).toBe(9_500);
    expect(tapTime(10_000 - TIMING_LATENCY_MS - 1, 10_000)).toBe(10_000);    // too old: a replayed or cheating tap
    expect(tapTime(10_500, 10_000)).toBe(10_000);                           // from the future
    expect(tapTime('soon', 10_000)).toBe(10_000);
  });

  it('a tap in the band (give or take a hair) passes; outside it, or before the start, fails', () => {
    const c = { kind: 'perfect_timing', prompt: '', answer: null, issuedAt: 0, params: p };
    const at = (x: number) => 1000 + x * 500;                                // on the way out
    expect(perfectTiming.check(c, { at: at(0.49) }, at(0.49) + 100)).toBe(true);
    expect(perfectTiming.check(c, { at: at(0.385) }, at(0.385))).toBe(true);  // within the tolerance
    expect(perfectTiming.check(c, { at: at(0.2) }, at(0.2) + 100)).toBe(false);
    expect(perfectTiming.check(c, { at: at(0.9) }, at(0.9))).toBe(false);
    expect(perfectTiming.check(c, { at: 900 }, 900)).toBe(false);
    // a tap claimed long ago (when the marker was in the band) is judged at arrival, when it isn't
    expect(perfectTiming.check(c, { at: at(0.49) }, at(0.49) + 2000 + 250)).toBe(false);
  });

  it('in a game: the holder gets the motion, not a success flag; the simulated hand finds a moment in the band', () => {
    const g = play(['perfect_timing'], 4);
    const s = st(g);
    const mine = projectGame(g, { kind: 'phone', pid: s.holder }, NOW).arenaView!.view as Record<string, unknown>;
    expect(mine).toMatchObject({ startAt: (s.challenge.params as TimingParams).startAt, periodMs: expect.any(Number), band: expect.any(Array) });
    let now = NOW + 2000, right: unknown = null;
    for (; right === null; now += 50) right = perfectTiming.solve(s.challenge, now);
    expect(gremlin.offers!({ now, rng: rng(1), connected: () => true } as never, s, s.holder)).toEqual([{ kind: 'answer', value: right }]);
    expect(act(g, s.holder, { kind: 'answer', value: { at: 'x' } }, now).error).toBeUndefined();   // a bad time is judged at arrival
    const ok = apply(g, act(g, s.holder, { kind: 'answer', value: right }, now));
    expect(st(ok).stage).toBe('pass_ready');
    const miss = apply(g, act(g, s.holder, { kind: 'answer', value: perfectTiming.miss(s.challenge, now) }, now));
    expect(st(miss).stage).toBe('holding');
  });
});

describe('the challenge mix', () => {
  it('each pass hands the next holder a different kind than the last, when the game has several', () => {
    const kinds = Object.keys(CHALLENGES);
    let last = 'quick_math';
    for (let seed = 1; seed < 40; seed++) {
      const c = nextChallenge(kinds, rng(seed), NOW, last);
      expect(c.kind).not.toBe(last);
      last = c.kind;
    }
    expect(nextChallenge(['memory_flash'], rng(1), NOW, 'memory_flash').kind).toBe('memory_flash');
  });

  it('by default a game uses all three; a game saved before M3 (one `challenge`) still loads; an unknown kind is refused', () => {
    expect(gremlin.validateConfig({}).challenges).toEqual(['quick_math', 'memory_flash', 'perfect_timing']);
    expect(gremlin.validateConfig({ challenge: 'quick_math' }).challenges).toEqual(['quick_math']);
    expect(() => gremlin.validateConfig({ challenges: ['juggling'] })).toThrow(/juggling/);
  });

  it('passing in a mixed game changes the kind', () => {
    let g = play(['quick_math', 'memory_flash', 'perfect_timing'], 2);
    for (let i = 0, now = NOW + 3000; i < 6; i++, now += 3000) {
      const s = st(g);
      let right: unknown = null, t = now;
      for (; right === null; t += 50) right = CHALLENGES[s.challenge.kind].solve(s.challenge, t);
      g = apply(g, act(g, s.holder, { kind: 'answer', value: right }, t));
      g = apply(g, act(g, s.holder, { kind: 'pass_random' }, t + 1500));
      expect(st(g).challenge.kind).not.toBe(s.challenge.kind);
      now = t + 1500;
    }
  });

  it("the host's choice on the setup screen is the game's: one kind, or the mix by default", () => {
    const duel = (settings: Record<string, unknown>) => {
      const g = table(2, { duelPick: 'off', duelAppGames: ['pass_the_gremlin'], voteWindow: 0, ...settings });
      g.currentTurn = { phase: 'duel', challengerId: 'p2', defenderId: 'p1', pot: 2, duelGame: { name: 'Pass the Gremlin', kind: 'app', app: 'pass_the_gremlin', via: 'picker' } };
      return st(apply(g, startDuelApp(g, { now: NOW, rng: rng(1) })));
    };
    expect(duel({ gremlinChallenges: 'memory_flash' }).cfg.challenges).toEqual(['memory_flash']);
    expect(duel({ gremlinChallenges: 'memory_flash' }).challenge.kind).toBe('memory_flash');
    expect(duel({}).cfg.challenges).toEqual(['quick_math', 'memory_flash', 'perfect_timing']);
    expect(duel({ gremlinChallenges: 'bogus' }).cfg.challenges).toHaveLength(3);
  });
});
