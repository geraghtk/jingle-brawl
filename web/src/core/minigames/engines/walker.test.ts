// The walker on its own: legs that take turns, a lean that grows by itself to an exact moment, a
// wall, a gap that needs both legs, a boost; and the rule every clock-settled engine keeps: settling
// in one go or step by step gives the same walk.
import { describe, expect, it } from 'vitest';
import { blocker, boostNear, DEFAULT_WALK as P, free, gapAhead, grab, jump, leanAt, newWalker, nextAt, posAt, push, settle, shiftWalker, smash, step, topplesAt, wallAhead } from './walker';
import type { Course, Leg, Walker } from './walker';

const T0 = 1000;
const open: Course = { length: 40, seed: 7, obstacles: [] };
const must = (d: { walker?: Walker; error?: string }): Walker => { if (!d.walker) throw new Error(d.error); return d.walker; };
/** Walk `n` good steps (legs in turn, each as the last lands), holding the lean near upright. */
function walk(w: Walker, n: number, at: number, course: Course, first: Leg = 0): { w: Walker; at: number } {
  let leg = first;
  for (let i = 0; i < n; i++) {
    w = settle(w, at, course);
    if (Math.abs(leanAt(w, at)) > 0.3) w = must(push(w, leanAt(w, at) > 0 ? -1 : 1, at));
    w = must(step(w, leg, at, course));
    at += P.stepMs;
    w = settle(w, at, course);
    leg = leg === 0 ? 1 : 0;
  }
  return { w, at };
}

describe('the walker: legs', () => {
  it('a step moves it one stride, over the time a step takes; screens see it part way along', () => {
    let w = must(step(newWalker(T0), 0, T0, open));
    expect(posAt(w, T0 + P.stepMs / 2)).toBeCloseTo(P.stride / 2);
    expect(step(w, 1, T0 + P.stepMs - 50, open).error).toMatch(/Too soon/);         // a hair early is let go
    w = settle(w, T0 + P.stepMs, open);
    expect(w.pos).toBe(P.stride);
    expect(w.last).toBe(0);
    expect(free(w, T0 + P.stepMs)).toBe(true);
  });

  it('a part moves whenever it is told to: a step while a foot is still in the air trips it where it is', () => {
    const w = must(step(newWalker(T0), 0, T0, open));
    const trip = must(step(w, 1, T0 + P.stepMs / 2, open));                           // the other leg, far too soon
    expect(trip).toMatchObject({ busy: 'stumble', busyUntil: T0 + P.stepMs / 2 + P.stumbleMs, last: -1 });
    expect(trip.move).toBeUndefined();
    expect(trip.pos).toBeCloseTo(P.stride / 2);                                      // down where it had got to
    expect(trip.events.at(-1)).toMatchObject({ kind: 'stumble', leg: 1 });
    expect(step(trip, 0, T0 + P.stepMs, open).error).toMatch(/getting up/);
    // in the air over a gap or on a rocket candle a leg does nothing at all
    const course: Course = { length: 40, seed: 3, obstacles: [{ id: 'b1', kind: 'boost', at: 1 }] };
    expect(step(must(grab(newWalker(T0), T0, course)), 0, T0 + 100, course).error).toMatch(/Too soon/);
  });

  it('legs take turns: the same leg twice is a stumble that costs time and mends itself', () => {
    let w = settle(must(step(newWalker(T0), 0, T0, open)), T0 + P.stepMs, open);
    w = must(step(w, 0, T0 + P.stepMs, open));
    expect(w).toMatchObject({ busy: 'stumble', busyUntil: T0 + P.stepMs + P.stumbleMs, pos: P.stride });
    expect(w.events.at(-1)).toMatchObject({ kind: 'stumble', leg: 0 });
    expect(step(w, 1, T0 + P.stepMs + 200, open).error).toMatch(/getting up/);
    const after = T0 + P.stepMs + P.stumbleMs;
    w = settle(w, after, open);
    expect(w.busy).toBeUndefined();
    expect(step(w, 0, after, open).error).toBeUndefined();           // either leg may start again
  });

  it('a step started as the last one lands is in stride and goes further; a late one is a plain step', () => {
    const a = settle(must(step(newWalker(T0), 0, T0, open)), T0 + P.stepMs, open);
    const quick = settle(must(step(a, 1, T0 + P.stepMs + 100, open)), T0 + 5000, { ...open, seed: 1 });
    const slow = settle(must(step(a, 1, T0 + P.stepMs + P.strideMs + 50, open)), T0 + 5000, { ...open, seed: 1 });
    expect(quick.pos - P.stride).toBeCloseTo(P.stride * P.strideBonus);
    expect(slow.pos - P.stride).toBeCloseTo(P.stride);
  });

  it('crossing the line ends the walk at the moment the foot lands', () => {
    const short: Course = { length: 3, seed: 1, obstacles: [] };
    let w = settle(must(step(newWalker(T0), 0, T0, short)), T0 + P.stepMs, short);
    w = must(step(w, 1, T0 + P.stepMs, short));
    w = settle(w, T0 + 10_000, short);
    expect(w.finishedAt).toBe(T0 + 2 * P.stepMs);
    expect(w.pos).toBe(3);
    expect(step(w, 0, T0 + 10_000, short).error).toMatch(/Over the line/);
    expect(nextAt(w)).toBeNull();
  });
});

describe('the walker: balance', () => {
  it('standing straight it stays straight; a step kicks it towards that leg; left alone the lean grows', () => {
    const still = newWalker(T0);
    expect(topplesAt(still)).toBeNull();
    const w = settle(must(step(still, 0, T0, open)), T0 + P.stepMs, open);
    expect(w.lean).toBeLessThan(0);                                    // the left leg: it tips left
    const now = T0 + P.stepMs;
    expect(Math.abs(leanAt(w, now + 2000))).toBeGreaterThan(Math.abs(leanAt(w, now)));
  });

  it('the moment it topples is known exactly: upright the instant before, down at it, and up again by itself', () => {
    const w = settle(must(step(newWalker(T0), 0, T0, open)), T0 + P.stepMs, open);
    const at = topplesAt(w)!;
    expect(nextAt(w)).toBe(at);
    expect(settle(w, at - 1, open).busy).toBeUndefined();
    const down = settle(w, at, open);
    expect(down).toMatchObject({ busy: 'topple', busyUntil: at + P.toppleMs, lean: 0 });
    expect(down.events.at(-1)).toMatchObject({ kind: 'topple', at });
    expect(step(down, 1, at + 100, open).error).toMatch(/getting up/);
    const up = settle(down, at + P.toppleMs, open);
    expect(up.busy).toBeUndefined();
    expect(topplesAt(up)).toBeNull();
    expect(step(up, 1, at + P.toppleMs, open).error).toBeUndefined();
  });

  it('a push sets the lean back (and puts off the fall); pushing the wrong way brings it on; not twice in a blink', () => {
    const w = settle(must(step(newWalker(T0), 0, T0, open)), T0 + P.stepMs, open);   // leaning left
    const now = T0 + P.stepMs + 1500;
    const before = leanAt(w, now);
    const right = must(push(settle(w, now, open), 1, now));
    expect(right.lean).toBeCloseTo(before + P.push);
    const wrong = must(push(settle(w, now, open), -1, now));
    expect(topplesAt(wrong)!).toBeLessThan(topplesAt(w)!);
    expect(push(right, 1, now + 50).error).toMatch(/Easy/);
    expect(push(right, 1, now + P.pushMs).error).toBeUndefined();
  });

  it('toppling in mid step drops it where it was, never past what is in its way', () => {
    const course: Course = { length: 40, seed: 7, obstacles: [{ id: 'w', kind: 'wall', at: 1 }] };
    let w = must(step(newWalker(T0), 0, T0, course));
    w = { ...w, lean: 0.9995, leanAsOf: T0 };
    w = settle(w, T0 + 400, course);
    expect(w.busy).toBe('topple');
    expect(w.pos).toBeLessThanOrEqual(1);
    expect(w.move).toBeUndefined();
  });
});

describe('the walker: what is in the way', () => {
  const walled: Course = { length: 40, seed: 3, obstacles: [{ id: 'w1', kind: 'wall', at: 5 }] };
  it('a wall stops the walk until the arms have hit it enough; out of reach there is nothing to smash', () => {
    const air = must(smash(newWalker(T0), T0, walled));                // the arms swing whenever they are told to: at nothing, it lurches
    expect(air.events.at(-1)).toMatchObject({ kind: 'whiff' });
    expect(Math.abs(air.lean)).toBeCloseTo(P.kick * 2);
    expect(air.hits).toEqual({});
    expect(smash(air, T0 + 10, walled).error).toMatch(/Easy/);
    let { w, at } = walk(newWalker(T0), 3, T0, walled);
    expect(w.pos).toBe(5);                                             // the step was cut short at the wall
    const bonk = must(step(w, w.last === 0 ? 1 : 0, at, walled));       // a step with the wall still up: straight into it
    expect(bonk).toMatchObject({ busy: 'stumble', pos: 5 });
    expect(bonk.events.at(-1)).toMatchObject({ kind: 'bump', id: 'w1' });
    expect(wallAhead(w, walled)?.id).toBe('w1');
    for (let i = 0; i < P.wallHits; i++) {
      expect(smash(w, at + 10, walled).error ?? (i ? 'Easy!' : undefined)).toBe(i ? 'Easy!' : undefined);
      w = must(smash(w, at, walled));
      at += P.hitMs;
    }
    expect(w.events.at(-1)).toMatchObject({ kind: 'smashed', id: 'w1' });
    expect(blocker(w, walled)).toBeNull();
    w = settle(w, at + 5000, walled);                                  // (it may have toppled meanwhile: it gets up)
    expect(step(settle(w, at + 9000, walled), 0, at + 9000, walled).error).toBeUndefined();
  });

  const gapped: Course = { length: 40, seed: 3, obstacles: [{ id: 'g1', kind: 'gap', at: 4, width: 4 }] };
  it('a gap needs both legs within the window: the first crouches, the second sends it over', () => {
    let { w, at } = walk(newWalker(T0), 2, T0, gapped);
    expect(w.pos).toBe(4);
    expect(gapAhead(w, gapped)?.id).toBe('g1');
    w = must(jump(w, 0, at, gapped));
    expect(w.crouch).toEqual({ leg: 0, at });
    expect(jump(w, 0, at + 100, gapped).error).toMatch(/other leg/);
    w = must(jump(w, 1, at + P.jumpWindowMs - 1, gapped));
    expect(w.move).toMatchObject({ kind: 'jump', from: 4, to: 8.5 });
    w = settle(w, at + P.jumpWindowMs - 1 + P.jumpMs, gapped);
    expect(w.pos).toBe(8.5);
    expect(blocker(w, gapped)).toBeNull();
    // no gap in front: both legs jumping is a hop that goes nowhere much (slower than walking), one alone still a stall
    const flat = settle(w, at + 9000, gapped);
    const hop = must(jump(must(jump(flat, 0, at + 9000, gapped)), 1, at + 9100, gapped));
    expect(hop.move).toMatchObject({ kind: 'jump', from: 8.5, to: 8.5 + P.stride * P.hop });
    expect((P.stride * P.hop) / P.jumpMs).toBeLessThan(P.stride / P.stepMs);
    expect(settle(must(jump(flat, 0, at + 9000, gapped)), at + 9000 + P.jumpWindowMs, gapped).busy).toBe('misjump');
  });

  it('one leg jumping alone gives up after the window (a short stall); a step into the gap falls in and climbs back out', () => {
    let { w, at } = walk(newWalker(T0), 2, T0, gapped);
    const lone = settle(must(jump(w, 1, at, gapped)), at + P.jumpWindowMs, gapped);
    expect(lone).toMatchObject({ busy: 'misjump', busyUntil: at + P.jumpWindowMs + P.misjumpMs, pos: 4 });
    expect(lone.crouch).toBeUndefined();
    w = must(step(w, w.last === 0 ? 1 : 0, at, gapped));
    expect(w).toMatchObject({ busy: 'climb', busyUntil: at + P.climbMs, pos: 4 });
    expect(w.events.at(-1)).toMatchObject({ kind: 'fell', id: 'g1' });
    w = settle(w, at + P.climbMs, gapped);
    expect(jump(w, 0, at + P.climbMs, gapped).error).toBeUndefined();   // back at the edge: try again
  });

  it('a boost is grabbed on the way past, once, and carries it forward but not through a wall', () => {
    const course: Course = { length: 40, seed: 3, obstacles: [{ id: 'b1', kind: 'boost', at: 2 }, { id: 'w1', kind: 'wall', at: 6 }] };
    expect(boostNear(newWalker(T0), course, T0)?.id).toBe('b1');
    let w = must(grab(newWalker(T0), T0, course));
    expect(w.move).toMatchObject({ kind: 'boost', from: 0, to: 6 });
    w = settle(w, T0 + P.boostMs, course);
    expect(w.pos).toBe(6);
    const far: Course = { length: 40, seed: 3, obstacles: [{ id: 'b1', kind: 'boost', at: 20 }] };
    const miss = must(grab(newWalker(T0), T0, far));                   // a grab at nothing: a lurch, and the candle is still there
    expect(miss.events.at(-1)).toMatchObject({ kind: 'whiff' });
    expect(miss.taken).toEqual([]);
    expect(miss.lean).not.toBe(0);
  });
});

describe('the walker: the clock', () => {
  const course: Course = { length: 40, seed: 11, obstacles: [{ id: 'g1', kind: 'gap', at: 6, width: 3 }] };
  it('settling in one go or moment by moment gives the same walk', () => {
    const start = must(step(newWalker(T0), 0, T0, course));
    const once = settle(start, T0 + 30_000, course);
    let slow = start;
    for (let t = T0; t <= T0 + 30_000; t += 37) slow = settle(slow, t, course);
    slow = settle(slow, T0 + 30_000, course);
    expect(slow).toEqual(once);
    expect(once.events.some((e) => e.kind === 'topple')).toBe(true);    // left alone, it went over (and got up, and stood)
  });

  it('a pause moves every time in the walk on by its length', () => {
    const w = must(jump(walk(newWalker(T0), 3, T0, course).w, 0, T0 + 3 * P.stepMs + 5, course));
    const later = shiftWalker(w, 60_000);
    expect(later.crouch!.at).toBe(w.crouch!.at + 60_000);
    expect(nextAt(later)).toBe(nextAt(w)! + 60_000);
    expect(leanAt(later, T0 + 61_000)).toBeCloseTo(leanAt(w, T0 + 1000));
  });

  it('the same course gives the same kicks to anyone who walks it the same way', () => {
    const a = walk(newWalker(T0), 6, T0, open).w, b = walk(newWalker(T0), 6, T0, open).w;
    expect(a).toEqual(b);
    expect(walk(newWalker(T0), 6, T0, { ...open, seed: 8 }).w.lean).not.toBe(a.lean);
  });
});
