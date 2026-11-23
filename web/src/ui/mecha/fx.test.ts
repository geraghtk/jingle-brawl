import { describe, expect, it } from 'vitest';
import { hangAt, JOIN_AT, LANDING_FRESH_MS, PUNCH_MS, punchAt, ROPE, SPLIT_AT, splitFor, STRIPS, touchdown, VAULT_ACROSS, VAULT_CLEAR, vaultAt } from './fx';

describe('Mecha Santa: the camera punch', () => {
  it('starts down by its size, dies away, and is still at the end', () => {
    expect(punchAt(0.4, 0)).toBeCloseTo(-0.4);
    const early = Math.max(...Array.from({ length: 10 }, (_, k) => Math.abs(punchAt(0.4, k * 10))));
    const late = Math.max(...Array.from({ length: 10 }, (_, k) => Math.abs(punchAt(0.4, PUNCH_MS - 100 + k * 10))));
    expect(late).toBeLessThan(early / 10);
    expect(punchAt(0.4, PUNCH_MS)).toBe(0);
    expect(punchAt(0.4, -5)).toBe(0);
    expect(punchAt(0, 50)).toBe(0);
  });
  it('bounces: it goes above its mark as well as below', () => {
    const ys = Array.from({ length: 50 }, (_, k) => punchAt(1, k * 10));
    expect(Math.min(...ys)).toBeLessThan(0);
    expect(Math.max(...ys)).toBeGreaterThan(0.1);
  });
});

describe('Mecha Santa: when a foot comes down', () => {
  const step = { kind: 'step' as const, leg: 1, at: 1000, lands: 1300 };
  it('watches a move while it is in the air', () => {
    expect(touchdown(null, step, 1100)).toEqual({ air: step, landed: null });
    expect(touchdown(step, step, 1200)).toEqual({ air: step, landed: null });
  });
  it('lands once: when its time has come, or the view has already moved on', () => {
    expect(touchdown(step, step, 1300)).toEqual({ air: null, landed: step });
    expect(touchdown(null, step, 1320)).toEqual({ air: null, landed: null });
    expect(touchdown(step, undefined, 1310).landed).toEqual(step);
    const next = { kind: 'step' as const, leg: 0, at: 1310, lands: 1610 };
    expect(touchdown(step, next, 1320)).toEqual({ air: next, landed: step });
  });
  it('does not land a move a topple cut short, or one that is old news', () => {
    expect(touchdown(step, undefined, 1200).landed).toBeNull();
    expect(touchdown(step, undefined, 1310, 'topple').landed).toBeNull();
    expect(touchdown(step, undefined, 1300 + LANDING_FRESH_MS).landed).toBeNull();
  });
});

describe('Mecha Santa: the split picture', () => {
  it('splits when one robot is too far behind for one camera, and joins only once they are well within range', () => {
    expect(splitFor(SPLIT_AT - 0.1, false)).toBe(false);
    expect(splitFor(SPLIT_AT + 0.1, false)).toBe(true);
    expect(splitFor(-(SPLIT_AT + 0.1), false)).toBe(true);                // whichever is in front
    expect(splitFor(SPLIT_AT - 1, true)).toBe(true);                      // between the two marks it stays as it is: no flicker
    expect(splitFor(JOIN_AT + 0.1, true)).toBe(true);
    expect(splitFor(JOIN_AT - 0.1, true)).toBe(false);
    expect(JOIN_AT).toBeLessThan(SPLIT_AT);
  });

  it('the two strips sit one above the other, clear of the panels above and the course bar below', () => {
    const [a, b] = STRIPS;
    expect(a.top).toBeGreaterThanOrEqual(0.2);
    expect(a.top + a.height).toBeCloseTo(b.top);
    expect(b.top + b.height).toBeLessThanOrEqual(0.92);
    expect(a.height).toBeCloseTo(b.height);
  });

  it('a robot on a rope hangs under the pivot and swings back and forward the same distance', () => {
    expect(hangAt(0).dx).toBeCloseTo(0);
    expect(hangAt(1).dx).toBeCloseTo(-hangAt(-1).dx);
    expect(hangAt(1).dx).toBeGreaterThan(1.5);
    expect(hangAt(0).y).toBeLessThan(hangAt(1).y);                        // lowest in the middle of the swing
    expect(hangAt(0).y).toBeCloseTo(ROPE.pivotY - ROPE.length - ROPE.hands);
    expect(hangAt(5)).toEqual(hangAt(1));                                 // never past its reach
  });
});

describe('Mecha Santa: over the top of a stack', () => {
  it('goes across the top clear of it, and only then down the far side: never through it', () => {
    const top = 3.1;
    // all the way across while it is still above the top
    for (let f = 0; f < VAULT_ACROSS; f += 0.05) expect(vaultAt(f, top).y, `f ${f}`).toBeGreaterThanOrEqual(top + VAULT_CLEAR - 1e-9);
    expect(vaultAt(VAULT_ACROSS, top).over).toBe(1);
    // then straight down: it is already across, and it only gets lower
    let last = Infinity;
    for (let f = VAULT_ACROSS; f <= 1.0001; f += 0.05) { const v = vaultAt(f, top); expect(v.over).toBe(1); expect(v.y).toBeLessThanOrEqual(last + 1e-9); last = v.y; }
    expect(vaultAt(1, top).y).toBeCloseTo(0);
    expect(vaultAt(0, top)).toMatchObject({ over: 0 });
    expect(vaultAt(0.3, top).over).toBeCloseTo(0.5);
  });
});
