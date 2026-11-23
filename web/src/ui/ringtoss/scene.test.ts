import { describe, expect, it } from 'vitest';
import { PEGS } from '../../core/minigames/ringtoss/ringtoss';
import { TRAIL } from './Flights';
import { DEER_Z, restOn, shadowOpacity, stackOn, TIP_H } from './scene';

describe('Ring Toss: the shadow under a ring in the air', () => {
  it('is there under a ring near the ground out in front, fainter as the ring rises', () => {
    expect(shadowOpacity(0.4, 0)).toBeGreaterThan(0.2);
    expect(shadowOpacity(0.4, 0.2)).toBeLessThan(shadowOpacity(0.4, 0.05));
    expect(shadowOpacity(0.4, 0.2)).toBeGreaterThan(0);
  });
  it('is gone under a ring high in the air: nothing on the snowdrift while a ring flies to an antler', () => {
    const lowestTip = Math.min(...Object.values(TIP_H));
    for (const z of [0.6, 1, 1.3]) for (const h of [0.5, lowestTip * 0.6, lowestTip, 1.5]) expect(shadowOpacity(z, h)).toBe(0);
  });
  it('is gone when the ring settles on an antler: no smudge on the snow below the reindeer', () => {
    for (const p of PEGS) expect(shadowOpacity(p.z, TIP_H[p.id])).toBe(0);
    expect(shadowOpacity(DEER_Z + 0.3, 0.2)).toBe(0);
  });
  it('fades in steps, not all at once, as the ring nears the reindeer', () => {
    expect(shadowOpacity(DEER_Z - 0.2, 0.1)).toBeGreaterThan(0);
    expect(shadowOpacity(DEER_Z - 0.2, 0.1)).toBeLessThan(shadowOpacity(DEER_Z - 0.6, 0.1));
  });
});

describe('Ring Toss: where a ringer rests on its antler', () => {
  const ringer = (n: number, at: number, pegId = 'left-inner') => ({ n, pegId, at });
  const landAt = (t: { at: number }) => t.at;
  it('rests below the rings still showing on that antler', () => {
    const throws = [ringer(1, 1000), ringer(2, 1400), ringer(3, 1500, 'golden'), ringer(4, 1800)];
    expect(stackOn(throws[0], throws, landAt)).toBe(0);
    expect(stackOn(throws[1], throws, landAt)).toBe(1);
    expect(stackOn(throws[3], throws, landAt)).toBe(2);
    expect(stackOn(throws[2], throws, landAt)).toBe(0);
  });
  it('the twentieth ringer of a long round still lands on the antler, not far below it', () => {
    const throws = Array.from({ length: 20 }, (_, i) => ringer(i + 1, 1000 + i * 3000));
    expect(stackOn(throws[19], throws, landAt)).toBe(0);
    const p = PEGS.find((x) => x.id === 'left-inner')!;
    expect(restOn(p, stackOn(throws[19], throws, landAt)).sy - restOn(p, 0).sy).toBe(0);
  });
  it('a burst of ringers on one antler never hangs more than a few deep', () => {
    const throws = Array.from({ length: 9 }, (_, i) => ringer(i + 1, 1000 + i * 50));
    expect(stackOn(throws[8], throws, landAt)).toBeLessThanOrEqual(3);
  });
});

describe('Ring Toss: the trail behind a ring', () => {
  it('is slight: two faint copies at most', () => {
    expect(TRAIL.length).toBeLessThanOrEqual(2);
    expect(Math.max(...TRAIL)).toBeLessThanOrEqual(0.2);
  });
});
