import { describe, expect, it } from 'vitest';
import { recoil, shotEvery, shotPhase, stagger } from './motion';

describe('Elf Tower Wars: what moves on the TV', () => {
  it('a tower kicks as its shot leaves, and is still again well before the next', () => {
    expect(recoil(0)).toBe(1);
    expect(recoil(0.1)).toBeGreaterThan(0.3);
    expect(recoil(0.1)).toBeLessThan(recoil(0.02));
    expect(recoil(0.5)).toBe(0);
    expect(recoil(0.99)).toBe(0);
    expect(recoil(NaN)).toBe(0);
  });
  it('the kick is in time with the shot: the same loop, whatever the tower', () => {
    const every = shotEvery(false, 1);
    expect(every).toBe(460);
    expect(shotEvery(true, 1)).toBeGreaterThan(every);
    expect(shotEvery(false, 4)).toBe(230);
    // the shot of tower 2 leaves when its phase wraps to 0: the kick is 1 there and nowhere near it half way
    const leaves = every - ((2 * 137) % every);
    expect(shotPhase(leaves, 2, every)).toBeCloseTo(0);
    expect(recoil(shotPhase(leaves, 2, every))).toBeCloseTo(1);
    expect(recoil(shotPhase(leaves + every / 2, 2, every))).toBe(0);
    for (const now of [0, 1, 999, 123456789]) { const k = shotPhase(now, 3, every); expect(k).toBeGreaterThanOrEqual(0); expect(k).toBeLessThan(1); }
  });
  it('toys are out of step with each other, each always the same', () => {
    const ids = ['0-1', '0-2', '0-3', '1-1', '1-2', '1-3'];
    const at = ids.map((id) => stagger(id, 800));
    expect(new Set(at).size).toBeGreaterThan(3);
    expect(at.every((ms) => ms >= 0 && ms < 800)).toBe(true);
    expect(stagger('0-2', 800)).toBe(at[1]);
  });
});
