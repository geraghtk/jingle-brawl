import { describe, expect, it } from 'vitest';
import { secsLeft } from './clock';

describe('secsLeft', () => {
  it('rounds up, so the last second reads 1 until it is over', () => {
    expect(secsLeft(10_000, 0)).toBe(10);
    expect(secsLeft(10_000, 1)).toBe(10);
    expect(secsLeft(10_000, 9_001)).toBe(1);
    expect(secsLeft(10_000, 9_999)).toBe(1);
    expect(secsLeft(10_000, 10_000)).toBe(0);
  });
  it('never goes below zero', () => {
    expect(secsLeft(10_000, 25_000)).toBe(0);
  });
});
