import { describe, expect, it } from 'vitest';
import { rng } from '../../rng';
import { COURSE_RULES, courseFaults, makeCourse, TROUBLE } from './course';

describe('a new course for every race', () => {
  it('the same numbers give the same course; different numbers a different one', () => {
    expect(makeCourse(rng(5), 96)).toEqual(makeCourse(rng(5), 96));
    const kinds = (seed: number) => makeCourse(rng(seed), 96).map((o) => `${o.kind}@${o.at}`).join(' ');
    expect(new Set(Array.from({ length: 30 }, (_, i) => kinds(i + 1))).size).toBeGreaterThan(25);
  });

  it('500 courses: every one keeps the rules, has one of everything, and the same amount of trouble', () => {
    for (let seed = 1; seed <= 500; seed++) {
      const c = makeCourse(rng(seed), 96);
      expect(courseFaults(c, 96), `seed ${seed}`).toEqual([]);
      const way = c.filter((o) => o.kind !== 'boost');
      expect(way).toHaveLength(COURSE_RULES.features);
      expect(way.reduce((n, o) => n + TROUBLE[o.kind as keyof typeof TROUBLE], 0), `seed ${seed}`).toBe(COURSE_RULES.budget);
      for (const kind of ['wall', 'gap', 'ice', 'climb', 'rope']) expect(c.some((o) => o.kind === kind), `seed ${seed}: ${kind}`).toBe(true);
      expect(c.filter((o) => o.kind === 'boost').length).toBeGreaterThanOrEqual(2);
      for (const o of c) if (o.kind === 'gap' || o.kind === 'rope' || o.kind === 'ice') expect(o.width).toBeGreaterThan(0);
      // a candle is never inside a gap, and the list is in order along the roof
      for (const b of c.filter((o) => o.kind === 'boost')) expect(c.some((o) => (o.kind === 'gap' || o.kind === 'rope') && b.at > o.at && b.at < o.at + o.width!)).toBe(false);
      expect(c.map((o) => o.at)).toEqual([...c.map((o) => o.at)].sort((a, b) => a - b));
    }
  });

  it('the faults a hand-made course can have are named', () => {
    expect(courseFaults([{ id: 'a', kind: 'climb', at: 20 }, { id: 'b', kind: 'rope', at: 30, width: 7 }], 96)).toEqual(['a and b: two hard ones in a row']);
    expect(courseFaults([{ id: 'a', kind: 'wall', at: 3 }], 96)).toEqual(['a is too near the start']);
    expect(courseFaults([{ id: 'a', kind: 'gap', at: 20, width: 4 }, { id: 'b', kind: 'wall', at: 25 }], 96)).toEqual(['a and b are too close']);
  });
});
