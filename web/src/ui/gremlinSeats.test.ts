import { describe, expect, it } from 'vitest';
import { passPath, seats, seatSize, WORDS } from './gremlinSeats';

describe('Pass the Gremlin: the seats round the table on the big screen', () => {
  it('everyone has a seat inside the scene, the first at the top', () => {
    for (let n = 2; n <= 20; n++) {
      const s = seats(n);
      expect(s).toHaveLength(n);
      for (const p of s) { expect(p.x).toBeGreaterThan(5); expect(p.x).toBeLessThan(95); expect(p.y).toBeGreaterThan(12); expect(p.y).toBeLessThan(92); }
      expect(s[0].x).toBeCloseTo(50, 0);
      expect(s[0].y).toBeLessThan(25);
    }
  });
  it('no two seats overlap, even at a table of twenty (pictures sized for the table)', () => {
    for (let n = 2; n <= 20; n++) {
      const s = seats(n), size = seatSize(n);
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
        // in units of the scene's height (the scene is 16:9)
        const d = Math.hypot((s[i].x - s[j].x) * (16 / 9), s[i].y - s[j].y);
        expect(d).toBeGreaterThan(size * 1.25);
      }
    }
  });
  it('the seats are evenly spread along the oval, not bunched at its ends', () => {
    const s = seats(12);
    const gaps = s.map((p, i) => { const q = s[(i + 1) % 12]; return Math.hypot((p.x - q.x) * (16 / 9), p.y - q.y); });
    expect(Math.max(...gaps) / Math.min(...gaps)).toBeLessThan(1.35);
  });
  it('a pass is drawn round the table, never through the words in the middle of it', () => {
    const points = (d: string) => d.slice(1).split(' L').map((p) => p.split(',').map(Number));
    for (let n = 2; n <= 20; n++) {
      const s = seats(n);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        if (i === j) continue;
        for (const inset of [0, 0.035, 0.07]) {
          const pts = points(passPath(s[i], s[j], inset));
          // it starts and ends on the two seats
          expect(pts[0][0]).toBeCloseTo(s[i].x, 1); expect(pts[0][1]).toBeCloseTo(s[i].y, 1);
          expect(pts.at(-1)![0]).toBeCloseTo(s[j].x, 1); expect(pts.at(-1)![1]).toBeCloseTo(s[j].y, 1);
          for (const [x, y] of pts) expect(x > WORDS.x0 && x < WORDS.x1 && y > WORDS.y0 && y < WORDS.y1, `${n} seats, ${i} to ${j}: ${x},${y}`).toBe(false);
        }
      }
    }
  });
  it('a pass goes the short way round', () => {
    const s = seats(8);
    const len = (d: string) => d.split(' L').length;
    expect(len(passPath(s[0], s[1]))).toBeLessThan(len(passPath(s[0], s[4])));
    expect(len(passPath(s[0], s[7]))).toBe(len(passPath(s[0], s[1])));
  });
});
