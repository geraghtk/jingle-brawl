import { describe, expect, it } from 'vitest';
import { seats, seatSize } from './gremlinSeats';

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
});
