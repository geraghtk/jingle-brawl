import { describe, expect, it } from 'vitest';
import { sketchGrid } from './sketchGrid';

describe('Sketch-Off: the wall of drawings on the big screen', () => {
  it('always has room for every drawing', () => {
    for (let n = 1; n <= 30; n++) { const g = sketchGrid(n); expect(g.cols * g.rows).toBeGreaterThanOrEqual(n); }
  });
  it('uses the screen: a few drawings are one big row, a full table several rows', () => {
    expect(sketchGrid(2)).toEqual({ cols: 2, rows: 1 });
    expect(sketchGrid(3).rows).toBe(1);
    expect(sketchGrid(6)).toEqual({ cols: 3, rows: 2 });
    expect(sketchGrid(8)).toEqual({ cols: 4, rows: 2 });
    expect(sketchGrid(20).rows).toBeGreaterThanOrEqual(3);
  });
  it('no arrangement of the same drawings would make them bigger', () => {
    for (let n = 1; n <= 30; n++) {
      const g = sketchGrid(n, 2), size = Math.min(2 / g.cols, 1 / g.rows);
      for (let cols = 1; cols <= n; cols++) expect(Math.min(2 / cols, 1 / Math.ceil(n / cols))).toBeLessThanOrEqual(size + 1e-9);
    }
  });
});
