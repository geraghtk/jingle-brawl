import { describe, expect, it } from 'vitest';
import { sketchGrid, sketchTile, WALL } from './sketchGrid';

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

describe('Sketch-Off: the wall keeps a margin (the drawings used to touch the frame of the TV)', () => {
  it('at any table and any screen shape the drawings stay a margin clear of every edge', () => {
    for (const [w, h] of [[1880, 880], [1240, 560], [1000, 900], [600, 800]]) for (let n = 1; n <= 30; n++) for (const captions of [true, false]) {
      const t = sketchTile(n, w, h, captions), pad = (WALL.pad * h) / 100;
      expect(t.tile).toBeGreaterThan(0);
      expect(t.width).toBeLessThanOrEqual(w - 2 * pad + 1e-6);
      expect(t.height).toBeLessThanOrEqual(h - 2 * pad + 1e-6);
    }
  });
  it('the gap between two drawings is wider than a done ring on each of them', () => {
    expect(WALL.gap).toBeGreaterThan(2 * 1.5);          // .sk-easel.done / .win: a ring of up to 1.5cqh
    expect(WALL.pad).toBeGreaterThan(1.5 + 1.2);        // the ring and the paper's shadow under it
  });
});
