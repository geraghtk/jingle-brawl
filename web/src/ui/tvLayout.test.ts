import { describe, expect, it } from 'vitest';
import { bestGrid, breakable, fitText, linesAt, PIC_W, railShape } from './tvLayout';

describe('the TV stage: names that fit', () => {
  it('a line breaks at spaces and after a hyphen, never inside a word', () => {
    expect(breakable('Bartholomew-Alexander')).toEqual(['Bartholomew-', 'Alexander']);
    expect(breakable('A really  long gift')).toEqual(['A', 'really', 'long', 'gift']);
    expect(breakable('Al')).toEqual(['Al']);
  });

  it('a short name is set at the biggest size', () => {
    expect(fitText('Bob', { width: 60, max: 10, min: 4 })).toEqual({ size: 10, lines: 1, tight: false });
  });

  it('a long name shrinks until it fits on one line', () => {
    const f = fitText('Bartholomew-Alexander', { width: 60, max: 10, min: 3 });
    expect(f.tight).toBe(false);
    expect(f.size).toBeLessThan(10);
    expect(linesAt('Bartholomew-Alexander', 60, f.size)).toBe(1);
  });

  it('with two lines allowed it wraps between the words and stays bigger', () => {
    const one = fitText('Bartholomew-Alexander', { width: 60, max: 10, min: 3 });
    const two = fitText('Bartholomew-Alexander', { width: 60, max: 10, min: 3, maxLines: 2 });
    expect(two.size).toBeGreaterThan(one.size);
    expect(two.lines).toBe(2);
    // no single piece is wider than the box at that size
    for (const part of breakable('Bartholomew-Alexander')) expect(part.length * 0.63 * two.size).toBeLessThanOrEqual(60);
  });

  it('capitals are wider, so they are set smaller', () => {
    const lower = fitText('Bartholomew-Alexander', { width: 60, max: 12, min: 3 });
    const upper = fitText('BARTHOLOMEW-ALEXANDER', { width: 60, max: 12, min: 3, upper: true });
    expect(upper.size).toBeLessThan(lower.size);
  });

  it('a word that cannot fit even at the smallest size is flagged, at the smallest size', () => {
    expect(fitText('x'.repeat(60), { width: 20, max: 8, min: 3 })).toEqual({ size: 3, lines: 1, tight: true });
  });

  it('a long gift name uses its lines', () => {
    const name = 'The World\'s Largest Novelty Inflatable Reindeer Lawn Ornament';
    const f = fitText(name, { width: 50, max: 7, min: 2.5, maxLines: 3 });
    expect(f.tight).toBe(false);
    expect(linesAt(name, 50, f.size)).toBeLessThanOrEqual(3);
  });
});

describe('the TV stage: grids', () => {
  it('fills the area with the biggest tiles that fit', () => {
    expect(bestGrid(1, 80, 60)).toEqual({ cols: 1, rows: 1, tile: 60 });
    const g = bestGrid(8, 80, 60, { ratio: 1.3, gap: 2 });
    expect(g.cols * g.rows).toBeGreaterThanOrEqual(8);
    expect(g.cols * g.tile + (g.cols - 1) * 2).toBeLessThanOrEqual(80 + 1e-6);
    expect(g.rows * g.tile * 1.3 + (g.rows - 1) * 2).toBeLessThanOrEqual(60 + 1e-6);
  });

  it('every table size from 1 to 30 fits its area', () => {
    for (let n = 1; n <= 30; n++) {
      const g = bestGrid(n, 84, 62, { ratio: 1.35, gap: 1.5, maxTile: 20 });
      expect(g.tile).toBeGreaterThan(0);
      expect(g.tile).toBeLessThanOrEqual(20);
      expect(g.cols * g.rows).toBeGreaterThanOrEqual(n);
      expect(g.cols * g.tile + (g.cols - 1) * 1.5).toBeLessThanOrEqual(84 + 1e-6);
      expect(g.rows * g.tile * 1.35 + (g.rows - 1) * 1.5).toBeLessThanOrEqual(62 + 1e-6);
    }
  });

  it('the rail: one row for a usual table, two rows of smaller tiles for a big one, always inside the picture', () => {
    expect(railShape(6)).toMatchObject({ rows: 1, cols: 6, compact: false });
    expect(railShape(8)).toMatchObject({ rows: 1, cols: 8, compact: false });
    expect(railShape(14)).toMatchObject({ rows: 2, cols: 7, compact: true });
    for (let n = 1; n <= 30; n++) {
      const r = railShape(n);
      expect(r.cols * r.rows).toBeGreaterThanOrEqual(n);
      expect(r.cols * r.tileW + (r.cols - 1) * 1.2).toBeLessThanOrEqual(PIC_W - 6 + 1e-6);
    }
  });
});
