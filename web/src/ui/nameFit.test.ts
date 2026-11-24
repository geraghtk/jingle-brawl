// Type that fits the name: a long name comes out smaller and whole, never broken inside a word.
import { describe, expect, it } from 'vitest';
import { fitName, nameWidth, nameWords } from './nameFit';
import { cleanName, NAME_MAX } from '../core/actions';

const LONGEST = 'Bartholomew-Alexander II';   // the most the join form takes

describe('fitName', () => {
  it('a short name is drawn at full size', () => {
    expect(fitName('Alice', { chars: 9 })).toEqual({ scale: 1, whole: true });
    expect(fitName('Bob', { chars: 5, lines: 1 })).toEqual({ scale: 1, whole: true });
    expect(fitName('', { chars: 5 })).toEqual({ scale: 1, whole: true });
  });

  it('a long single word shrinks until it fits one line, whole', () => {
    const fit = fitName('Christopher', { chars: 8 });
    expect(fit.whole).toBe(true);
    expect(fit.scale).toBeLessThan(1);
    expect(nameWidth('Christopher') * fit.scale).toBeLessThanOrEqual(8);
  });

  it('the longer the word, the smaller the type (never larger)', () => {
    const names = ['Al', 'Alice', 'Christopher', 'Bartholomew-Alexander', 'Wolfeschlegelsteinhausen'];
    const scales = names.map((n) => fitName(n, { chars: 9, lines: 2 }).scale);
    for (let i = 1; i < scales.length; i++) expect(scales[i]).toBeLessThanOrEqual(scales[i - 1]);
    expect(scales[0]).toBe(1);
    expect(scales[scales.length - 1]).toBeLessThan(0.6);
  });

  it('a name may break at a space or after a hyphen, so two short words need no shrinking', () => {
    expect(nameWords('Mary-Jane van Dyke')).toEqual(['Mary-', 'Jane', 'van', 'Dyke']);
    expect(fitName('Mary-Jane', { chars: 6, lines: 2 })).toEqual({ scale: 1, whole: true });
    expect(fitName('Mary-Jane', { chars: 6, lines: 1 }).scale).toBeLessThan(1);     // one line only: it has to shrink
    // the hyphenated name fits at a bigger size than the same letters unbroken
    expect(fitName('Bartholomew-Alexander', { chars: 9 }).scale).toBeGreaterThan(fitName('BartholomewAlexander', { chars: 9 }).scale);
  });

  it('counts the lines the place has: more words than lines shrinks too', () => {
    const two = fitName('Anna Maria Luisa de la Cruz', { chars: 8, lines: 2 });
    const three = fitName('Anna Maria Luisa de la Cruz', { chars: 8, lines: 3 });
    expect(two.whole && three.whole).toBe(true);
    expect(three.scale).toBeGreaterThan(two.scale);
  });

  it('capitals and wide letters take more room than narrow ones', () => {
    expect(nameWidth('CHRISTOPHER')).toBeGreaterThan(nameWidth('Christopher'));
    expect(nameWidth('Mmmmm')).toBeGreaterThan(nameWidth('Illii'));
    expect(fitName('WILHELMINA', { chars: 8 }).scale).toBeLessThan(fitName('Wilhelmina', { chars: 8 }).scale);
  });

  it('the longest name the join form allows still comes out whole in every big place', () => {
    expect(LONGEST.length).toBe(NAME_MAX);
    expect(cleanName(LONGEST + ' and then some more')).toBe(LONGEST);
    for (const spec of [{ chars: 7, lines: 3 }, { chars: 9, lines: 2 }, { chars: 12, lines: 2 }, { chars: 16, lines: 2 }]) {
      const fit = fitName(LONGEST, spec);
      expect(fit.scale).toBeGreaterThanOrEqual(0.4);
      expect(fit.whole).toBe(true);
    }
  });

  it('only a name no size can fit is allowed to break inside a word, and never below the smallest size', () => {
    const wall = 'W'.repeat(60);
    const fit = fitName(wall, { chars: 8, lines: 2, min: 0.4 });
    expect(fit).toEqual({ scale: 0.4, whole: false });
    expect(fitName('Christopher', { chars: 8, lines: 1, min: 0.5 }).whole).toBe(true);
  });
});
