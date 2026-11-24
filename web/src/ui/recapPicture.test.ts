// The share picture's layout: nothing overlaps, nothing leaves the page, long names are cut with an ellipsis.
import { describe, expect, it } from 'vitest';
import { recapOf } from '../core/recap';
import type { Game } from '../core/types';
import { PIC_H, PIC_W, fitText, layoutPicture, roughMeasure, stepNames } from './recapPicture';
import type { PicItem, PicText } from './recapPicture';

const WORDS = { brand: 'YuleDuel', title: 'The final standings', date: '23 November 2026' };
const LONG = 'Bartholomew Featherstonehaugh the Third of Little Snoring';

/** A finished table of `n` with scores, chips, duels and gifts all different (so every award is given). */
function night(n: number, name: (i: number) => string = (i) => `Player ${i + 1}`, flat = false): Game {
  const g: Game = { phase: 'ended', players: {}, gifts: {} };
  for (let i = 0; i < n; i++) {
    g.players![`p${i}`] = flat ? { name: name(i), chips: 10 } : {
      name: name(i), chips: 3 + i, partyPoints: (n - i) * 2, votingWins: i === 1 ? 6 : i % 3, duelsFought: i === 2 ? 7 : i % 2,
      duelWins: i === 0 ? 3 : 0, duelLosses: i === 2 ? 5 : 0, giftId: `g${i}`,
    };
    if (!flat) g.gifts![`g${i}`] = { name: `${name(i)}'s rather wonderful and very long-named present`, ownerId: `p${i}`, naughtyLevel: i === 3 ? 5 : 1 };
  }
  return g;
}

const texts = (items: PicItem[]) => items.filter((i): i is PicText => i.kind === 'text');
const overlap = (a: PicText, b: PicText) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;

function check(items: PicItem[]) {
  for (const it of items) {
    expect(it.x, JSON.stringify(it)).toBeGreaterThanOrEqual(0);
    expect(it.y, JSON.stringify(it)).toBeGreaterThanOrEqual(0);
    expect(it.x + it.w, JSON.stringify(it)).toBeLessThanOrEqual(PIC_W);
    expect(it.y + it.h, JSON.stringify(it)).toBeLessThanOrEqual(PIC_H);
    expect(it.w).toBeGreaterThan(0);
    expect(it.h).toBeGreaterThan(0);
  }
  const t = texts(items);
  for (let i = 0; i < t.length; i++) for (let j = i + 1; j < t.length; j++) {
    expect(overlap(t[i], t[j]), `"${t[i].full}" over "${t[j].full}"`).toBe(false);
  }
  for (const x of t) {
    expect(roughMeasure(x.text, x.size, x.weight, x.font), x.full).toBeLessThanOrEqual(x.w);
    expect(x.h, x.full).toBeGreaterThanOrEqual(x.size);                 // a line is at least as tall as its letters
  }
}

describe('the share picture', () => {
  it.each([2, 4, 8, 12])('%i players: no text box overlaps another and everything is on the page', (n) => {
    const recap = recapOf(night(n));
    expect(recap.awards.length).toBeGreaterThanOrEqual(n > 2 ? 3 : 1);
    const items = layoutPicture(recap, WORDS);
    check(items);
    for (const l of recap.lines) expect(texts(items).some((t) => t.full === l.name)).toBe(true);        // everyone is on it
    for (const l of recap.lines) expect(texts(items).some((t) => t.full === l.gift!.name)).toBe(true);  // and their gift
    for (const a of recap.awards) expect(texts(items).some((t) => t.full === a.title)).toBe(true);
    expect(items.filter((i) => i.kind === 'avatar').length).toBe(n + recap.podium.reduce((k, s) => k + Math.min(3, s.players.length), 0));
  });

  it.each([2, 4, 8, 12])('%i players with very long names: every name is cut to its box', (n) => {
    const items = layoutPicture(recapOf(night(n, (i) => `${LONG} ${i}`)), WORDS);
    check(items);
    const names = texts(items).filter((t) => t.full.startsWith(LONG) && !t.full.includes('present') && !t.full.includes(':'));
    expect(names.length).toBeGreaterThanOrEqual(n);
    for (const t of names) expect(t.text.endsWith('…')).toBe(true);
  });

  it('a podium of ties, a game nobody scored in and an empty one all lay out', () => {
    const tied = night(6);
    for (const p of Object.values(tied.players!)) Object.assign(p, { partyPoints: 2, votingWins: 0, duelsFought: 0 });
    const r = recapOf(tied);
    expect(r.podium).toHaveLength(1);
    expect(r.podium[0].players).toHaveLength(6);
    check(layoutPicture(r, WORDS));
    check(layoutPicture(recapOf(night(5, undefined, true)), WORDS));
    check(layoutPicture(recapOf({ phase: 'ended' }), WORDS));
    check(layoutPicture(recapOf(night(1)), WORDS));
  });

  it('a short name is left alone, a long one ends in an ellipsis and fits', () => {
    expect(fitText('Ann', 200, 20, 700, 'disp', roughMeasure)).toBe('Ann');
    const cut = fitText(LONG, 200, 20, 700, 'disp', roughMeasure);
    expect(cut.endsWith('…')).toBe(true);
    expect(roughMeasure(cut, 20, 700, 'disp')).toBeLessThanOrEqual(200);
    expect(LONG.startsWith(cut.slice(0, -1))).toBe(true);
    expect(fitText(LONG, 1, 20, 700, 'disp', roughMeasure)).toBe('…');
  });

  it('names on a shared step', () => {
    const n = (...names: string[]) => stepNames(names.map((name) => ({ name })));
    expect([n('Ann'), n('Ann', 'Bo'), n('Ann', 'Bo', 'Cy', 'Di')]).toEqual(['Ann', 'Ann & Bo', 'Ann, Bo +2']);
  });
});
