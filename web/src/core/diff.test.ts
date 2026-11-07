// Only what changed goes over the wire: applying the diff to the old view gives the new one.
import { describe, expect, it } from 'vitest';
import { rng } from './rng';
import { applyUpdates } from './tx';
import { diff } from './diff';
import type { Game } from './types';
import { Table } from '../sim/table';

describe('diff', () => {
  it('small changes are small; deletions are nulls', () => {
    const a = { phase: 'main', players: { p1: { name: 'A', chips: 10 }, p2: { name: 'B', chips: 10 } }, currentTurn: { phase: 'bidding', bids: { p2: { amount: 1 } } } } as unknown as Game;
    const b = { phase: 'main', players: { p1: { name: 'A', chips: 9 }, p2: { name: 'B', chips: 10 } } } as unknown as Game;
    expect(diff(a, b)).toEqual({ 'players/p1/chips': 9, currentTurn: null });
    expect(diff(a, a)).toEqual({});
  });
  it('replays a whole simulated party exactly, step by step', () => {
    const people = Array.from({ length: 5 }, (_, i) => ({ id: `p${i}`, name: `G${i}`, skills: { duel: 0.5, whack: 0.5, sketch: 0.5 } }));
    const t = new Table({ seed: 3, people, settings: { tiebreakerMethod: 'sketch_off' }, autoHost: { paceMs: 4000 } });
    const r = rng(3);
    let prev = t.game;
    let steps = 0;
    for (let tick = 0; tick < 3000 && !t.ended; tick++) {
      for (const p of t.people) { const os = t.offers(p.id); if (os.length && r() < 0.5) t.act(p.id, os[Math.floor(r() * os.length)]); }
      t.advance(3000);
      expect(applyUpdates(prev, diff(prev, t.game))).toEqual(t.game);
      prev = t.game;
      steps++;
    }
    expect(steps).toBeGreaterThan(20);
  });
});
