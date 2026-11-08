// The Gremlin simulation (smoke): every round ends; a party round always gives one of the duelists the win.
import { describe, expect, it } from 'vitest';
import { round } from './gremlin-sim';

const hands = [0, 0.3, 0.6, 1, 0.5].map((x) => ({ speed: x, accuracy: 0.75 + x * 0.2 }));

describe('Pass the Gremlin, simulated', () => {
  it('300 party rounds: none stalls, every one settles the duel on one of the two duelists', () => {
    for (let seed = 1; seed <= 300; seed++) {
      const r = round('party_with_duelists', hands, seed);
      expect(r.stalled).toBe(false);
      expect(['p1', 'p2']).toContain(r.winner);
    }
  });
  it('free-for-all rounds end with a loser and no winner', () => {
    for (let seed = 1; seed <= 100; seed++) {
      const r = round('free_for_all', hands, seed);
      expect(r.stalled).toBe(false);
      expect(r.winner).toBeUndefined();
      expect(r.passes).toBeGreaterThan(0);
    }
  });
});
