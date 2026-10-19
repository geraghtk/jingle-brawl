// Economic experiments on the simulator, shared by the tests (economics.test.ts) and the report
// (npm run analyze:economics). Payoff = the value (to you, 0–12) of the gift you end with;
// leftover chips are worth nothing at the end, exactly as in the game.
import type { Settings } from '../core/types';
import { playGame } from './simulate';
import type { GameResult } from './simulate';
import type { Strategy } from './strategies';

export interface Stat { mean: number; se: number; n: number }
const stat = (xs: number[]): Stat => {
  const n = xs.length;
  const mean = xs.reduce((a, b) => a + b, 0) / n;
  const sd = Math.sqrt(xs.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, n - 1));
  return { mean, se: sd / Math.sqrt(n), n };
};

/**
 * One player (rotating through the seats) plays `deviant`, everyone else plays `table`.
 * Returns the deviant's payoff minus a table player's, from the SAME games (paired → low noise).
 * > 0 means deviating pays: `table` is not a best response to itself.
 */
export function deviationGain(table: Strategy, deviant: Strategy, o: { players?: number; games?: number; settings?: Settings; seed?: number; chipWeight?: number } = {}): Stat {
  // chipWeight: what a leftover chip is worth at the end. 0 = nothing (the rules); the final
  // standings are sorted by chips, though, so a table playing for the standings values them > 0.
  const pay = (s: GameResult['seats'][0]) => s.giftValue + (o.chipWeight ?? 0) * s.chipsLeft;
  const n = o.players ?? 6;
  const diffs: number[] = [];
  for (let i = 0; i < (o.games ?? 400); i++) {
    const seat = i % n;
    const strategies = Array.from({ length: n }, (_, s) => (s === seat ? deviant : table));
    const r = playGame({ strategies, seed: (o.seed ?? 1) * 100003 + i, settings: o.settings });
    const mine = pay(r.seats[seat]);
    const others = r.seats.filter((_, s) => s !== seat).map(pay);
    diffs.push(mine - others.reduce((a, b) => a + b, 0) / others.length);
  }
  return stat(diffs);
}

/** A table where everyone plays `s`: what the game looks like (duels, chip flows, leftovers…). */
export function profile(s: Strategy, o: { players?: number; games?: number; settings?: Settings; seed?: number } = {}) {
  const n = o.players ?? 6;
  const runs: GameResult[] = [];
  for (let i = 0; i < (o.games ?? 300); i++) {
    runs.push(playGame({ strategies: Array.from({ length: n }, () => s), seed: (o.seed ?? 7) * 100003 + i, settings: o.settings }));
  }
  const per = (f: (r: GameResult) => number) => stat(runs.map(f));
  return {
    runs,
    duels: per((r) => r.duels.length),
    sealedDuels: per((r) => r.duels.filter((d) => d.kind === 'sealed').length),
    gambits: per((r) => r.gambits),
    payoff: per((r) => r.seats.reduce((a, x) => a + x.giftValue, 0) / n),
    giftless: per((r) => r.seats.filter((x) => !x.gotGift).length),
    chipsLeftShare: per((r) => r.ledger.end / r.ledger.start),
    supplyChange: per((r) => r.ledger.end - r.ledger.start),
    taxBurned: per((r) => r.ledger.taxBurned),
    costsBurned: per((r) => r.ledger.costsBurned),
    dividendsMinted: per((r) => r.ledger.dividendsMinted),
    maxNaughty: per((r) => r.maxNaughty),
  };
}

export const fmt = (s: Stat, d = 2) => `${s.mean >= 0 ? '+' : ''}${s.mean.toFixed(d)} ± ${(1.96 * s.se).toFixed(d)}`;
export const significant = (s: Stat) => Math.abs(s.mean) > 1.96 * s.se;
