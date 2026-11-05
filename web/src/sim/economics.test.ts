// Economic and game-theoretic properties of the rules, pinned as tests. Two kinds:
//  - INVARIANTS (must always hold): the chip ledger balances, no chips appear or vanish except
//    through the named flows, bid costs are what the rules say.
//  - FINDINGS (measured on the simulator, seeded so they're deterministic): what the incentives
//    actually are. If a rule change flips one, the test says so — re-run `npm run
//    analyze:economics` and update ECONOMICS.md rather than just editing the number.
import { describe, expect, it } from 'vitest';
import * as A from '../core/actions';
import * as D from '../core/duel';
import { config, minCost } from '../core/settings';
import { apply, NOW, table } from '../core/testkit';
import type { Game, Settings } from '../core/types';
import { deviationGain, profile, significant } from './experiments';
import { playGame } from './simulate';
import { aggressive, passive, random, valueScaled } from './strategies';

const GAMES = 300;

/** Two players (+ a spectator): p2 bids `bid` on p1's opened gift, the duel goes to `winner`. */
function sealedDuel(bid: number, winner: 'challenger' | 'defender', settings: Settings = {}) {
  let g: Game = table(3, settings);
  g.gifts = { gA: { name: 'A', ownerId: 'p1', naughtyLevel: 0 } };
  g.players!.p1.giftId = 'gA';
  g.currentTurn = { phase: 'reveal', openerId: 'p1', giftId: 'gA', bids: { p2: { amount: bid, submittedAt: NOW }, p3: { amount: 0, submittedAt: NOW } } };
  const before = { p1: g.players!.p1.chips, p2: g.players!.p2.chips };
  g = apply(g, A.startDuel(g, 'p2', 'p1'));
  g = apply(g, D.resolveDuel(g, winner === 'challenger' ? 'p2' : 'p1', NOW));
  return { g, challenger: g.players!.p2.chips - before.p2, defender: g.players!.p1.chips - before.p1 };
}

describe('invariants: the chip ledger', () => {
  it('in every simulated game, chips only change through Santa Tax, costs, dividends, the Elf Bonus (and zero-sum tolls)', () => {
    const mixes = [[aggressive(2), passive, aggressive(4), valueScaled, random, aggressive(1)], Array(8).fill(random), Array(4).fill(aggressive(4))];
    const settingsList: Settings[] = [{}, { loserDividendRule: 'everyone' }, { taxThreshold: 99 }, { maxWinnerPayout: 3, taxThreshold: 3, baseCost: 0 }];
    let games = 0;
    for (const strategies of mixes) for (const settings of settingsList) for (let seed = 1; seed <= 25; seed++) {
      const r = playGame({ strategies, seed, settings });
      const l = r.ledger;
      expect(r.ended, `game ended (seed ${seed})`).toBe(true);
      expect(l.end, `ledger (seed ${seed}, ${JSON.stringify(settings)})`).toBe(l.start - l.taxBurned - l.costsBurned + l.dividendsMinted + l.elfBonusMinted);
      for (const s of r.seats) expect(s.chipsLeft).toBeGreaterThanOrEqual(0);
      games++;
    }
    expect(games).toBe(300);
  });

  it('a sealed-bid duel: what each side gains or loses, per bid (defaults: pots of 2+ pay the winner at most 2)', () => {
    // [bid] → challenger if they win / lose, defender if they win / lose
    const table4 = [1, 2, 3, 4].map((b) => {
      const w = sealedDuel(b, 'challenger'), l = sealedDuel(b, 'defender');
      return { bid: b, challengerWins: w.challenger, challengerLoses: l.challenger, defenderWins: l.defender, defenderLoses: w.defender };
    });
    expect(table4).toEqual([
      { bid: 1, challengerWins: 0, challengerLoses: -1, defenderWins: 1, defenderLoses: 1 },
      { bid: 2, challengerWins: 0, challengerLoses: -2, defenderWins: 2, defenderLoses: 1 },
      { bid: 3, challengerWins: -1, challengerLoses: -3, defenderWins: 2, defenderLoses: 1 },
      { bid: 4, challengerWins: -2, challengerLoses: -4, defenderWins: 2, defenderLoses: 1 },
    ]);
  });

  it('…but a high winning bid also buys protection: Naughty = the top bid, so taking the gift back costs base + bid', () => {
    for (const bid of [1, 2, 3, 4]) {
      const { g } = sealedDuel(bid, 'challenger');
      const gift = g.gifts!.gA;
      expect(gift.ownerId).toBe('p2');
      expect(gift.naughtyLevel).toBe(bid);
      expect(minCost(config(g.settings), gift.naughtyLevel)).toBe(1 + bid); // Gambit / Reprisal / Steal price
    }
  });

  it('yielding is not "receiving nothing": the Opener goes back in the draw bag and opens another gift', () => {
    let g: Game = table(4);
    g.gifts = { gA: { name: 'A', ownerId: 'p1', naughtyLevel: 0 } };
    g.players!.p1.giftId = 'gA';
    g.currentTurn = { phase: 'yield_pending', openerId: 'p1', giftId: 'gA', yieldC1: 'p2', yieldC2: 'p3', bids: { p2: { amount: 2, submittedAt: NOW }, p3: { amount: 1, submittedAt: NOW } } };
    g = apply(g, A.startYieldDuel(g, 'p2', 'p3', true, { now: NOW, rng: () => 0.5 }));
    g = apply(g, D.resolveDuel(g, 'p2', NOW));
    expect(g.players!.p1.giftId ?? null).toBeNull(); // back in the bag → will be drawn again
    expect(g.gifts!.gA.ownerId).toBe('p2');
  });
});

describe('findings: incentives (6 players, defaults, coin-flip duels)', () => {
  it('bidding above 2 is NOT dominated: at a bid-2 table, outbidding to 3 ends with a better gift', () => {
    // The chip cost of a 3 is higher (see the table above) — but chips are worth nothing at the end,
    // and outbidding wins the right to duel. So the Santa Tax cap does not make 2 a ceiling.
    const g3 = deviationGain(aggressive(2), aggressive(3), { games: GAMES });
    expect(g3.mean).toBeGreaterThan(0);
    expect(significant(g3)).toBe(true);
  });

  it('the Santa Tax barely moves incentives: removing it does not change whether outbidding pays', () => {
    const withTax = deviationGain(aggressive(2), aggressive(4), { games: GAMES });
    const noTax = deviationGain(aggressive(2), aggressive(4), { games: GAMES, settings: { taxThreshold: 99 } });
    expect(Math.abs(withTax.mean - noTax.mean)).toBeLessThan(0.3);
  });

  it('a passive table (nobody bids, everyone Gambits) is not stable: any bidder does better there', () => {
    for (const d of [aggressive(1), aggressive(2)]) {
      const g = deviationGain(passive, d, { games: GAMES });
      expect(g.mean, d.name).toBeGreaterThan(0);
      expect(significant(g), d.name).toBe(true);
    }
  });

  it('the regime switch: if leftover chips count (players playing for the chip standings), passive play pays and high bids stop paying', () => {
    // The external analysis's "2 is the ceiling" and "passive equilibrium" are true in THIS regime only.
    const w = 0.5;
    const passiveAtBid2 = deviationGain(aggressive(2), passive, { games: GAMES, chipWeight: w });
    const bid4AtBid2 = deviationGain(aggressive(2), aggressive(4), { games: GAMES, chipWeight: w });
    expect(passiveAtBid2.mean).toBeGreaterThan(0);
    expect(bid4AtBid2.mean).toBeLessThan(0);
    expect(significant(bid4AtBid2)).toBe(true);
  });

  it('KNOWN ISSUE — chips go mostly unspent: even all-in bidders end with over half their chips', () => {
    // Leftover chips have no end-of-game value, so unspent chips are wasted choices. If you add
    // an end-game use for chips (or a reason to keep them), this should fall — update ECONOMICS.md.
    for (const s of [aggressive(2), aggressive(4), valueScaled]) {
      expect(profile(s, { games: 150 }).chipsLeftShare.mean, s.name).toBeGreaterThan(0.5);
    }
  });

  it('at the common bid (2), the chip drain is Gambit/Reprisal/Steal costs, not the Santa Tax', () => {
    const p = profile(aggressive(2), { games: 150 });
    expect(p.taxBurned.mean).toBe(0); // a pot of 2 pays the winner all of it
    expect(p.costsBurned.mean).toBeGreaterThan(p.dividendsMinted.mean); // so supply shrinks
    expect(p.supplyChange.mean).toBeLessThan(0);
  });
});
