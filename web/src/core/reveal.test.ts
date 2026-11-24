// The bids a phone lists after the reveal: in order, the top marked, a tie said plainly. And the
// other half: before the reveal a phone's own view of the game gives it nobody else's bid to list.
import { describe, expect, it } from 'vitest';
import * as A from './actions';
import * as D from './duel';
import { projectGame } from './projection';
import { nameList, revealedBids, revealSummary } from './reveal';
import { apply, ctx, NOW, table } from './testkit';
import type { Game } from './types';

/** A table of `n`, a gift opened, bids sealed: `amounts` in the order of the bidders' ids. */
function bidding(n: number, amounts: number[]): { g: Game; opener: string; bidders: string[] } {
  let g = table(n);
  g = apply(g, A.drawGift(g, 'Socks', 'gA', ctx(3)));
  const opener = g.currentTurn!.openerId!;
  const bidders = Object.keys(g.players!).filter((id) => id !== opener);
  amounts.forEach((amount, i) => { g = apply(g, A.submitBid(g, bidders[i], amount, NOW + i)); });
  return { g, opener, bidders };
}
const reveal = (g: Game) => apply(g, A.revealBids(g));
const phone = (g: Game, pid: string) => projectGame(g, { kind: 'phone', pid });

describe('revealedBids', () => {
  it('before the reveal a phone has nothing to list: not from its own view, and not if it tried', () => {
    const { g, bidders } = bidding(5, [4, 2, 0, 3]);
    for (const pid of Object.keys(g.players!)) {
      const view = phone(g, pid);
      expect(revealedBids(view)).toBeNull();
      // and the view itself holds no amount but the phone's own
      for (const [id, b] of Object.entries(view.currentTurn!.bids || {})) {
        if (id !== pid) expect(b).toMatchObject({ amount: 0, sealed: true });
      }
      const others = bidders.filter((id) => id !== pid).map((id) => g.currentTurn!.bids![id].amount).filter((a) => a > 0);
      const seen = Object.entries(view.currentTurn!.bids || {}).filter(([id]) => id !== pid).map(([, b]) => b.amount);
      expect(seen.some((a) => others.includes(a))).toBe(false);
    }
  });

  it('a view that still carries a sealed bid is never listed, whatever the phase says', () => {
    const { g } = bidding(4, [3, 1, 2]);
    const view = phone(g, 'p1');
    view.currentTurn!.phase = 'reveal';                       // a screen that got ahead of itself
    expect(revealedBids(view)).toBeNull();
  });

  it('after the reveal every phone lists every bid, highest first, passes last', () => {
    const { g: g0, opener, bidders } = bidding(5, [2, 4, 0, 3]);
    const g = reveal(g0);
    for (const pid of Object.keys(g.players!)) {
      const r = revealedBids(phone(g, pid))!;
      expect(r.rows.map((x) => x.amount)).toEqual([4, 3, 2, 0]);
      expect(r.rows.map((x) => x.id)).toEqual([bidders[1], bidders[3], bidders[0], bidders[2]]);
      expect(r.rows.map((x) => x.place)).toEqual([1, 2, 3, null]);
      expect(r.rows.map((x) => x.top)).toEqual([true, false, false, false]);
      expect(r).toMatchObject({ top: 4, leaders: [bidders[1]], tie: false, sitter: opener });
      expect(r.rows.some((x) => x.id === opener)).toBe(false);          // the opener does not bid
    }
  });

  it('a tie: every top bid is marked and shares first place; the first to seal is listed first', () => {
    const { g: g0, bidders } = bidding(5, [3, 1, 3, 3]);
    const r = revealedBids(phone(reveal(g0), bidders[0]))!;
    expect(r.tie).toBe(true);
    expect(r.leaders).toEqual([bidders[0], bidders[2], bidders[3]]);
    expect(r.rows.map((x) => x.place)).toEqual([1, 1, 1, 4]);
  });

  it('someone who never sent a bid is listed as such, after the passes; nobody bidding is no top at all', () => {
    const { g: g0, bidders } = bidding(4, [0, 0]);
    const r = revealedBids(reveal(g0))!;
    expect(r.rows).toEqual([
      { id: bidders[0], amount: 0, top: false, place: null },
      { id: bidders[1], amount: 0, top: false, place: null },
      { id: bidders[2], amount: 0, top: false, place: null, missing: true },
    ]);
    expect(r).toMatchObject({ top: 0, leaders: [], tie: false });
  });

  it('the Misfit auction: sealed while bidding, listed after its reveal, without the active player', () => {
    let g = table(4);
    g.phase = 'endgame';
    g.gifts = { m1: { name: 'Fruitcake', ownerId: 'misfit', naughtyLevel: 0, misfitOrder: 1 } };
    g.currentTurn = { phase: 'endgame_bidding', activePlayerId: 'p1', misfitId: 'm1', misfitName: 'Fruitcake' };
    g = apply(g, A.submitBid(g, 'p2', 2, NOW));
    g = apply(g, A.submitBid(g, 'p3', 4, NOW + 1));
    expect(revealedBids(phone(g, 'p2'))).toBeNull();
    expect(phone(g, 'p2').currentTurn!.bids!.p3).toMatchObject({ amount: 0, sealed: true });
    g = apply(g, D.revealAuction(g));
    expect(g.currentTurn!.phase).toBe('endgame_reveal');
    const r = revealedBids(phone(g, 'p2'))!;
    expect(r.rows.map((x) => [x.id, x.amount])).toEqual([['p3', 4], ['p2', 2], ['p4', 0]]);
    expect(r.sitter).toBe('p1');
  });

  it('nothing to list outside a reveal', () => {
    expect(revealedBids(table(3))).toBeNull();
  });
});

describe('revealSummary', () => {
  it('says who is on top, from my seat', () => {
    const { g: g0, bidders } = bidding(5, [2, 4, 0, 3]);
    const g = reveal(g0);
    const r = revealedBids(g)!;
    const top = g.players![bidders[1]].name;
    expect(revealSummary(g, r, bidders[0])).toBe(`${top} has the top bid: 4 chips.`);
    expect(revealSummary(g, r, bidders[1])).toBe('You have the top bid: 4 chips.');
  });
  it('says a tie plainly, with me last', () => {
    const { g: g0, bidders } = bidding(4, [1, 1, 0]);
    const g = reveal(g0);
    const r = revealedBids(g)!;
    const [a, b] = bidders.map((id) => g.players![id].name);
    expect(revealSummary(g, r)).toBe(`A tie at 1 chip: ${a} and ${b}.`);
    expect(revealSummary(g, r, bidders[0])).toBe(`A tie at 1 chip: ${b} and you.`);
    const three = bidding(5, [2, 2, 2, 1]);
    const g3 = reveal(three.g);
    expect(revealSummary(g3, revealedBids(g3)!)).toMatch(/^A 3-way tie at 2 chips: .+, .+ and .+\.$/);
  });
  it('nobody bid', () => {
    const { g: g0 } = bidding(3, [0, 0]);
    const g = reveal(g0);
    expect(revealSummary(g, revealedBids(g)!)).toBe('Nobody bid.');
  });
  it('nameList', () => {
    expect(nameList(['A'])).toBe('A');
    expect(nameList(['A', 'B'])).toBe('A and B');
    expect(nameList(['A', 'B', 'C'])).toBe('A, B and C');
  });
});
