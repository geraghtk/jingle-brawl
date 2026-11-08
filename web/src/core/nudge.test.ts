// The dot on the host's tab: lit when the game waits on the host to move it on, dark while the
// table is still deciding (bids coming in, an opener choosing) or when all the host has is an override.
import { describe, expect, it } from 'vitest';
import * as A from './actions';
import { hostsMove } from './nudge';
import { apply, ctx, NOW, table } from './testkit';
import type { Game } from './types';

describe('hostsMove', () => {
  it('the lobby: only once there are two to start with', () => {
    const one: Game = { phase: 'lobby', settings: {}, players: { p1: { name: 'A', chips: 10 } } };
    expect(hostsMove(one, NOW)).toBe(false);
    expect(hostsMove({ ...table(2), phase: 'lobby' }, NOW)).toBe(true);
  });

  it('between turns: lit while gifts are left to draw; End alone never lights it', () => {
    const g = table(3);                                               // three players, no gift drawn yet
    expect(hostsMove(g, NOW)).toBe(true);                             // Draw
    const allOpened: Game = { ...g, gifts: { a: { name: 'A', ownerId: 'p1', naughtyLevel: 0 }, b: { name: 'B', ownerId: 'p2', naughtyLevel: 0 }, c: { name: 'C', ownerId: 'p3', naughtyLevel: 0 } } };
    expect(hostsMove(allOpened, NOW)).toBe(false);                    // only End is left: the host's call, not a nudge
  });

  it('bidding: dark until every bid is in, then lit (Reveal)', () => {
    let g = table(3);
    g = apply(g, A.drawGift(g, 'Gift', 'gA', ctx(1)));
    const t = g.currentTurn!;
    expect(t.phase).toBe('bidding');
    const bidders = Object.keys(g.players!).filter((id) => id !== t.openerId);
    expect(hostsMove(g, NOW)).toBe(false);
    g = apply(g, A.submitBid(g, bidders[0], 1, NOW));
    expect(hostsMove(g, NOW)).toBe(false);
    g = apply(g, A.submitBid(g, bidders[1], 0, NOW));
    expect(hostsMove(g, NOW)).toBe(true);
  });

  it("someone who left doesn't hold up the dot", () => {
    let g = table(3);
    g = apply(g, A.drawGift(g, 'Gift', 'gA', ctx(1)));
    const bidders = Object.keys(g.players!).filter((id) => id !== g.currentTurn!.openerId);
    g.players![bidders[1]].connected = false;
    g = apply(g, A.submitBid(g, bidders[0], 1, NOW));
    expect(hostsMove(g, NOW)).toBe(true);
  });
});
