// Admin edits and the AI Draw Off flow keep gifts and players consistent.
import { describe, expect, it } from 'vitest';
import * as Admin from './admin';
import * as DO from './drawoff';
import { apply, ctx, NOW, table } from './testkit';
import type { Game } from './types';

/** Every gift owner holds exactly that gift, and vice versa. */
function consistent(g: Game) {
  for (const [id, gift] of Object.entries(g.gifts || {})) {
    if (gift.ownerId !== 'misfit') expect(g.players![gift.ownerId].giftId, `gift ${id}`).toBe(id);
  }
  for (const [id, p] of Object.entries(g.players || {})) {
    if (p.giftId) expect(g.gifts![p.giftId].ownerId, `player ${id}`).toBe(id);
  }
}

function withGifts(): Game {
  const g = table(3);
  g.gifts = { ga: { name: 'A', ownerId: 'p1', naughtyLevel: 2 }, gb: { name: 'B', ownerId: 'p2', naughtyLevel: 0 } };
  g.players!.p1.giftId = 'ga';
  g.players!.p2.giftId = 'gb';
  return g;
}

describe('admin', () => {
  it('re-owning a gift moves the new owner\'s old gift to the Misfit pile', () => {
    const g = apply(withGifts(), Admin.saveGift(withGifts(), 'ga', { naughtyLevel: 3, ownerId: 'p2' }, NOW));
    consistent(g);
    expect(g.gifts!.gb).toMatchObject({ ownerId: 'misfit', naughtyLevel: 0 });
    expect(g.players!.p1.giftId ?? null).toBeNull();
    expect(g.gifts!.ga.naughtyLevel).toBe(3);
  });
  it('Move to Misfit / delete / add keep links consistent', () => {
    let g = apply(withGifts(), Admin.moveGiftToMisfit(withGifts(), 'ga', NOW));
    consistent(g);
    expect(g.gifts!.ga.ownerId).toBe('misfit');
    g = apply(g, Admin.deleteGift(g, 'gb'));
    consistent(g);
    expect(g.gifts!.gb).toBeUndefined();
    g = apply(g, Admin.addGift(g, 'gc', 'C', 'p3', NOW));
    g = apply(g, Admin.addGift(g, 'gd', 'D', 'p3', NOW)); // p3's C goes to the pile
    consistent(g);
    expect(g.gifts!.gc.ownerId).toBe('misfit');
    expect(Admin.addGift(g, 'ge', '  ', 'misfit', NOW).error).toBeTruthy();
  });
});

describe('AI Draw Off', () => {
  function tie(goal: 'pick_one' | 'pick_two'): Game {
    const g = table(4, { tiebreakerMethod: 'draw_off', aiCriticEnabled: true, aiCriticUrl: 'http://x' });
    g.gifts = { gA: { name: 'A', ownerId: 'p1', naughtyLevel: 0 } };
    g.players!.p1.giftId = 'gA';
    g.currentTurn = {
      phase: 'minigame_draw_off', openerId: 'p1', giftId: 'gA', tiedBidders: ['p2', 'p3', 'p4'], minigameGoal: goal,
      bids: { p2: { amount: 3 }, p3: { amount: 3 }, p4: { amount: 3 } },
    };
    return g;
  }
  it('prompt → drawing → judged → the top-rated tied bidder duels the Opener', () => {
    let g = apply(tie('pick_one'), DO.startDrawOff(tie('pick_one'), ctx()));
    g = apply(g, DO.startDrawing(g, NOW));
    expect(DO.startDrawing(g, NOW).error).toBeTruthy();
    g = apply(g, DO.startJudging(g));
    g = apply(g, DO.judged(g, { p2: { rating: 40 }, p3: { rating: 90 }, p4: { rating: 60 } }, {}));
    expect((g.minigame as DO.DrawOff).winners).toEqual(['p3']);
    g = apply(g, DO.completeDrawOff(g));
    expect(g.minigame ?? null).toBeNull();
    expect(g.currentTurn).toMatchObject({ phase: 'duel', challengerId: 'p3', defenderId: 'p1', pot: 2, tax: 1 });
    expect(g.players!.p3.chips).toBe(7);
  });
  it("after a yield the critic's named winner and the next best duel each other", () => {
    let g = apply(tie('pick_two'), DO.startDrawOff(tie('pick_two'), ctx()));
    g = apply(g, DO.startDrawing(g, NOW));
    g = apply(g, DO.startJudging(g));
    g = apply(g, DO.judged(g, { p2: { rating: 40 }, p3: { rating: 90 }, p4: { rating: 60 } }, { winnerId: 'p2', text: 'Bold.' }));
    expect((g.minigame as DO.DrawOff).winners).toEqual(['p2', 'p3']);
    g = apply(g, DO.completeDrawOff(g));
    expect(g.currentTurn).toMatchObject({ isYieldDuel: true, challengerId: 'p2', defenderId: 'p3' });
  });
  it('refuses without the AI critic', () => {
    const g = tie('pick_one');
    g.settings!.aiCriticEnabled = false;
    expect(DO.startDrawOff(g, ctx()).error).toMatch(/AI Art Critic/);
  });
});
