// Admin edits and the AI Draw Off flow keep gifts and players consistent.
import { describe, expect, it } from 'vitest';
import * as Admin from './admin';
import { apply, NOW, table } from './testkit';
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
