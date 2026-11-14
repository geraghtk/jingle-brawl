// Anyone can create a room: what it stores as settings is only real settings, within their limits
// (a billion starting chips once made the room list a billion bids and took the whole server down).
import { describe, expect, it } from 'vitest';
import { LIMITS } from '../../../web/src/core/settings';
import type { Settings } from '../../../web/src/core/types';
import { MemoryStore } from '../store/store';
import { FakeClock } from '../testing/fakeClock';
import { Rooms } from './rooms';

describe('a new room’s settings', () => {
  it('are stored whitelisted and clamped, whatever was sent', () => {
    const store = new MemoryStore();
    const rooms = new Rooms(store, new FakeClock(), { secret: 'test' });
    const sent = { startingChips: 1e9, maxBid: 1e9, duelPick: 'wheel', junk: 'x'.repeat(50000), nested: { a: [1, 2, 3] } } as unknown as Settings;
    const { code } = rooms.create(sent);
    expect(store.load(code)!.game.settings).toEqual({ startingChips: LIMITS.startingChips[1], maxBid: LIMITS.maxBid[1], duelPick: 'wheel' });
    rooms.close();
  });
});
