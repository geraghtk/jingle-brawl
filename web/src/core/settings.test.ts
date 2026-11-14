// A room's settings come from whoever created it (anyone can): only real settings, each within its
// limits. An absurd one (a billion starting chips and no bid cap) once made the room list a billion bids.
import { describe, expect, it } from 'vitest';
import * as A from './actions';
import { config, DEFAULTS, LIMITS, sanitizeSettings } from './settings';
import { offers } from './table';
import { apply, ctx, NOW, table } from './testkit';
import type { Settings } from './types';

describe('settings from the network', () => {
  it('keeps what the setup screen sends, as it is', () => {
    const sent: Settings = {
      publicChips: true, maxReprisalDepth: 2, maxBid: 99, maxWinnerPayout: 3, loserDividendRule: 'everyone', spectatorBets: 1, voteWindow: 60,
      duelPick: 'wheel', duelTableGames: 'all', duelGames: 'Thumb War\nStaring Contest', duelAppGames: ['whack_elf', 'reindeer_ring_toss'], gremlinChallenges: 'quick_math',
      taxThreshold: 99, baseCost: 0, tiebreakerMethod: 'sketch_off', partyPrize: 'points', aiCriticBackend: 'gemini', startingChips: 20,
    };
    expect(sanitizeSettings(sent)).toEqual(sent);
    expect(sanitizeSettings({ duelGames: '' })).toEqual({ duelGames: '' });          // an empty list stays empty
  });

  it('clamps numbers, drops unknown keys and wrong kinds', () => {
    const s = sanitizeSettings({
      startingChips: 1e9, maxBid: 1e9, maxWinnerPayout: -5, voteWindow: Infinity, taxThreshold: '2', baseCost: 2.9, spectatorBets: NaN,
      duelPick: 'whenever', loserDividendRule: { toString: 1 }, publicChips: 'yes', aiCriticUrl: 'http://example.test/', junk: 'x'.repeat(100000),
      __proto__: { maxBid: 7 }, duelAppGames: ['whack_elf', 'whack_elf', '__proto__', 5, 'x'.repeat(200)], duelGames: ['Thumb War', 7, 'y'.repeat(500)],
    });
    expect(s).toEqual({
      startingChips: LIMITS.startingChips[1], maxBid: LIMITS.maxBid[1], maxWinnerPayout: 0, baseCost: 2,
      duelAppGames: ['whack_elf'], duelGames: `Thumb War\n${'y'.repeat(80)}`,
    });
    for (const junk of [null, undefined, 7, 'x', [1, 2], () => 1]) expect(sanitizeSettings(junk)).toEqual({});
  });

  it('a long list of duels is cut to a sensible one', () => {
    const many = Array.from({ length: 500 }, (_, i) => `Game ${i}`);
    expect((sanitizeSettings({ duelGames: many }).duelGames as string).split('\n')).toHaveLength(30);
    expect((sanitizeSettings({ duelGames: many.join('\n') }).duelGames as string).split('\n').length).toBeLessThanOrEqual(30);
    expect(sanitizeSettings({ duelAppGames: Array.from({ length: 500 }, (_, i) => `game_${i}`) }).duelAppGames).toHaveLength(20);
  });

  it('config() reads numbers within the limits too (a room stored before settings were checked)', () => {
    const cfg = config({ startingChips: 1e9, maxBid: 1e9, maxWinnerPayout: 1e9, taxThreshold: -3, voteWindow: 1e9, maxReprisalDepth: 1e9 } as Settings);
    expect(cfg).toMatchObject({ startingChips: 99, maxBid: 99, maxWinnerPayout: 99, taxThreshold: 0, voteWindow: 600, maxReprisalDepth: 10 });
    expect(config({ maxBid: 'lots' } as unknown as Settings).maxBid).toBe(DEFAULTS.maxBid);
    expect(config({ baseCost: 0, taxThreshold: 0 })).toMatchObject({ baseCost: 0, taxThreshold: 0 });   // 0 is a real value
    expect(config({}).startingChips).toBeNull();
  });

  it('a phone is never offered more bids than the limit allows, whatever the room was created with', () => {
    let g = table(3, { startingChips: 1e9, maxBid: 1e9 } as Settings);
    for (const p of Object.values(g.players!)) p.chips = 9999;          // the most the host's console can set
    g = apply(g, A.drawGift(g, 'Gift A', 'gA', ctx(1)));
    const bidder = Object.keys(g.players!).find((id) => id !== g.currentTurn!.openerId)!;
    const bids = offers(g, bidder, NOW).filter((o) => o.type === 'bid');
    expect(bids.length).toBe(LIMITS.maxBid[1] + 1);                      // 0 to 99
  });
});
