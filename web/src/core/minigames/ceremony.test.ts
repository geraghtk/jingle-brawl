import { describe, expect, it } from 'vitest';
import { ceremonyFor, myEnding } from './ceremony';
import type { GameResult } from './contracts';

const name = (id: string) => ({ a: 'Ann', b: 'Bob', c: 'Cy', d: 'Di', e: 'Ed' } as Record<string, string>)[id] || id;
const base = { gameId: 'g', instanceId: 'i', reason: 'first over the line' };

describe('the results ceremony: one announcement for every game', () => {
  it('a team game: the captain names the team, every winner is a star with the points they earned', () => {
    const r: GameResult = { ...base, status: 'resolved', mode: 'captain_teams', winners: ['c', 'a', 'd'], losers: ['b', 'e'],
      duel: { duelists: ['a', 'b'], winningDuelistId: 'a', losingDuelistId: 'b' }, rewards: ['a', 'c', 'd'].map((playerId) => ({ playerId, points: 1 })) };
    const c = ceremonyFor(r, name)!;
    expect(c).toMatchObject({ tone: 'win', headline: "Ann's team wins!", sub: 'First over the line' });
    expect(c.stars.map((s) => s.id)).toEqual(['a', 'c', 'd']);                // the captain first
    expect(c.stars.every((s) => s.points === 1)).toBe(true);
    expect(c.also).toEqual([]);
    expect(myEnding(r, 'd', ['a', 'b', 'c', 'd', 'e'])).toBe('won');
    expect(myEnding(r, 'e', ['a', 'b', 'c', 'd', 'e'])).toBe('lost');
    expect(myEnding(r, 'zed', ['a', 'b', 'c', 'd', 'e'])).toBe('watched');
  });

  it('a duel with the table playing along: the duel winner is the star, bystanders who earned points are listed after, scores best first', () => {
    const r: GameResult = { ...base, status: 'resolved', mode: 'party_with_duelists', winners: ['b'], losers: ['a'],
      duel: { duelists: ['a', 'b'], winningDuelistId: 'b', losingDuelistId: 'a', resolutionReason: 'more points' },
      rewards: [{ playerId: 'd', points: 2 }, { playerId: 'c', points: 3 }],
      performance: [{ playerId: 'a', score: 4 }, { playerId: 'b', score: 9 }, { playerId: 'c', score: 12 }, { playerId: 'd', score: 7 }],
      placements: [{ playerId: 'c', rank: 1 }, { playerId: 'b', rank: 2 }, { playerId: 'd', rank: 3 }, { playerId: 'a', rank: 4 }] };
    const c = ceremonyFor(r, name)!;
    expect(c).toMatchObject({ headline: 'Bob wins!', sub: 'More points' });
    expect(c.also.map((x) => [x.id, x.points])).toEqual([['c', 3], ['d', 2]]);
    expect(c.scores.map((x) => x.id)).toEqual(['c', 'b', 'd', 'a']);
    // a practice round pays nothing, so it promises nothing
    expect(ceremonyFor(r, name, true)!.also).toEqual([]);
  });

  it('nobody to pick: a tie goes to the host, a no contest says so, a cancelled game is not announced', () => {
    const tie: GameResult = { ...base, status: 'requires_tiebreaker', mode: 'direct_duel', winners: [], losers: [], duel: { duelists: ['a', 'b'], resolutionReason: 'a tie' } };
    expect(ceremonyFor(tie, name)).toMatchObject({ tone: 'level', headline: 'Too close to call!', sub: 'The host decides the duel.', stars: [] });
    expect(myEnding(tie, 'a', ['a', 'b'])).toBe('level');
    expect(ceremonyFor({ ...tie, status: 'no_contest', reason: 'the host called it a no contest', duel: undefined }, name)).toMatchObject({ headline: 'No contest', sub: 'The host called it a no contest' });
    expect(ceremonyFor({ ...tie, status: 'cancelled' }, name)).toBeNull();
    expect(ceremonyFor(null, name)).toBeNull();
  });

  it('a free-for-all with one loser invents no champion: it says who is out, and the rest were safe', () => {
    const r: GameResult = { ...base, status: 'resolved', mode: 'free_for_all', winners: [], losers: ['c'], reason: 'holding the Gremlin when it went berserk' };
    expect(ceremonyFor(r, name)).toMatchObject({ tone: 'over', headline: 'Cy is out!', sub: 'Holding the Gremlin when it went berserk' });
    expect(myEnding(r, 'a', ['a', 'b', 'c'])).toBe('safe');
    expect(myEnding(r, 'c', ['a', 'b', 'c'])).toBe('lost');
  });

  it('several winners with no captain are named together, a crowd is a tie at the top', () => {
    const r: GameResult = { ...base, status: 'resolved', mode: 'free_for_all', winners: ['a', 'b'], losers: [] };
    expect(ceremonyFor(r, name)!.headline).toBe('Ann and Bob win!');
    expect(ceremonyFor({ ...r, winners: ['a', 'b', 'c', 'd'] }, name)!.headline).toBe('A tie at the top!');
  });
});
