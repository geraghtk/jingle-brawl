import { describe, expect, it } from 'vitest';
import { ceremonyFor, myEnding, myPartyEnding, partyCeremony } from './ceremony';
import type { PartyGame } from '../party';
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
    expect(c).toMatchObject({ headline: 'Bob wins the duel!', sub: 'More points' });   // the table played too: say which contest this is
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
    expect(ceremonyFor(r, name)).toMatchObject({ tone: 'over', headline: 'Cy is out!', sub: 'Cy was holding the Gremlin when it went berserk' });
    expect(myEnding(r, 'a', ['a', 'b', 'c'])).toBe('safe');
    expect(myEnding(r, 'c', ['a', 'b', 'c'])).toBe('lost');
    // in a duel the same reason is said of the loser by name (under "Bob wins!" it read as if Bob had been holding it)
    const duel: GameResult = { ...r, mode: 'direct_duel', winners: ['b'], losers: ['a'], duel: { duelists: ['a', 'b'], winningDuelistId: 'b', losingDuelistId: 'a', resolutionReason: r.reason } };
    expect(ceremonyFor(duel, name)).toMatchObject({ headline: 'Bob wins!', sub: 'Ann was holding the Gremlin when it went berserk' });
    // two scores are only the duel again, and a board of noughts says nothing: no table for either
    expect(ceremonyFor({ ...duel, performance: [{ playerId: 'a', score: 1 }, { playerId: 'b', score: 2 }] }, name)!.scores).toEqual([]);
    expect(ceremonyFor({ ...duel, performance: ['a', 'b', 'c'].map((playerId) => ({ playerId, score: 0 })) }, name)!.scores).toEqual([]);
  });

  it('several winners with no captain are named together, a crowd is a tie at the top', () => {
    const r: GameResult = { ...base, status: 'resolved', mode: 'free_for_all', winners: ['a', 'b'], losers: [] };
    expect(ceremonyFor(r, name)!.headline).toBe('Ann and Bob win!');
    expect(ceremonyFor({ ...r, winners: ['a', 'b', 'c', 'd'] }, name)!.headline).toBe('A tie at the top!');
  });
});

// The two party games end on the same ceremony (ui/Ceremony.tsx PartyCeremony): who won the tie and whom they
// duel next, or the duel itself, the Party points, everyone's score.
describe('the results of a party game as the ceremony', () => {
  const party = (over: Partial<PartyGame> = {}): PartyGame => ({
    type: 'whack_elf', phase: 'results', seed: 7, participants: ['a', 'b', 'c', 'd'], tiedBidders: ['a', 'b'], minigameGoal: 'pick_one', prize: 'points_chips', startedAt: 0, phaseEndsAt: 9000,
    results: { ranking: ['c', 'b', 'a', 'd'], scoreOf: { c: 30, b: 22, a: 9 }, points: { c: 3, b: 2, a: 1 }, bonus: 'c', winners: ['b'] }, ...over,
  });
  it('a tie: the best-placed tied bidder wins it and goes on to duel the Opener; the table is ranked with its points', () => {
    const c = partyCeremony(party(), name, { opener: 'Di', gift: 'the Fuzzy socks' })!;
    expect(c).toMatchObject({ tone: 'win', headline: 'Bob wins the tie!', sub: 'Next: a duel with Di for the Fuzzy socks.', unit: ['pt', 'pts'], also: [] });
    expect(c.stars).toEqual([{ id: 'b', points: 2, score: 22 }]);
    expect(c.scores.map((r) => [r.id, r.rank, r.score, r.points, r.tag])).toEqual([
      ['c', 1, 30, 3, '+1 chip Elf Bonus'], ['b', 2, 22, 2, undefined], ['a', 3, 9, 1, 'tied'], ['d', 4, undefined, 0, undefined]]);
    expect(partyCeremony(party(), name)!.sub).toBe('Next: a duel with the Opener for the gift.');
    expect(myPartyEnding(party(), 'b')).toBe('won');
    expect(myPartyEnding(party(), 'a')).toBe('lost');
    expect(myPartyEnding(party(), 'c')).toBe('watched');       // not in the tie: played for Party points
  });
  it('after a yield two go through and duel each other', () => {
    const mg = party({ minigameGoal: 'pick_two', results: { ...party().results!, winners: ['b', 'a'] } });
    expect(partyCeremony(mg, name, { gift: 'A candle' })).toMatchObject({ headline: 'Bob and Ann win the tie!', sub: 'Now they duel each other for A candle.' });
  });
  it('deciding a duel: it says who won the duel (what they take comes with the stakes)', () => {
    const c = partyCeremony(party({ minigameGoal: 'duel' }), name)!;
    expect(c).toMatchObject({ headline: 'Bob wins the duel!', sub: '' });
    expect(c.scores.find((r) => r.id === 'a')!.tag).toBeUndefined();   // no "tied" in a duel
  });
  it('the Game Lab: who came top, and nothing promised (no points, no bonus)', () => {
    const c = partyCeremony(party({ minigameGoal: 'lab', type: 'sketch_off' }), name)!;
    expect(c).toMatchObject({ tone: 'win', headline: 'Cy comes top!', unit: ['vote', 'votes'] });
    expect([...c.stars, ...c.scores].every((r) => r.points === 0 && !r.tag)).toBe(true);
    expect(partyCeremony(party({ minigameGoal: 'lab', results: { ranking: ['a', 'b'], scoreOf: {}, points: {}, bonus: null, winners: [] } }), name)).toMatchObject({ tone: 'over', headline: 'Nobody scored!' });
    expect(myPartyEnding(party({ minigameGoal: 'lab' }), 'c')).toBe('won');
  });
  it('nothing until the results are in; no em dash in what it says', () => {
    expect(partyCeremony(party({ results: undefined }), name)).toBeNull();
    expect(partyCeremony(null, name)).toBeNull();
    for (const goal of ['pick_one', 'pick_two', 'duel', 'lab'] as const) expect(JSON.stringify(partyCeremony(party({ minigameGoal: goal }), name))).not.toMatch(/—/);
  });
});
