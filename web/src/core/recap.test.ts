// The end-of-night recap: the podium follows the standings, each award has one rule, nothing crashes.
import { describe, expect, it } from 'vitest';
import { projectGame } from './projection';
import { MAX_AWARDS, awardsOf, ordinal, recapOf } from './recap';
import { partyStandings } from './scoring';
import { table } from './testkit';
import type { Game, Player } from './types';

/** A finished table: p1..pN with the given numbers (everything else as the test kit deals it). */
function ended(stats: Partial<Player>[], gifts: Record<string, [owner: string, naughty: number]> = {}): Game {
  const g = table(stats.length);
  g.phase = 'ended';
  stats.forEach((s, i) => Object.assign(g.players![`p${i + 1}`], s));
  for (const [name, [owner, naughty]] of Object.entries(gifts)) {
    g.gifts![`g_${name}`] = { name, ownerId: owner, naughtyLevel: naughty };
    if (owner !== 'misfit') g.players![owner].giftId = `g_${name}`;
  }
  return g;
}
const award = (g: Game, id: string) => recapOf(g).awards.find((a) => a.id === id);
const all = (g: Game) => recapOf(g).awards.map((a) => [a.id, a.winner]);

describe('the podium', () => {
  it('is the top three of the standings, in the standings\' own order', () => {
    const g = ended([{ partyPoints: 1 }, { partyPoints: 5 }, { votingWins: 3 }, { duelsFought: 2 }, {}]);
    const r = recapOf(g);
    expect(r.scored).toBe(true);
    expect(r.podium.map((s) => [s.place, s.players.map((p) => p.id)])).toEqual([[1, ['p2']], [2, ['p3']], [3, ['p4']]]);
    expect(r.lines.map((l) => [l.id, l.place, l.score])).toEqual(partyStandings(g.players).map((s) => [s.id, s.place, s.score]));
  });

  it('ties share a step as they share a place: 1, 2, 2 has no third step', () => {
    const r = recapOf(ended([{ partyPoints: 4 }, { partyPoints: 2 }, { votingWins: 2 }, { partyPoints: 1 }]));
    expect(r.podium.map((s) => [s.place, s.players.map((p) => p.id)])).toEqual([[1, ['p1']], [2, ['p2', 'p3']]]);
    expect(r.lines.map((l) => l.place)).toEqual([1, 2, 2, 4]);
  });

  it('joint winners stand on the top step together, the next is third', () => {
    const r = recapOf(ended([{ partyPoints: 3 }, { duelsFought: 3 }, { votingWins: 1 }]));
    expect(r.podium.map((s) => [s.place, s.players.length])).toEqual([[1, 2], [3, 1]]);
  });

  it('nobody stands on it without a point, and with no points at all there are no places', () => {
    expect(recapOf(ended([{ partyPoints: 2 }, {}, {}])).podium.map((s) => s.players.length)).toEqual([1]);
    const none = recapOf(ended([{}, {}, {}]));
    expect(none.scored).toBe(false);
    expect(none.podium).toEqual([]);
    expect(none.lines.map((l) => l.place)).toEqual([0, 0, 0]);
  });
});

describe('everyone\'s final line', () => {
  it('has the gift they went home with, their avatar and their chips', () => {
    const g = ended([{ partyPoints: 1, avatar: 'fox', chips: 4 }, {}], { Socks: ['p1', 3], Mug: ['misfit', 0] });
    const [a, b] = recapOf(g).lines;
    expect(a).toMatchObject({ id: 'p1', name: 'P1', avatar: 'fox', chips: 4, place: 1, score: 1, gift: { name: 'Socks', naughty: 3 } });
    expect(b.gift).toBeNull();
  });

  it('is the same from a phone\'s view of the finished game as from the host\'s', () => {
    const g = ended([{ partyPoints: 2, chips: 3, duelWins: 2, duelsFought: 2 }, { votingWins: 2, chips: 9 }, { chips: 6 }], { Socks: ['p1', 2] });
    expect(recapOf(projectGame(g, { kind: 'phone', pid: 'p3' }))).toEqual(recapOf(projectGame(g, { kind: 'host' })));
    expect(recapOf(projectGame(g, { kind: 'tv' }))).toEqual(recapOf(g));
  });
});

describe('the awards', () => {
  it('Duel Champion: the most duels won, alone at the top', () => {
    expect(award(ended([{ duelWins: 2 }, { duelWins: 1 }, {}]), 'duel_champ')).toMatchObject({ winner: 'p1', title: 'Duel Champion', detail: 'Won 2 duels' });
    expect(award(ended([{ duelWins: 1 }, {}, {}]), 'duel_champ')?.detail).toBe('Won 1 duel');
    expect(award(ended([{ duelWins: 2 }, { duelWins: 2 }, {}]), 'duel_champ')).toBeUndefined();   // a tie: nobody clearly earned it
    expect(award(ended([{}, {}, {}]), 'duel_champ')).toBeUndefined();                              // a tie at zero
  });

  it('Crystal Ball: the most duels called right, at least two', () => {
    expect(award(ended([{ votingWins: 3 }, { votingWins: 2 }]), 'crystal_ball')).toMatchObject({ winner: 'p1', detail: 'Called 3 duels right' });
    expect(award(ended([{ votingWins: 1 }, {}]), 'crystal_ball')).toBeUndefined();
    expect(award(ended([{ votingWins: 2 }, { votingWins: 2 }]), 'crystal_ball')).toBeUndefined();
  });

  it('Life of the Party: the most points from the mini-games', () => {
    expect(award(ended([{ partyPoints: 1 }, { partyPoints: 4 }]), 'party_star')).toMatchObject({ winner: 'p2', detail: '4 points from the party games' });
    expect(award(ended([{ partyPoints: 3 }, { partyPoints: 3 }]), 'party_star')).toBeUndefined();
  });

  it('Top of the Naughty List: holds the naughtiest gift (a gift on the Misfit pile is nobody\'s)', () => {
    const g = ended([{}, {}, {}], { Socks: ['p1', 2], Candle: ['p2', 4], Mug: ['misfit', 5] });
    expect(award(g, 'naughty_list')).toMatchObject({ winner: 'p2', detail: 'Took home the most fought-over gift' });
    expect(award(ended([{}, {}], { Socks: ['p1', 3], Candle: ['p2', 3] }), 'naughty_list')).toBeUndefined();
    expect(award(ended([{}, {}], { Socks: ['p1', 0] }), 'naughty_list')).toBeUndefined();
  });

  it('Jingle Brawler: the most duels fought, at least two', () => {
    expect(award(ended([{ duelsFought: 4 }, { duelsFought: 3 }]), 'brawler')).toMatchObject({ winner: 'p1', detail: 'Fought 4 duels' });
    expect(award(ended([{ duelsFought: 1 }, {}]), 'brawler')).toBeUndefined();
  });

  it('Big Spender: the fewest chips left, alone at the bottom', () => {
    expect(award(ended([{ chips: 2 }, { chips: 5 }, { chips: 5 }]), 'big_spender')).toMatchObject({ winner: 'p1', detail: 'Only 2 chips left' });
    expect(award(ended([{ chips: 0 }, { chips: 5 }]), 'big_spender')?.detail).toBe('Spent every last chip');
    expect(award(ended([{ chips: 1 }, { chips: 1 }, { chips: 5 }]), 'big_spender')).toBeUndefined();
    expect(award(ended([{}, {}, {}]), 'big_spender')).toBeUndefined();                             // nobody spent a chip
  });

  it('Good Sport: the most duels lost, at least two', () => {
    expect(award(ended([{ duelLosses: 3, duelsFought: 3 }, { duelLosses: 1 }]), 'good_sport')).toMatchObject({ winner: 'p1', detail: 'Lost 3 duels and kept on smiling' });
    expect(award(ended([{ duelLosses: 1 }, {}]), 'good_sport')).toBeUndefined();
  });

  it('Silent Night: the only one at a table of three or more who never duelled', () => {
    expect(award(ended([{ duelsFought: 1 }, { duelsFought: 2 }, {}]), 'silent_night')).toMatchObject({ winner: 'p3' });
    expect(award(ended([{ duelsFought: 1 }, {}, {}]), 'silent_night')).toBeUndefined();
    expect(award(ended([{ duelsFought: 1 }, {}]), 'silent_night')).toBeUndefined();
  });

  it('Deep Pockets: the most chips left, alone at the top', () => {
    expect(award(ended([{ chips: 12 }, { chips: 5 }, { chips: 5 }]), 'deep_pockets')).toMatchObject({ winner: 'p1', detail: '12 chips still in the stocking' });
    expect(award(ended([{ chips: 5 }, { chips: 5 }, { chips: 2 }]), 'deep_pockets')).toBeUndefined();
  });

  it('one each while they last, and never more than five', () => {
    // p1 earns four on their own; the rest of the table earns one each
    const g = ended([
      { duelWins: 4, duelsFought: 5, votingWins: 4, partyPoints: 6, chips: 4 }, { duelLosses: 3, duelsFought: 3, chips: 0 },
      { chips: 15 }, { duelsFought: 1, chips: 5 }, { duelsFought: 1, chips: 5 },
    ], { Socks: ['p4', 4] });
    const got = all(g);
    expect(got).toEqual([['duel_champ', 'p1'], ['naughty_list', 'p4'], ['big_spender', 'p2'], ['silent_night', 'p3']]);
    expect(new Set(got.map(([, w]) => w)).size).toBe(got.length);
    expect(got.length).toBeLessThanOrEqual(MAX_AWARDS);
  });

  it('a second award for someone only to reach three, never a third', () => {
    const g = ended([{ duelWins: 3, duelsFought: 4, votingWins: 3, partyPoints: 5 }, { duelsFought: 4 }]);
    expect(all(g)).toEqual([['duel_champ', 'p1'], ['crystal_ball', 'p1']]);
    const lines = recapOf(g).lines;
    expect(awardsOf(g, lines).every((a) => a.winner === 'p1')).toBe(true);
  });

  it('a quiet game gives none', () => {
    expect(recapOf(ended([{}, {}, {}, {}])).awards).toEqual([]);
  });

  it('no award says anything with an em dash', () => {
    const g = ended([{ duelWins: 2, duelsFought: 2 }, { votingWins: 3 }, { partyPoints: 3 }, { chips: 1 }, { duelLosses: 2, duelsFought: 2 }], { Socks: ['p4', 2] });
    for (const a of recapOf(g).awards) expect(a.title + a.detail).not.toMatch(/—/);
  });
});

describe('nothing to recap', () => {
  it('an empty game, a missing one and a table of one do not crash', () => {
    expect(recapOf(null)).toEqual({ scored: false, lines: [], podium: [], awards: [], drawings: [] });
    expect(recapOf({ phase: 'ended' })).toMatchObject({ lines: [], podium: [], awards: [] });
    const one = recapOf(ended([{ partyPoints: 2, duelWins: 1 }]));
    expect(one.podium).toHaveLength(1);
    expect(one.lines).toHaveLength(1);
  });

  it('a player whose gift is missing from the view still has a line', () => {
    const g = ended([{ partyPoints: 1, giftId: 'gone' }, {}]);
    expect(recapOf(g).lines[0].gift).toEqual({ name: 'Mystery gift', naughty: 0 });
  });
});

describe('Sketch-Off drawings', () => {
  it('are listed only if the finished game still holds real pictures', () => {
    const g = ended([{ partyPoints: 1 }, {}]);
    expect(recapOf(g).drawings).toEqual([]);
    g.minigame = { type: 'sketch_off', phase: 'results', seed: 1, participants: ['p1', 'p2'], tiedBidders: [], minigameGoal: 'pick_one', prize: 'points', startedAt: 0, phaseEndsAt: 0,
      prompt: 'A snowman at the beach', submissions: { p1: 'data:image/png;base64,AAAA', p2: 'data:,' } };
    expect(recapOf(g)).toMatchObject({ drawings: [{ id: 'p1', name: 'P1', img: 'data:image/png;base64,AAAA' }], prompt: 'A snowman at the beach' });
  });
});

it('ordinal', () => {
  expect([1, 2, 3, 4, 11, 12, 13, 21, 22].map(ordinal)).toEqual(['1st', '2nd', '3rd', '4th', '11th', '12th', '13th', '21st', '22nd']);
});
