// Which duel gets played: the defender's pick or the wheel, then (for an in-app game) the game decides.
import { describe, expect, it } from 'vitest';
import * as A from './actions';
import * as D from './duel';
import * as G from './duelgame';
import { results } from './party';
import type { PartyGame } from './party';
import { advanceParty } from './partyflow';
import { rng } from './rng';
import { apply, NOW, table } from './testkit';
import type { Game, Settings } from './types';

/** p1 (defender) vs p2 (challenger) for gA; p3, p4 watch. */
function duel(settings: Settings = {}, extra: Partial<NonNullable<Game['currentTurn']>> = {}): Game {
  const g = table(4, { duelPick: 'defender', ...settings });
  g.gifts = { gA: { name: 'Gift A', ownerId: 'p1', naughtyLevel: 0 } };
  g.players!.p1.giftId = 'gA';
  g.currentTurn = { phase: 'duel', challengerId: 'p2', defenderId: 'p1', giftId: 'gA', pot: 2, ...extra };
  return g;
}
const ctx = (seed = 3) => ({ now: NOW, rng: rng(seed) });

describe('duel game: what gets played', () => {
  it('the options are the host\'s list plus the in-app games they ticked', () => {
    const g = duel({ duelGames: ['Thumb War', '  ', 'Staring Contest'], duelAppGames: ['whack_elf'] });
    expect(G.duelOptions(g)).toEqual([
      { name: 'Thumb War', kind: 'table' }, { name: 'Staring Contest', kind: 'table' },
      { name: 'Whack-an-Elf', kind: 'app', app: 'whack_elf' },
    ]);
  });

  it('a tiebreak duel never spins an in-app game (its winner goes on to the real duel, it is not settled like one)', () => {
    const g = duel({ duelGames: 'Thumb War', duelAppGames: ['whack_elf'] }, { isTiebreak: true });
    expect(G.duelOptions(g).map((o) => o.name)).toEqual(['Thumb War']);
  });

  it('the list is stored as text, so an emptied list stays empty (RTDB drops empty arrays → defaults)', () => {
    expect(G.duelOptions(duel({ duelGames: 'Thumb War\n\nArm Wrestle ' }))).toEqual([
      { name: 'Thumb War', kind: 'table' }, { name: 'Arm Wrestle', kind: 'table' },
    ]);
    expect(G.duelOptions(duel({ duelGames: '', duelAppGames: ['whack_elf'] })).map((o) => o.name)).toEqual(['Whack-an-Elf']);
  });

  it('a normal duel: the defender picks — nobody else can, and votes wait for the pick', () => {
    let g = duel({ duelGames: ['Thumb War', 'Arm Wrestle'] });
    expect(G.duelGameStep(g)).toBe('picker');
    expect(G.chooseDuelGame(g, 'p2', 0).error).toBe('Not your pick'); // the challenger
    expect(A.voteOnDuel(g, 'p3', 'p1', NOW).error).toBe('Wait until the duel is picked');
    g = apply(g, G.chooseDuelGame(g, 'p1', 1));
    expect(g.currentTurn?.duelGame).toEqual({ name: 'Arm Wrestle', kind: 'table', via: 'picker' });
    expect(G.duelGameStep(g)).toBeNull();
    expect(A.voteOnDuel(g, 'p3', 'p1', NOW).error).toBeUndefined();
  });

  it('a yield duel or a tiebreak has nobody really defending: the wheel picks', () => {
    expect(G.duelGameStep(duel({}, { isYieldDuel: true }))).toBe('wheel');
    expect(G.duelGameStep(duel({}, { isTiebreak: true }))).toBe('wheel');
    expect(G.duelGameStep(duel({ duelPick: 'wheel' }))).toBe('wheel');
    expect(G.chooseDuelGame(duel({}, { isYieldDuel: true }), 'p1', 0).error).toBe('Not your pick');
  });

  it('the wheel records every option and where it landed; the defender may ask for a spin', () => {
    const g0 = duel({ duelGames: ['A', 'B', 'C'] });
    const plan = G.spinDuelGame(g0, ctx(), 'p1');
    const g = apply(g0, plan);
    const w = g.currentTurn!.wheel!;
    expect(w.options.map((o) => o.name)).toEqual(['A', 'B', 'C']);
    expect(g.currentTurn?.duelGame).toEqual({ ...w.options[w.index], via: 'wheel' });
    expect(w.spunAt).toBe(NOW);
    expect(G.spinDuelGame(g0, ctx(), 'p3').error).toBe('Not your pick');
  });

  it('a pick and a spin racing: only the first lands (both claim the duel game)', () => {
    const g0 = duel({ duelGames: ['A', 'B'] });
    const g1 = apply(g0, G.chooseDuelGame(g0, 'p1', 0));
    const g2 = apply(g1, G.spinDuelGame(g0, ctx())); // built from the old state, applied after
    expect(g2.currentTurn?.duelGame?.via).toBe('picker');
    expect(g2.currentTurn?.wheel).toBeUndefined();
  });

  it('"off": people agree on a duel themselves (no pick, no wheel)', () => {
    expect(G.duelGameStep(duel({ duelPick: 'off' }))).toBeNull();
  });

  it('an in-app duel game: everyone plays, the better-placed duellist wins, the duel resolves with them', () => {
    let g = duel({ duelGames: [], duelAppGames: ['whack_elf'], voteWindow: 0 });
    g = apply(g, G.spinDuelGame(g, ctx()));
    expect(g.currentTurn?.duelGame?.app).toBe('whack_elf');
    g = apply(g, G.startDuelApp(g, ctx()));
    expect(g.currentTurn?.phase).toBe('minigame_party');
    const mg = g.minigame as PartyGame;
    expect(mg).toMatchObject({ minigameGoal: 'duel', tiedBidders: ['p2', 'p1'], participants: ['p1', 'p2', 'p3', 'p4'] });
    g = apply(g, advanceParty(g, 'intro', NOW));
    g.minigame = { ...(g.minigame as PartyGame), scores: { p1: { score: 9 }, p2: { score: 4 }, p3: { score: 12 }, p4: { score: 1 } } };
    expect(results(g.minigame as PartyGame).winners).toEqual(['p1']); // p3 scored most but isn't duelling
    g = apply(g, advanceParty(g, 'playing', NOW));
    g = apply(g, advanceParty(g, 'results', NOW));
    expect(g.currentTurn).toMatchObject({ phase: 'duel', duelGameWinner: 'p1' });
    expect(G.duelAppWinner(g.currentTurn)).toBe('p1');
    g = apply(g, D.resolveDuel(g, 'p1', NOW));
    expect(g.lastDuel).toMatchObject({ winnerId: 'p1', loserId: 'p2' });
    expect(g.players!.p3.partyPoints).toBe(3); // the rest of the table still played for points
  });

  describe('the Under-16 Exception (V2 rules): a duellist under 16 always picks the duel', () => {
    const young = (g: Game, ages: Record<string, number>) => {
      for (const [id, age] of Object.entries(ages)) g.players![id].age = age;
      return g;
    };
    it('an under-16 challenger picks instead of the defender', () => {
      const g = young(duel({ duelGames: 'Thumb War' }), { p2: 12 });
      expect(G.duelPicker(g)).toBe('p2');
      expect(G.chooseDuelGame(g, 'p1', 0).error).toBe('Not your pick');
      expect(apply(g, G.chooseDuelGame(g, 'p2', 0)).currentTurn?.duelGame).toMatchObject({ name: 'Thumb War', via: 'picker' });
    });
    it('both under 16: the youngest picks (a tie: the defender)', () => {
      expect(G.duelPicker(young(duel(), { p1: 14, p2: 9 }))).toBe('p2');
      expect(G.duelPicker(young(duel(), { p1: 10, p2: 10 }))).toBe('p1');
    });
    it('it overrides the wheel too: "always spin", yield duels and tiebreaks', () => {
      expect(G.duelPicker(young(duel({ duelPick: 'wheel' }), { p1: 11 }))).toBe('p1');
      expect(G.duelGameStep(young(duel({}, { isYieldDuel: true }), { p2: 8 }))).toBe('picker');
      expect(G.duelGameStep(young(duel({}, { isTiebreak: true }), { p1: 13 }))).toBe('picker');
    });
    it('the young picker may still hand it to the wheel', () => {
      const g = young(duel({ duelGames: 'A\nB' }, { isYieldDuel: true }), { p2: 8 });
      expect(G.spinDuelGame(g, ctx(), 'p1').error).toBe('Not your pick');
      expect(apply(g, G.spinDuelGame(g, ctx(), 'p2')).currentTurn?.duelGame?.via).toBe('wheel');
    });
    it('16 and over is not "young"; nobody young → the standard rules', () => {
      expect(G.duelPicker(young(duel(), { p2: 16 }))).toBe('p1');
      expect(G.duelGameStep(duel({}, { isYieldDuel: true }))).toBe('wheel');
    });
    it('joining: an age is only kept for under-16s', () => {
      let g: Game = { phase: 'lobby', settings: {} };
      g = apply(g, A.joinPlayer(g, { playerId: 'k', name: 'Kid', uid: 'u', now: NOW, age: 9 }));
      g = apply(g, A.joinPlayer(g, { playerId: 'a', name: 'Adult', uid: 'u', now: NOW, age: 40 }));
      expect(g.players!.k.age).toBe(9);
      expect(g.players!.a.age).toBeUndefined();
    });
  });

  it('an in-app duel game waits for the votes before it starts', () => {
    let g = duel({ duelGames: [], duelAppGames: ['whack_elf'] });
    g = apply(g, G.spinDuelGame(g, ctx()));
    g.currentTurn!.voteEndsAt = NOW + 30000;
    expect(G.startDuelApp(g, ctx()).error).toBe('Waiting for votes');
  });
});
