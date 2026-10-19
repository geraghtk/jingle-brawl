// The duel voting window: the host can't call a duel while the table is still voting.
import { describe, expect, it } from 'vitest';
import * as A from './actions';
import * as D from './duel';
import { apply, NOW, table } from './testkit';
import type { Game, Settings } from './types';
import * as V from './voting';

/** p1 (defender) vs p2 (challenger) for gA; p3, p4 watch. */
function duel(settings: Settings = {}): Game {
  const g = table(4, settings);
  g.gifts = { gA: { name: 'Gift A', ownerId: 'p1', naughtyLevel: 0 } };
  g.players!.p1.giftId = 'gA';
  g.currentTurn = { phase: 'duel', challengerId: 'p2', defenderId: 'p1', giftId: 'gA', pot: 2 };
  return g;
}

describe('duel voting window', () => {
  it('the host has to wait from the moment the duel starts (before the window is even open)', () => {
    const s = V.voteStatus(duel(), NOW);
    expect(s).toMatchObject({ eligible: ['p3', 'p4'], waiting: ['p3', 'p4'], accepting: true, blocking: true });
  });

  it('once open, the duel cannot be called until every spectator has voted', () => {
    let g = apply(duel(), V.openVoting(duel(), NOW));
    expect(g.currentTurn?.voteEndsAt).toBe(NOW + 30000);
    expect(D.resolveDuel(g, 'p1', NOW + 1000).error).toMatch(/Waiting for 2 votes/);
    g = apply(g, A.voteOnDuel(g, 'p3', 'p1', NOW + 2000));
    expect(D.resolveDuel(g, 'p1', NOW + 3000).error).toMatch(/Waiting for 1 vote$/);
    g = apply(g, A.voteOnDuel(g, 'p4', 'p2', NOW + 4000));
    expect(V.voteStatus(g, NOW + 5000).blocking).toBe(false);
    expect(D.resolveDuel(g, 'p1', NOW + 5000).error).toBeUndefined();
  });

  it('the window running out unlocks the duel, and late votes are refused', () => {
    const g = apply(duel(), V.openVoting(duel(), NOW));
    expect(D.resolveDuel(g, 'p1', NOW + 30001).error).toBeUndefined();
    expect(A.voteOnDuel(g, 'p3', 'p1', NOW + 30001).error).toBe('Voting has closed');
  });

  it('the host can close voting early', () => {
    let g = apply(duel(), V.openVoting(duel(), NOW));
    g = apply(g, V.closeVoting(g));
    expect(V.voteStatus(g, NOW + 1000)).toMatchObject({ accepting: false, blocking: false });
    expect(D.resolveDuel(g, 'p1', NOW + 1000).error).toBeUndefined();
  });

  it('a spectator who has dropped off does not hold the table up', () => {
    const g0 = duel();
    g0.players!.p4.connected = false;
    let g = apply(g0, V.openVoting(g0, NOW));
    g = apply(g, A.voteOnDuel(g, 'p3', 'p1', NOW + 1000));
    expect(V.voteStatus(g, NOW + 1000)).toMatchObject({ eligible: ['p3'], blocking: false });
  });

  it('"don\'t wait" (window 0): no gate, but votes are still taken', () => {
    const g = duel({ voteWindow: 0 });
    expect(V.openVoting(g, NOW).error).toBeDefined();
    expect(V.voteStatus(g, NOW)).toMatchObject({ accepting: true, blocking: false });
    expect(A.voteOnDuel(g, 'p3', 'p1', NOW).error).toBeUndefined();
  });

  it('tiebreak duels have no vote (the winner goes on to the real duel)', () => {
    const g = duel();
    g.currentTurn!.isTiebreak = true;
    expect(V.voteStatus(g, NOW)).toMatchObject({ eligible: [], blocking: false });
  });
});
