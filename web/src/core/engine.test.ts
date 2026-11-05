// The rules engine, played in memory. Each test mirrors a flow the Playwright suite drives
// through the real pages, so the React apps inherit exactly the behaviour those tests pin.
import { describe, expect, it } from 'vitest';
import * as A from './actions';
import * as D from './duel';
import * as P from './partyflow';
import { apply, chipsOf, ctx, NOW, table } from './testkit';
import type { Game } from './types';

/** p1 opens "Gift A" (seed chosen so the draw lands on p1). */
function opened(g: Game, giftId = 'gA', giftName = 'Gift A'): Game {
  for (let seed = 1; seed < 200; seed++) {
    const plan = A.drawGift(g, giftName, giftId, ctx(seed));
    if (plan.updates[`players/p1/giftId`] === giftId) return apply(g, plan);
  }
  throw new Error('no seed draws p1');
}
const bid = (g: Game, bids: Record<string, number>) =>
  Object.entries(bids).reduce((acc, [id, n]) => apply(acc, A.submitBid(acc, id, n, NOW)), g);

describe('lobby', () => {
  it('variable starting chips: everyone gets 12 at an 11-player table (joiners used to get 10 or 12)', () => {
    let g: Game = { phase: 'lobby', settings: {} };
    for (let i = 1; i <= 11; i++) g = apply(g, A.joinPlayer(g, { playerId: `p${i}`, name: `P${i}`, uid: 'u', now: NOW }));
    expect(new Set(Object.values(chipsOf(g)))).toEqual(new Set([10, 12])); // before Start: mixed
    g = apply(g, A.startGame(g));
    expect(new Set(Object.values(chipsOf(g)))).toEqual(new Set([12]));
  });
  it('a fixed Starting Chips setting is kept; names are cleaned', () => {
    let g: Game = { phase: 'lobby', settings: { startingChips: 15 } };
    g = apply(g, A.joinPlayer(g, { playerId: 'a', name: '  <b>Al</b>  ', uid: null, now: NOW }));
    g = apply(g, A.joinPlayer(g, { playerId: 'b', name: 'Bo', uid: null, now: NOW }));
    g = apply(g, A.startGame(g));
    expect(g.players!.a).toMatchObject({ name: 'bAl/b', chips: 15 });
    expect(A.startGame({ phase: 'lobby', players: { a: { name: 'x', chips: 1 } } }).error).toBeTruthy();
  });
});

describe('sealed-bid duels', () => {
  it('challenger wins: pays the bid, wins the pot back; first duel sets Naughty to the top bid', () => {
    let g = opened(table(3));
    g = bid(g, { p2: 2, p3: 0 });
    g = apply(g, A.revealBids(g));
    g = apply(g, A.startDuel(g, 'p2', 'p1'));
    expect(chipsOf(g)).toMatchObject({ p2: 8 });
    g = apply(g, D.resolveDuel(g, 'p2', NOW));
    expect(chipsOf(g)).toEqual({ p1: 11, p2: 10, p3: 10 }); // defender dividend +1, pot 2 back
    expect(g.gifts!.gA).toMatchObject({ ownerId: 'p2', naughtyLevel: 2 });
    expect(g.players!.p1.giftId ?? null).toBeNull();
    expect(g.currentTurn ?? null).toBeNull(); // no other gift to Reprisal against
    expect(g.lastDuel).toMatchObject({ winnerId: 'p2', loserId: 'p1', pot: 2, dividend: 1, loserPaid: 0 });
  });

  it('Santa Tax: a bid of 4 pays the winner 2', () => {
    let g = opened(table(3));
    g = bid(g, { p2: 4 });
    g = apply(g, A.startDuel(g, 'p2', 'p1'));
    expect(g.currentTurn).toMatchObject({ pot: 2, tax: 2 });
    g = apply(g, D.resolveDuel(g, 'p1', NOW));
    expect(chipsOf(g)).toMatchObject({ p1: 12, p2: 6 }); // challenger lost: no dividend (defenders only)
  });

  it('a double-tap cannot start or resolve a duel twice (claims)', () => {
    let g = opened(table(3));
    g = bid(g, { p2: 3 });
    const start = A.startDuel(g, 'p2', 'p1');
    g = apply(apply(g, start), A.startDuel(apply(g, start), 'p2', 'p1'));
    expect(g.players!.p2.chips).toBe(7);
    const win = D.resolveDuel(g, 'p2', NOW);
    const once = apply(g, win);
    expect(apply(once, win)).toEqual(once);
  });

  it('a Reprisal is offered when another gift exists; it can claim a Misfit outright', () => {
    let g = opened(table(3));
    g = apply(g, A.keepGift(g));
    g = { ...g, gifts: { ...g.gifts, gM: { name: 'Fruitcake', ownerId: 'misfit', naughtyLevel: 0, misfitOrder: 1 } } };
    // p2 opens gB, p3 bids and wins → p2 offered a Reprisal
    g = apply(g, { updates: { 'gifts/gB': { name: 'Gift B', ownerId: 'p2', naughtyLevel: 0 }, 'players/p2/giftId': 'gB', currentTurn: { phase: 'bidding', openerId: 'p2', activePlayerId: 'p2', giftId: 'gB' } }, events: [] });
    g = bid(g, { p3: 1 });
    g = apply(g, A.startDuel(g, 'p3', 'p2'));
    g = apply(g, D.resolveDuel(g, 'p3', NOW));
    expect(g.currentTurn).toMatchObject({ phase: 'reprisal', reprisalPlayerId: 'p2', excludedGiftId: 'gB', reprisalDepth: 1 });
    expect(D.chooseReprisal(g, 'gB', NOW).error).toBeTruthy(); // not the gift just lost
    g = apply(g, D.chooseReprisal(g, 'gM', NOW));
    expect(g.gifts!.gM.ownerId).toBe('p2');
    expect(g.players!.p2).toMatchObject({ giftId: 'gM', chips: 10 }); // 10 +1 dividend −1 cost
    expect(g.currentTurn ?? null).toBeNull();
  });

  it("swap: the loser's ORIGINAL gift goes to the Misfit pile, they keep the winner's old gift", () => {
    let g = table(3);
    g.gifts = { gW: { name: 'W', ownerId: 'p2', naughtyLevel: 1 }, gL: { name: 'L', ownerId: 'p3', naughtyLevel: 2 } };
    g.players!.p2.giftId = 'gW';
    g.players!.p3.giftId = 'gL';
    // p2 Reprisals p3's gift and wins
    g.currentTurn = { phase: 'duel', challengerId: 'p2', defenderId: 'p3', giftId: 'gL', pot: 0, isReprisal: true, reprisalDepth: 1 };
    g = apply(g, D.resolveDuel(g, 'p2', NOW));
    expect(g.players!.p2.giftId).toBe('gL');
    expect(g.players!.p3.giftId).toBe('gW');
    expect(g.gifts!.gL).toMatchObject({ ownerId: 'p2', naughtyLevel: 3 });
  });

  it('yield duel: both pay half (rounded down); the Opener is left giftless', () => {
    let g = opened(table(3));
    g = bid(g, { p2: 3, p3: 2 });
    g = apply(g, A.openerYield(g, 'p2', 'p3', ctx().rng));
    g = apply(g, A.startYieldDuel(g, 'p2', 'p3', false, ctx()));
    expect(chipsOf(g)).toMatchObject({ p2: 9, p3: 9 });
    expect(g.currentTurn).toMatchObject({ isYieldDuel: true, pot: 2, tax: 0 });
    g = apply(g, D.resolveDuel(g, 'p3', NOW));
    expect(g.gifts!.gA.ownerId).toBe('p3');
    expect(g.players!.p1.giftId ?? null).toBeNull();
    expect(chipsOf(g)).toMatchObject({ p2: 9, p3: 11 }); // no dividend in yield duels
  });

  it("Grinch's Gambit: costs 1 + Naughty, never targets the Misfit pile, no Reprisal after", () => {
    let g = table(3);
    g.gifts = { gX: { name: 'X', ownerId: 'p2', naughtyLevel: 1 }, gM: { name: 'M', ownerId: 'misfit', naughtyLevel: 0 } };
    g.players!.p2.giftId = 'gX';
    g = opened(g);
    g = apply(g, A.revealBids(g));
    expect(A.gambit(g, 'gM').error).toBeTruthy();
    g = apply(g, A.gambit(g, 'gX'));
    expect(g.players!.p1.chips).toBe(8);
    g = apply(g, D.resolveDuel(g, 'p2', NOW));
    expect(g.currentTurn ?? null).toBeNull();
  });

  it('"0 + Naughty" base cost really is free at Naughty 0', () => {
    let g = table(2, { baseCost: 0 });
    g.gifts = { gX: { name: 'X', ownerId: 'p2', naughtyLevel: 0 } };
    g.players!.p2.giftId = 'gX';
    g = opened(g);
    g = apply(g, A.revealBids(g));
    g = apply(g, A.gambit(g, 'gX'));
    expect(g.players!.p1.chips).toBe(10);
  });

  it('spectator votes count for Party Score; with bets on, right callers win the wrong callers\' chips', () => {
    let g = opened(table(4, { spectatorBets: 1 }));
    g = bid(g, { p2: 1 });
    g = apply(g, A.startDuel(g, 'p2', 'p1'));
    g = apply(g, A.voteOnDuel(g, 'p3', 'p2', NOW));
    g = apply(g, A.voteOnDuel(g, 'p4', 'p1', NOW));
    expect(A.voteOnDuel(g, 'p2', 'p2', NOW).error).toBeTruthy(); // duellists can't vote
    g = apply(g, D.resolveDuel(g, 'p2', NOW));
    expect(chipsOf(g)).toMatchObject({ p3: 11, p4: 9 });
    expect(g.players!.p3.votingWins).toBe(1);
    expect(g.players!.p4.votingLosses).toBe(1);
    expect(g.lastDuel!.bets).toEqual({ pool: 1, toBank: 0, winners: 1 });
  });
});

describe('ties', () => {
  it('a tie is always a whole-table party game: the default mixes Whack-an-Elf and Sketch-Off, never a two-player tiebreak duel', () => {
    const types = new Set<string>();
    for (let seed = 1; seed <= 12; seed++) {
      let g = opened(table(4));
      g = bid(g, { p2: 2, p3: 2, p4: 0 });
      g = apply(g, A.openerFight(g, 'p2', 'p3', true, ctx(seed).rng));
      g = apply(g, A.startTiebreaker(g, 'p2', 'p3', ctx(seed)));
      expect(g.currentTurn?.phase).toBe('minigame_party');
      expect(g.currentTurn?.isTiebreak).toBeFalsy();
      expect(g.minigame).toMatchObject({ phase: 'intro', tiedBidders: ['p2', 'p3'], participants: ['p1', 'p2', 'p3', 'p4'] });
      types.add((g.minigame as { type: string }).type);
    }
    expect(types).toEqual(new Set(['whack_elf', 'sketch_off']));
  });

  it('games saved with the retired methods ("random", "draw_off") now settle ties with a party game too', () => {
    for (const tiebreakerMethod of ['random', 'draw_off'] as const) {
      let g = opened(table(3, { tiebreakerMethod }));
      g = bid(g, { p2: 2, p3: 2 });
      g = apply(g, A.hostTieFight(g, ctx()));
      expect(g.currentTurn?.phase, tiebreakerMethod).toBe('minigame_party');
    }
  });

  it('Whack-an-Elf: everyone plays, the better tied bidder wins, the best other player gets the Elf Bonus', () => {
    let g = opened(table(3, { tiebreakerMethod: 'whack_elf' }));
    g = bid(g, { p2: 2, p3: 2 });
    g = apply(g, A.hostTieFight(g, ctx()));
    expect(g.currentTurn!.phase).toBe('minigame_party');
    expect(g.minigame).toMatchObject({ type: 'whack_elf', phase: 'intro', tiedBidders: ['p2', 'p3'], participants: ['p1', 'p2', 'p3'] });
    g = apply(g, P.advanceParty(g, 'intro', NOW));
    for (const [pid, score] of [['p1', 7], ['p2', 3], ['p3', 9]] as const) g = apply(g, P.reportWhack(g, pid, score, true));
    g = apply(g, P.advanceParty(g, 'playing', NOW));
    expect((g.minigame as { results: unknown }).results).toMatchObject({ winners: ['p3'], bonus: 'p1', points: { p3: 3, p1: 2, p2: 1 } });
    expect(chipsOf(g)).toMatchObject({ p1: 11 });
    expect(P.advanceParty(g, 'playing', NOW).error).toBeTruthy(); // that phase is over: no double pay
    g = apply(g, P.advanceParty(g, 'results', NOW));
    expect(g.minigame ?? null).toBeNull();
    expect(g.currentTurn).toMatchObject({ phase: 'duel', challengerId: 'p3', defenderId: 'p1', pot: 2 });
    expect(chipsOf(g)).toMatchObject({ p3: 8 });
  });

  it('Sketch-Off after a yield (3-way tie): the two most-voted tied bidders duel each other', () => {
    let g = opened(table(4, { tiebreakerMethod: 'sketch_off' }));
    g = bid(g, { p2: 2, p3: 2, p4: 2 });
    g = apply(g, A.startYieldDuel(g, 'p2', 'p3', false, ctx()));
    expect(g.minigame).toMatchObject({ type: 'sketch_off', minigameGoal: 'pick_two' });
    g = apply(g, P.advanceParty(g, 'intro', NOW));
    for (const pid of ['p1', 'p2', 'p3', 'p4']) g = apply(g, P.submitSketch(g, pid, 'data:x'));
    g = apply(g, P.advanceParty(g, 'playing', NOW));
    for (const [v, t] of [['p1', 'p4'], ['p2', 'p4'], ['p3', 'p2'], ['p4', 'p2']]) g = apply(g, P.voteSketch(g, v, t));
    g = apply(g, P.advanceParty(g, 'voting', NOW));
    g = apply(g, P.advanceParty(g, 'results', NOW));
    expect(g.currentTurn).toMatchObject({ phase: 'duel', isYieldDuel: true });
    expect(new Set([g.currentTurn!.challengerId, g.currentTurn!.defenderId])).toEqual(new Set(['p2', 'p4']));
  });
});

describe('Misfit Lottery', () => {
  function endgameTable(): Game {
    const g = table(3);
    g.gifts = {
      ga: { name: 'A', ownerId: 'p1', naughtyLevel: 0 }, gb: { name: 'B', ownerId: 'p2', naughtyLevel: 0 },
      gm: { name: 'Lonely Misfit', ownerId: 'misfit', naughtyLevel: 0, misfitOrder: 1 },
    };
    g.players!.p1.giftId = 'ga';
    g.players!.p2.giftId = 'gb';
    return g;
  }

  it('Path A steal, win: take their gift, they take the Misfit; the game then ends', () => {
    let g = apply(endgameTable(), D.startEndgame(endgameTable()));
    g = apply(g, D.drawEndgamePlayer(g, ctx().rng));
    expect(g.currentTurn).toMatchObject({ phase: 'endgame_choice', activePlayerId: 'p3', misfitId: 'gm' });
    g = apply(g, D.chooseSteal(g));
    g = apply(g, D.executeSteal(g, 'ga'));
    expect(g.players!.p3.chips).toBe(9);
    const plan = D.resolveDuel(g, 'p3', NOW);
    expect(plan.then).toBe('checkEndgameContinue');
    g = apply(g, plan);
    expect(g.players!.p3.giftId).toBe('ga');
    expect(g.players!.p1.giftId).toBe('gm');
    g = apply(g, D.checkEndgameContinue(g, ctx().rng));
    expect(g.phase).toBe('ended');
  });

  it('auction where everyone passes → the active player picks a defender; losing costs a 1-chip toll', () => {
    let g = apply(endgameTable(), D.startEndgame(endgameTable()));
    g = apply(g, D.drawEndgamePlayer(g, ctx().rng));
    g = apply(g, D.startAuction(g));
    g = bid(g, { p1: 0, p2: 0 });
    g = apply(g, D.revealAuction(g));
    expect(g.currentTurn!.phase).toBe('endgame_nobids');
    g = apply(g, D.noBidsDuel(g, 'p1'));
    g = apply(g, D.resolveDuel(g, 'p1', NOW));
    expect(g.players!.p3).toMatchObject({ giftId: 'gm', chips: 9 });
    expect(g.players!.p1.chips).toBe(11);
  });

  it('auction won by a giftless bidder: the active player goes back in the draw (stats still count)', () => {
    let g = endgameTable();
    g.players!.p2.giftId = null;
    g.gifts!.gb.ownerId = 'misfit';
    g.gifts!.gb.misfitOrder = 2;
    g = apply(g, D.startEndgame(g));
    g = apply(g, { updates: { currentTurn: { phase: 'endgame_bidding', activePlayerId: 'p3', misfitId: 'gm', misfitName: 'Lonely Misfit' } }, events: [] });
    g = bid(g, { p2: 2, p1: 1 });
    g = apply(g, D.revealAuction(g));
    g = apply(g, D.auctionDuel(g, 'p2'));
    const plan = D.resolveDuel(g, 'p2', NOW);
    expect(plan.then).toBe('drawEndgamePlayer');
    g = apply(g, plan);
    expect(g.players!.p2).toMatchObject({ giftId: 'gm', duelWins: 1 });
    expect(g.players!.p3.giftId ?? null).toBeNull();
  });
});
