// One announcement per result: an in-app game's ending says who won the duel AND what they take, and the
// duel is then settled without a second announcement. So what the ending promises (core/stakes.ts) must be
// exactly what the hand-off and resolveDuel then do, for every kind of duel, whoever wins.
import { beforeAll, describe, expect, it } from 'vitest';
import * as DO from './drawoff';
import { resolveDuel } from './duel';
import { startDuelApp } from './duelgame';
import { hostAuto } from './hostauto';
import { startLab } from './minigames/lab';
import { register } from './minigames/registry';
import { arenaAct, arenaOf, arenaStop, handoff, RESULT_HOLD_MS } from './minigames/runtime';
import { tapRace } from './minigames/testgames/tapRace';
import { advanceParty } from './partyflow';
import { announcedWinner, projectGame } from './projection';
import { rng } from './rng';
import { duelStakes, stakesWords } from './stakes';
import type { DuelStakes } from './stakes';
import { HOST, perform } from './table';
import { apply, NOW, table } from './testkit';
import type { Game, Settings, Turn } from './types';

beforeAll(() => register(tapRace));
const ctx = (now = NOW) => ({ now, rng: rng(5) });
const APP = { name: 'Tap Race', kind: 'app' as const, app: 'tap_race', via: 'picker' as const };
const LATER = NOW + RESULT_HOLD_MS;

/** Four at the table (p3, p4 away, so the game is a direct duel), gifts on the table, a Misfit in the pile. */
function room(turn: Partial<Turn>, settings: Settings = {}, edit: (g: Game) => void = () => {}): Game {
  const g = table(4, { duelPick: 'off', duelAppGames: ['tap_race'], voteWindow: 0, ...settings });
  g.players!.p3.connected = false; g.players!.p4.connected = false;
  g.gifts = {
    gA: { name: 'Fuzzy socks', ownerId: 'p1', naughtyLevel: 0 }, gB: { name: 'A candle', ownerId: 'p2', naughtyLevel: 0 },
    gC: { name: 'Mug', ownerId: 'p3', naughtyLevel: 0 }, gM: { name: 'Fruitcake', ownerId: 'misfit', naughtyLevel: 0, misfitOrder: 1 },
  };
  g.players!.p1.giftId = 'gA'; g.players!.p2.giftId = 'gB'; g.players!.p3.giftId = 'gC';
  g.currentTurn = { phase: 'duel', challengerId: 'p2', defenderId: 'p1', giftId: 'gA', pot: 0, duelGame: APP, ...turn };
  edit(g);
  return g;
}

/** Every kind of duel an in-app game can settle. */
const KINDS: Record<string, () => Game> = {
  'standard (a bid, both hold a gift)': () => room({ pot: 3, bids: { p2: { amount: 3 } }, openerId: 'p1' }),
  'standard (the challenger has no gift yet)': () => room({ pot: 2, bids: { p2: { amount: 2 } }, openerId: 'p1' }, {}, (g) => { g.players!.p2.giftId = null as never; g.gifts!.gB.ownerId = 'p4'; g.players!.p4.giftId = 'gB'; }),
  'after a tie-break, with Santa Tax': () => room({ pot: 4, tax: 2, bids: { p2: { amount: 6 }, p3: { amount: 6 } }, openerId: 'p1', isTiebreak: false }),
  'yield (two challengers, the opener gave it up)': () => room({ pot: 3, isYieldDuel: true, yieldingOpener: 'p3', giftId: 'gC', bids: { p1: { amount: 3 }, p2: { amount: 4 } }, openerId: 'p3' }),
  "Grump's Gambit": () => room({ pot: 2, isGambit: true, bids: { p2: { amount: 2 } }, originalGiftId: 'gB' }),
  'Reindeer Reprisal': () => room({ pot: 0, isReprisal: true, reprisalDepth: 1 }),
  'everyone gets a dividend': () => room({ pot: 1, bids: { p2: { amount: 1 } } }, { loserDividendRule: 'everyone' }),
  'endgame steal': () => room({ pot: 0, isEndgameSteal: true, isMisfitDuel: true, misfitId: 'gM', misfitName: 'Fruitcake' }, {}, (g) => { g.phase = 'endgame'; g.players!.p2.giftId = null as never; g.gifts!.gB.ownerId = 'p4'; g.players!.p4.giftId = 'gB'; }),
  'endgame auction (the bidder had a gift)': () => room({ pot: 2, isEndgameAuction: true, isMisfitDuel: true, giftId: 'gM', misfitId: 'gM', misfitName: 'Fruitcake', bidderHadGift: true, bidderOldGift: 'gA' }, {}, (g) => { g.phase = 'endgame'; g.players!.p2.giftId = null as never; g.gifts!.gB.ownerId = 'p4'; g.players!.p4.giftId = 'gB'; }),
  'endgame auction (the bidder had none)': () => room({ pot: 2, isEndgameAuction: true, isMisfitDuel: true, giftId: 'gM', misfitId: 'gM', misfitName: 'Fruitcake', bidderHadGift: false, bidderOldGift: null }, {}, (g) => {
    g.phase = 'endgame'; g.players!.p2.giftId = null as never; g.players!.p1.giftId = null as never; g.gifts!.gA.ownerId = 'p4'; g.players!.p4.giftId = 'gA'; g.gifts!.gB.ownerId = 'misfit'; g.gifts!.gB.misfitOrder = 2;
  }),
  'endgame, nobody bid (a Misfit Toll)': () => room({ pot: 0, isNoBidsDuel: true, isMisfitDuel: true, giftId: 'gM', misfitId: 'gM', misfitName: 'Fruitcake' }, {}, (g) => { g.phase = 'endgame'; g.players!.p2.giftId = null as never; g.gifts!.gB.ownerId = 'p4'; g.players!.p4.giftId = 'gB'; }),
};

/** Play Tap Race to its end with `winner` first to five. */
function won(g0: Game, winner: string): Game {
  let g = apply(g0, startDuelApp(g0, ctx()));
  for (let i = 0; i < 5; i++) g = apply(g, arenaAct(g, winner, { action: { tap: true } }, ctx()));
  expect(arenaOf(g)?.result).toMatchObject({ status: 'resolved', duel: { winningDuelistId: winner } });
  return g;
}
/** The room's own next steps: the hand-off when the ending has had its time, then the duel settled with the game's winner. */
function settle(g: Game, now = LATER): Game {
  g = apply(g, hostAuto(g).find((s) => s.key.endsWith('|handoff'))!.run(g, ctx(now)));
  const app = hostAuto(g).find((s) => s.key.startsWith('app|'));
  return app ? apply(g, app.run(g, ctx(now))) : g;
}
const giftOf = (g: Game, id: string) => { const gid = g.players?.[id]?.giftId; return gid ? g.gifts?.[gid]?.name : undefined; };
const chips = (g: Game, id: string) => g.players![id].chips;

describe('the stakes an in-app game announces are exactly what the duel then does', () => {
  for (const [kind, make] of Object.entries(KINDS)) {
    for (const winner of ['p1', 'p2']) {
      it(`${kind}: ${winner} wins`, () => {
        const g = won(make(), winner);
        const loser = winner === 'p1' ? 'p2' : 'p1';
        const s = projectGame(g, { kind: 'tv' }, NOW + 2000).arenaView?.stakes as DuelStakes;
        expect(s, 'the ending carries the stakes').toBeTruthy();
        expect(s).toMatchObject({ winnerId: winner, loserId: loser });
        // every screen is told the same, a phone that may not see the others' chips too
        const hidden = { ...g, settings: { ...g.settings, publicChips: false } };
        expect(projectGame(hidden, { kind: 'phone', pid: 'p3' }, NOW + 2000).arenaView?.stakes).toEqual(s);

        const after = settle(g);
        expect(after.lastDuel).toMatchObject({ winnerId: winner, loserId: loser, pot: s.pot, dividend: s.dividend, loserPaid: s.loserPaid, announced: true });
        // chips: the pot and a toll to the winner, the dividend to the loser (less a toll)
        expect(chips(after, winner) - chips(g, winner)).toBe(s.pot + s.toll);
        expect(chips(after, loser) - chips(g, loser)).toBe(s.dividend - s.toll);
        // gifts: what the winner ends with, what the loser is handed or left without, what goes to the pile
        const changed = giftOf(after, winner) !== giftOf(g, winner);
        if (changed) expect(s.gift).toBe(giftOf(after, winner));
        if (s.gift) { expect(giftOf(after, winner)).toBe(s.gift); expect(!!s.kept).toBe(!changed); } else expect(changed).toBe(false);
        const lAfter = giftOf(after, loser), lBefore = giftOf(g, loser);
        expect(s.loserGift).toBe(lAfter && lAfter !== lBefore ? lAfter : undefined);
        expect(!!s.loserGiftless).toBe(!lAfter && !!lBefore);
        const piled = Object.entries(after.gifts || {}).filter(([id, x]) => x.ownerId === 'misfit' && g.gifts![id].ownerId !== 'misfit').map(([, x]) => x.name);
        expect(s.toMisfit || []).toEqual(piled);
        expect(!!s.reprisal).toBe(after.currentTurn?.phase === 'reprisal');
        expect(s.tax).toBe(g.currentTurn?.tax || 0);

        // and in words: something for the winner whenever anything moved, nothing a player sees has an em dash
        const w = stakesWords(s, (id) => after.players?.[id]?.name || '?');
        if (s.gift || s.pot || s.toll) expect(w.prize).toContain(after.players![winner].name);
        expect(JSON.stringify(w)).not.toMatch(/—/);
      });
    }
  }

  it('the words follow the duel: takes or keeps the gift, the pot, the dividend, the Reprisal, the Fund', () => {
    const name = (id: string) => ({ p1: 'Ann', p2: 'Bob', p3: 'Cy', p4: 'Di' }[id] || id);
    const std = KINDS['after a tie-break, with Santa Tax']();
    std.currentTurn!.duelVotes = { p3: { votedFor: 'p2' }, p4: { votedFor: 'p1' } };
    std.currentTurn!.phase = 'minigame_arena';
    expect(stakesWords(duelStakes(std, 'p2', NOW)!, name)).toEqual({
      prize: 'Bob takes the Fuzzy socks and the pot of 4 chips',
      loser: ['gets A candle', "+1 chip Loser's Dividend", 'a Reprisal chance'],
      foot: ['North Pole Fund +2', 'Called it: Cy'],
    });
    expect(stakesWords(duelStakes(std, 'p1', NOW)!, name)).toEqual({
      prize: 'Ann keeps the Fuzzy socks and the pot of 4 chips',
      loser: ['paid 6 chips', 'a Reprisal chance'],
      foot: ['North Pole Fund +2', 'Called it: Di'],
    });
    const toll = KINDS['endgame, nobody bid (a Misfit Toll)']();
    expect(stakesWords(duelStakes(toll, 'p1', NOW)!, name)).toEqual({ prize: 'Ann takes a Misfit Toll of 1 chip', loser: ['gets the Fruitcake'], foot: [] });
    const again = KINDS['endgame auction (the bidder had none)']();
    expect(stakesWords(duelStakes(again, 'p1', NOW)!, name).loser).toContain('is drawn again for the next Misfit');
  });
});

describe('which duels are announced by the game (and so get no verdict after)', () => {
  const std = () => KINDS['standard (a bid, both hold a gift)']();

  it('a duel decided in the room (the host\'s pick) is not marked: it gets the normal verdict', () => {
    const g = std();
    const after = apply(g, resolveDuel(g, 'p2', NOW));
    expect(after.lastDuel?.announced).toBeUndefined();
    expect(announcedWinner(g)).toBeNull();
  });

  it('a practice round promises nothing and hands nothing off', () => {
    const g0 = room({ pot: 3 }, { practiceRound: true });
    let g = apply(g0, startDuelApp(g0, ctx()));
    expect(arenaOf(g)?.practice).toBeTruthy();
    for (let i = 0; i < 5; i++) g = apply(g, arenaAct(g, 'p2', { action: { tap: true } }, ctx()));
    expect(arenaOf(g)?.result?.status).toBe('resolved');
    expect(announcedWinner(g)).toBeNull();
    expect(projectGame(g, { kind: 'tv' }, NOW + 5000).arenaView?.stakes).toBeUndefined();
    expect(handoff(g, LATER).error).toBeTruthy();
  });

  it('the Game Lab promises nothing', () => {
    const g0 = table(2);
    g0.phase = 'lobby';
    let g = apply(g0, startLab(g0, { gameId: 'tap_race', players: ['p1', 'p2'], mode: 'direct_duel', duelists: ['p1', 'p2'] }, ctx()));
    for (let i = 0; i < 5; i++) g = apply(g, arenaAct(g, 'p2', { action: { tap: true } }, ctx()));
    expect(arenaOf(g)?.result?.status).toBe('resolved');
    expect(announcedWinner(g)).toBeNull();
    expect(projectGame(g, { kind: 'tv' }, NOW + 5000).arenaView?.stakes).toBeUndefined();
  });

  it('a game the host stopped (no contest) announces nobody: the host picks, and that gets the verdict', () => {
    const g0 = std();
    let g = apply(g0, startDuelApp(g0, ctx()));
    g = apply(g, arenaStop(g, 'no_contest', NOW));
    expect(announcedWinner(g)).toBeNull();
    expect(projectGame(g, { kind: 'tv' }, NOW).arenaView?.stakes).toBeUndefined();
    g = apply(g, handoff(g, LATER));
    expect(g.currentTurn?.duelAnnounced).toBeUndefined();
    g = apply(g, resolveDuel(g, 'p2', LATER));
    expect(g.lastDuel?.announced).toBeUndefined();
  });

  it('the host moved on before the ceremony came up: nothing was announced, so the verdict says it', () => {
    let g = won(std(), 'p2');
    g = apply(g, perform(g, HOST, { type: 'arena_continue' }, ctx(NOW + 500)));
    expect(g.currentTurn).toMatchObject({ phase: 'duel', duelGameWinner: 'p2' });
    expect(g.currentTurn?.duelAnnounced).toBeUndefined();
    g = apply(g, resolveDuel(g, 'p2', NOW + 600));
    expect(g.lastDuel?.announced).toBeUndefined();
  });

  it('the host overrules the game after the hand-off: that is news, so it gets the verdict', () => {
    let g = won(std(), 'p2');
    g = apply(g, handoff(g, LATER));
    expect(g.currentTurn?.duelAnnounced).toBe('p2');
    g = apply(g, resolveDuel(g, 'p1', LATER));
    expect(g.lastDuel).toMatchObject({ winnerId: 'p1' });
    expect(g.lastDuel?.announced).toBeUndefined();
  });

  it('a party game deciding a duel: its results carry the stakes, and the duel is settled without a verdict', () => {
    const g0 = room({ pot: 3, bids: { p2: { amount: 3 } }, duelGame: { name: 'Whack-an-Elf', kind: 'app', app: 'whack_elf', via: 'picker' } }, { duelAppGames: ['whack_elf'] });
    let g = apply(g0, startDuelApp(g0, ctx()));
    expect(g.minigame).toMatchObject({ type: 'whack_elf', minigameGoal: 'duel' });
    g = apply(g, advanceParty(g, 'intro', NOW));
    g = { ...g, minigame: { ...g.minigame!, scores: { p1: { score: 2, final: true }, p2: { score: 9, final: true } } } as Game['minigame'] };
    expect(announcedWinner(g)).toBeNull();                                    // still playing
    g = apply(g, advanceParty(g, 'playing', NOW + 20_000));
    expect(announcedWinner(g)).toBe('p2');
    const s = (projectGame(g, { kind: 'phone', pid: 'p1' }, NOW + 21_000).minigame as { stakes?: DuelStakes }).stakes!;
    expect(s).toMatchObject({ winnerId: 'p2', loserId: 'p1', pot: 3, gift: 'Fuzzy socks' });
    g = apply(g, advanceParty(g, 'results', NOW + 30_000));
    g = apply(g, hostAuto(g).find((x) => x.key.startsWith('app|'))!.run(g, ctx(NOW + 30_000)));
    expect(g.lastDuel).toMatchObject({ winnerId: 'p2', pot: s.pot, dividend: s.dividend, announced: true });
    expect(giftOf(g, 'p2')).toBe(s.gift);
  });

  it('a party game deciding a TIE announces no duel result: the duel it leads to gets its own verdict', () => {
    const g = room({ phase: 'minigame_party', challengerId: undefined, defenderId: undefined, openerId: 'p1', bids: { p2: { amount: 3 }, p3: { amount: 3 } } });
    g.minigame = { type: 'whack_elf', phase: 'results', seed: 1, participants: ['p1', 'p2', 'p3'], tiedBidders: ['p2', 'p3'], minigameGoal: 'pick_one', prize: 'points', startedAt: NOW, phaseEndsAt: NOW + 9000,
      results: { ranking: ['p2', 'p3', 'p1'], scoreOf: { p2: 5, p3: 2 }, points: {}, bonus: null, winners: ['p2'] } };
    expect(announcedWinner(g)).toBeNull();
    expect((projectGame(g, { kind: 'tv' }, NOW).minigame as { stakes?: unknown }).stakes).toBeUndefined();
    const next = apply(g, advanceParty(g, 'results', NOW + 9000));
    expect(next.currentTurn).toMatchObject({ phase: 'duel', challengerId: 'p2', defenderId: 'p1' });
    expect(next.currentTurn?.duelAnnounced).toBeUndefined();
  });

  it('AI Draw Off: the critic\'s results carry the stakes; a tie leaves it to the host and the verdict', () => {
    const start = () => {
      const g0 = room({ pot: 3, bids: { p2: { amount: 3 } }, duelGame: { name: 'AI Draw Off', kind: 'app', app: 'draw_off', via: 'picker' } }, { duelAppGames: ['draw_off'], aiCriticBackend: 'openai' });
      let g = apply(g0, startDuelApp(g0, ctx()));
      g = apply(g, DO.startDrawing(g, NOW));
      return apply(g, DO.startJudging(g));
    };
    let g = start();
    g = apply(g, DO.judged(g, { scores: { p2: { rating: 81 }, p1: { rating: 64 } }, verdict: 'P2 by a nose.' }, NOW));
    const s = (projectGame(g, { kind: 'tv' }, NOW).minigame as { stakes?: DuelStakes }).stakes!;
    expect(s).toMatchObject({ winnerId: 'p2', pot: 3, gift: 'Fuzzy socks' });
    g = apply(g, DO.completeDrawOff(g));
    g = apply(g, hostAuto(g).find((x) => x.key.startsWith('app|'))!.run(g, ctx()));
    expect(g.lastDuel).toMatchObject({ winnerId: 'p2', announced: true });

    let tie = start();
    tie = apply(tie, DO.judged(tie, { scores: { p2: { rating: 70 }, p1: { rating: 70 } } }, NOW));
    expect((projectGame(tie, { kind: 'tv' }, NOW).minigame as { stakes?: unknown }).stakes).toBeUndefined();
    tie = apply(tie, DO.completeDrawOff(tie));
    tie = apply(tie, resolveDuel(tie, 'p1', NOW));
    expect(tie.lastDuel?.announced).toBeUndefined();
  });
});
