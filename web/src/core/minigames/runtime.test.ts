// The mini-game runtime with a test game (Tap Race): lifecycle, action ids and versions, server
// deadlines, pause/resume, the host's overrides, per-viewer views, and the hand-off to the duel
// (only a resolved result with a winning duelist names a winner).
import { beforeAll, describe, expect, it } from 'vitest';
import { duelAppWinner, startDuelApp } from '../duelgame';
import { resolveDuel } from '../duel';
import { hostAuto } from '../hostauto';
import { projectGame } from '../projection';
import { rng } from '../rng';
import { HOST, offers, perform } from '../table';
import { apply, chipsOf, NOW, table } from '../testkit';
import type { Game } from '../types';
import { register } from './registry';
import { arenaAct, arenaOf, arenaPause, arenaResume, arenaStop, awaited, handoff, READY_COUNT_MS, RESULT_HOLD_MS } from './runtime';
import { tapRace } from './testgames/tapRace';

beforeAll(() => register(tapRace));
const ctx = (now = NOW) => ({ now, rng: rng(5) });

/** A duel between p2 and p1 whose duel game is Tap Race, started (anyone else is away: a direct duel). */
function racing(players = 2): Game {
  const g = table(players, { duelPick: 'off', duelAppGames: ['tap_race'], voteWindow: 0 });
  for (let i = 3; i <= players; i++) g.players![`p${i}`].connected = false;
  g.gifts = { gA: { name: 'Gift A', ownerId: 'p1', naughtyLevel: 0 } };
  g.players!.p1.giftId = 'gA';
  g.currentTurn = { phase: 'duel', challengerId: 'p2', defenderId: 'p1', giftId: 'gA', pot: 2, duelGame: { name: 'Tap Race', kind: 'app', app: 'tap_race', via: 'picker' } };
  return apply(g, startDuelApp(g, ctx()));
}
const tap = (g: Game, who: string, aid?: string, v?: number) => arenaAct(g, who, { action: { tap: true }, aid, v }, ctx());

describe('the mini-game runtime', () => {
  it('a registered game starts as the duel game: the turn waits in minigame_arena', () => {
    const g = racing();
    expect(g.currentTurn?.phase).toBe('minigame_arena');
    expect(arenaOf(g)).toMatchObject({ gameId: 'tap_race', mode: 'direct_duel', lifecycle: 'active', duelists: ['p2', 'p1'], stateVersion: 1 });
  });

  it('refuses a mode that can\'t always settle the duel (gift stakes)', () => {
    const g = table(4, { duelPick: 'off', duelAppGames: ['tap_race'], voteWindow: 0 });
    g.currentTurn = { phase: 'duel', challengerId: 'p2', defenderId: 'p1', pot: 2, duelGame: { name: 'Tap Race', kind: 'app', app: 'tap_race', via: 'picker' } };
    expect(startDuelApp(g, ctx()).error).toMatch(/can't settle this duel/);  // 4 at the table: everyone plays, and a tie is possible
  });

  it('only participants act; a resent action id is answered once; a stale version is refused', () => {
    let g = racing(3);
    g = apply(g, tap(g, 'p2', 'x1'));
    expect(tap(g, 'p2', 'x1').error).toMatch(/Already/);
    expect(tap(g, 'p3').error).toBeTruthy();                                 // p3 isn't in a direct duel
    expect(tap(g, 'p1', undefined, 0).error).toMatch(/moved on/);
    expect(tap(g, 'p1', undefined, arenaOf(g)!.stateVersion).error).toBeUndefined();
  });

  it('the result hands the duel its winner, once, and the duel resolves with them', () => {
    let g = racing();
    for (let i = 0; i < 5; i++) g = apply(g, tap(g, 'p2'));
    expect(arenaOf(g)).toMatchObject({ lifecycle: 'completed', result: { status: 'resolved', duel: { winningDuelistId: 'p2' } } });
    const step = hostAuto(g).find((s) => s.key.endsWith('|handoff'))!;
    g = apply(g, step.run(g, ctx()));
    expect(g.currentTurn).toMatchObject({ phase: 'duel', duelGameWinner: 'p2' });
    expect(arenaOf(g) ?? null).toBeNull();
    expect(handoff(g).error).toBeTruthy();
    expect(duelAppWinner(g.currentTurn)).toBe('p2');
    g = apply(g, resolveDuel(g, 'p2', NOW));
    expect(g.lastDuel).toMatchObject({ winnerId: 'p2' });
  });

  it("a finished game's ending stays up a few seconds before the duel takes over; the host can move on at once; a stopped game has nothing to show", () => {
    let g = racing();
    for (let i = 0; i < 5; i++) g = apply(g, tap(g, 'p2'));
    expect(arenaOf(g)!.endedAt).toBe(NOW);
    expect(hostAuto(g).find((s) => s.key.endsWith('|handoff'))!.at).toBe(NOW + RESULT_HOLD_MS);
    expect(projectGame(g, { kind: 'tv' }, NOW).arenaView).toMatchObject({ lifecycle: 'completed', endedAt: NOW });
    // the host's button: on to the duel now (nobody else's)
    expect(offers(g, HOST, NOW)).toContainEqual({ type: 'arena_continue' });
    expect(perform(g, 'p1', { type: 'arena_continue' }, ctx()).error).toBeTruthy();
    const on = apply(g, perform(g, HOST, { type: 'arena_continue' }, ctx()));
    expect(on.currentTurn).toMatchObject({ phase: 'duel', duelGameWinner: 'p2' });
    expect(perform(on, HOST, { type: 'arena_continue' }, ctx()).error).toBeTruthy();   // once
    // cancelled by the host: straight back to them
    const stopped = apply(racing(), arenaStop(racing(), 'cancelled', NOW));
    expect(hostAuto(stopped).find((s) => s.key.endsWith('|handoff'))!.at).toBe(0);
  });

  it('a deadline is the server\'s: the hostAuto step runs it when due', () => {
    let g = racing();
    g = apply(g, tap(g, 'p1'));
    const d = arenaOf(g)!.deadlines[0];
    const step = hostAuto(g).find((s) => s.key.startsWith(`arena|${arenaOf(g)!.instanceId}|end`))!;
    expect(step.at).toBe(d.at);
    g = apply(g, step.run(g, ctx(d.at)));
    expect(arenaOf(g)?.result).toMatchObject({ status: 'resolved', duel: { winningDuelistId: 'p1' } });
  });

  it('pause keeps the time that was left; resume gives it back', () => {
    let g = racing();
    const end = arenaOf(g)!.deadlines[0].at;
    g = apply(g, arenaPause(g, NOW + 4000));
    expect(arenaOf(g)?.lifecycle).toBe('paused');
    expect(tap(g, 'p2').error).toMatch(/paused/);
    expect(hostAuto(g).some((s) => s.key.startsWith('arena|') && !s.key.endsWith('handoff'))).toBe(false);
    g = apply(g, arenaResume(g, NOW + 60_000));
    expect(arenaOf(g)!.deadlines[0].at).toBe(NOW + 60_000 + (end - (NOW + 4000)));
    expect(arenaOf(g)!.overrides.map((o) => o.what)).toEqual(['paused', 'resumed']);
  });

  it('cancel or no contest: no winner, the duel goes back to the host, no chips or gifts move', () => {
    for (const stop of ['cancelled', 'no_contest'] as const) {
      let g = racing();
      const before = { chips: chipsOf(g), gifts: g.gifts };
      g = apply(g, arenaStop(g, stop, NOW));
      g = apply(g, handoff(g));
      expect(g.currentTurn?.duelGameWinner).toBeUndefined();
      expect(g.currentTurn?.duelGameOutcome?.status).toBe('no_contest');
      expect({ chips: chipsOf(g), gifts: g.gifts }).toEqual(before);
    }
  });

  it("the host's overrides are table offers", () => {
    const g = racing();
    expect(perform(g, HOST, { type: 'arena_pause' }, ctx()).error).toBeUndefined();
    expect(perform(g, 'p2', { type: 'arena_pause' }, ctx()).error).toBeTruthy();
    expect(perform(g, 'p2', { type: 'arena', action: { tap: true } }, ctx()).error).toBeUndefined();
  });

  it('the running game never leaves the server: each screen gets its own view of it', () => {
    let g = racing();
    g = apply(g, tap(g, 'p2'));
    const phone = projectGame(g, { kind: 'phone', pid: 'p2' }, NOW);
    expect(phone.arena).toBeUndefined();
    expect(phone.arenaView).toMatchObject({ gameId: 'tap_race', view: { mine: 1 } });
    expect(projectGame(g, { kind: 'tv' }, NOW).arenaView?.view).toEqual({ taps: { p2: 1 } });
    expect(JSON.stringify(phone)).not.toContain('"deadlines"');
  });

  describe('the instructions first: nobody starts until everyone is ready', () => {
    /** The same duel with the ready stage on (the house default). */
    function briefed(players = 2): Game {
      const g = table(players, { duelPick: 'off', duelAppGames: ['tap_race'], voteWindow: 0, readyCheck: true });
      for (let i = 3; i <= players; i++) g.players![`p${i}`].connected = false;
      g.currentTurn = { phase: 'duel', challengerId: 'p2', defenderId: 'p1', pot: 2, duelGame: { name: 'Tap Race', kind: 'app', app: 'tap_race', via: 'picker' } };
      return apply(g, startDuelApp(g, ctx()));
    }
    const say = (g: Game, who: string, type: 'arena_ready' | 'arena_start_now', now = NOW) => perform(g, who, { type }, ctx(now));

    it('the game is made but holds still: nobody can play, each player is offered Ready, the screens are told who is', () => {
      let g = briefed();
      expect(arenaOf(g)).toMatchObject({ lifecycle: 'briefing', paused: { at: NOW } });
      expect(arenaOf(g)!.deadlines?.length ?? 0).toBe(0);                    // its clock is not running
      expect(tap(g, 'p2').error).toBeTruthy();                               // not started
      expect(offers(g, 'p1', NOW)).toEqual([{ type: 'arena_ready' }]);
      expect(hostAuto(g).filter((s) => s.key.startsWith('arena|'))).toEqual([]);   // nothing happens by itself while somebody is awaited
      g = apply(g, say(g, 'p1', 'arena_ready'));
      expect(offers(g, 'p1', NOW)).toEqual([]);
      expect(say(g, 'p1', 'arena_ready').error).toBeTruthy();                // once
      expect(arenaOf(g)!.briefing!.goAt).toBeUndefined();                    // p2 is still reading
      expect(projectGame(g, { kind: 'tv' }, NOW).arenaView).toMatchObject({ lifecycle: 'briefing', briefing: { ready: ['p1'] }, pausedAt: NOW });
    });

    it('the last one ready starts a 3, 2, 1; only then does the game\'s clock run, with all its time', () => {
      let g = briefed();
      g = apply(g, say(g, 'p1', 'arena_ready', NOW + 20_000));
      g = apply(g, say(g, 'p2', 'arena_ready', NOW + 40_000));
      expect(arenaOf(g)!.briefing!.goAt).toBe(NOW + 40_000 + READY_COUNT_MS);
      const begin = hostAuto(g).find((s) => s.key.includes('|begin|'))!;
      expect(begin.at).toBe(NOW + 40_000 + READY_COUNT_MS);
      expect(begin.run(g, ctx(NOW + 41_000)).error).toBeTruthy();            // not before the count is out
      const at = NOW + 40_000 + READY_COUNT_MS;
      g = apply(g, begin.run(g, ctx(at)));
      const a = arenaOf(g)!;
      expect(a).toMatchObject({ lifecycle: 'active', startedAt: at });
      expect(a.briefing ?? null).toBeNull();
      expect(a.deadlines[0].at).toBe(at + 10_000);                           // the race's full ten seconds, from now
      expect(arenaAct(g, 'p2', { action: { tap: true } }, ctx(at)).error).toBeUndefined();
    });

    it('a practice bot is always ready; a phone that has dropped is not waited for', () => {
      const g0 = table(3, { duelPick: 'off', duelAppGames: ['tap_race'], voteWindow: 0, readyCheck: true });
      g0.players!.p3.connected = false;
      g0.players!.p1.isBot = true;
      g0.currentTurn = { phase: 'duel', challengerId: 'p2', defenderId: 'p1', pot: 2, duelGame: { name: 'Tap Race', kind: 'app', app: 'tap_race', via: 'picker' } };
      let g = apply(g0, startDuelApp(g0, ctx()));
      expect(awaited(g, arenaOf(g)!)).toEqual(['p2']);
      // p2's phone drops while the instructions are up: the room starts the count itself
      g = { ...g, players: { ...g.players, p2: { ...g.players!.p2, connected: false } } };
      const step = hostAuto(g).find((s) => s.key.endsWith('|allready'))!;
      g = apply(g, step.run(g, ctx(NOW + 5000)));
      expect(arenaOf(g)!.briefing!.goAt).toBe(NOW + 5000 + READY_COUNT_MS);
    });

    it('the host can start without waiting (logged), or cancel; a game with only bots counts down at once', () => {
      let g = briefed();
      expect(offers(g, HOST, NOW).map((o) => o.type)).toEqual(['arena_start_now', 'arena_cancel']);
      expect(say(g, 'p1', 'arena_start_now').error).toBeTruthy();            // not a player's button
      g = apply(g, say(g, HOST, 'arena_start_now'));
      expect(arenaOf(g)).toMatchObject({ briefing: { goAt: NOW + READY_COUNT_MS }, overrides: [{ what: 'started without waiting' }] });
      expect(offers(g, HOST, NOW).map((o) => o.type)).toEqual(['arena_cancel']);
      const bots = table(2, { duelPick: 'off', duelAppGames: ['tap_race'], voteWindow: 0, readyCheck: true });
      bots.players!.p1.isBot = true; bots.players!.p2.isBot = true;
      bots.currentTurn = { phase: 'duel', challengerId: 'p2', defenderId: 'p1', pot: 2, duelGame: { name: 'Tap Race', kind: 'app', app: 'tap_race', via: 'picker' } };
      expect(arenaOf(apply(bots, startDuelApp(bots, ctx())))!.briefing!.goAt).toBe(NOW + READY_COUNT_MS);
    });
  });
});
