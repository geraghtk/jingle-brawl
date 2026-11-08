// The mini-game runtime with a test game (Tap Race): lifecycle, action ids and versions, server
// deadlines, pause/resume, the host's overrides, per-viewer views, and the hand-off to the duel
// (only a resolved result with a winning duelist names a winner).
import { beforeAll, describe, expect, it } from 'vitest';
import { duelAppWinner, startDuelApp } from '../duelgame';
import { resolveDuel } from '../duel';
import { hostAuto } from '../hostauto';
import { projectGame } from '../projection';
import { rng } from '../rng';
import { HOST, perform } from '../table';
import { apply, chipsOf, NOW, table } from '../testkit';
import type { Game } from '../types';
import { register } from './registry';
import { arenaAct, arenaOf, arenaPause, arenaResume, arenaStop, handoff } from './runtime';
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
});
