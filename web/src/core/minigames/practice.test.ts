// The practice round (the first time a game comes up in a night), the relaxed pace, the stand-in for a
// teammate whose phone dropped, and "play again with the same people".
import { beforeAll, describe, expect, it } from 'vitest';
import { playAgain } from '../actions';
import { botMove, botsBusy, roomPlayed, standIns } from '../bots';
import { startDuelApp } from '../duelgame';
import { hostAuto } from '../hostauto';
import { dispatch } from '../intents';
import { GAMES } from '../party';
import { advanceParty, startPartyGame } from '../partyflow';
import { projectGame } from '../projection';
import { rng } from '../rng';
import { config, sanitizeSettings } from '../settings';
import { HOST, offers, perform } from '../table';
import { apply, NOW, table } from '../testkit';
import { Tx } from '../tx';
import type { Game, Settings } from '../types';
import { DEFAULTS as CONVEYOR } from './conveyor/conveyor';
import { DEFAULT_WALK } from './engines/walker';
import { startLab } from './lab';
import { pacedConfig, pacedMs } from './pace';
import { register } from './registry';
import { arenaAct, arenaOf, arenaOffers, arenaPause, arenaResume, arenaStop, handoff, PRACTICE_END_MS, PRACTICE_MS, READY_COUNT_MS, REAL_COUNT_MS } from './runtime';
import { tapRace } from './testgames/tapRace';

beforeAll(() => register(tapRace));
const ctx = (now = NOW) => ({ now, rng: rng(5) });
const tap = (g: Game, who: string, now = NOW) => arenaAct(g, who, { action: { tap: true } }, ctx(now));

/** A duel between p2 and p1 about to be settled by `app`. */
function duel(app: string, players = 2, settings: Settings = {}): Game {
  const g = table(players, { duelPick: 'off', duelAppGames: [app], voteWindow: 0, practiceRound: true, ...settings });
  g.currentTurn = { phase: 'duel', challengerId: 'p2', defenderId: 'p1', pot: 2, duelGame: { name: app, kind: 'app', app, via: 'picker' } };
  return g;
}
const auto = (g: Game, ends: string) => hostAuto(g).find((s) => s.key.includes(ends));

describe('a practice round the first time a game comes up', () => {
  it('the first go is practice: it runs for a while, then the same game starts afresh for real after a short count', () => {
    let g = apply(duel('tap_race'), startDuelApp(duel('tap_race'), ctx()));
    expect(arenaOf(g)).toMatchObject({ lifecycle: 'active', practice: { on: true, endsAt: NOW + PRACTICE_MS } });
    expect(g.triedGames).toEqual({ tap_race: true });
    expect(projectGame(g, { kind: 'tv' }, NOW).arenaView).toMatchObject({ practice: { endsAt: NOW + PRACTICE_MS } });
    g = apply(g, tap(g, 'p2'));                                              // they may play
    const first = arenaOf(g)!.instanceId;
    const step = auto(g, '|real|')!;
    expect(step.at).toBe(NOW + PRACTICE_MS);
    expect(step.run(g, ctx(NOW + 1000)).error).toMatch(/Still practising/);
    const at = NOW + PRACTICE_MS;
    g = apply(g, step.run(g, ctx(at)));
    const real = arenaOf(g)!;
    expect(real.instanceId).not.toBe(first);
    expect(real).toMatchObject({ lifecycle: 'briefing', briefing: { goAt: at + REAL_COUNT_MS, real: true, ready: { p1: true, p2: true } }, mode: 'direct_duel', duelists: ['p2', 'p1'] });
    expect(real.practice).toBeUndefined();
    expect((real.state as { taps?: Record<string, number> }).taps?.p2 ?? 0).toBe(0);     // nothing from practice is kept
    expect(projectGame(g, { kind: 'phone', pid: 'p1' }, at).arenaView).toMatchObject({ lifecycle: 'briefing', briefing: { real: true } });
    expect(offers(g, 'p1', at)).toEqual([]);                                  // nobody presses Ready again
    g = apply(g, auto(g, '|begin|')!.run(g, ctx(at + REAL_COUNT_MS)));
    expect(arenaOf(g)).toMatchObject({ lifecycle: 'active' });
    // and this one counts
    for (let i = 0; i < 5; i++) g = apply(g, tap(g, 'p2', at + REAL_COUNT_MS + 100));
    g = apply(g, auto(g, '|handoff')!.run(g, ctx(at + 60_000)));
    expect(g.currentTurn).toMatchObject({ phase: 'duel', duelGameWinner: 'p2' });
  });

  it('a practice that ends by itself decides nothing: no hand-off, the real game is set up a moment later', () => {
    let g = apply(duel('tap_race'), startDuelApp(duel('tap_race'), ctx()));
    for (let i = 0; i < 5; i++) g = apply(g, tap(g, 'p1'));
    expect(arenaOf(g)).toMatchObject({ lifecycle: 'completed', result: { winners: ['p1'] }, endedAt: NOW });
    expect(handoff(g).error).toMatch(/only practice/);
    expect(auto(g, '|handoff')).toBeUndefined();
    expect(offers(g, HOST, NOW).map((o) => o.type)).toEqual(['arena_skip_practice']);
    const step = auto(g, '|real|')!;
    expect(step.at).toBe(NOW + PRACTICE_END_MS);
    g = apply(g, step.run(g, ctx(NOW + PRACTICE_END_MS)));
    expect(arenaOf(g)).toMatchObject({ lifecycle: 'briefing', briefing: { real: true } });
    expect(arenaOf(g)!.result).toBeUndefined();
    expect(g.currentTurn?.duelGameWinner).toBeUndefined();
  });

  it('only the first time in a night; not for a game with a practice of its own; not when the house switched it off', () => {
    const again = { ...duel('tap_race'), triedGames: { tap_race: true as const } };
    expect(arenaOf(apply(again, startDuelApp(again, ctx())))!.practice).toBeUndefined();
    const off = duel('tap_race', 2, { practiceRound: false });
    expect(arenaOf(apply(off, startDuelApp(off, ctx())))!.practice).toBeUndefined();
    const rings = duel('reindeer_ring_toss');
    expect(arenaOf(apply(rings, startDuelApp(rings, ctx())))!.practice).toBeUndefined();
    expect(config({}).practiceRound).toBe(true);
    expect(sanitizeSettings({ practiceRound: false, pace: 'relaxed' })).toEqual({ practiceRound: false, pace: 'relaxed' });
    expect(sanitizeSettings({ practiceRound: 'no', pace: 'frantic' })).toEqual({});
  });

  it('with the instructions on, practice time starts when the game does; a pause stops its clock too', () => {
    const g0 = duel('tap_race', 2, { readyCheck: true });
    let g = apply(g0, startDuelApp(g0, ctx()));
    expect(arenaOf(g)).toMatchObject({ lifecycle: 'briefing', practice: { on: true } });
    expect(arenaOf(g)!.practice!.endsAt).toBeUndefined();
    expect(auto(g, '|real|')).toBeUndefined();
    for (const p of ['p1', 'p2']) g = apply(g, perform(g, p, { type: 'arena_ready' }, ctx(NOW + 2000)));
    const go = NOW + 2000 + READY_COUNT_MS;
    g = apply(g, auto(g, '|begin|')!.run(g, ctx(go)));
    expect(arenaOf(g)!.practice).toEqual({ on: true, endsAt: go + PRACTICE_MS });
    g = apply(g, arenaPause(g, go + 5000));
    expect(auto(g, '|real|')).toBeUndefined();                                // nothing happens while paused
    g = apply(g, arenaResume(g, go + 15_000));
    expect(arenaOf(g)!.practice).toEqual({ on: true, endsAt: go + PRACTICE_MS + 10_000 });
  });

  it('the host can cut practice short (logged); a game the host stops is stopped, not played again', () => {
    let g = apply(duel('tap_race'), startDuelApp(duel('tap_race'), ctx()));
    expect(offers(g, HOST, NOW).map((o) => o.type)).toContain('arena_skip_practice');
    expect(perform(g, 'p1', { type: 'arena_skip_practice' }, ctx()).error).toBeTruthy();
    const cut = apply(g, perform(g, HOST, { type: 'arena_skip_practice' }, ctx(NOW + 3000)));
    expect(arenaOf(cut)).toMatchObject({ lifecycle: 'briefing', briefing: { real: true, goAt: NOW + 3000 + REAL_COUNT_MS }, overrides: [{ what: 'practice cut short' }] });
    g = apply(g, arenaStop(g, 'cancelled', NOW + 1000));
    expect(auto(g, '|real|')).toBeUndefined();
    g = apply(g, auto(g, '|handoff')!.run(g, ctx(NOW + 1000)));
    expect(g.currentTurn).toMatchObject({ phase: 'duel', duelGameOutcome: { status: 'no_contest' } });
  });

  it('the Game Lab is practice already: no practice round there', () => {
    const g = { ...table(2, { practiceRound: true }), phase: 'lobby' as const };
    const lab = apply(g, startLab(g, { gameId: 'tap_race', players: ['p1', 'p2'] }, ctx()));
    expect(arenaOf(lab)!.practice).toBeUndefined();
  });
});

describe('a relaxed pace', () => {
  it('each game that has a "more time" gets it; whatever the caller chose still wins; a normal pace changes nothing', () => {
    expect(pacedConfig({}, 'gift_conveyor', { teams: 1 })).toEqual({ teams: 1 });
    expect(pacedConfig({ pace: 'relaxed' }, 'gift_conveyor', { teams: 1 })).toMatchObject({ teams: 1, baseSpeed: 0.08, roundMs: 110_000 });
    expect(pacedConfig({ pace: 'relaxed' }, 'gift_conveyor', { roundMs: 5 })).toMatchObject({ roundMs: 5 });
    expect(pacedConfig({ pace: 'relaxed' }, 'nutcracker_knockout', undefined)).toBeUndefined();   // a boxing match is no easier slowed down
    expect(pacedMs({ pace: 'relaxed' }, 60_000)).toBe(90_000);
    expect(pacedMs({}, 60_000)).toBe(60_000);
  });

  it('a team game started for a duel at a relaxed pace has the slower belt, and Mecha Santa the gentler lean', () => {
    const g = duel('gift_conveyor', 4, { pace: 'relaxed', practiceRound: false });
    const cfg = arenaOf(apply(g, startDuelApp(g, ctx())))!.config as typeof CONVEYOR;
    expect(cfg.baseSpeed).toBeLessThan(CONVEYOR.baseSpeed);
    expect(cfg.roundMs).toBeGreaterThan(CONVEYOR.roundMs);
    const m = duel('mecha_santa', 4, { pace: 'relaxed', practiceRound: false });
    const walk = (arenaOf(apply(m, startDuelApp(m, ctx())))!.config as { walk: typeof DEFAULT_WALK }).walk;
    expect(walk.leanRate).toBeLessThan(DEFAULT_WALK.leanRate);
    expect(walk.stride).toBe(DEFAULT_WALK.stride);                            // everything else as it was
    const n = duel('gift_conveyor', 4, { practiceRound: false });
    expect((arenaOf(apply(n, startDuelApp(n, ctx())))!.config as typeof CONVEYOR).baseSpeed).toBe(CONVEYOR.baseSpeed);
  });

  it('Sketch-Off gives half as long again to draw and to vote; Whack-an-Elf is the same length (its elves are timed to it)', () => {
    const sketch = (settings: Settings) => {
      const g = table(3, { readyCheck: false, ...settings });
      g.currentTurn = { phase: 'tiebreaker_pending' };
      const tx = new Tx(g);
      startPartyGame(tx, 'sketch_off', 'pick_one', ['p1', 'p2'], ctx());
      return apply(apply(g, tx.plan()), advanceParty(apply(g, tx.plan()), 'intro', NOW + 100));
    };
    expect(sketch({}).minigame).toMatchObject({ phase: 'playing', phaseEndsAt: NOW + 100 + GAMES.sketch_off.playMs });
    const slow = sketch({ pace: 'relaxed' });
    expect(slow.minigame).toMatchObject({ phaseEndsAt: NOW + 100 + GAMES.sketch_off.playMs * 1.5 });
    expect(apply(slow, advanceParty(slow, 'playing', NOW + 200)).minigame).toMatchObject({ phase: 'voting', phaseEndsAt: NOW + 200 + GAMES.sketch_off.voteMs! * 1.5 });
    const g = table(3, { readyCheck: false, pace: 'relaxed' });
    g.currentTurn = { phase: 'tiebreaker_pending' };
    const tx = new Tx(g);
    startPartyGame(tx, 'whack_elf', 'pick_one', ['p1', 'p2'], ctx());
    const w = apply(g, tx.plan());
    expect(apply(w, advanceParty(w, 'intro', NOW + 100)).minigame).toMatchObject({ phaseEndsAt: NOW + 100 + GAMES.whack_elf.playMs });
  });
});

describe('a stand-in for a teammate whose phone dropped', () => {
  const teamGame = () => { const g = duel('mecha_santa', 4, { practiceRound: false }); return apply(g, startDuelApp(g, ctx())); };

  it('in a running team game the room plays a dropped player until they are back; never anyone who is here', () => {
    let g = teamGame();
    expect(standIns(g)).toEqual([]);
    expect(botsBusy(g)).toBe(false);
    expect(botMove(g, 'p3', ctx())).toBeNull();                               // p3 is here: nobody presses for them
    g = { ...g, players: { ...g.players, p3: { ...g.players!.p3, connected: false } } };
    expect(standIns(g)).toEqual(['p3']);
    expect(roomPlayed(g)).toEqual(['p3']);
    expect(botsBusy(g)).toBe(true);                                           // so the room keeps its tick going
    let pressed = 0;
    for (let i = 0; i < 400 && !pressed; i++) {
      const at = NOW + 1000 + i * 150;
      for (const s of hostAuto(g)) if (s.at <= at) { const due = s.run(g, { now: at, rng: rng(i + 1) }); if (!due.error) g = apply(g, due); }   // the game's own clock
      const plan = botMove(g, 'p3', { now: at, rng: rng(i + 1) });
      if (plan && !plan.error) pressed++;
      // (their teammate, who is here, keeps their own leg going: a simulated hand presses for them in this test)
      for (const mate of ['p1', 'p2', 'p4']) { const m = arenaOffers(g, mate, at, rng(i + 7))[0]; if (m && !pressed) { const done = arenaAct(g, mate, { action: m }, { now: at, rng: rng(i) }); if (!done.error) g = apply(g, done); } }
    }
    expect(pressed).toBe(1);                                                  // their job gets done
    g = { ...g, players: { ...g.players, p3: { ...g.players!.p3, connected: true } } };
    expect(standIns(g)).toEqual([]);
  });

  it('only team games, and only while one is running: not a duel between two, not the instructions, not the bidding', () => {
    let two = apply(duel('tap_race', 2, { practiceRound: false }), startDuelApp(duel('tap_race', 2, { practiceRound: false }), ctx()));
    two = { ...two, players: { ...two.players, p1: { ...two.players!.p1, connected: false } } };
    expect(standIns(two)).toEqual([]);                                        // a duelist's own duel is theirs to lose
    const b0 = duel('mecha_santa', 4, { practiceRound: false, readyCheck: true });
    let briefing = apply(b0, startDuelApp(b0, ctx()));
    briefing = { ...briefing, players: { ...briefing.players, p3: { ...briefing.players!.p3, connected: false } } };
    expect(standIns(briefing)).toEqual([]);
    const idle = table(4);
    idle.players!.p3.connected = false;
    expect(standIns(idle)).toEqual([]);
  });
});

describe('play again with the same people', () => {
  function finished(): Game {
    const g = table(3, { startingChips: 8 });
    g.phase = 'ended';
    g.gifts = { gA: { name: 'Socks', ownerId: 'p1', naughtyLevel: 2 } };
    g.players!.p1 = { ...g.players!.p1, giftId: 'gA', chips: 3, duelWins: 2, partyPoints: 5, avatar: 'fox', age: 9, ownerUid: 'u1', joinedAt: 5 };
    g.players!.p2 = { ...g.players!.p2, chips: 0, duelLosses: 2, connected: false };
    g.lastDuel = { winnerId: 'p1', loserId: 'p2', pot: 2, dividend: 1, loserPaid: 1, at: NOW };
    g.teamPairs = { 'p1|p2': 1 }; g.triedGames = { tap_race: true }; g.betPool = 2;
    return g;
  }

  it('the host sends everyone back to the lobby: same seats, names and avatars; gifts, chips and scores start again', () => {
    const g = apply(finished(), dispatch(finished(), HOST, { type: 'admin_play_again' }, ctx()));
    expect(g.phase).toBe('lobby');
    expect(g.gifts ?? {}).toEqual({});
    expect(g.players!.p1).toEqual({ name: finished().players!.p1.name, chips: 8, connected: true, avatar: 'fox', age: 9, ownerUid: 'u1', joinedAt: 5 });
    expect(g.players!.p2).toMatchObject({ chips: 8, connected: false });      // still away: their seat is kept
    expect(g.players!.p2.duelLosses).toBeUndefined();
    for (const k of ['lastDuel', 'teamPairs', 'triedGames', 'betPool', 'currentTurn'] as const) expect(g[k] ?? null).toBeNull();
    expect(offers(g, HOST, NOW).map((o) => o.type)).toContain('start');        // and the host can start the next one
  });

  it('only the host, and only once the game is over', () => {
    expect(dispatch(finished(), 'p1', { type: 'admin_play_again' }, ctx()).error).toMatch(/Only the host/);
    expect(playAgain(table(3)).error).toMatch(/not over/);
    const again = apply(finished(), playAgain(finished()));
    expect(playAgain(again).error).toMatch(/not over/);                       // a double tap does nothing
  });
});
