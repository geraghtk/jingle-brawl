// The Game Lab: any registered mini-game from the lobby, nothing at stake. The host picks the game,
// the players, the mode and the duelists; the result stays on the screens (no hand-off, no Party
// points); play again or back to the lobby; the real game can't start meanwhile.
import { beforeAll, describe, expect, it } from 'vitest';
import { hostAuto } from '../hostauto';
import { parseIntent } from '../intents';
import { projectGame } from '../projection';
import { rng } from '../rng';
import { HOST, isOffered, offers, perform } from '../table';
import { apply, NOW, table } from '../testkit';
import type { Game } from '../types';
import { defaultLabMode, labModes, labRunning } from './lab';
import { getDef, register } from './registry';
import { arenaAct, arenaOf } from './runtime';
import { tapRace } from './testgames/tapRace';

beforeAll(() => register(tapRace));
const ctx = (now = NOW) => ({ now, rng: rng(7) });
const lobby = (n: number): Game => ({ ...table(n), phase: 'lobby' });
const lab = (g: Game, o: { gameId: string; players: string[]; mode?: string; duelists?: string[] }, now = NOW) => perform(g, HOST, { type: 'lab_start', ...o }, ctx(now));

describe('the Game Lab', () => {
  it('starts Ring Toss from the lobby: a direct duel for two, on the phones, nothing else changes', () => {
    const g0 = lobby(2);
    const g = apply(g0, lab(g0, { gameId: 'reindeer_ring_toss', players: ['p1', 'p2'] }));
    expect(g.phase).toBe('lobby');
    expect(g.currentTurn).toMatchObject({ phase: 'minigame_arena', lab: { gameId: 'reindeer_ring_toss', mode: 'direct_duel', duelists: ['p1', 'p2'] } });
    expect(arenaOf(g)).toMatchObject({ gameId: 'reindeer_ring_toss', mode: 'direct_duel', lifecycle: 'active' });
    expect(projectGame(g, { kind: 'phone', pid: 'p1' }, NOW).arenaView?.gameId).toBe('reindeer_ring_toss');
    expect(offers(g, 'p1', NOW).some((o) => o.type === 'arena')).toBe(true);
    expect(g.players).toEqual(g0.players);                       // chips untouched
  });

  it('one player can practise alone (a free-for-all), and the host picks the mode and duelists', () => {
    const g0 = lobby(4);
    const solo = apply(g0, lab(g0, { gameId: 'reindeer_ring_toss', players: ['p3'] }));
    expect(arenaOf(solo)).toMatchObject({ mode: 'free_for_all', participants: ['p3'] });
    expect(arenaOf(solo)!.duelists).toBeUndefined();
    const party = apply(g0, lab(g0, { gameId: 'pass_the_gremlin', players: ['p1', 'p2', 'p3', 'p4'], mode: 'party_with_duelists', duelists: ['p4', 'p2'] }));
    expect(arenaOf(party)).toMatchObject({ mode: 'party_with_duelists', duelists: ['p4', 'p2'], participants: ['p1', 'p2', 'p3', 'p4'] });
    const ffa = apply(g0, lab(g0, { gameId: 'pass_the_gremlin', players: ['p1', 'p2', 'p3'], mode: 'free_for_all' }));
    expect(arenaOf(ffa)!.mode).toBe('free_for_all');
  });

  it('the Gift Conveyor: the duelists captain, the lab deals the teams and does not remember them', () => {
    const g0 = lobby(4);
    const g = apply(g0, lab(g0, { gameId: 'gift_conveyor', players: ['p1', 'p2', 'p3', 'p4'], duelists: ['p3', 'p1'] }));
    const s = arenaOf(g)!.state as { teams: { captainId: string; members: string[] }[] };
    expect(s.teams.map((t) => t.captainId)).toEqual(['p3', 'p1']);
    expect(s.teams.flatMap((t) => t.members).sort()).toEqual(['p1', 'p2', 'p3', 'p4']);
    expect(g.teamPairs).toBeUndefined();
  });

  it('modes by table size, and refusals: not outside the lobby, not over a running game, bad picks', () => {
    const ring = getDef('reindeer_ring_toss')!;
    expect(labModes(ring, 1)).toEqual(['free_for_all']);
    expect(defaultLabMode(ring, 2)).toBe('direct_duel');
    expect(defaultLabMode(ring, 5)).toBe('party_with_duelists');
    expect(defaultLabMode(getDef('gift_conveyor')!, 1)).toBeUndefined();

    const main = table(2);
    expect(lab(main, { gameId: 'reindeer_ring_toss', players: ['p1', 'p2'] }).error).toBeTruthy();
    const g0 = lobby(3);
    expect(lab(g0, { gameId: 'nope', players: ['p1'] }).error).toMatch(/No such game/);
    expect(lab(g0, { gameId: 'reindeer_ring_toss', players: ['p1', 'p9'] }).error).toMatch(/in the room/);
    expect(lab(g0, { gameId: 'reindeer_ring_toss', players: ['p1', 'p2', 'p3'], mode: 'direct_duel' }).error).toMatch(/can't be played/);
    expect(lab(g0, { gameId: 'pass_the_gremlin', players: ['p1', 'p2', 'p3'], mode: 'party_with_duelists', duelists: ['p1', 'p1'] }).error).toMatch(/two different/);
    expect(lab(g0, { gameId: 'gift_conveyor', players: ['p1'] }).error).toMatch(/can't be played/);
    const g = apply(g0, lab(g0, { gameId: 'reindeer_ring_toss', players: ['p1', 'p2'] }));
    expect(lab(g, { gameId: 'reindeer_ring_toss', players: ['p1', 'p2'] }).error).toMatch(/still running/);
    expect(perform(g, HOST, { type: 'start' }, ctx()).error).toBeTruthy();            // not offered meanwhile
    expect(offers(g, HOST, NOW)).toEqual([{ type: 'arena_pause' }]);
    expect(isOffered(g, 'p1', { type: 'lab_end' }, NOW)).toBe(false);                // the host's only
  });

  it('the result stays (no hand-off, no Party points); play again; back to the lobby; then the real game', () => {
    const g0 = lobby(2);
    let g = apply(g0, lab(g0, { gameId: 'tap_race', players: ['p1', 'p2'] }));
    for (let i = 0; i < 5; i++) g = apply(g, arenaAct(g, 'p2', { action: { tap: true } }, ctx()));
    expect(arenaOf(g)!.lifecycle).toBe('completed');
    expect(arenaOf(g)!.result?.duel?.winningDuelistId).toBe('p2');
    expect(labRunning(g)).toBe(false);
    expect(hostAuto(g).filter((s) => s.key.includes('handoff'))).toEqual([]);         // the result stays on screen
    expect(g.players?.p2.partyPoints).toBeUndefined();

    const again = apply(g, lab(g, { gameId: 'tap_race', players: ['p1', 'p2'] }, NOW + 5000));
    expect(arenaOf(again)).toMatchObject({ lifecycle: 'active', stateVersion: 1 });
    expect(arenaOf(again)!.instanceId).not.toBe(arenaOf(g)!.instanceId);

    const back = apply(again, perform(again, HOST, { type: 'lab_end' }, ctx()));      // stops a running one too
    expect(back.currentTurn ?? null).toBeNull();
    expect(back.arena ?? null).toBeNull();
    expect(offers(back, HOST, NOW)).toEqual([{ type: 'start' }]);
    expect(apply(back, perform(back, HOST, { type: 'start' }, ctx())).phase).toBe('main');
  });

  it('the wire: lab_start is checked for shape', () => {
    expect(parseIntent({ type: 'lab_start', gameId: 'reindeer_ring_toss', players: ['p1', 'p2'], mode: 'direct_duel' })).not.toHaveProperty('error');
    expect(parseIntent({ type: 'lab_start', gameId: 'x', players: [] })).toHaveProperty('error');
    expect(parseIntent({ type: 'lab_start', gameId: 'x', players: ['p1'], mode: 'chaos' })).toHaveProperty('error');
    expect(parseIntent({ type: 'lab_start', gameId: 'x', players: ['<b>'] })).toHaveProperty('error');
    expect(parseIntent({ type: 'lab_end' })).not.toHaveProperty('error');
  });
});
