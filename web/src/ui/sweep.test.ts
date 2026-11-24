// Small fixes from the 2026-11-23 polish sweep, each pinned so it does not come back.
import { describe, expect, it } from 'vitest';
import { rng } from '../core/rng';
import { apply, NOW, table } from '../core/testkit';
import { Tx } from '../core/tx';
import type { Game } from '../core/types';
import type { GremlinState } from '../core/minigames/gremlin/gremlin';
import { arenaOf, arenaPause, arenaResume, startArena } from '../core/minigames/runtime';
import { joinPath } from '../host/TurnPanel';
import { splashFor } from '../host/RobotStage';
import type { RobotView } from '../core/minigames/robots/view';
import { theGift } from './notes';

describe('polish sweep fixes', () => {
  it('the join link is the join page from the console and from the TV (the TV\'s QR used to open another TV)', () => {
    expect(joinPath('/host.html')).toBe('/player.html');
    expect(joinPath('/tv.html')).toBe('/player.html');
    expect(joinPath('/party/tv.html')).toBe('/party/player.html');
  });

  it('a gift name that brings its own article keeps it ("takes the A jumper" was wrong)', () => {
    expect(theGift('Fuzzy socks')).toBe('the Fuzzy socks');
    expect(theGift('A hand-knitted reindeer jumper')).toBe('A hand-knitted reindeer jumper');
    expect(theGift('the big one')).toBe('the big one');
    expect(theGift('An umbrella')).toBe('An umbrella');
    expect(theGift('Anorak')).toBe('the Anorak');
  });

  it('Knockout: the stage does not announce the winner a second time under the results ceremony', () => {
    const v = { stage: 'done', order: ['a', 'b'], wins: { a: 2, b: 0 }, stageEndsAt: 0, round: 2, sudden: false } as unknown as RobotView;
    expect(splashFor(v, 0)).toBe('');
    expect(splashFor({ ...v, wins: { a: 1, b: 1 } }, 0)).toBe('Level!');
  });

  it('the Gremlin\'s clocks stand still through a pause (and so through the instructions): the hold timer, the mood, the secret moment', () => {
    const g: Game = table(3);
    g.currentTurn = { phase: 'duel', challengerId: 'p1', defenderId: 'p2', pot: 2 };
    const tx = new Tx(g);
    expect(startArena(tx, 'pass_the_gremlin', { mode: 'party_with_duelists', participants: ['p1', 'p2', 'p3'], duelists: ['p1', 'p2'] }, { now: NOW, rng: rng(3) }).error).toBeUndefined();
    let game = apply(g, tx.plan());
    const before = arenaOf(game)!.state as GremlinState;
    game = apply(game, arenaPause(game, NOW + 2000));
    game = apply(game, arenaResume(game, NOW + 32_000));
    const after = arenaOf(game)!.state as GremlinState;
    expect(after.holdSince).toBe(before.holdSince + 30_000);
    expect(after.roundStartedAt).toBe(before.roundStartedAt + 30_000);
    expect(after.doomAt).toBe(before.doomAt + 30_000);
    expect(after.challenge.issuedAt).toBe(before.challenge.issuedAt + 30_000);
    expect(arenaOf(game)!.deadlines.find((d) => d.id === 'doom')!.at).toBe(after.doomAt);   // the deadline and the state agree
  });
});
