// Only what changed goes over the wire: applying the diff to the old view gives the new one.
import { describe, expect, it } from 'vitest';
import { rng } from './rng';
import { applyUpdates } from './tx';
import { diff } from './diff';
import type { Game } from './types';
import { projectGame } from './projection';
import type { Viewer } from './projection';
import { Table } from '../sim/table';

/** Over the wire and back. */
const wire = <T>(v: T): T => JSON.parse(JSON.stringify(v));
const isMap = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);
/** A view as a screen may rely on it: nulls and empty maps are the same as nothing there (a screen
 *  that had the whole view keeps them, one that was patched doesn't); arrays arrive whole, as sent. */
function seen(v: unknown): unknown {
  if (!isMap(v)) return v;
  const out: Record<string, unknown> = {};
  for (const [k, x] of Object.entries(v)) {
    const s = seen(x);
    if (s == null || (isMap(s) && !Object.keys(s).length)) continue;
    out[k] = s;
  }
  return out;
}
/** One screen: the whole view first, then only patches, each made from the server's last view. */
function screen(first: Game) {
  let sent = first, held = wire(first);
  return {
    get held() { return held; },
    next(view: Game) {
      const patch = wire(diff(sent, view));
      held = applyUpdates(held, patch, 0);
      sent = view;
      return patch;
    },
  };
}

describe('diff', () => {
  it('small changes are small; deletions are nulls', () => {
    const a = { phase: 'main', players: { p1: { name: 'A', chips: 10 }, p2: { name: 'B', chips: 10 } }, currentTurn: { phase: 'bidding', bids: { p2: { amount: 1 } } } } as unknown as Game;
    const b = { phase: 'main', players: { p1: { name: 'A', chips: 9 }, p2: { name: 'B', chips: 10 } } } as unknown as Game;
    expect(diff(a, b)).toEqual({ 'players/p1/chips': 9, currentTurn: null });
    expect(diff(a, a)).toEqual({});
  });
  it('replays a whole simulated party exactly, step by step', () => {
    const people = Array.from({ length: 5 }, (_, i) => ({ id: `p${i}`, name: `G${i}`, skills: { duel: 0.5, whack: 0.5, sketch: 0.5 } }));
    const t = new Table({ seed: 3, people, settings: { tiebreakerMethod: 'sketch_off' }, autoHost: { paceMs: 4000 } });
    const r = rng(3);
    let prev = t.game;
    let steps = 0;
    for (let tick = 0; tick < 3000 && !t.ended; tick++) {
      for (const p of t.people) { const os = t.offers(p.id); if (os.length && r() < 0.5) t.act(p.id, os[Math.floor(r() * os.length)]); }
      t.advance(3000);
      expect(applyUpdates(prev, diff(prev, t.game))).toEqual(t.game);
      prev = t.game;
      steps++;
    }
    expect(steps).toBeGreaterThan(20);
  });

  // A phone's view with a mini-game running (Ring Toss, a tiebreak round starting), built by hand
  // so every awkward change is in one step: maps that empty, keys that go, arrays that shrink to
  // nothing, values that turn falsy.
  const ringToss = (view: Record<string, unknown>, more: Record<string, unknown> = {}) => ({
    phase: 'main',
    players: { p1: { name: 'Ada', chips: 10 }, p2: { name: 'Bo' }, p3: { name: 'Cy' } },
    currentTurn: { phase: 'minigame_arena', openerId: 'p1', bids: { p2: { amount: 0, sealed: true } } },
    arenaView: {
      instanceId: 'a1', gameId: 'reindeer_ring_toss', mode: 'party_with_duelists', lifecycle: 'active',
      participants: ['p1', 'p2', 'p3'], duelists: ['p1', 'p2'], ...more, view,
    },
  }) as unknown as Game;
  const before = ringToss({
    stage: 'scored', stageEndsAt: 9000, pool: ['p1', 'p2', 'p3'], tiebreakRound: 0, me: 'p1', myLeft: 2, myScore: 3, paused: true, note: 'go',
    scores: { p1: 3, p2: 3 }, left: { p1: 2, p2: 0, p3: 3 }, reloadAt: { p2: 8000 },
    got: { p1: { star: 1, bell: 1 }, p2: { holly: 1 } }, sets: { p1: 1 },
    throws: [{ n: 1, pid: 'p1', pegId: 'star', points: 1 }, { n: 2, pid: 'p2', pegId: null, points: 0, extra: {} }],
    teams: [{ captainId: 'p1', members: ['p1', 'p3'], holding: {} }], leader: 'p1', hint: null,
  }, { stateVersion: 7 });
  const after = ringToss({
    stage: 'tiebreak', stageEndsAt: 12000, pool: ['p1', 'p2'], tiebreakRound: 1, me: 'p1', myLeft: 0, myScore: 0, paused: false, note: '',
    scores: {}, left: { p1: 0, p2: 0 }, reloadAt: {},
    got: { p1: {} }, throws: [],
    teams: [], leader: null, hint: 'one ring each',
  }, { stateVersion: 8, result: { status: 'requires_tiebreaker', placements: [], rewards: {}, duel: { winningDuelistId: null } } });

  it('a running mini-game in the view: maps that empty and keys that go vanish, empty arrays and falsy values travel', () => {
    const phone = screen(before);
    const patch = phone.next(after);
    const view = (phone.held.arenaView as { view: Record<string, unknown> }).view;
    expect(seen(phone.held)).toStrictEqual(seen(wire(after)));
    // what a screen has to read defensively (CLAUDE.md): gone, not empty and not null
    for (const k of ['scores', 'reloadAt', 'got', 'sets', 'leader']) expect(k in view, k).toBe(false);
    expect((view.left as object)).toStrictEqual({ p1: 0, p2: 0 });
    // what it may rely on: arrays arrive, empty or not; 0, false and '' are values
    expect(view).toMatchObject({ throws: [], teams: [], pool: ['p1', 'p2'], myLeft: 0, myScore: 0, paused: false, note: '', hint: 'one ring each', tiebreakRound: 1 });
    expect((phone.held.arenaView as { result?: unknown }).result).toMatchObject({ status: 'requires_tiebreaker', placements: [] });
    // and only what changed went: nothing about the players, the turn or who is playing
    expect(Object.keys(patch).filter((p) => !p.startsWith('arenaView/'))).toEqual([]);
    expect(Object.keys(patch).some((p) => /participants|duelists|gameId|instanceId/.test(p))).toBe(false);
    expect(patch).toMatchObject({ 'arenaView/stateVersion': 8, 'arenaView/view/scores/p1': null, 'arenaView/view/got/p2': null, 'arenaView/view/throws': [], 'arenaView/view/myScore': 0 });
  });

  it('and back: maps, keys and arrays reappear from nothing; then the mini-game ends and its view goes', () => {
    const phone = screen(before);
    phone.next(after);
    phone.next(before);                                  // (as a patch onto a screen that holds no empty maps)
    expect(seen(phone.held)).toStrictEqual(seen(wire(before)));
    expect((phone.held.arenaView as { view: { throws: unknown[] } }).view.throws[1]).toStrictEqual({ n: 2, pid: 'p2', pegId: null, points: 0, extra: {} });   // inside an array: as sent
    const over = { phase: 'main', players: (before as unknown as { players: unknown }).players, arenaView: null, currentTurn: {} } as unknown as Game;
    expect(phone.next(over)).toEqual({ arenaView: null, 'currentTurn/phase': null, 'currentTurn/openerId': null, 'currentTurn/bids': null });
    expect(phone.held).toStrictEqual({ phase: 'main', players: wire((before as unknown as { players: unknown }).players) });
    // a screen that joins now is sent the whole view: the same to anything reading it defensively
    expect(seen(wire(over))).toStrictEqual(phone.held);
  });

  it('every screen of a party with in-app duel games holds its own projected view, patch after patch', () => {
    const people = Array.from({ length: 4 }, (_, i) => ({ id: `p${i}`, name: `G${i}`, skills: { duel: 0.5, whack: 0.5, sketch: 0.5 } }));
    const viewers: Viewer[] = [{ kind: 'host' }, { kind: 'tv' }, ...people.map((p) => ({ kind: 'phone' as const, pid: p.id }))];
    const played = new Set<string>();
    let arenaSteps = 0;
    for (const seed of [1, 2, 3]) {
      const t = new Table({ seed, people, settings: { duelGames: [], duelAppGames: ['pass_the_gremlin', 'reindeer_ring_toss', 'gift_conveyor'], duelPick: 'wheel' }, autoHost: { paceMs: 4000 } });
      const r = rng(seed);
      const screens = viewers.map((v) => screen(projectGame(t.game, v, t.now)));
      for (let tick = 0; tick < 1500 && !t.ended; tick++) {
        for (const p of t.people) { const os = t.offers(p.id); if (os.length && r() < 0.6) t.act(p.id, os[Math.floor(r() * os.length)]); }
        t.advance(1500);
        viewers.forEach((v, i) => {
          const view = projectGame(t.game, v, t.now);
          screens[i].next(view);
          expect(seen(screens[i].held)).toStrictEqual(seen(wire(view)));
        });
        if (t.game.arena) { arenaSteps++; played.add((t.game.arena as { gameId: string }).gameId); }
      }
    }
    expect([...played].sort()).toEqual(['gift_conveyor', 'pass_the_gremlin', 'reindeer_ring_toss']);
    expect(arenaSteps).toBeGreaterThan(100);
  });
});
