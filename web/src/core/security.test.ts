// What a stranger can send: anyone can create a room, so its host console is untrusted too. An id is a
// key into players / gifts, and every object already "has" __proto__, constructor and toString: none of
// them is a player, and no update path may ever reach a prototype (it would change every room at once).
import { afterEach, describe, expect, it } from 'vitest';
import * as Admin from './admin';
import { dispatch, parseIntent } from './intents';
import { GAMES, whackBounds, whackSequence } from './party';
import type { PartyGame } from './party';
import { advanceParty, startPartyGame } from './partyflow';
import { rng } from './rng';
import { HOST } from './table';
import { apply, NOW, table } from './testkit';
import { applyUpdates, getPath, Tx } from './tx';
import type { Game } from './types';

const ctx = { now: NOW, rng: rng(1) };
const NAMES = ['__proto__', 'constructor', 'prototype'];
const INHERITED = ['toString', 'hasOwnProperty', 'valueOf'];

function withGift(): Game {
  const g = table(3);
  g.gifts = { ga: { name: 'A', ownerId: 'p1', naughtyLevel: 0 } };
  g.players!.p1.giftId = 'ga';
  return g;
}

describe('Whack-an-Elf: a phone counts its own taps, the room holds the score to what was possible', () => {
  /** Three at the table, Whack-an-Elf just started playing at NOW. */
  function playing(): Game {
    const g = table(3);
    g.currentTurn = { phase: 'minigame_party' };
    const tx = new Tx(g);
    startPartyGame(tx, 'whack_elf', 'pick_one', ['p2', 'p3'], ctx);
    return apply(apply(g, tx.plan()), advanceParty(apply(g, tx.plan()), 'intro', NOW));
  }
  const scoreOf = (g: Game, pid: string) => (g.minigame as PartyGame).scores?.[pid]?.score;

  it('a claimed thousand is held to the most the pop-ups so far could score', () => {
    const g = playing();
    const seed = (g.minigame as PartyGame).seed;
    const whole = whackBounds(seed), early = whackBounds(seed, 2000), start = whackBounds(seed, 0);
    expect(whole.max).toBeGreaterThan(early.max);
    expect(early.max).toBeGreaterThan(0);
    expect(start.max).toBeLessThanOrEqual(early.max);
    // two seconds in, through the phone's own button (dispatch → perform → reportWhack)
    const at = { now: NOW + 2000, rng: rng(1) };
    expect(scoreOf(apply(g, dispatch(g, 'p2', { type: 'whack', score: 1000, final: true }, at)), 'p2')).toBe(early.max);
    expect(scoreOf(apply(g, dispatch(g, 'p2', { type: 'whack', score: -1000 }, at)), 'p2')).toBe(early.min);
    // an honest score passes untouched
    expect(scoreOf(apply(g, dispatch(g, 'p3', { type: 'whack', score: 2 }, at)), 'p3')).toBe(2);
    // at the end, the whole game's best is the ceiling
    const end = { now: NOW + GAMES.whack_elf.playMs - 1, rng: rng(1) };
    expect(scoreOf(apply(g, dispatch(g, 'p2', { type: 'whack', score: 1000, final: true }, end)), 'p2')).toBe(whole.max);
  });

  it('the ceiling is every elf at its fastest and no Grinch; the floor is every Grinch', () => {
    const seq = whackSequence(42);
    const b = whackBounds(42);
    expect(b.max).toBe(seq.filter((p) => p.kind === 'elf').length * 3 + seq.filter((p) => p.kind === 'gold').length * 6);
    expect(b.min).toBe(seq.filter((p) => p.kind === 'grinch').length * -3);
  });
});

describe('ids and paths from the network never reach a prototype', () => {
  afterEach(() => {
    // whatever a test did, the next one starts clean (and a failure here says which test leaked)
    for (const k of ['giftId', 'chips', 'name', 'x']) delete (Object.prototype as Record<string, unknown>)[k];
  });
  const clean = () => { for (const k of ['giftId', 'chips', 'name', 'x']) expect(({} as Record<string, unknown>)[k], `Object.prototype.${k}`).toBeUndefined(); };

  it('an intent whose id is a prototype name is not an intent', () => {
    for (const bad of NAMES) {
      expect(parseIntent({ type: 'admin_set_chips', pid: bad, chips: 5 })).toHaveProperty('error');
      expect(parseIntent({ type: 'admin_add_gift', name: 'x', ownerId: bad })).toHaveProperty('error');
      expect(parseIntent({ type: 'admin_save_gift', giftId: bad, naughtyLevel: 0, ownerId: null })).toHaveProperty('error');
      expect(parseIntent({ type: 'gambit', giftId: bad })).toHaveProperty('error');
      expect(parseIntent({ type: 'lab_start', gameId: 'pass_the_gremlin', players: ['p1', bad] })).toHaveProperty('error');
    }
    expect(parseIntent({ type: 'admin_set_chips', pid: 'p1', chips: 5 })).not.toHaveProperty('error');
  });

  it('the admin actions refuse a player or gift the game does not really have', () => {
    for (const bad of [...NAMES, ...INHERITED]) {
      const g = withGift();
      expect(Admin.setChips(g, bad, 5).error, bad).toBe('No such player');
      expect(Admin.addGift(g, 'g2', 'Socks', bad, NOW).error, bad).toBe('No such player');
      expect(Admin.removePlayerGift(g, bad, NOW).error, bad).toBe('Player has no gift');
      expect(Admin.saveGift(g, bad, { naughtyLevel: 0, ownerId: null }, NOW).error, bad).toBe('No such gift');
      expect(Admin.saveGift(g, 'ga', { naughtyLevel: 0, ownerId: bad }, NOW).error, bad).toBe('No such player');
      expect(Admin.moveGiftToMisfit(g, bad, NOW).error, bad).toBe('No such gift');
      expect(Admin.deleteGift(g, bad).error, bad).toBe('No such gift');
    }
    clean();
  });

  it('through dispatch too: the host console cannot write to every room at once', () => {
    const g = withGift();
    for (const bad of NAMES) {
      // as it would arrive if parsing ever let it through
      const plan = dispatch(g, HOST, { type: 'admin_add_gift', name: 'x', ownerId: bad }, ctx);
      expect(plan.error).toBe('No such player');
      expect(dispatch(g, HOST, { type: 'admin_set_chips', pid: bad, chips: 9 }, ctx).error).toBe('No such player');
    }
    clean();
    expect(table(2).players!.p1.giftId ?? null).toBeNull();          // another room's player holds nothing
  });

  it('re-owning a gift to someone not at the table is refused: no phantom player', () => {
    const g = withGift();
    expect(Admin.saveGift(g, 'ga', { naughtyLevel: 0, ownerId: 'nobody' }, NOW).error).toBe('No such player');
    // 'misfit' as the owner is the Misfit pile, not a player called misfit
    const out = apply(g, Admin.saveGift(g, 'ga', { naughtyLevel: 1, ownerId: 'misfit' }, NOW));
    expect(out.gifts!.ga).toMatchObject({ ownerId: 'misfit', naughtyLevel: 1 });
    expect(Object.keys(out.players!).sort()).toEqual(['p1', 'p2', 'p3']);
    expect(out.players!.p1.giftId ?? null).toBeNull();
  });

  it('an update path through a prototype throws and changes nothing; reads see only own keys', () => {
    const g = table(2);
    for (const bad of NAMES) {
      expect(() => applyUpdates(g, { [`players/${bad}/x`]: 1 })).toThrow(/unsafe update path/);
      expect(() => new Tx(g).set(`players/${bad}/x`, 1)).toThrow(/unsafe update path/);
      expect(() => new Tx(g).set(`${bad}/x`, 1)).toThrow(/unsafe update path/);
    }
    clean();
    expect(getPath(g, 'players/__proto__')).toBeUndefined();
    expect(getPath(g, 'players/toString')).toBeUndefined();
    expect(getPath(g, 'players/p1/name')).toBe(g.players!.p1.name);
    // a key that only looks inherited is written as the game's own, not onto the function it shadows
    const out = applyUpdates(g, { 'players/toString/chips': 3 });
    expect(Object.prototype.hasOwnProperty.call(out.players, 'toString')).toBe(true);
    expect(typeof ({}).toString).toBe('function');
  });
});
