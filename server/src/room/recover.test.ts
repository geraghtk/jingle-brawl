// After a restart the server loads its recent rooms (Rooms.recover) so their clocks run again with
// nobody connected: a party game's phase still ends, the Misfit Lottery still draws. Rooms nobody
// has touched for a day stay in the store until someone comes back.
import { describe, expect, it } from 'vitest';
import type { PartyGame } from '../../../web/src/core/party';
import { HOST } from '../../../web/src/core/table';
import { MemoryStore } from '../store/store';
import { FakeClock } from '../testing/fakeClock';
import { RECOVER_WITHIN_MS, Rooms } from './rooms';

const HOUR = 60 * 60 * 1000;

function server(store = new MemoryStore(), at?: number) {
  const clock = new FakeClock();
  if (at != null) clock.t = at;
  return { store, clock, rooms: new Rooms(store, clock, { secret: 'test', seed: () => 1 }) };
}

/** A room with four players at a tied bid: its party game is about to start by itself. */
function tied(rooms: Rooms) {
  const { code } = rooms.create({ duelPick: 'off', tiebreakerMethod: 'whack_elf' });
  const room = rooms.get(code)!;
  const pids = ['Ada', 'Bo', 'Cy', 'Di'].map((name) => (room.join({ name }) as { pid: string }).pid);
  let n = 0;
  const press = (who: string, intent: unknown) => expect(room.handle(who, `a${++n}`, intent)).toEqual({ ok: true });
  press(HOST, { type: 'start' });
  press(HOST, { type: 'draw', giftName: 'Socks' });
  const opener = room.state.game.currentTurn!.openerId!;
  for (const p of pids.filter((x) => x !== opener)) press(p, { type: 'bid', amount: 2 });
  press(HOST, { type: 'reveal' });
  press(opener, { type: 'fight' });
  expect(room.state.game.currentTurn!.phase).toBe('tiebreaker_pending');
  return code;
}

describe('recovering rooms after a restart', () => {
  it('loads the recent rooms and their clocks run with nobody connected', () => {
    const a = server();
    const code = tied(a.rooms);
    const lobby = a.rooms.create({}).code;
    // the process dies (no close, no timers) and starts again on the same store
    const b = server(a.store, a.clock.now() + 5000);
    expect(b.rooms.size).toBe(0);
    expect(b.rooms.recover()).toBe(2);
    expect(b.rooms.size).toBe(2);
    b.clock.advance(0);
    const game = () => b.rooms.get(code)!.state.game;
    expect(game().currentTurn!.phase).toBe('minigame_party');               // the tie's step came back from the state
    b.clock.advance((game().minigame as PartyGame).phaseEndsAt - b.clock.now() + 1000);
    expect((game().minigame as PartyGame).phase).toBe('playing');
    b.clock.advance(5 * 60_000);
    expect(game().minigame ?? null).toBeNull();                             // played out (nobody whacked) and handed back
    expect(Object.values(game().players!).every((p) => p.connected === false)).toBe(true);
    expect(b.rooms.get(lobby)!.state.game.phase).toBe('lobby');
    expect(b.store.load(code)!.version).toBe(b.rooms.get(code)!.state.version);
  });

  it('a Misfit Lottery caught between two turns draws its next player once recovered', () => {
    const a = server();
    const { code } = a.rooms.create({ duelPick: 'off' });
    const room = a.rooms.get(code)!;
    const [x, y, z] = ['Ada', 'Bo', 'Cy'].map((name) => (room.join({ name }) as { pid: string }).pid);
    room.handle(HOST, 's', { type: 'start' });
    room.debugWrite({
      gifts: { g1: { name: 'Socks', ownerId: x, naughtyLevel: 0 }, g2: { name: 'Mug', ownerId: y, naughtyLevel: 0 }, g3: { name: 'Candle', ownerId: 'misfit', naughtyLevel: 0, misfitOrder: 1 } },
      [`players/${x}/giftId`]: 'g1', [`players/${y}/giftId`]: 'g2',
    });
    expect(room.handle(HOST, 'e', { type: 'endgame' })).toEqual({ ok: true });
    expect(room.state.game.currentTurn ?? null).toBeNull();
    const b = server(a.store, a.clock.now() + 60_000);
    expect(b.rooms.recover()).toBe(1);
    b.clock.advance(0);
    expect(b.rooms.get(code)!.state.game.currentTurn).toMatchObject({ phase: 'endgame_choice', activePlayerId: z });
  });

  it('leaves rooms untouched for a day in the store: not loaded, not lost, back when someone returns', () => {
    const a = server();
    const old = a.rooms.create({}).code;
    a.clock.advance(RECOVER_WITHIN_MS + HOUR);
    const recent = a.rooms.create({}).code;
    const b = server(a.store, a.clock.now() + 1000);
    expect(b.rooms.recover()).toBe(1);
    expect(b.rooms.size).toBe(1);
    expect(b.rooms.get(recent)).not.toBeNull();
    expect(b.store.load(old)).not.toBeNull();
    expect(b.rooms.get(old)!.state.game.phase).toBe('lobby');               // a screen comes back: loaded on demand
    expect(b.rooms.size).toBe(2);
  });

  it('nothing to recover is not an error, and recovering twice loads nothing twice', () => {
    expect(server().rooms.recover()).toBe(0);
    const a = server();
    const code = a.rooms.create({}).code;
    const b = server(a.store, a.clock.now());
    expect(b.rooms.recover()).toBe(1);
    const room = b.rooms.get(code);
    expect(b.rooms.recover()).toBe(1);
    expect(b.rooms.get(code)).toBe(room);
    expect(b.rooms.size).toBe(1);
  });

  it('the token secret is kept with the rooms: a seat from before the restart still opens its room', () => {
    const store = new MemoryStore();
    const a = new Rooms(store, new FakeClock());
    const b = new Rooms(store, new FakeClock());
    expect(b.secret).toBe(a.secret);
    expect(a.secret.length).toBeGreaterThanOrEqual(32);
  });
});
