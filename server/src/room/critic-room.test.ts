// AI Draw Off in a room: the host's own key goes to the room's memory only, judges the duel, and
// is forgotten when the game ends. It never reaches the log, the store, or any screen.
import { describe, expect, it } from 'vitest';
import type { DrawOff } from '../../../web/src/core/drawoff';
import { HOST } from '../../../web/src/core/table';
import type { ServerMsg } from '../../../web/src/net/protocol';
import type { Critic } from '../critic/critic';
import { MemoryStore } from '../store/store';
import { FakeClock } from '../testing/fakeClock';
import { RoomActor } from './actor';
import type { Conn, CriticSource } from './actor';

const KEY = 'sk-SECRET-key-123456';

function spyCritics(opts: { badKey?: boolean } = {}) {
  const keys: string[] = [];
  const source: CriticSource = {
    make(_provider, key) {
      keys.push(key);
      const c: Critic = {
        backend: 'openai',
        async check() { if (opts.badKey) throw new Error(`Incorrect API key provided: ${key}`); },
        async judge(image) { return { rating: image.length % 101, description: 'a doodle', critique: 'Brave.', backend: 'openai', model: 'm' }; },
        async verdict(_p, winner) { return `${winner} wins!`; },
      };
      return c;
    },
    fallback: () => null,
  };
  return { source, keys };
}

const flush = async () => { for (let i = 0; i < 10; i++) await new Promise((r) => setTimeout(r, 0)); };

function setup(critics: CriticSource) {
  const store = new MemoryStore();
  const clock = new FakeClock();
  let s = 1;
  const room = RoomActor.create('DRAW', { duelPick: 'wheel', duelGames: [], duelAppGames: ['draw_off'], voteWindow: 0, aiCriticBackend: 'openai' },
    { store, clock, seed: () => s++, critics })!;
  const sent: ServerMsg[] = [];
  const screen = (viewer: Conn['viewer']) => { const conn: Conn = { viewer, send: (m) => sent.push(JSON.parse(JSON.stringify(m))) }; room.attach(conn); return conn; };
  const host = screen({ kind: 'host' });
  const pids = ['Ada', 'Bo', 'Cy'].map((name) => { const r = room.join({ name }); if ('error' in r) throw new Error(r.error); screen({ kind: 'phone', pid: r.pid }); return r.pid; });
  screen({ kind: 'tv' });
  let n = 0;
  const press = (who: string, intent: unknown) => room.handle(who, `a${++n}`, intent);
  return { store, clock, room, host, pids, sent, press, game: () => room.state.game };
}

/** Into a Draw Off duel between the first two players (the wheel picks it: the only option). */
function intoDrawOff(t: ReturnType<typeof setup>) {
  expect(t.press(HOST, { type: 'start' })).toEqual({ ok: true });
  t.room.debugWrite({ currentTurn: { phase: 'duel', challengerId: t.pids[0], defenderId: t.pids[1], pot: 2 } });
  t.clock.advance(10_000);                                          // the wheel spins and stops
  expect(t.game().currentTurn?.duelGame?.app).toBe('draw_off');
  expect(t.press(HOST, { type: 'start_duel_app' })).toEqual({ ok: true });
  const mg = () => t.game().minigame as DrawOff;
  t.clock.advance(mg().promptRevealEndTime - t.clock.now());
  expect(mg().phase).toBe('drawing');
  t.press(t.pids[0], { type: 'sketch', drawing: `data:image/jpeg;base64,${'A'.repeat(60)}` });   // scores 83 (length mod 101)
  t.press(t.pids[1], { type: 'sketch', drawing: `data:image/jpeg;base64,${'B'.repeat(10)}` });   // scores 33
  t.clock.advance(0);
  return mg;
}

describe('AI Draw Off in a room', () => {
  it("the host's key judges the duel; it never reaches the log, the store or any screen; it's forgotten at the end", async () => {
    const { source, keys } = spyCritics();
    const t = setup(source);
    t.room.setCriticKey(KEY);
    await flush();
    expect(t.sent.filter((m) => m.t === 'critic').at(-1)).toMatchObject({ ready: true, backend: 'openai' });
    const mg = intoDrawOff(t);
    await flush();                                                  // the critic answers
    expect(mg()).toMatchObject({ phase: 'results', winners: [t.pids[0]] });
    t.clock.advance(60_000);                                        // results → the duel resolves
    expect(t.game().lastDuel).toMatchObject({ winnerId: t.pids[0] });
    expect(keys).toEqual([KEY]);
    expect(JSON.stringify(t.sent)).not.toContain(KEY);
    expect(JSON.stringify(t.store.events('DRAW', 0, 1e9))).not.toContain(KEY);
    expect(JSON.stringify(t.store.load('DRAW'))).not.toContain(KEY);
    t.press(HOST, { type: 'admin_end_game' });
    expect(t.game().phase).toBe('ended');
    expect(t.sent.filter((m) => m.t === 'critic').at(-1)).toMatchObject({ ready: false });
  });

  it('no key: the critic step ends in "no contest" at once, and the host picks the winner', async () => {
    const t = setup(spyCritics().source);
    const mg = intoDrawOff(t);
    await flush();
    expect(mg()).toMatchObject({ phase: 'results', winners: [], failed: 'no AI critic key' });
    t.clock.advance(60_000);
    expect(t.game().currentTurn?.duelGameOutcome?.status).toBe('no_contest');
    expect(t.game().lastDuel ?? null).toBeNull();
    expect(t.press(HOST, { type: 'duel_winner', id: t.pids[1] })).toEqual({ ok: true });
    expect(t.game().lastDuel).toMatchObject({ winnerId: t.pids[1] });
  });

  it("a refused key is reported without the key in the message", async () => {
    const t = setup(spyCritics({ badKey: true }).source);
    t.room.setCriticKey(KEY);
    await flush();
    const status = t.sent.filter((m) => m.t === 'critic').at(-1) as Extract<ServerMsg, { t: 'critic' }>;
    expect(status.ready).toBe(false);
    expect(status.error).toContain('***');
    expect(JSON.stringify(t.sent)).not.toContain(KEY);
  });

  it('only the host is told about the critic', async () => {
    const t = setup(spyCritics().source);
    t.room.setCriticKey(KEY);
    await flush();
    const critics = t.sent.filter((m) => m.t === 'critic').length;
    expect(critics).toBeGreaterThan(0);
    // every critic message went to the host connection only (the host is the only one counted here)
    const t2 = setup(spyCritics().source);
    const phoneMsgs: ServerMsg[] = [];
    t2.room.attach({ viewer: { kind: 'phone', pid: t2.pids[0] }, send: (m) => phoneMsgs.push(m) });
    t2.room.setCriticKey(KEY);
    await flush();
    expect(phoneMsgs.some((m) => m.t === 'critic')).toBe(false);
  });
});
