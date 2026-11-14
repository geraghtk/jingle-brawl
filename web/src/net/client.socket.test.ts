// A screen's connection, one frame at a time: a scripted socket and a clock we move by hand, for
// the orderings a real server won't produce on demand (a lost message, a retry racing a restart,
// a press nobody will ever answer).
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Game } from '../core/types';
import { RoomClient, SILENT_MS } from './client';
import type { Ack } from './client';
import type { ClientMsg, ServerMsg } from './protocol';

class FakeSocket {
  static CONNECTING = 0; static OPEN = 1; static CLOSING = 2; static CLOSED = 3;
  static all: FakeSocket[] = [];
  readyState = FakeSocket.CONNECTING;
  sent: ClientMsg[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  constructor(readonly url: string) { FakeSocket.all.push(this); }
  send(data: string): void {
    if (this.readyState !== FakeSocket.OPEN) throw new Error('send on a socket that is not open');
    this.sent.push(JSON.parse(data));
  }
  /** The screen hangs up: closing now, closed a moment later (as in a browser). */
  close(): void {
    if (this.readyState >= FakeSocket.CLOSING) return;
    this.readyState = FakeSocket.CLOSING;
    setTimeout(() => this.drop(), 1);
  }
  // --- the server's side ---
  accept(): void { this.readyState = FakeSocket.OPEN; this.onopen?.(); }
  say(m: ServerMsg): void { if (this.readyState !== FakeSocket.CLOSED) this.onmessage?.({ data: JSON.stringify(m) }); }
  drop(): void { if (this.readyState === FakeSocket.CLOSED) return; this.readyState = FakeSocket.CLOSED; this.onclose?.(); }
  /** Accepted, welcomed and sent its whole view. */
  join(version: number, game: Game): this {
    this.accept();
    this.say({ t: 'welcome', viewer: 'host' });
    this.say({ t: 'state', version, game, now: Date.now() });
    return this;
  }
  intents(): ClientMsg[] { return this.sent.filter((m) => m.t === 'intent'); }
}

const sockets = () => FakeSocket.all;
const live = () => sockets().filter((s) => s.readyState <= FakeSocket.OPEN);
const last = () => sockets()[sockets().length - 1];
const game = (o: Record<string, unknown>) => ({ phase: 'lobby', players: { p1: { name: 'Ada', chips: 10 } }, ...o }) as unknown as Game;
const client = () => new RoomClient('ABCD', 'tok', 'ws://room/ws');
/** What a promise has come to by now ('hanging': nothing yet). */
const outcome = async (p: Promise<Ack>): Promise<Ack | 'hanging'> => {
  await vi.advanceTimersByTimeAsync(0);
  return Promise.race([p, Promise.resolve('hanging' as const)]);
};

beforeEach(() => {
  FakeSocket.all = [];
  vi.useFakeTimers();
  vi.stubGlobal('WebSocket', FakeSocket);
});
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });

describe('RoomClient, frame by frame', () => {
  it('a patch on the wrong base is never applied: the screen starts over with a whole view', () => {
    const c = client().start();
    const s0 = last().join(1, game({ note: 'v1' }));
    expect(c.version).toBe(1);
    s0.say({ t: 'patch', base: 3, version: 4, updates: { note: 'v4', phase: 'main' }, now: 0 });   // v2 and v3 never came
    expect(c.game).toEqual(game({ note: 'v1' }));
    expect(c.version).toBe(1);
    expect(s0.readyState).toBeGreaterThanOrEqual(FakeSocket.CLOSING);
    s0.say({ t: 'patch', base: 4, version: 5, updates: { note: 'v5' }, now: 0 });                    // still arriving as it closes
    expect(c.game).toEqual(game({ note: 'v1' }));
    vi.advanceTimersByTime(1000);
    expect(sockets().length).toBe(2);
    expect(live().length).toBe(1);
    last().join(5, game({ note: 'v5', phase: 'main' }));
    expect(last().sent[0]).toMatchObject({ t: 'hello', room: 'ABCD', token: 'tok' });
    expect(c.game).toEqual(game({ note: 'v5', phase: 'main' }));
    expect(c.version).toBe(5);
    last().say({ t: 'patch', base: 5, version: 6, updates: { note: 'v6' }, now: 0 });                // and on from there
    expect(c.game).toEqual(game({ note: 'v6', phase: 'main' }));
    expect(c.status).toBe('open');
  });

  it('a patch before any whole view is not applied either', () => {
    const c = client().start();
    last().accept();
    last().say({ t: 'welcome', viewer: 'host' });
    last().say({ t: 'patch', base: 0, version: 1, updates: { phase: 'main' }, now: 0 });
    expect(c.game).toBeNull();
    expect(last().readyState).toBeGreaterThanOrEqual(FakeSocket.CLOSING);
  });

  it('presses still waiting when the screen stops are answered "no", and never sent in a later session', async () => {
    const c = client().start();
    last().join(1, game({}));
    last().drop();
    const pressed = c.send({ type: 'start' });                 // tapped while offline
    expect(await outcome(pressed)).toBe('hanging');
    c.stop();
    expect(await outcome(pressed)).toEqual({ ok: false, error: 'left the room' });
    c.start();
    last().join(2, game({}));
    vi.advanceTimersByTime(10_000);
    expect(sockets().flatMap((s) => s.intents())).toEqual([]);
  });

  it('after a drop: presses made meanwhile go once on the new socket, with their ids, and the answers come back', async () => {
    const c = client().start();
    const s0 = last().join(1, game({}));
    const sentLive = c.send({ type: 'start' });                // sent, but the answer is lost with the socket
    const id0 = (s0.intents()[0] as { id: string }).id;
    s0.drop();
    expect(c.status).toBe('closed');
    const offline = c.send({ type: 'reveal' });
    vi.advanceTimersByTime(250);
    const s1 = last();
    expect(s1).not.toBe(s0);
    s1.accept();
    const early = c.send({ type: 'draw', giftName: 'Socks' }); // connected, not yet welcomed
    expect(s1.intents()).toEqual([]);                          // nothing before the room knows who this is
    s1.say({ t: 'welcome', viewer: 'host' });
    s1.say({ t: 'state', version: 7, game: game({ phase: 'main' }), now: 0 });
    const sent = s1.intents() as { id: string; intent: { type: string } }[];
    expect(sent.map((m) => m.intent.type)).toEqual(['start', 'reveal', 'draw']);
    expect(sent[0].id).toBe(id0);                              // the same press, so the room applies it once
    expect(new Set(sent.map((m) => m.id)).size).toBe(3);
    expect(c.game).toEqual(game({ phase: 'main' }));           // the room moved on meanwhile
    expect(c.version).toBe(7);
    s1.say({ t: 'ack', id: sent[0].id, ok: true });
    s1.say({ t: 'ack', id: sent[1].id, ok: false, error: 'not now' });
    s1.say({ t: 'ack', id: sent[2].id, ok: true });
    expect(await outcome(sentLive)).toEqual({ ok: true });
    expect(await outcome(offline)).toEqual({ ok: false, error: 'not now' });
    expect(await outcome(early)).toEqual({ ok: true });
    s1.drop();                                                 // answered presses are not sent again
    vi.advanceTimersByTime(250);
    last().join(8, game({ phase: 'main' }));
    expect(last().intents()).toEqual([]);
  });

  it('waits longer between tries, up to 5 s, and starts short again once it has connected', () => {
    client().start();
    const waits: number[] = [];
    for (let i = 0; i < 8; i++) {
      const n = sockets().length;
      last().drop();                                           // nobody home
      let ms = 0;
      while (sockets().length === n) { vi.advanceTimersByTime(50); ms += 50; if (ms > 60_000) throw new Error('gave up'); }
      waits.push(ms);
    }
    expect(waits).toEqual([250, 500, 1000, 2000, 4000, 5000, 5000, 5000]);
    last().join(1, game({}));
    const n = sockets().length;
    last().drop();
    vi.advanceTimersByTime(250);
    expect(sockets().length).toBe(n + 1);
    expect(live().length).toBe(1);
  });

  it('a socket that has gone quiet is replaced, by one new socket', () => {
    const c = client().start();
    const s0 = last().join(1, game({}));
    vi.advanceTimersByTime(SILENT_MS + 5000);                  // pings go out, nothing comes back
    expect(s0.sent.filter((m) => m.t === 'ping').length).toBeGreaterThan(1);
    expect(s0.readyState).toBeGreaterThanOrEqual(FakeSocket.CLOSING);
    vi.advanceTimersByTime(1000);
    expect(sockets().length).toBe(2);
    expect(live().length).toBe(1);
    last().join(2, game({ note: 'back' }));
    expect(c.status).toBe('open');
    expect(c.game).toEqual(game({ note: 'back' }));
    vi.advanceTimersByTime(4000);                              // and the new one is pinged on one timer, not two
    expect(last().sent.filter((m) => m.t === 'ping').length).toBe(1);
  });
});
