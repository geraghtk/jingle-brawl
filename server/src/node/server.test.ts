// The whole server, over real HTTP and WebSockets: a host creates a room, phones join, and a party
// plays to the end with everyone pressing only what their own view offers. Time is a fake clock.
import { afterEach, describe, expect, it } from 'vitest';
import WebSocket from 'ws';
import type { AddressInfo } from 'node:net';
import { rng } from '../../../web/src/core/rng';
import { HOST, offers } from '../../../web/src/core/table';
import type { Offer, Who } from '../../../web/src/core/table';
import type { Game } from '../../../web/src/core/types';
import { PROTOCOL } from '../../../web/src/net/protocol';
import type { CreateRoomRes, JoinRes, ServerMsg } from '../../../web/src/net/protocol';
import { Rooms } from '../room/rooms';
import { MemoryStore } from '../store/store';
import { FakeClock } from '../testing/fakeClock';
import { createAppServer } from './http';

let close: (() => void) | null = null;
afterEach(() => { close?.(); close = null; });

async function boot() {
  const clock = new FakeClock();
  const rooms = new Rooms(new MemoryStore(), clock, { secret: 'test' });
  const app = createAppServer({ rooms, staticDir: null, debugApi: true });
  await new Promise<void>((r) => app.server.listen(0, '127.0.0.1', () => r()));
  const base = `http://127.0.0.1:${(app.server.address() as AddressInfo).port}`;
  close = () => { app.close(); rooms.close(); };
  return { clock, base, rooms };
}

const post = async <T>(url: string, body: unknown): Promise<T> => (await fetch(url, { method: 'POST', body: JSON.stringify(body) })).json() as Promise<T>;

/** One screen: its socket, its latest view, and send-and-wait-for-ack. */
async function connect(base: string, room: string, token?: string) {
  const ws = new WebSocket(`${base.replace('http', 'ws')}/ws`);
  await new Promise((r, j) => { ws.once('open', r); ws.once('error', j); });
  const s = { ws, game: null as Game | null, frames: [] as ServerMsg[], acks: new Map<string, (m: ServerMsg) => void>() };
  ws.on('message', (d) => {
    const m = JSON.parse(String(d)) as ServerMsg;
    s.frames.push(m);
    if (m.t === 'state') s.game = m.game;
    if (m.t === 'ack') s.acks.get(m.id)?.(m);
  });
  ws.send(JSON.stringify({ t: 'hello', v: PROTOCOL, room, token }));
  for (let i = 0; i < 50 && !s.game; i++) await new Promise((r) => setTimeout(r, 10));
  let n = 0;
  const press = (intent: Offer) => new Promise<Extract<ServerMsg, { t: 'ack' }>>((r) => {
    const id = `i${++n}`;
    s.acks.set(id, (m) => r(m as Extract<ServerMsg, { t: 'ack' }>));
    ws.send(JSON.stringify({ t: 'intent', id, intent }));
  });
  return { ...s, get game() { return s.game; }, frames: s.frames, press };
}

describe('the room server', () => {
  it('a party over WebSockets: create, join, play to the end; each screen pressing only what its own view offers', async () => {
    const { clock, base } = await boot();
    const { code, token } = await post<CreateRoomRes>(`${base}/api/rooms`, { settings: { duelPick: 'off', voteWindow: 0, tiebreakerMethod: 'whack_elf' } });
    expect(code).toMatch(/^[A-Z2-9]{4}$/);
    const host = await connect(base, code, token);
    const tv = await connect(base, code);
    const phones: Record<string, Awaited<ReturnType<typeof connect>>> = {};
    for (const name of ['Ada', 'Bo', 'Cy', 'Di']) {
      const j = await post<JoinRes>(`${base}/api/rooms/${code}/join`, { name });
      phones[j.pid] = await connect(base, code, j.token);
    }
    const r = rng(7);
    const screens: [Who, typeof host][] = [[HOST, host], ...Object.entries(phones)];
    for (let round = 0; round < 400; round++) {
      const g = host.game!;
      if (g.phase === 'ended') break;
      for (const [who, s] of screens) {
        const mine = offers(s.game!, who, clock.now()).filter((o) => o.type !== 'end');
        if (!mine.length || r() < 0.4) continue;
        const o = mine[Math.floor(r() * mine.length)];
        const ack = await s.press(o.type === 'whack' ? { type: 'whack', score: Math.floor(r() * 20), final: true } : o);
        if (!ack.ok) expect(ack.error).toMatch(/isn't on|got there first|closed|Waiting/);  // a race with someone else's press
      }
      clock.advance(3000);
      await new Promise((res) => setTimeout(res, 5));
      if (!offers(host.game!, HOST, clock.now()).some((o) => o.type !== 'end') && host.game!.phase === 'main' && !host.game!.currentTurn
        && Object.values(host.game!.players || {}).every((p) => p.giftId)) await host.press({ type: 'end' });
    }
    expect(host.game!.phase).toBe('ended');
    expect(Object.values(host.game!.players!).every((p) => p.giftId)).toBe(true);
    // the TV only watched, and never saw a sealed bid
    const sealedLeak = tv.frames.some((m) => m.t === 'state' && m.game.currentTurn?.phase === 'bidding'
      && Object.values(m.game.currentTurn.bids || {}).some((b) => b.amount !== 0));
    expect(sealedLeak).toBe(false);
    expect((await tv.press({ type: 'reveal' })).ok).toBe(false);
  }, 60_000);

  it("refuses a token for another room, a forged token, and an unknown room", async () => {
    const { base } = await boot();
    const a = await post<CreateRoomRes>(`${base}/api/rooms`, {});
    const b = await post<CreateRoomRes>(`${base}/api/rooms`, {});
    const wrong = await connect(base, b.code, a.token);
    expect(wrong.frames.some((m) => m.t === 'error' && m.fatal)).toBe(true);
    const forged = await connect(base, a.code, `${Buffer.from(JSON.stringify({ room: a.code, role: 'host' })).toString('base64url')}.AAAA`);
    expect(forged.frames.some((m) => m.t === 'error' && m.fatal)).toBe(true);
    const lobby = await (await fetch(`${base}/api/rooms/ZZZZ/lobby`)).json();
    expect(lobby).toEqual({ exists: false });
  });
});
