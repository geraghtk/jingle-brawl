// A player's phone died: the host asks the room for a fresh link to their seat. Only the host may,
// only for someone at the table, and the token it gets seats exactly that player.
import { afterEach, describe, expect, it } from 'vitest';
import WebSocket from 'ws';
import type { AddressInfo } from 'node:net';
import { PROTOCOL } from '../../../web/src/net/protocol';
import type { CreateRoomRes, JoinRes, ServerMsg } from '../../../web/src/net/protocol';
import { verify } from '../auth/tokens';
import { Rooms } from '../room/rooms';
import { MemoryStore } from '../store/store';
import { FakeClock } from '../testing/fakeClock';
import { createAppServer } from './http';

let close: (() => void) | null = null;
afterEach(() => { close?.(); close = null; });

async function screen(base: string, room: string, token?: string) {
  const ws = new WebSocket(`${base.replace('http', 'ws')}/ws`);
  await new Promise((r) => ws.once('open', r));
  const frames: ServerMsg[] = [];
  ws.on('message', (d) => frames.push(JSON.parse(String(d))));
  ws.send(JSON.stringify({ t: 'hello', v: PROTOCOL, room, token }));
  const next = async (t: ServerMsg['t']) => {
    for (let i = 0; i < 100; i++) { const f = frames.find((m) => m.t === t); if (f) return f; await new Promise((r) => setTimeout(r, 10)); }
    throw new Error(`no ${t}`);
  };
  await next('welcome');
  return { ws, frames, next };
}

describe('a fresh link to a seat', () => {
  it('the host gets one for a player; a phone and the TV get nothing; the token seats that player', async () => {
    const rooms = new Rooms(new MemoryStore(), new FakeClock(), { secret: 'test' });
    const app = createAppServer({ rooms, staticDir: null, limits: false });
    await new Promise<void>((r) => app.server.listen(0, '127.0.0.1', () => r()));
    const base = `http://127.0.0.1:${(app.server.address() as AddressInfo).port}`;
    close = () => { app.close(); rooms.close(); };
    const post = async <T>(path: string, body: unknown) => (await fetch(base + path, { method: 'POST', body: JSON.stringify(body) })).json() as Promise<T>;

    const { code, token } = await post<CreateRoomRes>('/api/rooms', {});
    const ada = await post<JoinRes>(`/api/rooms/${code}/join`, { name: 'Ada' });
    const bo = await post<JoinRes>(`/api/rooms/${code}/join`, { name: 'Bo' });

    const phone = await screen(base, code, bo.token);
    phone.ws.send(JSON.stringify({ t: 'seat', pid: ada.pid }));
    expect(await phone.next('error')).toMatchObject({ error: 'no such player' });
    expect(phone.frames.some((m) => m.t === 'seat')).toBe(false);
    const tv = await screen(base, code);
    tv.ws.send(JSON.stringify({ t: 'seat', pid: ada.pid }));
    await tv.next('error');

    const host = await screen(base, code, token);
    host.ws.send(JSON.stringify({ t: 'seat', pid: 'p_nobody' }));
    await host.next('error');
    host.ws.send(JSON.stringify({ t: 'seat', pid: ada.pid }));
    const seat = (await host.next('seat')) as Extract<ServerMsg, { t: 'seat' }>;
    expect(seat.pid).toBe(ada.pid);
    expect(verify('test', seat.token)).toMatchObject({ room: code, role: 'player', pid: ada.pid });

    const newPhone = await screen(base, code, seat.token);
    expect(newPhone.frames.find((m) => m.t === 'welcome')).toMatchObject({ viewer: 'phone', pid: ada.pid });
    for (const s of [phone, tv, host, newPhone]) s.ws.close();
  });
});
