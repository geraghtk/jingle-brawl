// What the server does with traffic that is wrong: junk on a socket, a press before hello, an old
// page, a flood; a join for a room that is not there, without a name, or far too big. Each is
// answered (or dropped) and the room carries on.
import { afterEach, describe, expect, it } from 'vitest';
import WebSocket from 'ws';
import { request } from 'node:http';
import type { AddressInfo } from 'node:net';
import { PROTOCOL } from '../../../web/src/net/protocol';
import type { CreateRoomRes, JoinRes, ServerMsg } from '../../../web/src/net/protocol';
import { Rooms } from '../room/rooms';
import { MemoryStore } from '../store/store';
import { FakeClock } from '../testing/fakeClock';
import { createAppServer } from './http';
import type { ServerOpts } from './http';
import { MAX_FRAME } from './ws';

let close: (() => void) | null = null;
afterEach(() => { close?.(); close = null; });

async function boot(opts: Partial<ServerOpts> = {}) {
  const rooms = new Rooms(new MemoryStore(), new FakeClock(), { secret: 'test' });
  const app = createAppServer({ rooms, staticDir: null, limits: false, ...opts });
  await new Promise<void>((r) => app.server.listen(0, '127.0.0.1', () => r()));
  const base = `http://127.0.0.1:${(app.server.address() as AddressInfo).port}`;
  close = () => { app.close(); rooms.close(); };
  const made = (await (await fetch(`${base}/api/rooms`, { method: 'POST', body: '{}' })).json()) as CreateRoomRes;
  return { base, rooms, app, ...made };
}

/** A socket and everything it was sent; `until` waits for a frame (or the close). */
async function socket(base: string) {
  const ws = new WebSocket(`${base.replace('http', 'ws')}/ws`);
  const frames: ServerMsg[] = [];
  let closed: number | null = null;
  ws.on('message', (d) => frames.push(JSON.parse(String(d)) as ServerMsg));
  ws.on('close', (code) => { closed = code; });
  ws.on('error', () => {});
  await new Promise((r, j) => { ws.once('open', r); ws.once('error', j); });
  const until = async (ok: () => unknown) => { for (let i = 0; i < 200 && !ok(); i++) await new Promise((r) => setTimeout(r, 5)); return !!ok(); };
  return {
    ws, frames, until,
    say: (m: unknown) => ws.send(typeof m === 'string' ? m : JSON.stringify(m)),
    hello: (room: string, token?: string, v: unknown = PROTOCOL) => ws.send(JSON.stringify({ t: 'hello', v, room, token })),
    got: (t: ServerMsg['t']) => frames.filter((f) => f.t === t),
    closed: () => closed,
  };
}

describe('wrong traffic on a socket', () => {
  it('junk is answered, not obeyed: the socket stays open and still works', async () => {
    const { base, code, token, rooms } = await boot();
    const s = await socket(base);
    s.say('not json {');
    expect(await s.until(() => s.frames.length === 1)).toBe(true);
    expect(s.frames[0]).toEqual({ t: 'error', error: 'not JSON' });
    for (const odd of ['5', 'null', '"hello"', '[]', '{}', '{"t":"nonsense"}', '{"t":null}']) s.say(odd);   // JSON, but no message of ours: ignored
    s.say({ t: 'ping', c: 3 });
    expect(await s.until(() => s.got('pong').length)).toBe(true);
    expect(s.frames.length).toBe(2);
    expect(s.closed()).toBeNull();
    s.hello(code, token);
    expect(await s.until(() => s.got('state').length)).toBe(true);
    expect(rooms.get(code)!.screens).toBe(1);
  });

  it('a press, a key or a seat before hello does nothing', async () => {
    const { base, code, rooms } = await boot();
    const version = rooms.get(code)!.state.version;
    const s = await socket(base);
    s.say({ t: 'intent', id: 'a1', intent: { type: 'start' } });
    s.say({ t: 'critic_key', key: 'sk-nope' });
    s.say({ t: 'seat', pid: 'p_000000' });
    expect(await s.until(() => s.frames.length === 2)).toBe(true);
    expect(s.frames).toEqual([{ t: 'error', error: 'say hello first' }, { t: 'error', error: 'no such player' }]);
    expect(rooms.get(code)!.state.version).toBe(version);
    expect(rooms.get(code)!.screens).toBe(0);
  });

  it('an old page (another protocol version) is told to reload, for good; so is a hello for no room', async () => {
    const { base, code, token, rooms } = await boot();
    for (const [v, room, why] of [[PROTOCOL + 1, code, 'please reload: the game was updated'], [undefined, code, 'please reload: the game was updated'],
      [String(PROTOCOL), code, 'please reload: the game was updated'], [PROTOCOL, 'ZZZZ', 'no such room'], [PROTOCOL, '../x', 'no such room'], [PROTOCOL, { a: 1 }, 'no such room']] as [unknown, string, string][]) {
      const s = await socket(base);
      s.say({ t: 'hello', v, room, token });
      expect(await s.until(() => s.closed() != null)).toBe(true);
      expect([v, room, s.frames]).toEqual([v, room, [{ t: 'error', error: why, fatal: true }]]);
      expect(s.closed()).toBe(4000);
    }
    expect(rooms.get(code)!.screens).toBe(0);
  });

  it('a second hello does not change who a socket is; the TV and a phone cannot press the host’s buttons', async () => {
    const { base, code, token, rooms } = await boot();
    const ada = (await (await fetch(`${base}/api/rooms/${code}/join`, { method: 'POST', body: JSON.stringify({ name: 'Ada' }) })).json()) as JoinRes;
    await fetch(`${base}/api/rooms/${code}/join`, { method: 'POST', body: JSON.stringify({ name: 'Bo' }) });
    const tv = await socket(base);
    tv.hello(code);
    expect(await tv.until(() => tv.got('state').length)).toBe(true);
    tv.hello(code, token);                                                   // now with the host's token: too late, it is the TV
    tv.say({ t: 'intent', id: 't1', intent: { type: 'start' } });
    tv.say({ t: 'seat', pid: ada.pid });
    tv.say({ t: 'critic_key', key: 'sk-nope' });
    expect(await tv.until(() => tv.got('ack').length && tv.got('error').length)).toBe(true);
    expect(tv.got('ack')).toEqual([{ t: 'ack', id: 't1', ok: false, error: 'the TV only watches' }]);
    expect(tv.got('error')).toEqual([{ t: 'error', error: 'no such player' }]);
    expect(tv.got('seat')).toEqual([]);
    expect(tv.got('welcome')).toEqual([{ t: 'welcome', viewer: 'tv' }]);
    const phone = await socket(base);
    phone.hello(code, ada.token);
    expect(await phone.until(() => phone.got('state').length)).toBe(true);
    phone.say({ t: 'intent', id: 'p1', intent: { type: 'start' } });
    phone.say({ t: 'seat', pid: ada.pid });
    expect(await phone.until(() => phone.got('ack').length && phone.got('error').length)).toBe(true);
    expect(phone.got('ack')[0]).toMatchObject({ id: 'p1', ok: false });
    expect(phone.got('seat')).toEqual([]);
    expect(rooms.get(code)!.state.game.phase).toBe('lobby');
    expect(rooms.get(code)!.screens).toBe(2);
  });

  it('a press with a broken id or a broken intent is refused with an ack, and the room is untouched', async () => {
    const { base, code, token, rooms } = await boot();
    const version = rooms.get(code)!.state.version;
    const s = await socket(base);
    s.hello(code, token);
    expect(await s.until(() => s.got('state').length)).toBe(true);
    s.say({ t: 'intent', id: 'x'.repeat(81), intent: { type: 'start' } });
    s.say({ t: 'intent', intent: { type: 'start' } });
    s.say({ t: 'intent', id: { a: 1 }, intent: { type: 'start' } });
    s.say({ t: 'intent', id: 'j1', intent: 'start' });
    s.say({ t: 'intent', id: 'j2', intent: { type: '__proto__' } });
    s.say({ t: 'intent', id: 'j3' });
    expect(await s.until(() => s.got('ack').length === 6)).toBe(true);
    expect(s.got('ack').every((a) => a.t === 'ack' && a.ok === false && !!a.error)).toBe(true);
    expect(rooms.get(code)!.state.version).toBe(version);
    expect(s.closed()).toBeNull();
  });

  it('a flood is told to slow down and loses its extra messages; the socket recovers and nobody else is slowed', async () => {
    const { base, code, token } = await boot();
    const s = await socket(base);
    s.hello(code, token);
    expect(await s.until(() => s.got('state').length)).toBe(true);
    for (let i = 0; i < 200; i++) s.say({ t: 'ping', c: i });
    expect(await s.until(() => s.got('pong').length + s.got('error').length === 200)).toBe(true);
    const slow = s.got('error');
    expect(slow.length).toBeGreaterThan(100);                                // the burst is 40; a few more trickle in while it runs
    expect(s.got('pong').length).toBeGreaterThanOrEqual(39);
    expect(slow.every((e) => e.t === 'error' && e.error === 'slow down' && !e.fatal)).toBe(true);
    expect(s.closed()).toBeNull();
    const other = await socket(base);                                        // per socket: a second screen is not slowed
    other.say({ t: 'ping', c: 1 });
    expect(await other.until(() => other.got('pong').length === 1)).toBe(true);
    await new Promise((r) => setTimeout(r, 300));                            // 20 a second come back
    const before = s.got('pong').length;
    s.say({ t: 'ping', c: 999 });
    expect(await s.until(() => s.got('pong').length === before + 1)).toBe(true);
  });

  it('a frame bigger than a drawing closes that socket (1009) and nothing else', async () => {
    const { base, code, token, rooms } = await boot();
    const s = await socket(base);
    s.hello(code, token);
    expect(await s.until(() => s.got('state').length)).toBe(true);
    s.say({ t: 'intent', id: 'big', intent: { type: 'sketch', drawing: 'A'.repeat(MAX_FRAME + 10) } });
    expect(await s.until(() => s.closed() != null)).toBe(true);
    expect(s.closed()).toBe(1009);
    expect(await s.until(() => rooms.get(code)!.screens === 0)).toBe(true);
    const again = await socket(base);
    again.hello(code, token);
    expect(await again.until(() => again.got('state').length)).toBe(true);
  });
});

describe('wrong requests to the API', () => {
  const join = (base: string, code: string, body: string) => fetch(`${base}/api/rooms/${code}/join`, { method: 'POST', body });

  it('a join for a room that is not there is a 404 with words for the player; a code that is no code too', async () => {
    const { base } = await boot();
    for (const code of ['ZZZZ', 'zz', 'A-B-', '%2e%2ex', 'ABCDEFGHIJ']) {
      const res = await join(base, code, JSON.stringify({ name: 'Ada' }));
      expect([code, res.status]).toEqual([code, 404]);
      expect(await res.json()).toEqual({ error: 'Game not found! Check the room code.' });
    }
  });

  it('a join without a usable name is a 400 and seats nobody', async () => {
    const { base, code, rooms } = await boot();
    for (const body of ['{}', '', '{"name":""}', '{"name":"   "}', '{"name":null}', '{"name":"<>"}', 'null', '[]', '"Ada"', '7']) {
      const res = await join(base, code, body);
      expect([body, res.status]).toEqual([body, 400]);
      expect(await res.json()).toEqual({ error: 'Please enter your name' });
    }
    const res = await join(base, code, '{"name": "Ada"');                    // not JSON
    expect(res.status).toBe(400);
    expect(Object.keys(rooms.get(code)!.state.game.players || {})).toEqual([]);
    expect((await join(base, code, JSON.stringify({ name: '  Ada  ', age: 'old', avatar: { x: 1 } }))).status).toBe(200);
    expect(Object.values(rooms.get(code)!.state.game.players!).map((p) => p.name)).toEqual(['Ada']);
  });

  it('a body over the limit is refused (400) before it is read whole; the server and the room carry on', async () => {
    const { base, code, rooms } = await boot();
    const big = JSON.stringify({ name: 'Ada', pad: 'x'.repeat(200_000) });
    const tries = [() => join(base, code, big), () => fetch(`${base}/api/rooms`, { method: 'POST', body: big }),
      () => fetch(`${base}/api/client-error`, { method: 'POST', body: big })];
    for (const go of tries) {
      // (the server answers and stops reading; a client still sending may see the socket reset instead of the 400)
      const status = await go().then((r) => r.status, () => 'reset');
      expect(['reset', 400]).toContain(status);
    }
    // a body that does not say how long it is (chunked) is cut off at the limit too
    const chunked = await new Promise<number | 'reset'>((done) => {
      const req = request(`${base}/api/rooms/${code}/join`, { method: 'POST' }, (res) => { res.resume(); done(res.statusCode || 0); });
      req.on('error', () => done('reset'));
      req.write('{"name":"Ada","pad":"');
      let sent = 0;
      const more = () => { if (sent++ < 200 && !req.destroyed) { req.write('x'.repeat(1000)); setTimeout(more, 1); } };
      more();
    });
    expect(['reset', 400]).toContain(chunked);
    expect(Object.keys(rooms.get(code)!.state.game.players || {})).toEqual([]);
    expect(rooms.size).toBe(1);
    expect((await join(base, code, JSON.stringify({ name: 'Ada' }))).status).toBe(200);
    expect((await fetch(`${base}/healthz`)).status).toBe(200);
  });

  it('a full table refuses the next join with a 400 that says so', async () => {
    const { base, code, rooms } = await boot();
    for (let i = 0; i < 50; i++) expect(rooms.get(code)!.join({ name: `P${i}` })).toHaveProperty('pid');
    const res = await join(base, code, JSON.stringify({ name: 'One more' }));
    expect(res.status).toBe(400);
    expect(((await res.json()) as { error: string }).error).toMatch(/full/);
  });
});
