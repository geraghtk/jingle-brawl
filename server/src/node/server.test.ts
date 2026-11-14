// The whole server, over real HTTP and WebSockets: a host creates a room, phones join, and a party
// plays to the end with everyone pressing only what their own view offers. Time is a fake clock.
import { afterEach, describe, expect, it } from 'vitest';
import WebSocket from 'ws';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { request } from 'node:http';
import type { AddressInfo } from 'node:net';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { rng } from '../../../web/src/core/rng';
import { HOST, offers } from '../../../web/src/core/table';
import type { Offer, Who } from '../../../web/src/core/table';
import type { Game } from '../../../web/src/core/types';
import { PROTOCOL } from '../../../web/src/net/protocol';
import { viewsFrom } from '../../../web/src/net/views';
import type { CreateRoomRes, JoinRes, ServerMsg } from '../../../web/src/net/protocol';
import { Rooms } from '../room/rooms';
import { MemoryStore } from '../store/store';
import { FakeClock } from '../testing/fakeClock';
import { createAppServer, debugApiRefusal } from './http';
import type { ServerOpts } from './http';

const HOUR = 60 * 60 * 1000;
let close: (() => void) | null = null;
const dirs: string[] = [];
afterEach(() => { close?.(); close = null; for (const d of dirs.splice(0)) rmSync(d, { recursive: true, force: true }); });

async function boot(opts: Partial<ServerOpts> = {}) {
  const clock = new FakeClock();
  const store = new MemoryStore();
  const rooms = new Rooms(store, clock, { secret: 'test' });
  const app = createAppServer({ rooms, staticDir: null, debugApi: true, ...opts });
  await new Promise<void>((r) => app.server.listen(0, '127.0.0.1', () => r()));
  const port = (app.server.address() as AddressInfo).port;
  const base = `http://127.0.0.1:${port}`;
  close = () => { app.close(); rooms.close(); };
  return { clock, base, port, rooms, store, app };
}

/** One request exactly as written: the path is not tidied (fetch would resolve `..` itself) and any header can be set. */
function raw(port: number, path: string, { method = 'GET', headers = {}, body }: { method?: string; headers?: Record<string, string>; body?: string } = {}) {
  return new Promise<{ status: number; body: string }>((resolve, reject) => {
    const req = request({ host: '127.0.0.1', port, path, method, headers }, (res) => {
      const chunks: Buffer[] = [];
      res.on('data', (c) => chunks.push(c));
      res.on('end', () => resolve({ status: res.statusCode || 0, body: Buffer.concat(chunks).toString('utf8') }));
    });
    req.on('error', reject);
    req.end(body);
  });
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
    if (m.t === 'state' || m.t === 'patch') s.game = viewsFrom(s.frames).at(-1) ?? null;
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
    const sealedLeak = viewsFrom(tv.frames).some((g) => g.currentTurn?.phase === 'bidding'
      && Object.values(g.currentTurn.bids || {}).some((b) => b.amount !== 0));
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

describe('the gates', () => {
  it('without --debug-api nothing under /debug answers: no read, no write, no reload', async () => {
    const { base, rooms } = await boot({ debugApi: false });
    const { code } = await post<CreateRoomRes>(`${base}/api/rooms`, {});
    const version = rooms.get(code)!.state.version;
    const tries: [string, string, string?][] = [
      ['GET', `/debug/rooms/${code}`], ['GET', `/debug/rooms/${code}/phase`], ['GET', '/debug'], ['GET', '/debug/rooms/ZZZZ'],
      ['PUT', `/debug/rooms/${code}/phase`, '"ended"'], ['PATCH', `/debug/rooms/${code}`, '{"phase":"ended"}'],
      ['POST', `/debug/rooms/${code}/reload`], ['DELETE', `/debug/rooms/${code}`],
    ];
    for (const [method, path, body] of tries) {
      const res = await fetch(`${base}${path}`, { method, body });
      expect([method, path, res.status]).toEqual([method, path, 404]);
      expect(await res.json()).toEqual({ error: 'not found' });
    }
    expect(rooms.get(code)!.state.game.phase).toBe('lobby');
    expect(rooms.get(code)!.state.version).toBe(version);
  });

  it("a code that comes back is a new room: the old room's host and player tokens are refused on the socket", async () => {
    const { base, clock, rooms, store } = await boot();
    const { code, token } = await post<CreateRoomRes>(`${base}/api/rooms`, {});
    const ada = await post<JoinRes>(`${base}/api/rooms/${code}/join`, { name: 'Ada' });
    const before = [await connect(base, code, token), await connect(base, code, ada.token)];
    expect(before.every((s) => s.game)).toBe(true);                          // while the room lives, both open it
    for (const s of before) s.ws.close();
    clock.advance(25 * HOUR);
    rooms.sweep(HOUR);
    expect(rooms.purge()).toEqual([code]);
    // the code is dealt again (as Rooms.create would), and by chance the new room has a player with Ada's id
    store.create(code, { phase: 'lobby', createdAt: clock.now(), settings: {}, players: { [ada.pid]: { name: 'Zed', chips: 10 } } }, clock.now());
    for (const old of [token, ada.token]) {
      const s = await connect(base, code, old);
      expect(s.game).toBeNull();
      expect(s.frames).toEqual([{ t: 'error', error: 'not your room', fatal: true }]);
    }
    expect(rooms.get(code)!.screens).toBe(0);
    const fresh = rooms.token({ room: code, born: rooms.get(code)!.born, role: 'host' });
    expect((await connect(base, code, fresh)).game!.players![ada.pid].name).toBe('Zed');
  });

  it('static files: nothing outside the static folder is ever served, however the path is dressed up', async () => {
    const top = mkdtempSync(join(tmpdir(), 'jb-static-')); dirs.push(top);
    const dist = join(top, 'dist');
    mkdirSync(join(dist, 'assets'), { recursive: true });
    mkdirSync(join(top, 'dist-evil'));
    writeFileSync(join(dist, 'player.html'), '<p>the app</p>');
    writeFileSync(join(dist, 'assets', 'app.js'), '// app');
    writeFileSync(join(top, 'package.json'), 'SECRET');
    writeFileSync(join(top, 'dist-evil', 'x.txt'), 'SECRET');           // a sibling whose name starts like the folder's
    const { port } = await boot({ staticDir: dist, debugApi: false });
    expect((await raw(port, '/')).body).toBe('<p>the app</p>');
    expect((await raw(port, '/assets/app.js')).body).toBe('// app');
    const abs = join(top, 'package.json');
    const outside = [
      '/..%2f..%2fpackage.json', '/..%2fpackage.json', '/%2e%2e%2fpackage.json', '/assets/../../package.json', '/assets/..%2f..%2fpackage.json',
      '/assets/%2e%2e/%2e%2e/package.json', '/..%5cpackage.json', '/assets/..%5c..%5cpackage.json', '/assets\\..\\..\\package.json',
      '/%2e%2e/package.json', '/../package.json', '/../dist-evil/x.txt', '/..%2fdist-evil/x.txt', '/..%2fdist-evil%2fx.txt',
      '/%252e%252e%252fpackage.json', '/..%252fpackage.json', '/....//package.json',
      '/player.html%00', '/..%2fpackage.json%00', '/%00', '/assets/app.js%00.png', '/..%00/package.json',
      `/${encodeURIComponent(abs)}`, `/${abs.replace(/\\/g, '/')}`, `/${encodeURIComponent(`../${abs}`)}`,
      '/assets', '/assets/',
    ];
    for (const path of outside) {
      for (const method of ['GET', 'HEAD']) {
        const res = await raw(port, path, { method });
        expect([path, res.status]).toEqual([path, 404]);
        expect(res.body).not.toContain('SECRET');
      }
    }
    // two slashes first is still a path on this server, not another host's name; a bad escape is no file
    for (const path of ['//..%2fpackage.json', '//[', '//%zz/x', '///..%2f..%2fpackage.json', '/%zz', '/assets/%E0%A4%A']) {
      const res = await raw(port, path);
      expect([path, res.status]).toEqual([path, 404]);
    }
    expect((await raw(port, '//assets/app.js')).body).toBe('// app');
  });

  it('a socket upgrade on a path that is not a URL is dropped; the server stays up', async () => {
    const { port, base } = await boot();
    const upgrade = { Connection: 'Upgrade', Upgrade: 'websocket', 'Sec-WebSocket-Key': 'dGhlIHNhbXBsZSBub25jZQ==', 'Sec-WebSocket-Version': '13' };
    const crashes: unknown[] = [];
    const seen = (e: unknown) => crashes.push(e);
    process.on('uncaughtException', seen);
    try {
      for (const path of ['//[', '//..%2fws', '//%zz']) {
        expect(await raw(port, path, { headers: upgrade }).then(() => 'answered', () => 'dropped')).toBe('dropped');
      }
      await new Promise((r) => setTimeout(r, 20));
    } finally { process.off('uncaughtException', seen); }
    expect(crashes).toEqual([]);                                             // in production an uncaught error stops the process
    expect((await fetch(`${base}/healthz`)).status).toBe(200);
    const { code } = await post<CreateRoomRes>(`${base}/api/rooms`, {});
    expect((await connect(base, code)).game).toBeTruthy();                   // and real sockets still open
  });
});

describe('the debug API (tests and dev, this machine only)', () => {
  it('answers only a request that asked for this machine by its own name (a rebound name is refused)', async () => {
    const { port, base, rooms } = await boot();
    const { code } = await post<CreateRoomRes>(`${base}/api/rooms`, {});
    const path = `/debug/rooms/${code}/phase`;
    // what the suite and the dev tools use: http://localhost:4173, http://127.0.0.1:8787
    for (const host of ['localhost:4173', '127.0.0.1:8787', `127.0.0.1:${port}`, 'localhost', '127.0.0.1', '[::1]:8787', '[::1]', 'LOCALHOST:5']) {
      const res = await raw(port, path, { headers: { Host: host } });
      expect([host, res.status, res.body]).toEqual([host, 200, '"lobby"']);
    }
    for (const host of ['evil.example', 'evil.example:4173', 'localhost.evil.example', '127.0.0.1.evil.example:8787', 'evil.example:127.0.0.1',
      'notlocalhost', '127.0.0.10', '192.168.1.20:8787', 'localhost:', 'localhost:80x', 'localhost:8787@evil.example', 'localhost evil.example']) {
      for (const [method, p, body] of [['GET', path], ['PUT', path, '"ended"'], ['PATCH', `/debug/rooms/${code}`, '{"phase":"ended"}'], ['POST', `/debug/rooms/${code}/reload`]] as [string, string, string?][]) {
        const res = await raw(port, p, { method, body, headers: { Host: host } });
        expect([host, method, res.status]).not.toEqual([host, method, 200]);
        if (res.status !== 400) expect([host, method, res.status]).toEqual([host, method, 404]);   // (400: Node itself refuses a Host that is no host)
      }
    }
    expect(rooms.get(code)!.state.game.phase).toBe('lobby');
    // the app itself is not tied to a name: phones reach it by the PC's Wi-Fi address
    expect((await raw(port, `/api/rooms/${code}/lobby`, { headers: { Host: '192.168.1.20:8787' } })).status).toBe(200);
  });

  it('behind a proxy (trustProxy) it does not answer at all, like /metrics: every caller looks local there', async () => {
    const { base, rooms } = await boot({ debugApi: true, trustProxy: true, limits: false });
    const { code } = await post<CreateRoomRes>(`${base}/api/rooms`, {});
    expect((await fetch(`${base}/debug/rooms/${code}`)).status).toBe(404);
    expect((await fetch(`${base}/debug/rooms/${code}/phase`, { method: 'PUT', body: '"ended"' })).status).toBe(404);
    expect((await fetch(`${base}/debug/rooms/${code}/reload`, { method: 'POST' })).status).toBe(404);
    expect(rooms.get(code)!.state.game.phase).toBe('lobby');
  });

  it('the rule for starting: never with a proxy in front, never in production', () => {
    expect(debugApiRefusal({ debugApi: true, trustProxy: false, nodeEnv: undefined })).toBeNull();
    expect(debugApiRefusal({ debugApi: true, nodeEnv: 'test' })).toBeNull();
    expect(debugApiRefusal({ debugApi: true, nodeEnv: 'development' })).toBeNull();
    expect(debugApiRefusal({ debugApi: false, trustProxy: true, nodeEnv: 'production' })).toBeNull();
    expect(debugApiRefusal({ debugApi: true, trustProxy: true })).toMatch(/--trust-proxy/);
    expect(debugApiRefusal({ debugApi: true, nodeEnv: 'production' })).toMatch(/NODE_ENV=production/);
  });

  it('the server refuses to start with --debug-api behind a proxy or in production: a clear error, exit code 1, no port opened', () => {
    const start = (flags: string[], env: Record<string, string>) => {
      const clean = Object.fromEntries(Object.entries(process.env).filter(([k]) => !/^(NODE_ENV|PORT|JB_|VITEST)/.test(k)));
      return spawnSync(process.execPath, ['node_modules/vite-node/vite-node.mjs', 'server/src/node/main.ts', '--', '--port', '0', '--host', '127.0.0.1', '--db', ':memory:', '--static', 'none', ...flags],
        { env: { ...clean, ...env }, encoding: 'utf8', timeout: 20_000 });
    };
    const cases: [string[], Record<string, string>, RegExp][] = [
      [['--debug-api', '--trust-proxy'], {}, /Not starting: --debug-api cannot be used with --trust-proxy/],
      [['--debug-api'], { JB_TRUST_PROXY: '1' }, /Not starting: --debug-api cannot be used with --trust-proxy/],
      [['--debug-api'], { NODE_ENV: 'production' }, /Not starting: --debug-api cannot be used with NODE_ENV=production/],
    ];
    for (const [flags, env, why] of cases) {
      const r = start(flags, env);                       // (a server that did start would run into the timeout: status null)
      expect([flags, env, r.status]).toEqual([flags, env, 1]);
      expect(r.stderr).toMatch(why);
      expect(r.stdout).not.toMatch(/room server on/);
    }
  }, 90_000);
});
