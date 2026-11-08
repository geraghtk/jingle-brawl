// The per-address limits, and the server enforcing them over real HTTP and WebSockets.
import { afterEach, describe, expect, it } from 'vitest';
import WebSocket from 'ws';
import type { AddressInfo } from 'node:net';
import type { IncomingMessage } from 'node:http';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { Rooms } from '../room/rooms';
import { MemoryStore } from '../store/store';
import { FakeClock } from '../testing/fakeClock';
import { createAppServer } from './http';
import type { ServerOpts } from './http';
import { Limiter, clientIp } from './limits';

describe('Limiter', () => {
  it('allows a burst, then refills at the per-minute rate', () => {
    let t = 0;
    const l = new Limiter({ join: { burst: 3, perMinute: 6 } }, () => t);
    expect([l.take('join', 'a'), l.take('join', 'a'), l.take('join', 'a')]).toEqual([0, 0, 0]);
    expect(l.take('join', 'a')).toBe(10_000);       // one token per 10 s
    expect(l.take('join', 'b')).toBe(0);            // another address has its own bucket
    t += 10_000;
    expect(l.take('join', 'a')).toBe(0);
    expect(l.take('join', 'a')).toBeGreaterThan(0);
  });

  it('caps open sockets per address and frees a slot on close', () => {
    const l = new Limiter({ socketsPerIp: 2 });
    expect([l.openSocket('a'), l.openSocket('a'), l.openSocket('a')]).toEqual([true, true, false]);
    l.closeSocket('a');
    expect(l.openSocket('a')).toBe(true);
  });

  it('reads X-Forwarded-For only when told to trust the proxy, and then its last hop', () => {
    const req = { headers: { 'x-forwarded-for': '6.6.6.6, 203.0.113.9' }, socket: { remoteAddress: '::ffff:10.0.0.2' } } as unknown as IncomingMessage;
    expect(clientIp(req, false)).toBe('10.0.0.2');
    expect(clientIp(req, true)).toBe('203.0.113.9');
  });
});

let close: (() => void) | null = null;
afterEach(() => { close?.(); close = null; });

async function boot(opts: Partial<ServerOpts> = {}, maxLive?: number) {
  const clock = new FakeClock();
  const rooms = new Rooms(new MemoryStore(), clock, { secret: 'test', maxLive });
  const app = createAppServer({ rooms, staticDir: null, ...opts });
  await new Promise<void>((r) => app.server.listen(0, '127.0.0.1', () => r()));
  const base = `http://127.0.0.1:${(app.server.address() as AddressInfo).port}`;
  close = () => { app.close(); rooms.close(); };
  return { clock, base };
}
const create = (base: string, headers: Record<string, string> = {}) => fetch(`${base}/api/rooms`, { method: 'POST', body: '{}', headers });
const opens = (base: string) => new Promise<boolean>((r) => {
  const ws = new WebSocket(`${base.replace('http', 'ws')}/ws`);
  ws.once('open', () => r(true));
  ws.once('error', () => r(false));
});

describe('the room server under abuse', () => {
  it('answers 429 with Retry-After once an address makes rooms too fast, and lets it go again later', async () => {
    const { clock, base } = await boot({ limits: { create: { burst: 2, perMinute: 1 } } });
    expect((await create(base)).status).toBe(200);
    expect((await create(base)).status).toBe(200);
    const r = await create(base);
    expect(r.status).toBe(429);
    expect(r.headers.get('retry-after')).toBe('60');
    expect(((await r.json()) as { error: string }).error).toMatch(/wait a moment/);
    clock.advance(60_000);
    expect((await create(base)).status).toBe(200);
  });

  it('limits room-code lookups and joins per address', async () => {
    const { base } = await boot({ limits: { lobby: { burst: 3, perMinute: 1 }, join: { burst: 1, perMinute: 1 } } });
    const { code } = (await (await create(base)).json()) as { code: string };
    const codes = ['AAAA', 'BBBB', 'CCCC', 'DDDD'];
    const statuses = await Promise.all(codes.map(async (c) => (await fetch(`${base}/api/rooms/${c}/lobby`)).status));
    expect(statuses.sort()).toEqual([200, 200, 200, 429]);
    const join = () => fetch(`${base}/api/rooms/${code}/join`, { method: 'POST', body: JSON.stringify({ name: 'Ada' }) });
    expect((await join()).status).toBe(200);
    expect((await join()).status).toBe(429);
  });

  it('behind a trusted proxy each client address has its own limit; untrusted, the header is ignored', async () => {
    const limits = { create: { burst: 1, perMinute: 1 } };
    let { base } = await boot({ limits, trustProxy: true });
    expect((await create(base, { 'X-Forwarded-For': '203.0.113.1' })).status).toBe(200);
    expect((await create(base, { 'X-Forwarded-For': '203.0.113.2' })).status).toBe(200);
    expect((await create(base, { 'X-Forwarded-For': '203.0.113.1' })).status).toBe(429);
    close?.();
    ({ base } = await boot({ limits }));
    expect((await create(base, { 'X-Forwarded-For': '203.0.113.1' })).status).toBe(200);
    expect((await create(base, { 'X-Forwarded-For': '203.0.113.2' })).status).toBe(429);   // same real address
  });

  it('caps open sockets per address; a closed one frees its slot', async () => {
    const { base } = await boot({ limits: { socketsPerIp: 2 } });
    const a = new WebSocket(`${base.replace('http', 'ws')}/ws`);
    await new Promise((r) => a.once('open', r));
    expect(await opens(base)).toBe(true);
    expect(await opens(base)).toBe(false);
    a.close();
    await new Promise((r) => setTimeout(r, 50));
    expect(await opens(base)).toBe(true);
  });

  it('with no limits (tests, dev) nothing is refused', async () => {
    const { base } = await boot({ limits: false });
    const statuses = await Promise.all(Array.from({ length: 30 }, async () => (await create(base)).status));
    expect(statuses.every((s) => s === 200)).toBe(true);
  });

  it('when full, refuses new rooms with 503 while the rooms already there keep working', async () => {
    const { base } = await boot({ limits: false }, 1);
    const { code } = (await (await create(base)).json()) as { code: string };
    const r = await create(base);
    expect(r.status).toBe(503);
    expect(((await r.json()) as { error: string }).error).toMatch(/full/);
    expect((await fetch(`${base}/api/rooms/${code}/join`, { method: 'POST', body: JSON.stringify({ name: 'Ada' }) })).status).toBe(200);
  });

  it('sends the security headers: no sniffing, no referrer, no framing, our own scripts only', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'jb-static-'));
    writeFileSync(join(dir, 'player.html'), '<!doctype html><title>x</title>');
    const { base } = await boot({ staticDir: dir });
    for (const path of ['/player.html', '/api/rooms/ZZZZ/lobby']) {
      const h = (await fetch(base + path)).headers;
      expect(h.get('x-content-type-options')).toBe('nosniff');
      expect(h.get('referrer-policy')).toBe('no-referrer');
      expect(h.get('x-frame-options')).toBe('SAMEORIGIN');            // our own pages only (the host's tab)
      const csp = h.get('content-security-policy')!;
      expect(csp).toContain("script-src 'self'");
      expect(csp).toContain('https://fonts.googleapis.com');
      expect(csp).toContain('https://fonts.gstatic.com');
      expect(csp).toContain(`ws://${base.slice('http://'.length)}`);
      expect(csp).toContain("frame-ancestors 'self'");
    }
  });
});
