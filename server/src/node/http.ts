// HTTP: make a room (the host), look at one (the join screen), join one (a phone), the built app,
// and (only with --debug-api, only from this machine) the test helpers' direct reads and writes.
import { createReadStream, existsSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { networkInterfaces } from 'node:os';
import { extname, join, normalize, resolve, sep } from 'node:path';
import { getPath } from '../../../web/src/core/tx';
import { avatarOf } from '../../../web/src/core/avatars';
import type { JoinReq, LobbyRes } from '../../../web/src/net/protocol';
import { ServerFull } from '../room/rooms';
import type { Rooms } from '../room/rooms';
import { Limiter, clientIp } from './limits';
import type { Kind, Limits } from './limits';
import { Metrics, silentLogger } from './log';
import type { Logger } from './log';
import { wsServer } from './ws';

export interface ServerOpts {
  rooms: Rooms; staticDir?: string | null; debugApi?: boolean;
  /** per-address limits (false = none: tests and dev, where everyone arrives through loopback) */
  limits?: Partial<Limits> | false;
  /** behind a reverse proxy: take the client's address from X-Forwarded-For */
  trustProxy?: boolean;
  log?: Logger;
  /** lets /metrics answer a caller that is not this machine: `Authorization: Bearer <token>` */
  metricsToken?: string;
}

const TYPES: Record<string, string> = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json',
  '.png': 'image/png', '.webp': 'image/webp', '.svg': 'image/svg+xml', '.ico': 'image/x-icon', '.jpg': 'image/jpeg', '.woff2': 'font/woff2',
};
const MAX_BODY = 64_000;

/** On every response. Room codes ride in page URLs, so no Referer leaves the site; framing only by
 *  our own pages (the host's phone shows the console in a tab);
 *  pages run only our own scripts (fonts from Google; drawings are data:/blob: images). */
function secure(req: IncomingMessage, res: ServerResponse) {
  const host = /^[A-Za-z0-9.:[\]-]+$/.test(req.headers.host || '') ? req.headers.host : null;
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('Referrer-Policy', 'no-referrer');
  res.setHeader('X-Frame-Options', 'SAMEORIGIN');
  res.setHeader('Content-Security-Policy', [
    "default-src 'self'", "script-src 'self'", "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
    "font-src 'self' https://fonts.gstatic.com", "img-src 'self' data: blob:",
    `connect-src 'self'${host ? ` ws://${host} wss://${host}` : ''}`,
    "object-src 'none'", "base-uri 'none'", "form-action 'self'", "frame-ancestors 'self'", "frame-src 'self'",
  ].join('; '));
}

function json(res: ServerResponse, status: number, body: unknown, headers: Record<string, string> = {}) {
  res.writeHead(status, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store', ...headers });
  res.end(JSON.stringify(body ?? null));
}

async function body(req: IncomingMessage): Promise<unknown> {
  let size = 0;
  const chunks: Buffer[] = [];
  for await (const c of req) {
    size += (c as Buffer).length;
    if (size > MAX_BODY) throw new Error('too big');
    chunks.push(c as Buffer);
  }
  const s = Buffer.concat(chunks).toString('utf8');
  return s ? JSON.parse(s) : null;
}

/** This machine's IPv4 addresses on the local network (for the TV's join link). */
export const lanAddresses = () => Object.values(networkInterfaces()).flat()
  .filter((n) => n && n.family === 'IPv4' && !n.internal).map((n) => n!.address);

/** A browser's error report (POST /api/client-error): small, and only what helps find the bug. */
const clip = (v: unknown, n: number) => (typeof v === 'string' ? v.slice(0, n) : undefined);
export function clientError(b: unknown) {
  const r = (b && typeof b === 'object' ? b : {}) as Record<string, unknown>;
  const page = clip(r.page, 40);
  return { page: page && /^[a-z]+$/.test(page) ? page : 'unknown', message: clip(r.message, 500) || '(no message)',
    stack: clip(r.stack, 2000), build: clip(r.build, 40) };
}

/** Where a request is going, as a URL on this server. A path that starts with `//` would be read as
 *  another host's name (and may not parse at all): leading slashes count as one. Null: not a URL. */
function urlOf(req: IncomingMessage): URL | null {
  try { return new URL((req.url || '/').replace(/^\/{2,}/, '/'), 'http://x'); } catch { return null; }
}
const decoded = (path: string) => { try { return decodeURIComponent(path); } catch { return null; } };

const isLocal = (req: IncomingMessage) => ['127.0.0.1', '::1', '::ffff:127.0.0.1'].includes(req.socket.remoteAddress || '');

export function createAppServer({ rooms, staticDir, debugApi, limits, trustProxy = false, log = silentLogger, metricsToken }: ServerOpts) {
  const metrics = new Metrics();
  const ws = wsServer(rooms, { log, metrics });
  metrics.gauge('jb_rooms_live', () => rooms.size);
  metrics.gauge('jb_sockets_open', () => ws.size);
  const root = staticDir ? resolve(staticDir) : null;
  const limiter = limits === false ? null : new Limiter(limits || {}, () => rooms.clock.now());
  /** ms to wait if this caller is over its `kind` limit, else 0 */
  const over = (kind: Kind, req: IncomingMessage) => {
    const wait = limiter ? limiter.take(kind, clientIp(req, trustProxy)) : 0;
    if (wait) metrics.inc('jb_rate_limited_total', { kind });
    return wait;
  };
  /** /metrics: this machine (not through a proxy, where everyone looks local), or the token */
  const mayReadMetrics = (req: IncomingMessage) =>
    (!trustProxy && isLocal(req)) || (!!metricsToken && req.headers.authorization === `Bearer ${metricsToken}`);
  const tooMany = (res: ServerResponse, wait: number) =>
    json(res, 429, { error: 'Too many tries. Please wait a moment and try again.' }, { 'Retry-After': String(Math.ceil(wait / 1000)) });

  const server = createServer(async (req, res) => {
    try {
      secure(req, res);
      const url = urlOf(req);
      if (!url) return json(res, 404, { error: 'not found' });
      const parts = url.pathname.split('/').filter(Boolean);

      if (url.pathname === '/healthz') return json(res, 200, { ok: true, now: rooms.clock.now() });
      // the TV's Wi-Fi join link; behind a proxy the page's own address is the link
      if (url.pathname === '/api/info') return json(res, 200, { ips: trustProxy ? [] : lanAddresses() });
      if (url.pathname === '/metrics') {
        if (!mayReadMetrics(req)) return json(res, 404, { error: 'not found' });
        res.writeHead(200, { 'Content-Type': 'text/plain; version=0.0.4', 'Cache-Control': 'no-store' });
        return res.end(metrics.render());
      }
      if (url.pathname === '/api/client-error' && req.method === 'POST') {
        const wait = over('report', req);
        if (wait) return tooMany(res, wait);
        const e = clientError(await body(req));
        metrics.inc('jb_client_errors_total', { page: e.page });
        log.warn('client.error', e);
        res.writeHead(204);
        return res.end();
      }

      if (parts[0] === 'api' && parts[1] === 'rooms') {
        if (parts.length === 2 && req.method === 'POST') {
          const wait = over('create', req);
          if (wait) return tooMany(res, wait);
          const b = (await body(req)) as { settings?: unknown } | null;
          const settings = b?.settings && typeof b.settings === 'object' && !Array.isArray(b.settings) ? b.settings : {};
          const made = rooms.create(settings as Record<string, unknown>);
          metrics.inc('jb_rooms_created_total');
          log.info('room.created', { room: made.code });
          return json(res, 200, made);
        }
        const room = parts[2] ? rooms.get(parts[2]) : null;
        if (parts[3] === 'lobby' && req.method === 'GET') {
          const wait = over('lobby', req);
          if (wait) return tooMany(res, wait);
          if (!room) return json(res, 200, { exists: false } satisfies LobbyRes);
          const g = room.state.game;
          return json(res, 200, {
            exists: true, phase: g.phase,
            avatars: Object.values(g.players || {}).map((p) => avatarOf(p.name, p.avatar).id),
          } satisfies LobbyRes);
        }
        if (parts[3] === 'join' && req.method === 'POST') {
          const wait = over('join', req);
          if (wait) return tooMany(res, wait);
          if (!room) return json(res, 404, { error: 'Game not found! Check the room code.' });
          const out = room.join(((await body(req)) || {}) as JoinReq);
          if ('error' in out) return json(res, 400, out);
          metrics.inc('jb_joins_total');
          return json(res, 200, { pid: out.pid, token: rooms.token({ room: room.code, born: room.born, role: 'player', pid: out.pid }) });
        }
        return json(res, 404, { error: 'not found' });
      }

      if (parts[0] === 'debug') {
        if (!debugApi || !isLocal(req)) return json(res, 404, { error: 'not found' });
        const room = parts[1] === 'rooms' && parts[2] ? rooms.get(parts[2]) : null;
        if (!room) return json(res, 200, null);                     // like the database: a missing room reads as null
        const path = parts.slice(3).join('/');
        if (req.method === 'GET') return json(res, 200, path ? getPath(room.state.game, path) ?? null : room.state.game);
        if (req.method === 'POST' && path === 'reload') { rooms.evict(room.code); rooms.get(room.code); return json(res, 200, { ok: true }); }
        if (req.method === 'PUT' && path) { room.debugWrite({ [path]: await body(req) }); return json(res, 200, { ok: true }); }
        if (req.method === 'PATCH' && !path) { room.debugWrite(((await body(req)) || {}) as Record<string, unknown>); return json(res, 200, { ok: true }); }
        return json(res, 405, { error: 'method' });
      }

      if (root && (req.method === 'GET' || req.method === 'HEAD')) {
        const rel = url.pathname === '/' ? 'player.html' : decoded(url.pathname);
        const file = rel == null ? '' : normalize(join(root, rel));
        if ((file + sep).startsWith(root + sep) && existsSync(file) && statSync(file).isFile()) {
          const hashed = file.includes(`${sep}assets${sep}`);
          res.writeHead(200, {
            'Content-Type': TYPES[extname(file).toLowerCase()] || 'application/octet-stream',
            'Cache-Control': hashed ? 'public, max-age=31536000, immutable' : 'no-cache',
          });
          if (req.method === 'HEAD') return res.end();
          return createReadStream(file).pipe(res);
        }
      }
      json(res, 404, { error: 'not found' });
    } catch (e) {
      if (e instanceof ServerFull) {
        metrics.inc('jb_rooms_refused_total');
        log.warn('rooms.full', { live: rooms.size });
        return json(res, 503, { error: e.message }, { 'Retry-After': '300' });
      }
      log.warn('http.error', { path: (req.url || '').split('?')[0].slice(0, 100), error: (e as Error).message.slice(0, 200) });
      json(res, 400, { error: (e as Error).message.slice(0, 200) });
    }
  });

  server.on('upgrade', (req, socket, head) => {
    // (an error thrown here is uncaught: it would stop the server)
    if (urlOf(req)?.pathname !== '/ws') return socket.destroy();
    if (limiter) {
      const ip = clientIp(req, trustProxy);
      if (limiter.take('connect', ip) || !limiter.openSocket(ip)) {
        metrics.inc('jb_rate_limited_total', { kind: 'socket' });
        socket.end('HTTP/1.1 429 Too Many Requests\r\nRetry-After: 10\r\nContent-Length: 0\r\n\r\n');
        return;
      }
      socket.once('close', () => limiter.closeSocket(ip));
    }
    ws.upgrade(req, socket, head);
  });

  return {
    server, metrics,
    close() { ws.close(); server.close(); },
  };
}
