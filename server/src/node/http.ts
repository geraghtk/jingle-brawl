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
import type { Rooms } from '../room/rooms';
import { wsServer } from './ws';

export interface ServerOpts { rooms: Rooms; staticDir?: string | null; debugApi?: boolean }

const TYPES: Record<string, string> = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json',
  '.png': 'image/png', '.webp': 'image/webp', '.svg': 'image/svg+xml', '.ico': 'image/x-icon', '.jpg': 'image/jpeg', '.woff2': 'font/woff2',
};
const MAX_BODY = 64_000;

function json(res: ServerResponse, status: number, body: unknown) {
  res.writeHead(status, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
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

const isLocal = (req: IncomingMessage) => ['127.0.0.1', '::1', '::ffff:127.0.0.1'].includes(req.socket.remoteAddress || '');

export function createAppServer({ rooms, staticDir, debugApi }: ServerOpts) {
  const ws = wsServer(rooms);
  const root = staticDir ? resolve(staticDir) : null;

  const server = createServer(async (req, res) => {
    try {
      const url = new URL(req.url || '/', 'http://x');
      const parts = url.pathname.split('/').filter(Boolean);

      if (url.pathname === '/healthz') return json(res, 200, { ok: true, now: rooms.clock.now() });
      if (url.pathname === '/api/info') return json(res, 200, { ips: lanAddresses() });

      if (parts[0] === 'api' && parts[1] === 'rooms') {
        if (parts.length === 2 && req.method === 'POST') {
          const b = (await body(req)) as { settings?: unknown } | null;
          const settings = b?.settings && typeof b.settings === 'object' && !Array.isArray(b.settings) ? b.settings : {};
          return json(res, 200, rooms.create(settings as Record<string, unknown>));
        }
        const room = parts[2] ? rooms.get(parts[2]) : null;
        if (parts[3] === 'lobby' && req.method === 'GET') {
          if (!room) return json(res, 200, { exists: false } satisfies LobbyRes);
          const g = room.state.game;
          return json(res, 200, {
            exists: true, phase: g.phase,
            avatars: Object.values(g.players || {}).map((p) => avatarOf(p.name, p.avatar).id),
          } satisfies LobbyRes);
        }
        if (parts[3] === 'join' && req.method === 'POST') {
          if (!room) return json(res, 404, { error: 'Game not found! Check the room code.' });
          const out = room.join(((await body(req)) || {}) as JoinReq);
          if ('error' in out) return json(res, 400, out);
          return json(res, 200, { pid: out.pid, token: rooms.token({ room: room.code, role: 'player', pid: out.pid }) });
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
        const rel = url.pathname === '/' ? 'player.html' : decodeURIComponent(url.pathname);
        const file = normalize(join(root, rel));
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
      json(res, 400, { error: (e as Error).message.slice(0, 200) });
    }
  });

  server.on('upgrade', (req, socket, head) => {
    if (new URL(req.url || '/', 'http://x').pathname === '/ws') ws.upgrade(req, socket, head);
    else socket.destroy();
  });

  return {
    server,
    close() { ws.close(); server.close(); },
  };
}
