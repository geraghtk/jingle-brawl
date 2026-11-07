// The WebSocket side: one socket per screen. The first message says which room and who (a token;
// none = the TV). After that a screen sends intents and pings; the room sends it its own view.
import type { IncomingMessage } from 'node:http';
import type { Duplex } from 'node:stream';
import { WebSocketServer } from 'ws';
import type { WebSocket } from 'ws';
import { HOST } from '../../../web/src/core/table';
import type { Viewer } from '../../../web/src/core/projection';
import { PROTOCOL } from '../../../web/src/net/protocol';
import type { ClientMsg, ServerMsg } from '../../../web/src/net/protocol';
import { verify } from '../auth/tokens';
import type { Conn, RoomActor } from '../room/actor';
import type { Rooms } from '../room/rooms';

export const MAX_FRAME = 1_000_000;        // a drawing is the biggest thing a phone sends
const HELLO_WITHIN_MS = 10_000;
const BURST = 40, PER_SECOND = 20;         // per socket: a phone tapping Whack-an-Elf stays well under

export function wsServer(rooms: Rooms) {
  const wss = new WebSocketServer({ noServer: true, maxPayload: MAX_FRAME });

  wss.on('connection', (ws: WebSocket) => {
    let room: RoomActor | null = null;
    let conn: Conn | null = null;
    let who: string | null = null;
    let tokens = BURST, last = Date.now();
    const send = (m: ServerMsg) => { if (ws.readyState === ws.OPEN) ws.send(JSON.stringify(m)); };
    const fail = (error: string) => { send({ t: 'error', error, fatal: true }); ws.close(4000, error.slice(0, 100)); };
    const hello = setTimeout(() => fail('no hello'), HELLO_WITHIN_MS);

    ws.on('message', (data) => {
      const now = Date.now();
      tokens = Math.min(BURST, tokens + ((now - last) / 1000) * PER_SECOND);
      last = now;
      if (--tokens < 0) return send({ t: 'error', error: 'slow down' });
      let msg: ClientMsg;
      try { msg = JSON.parse(String(data)); } catch { return send({ t: 'error', error: 'not JSON' }); }
      if (!msg || typeof msg !== 'object') return;

      if (msg.t === 'ping') return send({ t: 'pong', c: Number(msg.c) || 0, s: rooms.clock.now() });
      if (msg.t === 'hello') {
        if (conn) return;
        clearTimeout(hello);
        if (msg.v !== PROTOCOL) return fail('please reload: the game was updated');
        const r = rooms.get(msg.room);
        if (!r) return fail('no such room');
        let viewer: Viewer = { kind: 'tv' };
        if (msg.token) {
          const c = verify(rooms.secret, msg.token);
          if (!c || c.room !== r.code) return fail('not your room');
          if (c.role === 'player' && !r.hasPlayer(c.pid)) return fail('you are not in this game any more');
          viewer = c.role === 'host' ? { kind: 'host' } : { kind: 'phone', pid: c.pid };
        }
        room = r;
        who = viewer.kind === 'host' ? HOST : viewer.kind === 'phone' ? viewer.pid : null;
        conn = { viewer, send };
        r.attach(conn);
        return;
      }
      if (msg.t === 'intent') {
        if (!room || !conn) return send({ t: 'error', error: 'say hello first' });
        if (!who) return send({ t: 'ack', id: String(msg.id), ok: false, error: 'the TV only watches' });
        const ack = room.handle(who, msg.id, msg.intent);
        return send({ t: 'ack', id: String(msg.id), ...ack });
      }
    });
    ws.on('close', () => { clearTimeout(hello); if (room && conn) room.detach(conn); });
    ws.on('error', () => {});
  });

  return {
    upgrade(req: IncomingMessage, socket: Duplex, head: Buffer) {
      wss.handleUpgrade(req, socket, head, (ws) => wss.emit('connection', ws, req));
    },
    close() { for (const c of wss.clients) c.terminate(); wss.close(); },
  };
}
