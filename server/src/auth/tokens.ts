// Session tokens the server hands out: the host gets one when it creates a room, a phone when it
// joins. HMAC-signed, so nobody can claim to be the host or another player; who you are always
// comes from the token, never from a message.
import { createHmac, randomBytes, timingSafeEqual } from 'node:crypto';

// `born`: when the room was made. Codes are reused once a room is deleted, so a token names the one
// room it was issued in, not just its code.
export type Claims = { room: string; born: number; role: 'host' } | { room: string; born: number; role: 'player'; pid: string };

const b64 = (b: Buffer | string) => Buffer.from(b).toString('base64url');

export function newSecret(): string {
  return randomBytes(32).toString('base64url');
}

export function sign(secret: string, claims: Claims): string {
  const body = b64(JSON.stringify(claims));
  return `${body}.${b64(createHmac('sha256', secret).update(body).digest())}`;
}

export function verify(secret: string, token: string | undefined): Claims | null {
  if (typeof token !== 'string' || token.length > 512) return null;
  const [body, mac] = token.split('.');
  if (!body || !mac) return null;
  const want = createHmac('sha256', secret).update(body).digest();
  const got = Buffer.from(mac, 'base64url');
  if (got.length !== want.length || !timingSafeEqual(got, want)) return null;
  try {
    const c = JSON.parse(Buffer.from(body, 'base64url').toString('utf8'));
    if (typeof c?.room !== 'string' || typeof c.born !== 'number') return null;
    if (c.role === 'host') return { room: c.room, born: c.born, role: 'host' };
    if (c.role === 'player' && typeof c.pid === 'string') return { room: c.room, born: c.born, role: 'player', pid: c.pid };
    return null;
  } catch {
    return null;
  }
}
