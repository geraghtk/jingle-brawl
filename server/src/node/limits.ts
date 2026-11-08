// Abuse limits, per client address: how fast one address may make rooms, join, look up room codes
// and open sockets, and how many sockets it may hold at once. Generous: a whole household plays
// behind one Wi-Fi router (one address), and phones reconnect a lot.
import type { IncomingMessage } from 'node:http';

/** A token bucket: `burst` at once, then `perMinute` more per minute. */
export interface Rate { burst: number; perMinute: number }

export interface Limits {
  create: Rate;          // POST /api/rooms (a host makes a room)
  join: Rate;            // POST /api/rooms/:code/join
  lobby: Rate;           // GET /api/rooms/:code/lobby (also what guessing room codes would use)
  connect: Rate;         // WebSocket handshakes
  report: Rate;          // POST /api/client-error (a browser's error reports)
  socketsPerIp: number;  // open WebSockets at once
}

export const DEFAULT_LIMITS: Limits = {
  create: { burst: 10, perMinute: 2 },
  join: { burst: 40, perMinute: 30 },
  lobby: { burst: 60, perMinute: 60 },
  connect: { burst: 120, perMinute: 120 },
  report: { burst: 20, perMinute: 10 },
  socketsPerIp: 150,
};

const MAX_TRACKED = 20_000;   // addresses remembered per bucket kind before idle ones are dropped

class Buckets {
  private m = new Map<string, { tokens: number; at: number }>();
  constructor(private rate: Rate) {}

  /** Take one token for `ip`: 0 if allowed, else how many ms until one is free. */
  take(ip: string, now: number): number {
    const { burst, perMinute } = this.rate;
    const perMs = perMinute / 60_000;
    const b = this.m.get(ip) || { tokens: burst, at: now };
    b.tokens = Math.min(burst, b.tokens + Math.max(0, now - b.at) * perMs);
    b.at = now;
    if (b.tokens < 1) { this.m.set(ip, b); return Math.ceil((1 - b.tokens) / perMs); }
    b.tokens -= 1;
    this.m.set(ip, b);
    if (this.m.size > MAX_TRACKED) this.prune(now);
    return 0;
  }

  private prune(now: number) {
    const perMs = this.rate.perMinute / 60_000;
    for (const [ip, b] of this.m) if (b.tokens + (now - b.at) * perMs >= this.rate.burst) this.m.delete(ip);
  }
}

export type Kind = 'create' | 'join' | 'lobby' | 'connect' | 'report';

export class Limiter {
  private buckets: Record<Kind, Buckets>;
  private open = new Map<string, number>();
  readonly limits: Limits;

  constructor(limits: Partial<Limits> = {}, private now: () => number = Date.now) {
    this.limits = { ...DEFAULT_LIMITS, ...limits };
    this.buckets = {
      create: new Buckets(this.limits.create), join: new Buckets(this.limits.join),
      lobby: new Buckets(this.limits.lobby), connect: new Buckets(this.limits.connect), report: new Buckets(this.limits.report),
    };
  }

  /** 0 = go ahead; else the ms to wait (for Retry-After). */
  take(kind: Kind, ip: string): number {
    return this.buckets[kind].take(ip, this.now());
  }

  /** A socket opens: false if this address already holds too many. */
  openSocket(ip: string): boolean {
    const n = this.open.get(ip) || 0;
    if (n >= this.limits.socketsPerIp) return false;
    this.open.set(ip, n + 1);
    return true;
  }

  closeSocket(ip: string): void {
    const n = (this.open.get(ip) || 1) - 1;
    if (n > 0) this.open.set(ip, n); else this.open.delete(ip);
  }
}

/** The caller's address. Behind a reverse proxy (`--trust-proxy`) that is the proxy's last hop in
 *  X-Forwarded-For (the address the proxy itself saw; earlier entries are whatever the client claimed). */
export function clientIp(req: IncomingMessage, trustProxy: boolean): string {
  if (trustProxy) {
    const xff = req.headers['x-forwarded-for'];
    const last = (Array.isArray(xff) ? xff.join(',') : xff || '').split(',').map((s) => s.trim()).filter(Boolean).at(-1);
    if (last) return last.replace(/^::ffff:/, '');
  }
  return (req.socket.remoteAddress || 'unknown').replace(/^::ffff:/, '');
}
