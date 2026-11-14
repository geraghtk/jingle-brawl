// Every room on this server: made on demand, loaded from the store after a restart, and the
// token secret they share.
import { randomInt } from 'node:crypto';
import { sanitizeSettings } from '../../../web/src/core/settings';
import type { Settings } from '../../../web/src/core/types';
import { newSecret, sign } from '../auth/tokens';
import type { Claims } from '../auth/tokens';
import type { Store } from '../store/store';
import { RoomActor } from './actor';
import type { Clock, CriticSource } from './actor';

const CODE_CHARS = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'; // no 0/O, 1/I: read off a TV across the room
export const RECOVER_WITHIN_MS = 24 * 60 * 60 * 1000;
export const MAX_LIVE_ROOMS = 2000;
/** How long rooms are kept: deleted after a day untouched, or two hours after the game ended. */
export const KEEP_IDLE_MS = 24 * 60 * 60 * 1000;
export const KEEP_ENDED_MS = 2 * 60 * 60 * 1000;

/** Thrown by `create` when the server already holds as many live rooms as it should. */
export class ServerFull extends Error {
  constructor() { super('The server is full right now. Please try again in a few minutes.'); }
}

export class Rooms {
  private live = new Map<string, RoomActor>();
  readonly secret: string;

  readonly maxLive: number;

  constructor(readonly store: Store, readonly clock: Clock, opts: { secret?: string; seed?: () => number; critics?: CriticSource; maxLive?: number } = {}) {
    this.maxLive = opts.maxLive ?? MAX_LIVE_ROOMS;
    this.secret = opts.secret || store.meta('token_secret') || (() => { const s = newSecret(); store.setMeta('token_secret', s); return s; })();
    this.seedFn = opts.seed;
    this.critics = opts.critics;
  }
  private seedFn?: () => number;
  private critics?: CriticSource;

  private deps() {
    return { store: this.store, clock: this.clock, seed: this.seedFn, critics: this.critics };
  }

  /** A new room with a fresh code (or ServerFull: rooms already playing keep working). */
  create(settings: Settings): { code: string; token: string } {
    if (this.live.size >= this.maxLive) throw new ServerFull();
    for (let tries = 0; tries < 50; tries++) {
      const code = Array.from({ length: 4 }, () => CODE_CHARS[randomInt(CODE_CHARS.length)]).join('');
      const room = RoomActor.create(code, sanitizeSettings(settings), this.deps());
      if (!room) continue;
      this.live.set(code, room);
      return { code, token: this.token({ room: code, born: room.born, role: 'host' }) };
    }
    throw new Error('no free room code');
  }

  get size(): number { return this.live.size; }

  get(code: string): RoomActor | null {
    const c = String(code || '').toUpperCase();
    if (!/^[A-Z0-9]{4,8}$/.test(c)) return null;
    let room = this.live.get(c) || null;
    if (!room) {
      room = RoomActor.load(c, this.deps());
      if (room) this.live.set(c, room);
    }
    return room;
  }

  /** After a restart: load the recent rooms so their timers run again without anyone connecting. */
  recover(): number {
    let n = 0;
    for (const code of this.store.activeCodes(this.clock.now() - RECOVER_WITHIN_MS)) if (this.get(code)) n++;
    return n;
  }

  /** Drop a room from memory (it reloads from the store on next use): the debug API's restart. */
  evict(code: string): void {
    this.live.get(code)?.close();
    this.live.delete(code);
  }

  /** Put away rooms nobody has touched for `idleMs`: they leave memory (and forget any AI key);
   *  the store keeps them, so a returning screen loads them again. */
  sweep(idleMs: number): number {
    let n = 0;
    for (const [code, room] of this.live) {
      if (this.clock.now() - room.lastActivity > idleMs) { this.evict(code); n++; }   // open screens just reconnect
    }
    return n;
  }

  /** Delete rooms nobody needs any more, from memory and the store (screens still open on one are
   *  told it is gone; its code can be used again, and its old tokens name the old room). */
  purge(keepIdleMs = KEEP_IDLE_MS, keepEndedMs = KEEP_ENDED_MS): string[] {
    const now = this.clock.now();
    const gone = this.store.expired(now - keepIdleMs, now - keepEndedMs);
    for (const code of gone) {
      const room = this.live.get(code);
      // someone is still here (a TV showing the standings, a phone that just came back): keep it,
      // unless nothing at all has happened in it for the whole idle time
      const quiet = room ? now - room.lastActivity : Infinity;
      if (room && (quiet <= Math.min(keepIdleMs, keepEndedMs) || (room.screens > 0 && quiet <= keepIdleMs))) continue;
      room?.close();
      this.live.delete(code);
      this.store.remove(code);
    }
    return gone.filter((c) => !this.live.has(c));
  }

  token(claims: Claims): string {
    return sign(this.secret, claims);
  }

  close(): void {
    for (const r of this.live.values()) r.close();
    this.live.clear();
  }
}
