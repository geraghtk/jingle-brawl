// Every room on this server: made on demand, loaded from the store after a restart, and the
// token secret they share.
import { randomInt } from 'node:crypto';
import type { Settings } from '../../../web/src/core/types';
import { newSecret, sign } from '../auth/tokens';
import type { Claims } from '../auth/tokens';
import type { Store } from '../store/store';
import { RoomActor } from './actor';
import type { Clock } from './actor';

const CODE_CHARS = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'; // no 0/O, 1/I: read off a TV across the room
export const RECOVER_WITHIN_MS = 24 * 60 * 60 * 1000;

export class Rooms {
  private live = new Map<string, RoomActor>();
  readonly secret: string;

  constructor(readonly store: Store, readonly clock: Clock, opts: { secret?: string; seed?: () => number } = {}) {
    this.secret = opts.secret || store.meta('token_secret') || (() => { const s = newSecret(); store.setMeta('token_secret', s); return s; })();
    this.seedFn = opts.seed;
  }
  private seedFn?: () => number;

  private deps() {
    return { store: this.store, clock: this.clock, seed: this.seedFn };
  }

  /** A new room with a fresh code. */
  create(settings: Settings): { code: string; token: string } {
    for (let tries = 0; tries < 50; tries++) {
      const code = Array.from({ length: 4 }, () => CODE_CHARS[randomInt(CODE_CHARS.length)]).join('');
      const room = RoomActor.create(code, settings, this.deps());
      if (!room) continue;
      this.live.set(code, room);
      return { code, token: this.token({ room: code, role: 'host' }) };
    }
    throw new Error('no free room code');
  }

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

  token(claims: Claims): string {
    return sign(this.secret, claims);
  }

  close(): void {
    for (const r of this.live.values()) r.close();
    this.live.clear();
  }
}
