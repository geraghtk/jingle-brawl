// Where rooms live between restarts: the current state of each room and an append-only log of
// everything that happened to it (enough to replay the room). Two implementations: SQLite on
// disk (the server) and memory (tests, `--db :memory:`).
import type { Game } from '../../../web/src/core/types';

export interface RoomRow { code: string; game: Game; version: number; updatedAt: number }
export interface EventRow { seq: number; at: number; who: string; kind: 'intent' | 'auto' | 'system' | 'debug'; body: unknown; ok: boolean }

export interface Store {
  /** A new room; false if the code is taken. */
  create(code: string, game: Game, now: number): boolean;
  load(code: string): RoomRow | null;
  /** Save a committed change and its event in one go. */
  commit(code: string, game: Game, version: number, event: Omit<EventRow, 'seq'>): void;
  /** Record a press (actionId) and how it went, so a resend gets the same answer. */
  rememberAction(code: string, actionId: string, ok: boolean, error?: string): void;
  recallAction(code: string, actionId: string): { ok: boolean; error?: string } | null;
  events(code: string, sinceSeq?: number, limit?: number): EventRow[];
  /** Rooms touched since `since` (restart recovery re-arms their timers). */
  activeCodes(since: number): string[];
  /** Rooms due for deletion: untouched since `idleBefore`, or ended and untouched since `endedBefore`. */
  expired(idleBefore: number, endedBefore: number): string[];
  /** Delete a room and everything kept about it (its log, its presses, its origin). */
  remove(code: string): void;
  meta(key: string): string | null;
  setMeta(key: string, value: string): void;
}

export class MemoryStore implements Store {
  private rooms = new Map<string, RoomRow>();
  private log = new Map<string, EventRow[]>();
  private actions = new Map<string, { ok: boolean; error?: string }>();
  private kv = new Map<string, string>();

  create(code: string, game: Game, now: number): boolean {
    if (this.rooms.has(code)) return false;
    this.rooms.set(code, { code, game: structuredClone(game), version: 1, updatedAt: now });
    this.log.set(code, []);
    return true;
  }
  load(code: string): RoomRow | null {
    const r = this.rooms.get(code);
    return r ? { ...r, game: structuredClone(r.game) } : null;
  }
  commit(code: string, game: Game, version: number, event: Omit<EventRow, 'seq'>): void {
    this.rooms.set(code, { code, game: structuredClone(game), version, updatedAt: event.at });
    const l = this.log.get(code) || [];
    l.push({ ...event, seq: l.length + 1 });
    this.log.set(code, l);
  }
  rememberAction(code: string, actionId: string, ok: boolean, error?: string): void {
    this.actions.set(`${code}|${actionId}`, { ok, ...(error ? { error } : {}) });
  }
  recallAction(code: string, actionId: string) {
    return this.actions.get(`${code}|${actionId}`) || null;
  }
  events(code: string, sinceSeq = 0, limit = 1000): EventRow[] {
    return (this.log.get(code) || []).filter((e) => e.seq > sinceSeq).slice(0, limit);
  }
  activeCodes(since: number): string[] {
    return [...this.rooms.values()].filter((r) => r.updatedAt >= since).map((r) => r.code);
  }
  expired(idleBefore: number, endedBefore: number): string[] {
    return [...this.rooms.values()]
      .filter((r) => r.updatedAt < idleBefore || (r.game.phase === 'ended' && r.updatedAt < endedBefore)).map((r) => r.code);
  }
  remove(code: string): void {
    this.rooms.delete(code);
    this.log.delete(code);
    this.kv.delete(`origin:${code}`);
    for (const k of this.actions.keys()) if (k.startsWith(`${code}|`)) this.actions.delete(k);
  }
  meta(key: string): string | null {
    return this.kv.get(key) ?? null;
  }
  setMeta(key: string, value: string): void {
    this.kv.set(key, value);
  }
}
