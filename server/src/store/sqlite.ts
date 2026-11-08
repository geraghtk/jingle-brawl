// Rooms on disk: SQLite (Node's built-in node:sqlite), one file for every room. Each commit
// writes the room's new state and its event in one transaction.
import { createRequire } from 'node:module';
import type { DatabaseSync as Db } from 'node:sqlite';

// loaded at run time: bundlers and Vite don't all know the node:sqlite builtin yet
const { DatabaseSync } = createRequire(import.meta.url)('node:sqlite') as typeof import('node:sqlite');
import type { Game } from '../../../web/src/core/types';
import type { EventRow, RoomRow, Store } from './store';

export class SqliteStore implements Store {
  private db: Db;

  constructor(path: string) {
    this.db = new DatabaseSync(path);
    this.db.exec(`
      PRAGMA journal_mode = WAL;
      PRAGMA synchronous = NORMAL;
      CREATE TABLE IF NOT EXISTS rooms (code TEXT PRIMARY KEY, game TEXT NOT NULL, version INTEGER NOT NULL, updated_at INTEGER NOT NULL);
      CREATE TABLE IF NOT EXISTS events (room TEXT NOT NULL, seq INTEGER NOT NULL, at INTEGER NOT NULL, who TEXT NOT NULL,
        kind TEXT NOT NULL, body TEXT NOT NULL, ok INTEGER NOT NULL, PRIMARY KEY (room, seq));
      CREATE TABLE IF NOT EXISTS actions (room TEXT NOT NULL, action_id TEXT NOT NULL, ok INTEGER NOT NULL, error TEXT,
        PRIMARY KEY (room, action_id));
      CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
      CREATE INDEX IF NOT EXISTS rooms_updated ON rooms (updated_at);
    `);
  }

  private tx<T>(fn: () => T): T {
    this.db.exec('BEGIN');
    try {
      const out = fn();
      this.db.exec('COMMIT');
      return out;
    } catch (e) {
      this.db.exec('ROLLBACK');
      throw e;
    }
  }

  create(code: string, game: Game, now: number): boolean {
    const r = this.db.prepare('INSERT OR IGNORE INTO rooms (code, game, version, updated_at) VALUES (?, ?, 1, ?)').run(code, JSON.stringify(game), now);
    return Number(r.changes) === 1;
  }

  load(code: string): RoomRow | null {
    const r = this.db.prepare('SELECT code, game, version, updated_at FROM rooms WHERE code = ?').get(code) as
      { code: string; game: string; version: number; updated_at: number } | undefined;
    return r ? { code: r.code, game: JSON.parse(r.game), version: Number(r.version), updatedAt: Number(r.updated_at) } : null;
  }

  commit(code: string, game: Game, version: number, event: Omit<EventRow, 'seq'>): void {
    this.tx(() => {
      this.db.prepare('UPDATE rooms SET game = ?, version = ?, updated_at = ? WHERE code = ?').run(JSON.stringify(game), version, event.at, code);
      const seq = Number((this.db.prepare('SELECT COALESCE(MAX(seq), 0) AS s FROM events WHERE room = ?').get(code) as { s: number }).s) + 1;
      this.db.prepare('INSERT INTO events (room, seq, at, who, kind, body, ok) VALUES (?, ?, ?, ?, ?, ?, ?)')
        .run(code, seq, event.at, event.who, event.kind, JSON.stringify(event.body ?? null), event.ok ? 1 : 0);
    });
  }

  rememberAction(code: string, actionId: string, ok: boolean, error?: string): void {
    this.db.prepare('INSERT OR REPLACE INTO actions (room, action_id, ok, error) VALUES (?, ?, ?, ?)').run(code, actionId, ok ? 1 : 0, error ?? null);
  }

  recallAction(code: string, actionId: string) {
    const r = this.db.prepare('SELECT ok, error FROM actions WHERE room = ? AND action_id = ?').get(code, actionId) as { ok: number; error: string | null } | undefined;
    return r ? { ok: !!r.ok, ...(r.error ? { error: r.error } : {}) } : null;
  }

  events(code: string, sinceSeq = 0, limit = 1000): EventRow[] {
    const rows = this.db.prepare('SELECT seq, at, who, kind, body, ok FROM events WHERE room = ? AND seq > ? ORDER BY seq LIMIT ?').all(code, sinceSeq, limit) as
      { seq: number; at: number; who: string; kind: EventRow['kind']; body: string; ok: number }[];
    return rows.map((r) => ({ seq: Number(r.seq), at: Number(r.at), who: r.who, kind: r.kind, body: JSON.parse(r.body), ok: !!r.ok }));
  }

  activeCodes(since: number): string[] {
    return (this.db.prepare('SELECT code FROM rooms WHERE updated_at >= ?').all(since) as { code: string }[]).map((r) => r.code);
  }

  expired(idleBefore: number, endedBefore: number): string[] {
    return (this.db.prepare(`SELECT code FROM rooms WHERE updated_at < ?
      OR (updated_at < ? AND json_extract(game, '$.phase') = 'ended')`).all(idleBefore, endedBefore) as { code: string }[]).map((r) => r.code);
  }

  remove(code: string): void {
    this.tx(() => {
      for (const t of ['rooms', 'events', 'actions']) this.db.prepare(`DELETE FROM ${t} WHERE ${t === 'rooms' ? 'code' : 'room'} = ?`).run(code);
      this.db.prepare('DELETE FROM meta WHERE k = ?').run(`origin:${code}`);
    });
  }

  meta(key: string): string | null {
    const r = this.db.prepare('SELECT v FROM meta WHERE k = ?').get(key) as { v: string } | undefined;
    return r ? r.v : null;
  }

  setMeta(key: string, value: string): void {
    this.db.prepare('INSERT OR REPLACE INTO meta (k, v) VALUES (?, ?)').run(key, value);
  }

  close(): void {
    this.db.close();
  }
}
