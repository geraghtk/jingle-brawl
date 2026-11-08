// Rooms are deleted for good, from memory and the store (memory and SQLite alike): a day after
// anyone last touched them, two hours after the game ended. A room in use is never deleted, and once
// a code comes back for a new room, tokens from the old one don't open it.
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import { verify } from '../auth/tokens';
import { MemoryStore } from '../store/store';
import type { Store } from '../store/store';
import { SqliteStore } from '../store/sqlite';
import { FakeClock } from '../testing/fakeClock';
import { Rooms } from './rooms';

const HOUR = 60 * 60 * 1000;
const dirs: string[] = [];
const cleanup: (() => void)[] = [];
afterEach(() => { for (const f of cleanup.splice(0)) f(); for (const d of dirs.splice(0)) rmSync(d, { recursive: true, force: true }); });

const stores: [string, () => Store][] = [
  ['memory', () => new MemoryStore()],
  ['sqlite', () => {
    const dir = mkdtempSync(join(tmpdir(), 'jb-purge-')); dirs.push(dir);
    const s = new SqliteStore(join(dir, 'rooms.db'));
    cleanup.push(() => s.close());
    return s;
  }],
];

describe.each(stores)('purging rooms (%s store)', (_name, makeStore) => {
  function boot() {
    const clock = new FakeClock();
    const store = makeStore();
    const rooms = new Rooms(store, clock, { secret: 'test' });
    cleanup.unshift(() => rooms.close());
    return { clock, store, rooms };
  }

  it('deletes a room a day after anyone touched it, with its log and its origin; a busy one stays', () => {
    const { clock, store, rooms } = boot();
    const idle = rooms.create({}).code;
    clock.advance(20 * HOUR);
    const busy = rooms.create({}).code;
    rooms.get(busy)!.join({ name: 'Ada' });
    expect(rooms.purge()).toEqual([]);
    clock.advance(5 * HOUR);
    rooms.sweep(2 * HOUR);                                         // both left memory hours ago
    expect(rooms.purge()).toEqual([idle]);
    expect(store.load(idle)).toBeNull();
    expect(store.events(idle)).toEqual([]);
    expect(store.meta(`origin:${idle}`)).toBeNull();
    expect(store.load(busy)).not.toBeNull();
    expect(rooms.get(idle)).toBeNull();
  });

  it('deletes an ended game two hours after it ended, not before', () => {
    const { clock, store, rooms } = boot();
    const code = rooms.create({}).code;
    rooms.get(code)!.debugWrite({ phase: 'ended' });
    clock.advance(HOUR);
    rooms.sweep(30 * 60 * 1000);
    expect(rooms.purge()).toEqual([]);
    clock.advance(HOUR + 1);
    expect(rooms.purge()).toEqual([code]);
    expect(store.load(code)).toBeNull();
  });

  it('never deletes a room someone is still in, even past its time', () => {
    const { clock, rooms } = boot();
    const code = rooms.create({}).code;
    const room = rooms.get(code)!;
    room.debugWrite({ phase: 'ended' });
    clock.advance(3 * HOUR);
    room.attach({ viewer: { kind: 'tv' }, send: () => {} });       // the TV is still showing the standings
    expect(rooms.purge()).toEqual([]);
    expect(rooms.get(code)).toBe(room);
  });

  it("a code used again is a new room: the old room's host token names the old one", () => {
    const { clock, store, rooms } = boot();
    const { code, token } = rooms.create({});
    const before = verify('test', token)!;
    expect(before.born).toBe(rooms.get(code)!.born);
    clock.advance(25 * HOUR);
    rooms.sweep(HOUR);
    rooms.purge();
    store.create(code, { phase: 'lobby', createdAt: clock.now(), settings: {} }, clock.now());   // the code comes back
    expect(rooms.get(code)!.born).not.toBe(before.born);
  });
});
