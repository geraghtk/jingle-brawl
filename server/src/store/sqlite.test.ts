// Rooms on disk survive the server: close the file, open it again, and the room is all there.
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import { HOST } from '../../../web/src/core/table';
import { RoomActor } from '../room/actor';
import { replay } from '../room/replay';
import { FakeClock } from '../testing/fakeClock';
import { SqliteStore } from './sqlite';

const dirs: string[] = [];
afterEach(() => { for (const d of dirs.splice(0)) rmSync(d, { recursive: true, force: true }); });

describe('SqliteStore', () => {
  it('a room, its log and its remembered presses survive closing and reopening the file', () => {
    const dir = mkdtempSync(join(tmpdir(), 'jb-')); dirs.push(dir);
    const file = join(dir, 'rooms.db');
    const clock = new FakeClock();
    let store = new SqliteStore(file);
    const room = RoomActor.create('WXYZ', {}, { store, clock })!;
    for (const name of ['Ada', 'Bo']) room.join({ name });
    expect(room.handle(HOST, 'go', { type: 'start' })).toEqual({ ok: true });
    const saved = room.state;
    room.close();
    store.close();

    store = new SqliteStore(file);
    const back = RoomActor.load('WXYZ', { store, clock })!;
    expect(back.state.game.phase).toBe('main');
    expect(back.state.version).toBeGreaterThanOrEqual(saved.version);
    expect(back.handle(HOST, 'go', { type: 'start' })).toEqual({ ok: true });   // the same press, remembered
    expect(back.state.game.phase).toBe('main');
    expect(replay(store, 'WXYZ')).toEqual(store.load('WXYZ')!.game);
    expect(store.activeCodes(0)).toEqual(['WXYZ']);
    back.close();
    store.close();
  });
});
