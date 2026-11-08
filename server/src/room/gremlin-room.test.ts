// Pass the Gremlin as a duel game in a room: the server's clock dooms the holder even if no
// timer ran, a late pass loses to the deadline, the duel settles once, a restart pauses the round
// with its time intact, and a reconnecting phone gets its same problem back.
import { describe, expect, it } from 'vitest';
import type { GremlinState } from '../../../web/src/core/minigames/gremlin/gremlin';
import { arenaOf } from '../../../web/src/core/minigames/runtime';
import { HOST } from '../../../web/src/core/table';
import type { ServerMsg } from '../../../web/src/net/protocol';
import { viewsFrom } from '../../../web/src/net/views';
import { MemoryStore } from '../store/store';
import { FakeClock } from '../testing/fakeClock';
import { RoomActor } from './actor';
import type { Conn } from './actor';

function setup(names = ['Ada', 'Bo']) {
  const store = new MemoryStore();
  const clock = new FakeClock();
  let s = 1;
  const deps = { store, clock, seed: () => s++ };
  const room = RoomActor.create('GREM', { duelPick: 'wheel', duelGames: [], duelAppGames: ['pass_the_gremlin'], voteWindow: 0 }, deps)!;
  const pids = names.map((name) => { const r = room.join({ name }); if ('error' in r) throw new Error(r.error); return r.pid; });
  const phones: Record<string, ServerMsg[]> = {};
  for (const p of pids) { phones[p] = []; room.attach({ viewer: { kind: 'phone', pid: p }, send: (m) => phones[p].push(JSON.parse(JSON.stringify(m))) }); }
  let n = 0;
  const press = (who: string, intent: unknown) => room.handle(who, `a${++n}`, intent);
  expect(press(HOST, { type: 'start' })).toEqual({ ok: true });
  room.debugWrite({ currentTurn: { phase: 'duel', challengerId: pids[0], defenderId: pids[1], pot: 2 } });
  clock.advance(10_000);                                             // the wheel picks the Gremlin
  expect(room.state.game.currentTurn?.duelGame?.app).toBe('pass_the_gremlin');
  expect(press(HOST, { type: 'start_duel_app' })).toEqual({ ok: true });
  return { store, clock, deps, room, pids, phones, press, st: () => arenaOf(room.state.game)!.state as GremlinState };
}

describe('Pass the Gremlin in a room', () => {
  it('two duelists: the server dooms the holder; a pass after the deadline loses to it; the duel settles once', () => {
    const t = setup();
    expect(arenaOf(t.room.state.game)).toMatchObject({ mode: 'direct_duel', lifecycle: 'active' });
    const holder = t.st().holder;
    expect(t.press(holder, { type: 'arena', action: { kind: 'answer', value: t.st().challenge.answer } })).toEqual({ ok: true });
    t.clock.jump(t.st().doomAt - t.clock.now() + 10);                // the deadline passes; no timer got to run
    const late = t.press(holder, { type: 'arena', action: { kind: 'pass_random' } });
    expect(late.ok).toBe(false);                                     // the deadline ran first: the holder is doomed
    t.clock.advance(1000);                                           // the hand-off and the duel resolve
    const g = t.room.state.game;
    expect(g.lastDuel).toMatchObject({ loserId: holder, winnerId: t.pids.find((p) => p !== holder) });
    const paid = Object.values(g.players!).reduce((sum, p) => sum + p.chips, 0);
    t.clock.advance(60_000);
    expect(Object.values(t.room.state.game.players!).reduce((sum, p) => sum + p.chips, 0)).toBe(paid);   // nothing paid twice
  });

  it('a restart mid-round pauses it with its time intact; the doom does not fire during the outage', () => {
    const t = setup();
    t.clock.advance(5000);
    // the server can't know when it died: it gives back what was left at the last save
    const left = t.st().doomAt - t.store.load('GREM')!.updatedAt;
    t.room.close();
    t.clock.jump(5 * 60_000);                                        // the server was down for five minutes
    const back = RoomActor.load('GREM', t.deps)!;
    const a = arenaOf(back.state.game)!;
    expect(a.lifecycle).toBe('paused');
    expect(a.paused!.left.doom).toBe(left);
    expect(a.overrides.at(-1)!.what).toMatch(/server restarted/);
    expect(back.handle(HOST, 'resume', { type: 'arena_resume' })).toEqual({ ok: true });
    expect(arenaOf(back.state.game)!.deadlines[0].at).toBe(t.clock.now() + left);
    back.close();
  });

  it("a phone that reconnects gets its same problem back (the room's state, not the phone's)", () => {
    const t = setup();
    const holder = t.st().holder;
    const prompt = t.st().challenge.prompt;
    const again: ServerMsg[] = [];
    const conn: Conn = { viewer: { kind: 'phone', pid: holder }, send: (m) => again.push(JSON.parse(JSON.stringify(m))) };
    t.room.attach(conn);
    const view = viewsFrom(again).at(-1)!.arenaView!.view as { prompt?: string };
    expect(view.prompt).toBe(prompt);
    expect(JSON.stringify(again)).not.toContain(String(t.st().challenge.answer) + ',');
  });

  it('three at the table: everyone plays, and the duel still ends with exactly one of the duelists winning', () => {
    const t = setup(['Ada', 'Bo', 'Cy']);
    expect(arenaOf(t.room.state.game)!.mode).toBe('party_with_duelists');
    for (let i = 0; i < 6 && !t.room.state.game.lastDuel; i++) t.clock.advance(120_000);   // dooms, maybe sudden death
    const g = t.room.state.game;
    expect(g.lastDuel).toBeTruthy();
    expect(t.pids.slice(0, 2)).toContain(g.lastDuel!.winnerId);
  });
});
