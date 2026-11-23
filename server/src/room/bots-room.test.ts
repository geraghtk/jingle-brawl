// Practice bots in a real room on a fake clock: the host adds them in the lobby, the room plays them,
// every Game Lab game (the runtime's and the two party games) runs to its end with nobody else at the
// table, and nothing of them is left when the real game starts.
import { describe, expect, it } from 'vitest';
import { BOT_TICK_MS, botIds } from '../../../web/src/core/bots';
import { arenaOf } from '../../../web/src/core/minigames/runtime';
import type { PartyGame } from '../../../web/src/core/party';
import { whackBounds } from '../../../web/src/core/party';
import { HOST } from '../../../web/src/core/table';
import { MemoryStore } from '../store/store';
import { FakeClock } from '../testing/fakeClock';
import { RoomActor } from './actor';

function room(bots: number, people = 0, seed = 1) {
  const store = new MemoryStore();
  const clock = new FakeClock();
  let s = seed, id = 0;
  const deps = { store, clock, seed: () => s++ };
  const r = RoomActor.create('BOTS', {}, deps)!;
  const press = (who: string, intent: unknown) => r.handle(who, `a${++id}`, intent);
  const pids: string[] = [];
  for (let i = 0; i < people; i++) { const j = r.join({ name: `P${i}` }); if ('pid' in j) pids.push(j.pid); }
  for (let i = 0; i < bots; i++) expect(press(HOST, { type: 'lab_add_bot' })).toEqual({ ok: true });
  return { r, clock, deps, store, press, pids, game: () => r.state.game, bots: () => botIds(r.state.game) };
}

describe('practice bots', () => {
  it('the host adds bots in the lobby: named, marked, here; and takes them away again', () => {
    const t = room(3);
    const bots = t.bots().map((id) => t.game().players![id]);
    expect(bots).toHaveLength(3);
    expect(new Set(bots.map((b) => b.name)).size).toBe(3);
    for (const b of bots) expect(b).toMatchObject({ isBot: true, connected: true });
    expect(t.press('p_someone', { type: 'lab_add_bot' })).toMatchObject({ ok: false });   // the host's button only
    expect(t.press(HOST, { type: 'lab_remove_bots' })).toEqual({ ok: true });
    expect(t.bots()).toHaveLength(0);
    for (let i = 0; i < 8; i++) t.press(HOST, { type: 'lab_add_bot' });
    expect(t.press(HOST, { type: 'lab_add_bot' })).toMatchObject({ ok: false });           // enough
  });

  it('bots are not players of the real game: two people are still needed, and Start sends the bots home', () => {
    const t = room(3, 1);
    expect(t.press(HOST, { type: 'start' })).toMatchObject({ ok: false });
    const more = t.r.join({ name: 'Second' });
    expect('pid' in more).toBe(true);
    expect(t.press(HOST, { type: 'start' })).toEqual({ ok: true });
    expect(t.bots()).toHaveLength(0);
    expect(Object.keys(t.game().players!)).toHaveLength(2);
    expect(t.press(HOST, { type: 'lab_add_bot' })).toMatchObject({ ok: false });           // the lobby only
  });

  for (const [gameId, n] of [['nutcracker_knockout', 2], ['gift_conveyor', 4], ['mecha_santa', 4], ['elf_tower_wars', 4], ['reindeer_ring_toss', 3], ['pass_the_gremlin', 3]] as const) {
    it(`bots alone play ${gameId} to a result`, () => {
      const t = room(n);
      expect(t.press(HOST, { type: 'lab_start', gameId, players: t.bots() })).toEqual({ ok: true });
      const v0 = arenaOf(t.game())!.stateVersion;
      t.clock.advance(20_000);
      expect(arenaOf(t.game())!.stateVersion).toBeGreaterThan(v0 + 3);                      // they are pressing things
      t.clock.advance(6 * 60_000);
      const a = arenaOf(t.game())!;
      expect(a.lifecycle).toBe('completed');
      expect(['resolved', 'requires_tiebreaker']).toContain(a.result!.status);
      expect(t.game().phase).toBe('lobby');
      expect(t.press(HOST, { type: 'lab_end' })).toEqual({ ok: true });
      expect(t.game().currentTurn ?? null).toBeNull();
    });
  }

  it('Pass the Gremlin in the lab: when it catches a bystander the bots play on into round 2, to a duel winner', () => {
    let played = 0;
    for (let seed = 1; seed < 40 && played < 3; seed++) {
      const t = room(5, 0, seed * 7);
      expect(t.press(HOST, { type: 'lab_start', gameId: 'pass_the_gremlin', players: t.bots() })).toEqual({ ok: true });
      expect(arenaOf(t.game())!.mode).toBe('party_with_duelists');
      const state = () => arenaOf(t.game())!.state as { out?: string[]; roundNo?: number; stage: string; pool: string[] };
      for (let i = 0; i < 110 && !arenaOf(t.game())!.result && !(state().out || []).length; i++) t.clock.advance(1000);
      if (!(state().out || []).length) continue;                    // a duelist was caught first: over in one round
      const [gone] = state().out!;
      expect(arenaOf(t.game())!.result).toBeUndefined();
      t.clock.advance(6000);                                        // the beat, then the next round starts by itself
      expect(state().roundNo).toBe(2);
      expect(state().pool).not.toContain(gone);
      t.clock.advance(6 * 60_000);
      const a = arenaOf(t.game())!;
      expect(a.lifecycle).toBe('completed');
      expect(t.bots().slice(0, 2)).toContain(a.result!.duel!.winningDuelistId);
      played++;
    }
    expect(played).toBeGreaterThan(0);
  });

  it('the Conveyor: bots deliver toys (they do the work, not just wait for the bell)', () => {
    const t = room(4);
    t.press(HOST, { type: 'lab_start', gameId: 'gift_conveyor', players: t.bots() });
    t.clock.advance(60_000);
    const teams = (arenaOf(t.game())!.state as { teams: { delivered: number }[] }).teams;
    expect(teams[0].delivered + teams[1].delivered).toBeGreaterThan(2);
  });

  it('Whack-an-Elf can be tried in the lab: bots score what a person could, the results stay, nothing is won', () => {
    const t = room(2, 1);
    const all = Object.keys(t.game().players!);
    expect(t.press(HOST, { type: 'lab_start', gameId: 'whack_elf', players: all })).toEqual({ ok: true });
    expect(t.game().currentTurn).toMatchObject({ phase: 'minigame_party', lab: { gameId: 'whack_elf' } });
    const mg = () => t.game().minigame as PartyGame;
    expect(mg()).toMatchObject({ type: 'whack_elf', minigameGoal: 'lab', prize: 'none', participants: all });
    t.clock.advance(7000 + 10_000);
    expect(mg().phase).toBe('playing');
    const max = whackBounds(mg().seed).max;
    for (const b of t.bots()) { expect(mg().scores![b].score).toBeGreaterThan(0); expect(mg().scores![b].score).toBeLessThan(max); }
    t.clock.advance(15_000);
    expect(mg().phase).toBe('results');
    for (const b of t.bots()) expect(mg().scores![b].final).toBe(true);
    expect(mg().results!.winners).toEqual([]);
    for (const p of Object.values(t.game().players!)) { expect(p.partyPoints ?? 0).toBe(0); expect(p.chips).toBe(10); }
    t.clock.advance(60_000);
    expect(mg().phase).toBe('results');                                                     // until the host moves on
    expect(t.press(HOST, { type: 'skip_party' })).toMatchObject({ ok: false });
    expect(t.press(HOST, { type: 'lab_start', gameId: 'whack_elf', players: all })).toEqual({ ok: true });   // play again
    expect(mg().phase).toBe('intro');
    expect(t.press(HOST, { type: 'start' })).toMatchObject({ ok: false });                 // not over a practice game
    expect(t.press(HOST, { type: 'lab_end' })).toEqual({ ok: true });
    expect(t.game().minigame ?? null).toBeNull();
  });

  it('Sketch-Off in the lab: bots hand in a real picture and vote', () => {
    const t = room(3);
    expect(t.press(HOST, { type: 'lab_start', gameId: 'sketch_off', players: t.bots().slice(0, 2) })).toMatchObject({ ok: false });   // a vote needs a choice
    expect(t.press(HOST, { type: 'lab_start', gameId: 'sketch_off', players: t.bots() })).toEqual({ ok: true });
    t.clock.advance(7000 + 60_000);
    const mg = () => t.game().minigame as PartyGame;
    for (const b of t.bots()) expect(mg().submissions![b]).toMatch(/^data:image\/png;base64,iVBORw0KGgo/);
    t.clock.advance(40_000);
    expect(mg().phase).toBe('results');
    expect(Object.keys(mg().votes!)).toHaveLength(3);
  });

  it('the bot tick stops when nothing is on, and a restart leaves the bots at the table', () => {
    const t = room(2);
    expect(t.r.nextWakeAt()).toBeNull();
    t.press(HOST, { type: 'lab_start', gameId: 'nutcracker_knockout', players: t.bots() });
    const v = t.r.state.version;
    t.clock.advance(BOT_TICK_MS * 60);
    expect(t.r.state.version).toBeGreaterThan(v);
    t.r.close();
    const again = RoomActor.load('BOTS', t.deps)!;
    for (const id of botIds(again.state.game)) expect(again.state.game.players![id].connected).toBe(true);
  });
});
