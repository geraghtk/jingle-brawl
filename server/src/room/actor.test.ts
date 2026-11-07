// The room on a fake clock: presses are checked and applied once, deadlines beat late presses,
// a restart picks the game up where it was, and no screen is sent a secret.
import { describe, expect, it } from 'vitest';
import type { PartyGame } from '../../../web/src/core/party';
import { HOST } from '../../../web/src/core/table';
import type { ServerMsg } from '../../../web/src/net/protocol';
import { sign, verify } from '../auth/tokens';
import { MemoryStore } from '../store/store';
import { PRESENCE_GRACE_MS, RoomActor } from './actor';
import type { Conn } from './actor';
import { replay } from './replay';
import { projectGame } from '../../../web/src/core/projection';
import { viewsFrom } from '../../../web/src/net/views';
import { FakeClock } from '../testing/fakeClock';

function screen(viewer: Conn['viewer']) {
  const got: ServerMsg[] = [];
  const conn: Conn = { viewer, send: (m) => got.push(JSON.parse(JSON.stringify(m))) };
  return { conn, got, views: () => viewsFrom(got), last: () => viewsFrom(got).at(-1) };
}

function setup(n = 4, settings = {}) {
  const store = new MemoryStore();
  const clock = new FakeClock();
  let s = 1;
  const deps = { store, clock, seed: () => s++ };
  const room = RoomActor.create('ABCD', { duelPick: 'off', ...settings }, deps)!;
  const host = screen({ kind: 'host' });
  room.attach(host.conn);
  const pids: string[] = [];
  const phones: Record<string, ReturnType<typeof screen>> = {};
  for (let i = 0; i < n; i++) {
    const r = room.join({ name: `P${i}` });
    if ('error' in r) throw new Error(r.error);
    pids.push(r.pid);
    phones[r.pid] = screen({ kind: 'phone', pid: r.pid });
    room.attach(phones[r.pid].conn);
  }
  let id = 0;
  const press = (who: string, intent: unknown) => room.handle(who, `a${++id}`, intent);
  return { store, clock, deps, room, host, phones, pids, press, game: () => room.state.game };
}

/** Start, draw, everyone but the opener bids `amount`, reveal. */
function revealed(t: ReturnType<typeof setup>, amount = 2) {
  expect(t.press(HOST, { type: 'start' })).toEqual({ ok: true });
  expect(t.press(HOST, { type: 'draw', giftName: 'Socks' })).toEqual({ ok: true });
  const opener = t.game().currentTurn!.openerId!;
  for (const p of t.pids.filter((x) => x !== opener)) expect(t.press(p, { type: 'bid', amount })).toEqual({ ok: true });
  expect(t.press(HOST, { type: 'reveal' })).toEqual({ ok: true });
  return opener;
}

describe('a room', () => {
  it('a press is applied once: the same action id again is answered, not redone', () => {
    const t = setup();
    expect(t.room.handle(HOST, 'x1', { type: 'start' })).toEqual({ ok: true });
    const v = t.room.state.version;
    expect(t.room.handle(HOST, 'x1', { type: 'start' })).toEqual({ ok: true });
    expect(t.room.state.version).toBe(v);
    expect(t.room.handle(HOST, 'x2', { type: 'start' }).ok).toBe(false);   // a second Start is no longer offered
  });

  it("refuses junk and presses that aren't on that screen", () => {
    const t = setup();
    expect(t.press(t.pids[0], { type: 'start' }).ok).toBe(false);
    expect(t.press(HOST, { type: 'bid', amount: 1, playerId: t.pids[0] }).ok).toBe(false);
    expect(t.press(HOST, 'start').ok).toBe(false);
  });

  it('every screen gets its own view: no phone or TV is ever sent another bid before the reveal', () => {
    const t = setup();
    const tv = screen({ kind: 'tv' });
    t.room.attach(tv.conn);
    t.press(HOST, { type: 'start' });
    t.press(HOST, { type: 'draw', giftName: 'Socks' });
    const opener = t.game().currentTurn!.openerId!;
    const bidders = t.pids.filter((x) => x !== opener);
    bidders.forEach((p, i) => t.press(p, { type: 'bid', amount: i + 1 }));
    const leaked = (msgs: ServerMsg[], me?: string) => viewsFrom(msgs).some((g) => ['bidding', 'endgame_bidding'].includes(g.currentTurn?.phase || '')
      && Object.entries(g.currentTurn!.bids || {}).some(([id, b]) => id !== me && b.amount !== 0));
    expect(leaked(tv.got)).toBe(false);
    expect(leaked(t.host.got)).toBe(false);
    for (const p of t.pids) expect(leaked(t.phones[p].got, p)).toBe(false);
    expect(t.phones[bidders[2]].last()!.currentTurn!.bids![bidders[2]].amount).toBe(3);
    t.press(HOST, { type: 'reveal' });
    expect(tv.last()!.currentTurn!.bids![bidders[2]].amount).toBe(3);
  });

  it('after the first whole view a screen is sent only what changed: a drawing goes over the wire once', () => {
    const t = setup(4, { tiebreakerMethod: 'sketch_off' });
    const opener = revealed(t);
    t.press(opener, { type: 'fight' });
    t.clock.advance(0);
    const mg = () => t.game().minigame as PartyGame;
    t.clock.advance(mg().phaseEndsAt - t.clock.now() + 1000);                 // drawing time
    const drawing = `data:image/jpeg;base64,${'A'.repeat(50_000)}`;
    for (const p of t.pids) t.press(p, { type: 'sketch', drawing });
    t.clock.advance(1000);                                                    // everyone drew: voting
    expect(mg().phase).toBe('voting');
    const phone = t.phones[t.pids[0]];
    const before = phone.got.length;
    t.press(t.pids[1], { type: 'vote_sketch', pick: t.pids[2] });            // one vote
    const sent = phone.got.slice(before).map((m) => JSON.stringify(m).length).reduce((a, b) => a + b, 0);
    expect(sent).toBeLessThan(1000);                                          // not 4 x 50 KB of drawings again
    expect(phone.last()).toEqual(projectGame(t.game(), { kind: 'phone', pid: t.pids[0] }));
  });

  it('runs the automatic steps on its own clock: a tie becomes a party game, which plays through and pays once', () => {
    const t = setup(4, { tiebreakerMethod: 'whack_elf' });
    const opener = revealed(t);
    expect(t.press(opener, { type: 'fight' })).toEqual({ ok: true });
    t.clock.advance(0);                                                    // the tie step is due now
    expect(t.game().currentTurn!.phase).toBe('minigame_party');
    const mg = () => t.game().minigame as PartyGame;
    t.clock.advance(mg().phaseEndsAt - t.clock.now() + 1000);             // intro → playing
    expect(mg().phase).toBe('playing');
    t.pids.forEach((p, i) => t.press(p, { type: 'whack', score: 10 - i }));
    t.clock.advance(60_000);                                              // results → the duel
    expect(t.game().minigame ?? null).toBeNull();
    const points = t.pids.reduce((s, p) => s + (t.game().players![p].partyPoints || 0), 0);
    expect(points).toBe(3 + 2 + 1);
  });

  it('a deadline beats a late press, even if the timer never got to run', () => {
    const t = setup(4, { tiebreakerMethod: 'whack_elf' });
    const opener = revealed(t);
    t.press(opener, { type: 'fight' });
    t.clock.advance(0);
    const mg = () => t.game().minigame as PartyGame;
    t.clock.advance(mg().phaseEndsAt - t.clock.now() + 1000);
    t.clock.jump(mg().phaseEndsAt - t.clock.now() + 60_000);              // play time is long over; no timer ran
    expect(t.press(t.pids[0], { type: 'whack', score: 50 }).ok).toBe(false);
    expect(mg()?.scores?.[t.pids[0]]).toBeUndefined();
  });

  it('a restart picks the room up where it was: timers come back from the state, nothing is paid twice', () => {
    const t = setup(4, { tiebreakerMethod: 'whack_elf' });
    const opener = revealed(t);
    t.press(opener, { type: 'fight' });
    t.clock.advance(0);
    const mg = t.game().minigame as PartyGame;
    t.clock.advance(mg.phaseEndsAt - t.clock.now() + 1000);
    t.pids.forEach((p, i) => t.press(p, { type: 'whack', score: 10 - i }));
    t.room.close();                                                       // the server dies mid-game
    const again = RoomActor.load('ABCD', t.deps)!;
    expect(again.state.game.minigame).toBeTruthy();
    t.clock.advance(60_000);                                              // nobody reconnects; the clock alone moves it on
    const g = again.state.game;
    expect(g.minigame ?? null).toBeNull();
    expect(t.pids.reduce((s, p) => s + (g.players![p].partyPoints || 0), 0)).toBe(6);
    expect(Object.values(g.players!).every((p) => p.connected === false)).toBe(true);
  });

  it('the log replays to the saved state', () => {
    const t = setup(4, { tiebreakerMethod: 'whack_elf' });
    const opener = revealed(t);
    t.press(opener, { type: 'fight' });
    t.clock.advance(120_000);
    expect(replay(t.store, 'ABCD')).toEqual(t.store.load('ABCD')!.game);
  });

  it('presence: a phone that drops is "away" only after a grace period (a quick reconnect is not)', () => {
    const t = setup(3);
    const p = t.pids[0];
    const connected = () => t.game().players![p].connected;
    expect(connected()).toBe(true);
    t.room.detach(t.phones[p].conn);
    t.clock.advance(PRESENCE_GRACE_MS - 1000);
    t.room.attach(screen({ kind: 'phone', pid: p }).conn);
    t.clock.advance(PRESENCE_GRACE_MS * 2);
    expect(connected()).toBe(true);
    t.room.detach([...(t.room as unknown as { conns: Set<Conn> }).conns].find((c) => c.viewer.kind === 'phone' && c.viewer.pid === p)!);
    t.clock.advance(PRESENCE_GRACE_MS + 1);
    expect(connected()).toBe(false);
  });
});

describe('tokens', () => {
  it('only the server can make one; a player token is not a host token; another room is another room', () => {
    const tok = sign('s3cret', { room: 'ABCD', role: 'player', pid: 'p_1' });
    expect(verify('s3cret', tok)).toEqual({ room: 'ABCD', role: 'player', pid: 'p_1' });
    expect(verify('other', tok)).toBeNull();
    const [body, mac] = tok.split('.');
    const forged = Buffer.from(JSON.stringify({ room: 'ABCD', role: 'host' })).toString('base64url');
    expect(verify('s3cret', `${forged}.${mac}`)).toBeNull();
    expect(verify('s3cret', `${body}.`)).toBeNull();
    expect(verify('s3cret', undefined)).toBeNull();
  });
});

