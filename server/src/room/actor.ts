// One room: the referee for one game. Everything that changes the game goes through here, one
// thing at a time (every entry point is synchronous, so Node runs them in order):
//   a screen's intent → parse → seen this press before? → run the automatic steps that are due
//   (a deadline beats a late press) → dispatch (refused unless it's on that screen) → commit →
//   save the state and the event → re-arm the timer → send every screen its own view.
// The actor knows nothing about sockets or files: it is handed a Clock, a Store and connections,
// so the same room runs in the Node server, in tests on a fake clock, or behind another runtime.
import { randomInt } from 'node:crypto';
import * as A from '../../../web/src/core/actions';
import { commitPlan } from '../../../web/src/core/commit';
import { checkEndgameContinue, drawEndgamePlayer } from '../../../web/src/core/duel';
import { hostAuto } from '../../../web/src/core/hostauto';
import { dispatch, parseIntent } from '../../../web/src/core/intents';
import type { Intent } from '../../../web/src/core/intents';
import { projectGame } from '../../../web/src/core/projection';
import type { Viewer } from '../../../web/src/core/projection';
import { rng } from '../../../web/src/core/rng';
import { dueStep } from '../../../web/src/core/scheduler';
import { HOST } from '../../../web/src/core/table';
import type { Who } from '../../../web/src/core/table';
import { Tx } from '../../../web/src/core/tx';
import type { Game, GameEvent, Plan, Settings } from '../../../web/src/core/types';
import type { JoinReq, ServerMsg } from '../../../web/src/net/protocol';
import type { EventRow, Store } from '../store/store';

export interface Clock {
  now(): number;
  /** Run fn at `when` (server time); returns a cancel. */
  at(when: number, fn: () => void): () => void;
}

export interface Conn { viewer: Viewer; send(msg: ServerMsg): void }

export interface Deps { store: Store; clock: Clock; seed?: () => number }

export type Ack = { ok: true } | { ok: false; error: string };

export const PRESENCE_GRACE_MS = 5000;  // a phone that drops its socket is "away" only after this
const THEN_DELAY_MS = 500;              // endgame follow-ups (the host page used to wait for its listener)

export class RoomActor {
  private game: Game;
  private version: number;
  private fired = new Set<string>();
  private conns = new Set<Conn>();
  private cancelTimer: (() => void) | null = null;
  private sockets = new Map<string, number>();
  private away = new Map<string, () => void>();
  private closed = false;

  private constructor(readonly code: string, private deps: Deps, game: Game, version: number) {
    this.game = game;
    this.version = version;
  }

  /** A new room (null if the code is taken). */
  static create(code: string, settings: Settings, deps: Deps): RoomActor | null {
    const now = deps.clock.now();
    const game = A.newGame({ hostId: HOST, hostUid: null, settings, now });
    if (!deps.store.create(code, game, now)) return null;
    deps.store.setMeta(`origin:${code}`, JSON.stringify(game));
    const room = new RoomActor(code, deps, game, 1);
    room.schedule();
    return room;
  }

  /** A room from the store (after a restart: its timers come back from its state). */
  static load(code: string, deps: Deps): RoomActor | null {
    const row = deps.store.load(code);
    if (!row) return null;
    const room = new RoomActor(code, deps, row.game, row.version);
    // nobody is connected to a room that was just loaded: their sockets reattach
    room.markAllAway();
    room.schedule();
    return room;
  }

  get state(): { game: Game; version: number } {
    return { game: this.game, version: this.version };
  }

  // --- screens ---------------------------------------------------------------------------------

  attach(conn: Conn): void {
    this.conns.add(conn);
    conn.send({ t: 'welcome', viewer: conn.viewer.kind, ...(conn.viewer.kind === 'phone' ? { pid: conn.viewer.pid } : {}) });
    if (conn.viewer.kind === 'phone') this.connected(conn.viewer.pid, +1);
    this.sendState(conn);
    if (conn.viewer.kind === 'host') {
      const log = this.deps.store.events(this.code, 0, 100000).flatMap(logLines).slice(-300);
      if (log.length) conn.send({ t: 'log', events: log });
    }
  }

  detach(conn: Conn): void {
    if (!this.conns.delete(conn)) return;
    if (conn.viewer.kind === 'phone') this.connected(conn.viewer.pid, -1);
  }

  get screens(): number {
    return this.conns.size;
  }

  // --- what people do --------------------------------------------------------------------------

  /** A press from a screen. `actionId` makes resends safe: the same press is answered, not redone. */
  handle(who: Who, actionId: string, raw: unknown): Ack {
    if (typeof actionId !== 'string' || !actionId || actionId.length > 80) return { ok: false, error: 'bad action id' };
    const seen = this.deps.store.recallAction(this.code, actionId);
    if (seen) return seen.ok ? { ok: true } : { ok: false, error: seen.error || 'refused' };
    const intent = parseIntent(raw);
    if ('error' in intent) return this.remember(actionId, { ok: false, error: intent.error });
    this.runDue();                                       // a deadline that has passed wins over this press
    const seed = this.seed();
    const plan = dispatch(this.game, who, intent as Intent, { now: this.now(), rng: rng(seed) });
    if (plan.error) return this.remember(actionId, { ok: false, error: plan.error });
    if (!this.commit(plan, who, 'intent', { intent, seed })) return this.remember(actionId, { ok: false, error: 'someone got there first' });
    this.afterChange();
    return this.remember(actionId, { ok: true });
  }

  /** A phone joining: the server picks the player id. */
  join(req: JoinReq): { pid: string } | { error: string } {
    const pid = `p_${randomInt(36 ** 6).toString(36).padStart(6, '0')}`;
    const plan = A.joinPlayer(this.game, { playerId: pid, name: String(req.name ?? ''), uid: null, now: this.now(),
      age: typeof req.age === 'number' ? req.age : null, avatar: typeof req.avatar === 'string' ? req.avatar : null });
    if (plan.error) return { error: plan.error };
    // they join "away" until their phone's socket arrives
    const tx = new Tx(this.game);
    for (const [k, v] of Object.entries(plan.updates)) tx.set(k, v);
    tx.set(`players/${pid}/connected`, false);
    this.commit(tx.plan(), pid, 'system', { join: { name: req.name } });
    this.afterChange();
    return { pid };
  }

  hasPlayer(pid: string): boolean {
    return !!this.game.players?.[pid];
  }

  // --- time ------------------------------------------------------------------------------------

  /** Run every automatic host step that is due now (party-game phases, votes, the wheel …). */
  private runDue(): boolean {
    let changed = false;
    for (let guard = 0; guard < 50; guard++) {
      const now = this.now();
      const step = dueStep(hostAuto(this.game), this.fired, now, now);
      if (!step) break;
      this.fired.add(step.key);
      const seed = this.seed();
      if (this.commit(step.run(this.game, { now, rng: rng(seed) }), HOST, 'auto', { step: step.key, seed })) changed = true;
    }
    return changed;
  }

  private schedule(): void {
    this.cancelTimer?.();
    this.cancelTimer = null;
    if (this.closed) return;
    const now = this.now();
    const live = hostAuto(this.game).filter((s) => !this.fired.has(s.key));
    if (!live.length) return;
    const at = Math.min(...live.map((s) => s.at || now));
    this.cancelTimer = this.deps.clock.at(Math.max(at, now), () => {
      this.cancelTimer = null;
      if (this.runDue()) this.afterChange();
      else this.schedule();
    });
  }

  /** Next time this room needs to wake up (a Durable Objects runtime would set its alarm here). */
  nextWakeAt(): number | null {
    const now = this.now();
    const live = hostAuto(this.game).filter((s) => !this.fired.has(s.key));
    return live.length ? Math.min(...live.map((s) => s.at || now)) : null;
  }

  // --- presence --------------------------------------------------------------------------------

  private connected(pid: string, delta: number): void {
    const n = Math.max(0, (this.sockets.get(pid) || 0) + delta);
    this.sockets.set(pid, n);
    if (n > 0) {
      this.away.get(pid)?.();
      this.away.delete(pid);
      if (this.game.players?.[pid] && this.game.players[pid].connected !== true) this.setConnected(pid, true);
    } else if (!this.away.has(pid)) {
      this.away.set(pid, this.deps.clock.at(this.now() + PRESENCE_GRACE_MS, () => {
        this.away.delete(pid);
        if (!this.sockets.get(pid) && this.game.players?.[pid]?.connected) this.setConnected(pid, false);
      }));
    }
  }

  private setConnected(pid: string, on: boolean): void {
    this.commit(new Tx(this.game).set(`players/${pid}/connected`, on).plan(), pid, 'system', { connected: on });
    this.afterChange();
  }

  private markAllAway(): void {
    const tx = new Tx(this.game);
    let any = false;
    for (const [id, p] of Object.entries(this.game.players || {})) if (p.connected) { tx.set(`players/${id}/connected`, false); any = true; }
    if (any) this.commit(tx.plan(), HOST, 'system', { restart: true });
  }

  // --- committing ------------------------------------------------------------------------------

  private commit(plan: Plan, who: Who, kind: EventRow['kind'], body: unknown): boolean {
    const at = this.now();
    const { game, applied } = commitPlan(this.game, plan, at);
    if (!applied) return false;
    this.game = game;
    this.version++;
    // the event keeps the plan too, so the room can be replayed from its log (replay.ts)
    this.deps.store.commit(this.code, game, this.version, {
      at, who, kind, ok: true,
      body: { ...(body as object), events: plan.events, plan: { updates: plan.updates, ...(plan.claim ? { claim: plan.claim } : {}) } },
    });
    if (plan.events.length) this.broadcastLog(plan.events.map((e) => ({ ...e, at })));
    if (plan.then) {
      const then = plan.then;
      this.deps.clock.at(at + THEN_DELAY_MS, () => {
        const seed = this.seed();
        const next = then === 'drawEndgamePlayer' ? drawEndgamePlayer(this.game, rng(seed)) : checkEndgameContinue(this.game, rng(seed));
        if (!next.error && this.commit(next, HOST, 'auto', { then, seed })) this.afterChange();
      });
    }
    return true;
  }

  /** Debug API (tests): write paths directly, as the old emulator helpers did. Logged. */
  debugWrite(updates: Record<string, unknown>): void {
    const tx = new Tx(this.game);
    for (const [k, v] of Object.entries(updates)) tx.set(k, v);
    this.commit(tx.plan(), HOST, 'debug', { updates: Object.keys(updates) });
    this.fired.clear();
    this.afterChange();
  }

  private afterChange(): void {
    this.schedule();
    for (const c of this.conns) this.sendState(c);
  }

  private sendState(conn: Conn): void {
    conn.send({ t: 'state', version: this.version, game: projectGame(this.game, conn.viewer), now: this.now() });
  }

  private broadcastLog(events: (GameEvent & { at: number })[]): void {
    for (const c of this.conns) if (c.viewer.kind === 'host') c.send({ t: 'log', events });
  }

  private remember(actionId: string, ack: Ack): Ack {
    this.deps.store.rememberAction(this.code, actionId, ack.ok, ack.ok ? undefined : ack.error);
    return ack;
  }

  private now(): number {
    return this.deps.clock.now();
  }

  private seed(): number {
    return this.deps.seed ? this.deps.seed() : randomInt(2 ** 31);
  }

  close(): void {
    this.closed = true;
    this.cancelTimer?.();
    for (const cancel of this.away.values()) cancel();
    this.away.clear();
  }
}

function logLines(e: EventRow): (GameEvent & { at: number })[] {
  const events = (e.body as { events?: GameEvent[] } | null)?.events || [];
  return events.map((x) => ({ ...x, at: e.at }));
}
