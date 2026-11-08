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
import { diff } from '../../../web/src/core/diff';
import { checkEndgameContinue, drawEndgamePlayer } from '../../../web/src/core/duel';
import { isDrawOff, judged } from '../../../web/src/core/drawoff';
import { arenaOf } from '../../../web/src/core/minigames/runtime';
import type { DrawOffScore } from '../../../web/src/core/drawoff';
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
import type { Critic, Provider } from '../critic/critic';

export interface Clock {
  now(): number;
  /** Run fn at `when` (server time); returns a cancel. */
  at(when: number, fn: () => void): () => void;
}

/** A screen's connection; close() drops it (the screen reconnects), as a restart would. */
export interface Conn { viewer: Viewer; send(msg: ServerMsg): void; close?(): void }

/** Where AI Draw Off's critic comes from: the host's own key (make), or a default (local critic, an env key, tests). */
export interface CriticSource {
  make(provider: Provider, key: string): Critic;
  /** A critic without a key from the host (the local critic, or the server's own env key), if any. */
  fallback(provider: Provider): Critic | null;
}

export interface Deps { store: Store; clock: Clock; seed?: () => number; critics?: CriticSource }

const CRITIC_CALLS_PER_10_MIN = 12;   // a stuck loop can't run up the host's bill

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
  /** What each screen was last sent (its view and the version), so the next send is only what changed. */
  private sent = new WeakMap<Conn, { view: Game; version: number }>();
  /** AI Draw Off's critic. Only in memory: never saved, logged, or put in any view. */
  private critic: { c: Critic; ready: boolean; error?: string; own: boolean } | null = null;
  private judgingFor: number | null = null;
  private criticCalls: number[] = [];
  /** When anyone last did anything here (an idle room is put away, forgetting its key). */
  lastActivity = 0;

  private constructor(readonly code: string, private deps: Deps, game: Game, version: number) {
    this.game = game;
    this.version = version;
    this.lastActivity = deps.clock.now();
    const fallback = this.provider() && deps.critics?.fallback(this.provider()!);
    if (fallback) this.critic = { c: fallback, ready: true, own: false };
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
    room.pauseArenaAfterRestart(row.updatedAt);
    room.schedule();
    return room;
  }

  /** When this room was made: tokens carry it, so one for an earlier room with this code is refused. */
  get born(): number { return this.game.createdAt ?? 0; }

  get state(): { game: Game; version: number } {
    return { game: this.game, version: this.version };
  }

  // --- screens ---------------------------------------------------------------------------------

  attach(conn: Conn): void {
    this.lastActivity = this.now();
    this.conns.add(conn);
    conn.send({ t: 'welcome', viewer: conn.viewer.kind, ...(conn.viewer.kind === 'phone' ? { pid: conn.viewer.pid } : {}) });
    if (conn.viewer.kind === 'phone') this.connected(conn.viewer.pid, +1);
    this.sendState(conn);
    if (conn.viewer.kind === 'host') {
      if (this.provider()) conn.send(this.criticStatus());
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
    this.lastActivity = this.now();
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

  /** A mini-game running when the server went down is paused with the time it had then; the host resumes it. */
  private pauseArenaAfterRestart(lastSaved: number): void {
    const a = arenaOf(this.game);
    if (a?.lifecycle !== 'active') return;
    const left = Object.fromEntries(a.deadlines.map((d) => [d.id, Math.max(0, d.at - lastSaved)]));
    const tx = new Tx(this.game).set('arena', { ...a, lifecycle: 'paused', paused: { at: this.now(), left }, deadlines: [], stateVersion: a.stateVersion + 1,
      overrides: [...a.overrides, { at: this.now(), what: 'paused: the server restarted' }] }).event('⏸️ Paused: the server restarted. The host can resume.', 'duel');
    this.commit(tx.plan(), HOST, 'system', { restart: 'arena paused' });
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
    this.lastActivity = at;
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

  /** Debug API (tests): write paths directly. Logged. */
  debugWrite(updates: Record<string, unknown>): void {
    const tx = new Tx(this.game);
    for (const [k, v] of Object.entries(updates)) tx.set(k, v);
    this.commit(tx.plan(), HOST, 'debug', { updates: Object.keys(updates) });
    this.fired.clear();
    this.afterChange();
  }

  private afterChange(): void {
    this.schedule();
    if (this.game.phase === 'ended' && this.critic?.own) this.forgetKey(); // the game is over: forget the host's key
    this.maybeJudge();
    const views = new Map<string, Game>();                // each distinct view is worked out once
    for (const c of this.conns) {
      const key = c.viewer.kind === 'phone' ? `phone:${c.viewer.pid}` : c.viewer.kind;
      if (!views.has(key)) views.set(key, projectGame(this.game, c.viewer));
      this.sendView(c, views.get(key)!);
    }
  }

  /** The whole view the first time, then only what changed in it. */
  private sendView(conn: Conn, view: Game): void {
    const last = this.sent.get(conn);
    const now = this.now();
    if (!last) conn.send({ t: 'state', version: this.version, game: view, now });
    else {
      const updates = diff(last.view, view);
      if (!Object.keys(updates).length) { this.sent.set(conn, { view, version: last.version }); return; }
      conn.send({ t: 'patch', base: last.version, version: this.version, updates, now });
    }
    this.sent.set(conn, { view, version: this.version });
  }

  private sendState(conn: Conn): void {
    this.sent.delete(conn);
    this.sendView(conn, projectGame(this.game, conn.viewer));
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

  // --- AI Draw Off's critic ----------------------------------------------------------------------

  private provider(): Provider | null {
    const b = this.game.settings?.aiCriticBackend;
    return b === 'openai' || b === 'gemini' || b === 'local' ? b : b === 'ollama' ? 'local' : null;
  }

  private get fallbackCritic(): Critic | null {
    return (this.provider() && this.deps.critics?.fallback(this.provider()!)) || null;
  }

  private criticStatus(): ServerMsg {
    return { t: 'critic', ready: !!this.critic?.ready, ...(this.critic ? { backend: this.critic.c.backend } : {}), ...(this.critic?.error ? { error: this.critic.error } : {}) };
  }

  private tellHost(): void {
    const msg = this.criticStatus();
    for (const c of this.conns) if (c.viewer.kind === 'host') c.send(msg);
  }

  /** The host gives (or forgets) its own key. Checked once with the provider; kept in memory only. */
  setCriticKey(key: string | null): void {
    this.lastActivity = this.now();
    const provider = this.provider();
    if (!key || !provider || provider === 'local' || !this.deps.critics) { this.forgetKey(); return; }
    const c = this.deps.critics.make(provider, key.slice(0, 300));
    this.critic = { c, ready: false, own: true };
    this.tellHost();
    c.check().then(() => {
      if (this.critic?.c !== c) return;
      this.critic = { c, ready: true, own: true };
      this.tellHost();
    }).catch((e) => {
      if (this.critic?.c !== c) return;
      this.critic = { c, ready: false, own: true, error: scrub((e as Error).message, key) };
      this.tellHost();
    });
  }

  private forgetKey(): void {
    const fb = this.fallbackCritic;
    this.critic = fb ? { c: fb, ready: true, own: false } : null;
    this.tellHost();
  }

  /** A Draw Off is waiting for its judge: ask the critic (once), then commit the scores or why it couldn't. */
  private maybeJudge(): void {
    const mg = this.game.minigame;
    if (!isDrawOff(mg) || mg.phase !== 'judging' || this.judgingFor === mg.startedAt) return;
    const startedAt = mg.startedAt;
    this.judgingFor = startedAt;
    const finish = (result: Parameters<typeof judged>[1]) => {
      const now = this.game.minigame;
      if (this.closed || !isDrawOff(now) || now.startedAt !== startedAt || now.phase !== 'judging') return;
      if (this.commit(judged(this.game, result, this.now()), HOST, 'auto', { judged: 'failed' in result ? 'failed' : 'scored' })) this.afterChange();
    };
    const critic = this.critic;
    if (!critic?.ready) return finish({ failed: critic?.error ? `the AI critic failed: ${critic.error}` : 'no AI critic key' });
    const now = this.now();
    this.criticCalls = this.criticCalls.filter((t) => t > now - 10 * 60_000);
    if (this.criticCalls.length >= CRITIC_CALLS_PER_10_MIN) return finish({ failed: 'too many critic calls, try again in a few minutes' });
    this.criticCalls.push(now, now, now);
    const names = Object.fromEntries(mg.participants.map((p) => [p, this.game.players?.[p]?.name || 'Someone']));
    (async () => {
      const scores: Record<string, DrawOffScore> = {};
      await Promise.all(mg.participants.map(async (p) => {
        const img = mg.submissions?.[p];
        scores[p] = img ? await critic.c.judge(img, mg.prompt) : { rating: 0, description: 'No drawing', critique: 'Nothing on the page!' };
      }));
      const [a, b] = mg.participants;
      const top = (scores[a].rating || 0) === (scores[b].rating || 0) ? null : (scores[a].rating > scores[b].rating ? a : b);
      const verdict = top ? await critic.c.verdict(mg.prompt, names[top], mg.participants.map((p) => ({ name: names[p], rating: scores[p].rating, description: scores[p].description || '' })))
        .catch(() => null) : null;
      finish({ scores, verdict });
    })().catch((e) => finish({ failed: `the AI critic failed: ${scrub((e as Error).message, '')}` }));
  }

  close(): void {
    this.closed = true;
    for (const c of [...this.conns]) c.close?.();
    this.conns.clear();
    this.cancelTimer?.();
    for (const cancel of this.away.values()) cancel();
    this.away.clear();
  }
}

function logLines(e: EventRow): (GameEvent & { at: number })[] {
  const events = (e.body as { events?: GameEvent[] } | null)?.events || [];
  return events.map((x) => ({ ...x, at: e.at }));
}

/** An error to show the host, without any key in it (providers echo part of a bad key back). */
function scrub(message: string, key: string): string {
  let m = String(message || 'error').slice(0, 160);
  if (key) m = m.split(key).join('***');
  return m.replace(/(sk-[A-Za-z0-9_-]{4,}|AIza[A-Za-z0-9_-]{8,})/g, '***');
}
