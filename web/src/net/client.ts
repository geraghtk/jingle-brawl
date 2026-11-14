// One screen's connection to its room on the server: the latest view of the game, presses that
// survive a dropped connection (resent with the same id, so they apply once), and the server's
// clock (so every screen agrees when a phase ends). Reconnects by itself.
import type { Intent } from '../core/intents';
import { applyUpdates } from '../core/tx';
import type { Game, GameEvent } from '../core/types';
import { PROTOCOL } from './protocol';
import type { ClientMsg, ServerMsg } from './protocol';

export type Status = 'connecting' | 'open' | 'closed';
export type Ack = { ok: boolean; error?: string };
type Log = GameEvent & { at: number };

const PING_MS = 5000;
const SAMPLES = 6;
/** No word from the room for this long: the connection is dead even if the browser hasn't noticed
 *  (a phone that slept, a Wi-Fi hop). Drop it and connect again. */
export const SILENT_MS = 12_000;

let seq = 0;
/** Unique per press (crypto.randomUUID needs a secure page; phones on the Wi-Fi use plain http). */
const actionId = () => `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}-${(++seq).toString(36)}`;

export class RoomClient {
  game: Game | null = null;
  version = 0;
  status: Status = 'connecting';
  /** Server time minus this device's time (ms). */
  offset = 0;
  /** A reason the room won't have this screen (no such room, not your token): stop trying. */
  fatal: string | null = null;
  /** The host's view of AI Draw Off's critic (null: the game has none). */
  critic: { ready: boolean; backend?: string; error?: string } | null = null;
  private criticKey: string | null | undefined = undefined;   // waiting to be sent (memory only)

  private ws: WebSocket | null = null;
  private pending = new Map<string, { intent: Intent; resolve: (a: Ack) => void }>();
  private listeners = new Set<() => void>();
  private logListeners = new Set<(events: Log[]) => void>();
  private samples: { rtt: number; offset: number }[] = [];
  private retry = 0;
  private retryTimer: ReturnType<typeof setTimeout> | null = null;
  private pinger: ReturnType<typeof setInterval> | null = null;
  private stopped = false;
  private heard = 0;
  private seats = new Map<string, (token: string | null) => void>();
  private onVisible = () => { if (document.visibilityState === 'visible') this.check(); };

  constructor(readonly room: string, private token: string | undefined, private url = defaultUrl()) {}

  start(): this {
    // again after stop() too: React's development mode mounts a screen, unmounts it and mounts it again
    this.stopped = false;
    if (!this.ws || this.ws.readyState > WebSocket.OPEN) this.open();   // not a second socket
    if (typeof document !== 'undefined') document.addEventListener('visibilitychange', this.onVisible);
    return this;
  }

  stop(): void {
    this.stopped = true;
    if (typeof document !== 'undefined') document.removeEventListener('visibilitychange', this.onVisible);
    if (this.pinger) { clearInterval(this.pinger); this.pinger = null; }
    if (this.retryTimer) { clearTimeout(this.retryTimer); this.retryTimer = null; }
    // let go of the socket now: whatever it still says (or its close) is not this screen's any more
    const ws = this.ws;
    this.ws = null;
    ws?.close();
    this.setStatus('closed');
    for (const p of this.pending.values()) p.resolve({ ok: false, error: 'left the room' });
    this.pending.clear();
  }

  /** The host: a token that seats `pid` on another phone (null if the room said no). */
  seatToken(pid: string): Promise<string | null> {
    return new Promise((resolve) => {
      if (this.status !== 'open') return resolve(null);
      this.seats.get(pid)?.(null);
      this.seats.set(pid, resolve);
      this.write({ t: 'seat', pid });
      setTimeout(() => { if (this.seats.get(pid) === resolve) { this.seats.delete(pid); resolve(null); } }, 5000);
    });
  }

  /** Ask the room for something. Resolves with the room's answer (kept across reconnects). */
  send(intent: Intent): Promise<Ack> {
    if (this.fatal) return Promise.resolve({ ok: false, error: this.fatal });
    return new Promise((resolve) => {
      const id = actionId();
      this.pending.set(id, { intent, resolve });
      this.write({ t: 'intent', id, intent });
    });
  }

  /** The host's own AI key for this room: sent once over this socket, kept nowhere here. null forgets it. */
  setCriticKey(key: string | null): void {
    this.criticKey = key;
    this.flushKey();
  }

  private flushKey(): void {
    if (this.criticKey === undefined || this.status !== 'open' || this.ws?.readyState !== WebSocket.OPEN) return;
    this.ws.send(JSON.stringify({ t: 'critic_key', key: this.criticKey }));
    this.criticKey = undefined;
  }

  subscribe(fn: () => void): () => void {
    this.listeners.add(fn);
    return () => { this.listeners.delete(fn); };
  }

  onLog(fn: (events: Log[]) => void): () => void {
    this.logListeners.add(fn);
    return () => { this.logListeners.delete(fn); };
  }

  // --- the socket ------------------------------------------------------------------------------

  private open(): void {
    if (this.stopped || this.fatal) return;
    // one socket at a time: a retry still waiting (a start() got here first) must not open another
    if (this.retryTimer) { clearTimeout(this.retryTimer); this.retryTimer = null; }
    this.setStatus('connecting');
    const ws = new WebSocket(this.url);
    this.ws = ws;
    ws.onopen = () => {
      if (this.ws !== ws) return;
      this.retry = 0;
      this.heard = Date.now();
      this.write({ t: 'hello', v: PROTOCOL, room: this.room, ...(this.token ? { token: this.token } : {}) });
      this.ping();
      if (this.pinger) clearInterval(this.pinger);
      this.pinger = setInterval(() => this.ping(), PING_MS);
    };
    ws.onmessage = (ev) => {
      if (this.ws !== ws) return;   // a socket we let go of can still deliver what was on its way
      this.heard = Date.now();
      let m: ServerMsg;
      try { m = JSON.parse(String(ev.data)); } catch { return; }
      this.receive(m);
    };
    ws.onclose = () => this.dropped(ws);
  }

  /** The socket is gone (closed, or silent too long): show it, and try again soon. */
  private dropped(ws: WebSocket): void {
    if (this.ws !== ws) return;
    this.ws = null;
    if (this.pinger) { clearInterval(this.pinger); this.pinger = null; }
    this.setStatus('closed');
    if (this.stopped || this.fatal) return;
    const wait = Math.min(5000, 250 * 2 ** this.retry++);
    this.retryTimer = setTimeout(() => { this.retryTimer = null; this.open(); }, wait);
  }

  /** A dead socket can look open for minutes (a phone waking up): if the room has gone quiet, start over. */
  private check(): void {
    const ws = this.ws;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (Date.now() - this.heard > SILENT_MS) {
      this.dropped(ws);
      try { ws.close(); } catch { /* already gone */ }
      this.retry = 0;
    } else this.ping();
  }

  private receive(m: ServerMsg): void {
    switch (m.t) {
      case 'welcome':
        this.setStatus('open');
        // presses sent while the connection was down: the room answers each id once
        for (const [id, p] of this.pending) this.write({ t: 'intent', id, intent: p.intent });
        this.flushKey();
        return;
      case 'state':
        if (m.version < this.version && this.game) return;
        this.game = m.game;
        this.version = m.version;
        return this.emit();
      case 'patch':
        // only what changed since the version this screen has; a gap means a lost message: start over
        if (!this.game || m.base !== this.version) { this.ws?.close(); return; }
        this.game = applyUpdates(this.game, m.updates, m.now);
        this.version = m.version;
        return this.emit();
      case 'ack': {
        const p = this.pending.get(m.id);
        if (!p) return;
        this.pending.delete(m.id);
        return p.resolve({ ok: m.ok, ...(m.error ? { error: m.error } : {}) });
      }
      case 'log':
        for (const fn of this.logListeners) fn(m.events);
        return;
      case 'pong': {
        const now = Date.now();
        const rtt = now - m.c;
        this.samples = [...this.samples, { rtt, offset: m.s - (m.c + now) / 2 }].slice(-SAMPLES);
        const best = this.samples.reduce((a, b) => (b.rtt < a.rtt ? b : a));
        if (Math.abs(best.offset - this.offset) > 1) { this.offset = Math.round(best.offset); this.emit(); }
        return;
      }
      case 'seat': {
        const done = this.seats.get(m.pid);
        this.seats.delete(m.pid);
        return done?.(m.token);
      }
      case 'critic':
        this.critic = { ready: m.ready, ...(m.backend ? { backend: m.backend } : {}), ...(m.error ? { error: m.error } : {}) };
        return this.emit();
      case 'error':
        if (m.fatal) { this.fatal = m.error; this.emit(); }
        return;
    }
  }

  private ping(): void {
    if (this.ws && Date.now() - this.heard > SILENT_MS) return this.check();
    this.write({ t: 'ping', c: Date.now() });
  }

  private write(msg: ClientMsg): void {
    if (this.ws?.readyState === WebSocket.OPEN && (this.status === 'open' || msg.t !== 'intent')) this.ws.send(JSON.stringify(msg));
  }

  private setStatus(s: Status): void {
    if (this.status !== s) { this.status = s; this.emit(); }
  }

  private emit(): void {
    for (const fn of this.listeners) fn();
  }
}

function defaultUrl(): string {
  return `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;
}
