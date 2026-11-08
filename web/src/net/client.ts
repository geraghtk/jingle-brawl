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
  private pinger: ReturnType<typeof setInterval> | null = null;
  private stopped = false;

  constructor(readonly room: string, private token: string | undefined, private url = defaultUrl()) {}

  start(): this {
    this.open();
    return this;
  }

  stop(): void {
    this.stopped = true;
    if (this.pinger) clearInterval(this.pinger);
    this.ws?.close();
    for (const p of this.pending.values()) p.resolve({ ok: false, error: 'left the room' });
    this.pending.clear();
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
    this.setStatus('connecting');
    const ws = new WebSocket(this.url);
    this.ws = ws;
    ws.onopen = () => {
      this.retry = 0;
      this.write({ t: 'hello', v: PROTOCOL, room: this.room, ...(this.token ? { token: this.token } : {}) });
      this.ping();
      if (this.pinger) clearInterval(this.pinger);
      this.pinger = setInterval(() => this.ping(), PING_MS);
    };
    ws.onmessage = (ev) => {
      let m: ServerMsg;
      try { m = JSON.parse(String(ev.data)); } catch { return; }
      this.receive(m);
    };
    ws.onclose = () => {
      if (this.ws !== ws) return;
      this.ws = null;
      if (this.pinger) { clearInterval(this.pinger); this.pinger = null; }
      this.setStatus('closed');
      if (this.stopped || this.fatal) return;
      const wait = Math.min(5000, 250 * 2 ** this.retry++);
      setTimeout(() => this.open(), wait);
    };
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
      case 'critic':
        this.critic = { ready: m.ready, ...(m.backend ? { backend: m.backend } : {}), ...(m.error ? { error: m.error } : {}) };
        return this.emit();
      case 'error':
        if (m.fatal) { this.fatal = m.error; this.emit(); }
        return;
    }
  }

  private ping(): void {
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
