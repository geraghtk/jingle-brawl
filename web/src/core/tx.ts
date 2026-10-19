// Building one atomic multi-path update, and applying updates in memory (tests, previews).
import type { EventKind, Game, GameEvent, Plan, Updates } from './types';
import { SERVER_TIME } from './types';

const parts = (path: string) => path.split('/').filter(Boolean);
const isObj = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);
const clone = <T>(v: T): T => (v === undefined ? v : JSON.parse(JSON.stringify(v)));

export function getPath(root: unknown, path: string): unknown {
  let cur: unknown = root;
  for (const k of parts(path)) {
    if (!isObj(cur) && !Array.isArray(cur)) return undefined;
    cur = (cur as Record<string, unknown>)[k];
  }
  return cur;
}

/** RTDB semantics: null/undefined deletes; empty objects vanish. */
function setPath(root: Record<string, unknown>, path: string, value: unknown) {
  const ks = parts(path);
  let cur: Record<string, unknown> = root;
  for (const k of ks.slice(0, -1)) {
    if (!isObj(cur[k])) cur[k] = {};
    cur = cur[k] as Record<string, unknown>;
  }
  const last = ks[ks.length - 1];
  if (value === null || value === undefined) delete cur[last];
  else cur[last] = clone(value);
}

function prune(v: unknown): unknown {
  if (!isObj(v)) return v;
  for (const k of Object.keys(v)) {
    const p = prune(v[k]);
    if (p === undefined || (isObj(p) && !Object.keys(p).length)) delete v[k];
    else v[k] = p;
  }
  return v;
}

function resolveServerTime(v: unknown, now: number): unknown {
  if (isObj(v) && v['.sv'] === 'timestamp') return now;
  if (isObj(v)) for (const k of Object.keys(v)) v[k] = resolveServerTime(v[k], now);
  return v;
}

/** Apply a multi-path update to a game in memory (throws on overlapping paths, like RTDB). */
export function applyUpdates(game: Game, updates: Updates, now = Date.now()): Game {
  const keys = Object.keys(updates);
  for (const a of keys) for (const b of keys) {
    if (a !== b && (b + '/').startsWith(a + '/')) throw new Error(`overlapping update paths: ${a} / ${b}`);
  }
  const out = clone(game) as unknown as Record<string, unknown>;
  for (const [path, value] of Object.entries(updates)) setPath(out, path, resolveServerTime(clone(value), now));
  return prune(out) as unknown as Game;
}

/**
 * Accumulates one action's writes. Reads see earlier writes in the same action; the
 * resulting update never has overlapping paths (a write under an already-written object
 * merges into it, a write over written children replaces them).
 */
export class Tx {
  private updates: Updates = {};
  private view: Record<string, unknown>;
  private log: GameEvent[] = [];
  private claimed?: Plan['claim'];

  constructor(readonly game: Game) {
    this.view = clone(game) as unknown as Record<string, unknown>;
  }

  get<T = unknown>(path: string): T {
    return getPath(this.view, path) as T;
  }

  set(path: string, value: unknown): this {
    const p = parts(path).join('/');
    for (const k of Object.keys(this.updates)) {
      if ((k + '/').startsWith(p + '/')) delete this.updates[k]; // replaced by this write
    }
    const ancestor = Object.keys(this.updates).find((k) => (p + '/').startsWith(k + '/') && k !== p);
    if (ancestor) {
      const base = isObj(this.updates[ancestor]) ? (this.updates[ancestor] as Record<string, unknown>) : {};
      setPath(base, p.slice(ancestor.length + 1), value);
      this.updates[ancestor] = base;
    } else {
      this.updates[p] = value === undefined ? null : clone(value);
    }
    setPath(this.view, p, value);
    return this;
  }

  /** chips/points helper: add n to a numeric field. */
  add(path: string, n: number): this {
    return this.set(path, (Number(this.get(path)) || 0) + n);
  }

  event(text: string, kind: EventKind = 'bid'): this {
    this.log.push({ text, kind });
    return this;
  }

  /** The whole plan only applies if this set-if-empty claim wins (first claim in a plan counts). */
  claim(path: string, value: unknown): this {
    this.claimed ??= { path, value };
    return this;
  }

  then?: Plan['then'];

  plan(then?: Plan['then']): Plan {
    const next = then ?? this.then;
    return {
      updates: this.updates, events: this.log,
      ...(next ? { then: next } : {}), ...(this.claimed ? { claim: this.claimed } : {}),
    };
  }
}

export const refuse = (error: string): Plan => ({ updates: {}, events: [], error });
export { SERVER_TIME };
