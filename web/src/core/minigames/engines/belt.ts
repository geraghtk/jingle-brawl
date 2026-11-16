// A moving line, for any game that wants one: things enter at one end, ride along at the belt's
// speed past a row of stations, and fall off the other end. Pure and settled by the clock: the room
// knows where everything is at any moment, and every screen draws the same belt from the same few
// numbers (a position as of a time, and a speed), with no message per frame.
// Nothing here knows what rides the belt (toys, gifts, snowballs).
//
// Positions run from 0 (where things enter) to 1 (the end). Speed is belt lengths a second. Things
// enter a fixed DISTANCE apart, not a fixed time: a faster belt brings them sooner, still evenly spaced.
export const BELT_VERSION = '1';

export interface BeltItem<T> { id: string; pos: number; data: T }
export interface Belt<T> {
  /** Positions are as of this time; since then everything has moved speed x time. */
  asOf: number;
  speed: number;
  /** How far the belt has run since it started, and how far it will have run when the next thing enters. */
  travel: number;
  nextEntry: number;
  /** Things entered so far (the next one's number). */
  n: number;
  items: BeltItem<T>[];
}
/** A stretch of the belt where something can be done to whatever is passing. */
export interface Zone { from: number; to: number }

const EPS = 1e-9;

export const newBelt = <T>(now: number, speed: number): Belt<T> => ({ asOf: now, speed, travel: 0, nextEntry: 0, n: 0, items: [] });

/** A stored belt loses its empty list: put it back. */
export const wholeBelt = <T>(b: Belt<T>): Belt<T> => ({ ...b, items: b.items || [], travel: b.travel || 0, nextEntry: b.nextEntry || 0, n: b.n || 0 });

/** Where something is at `now` (it never shows past the end). */
export const posAt = (b: { asOf: number; speed: number }, pos: number, now: number): number => Math.min(1, pos + (b.speed * Math.max(0, now - b.asOf)) / 1000);

/** The belt moved on to `now` at its speed (nothing enters or leaves: see `next`). */
export function advance<T>(b: Belt<T>, now: number): Belt<T> {
  const d = (b.speed * Math.max(0, now - b.asOf)) / 1000;
  return { ...b, asOf: Math.max(now, b.asOf), travel: b.travel + d, items: b.items.map((it) => ({ ...it, pos: it.pos + d })) };
}

/** Change the speed from the belt's own moment on (advance first). */
export const atSpeed = <T>(b: Belt<T>, speed: number): Belt<T> => ({ ...b, speed });

/** When the next thing happens by itself: something enters, or the leading thing reaches the end. */
export function nextAt<T>(b: Belt<T>): number {
  if (!(b.speed > 0)) return Infinity;
  const lead = Math.max(-Infinity, ...b.items.map((it) => it.pos));
  const toEnd = b.items.length ? Math.max(0, 1 - lead) : Infinity;
  const toEntry = Math.max(0, b.nextEntry - b.travel);
  return b.asOf + (Math.min(toEnd, toEntry) / b.speed) * 1000;
}

export type BeltEvent<T> = { kind: 'off'; at: number; item: BeltItem<T> } | { kind: 'enter'; at: number; item: BeltItem<T> };

/** The one next thing, if it is due by `now`: the belt moved to that moment, with it done. Else the
 *  belt moved to `now` and no event. The caller loops (it may change the speed between events).
 *  `make(n)`: what enters as number n; `gap`: how far apart things enter. */
export function next<T>(b0: Belt<T>, now: number, gap: number, make: (n: number) => { id: string; data: T }): { belt: Belt<T>; event: BeltEvent<T> | null } {
  const t = nextAt(b0);
  if (t > now) return { belt: advance(b0, now), event: null };
  // which it is, decided BEFORE moving (by distance still to go): after the move, rounding can leave the
  // leader a hair short of the end, and the wrong guess here once let a toy enter at every step for ever
  const led = b0.items.reduce<BeltItem<T> | null>((m, it) => (!m || it.pos > m.pos ? it : m), null);
  const toEnd = led ? Math.max(0, 1 - led.pos) : Infinity, toEntry = Math.max(0, b0.nextEntry - b0.travel);
  const b = advance(b0, t);
  if (led && toEnd <= toEntry) {
    // off the end first (the far end is the older thing)
    return { belt: { ...b, items: b.items.filter((it) => it.id !== led.id) }, event: { kind: 'off', at: t, item: { ...led, pos: 1 } } };
  }
  const made = make(b.n);
  const item: BeltItem<T> = { id: made.id, pos: 0, data: made.data };
  // the belt has run exactly to the entry point (not a hair short of it: the next one must be a whole gap on)
  return { belt: { ...b, travel: Math.max(b.travel, b0.nextEntry), items: [...b.items, item], n: b.n + 1, nextEntry: b0.nextEntry + gap }, event: { kind: 'enter', at: t, item } };
}

/** Which zone a position is in (-1: between stations). */
export const zoneOf = (zones: Zone[], pos: number): number => zones.findIndex((z) => pos >= z.from - EPS && pos <= z.to + EPS);

/** `count` stations spread evenly along the belt, each `width` wide. */
export function evenZones(count: number, width: number): Zone[] {
  return Array.from({ length: count }, (_, k) => { const c = (k + 1) / (count + 1); return { from: c - width / 2, to: c + width / 2 }; });
}
