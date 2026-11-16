// A two-fighter exchange of punches, blocks and dodges, for any game that wants one: pure and settled
// by the clock, so the room decides every blow and every screen draws the same fight from the same
// numbers. Nothing here knows what the fighters are (robots, snowmen, elves).
//
// Two punches and two defences, each good against something:
//   jab       quick, light. A guard stops it and leaves the jabber open.
//   heavy     slow, hits hard, and smashes through a guard (the guard breaks; some of it gets through).
//   guard     held: stops jabs. Held too long, or hit by a heavy, it breaks and its fighter is open.
//   dodge     a tap: for a moment every punch misses, and the puncher is left badly open. Then the
//             dodger is exposed for a moment (it can't be spammed: a jab after it lands).
// And a punch that lands on a fighter who is winding up their own is a counter: extra damage, and
// their punch never lands (so a jab beats a heavy's long wind-up).
// Versioned: a result says which rules made it. v2 (2026-11-16): jab/heavy and the dodge; the timed
// "parry" guard of v1 is gone (the dodge is the timed defence now).

export type PunchKind = 'jab' | 'heavy';
export interface FightParams {
  version: string;
  health: number;         // damage a fighter can take
  jabMs: number;          // a jab: from starting it to it landing (what the other side has to read)
  heavyMs: number;        // a heavy: the same, much longer
  jabDamage: number;
  heavyDamage: number;
  smashDamage: number;    // what a heavy still does through a guard
  counter: number;        // extra damage for landing on a fighter who was winding up
  recoverMs: number;      // after a punch that hit, before the puncher can act again
  blockedMs: number;      // after a jab that was blocked (longer than a jab: the blocker can hit back)
  whiffMs: number;        // after a punch that was dodged (long: a dodge timed well earns a heavy, timed late a jab)
  flinchMs: number;       // a fighter who is hit can do nothing for this long
  dodgeMs: number;        // how long a dodge makes punches miss
  dodgeRestMs: number;    // ...and how long its fighter is exposed afterwards
  guardMaxMs: number;     // a guard held this long breaks
  guardBreakMs: number;   // ...and its fighter is open for this long
}
export const DEFAULT_FIGHT: FightParams = {
  version: '2', health: 12, jabMs: 340, heavyMs: 720, jabDamage: 1, heavyDamage: 2, smashDamage: 1, counter: 1,
  recoverMs: 300, blockedMs: 620, whiffMs: 1200, flinchMs: 280, dodgeMs: 380, dodgeRestMs: 400, guardMaxMs: 1600, guardBreakMs: 850,
};

export type Busy = 'recover' | 'blocked' | 'whiff' | 'flinch' | 'guard_break' | 'dodge';
export interface Fighter {
  health: number;
  /** Nothing can be started before this (and why: for the screens). */
  busyUntil: number;
  busy?: Busy;
  punch?: { kind: PunchKind; at: number; lands: number };
  guard?: { since: number };
  /** Punches landing from `at` up to `until` miss. */
  dodge?: { at: number; until: number };
}
export type FightEventKind = 'hit' | 'counter' | 'blocked' | 'smash' | 'dodged' | 'guard_break';
export interface FightEvent { n: number; at: number; kind: FightEventKind; by: 0 | 1; punch?: PunchKind; damage?: number }
export interface Fight {
  f: [Fighter, Fighter];
  /** The last few things that happened (the screens play them once each, by `n`). */
  events: FightEvent[];
  n: number;
  /** Per fighter: punches thrown and landed (a tie-break, and the stats). */
  thrown: [number, number];
  landed: [number, number];
}
const KEEP_EVENTS = 10;

export const newFight = (p: FightParams = DEFAULT_FIGHT): Fight => ({
  f: [{ health: p.health, busyUntil: 0 }, { health: p.health, busyUntil: 0 }], events: [], n: 0, thrown: [0, 0], landed: [0, 0],
});

/** A stored fight loses its empty lists: put them back. */
export function wholeFight(x: Fight): Fight {
  return { ...x, f: [{ ...x.f[0] }, { ...x.f[1] }] as [Fighter, Fighter], events: x.events || [], thrown: x.thrown || [0, 0], landed: x.landed || [0, 0], n: x.n || 0 };
}

/** Every time in the fight moved on by `ms` (the game was paused that long). */
export function shiftFight(fight: Fight, ms: number): Fight {
  const x = wholeFight(fight);
  x.f = x.f.map((f): Fighter => ({
    ...f, busyUntil: f.busyUntil ? f.busyUntil + ms : f.busyUntil,
    ...(f.punch ? { punch: { ...f.punch, at: f.punch.at + ms, lands: f.punch.lands + ms } } : {}),
    ...(f.guard ? { guard: { since: f.guard.since + ms } } : {}),
    ...(f.dodge ? { dodge: { at: f.dodge.at + ms, until: f.dodge.until + ms } } : {}),
  })) as [Fighter, Fighter];
  x.events = x.events.map((e) => ({ ...e, at: e.at + ms }));
  return x;
}

const other = (i: 0 | 1): 0 | 1 => (i === 0 ? 1 : 0);
const down = (x: Fight): 0 | 1 | null => (x.f[0].health <= 0 ? 0 : x.f[1].health <= 0 ? 1 : null);
export const windup = (kind: PunchKind, p: FightParams = DEFAULT_FIGHT) => (kind === 'heavy' ? p.heavyMs : p.jabMs);

/** When the next thing happens by itself (a punch lands, a guard breaks), or null. */
export function nextAt(x: Fight, p: FightParams = DEFAULT_FIGHT): number | null {
  const times = x.f.flatMap((f) => [f.punch?.lands, f.guard ? f.guard.since + p.guardMaxMs : undefined]).filter((t): t is number => t != null);
  return times.length ? Math.min(...times) : null;
}

/** Everything due by `now`, in the order it happened. Stops at a knockout: `ko` is who went down
 *  (both in the same instant: fighter 1; a real double is that rare). */
export function settle(fight: Fight, now: number, p: FightParams = DEFAULT_FIGHT): { fight: Fight; ko: 0 | 1 | null } {
  const x = wholeFight(fight);
  const say = (e: Omit<FightEvent, 'n'>) => { x.n += 1; x.events = [...x.events, { ...e, n: x.n }].slice(-KEEP_EVENTS); };
  for (let turns = 0; turns < 64; turns++) {
    if (down(x) != null) break;
    const t = nextAt(x, p);
    if (t == null || t > now) break;
    // punches landing in the same instant are a trade: both are judged on how things stood before either
    const landing = ([0, 1] as const).filter((i) => x.f[i].punch?.lands === t);
    const before = [{ ...x.f[0] }, { ...x.f[1] }];
    for (const i of landing) {
      const j = other(i), me = x.f[i], them = x.f[j], was = before[j];
      const kind = me.punch!.kind;
      me.punch = undefined;
      const hurt = (damage: number) => { them.health = Math.max(0, them.health - damage); x.landed[i] += 1; };
      if (was.dodge && was.dodge.at <= t && t < was.dodge.until) {
        me.busyUntil = t + p.whiffMs;
        me.busy = 'whiff';
        say({ at: t, kind: 'dodged', by: i, punch: kind });
      } else if (was.guard && was.guard.since <= t && kind === 'jab') {
        me.busyUntil = t + p.blockedMs;
        me.busy = 'blocked';
        say({ at: t, kind: 'blocked', by: i, punch: kind });
      } else if (was.guard && was.guard.since <= t) {
        // a heavy goes through a guard: it breaks, and some of the blow lands
        hurt(p.smashDamage);
        them.guard = undefined;
        them.busyUntil = Math.max(them.busyUntil, t + p.guardBreakMs);
        them.busy = 'guard_break';
        me.busyUntil = t + p.recoverMs;
        me.busy = 'recover';
        say({ at: t, kind: 'smash', by: i, punch: kind, damage: p.smashDamage });
      } else {
        const counter = !!was.punch && was.punch.lands > t;            // caught winding up (a trade is two plain hits)
        const damage = (kind === 'heavy' ? p.heavyDamage : p.jabDamage) + (counter ? p.counter : 0);
        hurt(damage);
        if (counter) them.punch = undefined;
        them.guard = undefined;
        them.busyUntil = Math.max(them.busyUntil, t + p.flinchMs);
        them.busy = 'flinch';
        if (me.busy !== 'flinch' || me.busyUntil <= t) { me.busyUntil = t + p.recoverMs; me.busy = 'recover'; }
        say({ at: t, kind: counter ? 'counter' : 'hit', by: i, punch: kind, damage });
      }
    }
    for (const i of [0, 1] as const) {
      const f = x.f[i];
      if (f.guard && f.guard.since + p.guardMaxMs === t && !landing.length) {
        f.guard = undefined;
        f.busyUntil = t + p.guardBreakMs;
        f.busy = 'guard_break';
        say({ at: t, kind: 'guard_break', by: i });
      }
    }
  }
  const ko = down(x);
  const both = x.f[0].health <= 0 && x.f[1].health <= 0;
  return { fight: x, ko: both ? 1 : ko };
}

/** May fighter `i` start something at `now` (not busy, not already punching)? */
export const free = (x: Fight, i: 0 | 1, now: number): boolean => now >= (x.f[i].busyUntil || 0) && !x.f[i].punch;
export const canPunch = (x: Fight, i: 0 | 1, now: number): boolean => free(x, i, now) && !x.f[i].guard;
export const canGuard = (x: Fight, i: 0 | 1, now: number): boolean => free(x, i, now) && !x.f[i].guard;
export const canDodge = canPunch;

/** Start a punch (settle first; check canPunch). */
export function punch(fight: Fight, i: 0 | 1, kind: PunchKind, now: number, p: FightParams = DEFAULT_FIGHT): Fight {
  const x = wholeFight(fight);
  x.f[i] = { ...x.f[i], punch: { kind, at: now, lands: now + windup(kind, p) }, busy: undefined, dodge: undefined };
  x.thrown[i] += 1;
  return x;
}

/** Raise the guard (check canGuard) or drop it. Dropping is free: a guard is given up, never stuck. */
export function guard(fight: Fight, i: 0 | 1, on: boolean, now: number): Fight {
  const x = wholeFight(fight);
  x.f[i] = { ...x.f[i], guard: on ? { since: now } : undefined, ...(on ? { busy: undefined, dodge: undefined } : {}) };
  return x;
}

/** Dodge (check canDodge): punches miss for a moment, then the fighter is exposed until they can act again. */
export function dodge(fight: Fight, i: 0 | 1, now: number, p: FightParams = DEFAULT_FIGHT): Fight {
  const x = wholeFight(fight);
  x.f[i] = { ...x.f[i], dodge: { at: now, until: now + p.dodgeMs }, busyUntil: now + p.dodgeMs + p.dodgeRestMs, busy: 'dodge' };
  return x;
}
