// A two-fighter exchange of punches and blocks, for any game that wants one: pure and settled by the
// clock, so the room decides every hit and every screen draws the same fight from the same numbers.
// Nothing here knows what the fighters are (robots, snowmen, elves).
//
// A punch winds up, then lands. What it does when it lands depends on the other fighter at that moment:
//   guarding            blocked: no damage, and the puncher is stuck recovering (long enough to be hit back)
//   guard just raised   parried: the puncher is stunned (a block timed to the punch is the best answer)
//   winding up a punch  a counter hit: double damage, and their punch is cancelled
//   anything else       a hit
// A guard can't be held for ever: it breaks, and the fighter is open for a moment. So mashing loses to
// blocking, blocking for ever loses to waiting, and waiting loses to punching.
// Versioned: a result says which rules made it.

export interface FightParams {
  version: string;
  health: number;         // hits a fighter can take
  windupMs: number;       // from starting a punch to it landing (what the other side has to read)
  recoverMs: number;      // after a punch that hit, before the puncher can act again
  blockedMs: number;      // after a punch that was blocked (longer than a wind-up: the blocker can hit back)
  parryMs: number;        // a guard raised no longer than this before the punch lands is a parry
  stunMs: number;         // a parried puncher can do nothing for this long
  flinchMs: number;       // a fighter who is hit can do nothing for this long
  guardMaxMs: number;     // a guard held this long breaks
  guardBreakMs: number;   // ...and its fighter is open for this long
  counter: number;        // damage of a hit on a fighter who was winding up
}
export const DEFAULT_FIGHT: FightParams = {
  version: '1', health: 6, windupMs: 420, recoverMs: 300, blockedMs: 620, parryMs: 240, stunMs: 950, flinchMs: 280,
  guardMaxMs: 1600, guardBreakMs: 850, counter: 2,
};

export type Hand = 'L' | 'R';
export type Busy = 'recover' | 'blocked' | 'stunned' | 'flinch' | 'guard_break';
export interface Fighter {
  health: number;
  /** Nothing can be started before this (and why: for the screens). */
  busyUntil: number;
  busy?: Busy;
  punch?: { hand: Hand; at: number; lands: number };
  guard?: { since: number };
}
export type FightEventKind = 'hit' | 'counter' | 'blocked' | 'parried' | 'guard_break';
export interface FightEvent { n: number; at: number; kind: FightEventKind; by: 0 | 1; hand?: Hand; damage?: number }
export interface Fight {
  f: [Fighter, Fighter];
  /** The last few things that happened (the screens play them once each, by `n`). */
  events: FightEvent[];
  n: number;
  /** Per fighter: punches thrown, landed, blocked or parried against them (a tie-break, and the stats). */
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

const other = (i: 0 | 1): 0 | 1 => (i === 0 ? 1 : 0);
const down = (x: Fight): 0 | 1 | null => (x.f[0].health <= 0 ? 0 : x.f[1].health <= 0 ? 1 : null);

/** When the next thing happens by itself (a punch lands, a guard breaks), or null. */
export function nextAt(x: Fight, p: FightParams = DEFAULT_FIGHT): number | null {
  const times = x.f.flatMap((f) => [f.punch?.lands, f.guard ? f.guard.since + p.guardMaxMs : undefined]).filter((t): t is number => t != null);
  return times.length ? Math.min(...times) : null;
}

/** Everything due by `now`, in the order it happened. Stops at a knockout: `ko` is who went down
 *  (both in the same instant: the one with less health left, then fighter 1; a real double is that rare). */
export function settle(fight: Fight, now: number, p: FightParams = DEFAULT_FIGHT): { fight: Fight; ko: 0 | 1 | null } {
  const x = wholeFight(fight);
  const say = (e: Omit<FightEvent, 'n'>) => { x.n += 1; x.events = [...x.events, { ...e, n: x.n }].slice(-KEEP_EVENTS); };
  for (let guard = 0; guard < 64; guard++) {
    if (down(x) != null) break;
    const t = nextAt(x, p);
    if (t == null || t > now) break;
    // punches landing in the same instant are a trade: both are judged on how things stood before either
    const landing = ([0, 1] as const).filter((i) => x.f[i].punch?.lands === t);
    const before = [{ ...x.f[0] }, { ...x.f[1] }];
    for (const i of landing) {
      const j = other(i), me = x.f[i], them = x.f[j], was = before[j];
      const hand = me.punch!.hand;
      me.punch = undefined;
      if (was.guard && was.guard.since <= t) {
        const parried = t - was.guard.since <= p.parryMs;
        me.busyUntil = t + (parried ? p.stunMs : p.blockedMs);
        me.busy = parried ? 'stunned' : 'blocked';
        say({ at: t, kind: parried ? 'parried' : 'blocked', by: i, hand });
      } else {
        const counter = !!was.punch && was.punch.lands > t;            // caught winding up (a trade is two plain hits)
        const damage = counter ? p.counter : 1;
        them.health = Math.max(0, them.health - damage);
        if (counter) them.punch = undefined;
        them.guard = undefined;
        them.busyUntil = Math.max(them.busyUntil, t + p.flinchMs);
        them.busy = 'flinch';
        if (me.busy !== 'flinch' || me.busyUntil <= t) { me.busyUntil = t + p.recoverMs; me.busy = 'recover'; }
        x.landed[i] += 1;
        say({ at: t, kind: counter ? 'counter' : 'hit', by: i, hand, damage });
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

/** Start a punch (settle first; check canPunch). */
export function punch(fight: Fight, i: 0 | 1, hand: Hand, now: number, p: FightParams = DEFAULT_FIGHT): Fight {
  const x = wholeFight(fight);
  x.f[i] = { ...x.f[i], punch: { hand, at: now, lands: now + p.windupMs }, busy: undefined };
  x.thrown[i] += 1;
  return x;
}

/** Raise the guard (check canGuard) or drop it. Dropping is free: a guard is given up, never stuck. */
export function guard(fight: Fight, i: 0 | 1, on: boolean, now: number): Fight {
  const x = wholeFight(fight);
  x.f[i] = { ...x.f[i], guard: on ? { since: now } : undefined, ...(on ? { busy: undefined } : {}) };
  return x;
}
