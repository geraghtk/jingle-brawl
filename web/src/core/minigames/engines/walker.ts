// One big clumsy thing walked by several people at once (theme-free: a robot, a pantomime horse, a
// giant snowman): two legs that must take turns, a body that leans further the longer it is left
// alone, and arms for what is in the way. Pure and settled by the clock, like the belt and the fight:
// the state is a few numbers and times, the room decides everything, and every screen draws the same
// walk from the same numbers (where it is now: `posAt`, how far it leans now: `leanAt`).
//
//   legs      a step each, in turn. The same leg twice is a stumble. A step started just as the last
//             one lands is "in stride" and goes further.
//   balance   every step kicks the lean a little; left alone the lean grows by itself (an upside-down
//             pendulum: lean * e^(kt)), so the moment it topples is known exactly. A push sets it back.
//   a wall    stops the walk until the arms have hit it enough.
//   a gap     needs a jump: both legs within a moment of each other. A step into it falls in.
//   a boost   is there to be grabbed on the way past.
//   a climb   a tall thing with rungs: the arms grab hold, then the legs go up it a rung each, in turn. The
//             same leg twice slips down a rung; leaning right over drops it off.
//   a rope    over a gap too wide to jump: the arms grab it, it swings by the clock, and both legs jump to
//             let go. On the forward part of the swing it lands across; otherwise it drops into the gap.
//             A push from the balance while it swings makes the swing bigger (the good moment longer).
//   ice       a stretch where the lean grows faster.
// Nothing is refused for being the wrong moment (Kevin, 2026-11-22: the fun is having to call it out): a
// part moves whenever it is told to. A step while a foot is still in the air trips it; a step at a wall
// bonks into it; the arms swung at nothing, or a jump where there is no gap, throw it off balance or
// waste the time. Every mistake costs time and mends itself: there is no dead end.
// Versioned: a result says which rules made it.

export const WALKER_VERSION = '2';
export type Leg = 0 | 1;
export const NO_LEG = -1;

export interface WalkParams {
  version: string;
  stride: number;         // how far a step goes (course units)
  stepMs: number;         // a step: from starting it to its foot landing
  strideMs: number;       // a step started within this of the last landing is in stride...
  strideBonus: number;    // ...and goes this much further (a factor)
  stumbleMs: number;      // the same leg twice
  kick: number;           // how much a step kicks the lean (towards its own side), on average
  leanRate: number;       // the lean grows by e^(leanRate * seconds)
  push: number;           // what one push of the balance sets it back
  pushMs: number;         // between pushes
  toppleMs: number;       // getting up after falling over (the lean reaching 1)
  wallHits: number;       // hits to break a wall
  hitMs: number;          // between hits
  reach: number;          // how close the arms need to be (to a wall, to a boost)
  jumpWindowMs: number;   // both legs must press within this
  jumpMs: number;         // in the air
  misjumpMs: number;      // one leg jumped alone
  climbMs: number;        // out of a gap after stepping into it
  boost: number;          // how far a boost carries
  boostMs: number;
  graceMs: number;        // a step this close to the last foot landing is only early, not a trip (it is ignored)
  hop: number;            // how far a jump goes where there is no gap (strides): slower than walking
  rungs: number;          // how many rungs a climb has (unless it says)
  rungMs: number;         // one rung
  slipMs: number;         // a slip on the rungs, or letting go of them
  pullMs: number;         // the arms pulling it in to what they grabbed
  vaultMs: number;        // over the top of a climb and down the far side
  vault: number;          // how far beyond the climb it comes down
  swingMs: number;        // a rope: there and back
  swingGood: number;      // let go with the swing (-1 back to 1 forward) at least this far forward...
  swingPumped: number;    // ...or this far, once the balance has given it a push
  iceLean: number;        // on ice the lean grows this many times as fast
}
export const DEFAULT_WALK: WalkParams = {
  version: WALKER_VERSION, stride: 2, stepMs: 520, strideMs: 320, strideBonus: 1.3, stumbleMs: 1100, kick: 0.11, leanRate: 0.55, push: 0.3, pushMs: 220,
  toppleMs: 2500, wallHits: 4, hitMs: 260, reach: 2.4, jumpWindowMs: 900, jumpMs: 800, misjumpMs: 700, climbMs: 2600, boost: 7, boostMs: 900, graceMs: 130, hop: 0.5,
  rungs: 4, rungMs: 380, slipMs: 700, pullMs: 300, vaultMs: 700, vault: 3.5, swingMs: 1800, swingGood: 0.5, swingPumped: 0.1, iceLean: 2,
};

export type ObstacleKind = 'wall' | 'gap' | 'boost' | 'climb' | 'rope' | 'ice';
/** `width`: a gap's, a rope's gap's, a stretch of ice's. `rungs`: a climb's (else the params'). */
export interface Obstacle { id: string; kind: ObstacleKind; at: number; width?: number; rungs?: number }
export interface Course { length: number; seed: number; obstacles: Obstacle[] }

export type MoveKind = 'step' | 'jump' | 'boost' | 'pull' | 'rung' | 'vault';
export type Busy = 'stumble' | 'topple' | 'misjump' | 'climb';
export type WalkEventKind = 'step' | 'stride' | 'stumble' | 'topple' | 'bump' | 'hit' | 'smashed' | 'crouch' | 'jump' | 'misjump' | 'fell' | 'boost' | 'saved' | 'finish' | 'whiff'
  | 'grabbed' | 'rung' | 'slip' | 'topped' | 'dropped' | 'pump';
export interface WalkEvent { n: number; at: number; kind: WalkEventKind; leg?: number; id?: string }

export interface Walker {
  /** Where it stands (while a move is in the air: where it left the ground). */
  pos: number;
  move?: { kind: MoveKind; leg: number; at: number; lands: number; from: number; to: number };
  /** The leg that stepped last (NO_LEG: either may start), and when its foot landed. */
  last: number;
  landedAt: number;
  /** The lean as of `leanAsOf`: -1 (flat on its left side) to 1. */
  lean: number;
  leanAsOf: number;
  pushedAt: number;
  busyUntil: number;
  busy?: Busy;
  /** Which way it last fell over (-1 or 1; 0: it has not), for the screens. */
  fell: number;
  /** A jump begun by one leg, waiting for the other. */
  crouch?: { leg: number; at: number };
  /** Holding on to a climb (which rung it is on) or a rope (swinging since `at`; `pumped`: the balance gave it a push). */
  grip?: { id: string; kind: 'climb' | 'rope'; at: number; rung: number; pumped?: boolean };
  /** Standing on ice: the lean grows faster (as of `leanAsOf`). */
  slick?: boolean;
  /** Hits landed on each wall; the boosts taken. */
  hits: Record<string, number>;
  hitAt: number;
  taken: string[];
  /** When it crossed the line (0: not yet). */
  finishedAt: number;
  steps: number;
  events: WalkEvent[];
  n: number;
  /** For the stats and the tie-break: stumbles, topples, falls. */
  slips: number;
}
const KEEP_EVENTS = 10;
const EPS = 1e-6;

export const newWalker = (now: number): Walker => ({
  pos: 0, last: NO_LEG, landedAt: 0, lean: 0, leanAsOf: now, pushedAt: 0, busyUntil: 0, fell: 0, hits: {}, hitAt: 0, taken: [], finishedAt: 0, steps: 0, events: [], n: 0, slips: 0,
});

/** A stored walker loses its empty lists and maps: put them back. */
export function wholeWalker(w: Walker): Walker {
  return { ...w, hits: { ...(w.hits || {}) }, taken: [...(w.taken || [])], events: w.events || [], n: w.n || 0, steps: w.steps || 0, slips: w.slips || 0,
    finishedAt: w.finishedAt || 0, busyUntil: w.busyUntil || 0, landedAt: w.landedAt || 0, pushedAt: w.pushedAt || 0, hitAt: w.hitAt || 0, lean: w.lean || 0, pos: w.pos || 0, fell: w.fell || 0,
    ...(w.grip ? { grip: { ...w.grip, rung: w.grip.rung || 0 } } : {}) };
}

/** Every time in the walk moved on by `ms` (the game was paused that long). */
export function shiftWalker(walker: Walker, ms: number): Walker {
  const w = wholeWalker(walker);
  const later = (t: number) => (t ? t + ms : t);
  return {
    ...w, leanAsOf: w.leanAsOf + ms, landedAt: later(w.landedAt), pushedAt: later(w.pushedAt), busyUntil: later(w.busyUntil), hitAt: later(w.hitAt), finishedAt: later(w.finishedAt),
    ...(w.move ? { move: { ...w.move, at: w.move.at + ms, lands: w.move.lands + ms } } : {}),
    ...(w.crouch ? { crouch: { ...w.crouch, at: w.crouch.at + ms } } : {}),
    ...(w.grip ? { grip: { ...w.grip, at: w.grip.at + ms } } : {}),
    events: w.events.map((e) => ({ ...e, at: e.at + ms })),
  };
}

/** A number from 0 to 1 that is the same for the same seed and count (so both teams get the same kicks). */
export function chance(seed: number, i: number): number {
  let x = (seed ^ Math.imul(i + 1, 0x9e3779b1)) >>> 0;
  x = Math.imul(x ^ (x >>> 16), 0x85ebca6b) >>> 0;
  x = Math.imul(x ^ (x >>> 13), 0xc2b2ae35) >>> 0;
  return ((x ^ (x >>> 16)) >>> 0) / 4294967296;
}

/** How fast the lean grows where it stands (faster on ice). */
const rateOf = (w: Pick<Walker, 'slick'>, p: Pick<WalkParams, 'leanRate'> & Partial<Pick<WalkParams, 'iceLean'>>) => p.leanRate * (w.slick ? p.iceLean ?? DEFAULT_WALK.iceLean : 1);

/** How far it leans at `now` (it grows by itself; it stops at flat on its side). */
export function leanAt(w: Pick<Walker, 'lean' | 'leanAsOf' | 'slick'>, now: number, p: Pick<WalkParams, 'leanRate'> & Partial<Pick<WalkParams, 'iceLean'>> = DEFAULT_WALK): number {
  const v = (w.lean || 0) * Math.exp(rateOf(w, p) * Math.max(0, now - w.leanAsOf) / 1000);
  return Math.max(-1, Math.min(1, v));
}
/** When it will fall over if nobody pushes it back (null: it is standing straight). */
export function topplesAt(w: Walker, p: WalkParams = DEFAULT_WALK): number | null {
  const a = Math.abs(w.lean || 0);
  if (a < EPS || w.busy === 'topple') return null;
  if (a >= 1) return w.leanAsOf;
  return w.leanAsOf + Math.ceil(Math.log(1 / a) / rateOf(w, p) * 1000);
}

/** Where it is at `now`: a move in the air is part way along. */
export function posAt(w: Pick<Walker, 'pos' | 'move'>, now: number): number {
  const m = w.move;
  if (!m) return w.pos;
  if (now >= m.lands) return m.to;
  const f = Math.max(0, (now - m.at) / Math.max(1, m.lands - m.at));
  return m.from + (m.to - m.from) * f;
}

const cleared = (w: Walker, o: Obstacle, p: WalkParams) => (o.kind === 'wall' ? (w.hits[o.id] || 0) >= p.wallHits
  : o.kind === 'gap' || o.kind === 'rope' ? w.pos >= o.at + (o.width || 0) - EPS : o.kind === 'climb' ? w.taken.includes(o.id) : true);
/** What stops a walk until it is dealt with (a boost and ice do not). */
const inTheWay = (o: Obstacle) => o.kind !== 'boost' && o.kind !== 'ice';
/** Is `pos` on a stretch of ice? */
export const onIce = (pos: number, course: Course) => course.obstacles.some((o) => o.kind === 'ice' && pos >= o.at - EPS && pos < o.at + (o.width || 0) - EPS);
/** How many rungs this climb has. */
export const rungsOf = (o: Pick<Obstacle, 'rungs'> | undefined, p: Pick<WalkParams, 'rungs'> = DEFAULT_WALK) => Math.max(1, Math.round(o?.rungs || p.rungs));
/** Where a rope's swing is at `now`: -1 (right back) to 1 (right forward); it starts at the back. */
export const swingAt = (grip: { at: number }, now: number, p: Pick<WalkParams, 'swingMs'> = DEFAULT_WALK) => -Math.cos((2 * Math.PI * Math.max(0, now - grip.at)) / p.swingMs);
/** The wall, gap, climb or rope that is next in the way (not broken, not crossed), or null. */
export function blocker(w: Walker, course: Course, p: WalkParams = DEFAULT_WALK): Obstacle | null {
  return course.obstacles.filter((o) => inTheWay(o) && !cleared(w, o, p) && o.at >= w.pos - EPS).sort((a, b) => a.at - b.at)[0] || null;
}
/** How far it can walk before something stops it. */
const limit = (w: Walker, course: Course, p: WalkParams) => blocker(w, course, p)?.at ?? Infinity;
/** The gap it is standing at the edge of (close enough to jump), or null. */
export function gapAhead(w: Walker, course: Course, p: WalkParams = DEFAULT_WALK): Obstacle | null {
  const b = blocker(w, course, p);
  return b && b.kind === 'gap' && b.at - w.pos <= p.stride * 0.5 + EPS ? b : null;
}
/** The wall the arms can reach, or null. */
export function wallAhead(w: Walker, course: Course, p: WalkParams = DEFAULT_WALK): Obstacle | null {
  const b = blocker(w, course, p);
  return b && b.kind === 'wall' && b.at - w.pos <= p.reach + EPS ? b : null;
}
/** The climb or rope the arms can reach (and are not yet holding), or null. */
export function gripAhead(w: Walker, course: Course, p: WalkParams = DEFAULT_WALK): Obstacle | null {
  const b = blocker(w, course, p);
  return !w.grip && b && (b.kind === 'climb' || b.kind === 'rope') && b.at - w.pos <= p.reach + EPS ? b : null;
}
/** How far up a climb it is at `now`, 0 (the foot) to 1 (the top); 0 when it is not on one. */
export function climbAt(w: Walker, now: number, course: Course, p: WalkParams = DEFAULT_WALK): number {
  if (w.grip?.kind !== 'climb') return 0;
  const n = rungsOf(course.obstacles.find((o) => o.id === w.grip!.id), p), m = w.move;
  const part = m && m.kind === 'rung' ? Math.max(0, Math.min(1, (now - m.at) / Math.max(1, m.lands - m.at))) : 0;
  return Math.min(1, (w.grip.rung + part) / n);
}
/** What is coming up within `range` (or is being dealt with now): the next thing not yet behind it, of any kind. */
export function comingUp(w: Walker, now: number, course: Course, p: WalkParams = DEFAULT_WALK, range = 12): Obstacle | null {
  if (w.finishedAt) return null;
  if (w.grip) return course.obstacles.find((o) => o.id === w.grip!.id) || null;
  const at = posAt(w, now);
  return course.obstacles.filter((o) => (o.kind === 'boost' ? !w.taken.includes(o.id) && o.at >= at - 0.5 : o.kind === 'ice' ? at < o.at + (o.width || 0) : !cleared(w, o, p) && o.at >= w.pos - EPS) && o.at - at <= range)
    .sort((a, b) => a.at - b.at)[0] || null;
}
/** A boost within reach that has not been taken, or null. */
export function boostNear(w: Walker, course: Course, now: number, p: WalkParams = DEFAULT_WALK): Obstacle | null {
  const at = posAt(w, now);
  return course.obstacles.find((o) => o.kind === 'boost' && !w.taken.includes(o.id) && Math.abs(o.at - at) <= p.reach + EPS) || null;
}

export const moving = (w: Walker, now: number) => !!w.move && now < w.move.lands;
/** Standing and able to start something. */
export const free = (w: Walker, now: number) => !w.finishedAt && now >= w.busyUntil && !moving(w, now);

/** When the next thing happens by itself (a foot lands, it topples, a lone jump gives up), or null. */
export function nextAt(w: Walker, p: WalkParams = DEFAULT_WALK): number | null {
  if (w.finishedAt) return null;
  const times = [w.move?.lands, topplesAt(w, p) ?? undefined, w.crouch ? w.crouch.at + p.jumpWindowMs : undefined].filter((t): t is number => t != null);
  return times.length ? Math.min(...times) : null;
}

function say(w: Walker, e: Omit<WalkEvent, 'n'>) { w.n += 1; w.events = [...w.events, { ...e, n: w.n }].slice(-KEEP_EVENTS); }
/** The lean as of `at`, then kicked by `by`. */
function kickLean(w: Walker, at: number, by: number, p: WalkParams) {
  w.lean = Math.max(-0.999, Math.min(0.999, leanAt(w, at, p) + by));
  w.leanAsOf = at;
}
function stall(w: Walker, at: number, busy: Busy, ms: number) { w.busy = busy; w.busyUntil = at + ms; w.slips += 1; delete w.crouch; }

/** Everything due by `now`, in the order it happened. */
export function settle(walker: Walker, now: number, course: Course, p: WalkParams = DEFAULT_WALK): Walker {
  const w = wholeWalker(walker);
  for (let turns = 0; turns < 200; turns++) {
    const t = nextAt(w, p);
    if (t == null || t > now) break;
    const m = w.move;
    if (m && m.lands === t) {
      // a foot (or the whole thing) lands
      w.pos = m.to;
      delete w.move;
      w.landedAt = t;
      if (m.kind === 'step') { w.last = m.leg; w.steps += 1; kickLean(w, t, (m.leg === 0 ? -1 : 1) * p.kick * (0.4 + 1.2 * chance(course.seed, w.steps)), p); }
      else if (m.kind === 'pull') { w.last = NO_LEG; kickLean(w, t, 0, p); }
      else if (m.kind === 'rung') {
        // a rung higher; the last one and it is over the top
        w.last = m.leg; w.steps += 1;
        kickLean(w, t, (m.leg === 0 ? -1 : 1) * p.kick * (0.4 + 1.2 * chance(course.seed, w.steps)), p);
        if (w.grip) {
          w.grip.rung += 1;
          if (w.grip.rung >= rungsOf(course.obstacles.find((o) => o.id === w.grip!.id), p)) {
            w.taken = [...w.taken, w.grip.id];
            say(w, { at: t, kind: 'topped', id: w.grip.id });
            delete w.grip;
            w.last = NO_LEG;
            w.move = { kind: 'vault', leg: NO_LEG, at: t, lands: t + p.vaultMs, from: w.pos, to: Math.min(course.length, w.pos + p.vault, limit(w, course, p)) };
          }
        }
      }
      else { w.last = NO_LEG; kickLean(w, t, (chance(course.seed, 7000 + w.n) < 0.5 ? -1 : 1) * p.kick * 1.5, p); }
      w.slick = onIce(w.pos, course);
      if (w.pos >= course.length - EPS) { w.finishedAt = t; w.lean = 0; say(w, { at: t, kind: 'finish' }); break; }
      continue;
    }
    if (w.crouch && w.crouch.at + p.jumpWindowMs === t) {
      if (w.grip?.kind === 'rope') {
        // one leg let go of the rope alone: down it goes
        say(w, { at: t, kind: 'fell', id: w.grip.id });
        delete w.grip;
        stall(w, t, 'climb', p.climbMs);
        continue;
      }
      say(w, { at: t, kind: 'misjump', leg: w.crouch.leg });
      stall(w, t, 'misjump', p.misjumpMs);
      continue;
    }
    // it has leaned all the way over
    const mid = w.move ? posAt(w, t) : w.pos;
    const stop = Math.min(mid, limit({ ...w, move: undefined }, course, p));
    delete w.move;
    w.pos = Math.max(w.pos, stop);
    w.fell = w.lean < 0 ? -1 : 1;
    delete w.grip;                                    // (off whatever it was holding on to)
    w.slick = onIce(w.pos, course);
    w.lean = 0; w.leanAsOf = t + p.toppleMs; w.last = NO_LEG;
    say(w, { at: t, kind: 'topple' });
    stall(w, t, 'topple', p.toppleMs);
  }
  if (w.busy && now >= w.busyUntil) delete w.busy;
  return w;
}

export type Did = { walker: Walker; error?: undefined } | { walker?: undefined; error: string };

/** A leg steps (settle to `now` first). The same leg twice stumbles; a step into a gap falls in. */
export function step(walker: Walker, leg: Leg, now: number, course: Course, p: WalkParams = DEFAULT_WALK): Did {
  const w = wholeWalker(walker);
  if (w.finishedAt) return { error: 'Over the line!' };
  if (now < w.busyUntil) return { error: 'Wait: getting up' };
  if (w.grip) {
    if (w.grip.kind === 'rope') return { error: 'Hanging on: jump to let go' };
    // on the rungs: a rung each, in turn; the same leg twice (or a leg before the last is on its rung) slips down one
    const early = moving(w, now);
    if (early && (w.move!.kind !== 'rung' || now >= w.move!.lands - p.graceMs)) return { error: 'Too soon' };
    if (early || w.last === leg) {
      delete w.move;
      w.grip.rung = Math.max(0, w.grip.rung - 1);
      say(w, { at: now, kind: 'slip', leg, id: w.grip.id });
      stall(w, now, 'stumble', p.slipMs);
      w.last = NO_LEG;
      kickLean(w, now, (leg === 0 ? -1 : 1) * p.kick, p);
      return { walker: w };
    }
    w.move = { kind: 'rung', leg, at: now, lands: now + p.rungMs, from: w.pos, to: w.pos };
    say(w, { at: now, kind: 'rung', leg, id: w.grip.id });
    return { walker: w };
  }
  if (moving(w, now)) {
    // a foot (or all of it) is still in the air: a hair early is let go, in a jump or a boost nothing happens,
    // otherwise it trips over its own feet where it is
    if (w.move!.kind !== 'step' || now >= w.move!.lands - p.graceMs) return { error: 'Too soon: wait for the foot to land' };
    kickLean(w, now, 0, p);
    w.pos = Math.min(posAt(w, now), limit({ ...w, move: undefined }, course, p));
    w.slick = onIce(w.pos, course);
    delete w.move;
    say(w, { at: now, kind: 'stumble', leg });
    stall(w, now, 'stumble', p.stumbleMs);
    w.last = NO_LEG;
    kickLean(w, now, (leg === 0 ? -1 : 1) * p.kick, p);
    return { walker: w };
  }
  if (w.last === leg) {
    say(w, { at: now, kind: 'stumble', leg });
    stall(w, now, 'stumble', p.stumbleMs);
    w.last = NO_LEG;
    kickLean(w, now, (leg === 0 ? -1 : 1) * p.kick, p);
    return { walker: w };
  }
  const b = blocker(w, course, p);
  const room = (b?.at ?? Infinity) - w.pos;
  if (b && room <= EPS) {
    if (b.kind === 'wall' || b.kind === 'climb') {
      // straight into the wall (or the foot of a climb nobody has hold of): a bonk, and a moment to shake it off
      say(w, { at: now, kind: 'bump', id: b.id });
      stall(w, now, 'stumble', p.stumbleMs);
      w.last = NO_LEG;
      return { walker: w };
    }
    say(w, { at: now, kind: 'fell', id: b.id });
    stall(w, now, 'climb', p.climbMs);
    w.last = NO_LEG;
    return { walker: w };
  }
  const inStride = w.last !== NO_LEG && now - w.landedAt <= p.strideMs;
  const to = Math.min(w.pos + p.stride * (inStride ? p.strideBonus : 1), b?.at ?? Infinity, course.length);
  w.move = { kind: 'step', leg, at: now, lands: now + p.stepMs, from: w.pos, to };
  delete w.crouch;
  say(w, { at: now, kind: inStride ? 'stride' : 'step', leg });
  if (b && b.kind === 'wall' && to >= b.at - EPS) say(w, { at: now, kind: 'bump', id: b.id });
  return { walker: w };
}

/** A leg jumps: the first crouches, the other within the window sends it over the gap. Where there is no gap it
 *  is a hop on the spot, more or less (never past whatever is in the way). */
export function jump(walker: Walker, leg: Leg, now: number, course: Course, p: WalkParams = DEFAULT_WALK): Did {
  const w = wholeWalker(walker);
  if (!free(w, now)) return { error: 'Not now' };
  if (w.grip?.kind === 'climb') {
    // a jump off the rungs: it lets go and is back at the bottom
    say(w, { at: now, kind: 'dropped', id: w.grip.id });
    delete w.grip;
    stall(w, now, 'stumble', p.slipMs);
    w.last = NO_LEG;
    return { walker: w };
  }
  const gap = gapAhead(w, course, p);
  if (!w.crouch) {
    w.crouch = { leg, at: now };
    say(w, { at: now, kind: 'crouch', leg });
    return { walker: w };
  }
  if (w.crouch.leg === leg) return { error: 'Waiting for the other leg!' };
  const called = w.crouch.at;
  delete w.crouch;
  if (w.grip?.kind === 'rope') {
    // both legs let go of the rope: forward enough in its swing and it lands across, else it drops into the gap.
    // The moment that counts is the first leg's (when the jump was called): the second only has to follow in time.
    const rope = course.obstacles.find((o) => o.id === w.grip!.id);
    const good = !!rope && swingAt(w.grip, called, p) >= (w.grip.pumped ? p.swingPumped : p.swingGood);
    delete w.grip;
    if (!good || !rope) {
      say(w, { at: now, kind: 'fell', ...(rope ? { id: rope.id } : {}) });
      stall(w, now, 'climb', p.climbMs);
      w.last = NO_LEG;
      return { walker: w };
    }
    w.move = { kind: 'jump', leg: NO_LEG, at: now, lands: now + p.jumpMs, from: w.pos, to: Math.min(course.length, rope.at + (rope.width || 0) + 0.5) };
    say(w, { at: now, kind: 'jump', id: rope.id });
    return { walker: w };
  }
  const to = gap ? gap.at + (gap.width || 0) + 0.5 : Math.min(w.pos + p.stride * p.hop, limit(w, course, p));
  w.move = { kind: 'jump', leg: NO_LEG, at: now, lands: now + p.jumpMs, from: w.pos, to: Math.min(course.length, to) };
  say(w, { at: now, kind: 'jump', ...(gap ? { id: gap.id } : {}) });
  return { walker: w };
}

/** The balance pushes it left (-1) or right (1). */
export function push(walker: Walker, dir: -1 | 1, now: number, p: WalkParams = DEFAULT_WALK): Did {
  const w = wholeWalker(walker);
  if (w.finishedAt) return { error: 'Over the line!' };
  if (w.busy === 'topple' && now < w.busyUntil) return { error: 'Wait: getting up' };
  if (now < w.pushedAt + p.pushMs) return { error: 'Easy!' };
  if (w.grip?.kind === 'rope') {
    // on the rope there is nothing to balance: a push swings it harder (once is enough)
    w.pushedAt = now;
    if (!w.grip.pumped) { w.grip.pumped = true; say(w, { at: now, kind: 'pump', id: w.grip.id }); }
    return { walker: w };
  }
  const before = leanAt(w, now, p);
  kickLean(w, now, dir * p.push, p);
  w.pushedAt = now;
  if (Math.abs(before) >= 0.6 && Math.abs(w.lean) < Math.abs(before)) say(w, { at: now, kind: 'saved' });
  return { walker: w };
}

/** The arms swung at nothing: the whole thing lurches (one way or the other). */
function whiff(w: Walker, now: number, course: Course, p: WalkParams): Did {
  w.hitAt = now;
  kickLean(w, now, (chance(course.seed, 9000 + w.n) < 0.5 ? -1 : 1) * p.kick * 2, p);
  say(w, { at: now, kind: 'whiff' });
  return { walker: w };
}

/** The arms hit the wall in front (at nothing: a lurch). */
export function smash(walker: Walker, now: number, course: Course, p: WalkParams = DEFAULT_WALK): Did {
  const w = wholeWalker(walker);
  if (w.finishedAt || now < w.busyUntil) return { error: 'Not now' };
  const wall = wallAhead(w, course, p);
  if (now < w.hitAt + p.hitMs) return { error: 'Easy!' };
  if (!wall) return whiff(w, now, course, p);
  w.hits[wall.id] = (w.hits[wall.id] || 0) + 1;
  w.hitAt = now;
  say(w, { at: now, kind: w.hits[wall.id] >= p.wallHits ? 'smashed' : 'hit', id: wall.id });
  return { walker: w };
}

/** The arms grab a boost on the way past: it carries the walk forward (as far as the next thing in the way). */
export function grab(walker: Walker, now: number, course: Course, p: WalkParams = DEFAULT_WALK): Did {
  const w = wholeWalker(walker);
  if (w.finishedAt || now < w.busyUntil) return { error: 'Not now' };
  if (w.grip) return { error: 'Holding on!' };
  const b = boostNear(w, course, now, p);
  const hold = b ? null : gripAhead(w, course, p);
  if (hold) {
    // a climb or a rope in reach: the arms take hold and pull it in (not while a foot is in the air)
    if (!free(w, now)) return { error: 'Not yet' };
    delete w.crouch;
    w.grip = { id: hold.id, kind: hold.kind as 'climb' | 'rope', at: now, rung: 0 };
    if (hold.at - w.pos > EPS) w.move = { kind: 'pull', leg: NO_LEG, at: now, lands: now + p.pullMs, from: w.pos, to: hold.at };
    if (hold.kind === 'rope') { w.lean = 0; w.leanAsOf = now; }       // hanging: there is nothing to balance
    w.hitAt = now; w.last = NO_LEG;
    say(w, { at: now, kind: 'grabbed', id: hold.id });
    return { walker: w };
  }
  if (!b) return now < w.hitAt + p.hitMs ? { error: 'Easy!' } : whiff(w, now, course, p);
  const from = posAt(w, now);
  w.taken = [...w.taken, b.id];
  w.pos = from;
  delete w.crouch;
  const to = Math.min(from + p.boost, limit({ ...w, move: undefined }, course, p), course.length);
  w.move = { kind: 'boost', leg: NO_LEG, at: now, lands: now + p.boostMs, from, to: Math.max(from, to) };
  say(w, { at: now, kind: 'boost', id: b.id });
  return { walker: w };
}
