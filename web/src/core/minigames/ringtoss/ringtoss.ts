// Reindeer Ring Toss (spec, game #2). The TV is the arena: a reindeer whose antler tips are pegs.
// Everyone swipes rings from their phone; the server works out where each lands (engines/: the
// swipe's launch, a deterministic flight, peg scoring) and the ring stays on the TV in its
// thrower's colour. A few practice throws first (not scored), then each player's rings against
// the clock.
//   direct_duel / party_with_duelists  the two duelists are compared by score (everyone else's
//       throws count for the table, not the duel). Level: each duelist throws one more ring, up
//       to a few times; still level: whoever's ring came closest to the golden peg.
//   free_for_all  placements by score, no duel.
import type { PlayerId } from '../../types';
import type { GameDefinition, GameResult, MatchCtx, Transition } from '../contracts';
import { GESTURE_VERSION, launchFrom, launchFromControls, parsePath, pathFor } from '../engines/gesture';
import type { Launch } from '../engines/gesture';
import { DEFAULT_FLIGHT, landing } from '../engines/projectile';
import type { FlightParams } from '../engines/projectile';
import { score, zoneDistance } from '../engines/targets';
import type { Peg } from '../engines/targets';
import { headToHead } from '../scoring';

/** The reindeer: two antlers (the far prong 1, the tall prong 3) and its party hat between them (5). */
export const PEGS: Peg[] = [
  // under the antler tips and the hat's tip of the reindeer (web/public/art/ringtoss/, tools/art/reindeer.py
  // variant g seed 4, placed by RingTossStage's DEER), which stands at depth 1.65: the far prongs, the
  // tall ones, the hat. Forgiving (playtest, 2026-11-12: "pretty hard to get a hit"): about 2 degrees of
  // aim and +-0.11 to 0.17 of distance (the 5-point hat smallest, the 1-point prongs biggest); where a far and a tall prong's zones overlap, the tall one counts.
  { id: 'left-outer', x: -0.3263, z: 1.65, rx: 0.07, rz: 0.17, points: 1 },
  { id: 'left-inner', x: -0.2461, z: 1.65, rx: 0.05, rz: 0.14, points: 3 },
  { id: 'golden', x: 0, z: 1.65, rx: 0.045, rz: 0.11, points: 5 },
  { id: 'right-inner', x: 0.2461, z: 1.65, rx: 0.05, rz: 0.14, points: 3 },
  { id: 'right-outer', x: 0.3263, z: 1.65, rx: 0.07, rz: 0.17, points: 1 },
];
const GOLDEN = PEGS.find((p) => p.id === 'golden')!;

export interface RingConfig {
  practiceThrows: number; practiceMs: number;
  rings: number; roundMs: number;
  tiebreakRounds: number; tiebreakMs: number;
  cooldownMs: number;                     // between one player's throws
  flight: FlightParams;
}
export const DEFAULTS: RingConfig = {
  practiceThrows: 3, practiceMs: 20_000, rings: 8, roundMs: 45_000, tiebreakRounds: 3, tiebreakMs: 15_000, cooldownMs: 600, flight: DEFAULT_FLIGHT,
};

export type Stage = 'practice' | 'throwing' | 'tiebreak' | 'done';
export interface Throw {
  n: number; pid: PlayerId; at: number; stage: Exclude<Stage, 'done'>;
  angle: number; power: number;           // the launch the server made of the swipe
  x: number; z: number; ms: number;       // where and when it lands
  pegId: string | null; points: number;   // points only count outside practice
  off: number;                            // from the golden peg's centre, in its radii (the last tiebreak)
}

export interface RingState {
  cfg: RingConfig;
  stage: Stage;
  stageEndsAt: number;
  pool: PlayerId[];                        // who throws in this stage
  left: Record<PlayerId, number>;          // rings left in this stage
  scores: Record<PlayerId, number>;
  throws: Throw[];
  lastAt: Record<PlayerId, number>;
  tiebreakRound: number;
  n: number;
}

export type RingAction = { kind: 'throw'; launch: Launch; via: 'swipe' | 'controls' };

const each = (ids: PlayerId[], v: number) => Object.fromEntries(ids.map((id) => [id, v]));

/** Stored state loses empty objects and lists: put them back before reading. */
const whole = (s: RingState): RingState => ({ ...s, pool: s.pool || [], left: s.left || {}, scores: s.scores || {}, throws: s.throws || [], lastAt: s.lastAt || {} });

function stageTo(ctx: MatchCtx, s: RingState, stage: Stage, pool: PlayerId[], rings: number, ms: number, extra: Partial<RingState> = {}): Transition<RingState> {
  const state: RingState = { ...s, ...extra, stage, pool, left: each(pool, rings), stageEndsAt: ctx.now + ms };
  return { state, deadlines: [{ id: 'stage', at: state.stageEndsAt }] };
}

/** Whoever came closest to the golden peg with a scored ring (the last word on a tie). */
function closest(s: RingState, pid: PlayerId): number {
  return Math.min(Infinity, ...s.throws.filter((t) => t.pid === pid && t.stage !== 'practice').map((t) => t.off));
}

function placements(s: RingState, ids: PlayerId[]) {
  const sorted = [...ids].sort((a, b) => (s.scores[b] || 0) - (s.scores[a] || 0));
  return sorted.map((playerId) => ({ playerId, rank: 1 + sorted.filter((q) => (s.scores[q] || 0) > (s.scores[playerId] || 0)).length }));
}

function finish(ctx: MatchCtx, s: RingState, reason: string): Transition<RingState> {
  const done: RingState = { ...s, stage: 'done', pool: [], left: {} };
  const metrics = { scores: s.scores, throws: s.throws.filter((t) => t.stage !== 'practice').length, physics: s.cfg.flight.version, gesture: GESTURE_VERSION };
  const performance = ctx.participants.map((playerId) => ({ playerId, score: s.scores[playerId] || 0 }));
  let result: GameResult;
  if (ctx.duelists) {
    const [a, b] = ctx.duelists;
    const h2h = headToHead.result(ringToss.id, ctx.instanceId, ctx.mode, { scores: s.scores, reason }, ctx.duelists);
    if (h2h.status === 'resolved') result = h2h;
    else {
      // level after the extra rings: the ring closest to the golden peg
      const ca = closest(s, a), cb = closest(s, b);
      if (ca === cb) result = { ...h2h, reason: 'level, and no ring closer to the golden peg' };
      else {
        const [w, l] = ca < cb ? [a, b] : [b, a];
        result = { ...h2h, status: 'resolved', winners: [w], losers: [l], reason: 'level on points: closest ring to the golden peg',
          duel: { duelists: ctx.duelists, winningDuelistId: w, losingDuelistId: l, resolutionReason: 'closest ring to the golden peg' } };
      }
    }
  } else {
    const pl = placements(s, ctx.participants);
    result = { gameId: ringToss.id, instanceId: ctx.instanceId, status: 'resolved', mode: ctx.mode, reason,
      winners: pl.filter((p) => p.rank === 1).map((p) => p.playerId), losers: [], placements: pl, scoringPolicyVersion: 'ring_points@1' };
  }
  return { state: done, result: { ...result, performance, placements: placements(s, ctx.participants), metrics: { ...result.metrics, ...metrics } }, events: [`🦌 Ring Toss: ${reason}`] };
}

/** The stage is over (everyone threw, or time ran out): practice → throwing → (tiebreak…) → done. */
function advance(ctx: MatchCtx, s: RingState): Transition<RingState> {
  const c = s.cfg;
  if (s.stage === 'practice') return stageTo(ctx, s, 'throwing', [...ctx.participants], c.rings, c.roundMs);
  const d = ctx.duelists;
  const level = d && (s.scores[d[0]] || 0) === (s.scores[d[1]] || 0);
  if (level && s.tiebreakRound < c.tiebreakRounds) {
    return { ...stageTo(ctx, s, 'tiebreak', [...d!], 1, c.tiebreakMs, { tiebreakRound: s.tiebreakRound + 1 }), events: ['🦌 Level! One more ring each for the duelists'] };
  }
  return finish(ctx, s, s.stage === 'tiebreak' ? 'after the extra rings' : 'all rings thrown');
}

export const ringToss: GameDefinition<RingConfig, RingState, RingAction> = {
  id: 'reindeer_ring_toss', version: '1', label: 'Reindeer Ring Toss', emoji: '🦌',
  rules: 'Swipe rings from your phone onto the reindeer on the TV: antler tips 1 and 3, the golden peg 5. A few practice throws first.',
  categories: ['real_time_action', 'physical_coordination', 'cross_device'],
  supportedModes: ['direct_duel', 'party_with_duelists', 'free_for_all'],
  compatibility: {
    version: 1, environments: ['in_person', 'remote', 'hybrid'], displayRole: 'required_shared_world', topology: ['phone_to_shared_display'],
    devices: [{ feature: 'touch', required: false, fallback: 'aim and hold-to-charge buttons' }, { feature: 'pointer', required: false }],
    players: { min: 2, max: 30 }, modes: ['direct_duel', 'party_with_duelists', 'free_for_all'],
  },

  validateConfig(raw) {
    const r = (raw && typeof raw === 'object' ? raw : {}) as Partial<RingConfig>;
    const c: RingConfig = { ...DEFAULTS, ...r, flight: { ...DEFAULT_FLIGHT, ...(r.flight || {}) } };
    if (!(c.rings >= 1 && c.roundMs > 0 && c.practiceThrows >= 0 && c.tiebreakRounds >= 0 && c.tiebreakMs > 0)) throw new Error('bad Ring Toss settings');
    return c;
  },

  // extra rings, then the closest ring, settle almost any tie; a dead level (nobody threw) goes to the host
  canAlwaysResolve: (mode) => (mode === 'free_for_all' ? { ok: false, reason: 'a free-for-all has no duel winner' } : { ok: true }),

  createInitialState(ctx, c) {
    const base: RingState = { cfg: c, stage: 'practice', stageEndsAt: 0, pool: [], left: {}, scores: each(ctx.participants, 0), throws: [], lastAt: {}, tiebreakRound: 0, n: 0 };
    return c.practiceThrows > 0 ? stageTo(ctx, base, 'practice', [...ctx.participants], c.practiceThrows, c.practiceMs)
      : stageTo(ctx, base, 'throwing', [...ctx.participants], c.rings, c.roundMs);
  },

  validateAction(ctx, s, who, raw) {
    s = whole(s);
    const a = raw as { kind?: unknown; path?: unknown; aim?: unknown; power?: unknown } | null;
    if (s.stage === 'done') return { ok: false, error: 'All rings are thrown' };
    if (!s.pool.includes(who)) return { ok: false, error: s.stage === 'tiebreak' ? 'Just the duelists now' : "You're not throwing" };
    if (!((s.left[who] || 0) > 0)) return { ok: false, error: s.stage === 'practice' ? 'Practice over: wait for the start' : 'No rings left' };
    if (ctx.now < (s.lastAt[who] || 0) + s.cfg.cooldownMs) return { ok: false, error: 'Next ring coming…' };
    if (a?.kind !== 'throw') return { ok: false, error: 'Not a throw' };
    const path = a.path !== undefined ? parsePath(a.path) : null;
    const launch = path ? launchFrom(path) : a.path === undefined ? launchFromControls(a) : null;
    if (!launch) return { ok: false, error: 'Flick up to throw' };
    return { ok: true, action: { kind: 'throw', launch, via: path ? 'swipe' : 'controls' } };
  },

  reduce(ctx, s, who, a) {
    s = whole(s);
    const land = landing(a.launch, s.cfg.flight);
    const hit = score(PEGS, land.x, land.z);
    const scored = s.stage !== 'practice';
    const t: Throw = {
      n: s.n + 1, pid: who, at: ctx.now, stage: s.stage as Throw['stage'], angle: a.launch.angle, power: a.launch.power,
      x: land.x, z: land.z, ms: land.ms, pegId: hit.pegId, points: scored ? hit.points : 0, off: zoneDistance(GOLDEN, land.x, land.z),
    };
    const next: RingState = {
      ...s, n: t.n, throws: [...s.throws, t], lastAt: { ...s.lastAt, [who]: ctx.now }, left: { ...s.left, [who]: (s.left[who] || 0) - 1 },
      scores: scored ? { ...s.scores, [who]: (s.scores[who] || 0) + t.points } : s.scores,
    };
    // everyone in this stage has thrown: move on now (after the last ring lands)
    if (next.pool.every((p) => !((next.left[p] || 0) > 0))) {
      return { state: next, deadlines: [{ id: 'stage', at: ctx.now + land.ms + 300 }] };
    }
    return { state: next };
  },

  onDeadline(ctx, s, id) {
    s = whole(s);
    if (id !== 'stage' || s.stage === 'done') return { state: s };
    return advance(ctx, s);
  },

  offers(ctx, s, who) {
    s = whole(s);
    if (s.stage === 'done' || !s.pool.includes(who) || !((s.left[who] || 0) > 0) || ctx.now < (s.lastAt[who] || 0) + s.cfg.cooldownMs) return [];
    // a simulated hand: aims at a peg, a little off (the simulator's skill decides how much)
    const peg = PEGS[Math.floor(ctx.rng() * PEGS.length)];
    const range = Math.hypot(peg.x, peg.z);
    const launch: Launch = { angle: Math.atan2(peg.x, peg.z) + (ctx.rng() - 0.5) * 0.06, power: (range - s.cfg.flight.baseRange) / s.cfg.flight.rangePerPower + (ctx.rng() - 0.5) * 0.12 };
    return [{ kind: 'throw', launch, via: 'swipe', path: pathFor(launch) } as RingAction];
  },

  projectForPlayer: (raw, pid, now) => { const s = whole(raw); return { ...publicView(s, now), me: pid, myLeft: s.left[pid] || 0, myScore: s.scores[pid] || 0, nextAt: (s.lastAt[pid] || 0) + s.cfg.cooldownMs }; },
  projectForTv: (s, now) => publicView(whole(s), now),
  projectForHost: (s, now) => publicView(whole(s), now),
};

/** Nothing about a ring toss is secret: every throw is on the TV. (The recent ones: older rings stay as dots.) */
function publicView(s: RingState, _now: number) {
  return {
    stage: s.stage, stageEndsAt: s.stageEndsAt, pool: s.pool, left: s.left, scores: s.scores, tiebreakRound: s.tiebreakRound,
    rings: s.cfg.rings, practiceThrows: s.cfg.practiceThrows, flight: s.cfg.flight, pegs: PEGS,
    throws: s.throws.slice(-120).map(({ n, pid, at, stage, angle, power, x, z, ms, pegId, points }) => ({ n, pid, at, stage, angle, power, x, z, ms, pegId, points })),
  };
}
