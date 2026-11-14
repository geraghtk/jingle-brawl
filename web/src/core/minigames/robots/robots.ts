// Nutcracker Knockout (game #4). Two wind-up toy robots on the TV, one for each duelist, driven from
// their phones: punch left, punch right, hold to block. The room is the referee (engines/fight.ts):
// a punch winds up and lands a moment later, a block stops it and leaves the puncher open, a block
// timed to the punch stuns them, a punch into a wind-up counts double. Every hit knocks a part off
// (PARTS, in order); the last hit takes the head, and that is the round. Best of three.
//   direct_duel  the two duelists fight; everyone else watches the TV.
// A round that runs out of time goes to the robot with more parts left; level: sudden death (the next
// hit takes the head). Still level: most punches landed over the match, then the host decides.
import type { PlayerId } from '../../types';
import type { GameDefinition, GameResult, MatchCtx, Transition } from '../contracts';
import { canGuard, canPunch, DEFAULT_FIGHT, guard, newFight, nextAt, punch, settle, wholeFight } from '../engines/fight';
import type { Fight, FightParams, Hand } from '../engines/fight';
import { headToHead } from '../scoring';
import type { RobotView } from './view';

/** What a robot loses, hit by hit: the last is always the head. As many as a robot has health. */
export const PARTS = ['hat', 'left_shoulder', 'right_shoulder', 'chest_door', 'key', 'head'] as const;
export type Part = (typeof PARTS)[number];
/** The parts a robot with `health` left has lost, first to last. */
export const lost = (health: number, max: number = PARTS.length): Part[] => PARTS.slice(0, Math.max(0, Math.min(PARTS.length, max - health)));

export interface RobotConfig {
  roundsToWin: number;
  introMs: number;          // "Round 1... Fight!"
  roundMs: number;
  suddenMs: number;         // sudden death, once
  koMs: number;             // the head flies: before the next round
  fight: FightParams;
}
export const DEFAULTS: RobotConfig = { roundsToWin: 2, introMs: 3500, roundMs: 45_000, suddenMs: 20_000, koMs: 4000, fight: DEFAULT_FIGHT };

export type Stage = 'intro' | 'fighting' | 'ko' | 'done';
export interface RobotState {
  cfg: RobotConfig;
  /** The two duelists: fighter 0 (left of the TV) and fighter 1. */
  order: [PlayerId, PlayerId];
  stage: Stage;
  stageEndsAt: number;      // intro and ko: when it moves on; fighting: when the round's time is up
  round: number;
  wins: Record<PlayerId, number>;
  fight: Fight;
  sudden: boolean;
  /** How the last round ended (the TV's knockout, the phones' "you lost your head"). */
  last?: { round: number; winner: PlayerId; loser: PlayerId; how: 'knockout' | 'time' };
  /** Punches landed over the whole match (the last tie-break). */
  landed: Record<PlayerId, number>;
}

export type RobotAction = { kind: 'punch'; hand: Hand } | { kind: 'guard'; on: boolean };

/** Stored state loses empty maps and lists: put them back before reading. */
const whole = (s: RobotState): RobotState => ({ ...s, wins: s.wins || {}, landed: s.landed || {}, fight: wholeFight(s.fight) });
const seat = (s: RobotState, pid: PlayerId): 0 | 1 | null => (s.order[0] === pid ? 0 : s.order[1] === pid ? 1 : null);

/** The deadlines of a fight in progress: the next thing that happens by itself, and the bell. */
function fightDeadlines(s: RobotState) {
  const t = nextAt(s.fight, s.cfg.fight);
  return [{ id: 'tick', at: Math.min(t ?? Infinity, s.stageEndsAt) }];
}

function startRound(ctx: MatchCtx, s: RobotState, round: number): Transition<RobotState> {
  const state: RobotState = { ...s, stage: 'intro', round, sudden: false, fight: newFight(s.cfg.fight), stageEndsAt: ctx.now + s.cfg.introMs };
  return { state, deadlines: [{ id: 'stage', at: state.stageEndsAt }] };
}

function finish(ctx: MatchCtx, s: RobotState, reason: string): Transition<RobotState> {
  const done: RobotState = { ...s, stage: 'done' };
  const duelists = ctx.duelists || s.order;
  const metrics = { rounds: s.round, wins: s.wins, landed: s.landed, fight: s.cfg.fight.version };
  let result: GameResult = headToHead.result(robots.id, ctx.instanceId, ctx.mode, { scores: s.wins, reason }, duelists);
  if (result.status !== 'resolved') {
    // level on rounds (it ran out of rounds without a winner): the one who landed more punches
    const byLanded = headToHead.result(robots.id, ctx.instanceId, ctx.mode, { scores: s.landed, reason: 'level on rounds: more punches landed' }, duelists);
    if (byLanded.status === 'resolved') result = byLanded;
  }
  return { state: done, result: { ...result, metrics: { ...result.metrics, ...metrics } }, events: [`🥊 Nutcracker Knockout: ${reason}`] };
}

/** A round is over: count it, then the next round or the match. */
function roundTo(ctx: MatchCtx, s: RobotState, winnerSeat: 0 | 1, how: 'knockout' | 'time'): Transition<RobotState> {
  const winner = s.order[winnerSeat], loser = s.order[winnerSeat === 0 ? 1 : 0];
  const wins = { ...s.wins, [winner]: (s.wins[winner] || 0) + 1 };
  const landed = { [s.order[0]]: (s.landed[s.order[0]] || 0) + s.fight.landed[0], [s.order[1]]: (s.landed[s.order[1]] || 0) + s.fight.landed[1] };
  const state: RobotState = { ...s, wins, landed, stage: 'ko', stageEndsAt: ctx.now + s.cfg.koMs, last: { round: s.round, winner, loser, how } };
  const name = how === 'knockout' ? 'a knockout' : 'on time';
  return { state, deadlines: [{ id: 'stage', at: state.stageEndsAt }], events: [`🥊 Round ${s.round}: ${name}`] };
}

/** Bring a fight up to `now`: punches land, guards break, a head comes off, the bell rings. */
function tick(ctx: MatchCtx, s: RobotState): Transition<RobotState> {
  const out = settle(s.fight, Math.min(ctx.now, s.stageEndsAt), s.cfg.fight);
  const next: RobotState = { ...s, fight: out.fight };
  if (out.ko != null) return roundTo(ctx, next, out.ko === 0 ? 1 : 0, 'knockout');
  if (ctx.now < s.stageEndsAt) return { state: next, deadlines: fightDeadlines(next) };
  // the bell: more parts left wins the round
  const [a, b] = next.fight.f;
  if (a.health !== b.health) return roundTo(ctx, next, a.health > b.health ? 0 : 1, 'time');
  if (!s.sudden) {
    // level: sudden death, one hit takes the head
    const fight = wholeFight(next.fight);
    fight.f[0].health = 1; fight.f[1].health = 1;
    const state: RobotState = { ...next, fight, sudden: true, stageEndsAt: ctx.now + s.cfg.suddenMs };
    return { state, deadlines: fightDeadlines(state), events: ['🥊 Level at the bell: sudden death!'] };
  }
  // nobody landed one in sudden death either: the match is decided on what there is
  const landed = { [s.order[0]]: (s.landed[s.order[0]] || 0) + next.fight.landed[0], [s.order[1]]: (s.landed[s.order[1]] || 0) + next.fight.landed[1] };
  return finish(ctx, { ...next, landed }, 'nobody could finish it');
}

export const robots: GameDefinition<RobotConfig, RobotState, RobotAction> = {
  id: 'nutcracker_knockout', version: '1', label: 'Nutcracker Knockout', emoji: '🥊',
  rules: 'The two duellists each drive a wind-up robot on the TV from their phone: punch left, punch right, hold to block. A block stops a punch and leaves the puncher open; every hit knocks a part off, and the last one takes the head. Best of three rounds.',
  categories: ['real_time_action', 'physical_coordination', 'cross_device'],
  supportedModes: ['direct_duel'],
  compatibility: {
    version: 1, environments: ['in_person', 'remote', 'hybrid'], displayRole: 'required_shared_world', topology: ['phone_to_shared_display'],
    devices: [{ feature: 'touch', required: false, fallback: 'the buttons work with a mouse or the keyboard' }],
    players: { min: 2, max: 2 }, modes: ['direct_duel'],
  },

  validateConfig(raw) {
    const r = (raw && typeof raw === 'object' ? raw : {}) as Partial<RobotConfig>;
    const c: RobotConfig = { ...DEFAULTS, ...r, fight: { ...DEFAULT_FIGHT, ...(r.fight || {}) } };
    // a robot has as many hits in it as it has parts to lose
    c.fight = { ...c.fight, health: Math.max(1, Math.min(PARTS.length, Math.floor(c.fight.health) || PARTS.length)) };
    if (!(c.roundsToWin >= 1 && c.roundsToWin <= 5 && c.roundMs > 0 && c.introMs >= 0 && c.koMs >= 0 && c.suddenMs > 0 && c.fight.windupMs > 0)) throw new Error('bad Nutcracker Knockout settings');
    return c;
  },

  // rounds, then sudden death, then punches landed settle almost anything; two robots that never throw a punch go to the host
  canAlwaysResolve: (mode) => (mode === 'direct_duel' ? { ok: true } : { ok: false, reason: 'two duelists only' }),

  createInitialState(ctx, c) {
    const order = (ctx.duelists || [ctx.participants[0], ctx.participants[1]]) as [PlayerId, PlayerId];
    const base: RobotState = { cfg: c, order, stage: 'intro', stageEndsAt: 0, round: 1, wins: { [order[0]]: 0, [order[1]]: 0 }, fight: newFight(c.fight), sudden: false, landed: {} };
    return startRound(ctx, base, 1);
  },

  validateAction(ctx, raw, who, action) {
    const s = whole(raw);
    const i = seat(s, who);
    if (i == null) return { ok: false, error: 'Only the two duellists fight' };
    if (s.stage !== 'fighting') return { ok: false, error: s.stage === 'done' ? 'The fight is over' : 'Wait for the bell' };
    const a = action as { kind?: unknown; hand?: unknown; on?: unknown } | null;
    // as things stand now (a punch that has landed by now has landed)
    const now = settle(s.fight, Math.min(ctx.now, s.stageEndsAt), s.cfg.fight);
    if (now.ko != null || ctx.now >= s.stageEndsAt) return { ok: false, error: 'Wait for the bell' };
    if (a?.kind === 'punch') {
      if (a.hand !== 'L' && a.hand !== 'R') return { ok: false, error: 'Left or right?' };
      if (!canPunch(now.fight, i, ctx.now)) return { ok: false, error: now.fight.f[i].guard ? 'Drop your guard to punch' : 'Not yet' };
      return { ok: true, action: { kind: 'punch', hand: a.hand } };
    }
    if (a?.kind === 'guard') {
      if (typeof a.on !== 'boolean') return { ok: false, error: 'Guard up or down?' };
      if (a.on && !canGuard(now.fight, i, ctx.now)) return { ok: false, error: 'Not yet' };
      return { ok: true, action: { kind: 'guard', on: a.on } };
    }
    return { ok: false, error: 'Punch or block' };
  },

  reduce(ctx, raw, who, a) {
    const s = whole(raw);
    const i = seat(s, who)!;
    const t = tick(ctx, s);
    if (t.state.stage !== 'fighting' || t.result) return t;            // the punch that just landed ended the round
    const fight = a.kind === 'punch' ? punch(t.state.fight, i, a.hand, ctx.now, s.cfg.fight) : guard(t.state.fight, i, a.on, ctx.now);
    const state: RobotState = { ...t.state, fight };
    return { state, deadlines: fightDeadlines(state), events: t.events };
  },

  onDeadline(ctx, raw, id) {
    const s = whole(raw);
    if (s.stage === 'done') return { state: s };
    if (s.stage === 'intro' && id === 'stage') {
      const state: RobotState = { ...s, stage: 'fighting', stageEndsAt: ctx.now + s.cfg.roundMs };
      return { state, deadlines: fightDeadlines(state) };
    }
    if (s.stage === 'ko' && id === 'stage') {
      const best = Math.max(...s.order.map((p) => s.wins[p] || 0));
      return best >= s.cfg.roundsToWin ? finish(ctx, s, `${best} round${best === 1 ? '' : 's'} to ${Math.min(...s.order.map((p) => s.wins[p] || 0))}`) : startRound(ctx, s, s.round + 1);
    }
    if (s.stage === 'fighting') return tick(ctx, s);
    return { state: s };
  },

  offers(ctx, raw, who) {
    const s = whole(raw);
    const i = seat(s, who);
    if (i == null || s.stage !== 'fighting' || ctx.now >= s.stageEndsAt) return [];
    const now = settle(s.fight, ctx.now, s.cfg.fight);
    if (now.ko != null) return [];
    const me = now.fight.f[i], them = now.fight.f[i === 0 ? 1 : 0];
    // a simulated hand: blocks a punch it sees coming (not always), drops a guard that has done its job, else swings
    if (me.guard) return them.punch ? [] : [{ kind: 'guard', on: false }];
    if (!canPunch(now.fight, i, ctx.now)) return [];
    if (them.punch && ctx.rng() < 0.6) return [{ kind: 'guard', on: true }];
    return [{ kind: 'punch', hand: ctx.rng() < 0.5 ? 'L' : 'R' }];
  },

  projectForPlayer: (raw, pid, now) => { const s = whole(raw); return { ...publicView(s, now), me: pid, mySeat: seat(s, pid) }; },
  projectForTv: (s, now) => publicView(whole(s), now),
  projectForHost: (s, now) => publicView(whole(s), now),
};

/** Nothing in a fight is secret: both robots are on the TV. The fight is shown as it stands at `now`
 *  (a punch due by now has landed), with the times a screen needs to draw what is in the air. */
function publicView(s: RobotState, now: number): RobotView {
  const fight = s.stage === 'fighting' ? settle(s.fight, Math.min(now, s.stageEndsAt), s.cfg.fight).fight : s.fight;
  return {
    stage: s.stage, stageEndsAt: s.stageEndsAt, round: s.round, roundsToWin: s.cfg.roundsToWin, order: s.order, wins: s.wins, sudden: s.sudden,
    maxHealth: s.cfg.fight.health, windupMs: s.cfg.fight.windupMs, guardMaxMs: s.cfg.fight.guardMaxMs,
    fighters: fight.f.map((f) => ({
      health: f.health, lost: lost(f.health, s.cfg.fight.health), busyUntil: f.busyUntil || 0,
      ...(f.busy && f.busyUntil > now ? { busy: f.busy } : {}), ...(f.punch ? { punch: f.punch } : {}), ...(f.guard ? { guard: f.guard } : {}),
    })) as RobotView['fighters'],
    events: fight.events, ...(s.last ? { last: s.last } : {}),
  };
}
