// Mecha Santa Mayhem (spec game #6). Each team drives one giant wind-up Santa along the same rooftop
// course; first over the line wins, else the one further along at the bell. Mode `captain_teams` only:
// the two duelists captain the teams (dealt by engines/teams.ts) and take a job like everyone else.
// Four jobs, dealt with engines/roles.ts (a small team doubles up, a big one shares, a dropped phone's
// job is open to its team): the left leg, the right leg, the balance, the arms. The robot is
// engines/walker.ts: legs in turn, a lean that grows by itself, walls to smash, gaps to jump with both
// legs, boosts to grab. Settled by the clock: nothing is sent per frame, a screen draws each robot
// where it is now from the times in the view. Both teams walk the same course with the same kicks.
import type { PlayerId } from '../../types';
import type { GameDefinition, GameResult, MatchCtx, Transition } from '../contracts';
import { crewFor, mayWork } from '../engines/roles';
import { assignTeams, checkAssignment } from '../engines/teams';
import type { TeamAssignment } from '../engines/teams';
import {
  blocker, boostNear, DEFAULT_WALK, free, gapAhead, grab, jump, leanAt, newWalker, nextAt, posAt, push, settle as settleWalk, shiftWalker, smash, step,
  WALKER_VERSION, wallAhead, wholeWalker, NO_LEG,
} from '../engines/walker';
import type { Course, Did, Leg, Obstacle, WalkParams, Walker } from '../engines/walker';
import { JOBS } from './view';
import type { MechaView } from './view';

export interface MechaConfig {
  roundMs: number;
  overtimeMs: number;
  /** Party points the game recommends for each player on the winning team (the losers: none). */
  rewardBase: number;
  walk: WalkParams;
  /** A team of two or three has fewer hands for the same four jobs: its robot strides this much further
   *  (a factor by team size; four or more: 1). Set from the simulation so team size does not decide a race. */
  lightFoot: Record<number, number>;
  /** How long the course is and what is on it (the same for both teams). */
  length: number;
  obstacles: Obstacle[];
  /** The teams, when the experience dealt them (duelgame / the lab); else dealt here. */
  teams?: TeamAssignment;
}
export const DEFAULTS: MechaConfig = {
  // (npm run analyze:mecha: an average team of four is over the line in about 50 s, a slow one is still out at the bell)
  roundMs: 90_000, overtimeMs: 15_000, rewardBase: 1, walk: DEFAULT_WALK, lightFoot: { 2: 1.08, 3: 1.02 }, length: 96,
  obstacles: [
    { id: 'b1', kind: 'boost', at: 8 }, { id: 'w1', kind: 'wall', at: 16 }, { id: 'g1', kind: 'gap', at: 28, width: 4 }, { id: 'b2', kind: 'boost', at: 38 },
    { id: 'w2', kind: 'wall', at: 46 }, { id: 'g2', kind: 'gap', at: 58, width: 4 }, { id: 'b3', kind: 'boost', at: 66 }, { id: 'w3', kind: 'wall', at: 74 },
    { id: 'g3', kind: 'gap', at: 84, width: 4 },
  ],
};

export interface TeamState {
  teamId: 'A' | 'B'; captainId: PlayerId; members: PlayerId[];
  /** Who does each job (JOBS order). */
  crew: PlayerId[][];
  /** A job shared by two or more is done in turns: how many times each job has been done (whose go it is). */
  turn: number[];
  /** This robot's own numbers (a small team's strides further: `lightFoot`). */
  walk: WalkParams;
  walker: Walker;
  /** What each member did for the walk (for show: it is not paid). */
  credit: Record<PlayerId, number>;
}
export interface MechaState {
  cfg: MechaConfig;
  stage: 'running' | 'overtime' | 'done';
  startedAt: number; endsAt: number;
  course: Course;
  teams: [TeamState, TeamState];
  winner?: 'A' | 'B';
}

export type MechaAction = { kind: 'step'; leg: Leg } | { kind: 'jump'; leg: Leg } | { kind: 'lean'; dir: -1 | 1 } | { kind: 'smash' } | { kind: 'grab' };
/** Which job an action belongs to. */
const jobOf = (a: MechaAction): number => (a.kind === 'step' || a.kind === 'jump' ? a.leg : a.kind === 'lean' ? 2 : 3);

/** Stored state loses empty lists and maps: put them back before reading. */
function whole(s: MechaState): MechaState {
  return {
    ...s, course: { ...s.course, obstacles: s.course.obstacles || [] },
    teams: s.teams.map((t): TeamState => ({ ...t, members: t.members || [], crew: JOBS.map((_, k) => t.crew?.[k] || []), turn: JOBS.map((_, k) => t.turn?.[k] || 0), walk: t.walk || s.cfg.walk, walker: wholeWalker(t.walker), credit: t.credit || {} })) as [TeamState, TeamState],
  };
}
const teamOf = (s: MechaState, pid: PlayerId) => s.teams.findIndex((t) => t.members.includes(pid));

/** One of job `k`'s crew, in turn (the simulator's: which of two on a leg presses next). */
export const onDuty = (t: Pick<TeamState, 'crew' | 'turn'>, k: number): PlayerId => t.crew[k][(t.turn[k] || 0) % t.crew[k].length];
/** May `who` do job `k`: it is theirs (two who share a job may both press whenever: they have to agree who
 *  does, or the leg goes twice), or its whole crew has dropped out and they are on the team. */
const mayDo = (connected: (p: PlayerId) => boolean, t: TeamState, who: PlayerId, k: number): boolean => mayWork(connected, t.crew, t.members, who, k);
const jobsNow = (connected: (p: PlayerId) => boolean, t: TeamState, who: PlayerId) => JOBS.map((_, k) => k).filter((k) => mayDo(connected, t, who, k));

function teamsFrom(ctx: MatchCtx, c: MechaConfig): TeamAssignment {
  const given = checkAssignment(c.teams, ctx.duelists);
  return given || assignTeams(ctx.duelists!, ctx.participants, ctx.rng);
}

function finish(ctx: MatchCtx, s: MechaState, winner: 'A' | 'B' | null, reason: string): Transition<MechaState> {
  const done: MechaState = { ...s, stage: 'done', ...(winner ? { winner } : {}) };
  const at = Math.min(ctx.now, s.endsAt);
  const [a, b] = s.teams;
  const metrics = {
    distance: { A: Math.round(posAt(a.walker, at) * 100) / 100, B: Math.round(posAt(b.walker, at) * 100) / 100 }, length: s.course.length,
    finishedAt: { A: a.walker.finishedAt, B: b.walker.finishedAt }, slips: { A: a.walker.slips, B: b.walker.slips },
    teams: s.teams.map((t) => ({ teamId: t.teamId, captainId: t.captainId, members: t.members })), walker: WALKER_VERSION,
  };
  const performance = s.teams.flatMap((t) => t.members.map((playerId) => ({ playerId, contribution: Math.round((t.credit[playerId] || 0) * 100) / 100 })));
  const base = { gameId: mecha.id, instanceId: ctx.instanceId, mode: ctx.mode, performance, metrics, scoringPolicyVersion: 'captain_team@2', reason };
  const duelists = ctx.duelists!;
  if (!winner) return { state: done, result: { ...base, status: 'requires_tiebreaker', winners: [], losers: [], duel: { duelists, resolutionReason: reason } }, events: [`🤖 Mecha Santa: ${reason}`] };
  const win = s.teams.find((t) => t.teamId === winner)!, lose = s.teams.find((t) => t.teamId !== winner)!;
  const result: GameResult = {
    ...base, status: 'resolved', winners: win.members, losers: lose.members,
    rewards: win.members.map((playerId) => ({ playerId, points: s.cfg.rewardBase, reason: 'the winning team' })),
    duel: { duelists, winningDuelistId: win.captainId, losingDuelistId: lose.captainId, resolutionReason: reason },
  };
  return { state: done, result, events: [`🤖 Mecha Santa: team ${win.teamId} (${reason})`] };
}

/** What is due next by itself: a foot landing, a robot toppling, a lone jump giving up, the bell. */
function deadlines(s: MechaState) {
  const times = s.teams.map((t) => nextAt(t.walker, t.walk)).filter((t): t is number => t != null);
  const soonest = times.length ? Math.min(...times) : Infinity;
  return [{ id: 'end', at: s.endsAt }, ...(soonest < s.endsAt ? [{ id: 'tick', at: soonest }] : [])];
}

/** Both robots brought up to `now` (never past the bell), and what follows: over the line, the bell, overtime. */
function settle(ctx: MatchCtx, s: MechaState): { state: MechaState; over?: Transition<MechaState> } {
  const now = Math.min(ctx.now, s.endsAt);
  const state: MechaState = { ...s, teams: s.teams.map((t): TeamState => ({ ...t, walker: settleWalk(t.walker, now, s.course, t.walk) })) as [TeamState, TeamState] };
  const [a, b] = state.teams.map((t) => t.walker);
  const fewer = a.slips !== b.slips ? (a.slips < b.slips ? 'A' : 'B') : null;
  if (a.finishedAt || b.finishedAt) {
    if (a.finishedAt && b.finishedAt && a.finishedAt === b.finishedAt) return { state, over: finish(ctx, state, fewer, fewer ? 'over the line together: fewer tumbles' : 'over the line together, tumble for tumble') };
    const first = !b.finishedAt || (a.finishedAt && a.finishedAt < b.finishedAt) ? 'A' : 'B';
    return { state, over: finish(ctx, state, first, 'first over the line') };
  }
  if (ctx.now < s.endsAt) return { state };
  // the bell: further along wins; dead level: a little longer, once; then fewer tumbles; then the host
  const pa = posAt(a, now), pb = posAt(b, now);
  if (Math.abs(pa - pb) > 1e-6) return { state, over: finish(ctx, state, pa > pb ? 'A' : 'B', s.stage === 'overtime' ? 'further along after overtime' : 'further along at the bell') };
  if (s.stage === 'running') return { state: { ...state, stage: 'overtime', endsAt: ctx.now + s.cfg.overtimeMs } };
  return { state, over: finish(ctx, state, fewer, fewer ? 'level after overtime: fewer tumbles' : 'level after overtime, tumble for tumble') };
}

/** One action on a robot that is already settled to `now`. */
function apply(w: Walker, a: MechaAction, now: number, course: Course, p: WalkParams): Did {
  switch (a.kind) {
    case 'step': return step(w, a.leg, now, course, p);
    case 'jump': return jump(w, a.leg, now, course, p);
    case 'lean': return push(w, a.dir, now, p);
    case 'smash': return smash(w, now, course, p);
    case 'grab': return grab(w, now, course, p);
  }
}
/** What a move was worth to the team (read off the walk's newest event). */
function worth(before: Walker, after: Walker): number {
  if (after.n === before.n) return 0;
  const e = after.events[after.events.length - 1];
  return ({ step: 1, stride: 1.5, jump: 2, crouch: 0.5, hit: 0.5, smashed: 1, boost: 2, saved: 2, stumble: -1, fell: -2, whiff: -0.5, bump: -1 } as Record<string, number>)[e.kind] ?? 0;
}

function parse(raw: unknown): MechaAction | null {
  const a = raw as { kind?: unknown; leg?: unknown; dir?: unknown } | null;
  if (a?.kind === 'step' || a?.kind === 'jump') return a.leg === 0 || a.leg === 1 ? { kind: a.kind, leg: a.leg } : null;
  if (a?.kind === 'lean') return a.dir === -1 || a.dir === 1 ? { kind: 'lean', dir: a.dir } : null;
  if (a?.kind === 'smash' || a?.kind === 'grab') return { kind: a.kind };
  return null;
}

export const mecha: GameDefinition<MechaConfig, MechaState, MechaAction> = {
  id: 'mecha_santa', version: '1', label: 'Mecha Santa Mayhem', emoji: '🤖',
  rules: 'The duellists captain two teams, each driving one giant wind-up Santa on the TV. One of you is the left leg, one the right, one keeps it upright, one works the arms. Every part moves whenever its player presses, so call it out: step in turn, smash the walls, jump the gaps together. First robot over the line wins.',
  categories: ['team_based', 'cross_device', 'real_time_action', 'asymmetric_cooperative'],
  supportedModes: ['captain_teams'],
  compatibility: {
    version: 1, environments: ['in_person', 'remote', 'hybrid'], displayRole: 'required_shared_world', topology: ['phone_to_shared_display', 'shared_object_multi_controller'],
    devices: [{ feature: 'touch', required: false, fallback: 'buttons work with a mouse or keyboard' }, { feature: 'haptics', required: false }], players: { min: 2, max: 40 }, modes: ['captain_teams'],
  },

  validateConfig(raw) {
    const r = (raw && typeof raw === 'object' ? raw : {}) as Partial<MechaConfig>;
    const c: MechaConfig = { ...DEFAULTS, ...r, walk: { ...DEFAULT_WALK, ...(r.walk || {}) } };
    const w = c.walk;
    const ok = c.roundMs > 0 && c.overtimeMs > 0 && c.length > 0 && Array.isArray(c.obstacles) && w.stride > 0 && w.stepMs > 0 && w.leanRate > 0 && w.kick >= 0 && w.push > 0
      && w.wallHits >= 1 && w.jumpWindowMs > 0 && w.jumpMs > 0 && w.boost > 0 && Object.values(c.lightFoot || {}).every((f) => f >= 1 && f <= 2)
      && c.obstacles.every((o) => typeof o.id === 'string' && ['wall', 'gap', 'boost'].includes(o.kind) && o.at > 0 && o.at < c.length && (o.kind !== 'gap' || ((o.width || 0) > 0 && o.at + (o.width || 0) < c.length)));
    if (!ok) throw new Error('bad Mecha Santa settings');
    return c;
  },

  // overtime, then fewer tumbles, settle nearly every tie; a dead level goes to the host
  canAlwaysResolve: (mode) => (mode === 'captain_teams' ? { ok: true } : { ok: false, reason: 'captains and teams only' }),

  createInitialState(ctx, c) {
    if (!ctx.duelists) throw new Error('Mecha Santa needs two captains');
    const deal = teamsFrom(ctx, c);
    const course: Course = { length: c.length, seed: Math.floor(ctx.rng() * 0x7fffffff), obstacles: [...c.obstacles].sort((a, b) => a.at - b.at) };
    const state: MechaState = {
      cfg: c, stage: 'running', startedAt: ctx.now, endsAt: ctx.now + c.roundMs, course,
      teams: deal.teams.map((t): TeamState => ({
        teamId: t.teamId as 'A' | 'B', captainId: t.captainId, members: t.memberIds, crew: crewFor(t.memberIds, JOBS.length), turn: JOBS.map(() => 0),
        walk: { ...c.walk, stride: c.walk.stride * (c.lightFoot[t.memberIds.length] || 1) }, walker: newWalker(ctx.now), credit: {},
      })) as [TeamState, TeamState],
    };
    return { state, deadlines: deadlines(state), events: [`🤖 Team ${deal.teams[0].teamId} vs team ${deal.teams[1].teamId}: wind them up!`] };
  },

  validateAction(ctx, raw, who, a0) {
    const s0 = whole(raw);
    if (s0.stage === 'done') return { ok: false, error: 'The race is over' };
    const ti = teamOf(s0, who);
    if (ti < 0) return { ok: false, error: "You're not on a team" };
    const a = parse(a0);
    if (!a) return { ok: false, error: 'Not a robot move' };
    const now = settle(ctx, s0);
    if (now.over) return { ok: false, error: 'The race is over' };
    const t = now.state.teams[ti];
    if (!mayDo(ctx.connected, t, who, jobOf(a))) return { ok: false, error: 'That is not your job' };
    const did = apply(t.walker, a, ctx.now, s0.course, t.walk);
    return did.error ? { ok: false, error: did.error } : { ok: true, action: a };
  },

  reduce(ctx, raw, who, a) {
    const now = settle(ctx, whole(raw));
    if (now.over) return now.over;
    const s = now.state;
    const ti = teamOf(s, who);
    const t = s.teams[ti];
    const did = apply(t.walker, a, ctx.now, s.course, t.walk);
    if (!did.walker) return { state: s, deadlines: deadlines(s) };
    const next: TeamState = { ...t, walker: did.walker, turn: t.turn.map((n, k) => (k === jobOf(a) ? n + 1 : n)), credit: { ...t.credit, [who]: (t.credit[who] || 0) + worth(t.walker, did.walker) } };
    const state: MechaState = { ...s, teams: s.teams.map((x, i) => (i === ti ? next : x)) as [TeamState, TeamState] };
    return { state, deadlines: deadlines(state) };
  },

  onDeadline(ctx, raw, id) {
    const s0 = whole(raw);
    if (s0.stage === 'done') return { state: s0 };
    // the bell is its deadline coming due (the room's clock is the judge of that, not this state's note of it)
    const s = id === 'end' ? { ...s0, endsAt: Math.min(s0.endsAt, ctx.now) } : s0;
    const now = settle(ctx, s);
    if (now.over) return now.over;
    return { state: now.state, deadlines: deadlines(now.state), ...(now.state.stage !== s.stage ? { events: ['🤖 Dead level! A little longer: further along wins'] } : {}) };
  },

  shiftTime(raw, ms) {
    const s = whole(raw);
    return { ...s, startedAt: s.startedAt + ms, endsAt: s.endsAt + ms, teams: s.teams.map((t): TeamState => ({ ...t, walker: shiftWalker(t.walker, ms) })) as [TeamState, TeamState] };
  },

  // a simulated hand does the right thing for each of its jobs (the simulator decides how often and how late)
  offers(ctx, raw, who) {
    const s0 = whole(raw);
    if (s0.stage === 'done') return [];
    const ti = teamOf(s0, who);
    if (ti < 0) return [];
    const now = settle(ctx, s0);
    if (now.over) return [];
    const t = now.state.teams[ti], w = t.walker, p = t.walk, course = s0.course;
    const out: MechaAction[] = [];
    for (const k of jobsNow(ctx.connected, t, who)) {
      if (t.crew[k].includes(who) && onDuty(t, k) !== who && ctx.connected(onDuty(t, k))) continue;   // two on a job: a simulated pair take it in turns
      if (k <= 1 && free(w, ctx.now)) {
        const leg = k as Leg;
        if (gapAhead(w, course, p)) { if (w.crouch?.leg !== leg) out.push({ kind: 'jump', leg }); continue; }
        const b = blocker(w, course, p);
        if (b && b.kind === 'wall' && b.at - w.pos <= 1e-6) continue;                    // up against a wall: the arms' turn
        if (w.last === NO_LEG ? leg === 0 : w.last !== leg) out.push({ kind: 'step', leg });
      }
      if (k === 2 && !(w.busy === 'topple') && ctx.now >= w.pushedAt + p.pushMs) {
        const lean = leanAt(w, ctx.now, p);
        if (Math.abs(lean) > 0.3) out.push({ kind: 'lean', dir: lean > 0 ? -1 : 1 });
      }
      if (k === 3 && ctx.now >= w.busyUntil) {
        if (wallAhead(w, course, p) && ctx.now >= w.hitAt + p.hitMs) out.push({ kind: 'smash' });
        else if (boostNear(w, course, ctx.now, p)) out.push({ kind: 'grab' });
      }
    }
    return out;
  },

  projectForPlayer(raw, pid, now) {
    const s = whole(raw);
    const ti = teamOf(s, pid);
    const mine = ti >= 0 ? JOBS.map((_, k) => k).filter((k) => s.teams[ti].crew[k].includes(pid)) : [];
    return { ...publicView(s, now), myTeam: ti >= 0 ? s.teams[ti].teamId : null, myJobs: mine };
  },
  projectForTv: (raw, now) => publicView(whole(raw), now),
  projectForHost: (raw, now) => publicView(whole(raw), now),
};

/** Both robots as every screen draws them: each as it stands at `now`, with the times of what it is
 *  in the middle of (a screen works out where it is this instant), the course, who has which job. */
function publicView(s: MechaState, now: number): MechaView {
  const at = Math.min(now, s.endsAt);
  return {
    stage: s.stage, startedAt: s.startedAt, endsAt: s.endsAt, winner: s.winner ?? null, course: s.course,
    teams: s.teams.map((t) => ({
      teamId: t.teamId, captainId: t.captainId, members: t.members, crew: t.crew, walk: t.walk,
      walker: s.stage === 'done' ? t.walker : settleWalk(t.walker, at, s.course, t.walk),
    })),
  };
}
