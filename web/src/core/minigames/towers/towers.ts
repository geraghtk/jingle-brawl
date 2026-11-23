// Elf Tower Wars (spec game #9). Two workshops face each other across three lanes. A team sends toys
// down a lane at the other workshop; the armies pass each other, and only towers, a thrown snowball and
// a raised shield stop them. A workshop at nothing loses; else the healthier one after the last round.
// Mode `captain_teams` only: the two duelists captain the teams (dealt by engines/teams.ts) and play
// like everyone else. There are no jobs: every player has every move and their own energy.
//
// It is played in rounds of two halves (Kevin, 2026-11-23: with everything live at once "I'm always
// looking at my phone screen"; the shape is Legion TD's build phase and battle phase):
//   PLAN    the field stands still. Everyone gets their share of the round's energy and spends it on
//           their phone: towers go up at once, toys are lined up at the workshop door (`queue`). It ends
//           when the time is up, or as soon as everyone has said they are done.
//   MARCH   the queued toys of both teams march out together and the battle runs (engines/lanes.ts, a
//           fixed-step simulation the room advances lazily). Eyes on the TV: a phone has only its three
//           reactions (a snowball, a shield, the two-person avalanche). It ends after `marchMs`, or as
//           soon as the field is empty; whatever is still walking stands where it is until the next march.
// Toys are sturdier each round, so it cannot end in a stand-off. A team earns the same energy however
// many it is (`income`, shared among the phones that are here: a dropped phone's share goes to the rest).
import type { PlayerId } from '../../types';
import type { GameDefinition, GameResult, MatchCtx, Transition } from '../contracts';
import { advance, arm, armed, build, busy, DEFAULT_LANES, LANES_VERSION, newField, PACK_GAP, send, shield, shiftField, wholeField, zap } from '../engines/lanes';
import type { Did, Field, LaneParams, Purse } from '../engines/lanes';
import { assignTeams, checkAssignment } from '../engines/teams';
import type { TeamAssignment } from '../engines/teams';
import { energyOf } from './view';
import type { Queued, TowerView } from './view';

export interface TowerConfig {
  /** How many rounds, and how long each half of one lasts at most. */
  rounds: number; planMs: number; marchMs: number;
  /** A march with nothing left on the field ends early, but not before this (so a quiet round still reads as one). */
  minMarchMs: number;
  /** Once everyone has said they are done planning, the march starts after this. */
  doneMs: number;
  /** How often the room wakes while toys are on the field (the screens glide in between). */
  tickMs: number;
  /** Party points the game recommends for each player on the winning team (the losers: none). */
  rewardBase: number;
  /** Toys get sturdier as the battle goes on (so it cannot end in a stand-off): this much more for each round gone. */
  toughen: number;
  /** Energy a whole team gets at the start of round 1, and how much more each round after, shared among its phones
   *  (so three are no richer than two); a big table gets more, so nobody's share falls under `minShare`. */
  income: number; incomeStep: number; minShare: number;
  /** The bigger team gets this much more for each player it has over the smaller one: a share split more ways buys
   *  less (it cannot reach the dear things), which the simulation showed costs a bigger team games. */
  crowd: number;
  /** How many toys one team may line up in one lane in one round. */
  maxQueue: number;
  rules: LaneParams;
  /** The teams, when the experience dealt them (duelgame / the lab); else dealt here. */
  teams?: TeamAssignment;
}
export const DEFAULTS: TowerConfig = {
  rounds: 5, planMs: 12_000, marchMs: 13_000, minMarchMs: 4000, doneMs: 1200, tickMs: 400, rewardBase: 1, toughen: 0.3,
  income: 16, incomeStep: 5, minShare: 4, crowd: 1, maxQueue: 5,
  // (energy comes in lumps now, a round at a time: nothing trickles in by the clock)
  // (a workshop of 16 and a slightly keener guard: set from `npm run analyze:towers` so most battles are decided on
  // health after five rounds and a few by a workshop falling)
  rules: { ...DEFAULT_LANES, purse: { start: 0, rate: 0, max: 24 }, sendMs: 0, baseHp: 16, guard: 3.25 },
};

export interface TeamState {
  teamId: 'A' | 'B'; captainId: PlayerId; members: PlayerId[];
  purses: Record<PlayerId, Purse>;
  /** Who is here (and so shares the round's energy; see `present`). */
  active: PlayerId[];
  /** Who has said they are done planning this round. */
  done: PlayerId[];
  credit: Record<PlayerId, number>;
}
export interface TowerState {
  cfg: TowerConfig;
  stage: 'running' | 'overtime' | 'done';
  /** Which half of the round it is, which round (from 1), when this half began and when it ends at the latest. */
  phase: 'plan' | 'march';
  round: number;
  phaseAt: number; phaseEndsAt: number;
  startedAt: number;
  field: Field;
  /** Toys lined up at the workshop doors, in the order they were played: they march out together. */
  queue: Queued[];
  teams: [TeamState, TeamState];
  winner?: 'A' | 'B';
}

export type TowerAction =
  | { kind: 'build'; lane: number; slot: number; tower: string } | { kind: 'send'; lane: number; unit: string }
  | { kind: 'zap'; lane: number } | { kind: 'shield'; lane: number } | { kind: 'combo'; lane: number } | { kind: 'done' };
export const costOf = (a: TowerAction, p: LaneParams): number =>
  a.kind === 'build' ? p.towers[a.tower]?.cost ?? Infinity : a.kind === 'send' ? p.units[a.unit]?.cost ?? Infinity : a.kind === 'zap' ? p.zapCost : a.kind === 'shield' ? p.shieldCost : a.kind === 'combo' ? p.comboCost : 0;
/** Which half of a round a move belongs to. */
export const phaseOf = (a: TowerAction): 'plan' | 'march' => (a.kind === 'build' || a.kind === 'send' || a.kind === 'done' ? 'plan' : 'march');

/** Stored state loses empty lists and maps: put them back before reading. */
function whole(s: TowerState): TowerState {
  return {
    ...s, field: wholeField(s.field, s.cfg.rules), queue: s.queue || [],
    teams: s.teams.map((t): TeamState => ({ ...t, members: t.members || [], purses: t.purses || {}, active: t.active || [], done: t.done || [], credit: t.credit || {} })) as [TeamState, TeamState],
  };
}
/** A player's energy (it only changes when they spend it or a round's share arrives). */
const energyNow = (t: TeamState, pid: PlayerId) => energyOf(t, pid);
/** What a whole team gets at the start of round `round` (the same for both, whatever their sizes). */
export const incomeOf = (s: Pick<TowerState, 'cfg' | 'teams'>, round: number) =>
  Math.max(s.cfg.income + s.cfg.incomeStep * (round - 1), s.cfg.minShare * Math.max(...s.teams.map((t) => (t.members || []).length)));
/** Who is here on each team (nobody at all: as if everyone were). */
function present(ctx: MatchCtx, s: TowerState): TowerState {
  let changed = false;
  const teams = s.teams.map((t): TeamState => {
    const here = t.members.filter((m) => ctx.connected(m));
    const active = here.length ? here : t.members;
    if (active.length === t.active.length && active.every((m) => t.active.includes(m))) return t;
    changed = true;
    return { ...t, active };
  });
  return changed ? { ...s, teams: teams as [TeamState, TeamState] } : s;
}
/** The round's energy, shared among whoever is here on each team (on top of what they kept back). */
function paid(s: TowerState, round: number): [TeamState, TeamState] {
  const max = s.cfg.rules.purse.max;
  return s.teams.map((t): TeamState => {
    const small = Math.min(...s.teams.map((x) => x.members.length));
    const share = (incomeOf(s, round) + s.cfg.crowd * Math.max(0, t.members.length - small)) / Math.max(1, t.active.length);
    return { ...t, done: [], purses: Object.fromEntries(t.members.map((m) => [m, { energy: Math.min(max, (t.purses[m]?.energy || 0) + (t.active.includes(m) ? share : 0)), asOf: s.phaseAt }])) };
  }) as [TeamState, TeamState];
}
/** How much sturdier than its kind a toy sent in round `round` is. */
export const toughAt = (s: Pick<TowerState, 'cfg'>, round: number) => 1 + s.cfg.toughen * Math.max(0, round - 1);
const teamOf = (s: TowerState, pid: PlayerId) => s.teams.findIndex((t) => t.members.includes(pid));

function finish(ctx: MatchCtx, s: TowerState, winner: 'A' | 'B' | null, reason: string): Transition<TowerState> {
  const done: TowerState = { ...s, stage: 'done', ...(winner ? { winner } : {}) };
  const [a, b] = s.field.sides;
  const metrics = {
    health: { A: a.hp, B: b.hp }, sent: { A: a.sent, B: b.sent }, stopped: { A: a.stopped, B: b.stopped }, towers: { A: a.towers.length, B: b.towers.length }, rounds: s.round,
    teams: s.teams.map((t) => ({ teamId: t.teamId, captainId: t.captainId, members: t.members })), lanes: LANES_VERSION,
  };
  const performance = s.teams.flatMap((t) => t.members.map((playerId) => ({ playerId, contribution: Math.round((t.credit[playerId] || 0) * 100) / 100 })));
  const base = { gameId: towers.id, instanceId: ctx.instanceId, mode: ctx.mode, performance, metrics, scoringPolicyVersion: 'captain_team@3', reason };
  const duelists = ctx.duelists!;
  if (!winner) return { state: done, result: { ...base, status: 'requires_tiebreaker', winners: [], losers: [], duel: { duelists, resolutionReason: reason } }, events: [`🏰 Elf Tower Wars: ${reason}`] };
  const win = s.teams.find((t) => t.teamId === winner)!, lose = s.teams.find((t) => t.teamId !== winner)!;
  const result: GameResult = {
    ...base, status: 'resolved', winners: win.members, losers: lose.members,
    rewards: win.members.map((playerId) => ({ playerId, points: s.cfg.rewardBase, reason: 'the winning team' })),
    duel: { duelists, winningDuelistId: win.captainId, losingDuelistId: lose.captainId, resolutionReason: reason },
  };
  return { state: done, result, events: [`🏰 Elf Tower Wars: team ${win.teamId} (${reason})`] };
}

/** The end of this half, and a wake-up while anything is walking. */
function deadlines(s: TowerState, now: number) {
  const tick = s.phase === 'march' && busy(s.field) ? Math.max(now, s.field.asOf) + s.cfg.tickMs : Infinity;
  // (an empty field ends a march early: the room looks again when that may be)
  const early = s.phase === 'march' && !busy(s.field) ? Math.max(now, s.phaseAt + s.cfg.minMarchMs) : Infinity;
  const next = Math.min(tick, early);
  return [{ id: 'phase', at: s.phaseEndsAt }, ...(next < s.phaseEndsAt ? [{ id: 'tick', at: next }] : [])];
}

/** The march begins at `at`: the field's clock starts again from here (it stood still while everyone planned) and
 *  the toys lined up at each door walk out one behind the other, in the order they were played. */
function toMarch(s: TowerState, at: number): TowerState {
  const p = s.cfg.rules;
  let f: Field = { ...s.field, asOf: at };
  const behind: Record<string, number> = {};
  for (const q of s.queue) {
    const did = send(f, q.side, q.lane, q.unit, at, p, toughAt(s, s.round));
    if (!did.field) continue;
    const pack = Math.max(1, p.units[q.unit]?.pack || 1), key = `${q.side}|${q.lane}`, back = behind[key] || 0;
    // (the newest `pack` toys are this card's: stood that much further back, inside the workshop, so they come out in turn)
    f = { ...did.field, units: did.field.units.map((u, i) => (i >= did.field.units.length - pack ? { ...u, pos: u.pos - back } : u)) };
    behind[key] = back + pack * PACK_GAP + PACK_GAP * 1.5;
  }
  return { ...s, field: f, queue: [], phase: 'march', phaseAt: at, phaseEndsAt: at + s.cfg.marchMs };
}

/** The battle brought up to `now`, half by half, and what follows: a workshop down, the last round over, overtime. */
function settle(ctx: MatchCtx, s0: TowerState): { state: TowerState; over?: Transition<TowerState>; said?: string[] } {
  let s = present(ctx, s0);
  const said: string[] = [];
  for (let turns = 0; turns < 2 * (s.cfg.rounds + 3); turns++) {
    if (s.phase === 'plan') {
      if (ctx.now < s.phaseEndsAt) break;
      s = toMarch(s, s.phaseEndsAt);
      continue;
    }
    // marching: up to now, or to the end of the march, whichever is first
    const until = Math.min(ctx.now, s.phaseEndsAt);
    s = { ...s, field: advance(s.field, until, s.cfg.rules) };
    const [a, b] = s.field.sides;
    if (a.hp <= 0 || b.hp <= 0) {
      if (a.hp <= 0 && b.hp <= 0) return { state: s, over: finish(ctx, s, null, 'both workshops fell in the same moment') };
      return { state: s, over: finish(ctx, s, a.hp <= 0 ? 'B' : 'A', 'the other workshop fell') };
    }
    const empty = !busy(s.field) && ctx.now >= s.phaseAt + s.cfg.minMarchMs;
    if (ctx.now < s.phaseEndsAt && !empty) break;
    const at = empty ? Math.max(s.phaseAt + s.cfg.minMarchMs, Math.min(ctx.now, s.phaseEndsAt)) : s.phaseEndsAt;
    // the march is over. The last round: the healthier workshop; level: one more round, once; then whoever stopped
    // more toys; then the host
    if (s.round >= s.cfg.rounds) {
      if (a.hp !== b.hp) return { state: s, over: finish(ctx, s, a.hp > b.hp ? 'A' : 'B', s.stage === 'overtime' ? 'the stronger workshop after the extra round' : 'the stronger workshop after the last round') };
      if (s.stage === 'overtime') {
        if (a.stopped !== b.stopped) return { state: s, over: finish(ctx, s, a.stopped > b.stopped ? 'A' : 'B', 'level after the extra round: stopped more toys') };
        return { state: s, over: finish(ctx, s, null, 'level after the extra round') };
      }
      s = { ...s, stage: 'overtime', cfg: { ...s.cfg, rounds: s.cfg.rounds + 1 } };
      said.push('🏰 Level! One more round');
    }
    const round = s.round + 1;
    s = { ...s, round, phase: 'plan', phaseAt: at, phaseEndsAt: at + s.cfg.planMs };
    s = { ...s, teams: paid(s, round) };
  }
  return { state: s, said };
}

function parse(raw: unknown): TowerAction | null {
  const a = raw as Record<string, unknown> | null;
  const int = (v: unknown): v is number => Number.isInteger(v) && (v as number) >= 0 && (v as number) < 16;
  if (a?.kind === 'build') return int(a.lane) && int(a.slot) && typeof a.tower === 'string' ? { kind: 'build', lane: a.lane, slot: a.slot, tower: a.tower } : null;
  if (a?.kind === 'send') return int(a.lane) && typeof a.unit === 'string' ? { kind: 'send', lane: a.lane, unit: a.unit } : null;
  if (a?.kind === 'zap' || a?.kind === 'shield' || a?.kind === 'combo') return int(a.lane) ? { kind: a.kind, lane: a.lane } : null;
  if (a?.kind === 'done') return { kind: 'done' };
  return null;
}
/** How many toys `side` has lined up in `lane`. */
const queued = (s: TowerState, side: number, lane: number) => s.queue.filter((q) => q.side === side && q.lane === lane).length;
/** One move, on a state already brought up to `now`: what it does to the field and the queue, or why not. */
function apply(s: TowerState, side: number, a: TowerAction, now: number, who: PlayerId, alone: boolean): { field: Field; queue: Queued[]; error?: undefined } | { error: string } {
  const p = s.cfg.rules, f = s.field;
  if (phaseOf(a) !== s.phase) return { error: s.phase === 'plan' ? 'Wait for the march' : 'Plan it in the next round' };
  let did: Did;
  switch (a.kind) {
    case 'done': return { field: f, queue: s.queue };
    case 'send':
      if (!p.units[a.unit]) return { error: 'No such toy' };
      if (!(a.lane >= 0 && a.lane < p.lanes)) return { error: 'No such lane' };
      if (queued(s, side, a.lane) >= s.cfg.maxQueue) return { error: 'That door is full' };
      return { field: f, queue: [...s.queue, { side, lane: a.lane, unit: a.unit, by: who }] };
    case 'build': did = build(f, side, a.lane, a.slot, a.tower, now, p); break;
    case 'combo': did = arm(f, side, a.lane, who, now, p, alone); break;
    case 'zap': did = zap(f, side, a.lane, now, p); break;
    case 'shield': did = shield(f, side, a.lane, now, p); break;
  }
  return did.field ? { field: did.field, queue: s.queue } : { error: did.error };
}

export const towers: GameDefinition<TowerConfig, TowerState, TowerAction> = {
  id: 'elf_tower_wars', version: '2', label: 'Elf Tower Wars', emoji: '🏰',
  rules: 'The duellists captain two teams, each with a workshop on the TV. It is played in rounds. First you plan: everyone gets energy and spends it on their phone, on towers (build one again to make it better) and on toys lined up at the workshop door. Then both armies march and everyone watches the TV, with a snowball, a shield and a two-person avalanche in hand. Every toy that reaches the other workshop hurts it. Knock theirs down, or have the stronger one after the last round.',
  categories: ['team_based', 'cross_device', 'real_time_action'],
  supportedModes: ['captain_teams'],
  compatibility: {
    version: 1, environments: ['in_person', 'remote', 'hybrid'], displayRole: 'required_shared_world', topology: ['phone_to_shared_display'],
    devices: [{ feature: 'touch', required: false, fallback: 'buttons work with a mouse or keyboard' }], players: { min: 2, max: 40 }, modes: ['captain_teams'],
  },

  validateConfig(raw) {
    const r = (raw && typeof raw === 'object' ? raw : {}) as Partial<TowerConfig>;
    const c: TowerConfig = { ...DEFAULTS, ...r, rules: { ...DEFAULTS.rules, ...(r.rules || {}) } };
    const p = c.rules;
    const ok = c.rounds >= 1 && c.rounds <= 12 && c.planMs > 0 && c.marchMs > 0 && c.minMarchMs >= 0 && c.minMarchMs <= c.marchMs && c.doneMs >= 0 && c.tickMs >= 100 && p.lanes >= 1 && p.lanes <= 5 && p.stepMs >= 20
      && p.baseHp > 0 && p.slots.length >= 1 && p.slots.every((q) => q > 0 && q < 1)
      && Object.values(p.units).every((u) => u.cost > 0 && u.hp > 0 && u.speed > 0 && u.damage > 0) && Object.values(p.towers).every((t) => t.cost > 0 && t.range > 0 && t.slow > 0 && t.slow <= 1)
      && p.purse.max > 0 && p.maxLevel >= 1 && c.income > 0 && c.incomeStep >= 0 && c.minShare >= 0 && c.crowd >= 0 && c.toughen >= 0 && c.maxQueue >= 1;
    if (!ok) throw new Error('bad Elf Tower Wars settings');
    return c;
  },

  // an extra round, then toys stopped, settle nearly every tie; a dead level goes to the host
  canAlwaysResolve: (mode) => (mode === 'captain_teams' ? { ok: true } : { ok: false, reason: 'captains and teams only' }),

  createInitialState(ctx, c) {
    if (!ctx.duelists) throw new Error('Elf Tower Wars needs two captains');
    const deal = checkAssignment(c.teams, ctx.duelists) || assignTeams(ctx.duelists, ctx.participants, ctx.rng);
    const first: TowerState = {
      cfg: c, stage: 'running', phase: 'plan', round: 1, phaseAt: ctx.now, phaseEndsAt: ctx.now + c.planMs, startedAt: ctx.now, field: newField(ctx.now, c.rules), queue: [],
      teams: deal.teams.map((t): TeamState => {
        const here = t.memberIds.filter((m) => ctx.connected(m));
        return { teamId: t.teamId as 'A' | 'B', captainId: t.captainId, members: t.memberIds, purses: {}, active: here.length ? here : t.memberIds, done: [], credit: {} };
      }) as [TeamState, TeamState],
    };
    const state: TowerState = { ...first, teams: paid(first, 1) };
    return { state, deadlines: deadlines(state, ctx.now), events: [`🏰 Team ${deal.teams[0].teamId} vs team ${deal.teams[1].teamId}: plan your first march!`] };
  },

  validateAction(ctx, raw, who, a0) {
    const s0 = whole(raw);
    if (s0.stage === 'done') return { ok: false, error: 'The battle is over' };
    const ti = teamOf(s0, who);
    if (ti < 0) return { ok: false, error: "You're not on a team" };
    const a = parse(a0);
    if (!a) return { ok: false, error: 'Not a workshop move' };
    const now = settle(ctx, s0);
    if (now.over) return { ok: false, error: 'The battle is over' };
    const s = now.state, t = s.teams[ti];
    if (a.kind === 'done' && t.done.includes(who)) return { ok: false, error: 'Already done' };
    if (energyNow(t, who) + 1e-9 < costOf(a, s.cfg.rules)) return { ok: false, error: 'Not enough energy' };
    const did = apply(s, ti, a, ctx.now, who, t.active.length < 2);
    return did.error != null ? { ok: false, error: did.error } : { ok: true, action: a };
  },

  reduce(ctx, raw, who, a) {
    const now = settle(ctx, whole(raw));
    if (now.over) return now.over;
    const s = now.state, ti = teamOf(s, who), t = s.teams[ti];
    const have = energyNow(t, who), cost = costOf(a, s.cfg.rules);
    const did = have + 1e-9 >= cost ? apply(s, ti, a, ctx.now, who, t.active.length < 2) : null;
    if (!did || did.error != null) return { state: s, deadlines: deadlines(s, ctx.now) };
    const next: TeamState = { ...t, purses: { ...t.purses, [who]: { energy: have - cost, asOf: ctx.now } }, credit: { ...t.credit, [who]: (t.credit[who] || 0) + cost },
      ...(a.kind === 'done' ? { done: [...t.done, who] } : {}) };
    let state: TowerState = { ...s, field: did.field, queue: did.queue, teams: s.teams.map((x, i) => (i === ti ? next : x)) as [TeamState, TeamState] };
    // everyone who is here has said they are done: no point waiting out the clock
    if (a.kind === 'done' && state.phase === 'plan' && state.teams.every((x) => x.active.every((m) => x.done.includes(m)))) state = { ...state, phaseEndsAt: Math.min(state.phaseEndsAt, ctx.now + s.cfg.doneMs) };
    return { state, deadlines: deadlines(state, ctx.now) };
  },

  onDeadline(ctx, raw) {
    const s0 = whole(raw);
    if (s0.stage === 'done') return { state: s0 };
    const now = settle(ctx, s0);
    if (now.over) return now.over;
    return { state: now.state, deadlines: deadlines(now.state, ctx.now), ...(now.said?.length ? { events: now.said } : {}) };
  },

  shiftTime(raw, ms) {
    const s = whole(raw);
    return { ...s, startedAt: s.startedAt + ms, phaseAt: s.phaseAt + ms, phaseEndsAt: s.phaseEndsAt + ms, field: shiftField(s.field, ms, s.cfg.rules),
      teams: s.teams.map((t): TeamState => ({ ...t, purses: Object.fromEntries(Object.entries(t.purses).map(([m, u]) => [m, { ...u, asOf: u.asOf + ms }])) })) as [TeamState, TeamState] };
  },

  // a simulated hand does something sensible (the simulator decides how often): `handOffers`
  offers: (ctx, raw, who) => handOffers(ctx, raw, who),

  projectForPlayer(raw, pid, now) {
    const s = whole(raw);
    const ti = teamOf(s, pid);
    return { ...publicView(s, now), myTeam: ti >= 0 ? s.teams[ti].teamId : null };
  },
  projectForTv: (raw, now) => publicView(whole(raw), now),
  projectForHost: (raw, now) => publicView(whole(raw), now),
};

/** The battle as every screen draws it: the field as of its last step (while it marches a screen glides each toy
 *  on from there; while everyone plans it stands still), both workshops, what is lined up, everyone's energy. */
function publicView(s: TowerState, now: number): TowerView {
  return {
    stage: s.stage, phase: s.phase, round: s.round, rounds: s.cfg.rounds, phaseAt: s.phaseAt, phaseEndsAt: s.phaseEndsAt, startedAt: s.startedAt, winner: s.winner ?? null, rules: s.cfg.rules,
    field: s.stage === 'done' || s.phase === 'plan' ? s.field : advance(s.field, Math.min(now, s.phaseEndsAt), s.cfg.rules),
    queue: s.queue,
    teams: s.teams.map((t) => ({ teamId: t.teamId, captainId: t.captainId, members: t.members, purses: t.purses, active: t.active, done: t.done })),
  };
}

export type Leaning = 'all' | 'send' | 'build' | 'defend';
/** What a simulated hand would press now. Planning: by default an all-rounder, as a person is (a tower or a better
 *  one while the team's towers are behind the round, else toys; then "done"). Marching: it minds the workshop
 *  (a snowball at what is nearly in, a shield for a crowd, the avalanche with a teammate). A `lean` makes it a
 *  hand that mostly does one thing (the simulator's habits). */
export function handOffers(ctx: MatchCtx, raw: TowerState, who: PlayerId, lean?: Leaning): TowerAction[] {
  const s0 = whole(raw);
  if (s0.stage === 'done') return [];
  const ti = teamOf(s0, who);
  if (ti < 0) return [];
  const now = settle(ctx, s0);
  if (now.over) return [];
  const s = now.state, t = s.teams[ti], p = s.cfg.rules, f = s.field, me = f.sides[ti];
  const have = energyNow(t, who);
  const leaning: Leaning = lean || 'all';
  const coming = (lane: number) => f.units.filter((u) => u.side !== ti && u.lane === lane);

  if (s.phase === 'march') {
    if (leaning === 'send') return [];
    const out: TowerAction[] = [];
    for (let lane = 0; lane < p.lanes; lane++) {
      if (have >= p.shieldCost && ctx.now >= me.shieldAt && coming(lane).filter((u) => 1 - u.pos <= 0.12).length >= 2) out.push({ kind: 'shield', lane });
      else if (have >= p.zapCost && ctx.now >= me.zapAt && coming(lane).some((u) => 1 - u.pos <= (leaning === 'defend' ? p.zapReach : 0.2))) out.push({ kind: 'zap', lane });
    }
    // the avalanche: join a teammate's at once; start one when a crowd is on our half and somebody is there to join
    if (have >= p.comboCost && ctx.now >= me.comboAt) {
      if (armed(me, ctx.now, p)) { if (me.armBy !== who) out.unshift({ kind: 'combo', lane: me.armLane }); }
      else if (t.active.length >= 2) for (let lane = 0; lane < p.lanes; lane++) if (coming(lane).filter((u) => 1 - u.pos <= p.comboReach).length >= 3) out.push({ kind: 'combo', lane });
    }
    return out;
  }

  // planning
  if (t.done.includes(who)) return [];
  const doneNow: TowerAction[] = [{ kind: 'done' }];
  const sends = (keep = 0): TowerAction[] => {
    const out: TowerAction[] = [];
    for (let lane = 0; lane < p.lanes; lane++) {
      if (queued(s, ti, lane) >= s.cfg.maxQueue) continue;
      for (const [unit, k] of Object.entries(p.units)) if (k.cost <= have - keep) out.push({ kind: 'send', lane, unit });
    }
    return out;
  };
  /** The next tower worth having: an empty spot first (where toys are coming), then a better one over a cheaper. */
  const wanted = (): { lane: number; slot: number; tower: string; cost: number } | null => {
    const kinds = Object.entries(p.towers).filter(([, k]) => k.dps > 0).sort((x, y) => x[1].cost - y[1].cost);
    const lanes = Array.from({ length: p.lanes }, (_, l) => l).sort((x, y) => coming(y).length - coming(x).length || x - y);
    for (const slot of p.slots.map((_, i) => i)) for (const lane of lanes) {
      if (me.towers.some((x) => x.lane === lane && x.slot === slot)) continue;
      const pick = kinds[slot % kinds.length];
      if (pick) return { lane, slot, tower: pick[0], cost: pick[1].cost };
    }
    const low = [...me.towers].filter((x) => (x.level || 1) < p.maxLevel).sort((x, y) => (x.level || 1) - (y.level || 1) || coming(y.lane).length - coming(x.lane).length || x.lane - y.lane || x.slot - y.slot)[0];
    return low ? { lane: low.lane, slot: low.slot, tower: low.kind, cost: p.towers[low.kind].cost } : null;
  };
  const builds = (): TowerAction[] => { const w = wanted(); return w && w.cost <= have ? [{ kind: 'build', lane: w.lane, slot: w.slot, tower: w.tower }] : []; };
  if (leaning === 'send') { const o = sends(); return o.length ? o : doneNow; }
  if (leaning === 'build') { const b = builds(); if (b.length) return b; if (wanted()) return doneNow; const o = sends(); return o.length ? o : doneNow; }   // (saving up for the next tower)
  if (leaning === 'defend') { const b = builds(); if (b.length) return b; const o = sends(p.shieldCost); return o.length ? o : doneNow; }                  // keeps a shield's worth back
  // an all-rounder: the team's towers keep pace with the rounds (about two levels a round between them); the rest
  // goes on toys, with a snowball's worth kept back now and then
  const built = me.towers.reduce((n, x) => n + (x.level || 1), 0);
  if (built < 2 * s.round) { const b = builds(); if (b.length) return b; }
  const o = sends(t.members.indexOf(who) % 2 === 0 ? p.zapCost : 0);
  return o.length ? o : doneNow;
}
