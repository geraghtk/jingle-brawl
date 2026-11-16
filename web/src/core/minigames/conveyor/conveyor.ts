// The Great Gift Conveyor (game #3; rebuilt 2026-11-16 as a real belt, Kevin's design). The two
// duelists captain opposing teams (the experience deals everyone else out: engines/teams.ts). Each
// team has a conveyor on the TV (engines/belt.ts): unfinished toys roll along it past a row of
// stations, one for each part (PARTS). Every player works a station from their phone: as a toy
// passes, fit the piece its order asks for (one of VARIANTS) before it has gone by. A toy that
// reaches the end with every piece right goes on the sleigh; anything else falls in the reject bin.
// Now and then a raccoon rides the belt: shoo it off at any station, or it makes off with a toy.
//   Speed: each belt runs faster the more its own team has delivered (so the team in front has the
//   harder belt), and both creep up as the round goes on.
//   Fair: both teams get the same orders in the same order, and the same stations.
//   Won: first to the quota; at the bell, more toys; level, overtime (the first to get ahead); still
//   level, fewer rejects; still level, the host decides. The winning captain wins the duel.
//   Rewards: a recommendation only (the experience applies it): a flat point to the winning team.
import type { Rng } from '../../rng';
import type { PlayerId } from '../../types';
import type { GameDefinition, GameResult, MatchCtx, Transition } from '../contracts';
import { atSpeed, BELT_VERSION, evenZones, newBelt, next, nextAt, wholeBelt, zoneOf } from '../engines/belt';
import type { Belt, BeltItem, Zone } from '../engines/belt';
import { assignTeams, checkAssignment } from '../engines/teams';
import type { TeamAssignment } from '../engines/teams';
import type { ConveyorView } from './view';

/** The stations along the belt, in the order a toy passes them: one piece each. */
export const PARTS = ['wheels', 'lights', 'paint', 'bow'] as const;
export type PartKind = (typeof PARTS)[number];
/** How many kinds of each piece there are to choose from. */
export const VARIANTS = 3;
const NONE = -1;                 // a piece not fitted yet (a null would vanish from stored state)

export interface Order { want: number[]; stowaway: boolean }
/** A toy on the belt: what its order wants at each station, what has been fitted (NONE: nothing yet),
 *  and who fitted it. A stowaway is no toy at all. */
export interface Toy { n: number; want: number[]; fitted: number[]; by: PlayerId[]; stowaway: boolean }

export interface ConveyorConfig {
  quota: number; roundMs: number; overtimeMs: number;
  /** Belt lengths a second at level 0, and how much faster each level is (a share of that). */
  baseSpeed: number; speedStep: number; maxLevel: number;
  /** A belt goes up a level for every so many toys ITS team has delivered, and both go up one every so long. */
  perLevel: number; timeLevelMs: number;
  /** How far apart toys enter (belt lengths), and how wide a station's reach is. */
  gap: number; zone: number;
  stowawayChance: number; rewardBase: number;
  teams?: TeamAssignment;                      // the experience's deal (else the game deals, balanced)
}
export const DEFAULTS: ConveyorConfig = {
  quota: 8, roundMs: 90_000, overtimeMs: 20_000, baseSpeed: 0.1, speedStep: 0.14, maxLevel: 7, perLevel: 2, timeLevelMs: 25_000,
  gap: 0.36, zone: 0.17, stowawayChance: 0.14, rewardBase: 1,
};

export interface TeamState {
  teamId: 'A' | 'B'; captainId: PlayerId; members: PlayerId[];
  /** Who works each station (by PARTS): a small team's players take several, a big team shares. */
  crew: PlayerId[][];
  belt: Belt<Toy>;
  delivered: number; rejected: number; stolen: number;
  credit: Record<PlayerId, number>;            // contribution: a share of each delivered toy, and raccoons shooed
}
export interface ConveyorState {
  cfg: ConveyorConfig;
  stage: 'running' | 'overtime' | 'done';
  startedAt: number;
  endsAt: number;
  orders: Order[];
  teams: [TeamState, TeamState];
  winner?: 'A' | 'B' | null;
}

export type ConveyorAction = { kind: 'fit'; toy: string; variant: number } | { kind: 'shoo'; toy: string };

const int = (rng: Rng, n: number) => Math.floor(rng() * n);

/** The orders, drawn once: both teams work the same ones in the same order. */
export function makeOrders(rng: Rng, count: number, stowawayChance: number): Order[] {
  return Array.from({ length: count }, (_, i) => ({
    want: PARTS.map(() => int(rng, VARIANTS)),
    stowaway: i > 1 && rng() < stowawayChance,        // never the first two: the table is still finding its stations
  }));
}

/** Stored state drops empty objects and lists: put them back. */
function whole(s: ConveyorState): ConveyorState {
  return {
    ...s, orders: s.orders || [],
    teams: s.teams.map((t) => {
      const belt = wholeBelt(t.belt);
      return {
        ...t, credit: t.credit || {}, crew: PARTS.map((_, k) => t.crew?.[k] || []),
        belt: { ...belt, items: belt.items.map((it) => ({ ...it, data: { ...it.data, want: it.data.want || [], fitted: it.data.fitted || PARTS.map(() => NONE), by: it.data.by || PARTS.map(() => ''), stowaway: !!it.data.stowaway } })) },
      };
    }) as [TeamState, TeamState],
  };
}

export const zones = (c: ConveyorConfig): Zone[] => evenZones(PARTS.length, c.zone);
const teamOf = (s: ConveyorState, pid: PlayerId) => s.teams.findIndex((t) => t.members.includes(pid));
const isRight = (toy: Toy) => !toy.stowaway && toy.want.every((v, k) => toy.fitted[k] === v);

/** A belt's level at `at`: its team's own deliveries, plus the round's age. */
export function levelAt(s: ConveyorState, t: TeamState, at: number): number {
  const c = s.cfg;
  return Math.max(0, Math.min(c.maxLevel, Math.floor(t.delivered / c.perLevel) + Math.floor(Math.max(0, at - s.startedAt) / c.timeLevelMs)));
}
export const speedAt = (c: ConveyorConfig, level: number) => c.baseSpeed * (1 + c.speedStep * level);
/** When the round's age next pushes the belts up a level. */
const nextTimeLevel = (s: ConveyorState, at: number) => s.startedAt + (Math.floor(Math.max(0, at - s.startedAt) / s.cfg.timeLevelMs) + 1) * s.cfg.timeLevelMs;

/** Who works each station: the members in turn; a station nobody drew goes to a member who already has one. */
export function crewFor(members: PlayerId[]): PlayerId[][] {
  return PARTS.map((_, k) => {
    const drawn = members.filter((_m, i) => i % PARTS.length === k);
    return drawn.length ? drawn : [members[k % members.length]];
  });
}
/** May `who` work station `k`: it is theirs, or its whole crew has dropped out and they are on the team. */
function mayWork(ctx: { connected: (p: PlayerId) => boolean }, t: TeamState, who: PlayerId, k: number): boolean {
  if (t.crew[k].includes(who)) return true;
  return t.members.includes(who) && t.crew[k].every((p) => !ctx.connected(p));
}

function teamsFrom(ctx: MatchCtx, c: ConveyorConfig): TeamAssignment {
  const given = checkAssignment(c.teams, ctx.duelists);
  if (given) return given;
  return assignTeams(ctx.duelists!, ctx.participants, ctx.rng);
}

/** One team's belt brought up to `now`: toys enter, ride, and fall off the end onto the sleigh or into
 *  the bin; the speed follows the level. Stops the moment the team reaches the quota (`wonAt`). */
function runTeam(s: ConveyorState, t0: TeamState, now: number): { team: TeamState; wonAt: number | null; events: string[] } {
  let t = t0, belt = t0.belt;
  const events: string[] = [];
  const make = (n: number) => {
    const o = s.orders[n % s.orders.length];
    return { id: `${t0.teamId}-${n}`, data: { n, want: o.want, fitted: PARTS.map(() => NONE), by: PARTS.map(() => ''), stowaway: o.stowaway } };
  };
  for (let turns = 0; turns < 2000; turns++) {
    belt = atSpeed(belt, speedAt(s.cfg, levelAt(s, t, belt.asOf)));
    const until = Math.min(now, nextTimeLevel(s, belt.asOf));
    const r = next(belt, until, s.cfg.gap, make);
    belt = r.belt;
    if (!r.event) { if (until >= now) break; continue; }             // the round's age moved the level: set the speed again
    if (r.event.kind !== 'off') continue;
    const toy = r.event.item.data;
    if (toy.stowaway) {
      const took = t.delivered > 0 ? 1 : 0;
      t = { ...t, delivered: t.delivered - took, stolen: t.stolen + 1 };
      events.push(took ? `🎁 A raccoon made off with one of team ${t.teamId}'s toys!` : `🎁 A raccoon rode team ${t.teamId}'s belt to the end and found nothing to take`);
    } else if (isRight(toy)) {
      const credit = { ...t.credit };
      for (const p of toy.by) if (p) credit[p] = (credit[p] || 0) + 1 / PARTS.length;
      t = { ...t, delivered: t.delivered + 1, credit };
      if (t.delivered >= s.cfg.quota) return { team: { ...t, belt }, wonAt: r.event.at, events };
    } else t = { ...t, rejected: t.rejected + 1 };
  }
  return { team: { ...t, belt }, wonAt: null, events };
}

function finish(ctx: MatchCtx, s: ConveyorState, winner: 'A' | 'B' | null, reason: string, events: string[] = []): Transition<ConveyorState> {
  const done: ConveyorState = { ...s, stage: 'done', winner, teams: s.teams.map((t): TeamState => ({ ...t, belt: { ...t.belt, speed: 0, items: [] } })) as [TeamState, TeamState] };
  const [a, b] = s.teams;
  const metrics = {
    delivered: { A: a.delivered, B: b.delivered }, rejected: { A: a.rejected, B: b.rejected }, stolen: { A: a.stolen, B: b.stolen },
    teams: s.teams.map((t) => ({ teamId: t.teamId, captainId: t.captainId, members: t.members })), belt: BELT_VERSION,
  };
  const performance = s.teams.flatMap((t) => t.members.map((playerId) => ({ playerId, contribution: Math.round((t.credit[playerId] || 0) * 100) / 100 })));
  const base = { gameId: conveyor.id, instanceId: ctx.instanceId, mode: ctx.mode, performance, metrics, scoringPolicyVersion: 'captain_team@2', reason };
  const duelists = ctx.duelists!;
  if (!winner) {
    return { state: done, result: { ...base, status: 'requires_tiebreaker', winners: [], losers: [], duel: { duelists, resolutionReason: reason } }, events: [...events, `🎁 Gift Conveyor: ${reason}`] };
  }
  const win = s.teams.find((t) => t.teamId === winner)!, lose = s.teams.find((t) => t.teamId !== winner)!;
  const rewards = win.members.map((playerId) => ({ playerId, points: s.cfg.rewardBase, reason: 'the winning team' }));
  const result: GameResult = {
    ...base, status: 'resolved', winners: win.members, losers: lose.members, rewards,
    duel: { duelists, winningDuelistId: win.captainId, losingDuelistId: lose.captainId, resolutionReason: reason },
  };
  return { state: done, result, events: [...events, `🎁 Gift Conveyor: team ${win.teamId} (${reason})`] };
}

/** What is due next by itself: a toy entering or leaving either belt, the level going up, the bell. */
function deadlines(s: ConveyorState) {
  const soonest = Math.min(...s.teams.map((t) => nextAt(atSpeed(t.belt, speedAt(s.cfg, levelAt(s, t, t.belt.asOf))))), nextTimeLevel(s, Math.max(...s.teams.map((t) => t.belt.asOf))));
  return [{ id: 'end', at: s.endsAt }, ...(soonest < s.endsAt ? [{ id: 'tick', at: soonest }] : [])];
}

/** Both belts brought up to `now` (never past the bell), and everything that follows from it: a team
 *  reaching the quota, the bell, overtime. `over`: the game ended (its result). */
function settle(ctx: MatchCtx, s: ConveyorState): { state: ConveyorState; over?: Transition<ConveyorState>; events: string[] } {
  const now = Math.min(ctx.now, s.endsAt);
  const ran = s.teams.map((t) => runTeam(s, t, now));
  const events = ran.flatMap((r) => r.events);
  const state: ConveyorState = { ...s, teams: ran.map((r) => r.team) as [TeamState, TeamState] };
  const [a, b] = ran;
  if (a.wonAt != null || b.wonAt != null) {
    const first = b.wonAt == null || (a.wonAt != null && a.wonAt <= b.wonAt) ? 'A' : 'B';
    return { state, events, over: finish(ctx, state, first, `first to ${s.cfg.quota} toys`, events) };
  }
  const [ta, tb] = state.teams;
  // overtime is sudden: the moment one team is ahead, it has won
  if (s.stage === 'overtime' && ta.delivered !== tb.delivered) return { state, events, over: finish(ctx, state, ta.delivered > tb.delivered ? 'A' : 'B', 'ahead in overtime', events) };
  if (ctx.now < s.endsAt) return { state, events };
  // the bell: more toys wins; level → overtime once; then fewer rejects; then the host
  if (ta.delivered !== tb.delivered) return { state, events, over: finish(ctx, state, ta.delivered > tb.delivered ? 'A' : 'B', 'more toys on the sleigh at the bell', events) };
  if (s.stage === 'running') {
    const extra: ConveyorState = { ...state, stage: 'overtime', endsAt: ctx.now + s.cfg.overtimeMs };
    return { state: extra, events: [...events, '🎁 Level! Overtime: the first team ahead wins'] };
  }
  if (ta.rejected !== tb.rejected) return { state, events, over: finish(ctx, state, ta.rejected < tb.rejected ? 'A' : 'B', 'level after overtime: fewer rejects', events) };
  return { state, events, over: finish(ctx, state, null, 'level after overtime, with as many rejects', events) };
}

export const conveyor: GameDefinition<ConveyorConfig, ConveyorState, ConveyorAction> = {
  id: 'gift_conveyor', version: '2', label: 'The Great Gift Conveyor', emoji: '🎁',
  rules: 'The duellists captain two teams, each with a conveyor on the TV. Unfinished toys roll past your station: fit the piece each one asks for before it has gone by. Finished toys go on the sleigh; first team to fill it wins. The better you do, the faster your belt runs. Shoo off any raccoon!',
  categories: ['team_based', 'cross_device', 'real_time_action'],
  supportedModes: ['captain_teams'],
  compatibility: {
    version: 1, environments: ['in_person', 'remote', 'hybrid'], displayRole: 'required_shared_world', topology: ['phone_to_shared_display'],
    devices: [{ feature: 'touch', required: false, fallback: 'buttons work with a mouse or keyboard' }], players: { min: 2, max: 40 }, modes: ['captain_teams'],
  },

  validateConfig(raw) {
    const r = (raw && typeof raw === 'object' ? raw : {}) as Partial<ConveyorConfig>;
    const c: ConveyorConfig = { ...DEFAULTS, ...r };
    const ok = c.quota >= 1 && c.roundMs > 0 && c.overtimeMs > 0 && c.baseSpeed > 0 && c.baseSpeed <= 1 && c.speedStep >= 0 && c.maxLevel >= 0 && c.perLevel >= 1
      && c.timeLevelMs > 0 && c.gap > 0.05 && c.zone > 0 && c.zone < 1 / (PARTS.length + 1) && c.zone < c.gap && c.stowawayChance >= 0 && c.stowawayChance <= 1;
    if (!ok) throw new Error('bad Gift Conveyor settings');
    return c;
  },

  // overtime, then fewer rejects, settle nearly every tie; a dead level goes to the host
  canAlwaysResolve: (mode) => (mode === 'captain_teams' ? { ok: true } : { ok: false, reason: 'captains and teams only' }),

  createInitialState(ctx, c) {
    if (!ctx.duelists) throw new Error('the Gift Conveyor needs two captains');
    const deal = teamsFrom(ctx, c);
    const orders = makeOrders(ctx.rng, Math.max(24, c.quota * 4), c.stowawayChance);
    const state: ConveyorState = {
      cfg: c, stage: 'running', startedAt: ctx.now, endsAt: ctx.now + c.roundMs, orders,
      teams: deal.teams.map((t) => ({
        teamId: t.teamId, captainId: t.captainId, members: t.memberIds, crew: crewFor(t.memberIds),
        belt: newBelt<Toy>(ctx.now, speedAt(c, 0)), delivered: 0, rejected: 0, stolen: 0, credit: {},
      })) as [TeamState, TeamState],
    };
    return { state, deadlines: deadlines(state), events: [`🎁 Team ${deal.teams[0].teamId} vs team ${deal.teams[1].teamId}: go!`] };
  },

  validateAction(ctx, raw, who, a0) {
    const s0 = whole(raw);
    const a = a0 as { kind?: unknown; toy?: unknown; variant?: unknown } | null;
    if (s0.stage === 'done') return { ok: false, error: 'The sleigh has left' };
    const ti = teamOf(s0, who);
    if (ti < 0) return { ok: false, error: "You're not on a team" };
    // as the belts stand right now
    const now = settle(ctx, s0);
    if (now.over) return { ok: false, error: 'The sleigh has left' };
    const t = now.state.teams[ti];
    const it = typeof a?.toy === 'string' ? t.belt.items.find((x) => x.id === a.toy) : undefined;
    if (!it) return { ok: false, error: 'That one has gone by' };
    const k = zoneOf(zones(s0.cfg), it.pos);
    if (k < 0) return { ok: false, error: 'It is between stations' };
    if (!mayWork(ctx, t, who, k)) return { ok: false, error: 'That is not your station' };
    if (a?.kind === 'shoo') return it.data.stowaway ? { ok: true, action: { kind: 'shoo', toy: it.id } } : { ok: false, error: 'Nothing to shoo' };
    if (a?.kind !== 'fit') return { ok: false, error: 'Fit a piece, or shoo' };
    if (it.data.stowaway) return { ok: false, error: 'That is a raccoon: shoo it off!' };
    if (!Number.isInteger(a.variant) || (a.variant as number) < 0 || (a.variant as number) >= VARIANTS) return { ok: false, error: 'Pick one of the pieces' };
    if (it.data.fitted[k] !== NONE) return { ok: false, error: 'That piece is already on' };
    return { ok: true, action: { kind: 'fit', toy: it.id, variant: a.variant as number } };
  },

  reduce(ctx, raw, who, a) {
    const s0 = whole(raw);
    const now = settle(ctx, s0);
    if (now.over) return now.over;
    const s = now.state;
    const ti = teamOf(s, who);
    let t = s.teams[ti];
    const it = t.belt.items.find((x) => x.id === a.toy);
    if (!it) return { state: s, deadlines: deadlines(s), events: now.events };
    const k = zoneOf(zones(s.cfg), it.pos);
    let items: BeltItem<Toy>[];
    if (a.kind === 'shoo') {
      items = t.belt.items.filter((x) => x !== it);
      t = { ...t, credit: { ...t.credit, [who]: (t.credit[who] || 0) + 0.25 } };
    } else {
      const fitted = it.data.fitted.map((v, i) => (i === k ? a.variant : v)), by = it.data.by.map((p, i) => (i === k ? who : p));
      items = t.belt.items.map((x) => (x === it ? { ...x, data: { ...x.data, fitted, by } } : x));
    }
    t = { ...t, belt: { ...t.belt, items } };
    const state: ConveyorState = { ...s, teams: s.teams.map((x, i) => (i === ti ? t : x)) as [TeamState, TeamState] };
    return { state, deadlines: deadlines(state), ...(now.events.length ? { events: now.events } : {}) };
  },

  onDeadline(ctx, raw) {
    const s = whole(raw);
    if (s.stage === 'done') return { state: s };
    const now = settle(ctx, s);
    if (now.over) return now.over;
    return { state: now.state, deadlines: deadlines(now.state), ...(now.events.length ? { events: now.events } : {}) };
  },

  offers(ctx, raw, who) {
    const s0 = whole(raw);
    if (s0.stage === 'done') return [];
    const ti = teamOf(s0, who);
    if (ti < 0) return [];
    const now = settle(ctx, s0);
    if (now.over) return [];
    const t = now.state.teams[ti], zs = zones(s0.cfg);
    // a simulated hand fits the right piece on whatever is at one of its stations (the simulator decides when it slips)
    for (const it of t.belt.items) {
      const k = zoneOf(zs, it.pos);
      if (k < 0 || !mayWork(ctx, t, who, k)) continue;
      if (it.data.stowaway) return [{ kind: 'shoo', toy: it.id }];
      if (it.data.fitted[k] === NONE) return [{ kind: 'fit', toy: it.id, variant: it.data.want[k] }];
    }
    return [];
  },

  projectForPlayer(raw, pid, now) {
    const s = whole(raw);
    const ti = teamOf(s, pid);
    const mine = ti >= 0 ? PARTS.map((_, k) => k).filter((k) => s.teams[ti].crew[k].includes(pid)) : [];
    return { ...publicView(s, now), myTeam: ti >= 0 ? s.teams[ti].teamId : null, myStations: mine };
  },
  projectForTv: (raw, now) => publicView(whole(raw), now),
  projectForHost: (raw, now) => publicView(whole(raw), now),
};

/** Both belts as every screen draws them: where each toy was at `asOf` and how fast the belt runs (a
 *  screen works out where it is now), what each toy wants and has, who works which station. */
function publicView(s: ConveyorState, now: number): ConveyorView {
  return {
    stage: s.stage, endsAt: s.endsAt, quota: s.cfg.quota, winner: s.winner ?? null, parts: PARTS, variants: VARIANTS, zones: zones(s.cfg),
    teams: s.teams.map((t) => {
      const level = levelAt(s, t, Math.min(now, s.endsAt));
      return {
        teamId: t.teamId, captainId: t.captainId, members: t.members, crew: t.crew, delivered: t.delivered, rejected: t.rejected, stolen: t.stolen,
        level, speed: s.stage === 'done' ? 0 : t.belt.speed, asOf: t.belt.asOf,
        toys: t.belt.items.map((it) => ({ id: it.id, pos: it.pos, want: it.data.want, fitted: it.data.fitted, stowaway: it.data.stowaway })),
      };
    }),
  };
}
