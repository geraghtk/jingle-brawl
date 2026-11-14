// The Great Gift Conveyor (spec, game #3): the two duelists captain opposing teams (the experience
// deals everyone else out: engines/teams.ts) and race to load correct gifts onto the sleigh. Each
// gift goes down the team's line (engines/workflow.ts): pick the toy, wrap it, label it, then check
// and load it, one phone after another; the TV shows both factories. Sometimes a raccoon or a penguin
// sneaks into a box: the checker shoos it out. A wrong gift sent to the sleigh doesn't count.
//   Fair: both teams get the same orders in the same order, and the same number of stations (the
//   smaller team's size, up to four), each working one gift at a time.
//   Won: first to the quota; at the bell, more gifts; level, a short overtime; still level, fewer
//   wrong gifts; still level, the host decides. The winning captain wins the duel, whoever did most.
//   Rewards: a recommendation only (the experience applies it): the winning team shares a base plus
//   a bonus by contribution (each step of a delivered gift counts the same, whatever the station);
//   the losing team gets none.
import type { Rng } from '../../rng';
import type { PlayerId } from '../../types';
import type { GameDefinition, GameResult, MatchCtx, Transition } from '../contracts';
import { assignTeams, checkAssignment } from '../engines/teams';
import type { TeamAssignment } from '../engines/teams';
import { add, complete, dispatch, holding, layout, release, remove, stationOf, WORKFLOW_VERSION } from '../engines/workflow';
import type { Floor, Line } from '../engines/workflow';
import type { ConveyorView } from './view';

export const TOYS = ['🧸', '🚂', '🎸', '🛹', '🪁', '🦖'];
export const PAPERS = ['red', 'green', 'gold', 'blue', 'purple'];
export const TAGS = ['Grandma', 'Uncle Bob', 'Little Sam', 'Aunt Jo', 'Cousin Lee', 'Mrs Claus'];
export const STOWAWAYS = ['🦝', '🐧'];
export const STEPS = ['pick', 'wrap', 'label', 'check'] as const;
export type StepKind = (typeof STEPS)[number];

export interface Order { toy: number; paper: number; tag: number; stowaway: number | null; options: { toy: number[]; paper: number[]; tag: number[] } }
export interface Gift { n: number; toy: number | null; paper: number | null; tag: number | null; stowaway: number | null }

export interface ConveyorConfig {
  quota: number; roundMs: number; overtimeMs: number; maxStations: number;
  stowawayChance: number; rewardBase: number; rewardBonus: number;
  teams?: TeamAssignment;                      // the experience's deal (else the game deals, balanced)
}
export const DEFAULTS: ConveyorConfig = { quota: 5, roundMs: 90_000, overtimeMs: 20_000, maxStations: 4, stowawayChance: 0.25, rewardBase: 2, rewardBonus: 3 };

export interface TeamState {
  teamId: 'A' | 'B'; captainId: PlayerId; members: PlayerId[];
  line: Line; floor: Floor<Gift>;
  next: number;                                // the next order to start
  delivered: number; wrong: number;
  credit: Record<PlayerId, number>;            // contribution: shares of delivered gifts + fixes
  fixes: Record<PlayerId, number>;
}
export interface ConveyorState {
  cfg: ConveyorConfig;
  stage: 'running' | 'overtime' | 'done';
  endsAt: number;
  orders: Order[];
  teams: [TeamState, TeamState];
  winner?: 'A' | 'B' | null;
}

export type ConveyorAction =
  | { kind: 'pick' | 'wrap' | 'label'; item: string; choice: number }
  | { kind: 'shoo' | 'load' | 'back'; item: string };

const TICK_MS = 3000;          // a dropped phone's gift goes back to its station this often

const int = (rng: Rng, n: number) => Math.floor(rng() * n);
function options(rng: Rng, right: number, of: number, k = 4): number[] {
  const pool = Array.from({ length: of }, (_, i) => i).filter((i) => i !== right);
  const pick: number[] = [right];
  while (pick.length < Math.min(k, of)) pick.push(pool.splice(int(rng, pool.length), 1)[0]);
  for (let i = pick.length - 1; i > 0; i--) { const j = int(rng, i + 1); [pick[i], pick[j]] = [pick[j], pick[i]]; }
  return pick;
}

/** The orders, drawn once: both teams work the same ones in the same order. */
export function makeOrders(rng: Rng, count: number, stowawayChance: number): Order[] {
  return Array.from({ length: count }, () => {
    const toy = int(rng, TOYS.length), paper = int(rng, PAPERS.length), tag = int(rng, TAGS.length);
    const stowaway = rng() < stowawayChance ? int(rng, STOWAWAYS.length) : null;
    return { toy, paper, tag, stowaway, options: { toy: options(rng, toy, TOYS.length), paper: options(rng, paper, PAPERS.length), tag: options(rng, tag, TAGS.length) } };
  });
}

/** Stored state drops empty objects and lists: put them back. */
function whole(s: ConveyorState): ConveyorState {
  return {
    ...s, orders: s.orders || [],
    teams: s.teams.map((t) => ({
      ...t, credit: t.credit || {}, fixes: t.fixes || {},
      floor: { rr: t.floor?.rr || [], items: Object.fromEntries(Object.entries(t.floor?.items || {}).map(([k, it]) => [k, {
        ...it, holder: it.holder ?? null, touched: it.touched || [],
        // a null field can come back missing: "no toy yet", "no stowaway" is null again
        data: { ...it.data, toy: it.data.toy ?? null, paper: it.data.paper ?? null, tag: it.data.tag ?? null, stowaway: it.data.stowaway ?? null },
      }])) },
    })) as [TeamState, TeamState],
  };
}

const teamOf = (s: ConveyorState, pid: PlayerId) => s.teams.findIndex((t) => t.members.includes(pid));
const orderOf = (s: ConveyorState, g: Gift) => s.orders[g.n % s.orders.length];
const isRight = (o: Order, g: Gift) => g.toy === o.toy && g.paper === o.paper && g.tag === o.tag && g.stowaway === null;

/** Keep the line fed: a new gift when the first station is free and the line has room (at most one
 *  gift per station on the floor: a stuck station holds the line up, it doesn't pile gifts behind it). */
function feed(s: ConveyorState, t: TeamState, available: (p: PlayerId) => boolean): TeamState {
  let floor = t.floor, next = t.next;
  const onFloor = Object.values(floor.items);
  const firstBusy = onFloor.some((it) => stationOf(t.line, it.step) === 0);
  if (!firstBusy && onFloor.length < t.line.stations.length) {
    const o = s.orders[next % s.orders.length];
    floor = add(floor, `${t.teamId}-${next}`, { n: next, toy: null, paper: null, tag: null, stowaway: o.stowaway });
    next++;
  }
  return { ...t, floor: dispatch(t.line, floor, available), next };
}

function teamsFrom(ctx: MatchCtx, c: ConveyorConfig): TeamAssignment {
  const given = checkAssignment(c.teams, ctx.duelists);
  if (given) return given;
  return assignTeams(ctx.duelists!, ctx.participants, ctx.rng);
}

function finish(ctx: MatchCtx, s: ConveyorState, winner: 'A' | 'B' | null, reason: string): Transition<ConveyorState> {
  const done: ConveyorState = { ...s, stage: 'done', winner, teams: s.teams.map((t) => ({ ...t, floor: { items: {}, rr: t.floor.rr } })) as [TeamState, TeamState] };
  const [a, b] = s.teams;
  const metrics = { delivered: { A: a.delivered, B: b.delivered }, wrong: { A: a.wrong, B: b.wrong }, teams: s.teams.map((t) => ({ teamId: t.teamId, captainId: t.captainId, members: t.members })), workflow: WORKFLOW_VERSION };
  const performance = s.teams.flatMap((t) => t.members.map((playerId) => ({ playerId, contribution: Math.round((t.credit[playerId] || 0) * 100) / 100 })));
  const base = { gameId: conveyor.id, instanceId: ctx.instanceId, mode: ctx.mode, performance, metrics, scoringPolicyVersion: 'captain_team@1', reason };
  const duelists = ctx.duelists!;
  if (!winner) {
    return { state: done, result: { ...base, status: 'requires_tiebreaker', winners: [], losers: [], duel: { duelists, resolutionReason: reason } }, events: [`🎁 Gift Conveyor: ${reason}`] };
  }
  const win = s.teams.find((t) => t.teamId === winner)!, lose = s.teams.find((t) => t.teamId !== winner)!;
  const top = Math.max(0, ...win.members.map((p) => win.credit[p] || 0));
  const rewards = win.members.map((playerId) => ({
    playerId, points: s.cfg.rewardBase + (top > 0 ? Math.round((s.cfg.rewardBonus * (win.credit[playerId] || 0)) / top) : 0), reason: 'the winning team',
  }));
  const result: GameResult = {
    ...base, status: 'resolved', winners: win.members, losers: lose.members, rewards,
    duel: { duelists, winningDuelistId: win.captainId, losingDuelistId: lose.captainId, resolutionReason: reason },
  };
  return { state: done, result, events: [`🎁 Gift Conveyor: team ${win.teamId} (${reason})`] };
}

/** The bell: more gifts wins; level → overtime once; then fewer wrong gifts; then the host. */
function bell(ctx: MatchCtx, s: ConveyorState): Transition<ConveyorState> {
  const [a, b] = s.teams;
  if (a.delivered !== b.delivered) return finish(ctx, s, a.delivered > b.delivered ? 'A' : 'B', 'more gifts on the sleigh at the bell');
  if (s.stage === 'running') {
    const state = { ...s, stage: 'overtime' as const, endsAt: ctx.now + s.cfg.overtimeMs };
    return { state, deadlines: [{ id: 'end', at: state.endsAt }, { id: 'tick', at: ctx.now + TICK_MS }], events: ['🎁 Level! Overtime'] };
  }
  if (a.wrong !== b.wrong) return finish(ctx, s, a.wrong < b.wrong ? 'A' : 'B', 'level after overtime: fewer wrong gifts');
  return finish(ctx, s, null, 'level after overtime, with as many wrong gifts');
}

export const conveyor: GameDefinition<ConveyorConfig, ConveyorState, ConveyorAction> = {
  id: 'gift_conveyor', version: '1', label: 'The Great Gift Conveyor', emoji: '🎁',
  rules: 'The duellists captain two teams. Pick, wrap, label and check gifts, phone to phone, and load the right ones onto the sleigh first. Watch for stowaways!',
  categories: ['team_based', 'cross_device', 'asymmetric_cooperative'],
  supportedModes: ['captain_teams'],
  compatibility: {
    version: 1, environments: ['in_person', 'remote', 'hybrid'], displayRole: 'required_shared_world', topology: ['phone_to_phone', 'phone_to_shared_display'],
    devices: [{ feature: 'touch', required: false, fallback: 'buttons work with a mouse or keyboard' }], players: { min: 2, max: 40 }, modes: ['captain_teams'],
  },

  validateConfig(raw) {
    const r = (raw && typeof raw === 'object' ? raw : {}) as Partial<ConveyorConfig>;
    const c: ConveyorConfig = { ...DEFAULTS, ...r };
    if (!(c.quota >= 1 && c.roundMs > 0 && c.overtimeMs > 0 && c.maxStations >= 1 && c.stowawayChance >= 0 && c.stowawayChance <= 1)) throw new Error('bad Gift Conveyor settings');
    return c;
  },

  // overtime, then fewer wrong gifts, settle nearly every tie; a dead level goes to the host
  canAlwaysResolve: (mode) => (mode === 'captain_teams' ? { ok: true } : { ok: false, reason: 'captains and teams only' }),

  createInitialState(ctx, c) {
    if (!ctx.duelists) throw new Error('the Gift Conveyor needs two captains');
    const deal = teamsFrom(ctx, c);
    const stations = Math.max(1, Math.min(c.maxStations, ...deal.teams.map((t) => t.memberIds.length)));
    const orders = makeOrders(ctx.rng, c.quota * 4, c.stowawayChance);
    const s0: ConveyorState = { cfg: c, stage: 'running', endsAt: ctx.now + c.roundMs, orders, teams: [] as unknown as [TeamState, TeamState] };
    s0.teams = deal.teams.map((t) => feed(s0, {
      teamId: t.teamId, captainId: t.captainId, members: t.memberIds, line: layout([...STEPS], stations, t.memberIds),
      floor: { items: {}, rr: [] }, next: 0, delivered: 0, wrong: 0, credit: {}, fixes: {},
    }, ctx.connected)) as [TeamState, TeamState];
    return { state: s0, deadlines: [{ id: 'end', at: s0.endsAt }, { id: 'tick', at: ctx.now + TICK_MS }], events: [`🎁 Team ${deal.teams[0].teamId} vs team ${deal.teams[1].teamId}: go!`] };
  },

  validateAction(_ctx, raw, who, a0) {
    const s = whole(raw);
    const a = a0 as { kind?: unknown; item?: unknown; choice?: unknown } | null;
    if (s.stage === 'done') return { ok: false, error: 'The sleigh has left' };
    const ti = teamOf(s, who);
    if (ti < 0) return { ok: false, error: "You're not on a team" };
    const t = s.teams[ti];
    const it = typeof a?.item === 'string' ? t.floor.items[a.item] : undefined;
    if (!it) return { ok: false, error: 'That gift has moved on' };
    if (it.holder !== who) return { ok: false, error: "That gift isn't on your table" };
    const step = STEPS[it.step];
    const o = orderOf(s, it.data);
    if (a?.kind !== step && !(step === 'check' && (a?.kind === 'shoo' || a?.kind === 'load' || a?.kind === 'back'))) return { ok: false, error: `This gift needs ${step === 'check' ? 'checking' : step === 'pick' ? 'a toy' : step === 'wrap' ? 'wrapping' : 'a label'}` };
    if (step === 'pick' || step === 'wrap' || step === 'label') {
      const opts = o.options[step === 'pick' ? 'toy' : step === 'wrap' ? 'paper' : 'tag'];
      if (typeof a.choice !== 'number' || !opts.includes(a.choice)) return { ok: false, error: 'Pick one of these' };
      return { ok: true, action: { kind: step, item: it.id, choice: a.choice } };
    }
    if (a.kind === 'shoo' && it.data.stowaway === null) return { ok: false, error: 'Nothing to shoo' };
    if (a.kind === 'load' && it.data.stowaway !== null) return { ok: false, error: `Shoo the ${STOWAWAYS[it.data.stowaway] === '🦝' ? 'raccoon' : 'penguin'} out first!` };
    return { ok: true, action: { kind: a.kind as 'shoo' | 'load' | 'back', item: it.id } };
  },

  reduce(ctx, raw, who, a) {
    const s = whole(raw);
    const ti = teamOf(s, who);
    let t = s.teams[ti];
    const it = t.floor.items[a.item];
    const o = orderOf(s, it.data);
    let events: string[] | undefined;
    if (a.kind === 'pick' || a.kind === 'wrap' || a.kind === 'label') {
      const field = a.kind === 'pick' ? 'toy' : a.kind === 'wrap' ? 'paper' : 'tag';
      const r = complete(t.line, t.floor, it.id, who, { ...it.data, [field]: a.choice }, undefined, ctx.connected);
      if ('error' in r) return { state: s };
      t = { ...t, floor: r.floor };
    } else if (a.kind === 'shoo') {
      t = { ...t, floor: { ...t.floor, items: { ...t.floor.items, [it.id]: { ...it, data: { ...it.data, stowaway: null } } } }, fixes: { ...t.fixes, [who]: (t.fixes[who] || 0) + 1 },
        credit: { ...t.credit, [who]: (t.credit[who] || 0) + 0.25 } };
    } else if (a.kind === 'back') {
      // back to the start of the line, the same order, emptied (a fix if it was wrong)
      const wasWrong = !isRight(o, it.data);
      const r = complete(t.line, t.floor, it.id, who, { ...it.data, toy: null, paper: null, tag: null }, 0, ctx.connected);
      if ('error' in r) return { state: s };
      t = { ...t, floor: r.floor, ...(wasWrong ? { fixes: { ...t.fixes, [who]: (t.fixes[who] || 0) + 1 }, credit: { ...t.credit, [who]: (t.credit[who] || 0) + 0.25 } } : {}) };
    } else {
      // load it on the sleigh: a right gift counts; each step's worker gets an equal share of it
      const right = isRight(o, it.data);
      const credit = { ...t.credit };
      if (right) for (const tch of [...it.touched, { by: who, step: it.step }]) credit[tch.by] = (credit[tch.by] || 0) + 1 / STEPS.length;
      t = { ...t, floor: dispatch(t.line, remove(t.floor, it.id), ctx.connected), delivered: t.delivered + (right ? 1 : 0), wrong: t.wrong + (right ? 0 : 1), credit };
      events = right ? undefined : [`🎁 Team ${t.teamId} loaded a wrong gift!`];
    }
    t = feed(s, t, ctx.connected);
    const teams = s.teams.map((x, i) => (i === ti ? t : x)) as [TeamState, TeamState];
    const next: ConveyorState = { ...s, teams };
    if (t.delivered >= s.cfg.quota) return finish(ctx, next, t.teamId, `first to ${s.cfg.quota} gifts`);
    return { state: next, ...(events ? { events } : {}) };
  },

  onDeadline(ctx, raw, id) {
    const s = whole(raw);
    if (s.stage === 'done') return { state: s };
    if (id === 'end') return bell(ctx, s);
    // tick: gifts held by phones that dropped out go back to their station's next person
    const teams = s.teams.map((t) => feed(s, { ...t, floor: release(t.line, t.floor, (p) => !ctx.connected(p), ctx.connected) }, ctx.connected)) as [TeamState, TeamState];
    return { state: { ...s, teams }, deadlines: [{ id: 'end', at: s.endsAt }, { id: 'tick', at: ctx.now + TICK_MS }] };
  },

  offers(_ctx, raw, who) {
    const s = whole(raw);
    if (s.stage === 'done') return [];
    const ti = teamOf(s, who);
    if (ti < 0) return [];
    const it = holding(s.teams[ti].floor, who);
    if (!it) return [];
    const o = orderOf(s, it.data);
    const step = STEPS[it.step];
    // a simulated hand gets it right (the simulator decides when it slips)
    if (step === 'pick') return [{ kind: 'pick', item: it.id, choice: o.toy }];
    if (step === 'wrap') return [{ kind: 'wrap', item: it.id, choice: o.paper }];
    if (step === 'label') return [{ kind: 'label', item: it.id, choice: o.tag }];
    if (it.data.stowaway !== null) return [{ kind: 'shoo', item: it.id }];
    return [isRight(o, it.data) ? { kind: 'load', item: it.id } : { kind: 'back', item: it.id }];
  },

  projectForPlayer(raw, pid, now) {
    const s = whole(raw);
    const ti = teamOf(s, pid);
    const mine = ti >= 0 ? holding(s.teams[ti].floor, pid) : null;
    const job = mine && {
      item: mine.id, step: STEPS[mine.step], order: (({ options: _o, ...rest }) => rest)(orderOf(s, mine.data)),
      options: orderOf(s, mine.data).options, gift: mine.data,
    };
    return { ...publicView(s, now), myTeam: ti >= 0 ? s.teams[ti].teamId : null, myStation: ti >= 0 ? s.teams[ti].line.workers.findIndex((w) => w.includes(pid)) : -1, job };
  },
  projectForTv: (raw, now) => publicView(whole(raw), now),
  projectForHost: (raw, now) => publicView(whole(raw), now),
};

/** Both factories as the TV draws them: the line, who works where, where each gift is. */
function publicView(s: ConveyorState, _now: number): ConveyorView {
  return {
    stage: s.stage, endsAt: s.endsAt, quota: s.cfg.quota, winner: s.winner ?? null, steps: STEPS,
    teams: s.teams.map((t) => ({
      teamId: t.teamId, captainId: t.captainId, members: t.members, delivered: t.delivered, wrong: t.wrong,
      stations: t.line.stations.map((steps, i) => ({ steps: steps.map((k) => STEPS[k]), workers: t.line.workers[i] })),
      gifts: Object.values(t.floor.items).map((it) => ({ id: it.id, step: it.step, station: stationOf(t.line, it.step), working: it.holder !== null, stowaway: it.data.stowaway })),
    })),
  };
}
