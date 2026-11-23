// Elf Tower Wars (spec game #9). Two workshops face each other across three lanes. A team sends toys
// down a lane at the other workshop; the armies pass each other, and only towers, a thrown snowball and
// a raised shield stop them. A workshop at nothing loses; else the healthier one at the bell. Mode
// `captain_teams` only: the two duelists captain the teams (dealt by engines/teams.ts) and play like
// everyone else. There are no jobs (there were: a builder, an attacker, a defender; Kevin, 2026-11-22:
// the builder mostly waited): every player has every move (towers on the team's six spots, toys down a
// lane, a snowball, a shield, glue) and their own energy to pay for them, refilling by the clock, so
// what to spend it on is the game and nobody can spend a teammate's share. A team earns the same
// energy however many it is (`teamEnergy`, shared among the phones that are here: a dropped phone's
// share goes to the rest). The battle is
// engines/lanes.ts: a fixed-step simulation the room advances lazily, waking a few times a second
// while toys are on the field.
import type { PlayerId } from '../../types';
import type { GameDefinition, GameResult, MatchCtx, Transition } from '../contracts';
import { advance, arm, armed, build, busy, DEFAULT_LANES, LANES_VERSION, newField, patch, send, shield, shiftField, wholeField, zap } from '../engines/lanes';
import type { Did, Field, LaneParams, Purse } from '../engines/lanes';
import { assignTeams, checkAssignment } from '../engines/teams';
import type { TeamAssignment } from '../engines/teams';
import { energyOf } from './view';
import type { TowerView } from './view';

export interface TowerConfig {
  roundMs: number;
  overtimeMs: number;
  /** How often the room wakes while toys are on the field (the screens glide in between). */
  tickMs: number;
  /** Party points the game recommends for each player on the winning team (the losers: none). */
  rewardBase: number;
  /** Toys get sturdier as the battle goes on (so it cannot end in a stand-off): this much more for each minute gone. */
  toughen: number;
  /** Energy a whole team earns each second, shared among its phones (so three are no richer than two); a big
   *  table earns more, so nobody's share falls under `minShare`. `teamStart`: what a team begins with, shared the same way. */
  teamEnergy: number; minShare: number; teamStart: number;
  /** The finale: for the last `rushMs` (and any overtime) energy comes `rush` times as fast. */
  rushMs: number; rush: number;
  rules: LaneParams;
  /** The teams, when the experience dealt them (duelgame / the lab); else dealt here. */
  teams?: TeamAssignment;
}
export const DEFAULTS: TowerConfig = { roundMs: 120_000, overtimeMs: 20_000, tickMs: 400, rewardBase: 1, toughen: 1.1, teamEnergy: 3, minShare: 0.6, teamStart: 12, rushMs: 30_000, rush: 2, rules: DEFAULT_LANES };

export interface TeamState {
  teamId: 'A' | 'B'; captainId: PlayerId; members: PlayerId[];
  purses: Record<PlayerId, Purse>;
  /** Who is earning, and how much a second each (see `present`). */
  active: PlayerId[];
  rate: number;
  credit: Record<PlayerId, number>;
}
export interface TowerState {
  cfg: TowerConfig;
  stage: 'running' | 'overtime' | 'done';
  startedAt: number; endsAt: number;
  rushAt: number;
  field: Field;
  teams: [TeamState, TeamState];
  winner?: 'A' | 'B';
}

export type TowerAction =
  | { kind: 'build'; lane: number; slot: number; tower: string } | { kind: 'send'; lane: number; unit: string }
  | { kind: 'zap'; lane: number } | { kind: 'shield'; lane: number } | { kind: 'patch' } | { kind: 'combo'; lane: number };
export const costOf = (a: TowerAction, p: LaneParams): number =>
  a.kind === 'build' ? p.towers[a.tower]?.cost ?? Infinity : a.kind === 'send' ? p.units[a.unit]?.cost ?? Infinity : a.kind === 'zap' ? p.zapCost : a.kind === 'shield' ? p.shieldCost : a.kind === 'combo' ? p.comboCost : p.patchCost;

/** Stored state loses empty lists and maps: put them back before reading. */
function whole(s: TowerState): TowerState {
  return {
    ...s, field: wholeField(s.field, s.cfg.rules),
    teams: s.teams.map((t): TeamState => ({ ...t, members: t.members || [], purses: t.purses || {}, active: t.active || [], rate: t.rate || 0, credit: t.credit || {} })) as [TeamState, TeamState],
  };
}
/** A player's energy at `now` (the finale counted in). */
const energyNow = (s: Pick<TowerState, 'cfg' | 'rushAt'>, t: TeamState, pid: PlayerId, now: number) => energyOf(t, pid, now, s.cfg.rules, { rushAt: s.rushAt, rush: s.cfg.rush });
/** What a whole team earns a second (the same for both, whatever their sizes). */
const teamRate = (s: Pick<TowerState, 'cfg' | 'teams'>) => Math.max(s.cfg.teamEnergy, s.cfg.minShare * Math.max(...s.teams.map((t) => (t.members || []).length)));
/** The team's energy goes to whoever is here: when a phone drops out or comes back, everyone's energy is
 *  banked as it stands and the rate is shared out again (nobody here at all: as if everyone were). */
function present(ctx: MatchCtx, s: TowerState): TowerState {
  let changed = false;
  const teams = s.teams.map((t): TeamState => {
    const here = t.members.filter((m) => ctx.connected(m));
    const active = here.length ? here : t.members;
    if (active.length === t.active.length && active.every((m) => t.active.includes(m))) return t;
    changed = true;
    const at = Math.min(ctx.now, s.endsAt);
    return { ...t, active, rate: teamRate(s) / active.length, purses: Object.fromEntries(t.members.map((m) => [m, { energy: energyNow(s, t, m, at), asOf: at }])) };
  });
  return changed ? { ...s, teams: teams as [TeamState, TeamState] } : s;
}
/** How much sturdier a toy sent at `now` is than its kind. */
export const toughAt = (s: Pick<TowerState, 'cfg' | 'startedAt'>, now: number) => 1 + (s.cfg.toughen * Math.max(0, now - s.startedAt)) / 60_000;
const teamOf = (s: TowerState, pid: PlayerId) => s.teams.findIndex((t) => t.members.includes(pid));

function finish(ctx: MatchCtx, s: TowerState, winner: 'A' | 'B' | null, reason: string): Transition<TowerState> {
  const done: TowerState = { ...s, stage: 'done', ...(winner ? { winner } : {}) };
  const [a, b] = s.field.sides;
  const metrics = {
    health: { A: a.hp, B: b.hp }, sent: { A: a.sent, B: b.sent }, stopped: { A: a.stopped, B: b.stopped }, towers: { A: a.towers.length, B: b.towers.length },
    teams: s.teams.map((t) => ({ teamId: t.teamId, captainId: t.captainId, members: t.members })), lanes: LANES_VERSION,
  };
  const performance = s.teams.flatMap((t) => t.members.map((playerId) => ({ playerId, contribution: Math.round((t.credit[playerId] || 0) * 100) / 100 })));
  const base = { gameId: towers.id, instanceId: ctx.instanceId, mode: ctx.mode, performance, metrics, scoringPolicyVersion: 'captain_team@2', reason };
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

/** The bell, and a wake-up while anything is walking. */
function deadlines(s: TowerState, now: number) {
  const tick = busy(s.field) ? Math.max(now, s.field.asOf) + s.cfg.tickMs : Infinity;
  return [{ id: 'end', at: s.endsAt }, ...(tick < s.endsAt ? [{ id: 'tick', at: tick }] : [])];
}

/** The battle brought up to `now` (never past the bell), and what follows: a workshop down, the bell, overtime. */
function settle(ctx: MatchCtx, s0: TowerState): { state: TowerState; over?: Transition<TowerState> } {
  const s = present(ctx, s0);
  const now = Math.min(ctx.now, s.endsAt);
  const state: TowerState = { ...s, field: advance(s.field, now, s.cfg.rules) };
  const [a, b] = state.field.sides;
  if (a.hp <= 0 || b.hp <= 0) {
    if (a.hp <= 0 && b.hp <= 0) return { state, over: finish(ctx, state, null, 'both workshops fell in the same moment') };
    return { state, over: finish(ctx, state, a.hp <= 0 ? 'B' : 'A', 'the other workshop fell') };
  }
  if (ctx.now < s.endsAt) return { state };
  // the bell: the healthier workshop; level: a little longer, once; then whoever stopped more toys; then the host
  if (a.hp !== b.hp) return { state, over: finish(ctx, state, a.hp > b.hp ? 'A' : 'B', s.stage === 'overtime' ? 'the stronger workshop after overtime' : 'the stronger workshop at the bell') };
  if (s.stage === 'running') return { state: { ...state, stage: 'overtime', endsAt: ctx.now + s.cfg.overtimeMs } };
  if (a.stopped !== b.stopped) return { state, over: finish(ctx, state, a.stopped > b.stopped ? 'A' : 'B', 'level after overtime: stopped more toys') };
  return { state, over: finish(ctx, state, null, 'level after overtime') };
}

function parse(raw: unknown): TowerAction | null {
  const a = raw as Record<string, unknown> | null;
  const int = (v: unknown): v is number => Number.isInteger(v) && (v as number) >= 0 && (v as number) < 16;
  if (a?.kind === 'build') return int(a.lane) && int(a.slot) && typeof a.tower === 'string' ? { kind: 'build', lane: a.lane, slot: a.slot, tower: a.tower } : null;
  if (a?.kind === 'send') return int(a.lane) && typeof a.unit === 'string' ? { kind: 'send', lane: a.lane, unit: a.unit } : null;
  if (a?.kind === 'zap' || a?.kind === 'shield' || a?.kind === 'combo') return int(a.lane) ? { kind: a.kind, lane: a.lane } : null;
  if (a?.kind === 'patch') return { kind: 'patch' };
  return null;
}
/** One action on a field that is already brought up to `now`. */
function apply(f: Field, side: number, a: TowerAction, now: number, p: LaneParams, tough = 1, who = '', alone = false): Did {
  switch (a.kind) {
    case 'combo': return arm(f, side, a.lane, who, now, p, alone);
    case 'build': return build(f, side, a.lane, a.slot, a.tower, now, p);
    case 'send': return send(f, side, a.lane, a.unit, now, p, tough);
    case 'zap': return zap(f, side, a.lane, now, p);
    case 'shield': return shield(f, side, a.lane, now, p);
    case 'patch': return patch(f, side, now, p);
  }
}

export const towers: GameDefinition<TowerConfig, TowerState, TowerAction> = {
  id: 'elf_tower_wars', version: '1', label: 'Elf Tower Wars', emoji: '🏰',
  rules: 'The duellists captain two teams, each with a workshop on the TV. Everyone has their own energy and spends it how they like: toys marching at the other workshop, towers to stop theirs (build one again to make it better), a snowball, a shield, glue, and an avalanche that takes two of you at once. Energy doubles for the last half minute. Knock their workshop down, or have the stronger one at the bell.',
  categories: ['team_based', 'cross_device', 'real_time_action'],
  supportedModes: ['captain_teams'],
  compatibility: {
    version: 1, environments: ['in_person', 'remote', 'hybrid'], displayRole: 'required_shared_world', topology: ['phone_to_shared_display'],
    devices: [{ feature: 'touch', required: false, fallback: 'buttons work with a mouse or keyboard' }], players: { min: 2, max: 40 }, modes: ['captain_teams'],
  },

  validateConfig(raw) {
    const r = (raw && typeof raw === 'object' ? raw : {}) as Partial<TowerConfig>;
    const c: TowerConfig = { ...DEFAULTS, ...r, rules: { ...DEFAULT_LANES, ...(r.rules || {}) } };
    const p = c.rules;
    const ok = c.roundMs > 0 && c.overtimeMs > 0 && c.tickMs >= 100 && p.lanes >= 1 && p.lanes <= 5 && p.stepMs >= 20 && p.baseHp > 0 && p.slots.length >= 1 && p.slots.every((q) => q > 0 && q < 1)
      && Object.values(p.units).every((u) => u.cost > 0 && u.hp > 0 && u.speed > 0 && u.damage > 0) && Object.values(p.towers).every((t) => t.cost > 0 && t.range > 0 && t.slow > 0 && t.slow <= 1)
      && p.purse.max > 0 && c.rush >= 1 && c.rushMs >= 0 && p.maxLevel >= 1 && c.teamEnergy > 0 && c.minShare >= 0 && c.teamStart >= 0;
    if (!ok) throw new Error('bad Elf Tower Wars settings');
    return c;
  },

  // overtime, then toys stopped, settle nearly every tie; a dead level goes to the host
  canAlwaysResolve: (mode) => (mode === 'captain_teams' ? { ok: true } : { ok: false, reason: 'captains and teams only' }),

  createInitialState(ctx, c) {
    if (!ctx.duelists) throw new Error('Elf Tower Wars needs two captains');
    const deal = checkAssignment(c.teams, ctx.duelists) || assignTeams(ctx.duelists, ctx.participants, ctx.rng);
    const state: TowerState = {
      cfg: c, stage: 'running', startedAt: ctx.now, endsAt: ctx.now + c.roundMs, rushAt: ctx.now + c.roundMs - c.rushMs, field: newField(ctx.now, c.rules),
      teams: deal.teams.map((t): TeamState => ({ teamId: t.teamId as 'A' | 'B', captainId: t.captainId, members: t.memberIds, purses: {}, active: [], rate: 0, credit: {} })) as [TeamState, TeamState],
    };
    // everyone starts with their share of the team's opening energy; who is here earns (`present`)
    const big = Math.max(...state.teams.map((t) => t.members.length));
    for (const t of state.teams) {
      const here = t.members.filter((m) => ctx.connected(m));
      t.active = here.length ? here : t.members;
      t.rate = teamRate(state) / t.active.length;
      const start = Math.min(c.rules.purse.max, Math.max(c.teamStart, c.rules.purse.start * big) / t.members.length);
      t.purses = Object.fromEntries(t.members.map((m) => [m, { energy: start, asOf: ctx.now }]));
    }
    return { state, deadlines: deadlines(state, ctx.now), events: [`🏰 Team ${deal.teams[0].teamId} vs team ${deal.teams[1].teamId}: to your workshops!`] };
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
    const t = now.state.teams[ti], p = s0.cfg.rules;
    if (a.kind === 'build' && s0.stage === 'overtime') return { ok: false, error: 'No building in overtime' };
    if (energyNow(now.state, t, who, ctx.now) + 1e-9 < costOf(a, p)) return { ok: false, error: 'Not enough energy yet' };
    const did = apply(now.state.field, ti, a, ctx.now, p, 1, who, t.active.length < 2);
    return did.error ? { ok: false, error: did.error } : { ok: true, action: a };
  },

  reduce(ctx, raw, who, a) {
    const now = settle(ctx, whole(raw));
    if (now.over) return now.over;
    const s = now.state, ti = teamOf(s, who), t = s.teams[ti], p = s.cfg.rules;
    const have = energyNow(s, t, who, ctx.now), cost = costOf(a, p);
    const did = have + 1e-9 >= cost ? apply(s.field, ti, a, ctx.now, p, toughAt(s, ctx.now), who, t.active.length < 2) : null;
    if (!did?.field) return { state: s, deadlines: deadlines(s, ctx.now) };
    const next: TeamState = { ...t, purses: { ...t.purses, [who]: { energy: have - cost, asOf: ctx.now } }, credit: { ...t.credit, [who]: (t.credit[who] || 0) + cost } };
    const state: TowerState = { ...s, field: did.field, teams: s.teams.map((x, i) => (i === ti ? next : x)) as [TeamState, TeamState] };
    return { state, deadlines: deadlines(state, ctx.now) };
  },

  onDeadline(ctx, raw, id) {
    const s0 = whole(raw);
    if (s0.stage === 'done') return { state: s0 };
    // the bell is its deadline coming due (the room's clock is the judge of that, not this state's note of it)
    const s = id === 'end' ? { ...s0, endsAt: Math.min(s0.endsAt, ctx.now) } : s0;
    const now = settle(ctx, s);
    if (now.over) return now.over;
    return { state: now.state, deadlines: deadlines(now.state, ctx.now), ...(now.state.stage !== s.stage ? { events: ['🏰 Level! Overtime: no more building'] } : {}) };
  },

  shiftTime(raw, ms) {
    const s = whole(raw);
    return { ...s, startedAt: s.startedAt + ms, endsAt: s.endsAt + ms, rushAt: s.rushAt + ms, field: shiftField(s.field, ms, s.cfg.rules),
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

/** The battle as every screen draws it: the field as of its last step (a screen glides each toy on from
 *  there), both workshops, everyone's energy. */
function publicView(s: TowerState, now: number): TowerView {
  return {
    stage: s.stage, startedAt: s.startedAt, endsAt: s.endsAt, rushAt: s.rushAt, rush: s.cfg.rush, winner: s.winner ?? null, rules: s.cfg.rules,
    field: s.stage === 'done' ? s.field : advance(s.field, Math.min(now, s.endsAt), s.cfg.rules),
    teams: s.teams.map((t) => ({ teamId: t.teamId, captainId: t.captainId, members: t.members, purses: t.purses, active: t.active, rate: t.rate })),
  };
}

export type Leaning = 'all' | 'send' | 'build' | 'defend';
/** What a simulated hand would press now. By default it is an all-rounder, as a person is: it minds the workshop
 *  when toys are nearly in, saves for the next tower while the team's towers are behind the clock, and otherwise sends toys. A `lean` makes it a hand that mostly does one thing (the simulator's habits). */
export function handOffers(ctx: MatchCtx, raw: TowerState, who: PlayerId, lean?: Leaning): TowerAction[] {
  {
    const s0 = whole(raw);
    if (s0.stage === 'done') return [];
    const ti = teamOf(s0, who);
    if (ti < 0) return [];
    const now = settle(ctx, s0);
    if (now.over) return [];
    const s = now.state, t = s.teams[ti], p = s.cfg.rules, f = s.field, me = f.sides[ti];
    const have = energyNow(s, t, who, ctx.now);
    const leaning: Leaning = lean || 'all';
    const coming = (lane: number) => f.units.filter((u) => u.side !== ti && u.lane === lane);
    const sends = (): TowerAction[] => {
      const out: TowerAction[] = [];
      for (let lane = 0; lane < p.lanes; lane++) {
        if (ctx.now < me.sendAt[lane]) continue;
        for (const [unit, k] of Object.entries(p.units)) if (k.cost <= have) out.push({ kind: 'send', lane, unit });
      }
      return out;
    };
    const defends = (): TowerAction[] => {
      const out: TowerAction[] = [];
      const near = (lane: number, within: number) => coming(lane).some((u) => 1 - u.pos <= within);
      for (let lane = 0; lane < p.lanes; lane++) {
        if (have >= p.shieldCost && ctx.now >= me.shieldAt && coming(lane).filter((u) => 1 - u.pos <= 0.12).length >= 2) out.push({ kind: 'shield', lane });
        else if (have >= p.zapCost && ctx.now >= me.zapAt && near(lane, leaning === 'all' ? 0.2 : p.zapReach)) out.push({ kind: 'zap', lane });   // (an all-rounder throws only at what is nearly in)
      }
      if (have >= p.patchCost && ctx.now >= me.patchAt && me.hp <= p.baseHp - p.patch) out.push({ kind: 'patch' });
      // the avalanche: join a teammate's at once; start one when a crowd is on our half and somebody is there to join
      if (have >= p.comboCost && ctx.now >= me.comboAt) {
        if (armed(me, ctx.now, p)) { if (me.armBy !== who) out.unshift({ kind: 'combo', lane: me.armLane }); }
        else if (t.active.length >= 2) for (let lane = 0; lane < p.lanes; lane++) if (coming(lane).filter((u) => 1 - u.pos <= p.comboReach).length >= 3) out.push({ kind: 'combo', lane });
      }
      return out;
    };
    /** The next tower worth having: an empty spot first (where toys are coming), then a better one over a cheaper. */
    const wanted = (): { lane: number; slot: number; tower: string; cost: number } | null => {
      if (s.stage !== 'running') return null;
      // one that hits a single toy hard by the workshop, one that hits a crowd further out; then make them better
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
    if (leaning === 'all') {
      const d = defends();
      // the team's towers should keep pace with the clock (one more, or one made better, every few seconds): while
      // they are behind, a hand saves for the next; otherwise it sends toys (so it never waits on a teammate to build)
      const built = me.towers.reduce((n, x) => n + (x.level || 1), 0);
      const w = built < 1 + Math.floor((ctx.now - s.startedAt) / 8000) ? wanted() : null;
      // towers before snowballs (a team that only ever throws never gets its towers up, and is overrun): while saving
      // it still raises a shield or joins an avalanche; with the towers in hand it defends about half the time
      if (w) return w.cost <= have ? [{ kind: 'build', lane: w.lane, slot: w.slot, tower: w.tower }] : d.filter((o) => o.kind === 'shield' || (o.kind === 'combo' && armed(me, ctx.now, p)));
      if (d.length && Math.floor(ctx.now / 700 + t.members.indexOf(who)) % 2 === 0) return d;
      return sends();
    }
    if (leaning === 'build') {
      const w = wanted();
      if (w) return w.cost <= have ? [{ kind: 'build', lane: w.lane, slot: w.slot, tower: w.tower }] : [];   // saving up
      return sends();
    }
    if (leaning === 'defend') {
      const d = defends();
      return d.length ? d : have >= p.purse.max * 0.6 ? sends() : [];                                       // keeps something back
    }
    return sends();
  }
}
