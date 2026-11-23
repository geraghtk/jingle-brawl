// The mini-game runtime: runs any registered game the same way. The running game lives at
// game.arena, which never leaves the server (each screen gets game.arenaView instead, made by the
// game's own projectFor*). Every change is one plan (one commit): an action from a participant,
// a deadline, or the host's pause / resume / cancel / no-contest (logged as overrides).
// When the game ends, handoff() returns to the duel: only a `resolved` result with a nominated
// winning duelist names a winner; anything else gives the duel back to the host to decide.
import type { Rng } from '../rng';
import { refuse, Tx } from '../tx';
import type { Game, Plan, PlayerId } from '../types';
import type { Deadline, GameDefinition, GameResult, Lifecycle, MatchCtx, MatchMode, Transition } from './contracts';
import { getDef } from './registry';

export interface ArenaInstance {
  instanceId: string;
  gameId: string;
  defVersion: string;
  mode: MatchMode;
  participants: PlayerId[];
  duelists?: [PlayerId, PlayerId];
  config: unknown;
  lifecycle: Lifecycle;
  stateVersion: number;
  state: unknown;
  deadlines: Deadline[];
  /** While paused: what was left on each deadline. */
  paused?: { at: number; left: Record<string, number> };
  /** Recent action ids: a resent action is answered, not replayed. */
  recent: string[];
  result?: GameResult;
  overrides: { at: number; what: string }[];
  startedAt: number;
  /** When the game itself ended (not when the host stopped it): its ending stays on the screens for RESULT_HOLD_MS. */
  endedAt?: number;
  /** Before it starts (lifecycle `briefing`): who has said they are ready, and when the 3, 2, 1 ends. */
  briefing?: { ready: Record<PlayerId, true>; goAt?: number };
}

/** Once everyone is ready: 3, 2, 1 and the game starts. */
export const READY_COUNT_MS = 3000;

/** A finished game's ending stays up this long before the duel takes over (the host can move on sooner). */
export const RESULT_HOLD_MS = 6000;

/** What each screen is sent instead of the instance (projection.ts). */
export interface ArenaView {
  instanceId: string; gameId: string; mode: MatchMode; lifecycle: Lifecycle; stateVersion: number;
  /** Paused: the game's clock stands at this moment (a screen that draws from the time uses it as "now"). */
  pausedAt?: number;
  /** The game ended by itself at this moment: its ending is on screen until the duel takes over. */
  endedAt?: number;
  participants: PlayerId[]; duelists?: [PlayerId, PlayerId]; result?: GameResult; view: unknown;
  /** The instructions are up: who is ready so far, and (once everyone is) when the countdown ends. */
  briefing?: { ready: PlayerId[]; goAt?: number };
}

const RECENT = 64;
export const arenaOf = (game: Game) => game.arena as ArenaInstance | null | undefined;

function ctxFor(game: Game, a: Pick<ArenaInstance, 'instanceId' | 'mode' | 'participants' | 'duelists'>, now: number, rng: Rng): MatchCtx {
  return { instanceId: a.instanceId, mode: a.mode, participants: a.participants, duelists: a.duelists, now, rng,
    connected: (pid) => game.players?.[pid]?.connected !== false };
}

/** Apply a transition: the new state, deadlines, result, events, one version up. */
function commit(tx: Tx, a: ArenaInstance, tr: Transition<unknown>, now: number, extra: Partial<ArenaInstance> = {}) {
  const next: ArenaInstance = {
    ...a, ...extra, state: tr.state, stateVersion: a.stateVersion + 1,
    deadlines: tr.result ? [] : tr.deadlines ?? a.deadlines,
    ...(tr.result ? { result: tr.result, lifecycle: tr.result.status === 'cancelled' ? 'cancelled' : 'completed', endedAt: now } : {}),
  };
  tx.set('arena', next);
  for (const e of tr.events || []) tx.event(e, 'duel');
}

/** Start a game inside the caller's transaction. With duelists (gift stakes) the mode must always resolve them. */
export function startArena(tx: Tx, gameId: string, o: { mode: MatchMode; participants: PlayerId[]; duelists?: [PlayerId, PlayerId]; config?: unknown; briefing?: boolean },
  ctx: { now: number; rng: Rng }): { error?: string } {
  const def = getDef(gameId);
  if (!def) return { error: `no game ${gameId}` };
  if (!def.supportedModes.includes(o.mode)) return { error: `${def.label} can't be played as ${o.mode.replace(/_/g, ' ')}` };
  const config = def.validateConfig(o.config ?? {});
  if (o.duelists) {
    const can = def.canAlwaysResolve(o.mode, config, o.participants.length);
    if (!can.ok) return { error: `${def.label} can't settle this duel: ${can.reason}` };
  }
  const instanceId = `${gameId}-${ctx.now.toString(36)}-${Math.floor(ctx.rng() * 1e6).toString(36)}`;
  const base = { instanceId, mode: o.mode, participants: o.participants, duelists: o.duelists };
  const tr = def.createInitialState(ctxFor(tx.game, base, ctx.now, ctx.rng), config);
  const a: ArenaInstance = {
    ...base, gameId, defVersion: def.version, config, lifecycle: 'active', stateVersion: 0, state: tr.state,
    deadlines: tr.deadlines || [], recent: [], overrides: [], startedAt: ctx.now,
  };
  commit(tx, a, tr, ctx.now);
  // the instructions first: the game is made but holds still (like a pause) until everyone is ready
  const made = tx.get('arena') as ArenaInstance;
  if (o.briefing && made.lifecycle === 'active') {
    const left = Object.fromEntries(made.deadlines.map((d) => [d.id, Math.max(0, d.at - ctx.now)]));
    const ready = Object.fromEntries(o.participants.filter((id) => tx.game.players?.[id]?.isBot).map((id) => [id, true as const]));
    const waiting: ArenaInstance = { ...made, lifecycle: 'briefing', paused: { at: ctx.now, left }, deadlines: [], briefing: { ready } };
    tx.set('arena', awaited(tx.game, waiting).length ? waiting : { ...waiting, briefing: { ready, goAt: ctx.now + READY_COUNT_MS } });
  }
  tx.set('currentTurn/phase', 'minigame_arena').event(`${def.emoji} ${def.label}!`, 'duel');
  return {};
}

/** Who the room is still waiting for before the game starts (a bot is always ready; a dropped phone is not waited for). */
export function awaited(game: Game, a: ArenaInstance): PlayerId[] {
  if (a.lifecycle !== 'briefing') return [];
  return a.participants.filter((id) => { const p = game.players?.[id]; return !!p && !p.isBot && p.connected !== false && !a.briefing?.ready?.[id]; });
}

/** A participant says they are ready; the last one starts the countdown. */
export function arenaReady(game: Game, who: PlayerId, now: number): Plan {
  const a = arenaOf(game);
  if (a?.lifecycle !== 'briefing') return refuse('The game has started');
  if (!a.participants.includes(who)) return refuse("You're not in this game");
  if (a.briefing?.ready?.[who]) return refuse('Already ready');
  const next: ArenaInstance = { ...a, stateVersion: a.stateVersion + 1, briefing: { ...a.briefing, ready: { ...(a.briefing?.ready || {}), [who]: true } } };
  if (!next.briefing!.goAt && !awaited(game, next).length) next.briefing!.goAt = now + READY_COUNT_MS;
  return new Tx(game).set('arena', next).plan();
}

/** Start the countdown without waiting any longer: the host's button, and the room's own step when nobody is left to wait for. */
export function arenaCountdown(game: Game, now: number, byHost = false): Plan {
  const a = arenaOf(game);
  if (a?.lifecycle !== 'briefing') return refuse('The game has started');
  if (a.briefing?.goAt) return refuse('Already counting down');
  if (!byHost && awaited(game, a).length) return refuse('Still waiting for someone');
  return new Tx(game).set('arena', { ...a, stateVersion: a.stateVersion + 1, briefing: { ready: a.briefing?.ready || {}, goAt: now + READY_COUNT_MS },
    ...(byHost ? { overrides: [...a.overrides, { at: now, what: 'started without waiting' }] } : {}) }).plan();
}

/** The countdown is over: the game's clock starts now (its times move on by however long the instructions were up). */
export function arenaBegin(game: Game, now: number): Plan {
  const a = arenaOf(game);
  if (a?.lifecycle !== 'briefing' || !a.paused) return refuse('The game has started');
  if (!a.briefing?.goAt || a.briefing.goAt > now) return refuse('Not yet');
  const { paused, briefing: _b, ...rest } = a;
  void _b;
  const def = getDef(a.gameId);
  const state = def?.shiftTime ? def.shiftTime(a.state, Math.max(0, now - paused.at)) : a.state;
  const deadlines = Object.entries(paused.left || {}).map(([id, ms]) => ({ id, at: now + ms }));   // (stored state drops an empty map)
  return new Tx(game).set('arena', { ...rest, state, lifecycle: 'active', deadlines, stateVersion: a.stateVersion + 1, startedAt: now }).plan();
}

/** A participant's action: actor, lifecycle, version and id checked; the game validates the rest. */
export function arenaAct(game: Game, who: PlayerId, msg: { action: unknown; aid?: string; v?: number }, ctx: { now: number; rng: Rng }): Plan {
  const a = arenaOf(game);
  const def = getDef(a?.gameId);
  if (!a || !def) return refuse('No game running');
  if (a.lifecycle !== 'active') return refuse(a.lifecycle === 'paused' ? 'The game is paused' : 'The game is over');
  if (!a.participants.includes(who)) return refuse("You're not in this game");
  if (msg.aid && a.recent.includes(msg.aid)) return refuse('Already done');
  if (msg.v != null && msg.v !== a.stateVersion) return refuse('The game moved on: try again');
  const mctx = ctxFor(game, a, ctx.now, ctx.rng);
  const check = def.validateAction(mctx, a.state, who, msg.action);
  if (!check.ok) return refuse(check.error);
  const tx = new Tx(game);
  commit(tx, a, def.reduce(mctx, a.state, who, check.action), ctx.now, msg.aid ? { recent: [...a.recent, msg.aid].slice(-RECENT) } : {});
  return tx.plan();
}

/** Can `who` act at all right now (the table offers it)? The game's own checks come with the action. */
export function canAct(game: Game, who: PlayerId): boolean {
  const a = arenaOf(game);
  return !!a && a.lifecycle === 'active' && a.participants.includes(who);
}

/** Example actions for `who` (the table's offers, the simulator's presses). */
export function arenaOffers(game: Game, who: PlayerId, now: number, rng: Rng): unknown[] {
  const a = arenaOf(game);
  const def = getDef(a?.gameId);
  if (!a || !def || !canAct(game, who)) return [];
  return def.offers(ctxFor(game, a, now, rng), a.state, who);
}

/** A deadline came due (the server's clock): the game decides what happens. */
export function arenaDeadline(game: Game, deadlineId: string, ctx: { now: number; rng: Rng }): Plan {
  const a = arenaOf(game);
  const def = getDef(a?.gameId);
  if (!a || !def || a.lifecycle !== 'active') return refuse('No game running');
  const d = a.deadlines.find((x) => x.id === deadlineId);
  if (!d || d.at > ctx.now) return refuse('Not due');
  const tx = new Tx(game);
  commit(tx, a, def.onDeadline(ctxFor(game, a, ctx.now, ctx.rng), a.state, deadlineId), ctx.now);
  return tx.plan();
}

// --- the host -----------------------------------------------------------------------------------

export function arenaPause(game: Game, now: number): Plan {
  const a = arenaOf(game);
  if (a?.lifecycle !== 'active') return refuse('Nothing to pause');
  const left = Object.fromEntries(a.deadlines.map((d) => [d.id, Math.max(0, d.at - now)]));
  return new Tx(game).set('arena', { ...a, lifecycle: 'paused', paused: { at: now, left }, deadlines: [], stateVersion: a.stateVersion + 1,
    overrides: [...a.overrides, { at: now, what: 'paused' }] }).event('⏸️ The host paused the game', 'duel').plan();
}

export function arenaResume(game: Game, now: number): Plan {
  const a = arenaOf(game);
  if (a?.lifecycle !== 'paused' || !a.paused) return refuse('Not paused');
  const deadlines = Object.entries(a.paused.left).map(([id, ms]) => ({ id, at: now + ms }));
  const { paused: _gone, ...rest } = a;
  void _gone;
  // a game settled by the clock keeps times in its state: they move on by the length of the pause too
  const def = getDef(a.gameId);
  const state = def?.shiftTime ? def.shiftTime(a.state, Math.max(0, now - a.paused.at)) : a.state;
  return new Tx(game).set('arena', { ...rest, state, lifecycle: 'active', deadlines, stateVersion: a.stateVersion + 1,
    overrides: [...a.overrides, { at: now, what: 'resumed' }] }).event('▶️ The host resumed the game', 'duel').plan();
}

/** The host ends it with no result: the duel goes back to the host, nothing is won or lost. */
export function arenaStop(game: Game, status: 'cancelled' | 'no_contest', now: number): Plan {
  const a = arenaOf(game);
  if (!a || a.lifecycle === 'completed' || a.lifecycle === 'cancelled') return refuse('Nothing to stop');
  const reason = status === 'cancelled' ? 'the host cancelled the game' : 'the host called it a no contest';
  const result: GameResult = { gameId: a.gameId, instanceId: a.instanceId, status, mode: a.mode, winners: [], losers: [], reason,
    ...(a.duelists ? { duel: { duelists: a.duelists, resolutionReason: reason } } : {}) };
  return new Tx(game).set('arena', { ...a, lifecycle: status === 'cancelled' ? 'cancelled' : 'completed', deadlines: [], result, stateVersion: a.stateVersion + 1,
    overrides: [...a.overrides, { at: now, what: status }] }).event(`⏹️ ${reason[0].toUpperCase()}${reason.slice(1)}`, 'duel').plan();
}

/** The game is over: back to the duel. Only a resolved result with a nominated winning duelist names a winner. */
export function handoff(game: Game): Plan {
  const a = arenaOf(game);
  const t = game.currentTurn;
  if (!a?.result || !t) return refuse('The game is not over');
  const tx = new Tx(game).claim('currentTurn/arenaHandedOff', a.instanceId).set('arena', null);
  // the game's reward recommendation, as Party points (once: the claim above guards a second hand-off)
  const paid: string[] = [];
  for (const r of a.result.rewards || []) {
    const p = game.players?.[r.playerId];
    if (p && Number.isFinite(r.points) && r.points > 0) {
      tx.set(`players/${r.playerId}/partyPoints`, (p.partyPoints || 0) + Math.round(r.points));
      paid.push(`${p.name || 'Someone'} +${Math.round(r.points)}`);
    }
  }
  if (paid.length) tx.event(`⭐ Party points: ${paid.join(', ')}`, 'duel');
  if (t.phase !== 'minigame_arena') return tx.plan();                       // a standalone round: nothing to settle
  tx.set('currentTurn/phase', 'duel');
  const w = a.result.duel?.winningDuelistId;
  if (a.result.status === 'resolved' && w && a.duelists?.includes(w)) tx.set('currentTurn/duelGameWinner', w);
  else tx.set('currentTurn/duelGameOutcome', { status: 'no_contest', reason: a.result.reason || 'the game could not decide' });
  return tx.plan();
}

/** The view a screen gets: the game's own projection for that viewer. */
export function arenaView(a: ArenaInstance, viewer: { kind: 'host' } | { kind: 'tv' } | { kind: 'phone'; pid: PlayerId }, now: number): ArenaView {
  const def = getDef(a.gameId) as GameDefinition;
  // while paused the game's clock stands still at the moment of the pause: the view is as it was then,
  // and says so (`pausedAt`), so a screen that draws from the time (a belt, a punch) holds still too
  const pausedAt = (a.lifecycle === 'paused' || a.lifecycle === 'briefing') && a.paused ? a.paused.at : null;
  const at = pausedAt ?? now;
  const view = !def ? null : viewer.kind === 'host' ? def.projectForHost(a.state, at) : viewer.kind === 'tv' ? def.projectForTv(a.state, at) : def.projectForPlayer(a.state, viewer.pid, at);
  return { instanceId: a.instanceId, gameId: a.gameId, mode: a.mode, lifecycle: a.lifecycle, stateVersion: a.stateVersion, ...(pausedAt != null ? { pausedAt } : {}),
    ...(a.endedAt != null ? { endedAt: a.endedAt } : {}),
    ...(a.lifecycle === 'briefing' ? { briefing: { ready: Object.keys(a.briefing?.ready || {}), ...(a.briefing?.goAt ? { goAt: a.briefing.goAt } : {}) } } : {}),
    participants: a.participants, ...(a.duelists ? { duelists: a.duelists } : {}), ...(a.result ? { result: a.result } : {}), view };
}
