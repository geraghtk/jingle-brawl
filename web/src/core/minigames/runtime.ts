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
}

/** What each screen is sent instead of the instance (projection.ts). */
export interface ArenaView {
  instanceId: string; gameId: string; mode: MatchMode; lifecycle: Lifecycle; stateVersion: number;
  participants: PlayerId[]; duelists?: [PlayerId, PlayerId]; result?: GameResult; view: unknown;
}

const RECENT = 64;
export const arenaOf = (game: Game) => game.arena as ArenaInstance | null | undefined;

function ctxFor(game: Game, a: Pick<ArenaInstance, 'instanceId' | 'mode' | 'participants' | 'duelists'>, now: number, rng: Rng): MatchCtx {
  return { instanceId: a.instanceId, mode: a.mode, participants: a.participants, duelists: a.duelists, now, rng,
    connected: (pid) => game.players?.[pid]?.connected !== false };
}

/** Apply a transition: the new state, deadlines, result, events, one version up. */
function commit(tx: Tx, a: ArenaInstance, tr: Transition<unknown>, extra: Partial<ArenaInstance> = {}) {
  const next: ArenaInstance = {
    ...a, ...extra, state: tr.state, stateVersion: a.stateVersion + 1,
    deadlines: tr.result ? [] : tr.deadlines ?? a.deadlines,
    ...(tr.result ? { result: tr.result, lifecycle: tr.result.status === 'cancelled' ? 'cancelled' : 'completed' } : {}),
  };
  tx.set('arena', next);
  for (const e of tr.events || []) tx.event(e, 'duel');
}

/** Start a game inside the caller's transaction. With duelists (gift stakes) the mode must always resolve them. */
export function startArena(tx: Tx, gameId: string, o: { mode: MatchMode; participants: PlayerId[]; duelists?: [PlayerId, PlayerId]; config?: unknown },
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
  commit(tx, a, tr);
  tx.set('currentTurn/phase', 'minigame_arena').event(`${def.emoji} ${def.label}!`, 'duel');
  return {};
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
  commit(tx, a, def.reduce(mctx, a.state, who, check.action), msg.aid ? { recent: [...a.recent, msg.aid].slice(-RECENT) } : {});
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
  commit(tx, a, def.onDeadline(ctxFor(game, a, ctx.now, ctx.rng), a.state, deadlineId));
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
  return new Tx(game).set('arena', { ...rest, lifecycle: 'active', deadlines, stateVersion: a.stateVersion + 1,
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
  const view = !def ? null : viewer.kind === 'host' ? def.projectForHost(a.state, now) : viewer.kind === 'tv' ? def.projectForTv(a.state, now) : def.projectForPlayer(a.state, viewer.pid, now);
  return { instanceId: a.instanceId, gameId: a.gameId, mode: a.mode, lifecycle: a.lifecycle, stateVersion: a.stateVersion,
    participants: a.participants, ...(a.duelists ? { duelists: a.duelists } : {}), ...(a.result ? { result: a.result } : {}), view };
}
