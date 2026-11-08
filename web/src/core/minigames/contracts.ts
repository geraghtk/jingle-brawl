// The mini-game platform's contracts (yule_duel_agent_spec_updated.md, M1). A game is a pure
// definition: config, initial state, action validation, a reducer, deadlines, per-viewer views and
// a result. The runtime (runtime.ts) runs any registered definition the same way, inside the room
// server: one action at a time, server deadlines, ids and versions on every action. A game never
// moves gifts or chips: it returns a result, and the duel (handoff.ts) acts on it only when it is
// `resolved` with a nominated winning duelist. Nothing here is about Christmas.
import type { Rng } from '../rng';
import type { PlayerId } from '../types';
import type { CompatibilityManifest } from './stubs';

export type GameCategory =
  | 'real_time_action' | 'strategy_bluffing' | 'creative_funny' | 'team_based' | 'party_social'
  | 'puzzle_brain' | 'physical_coordination' | 'cross_device' | 'asymmetric_cooperative';
export type MatchMode = 'direct_duel' | 'free_for_all' | 'party_with_duelists' | 'captain_teams' | 'cooperative';
export type Lifecycle = 'active' | 'paused' | 'completed' | 'cancelled';
export type ResultStatus = 'resolved' | 'requires_tiebreaker' | 'no_contest' | 'cancelled';

export interface DuelResolution {
  duelists: [PlayerId, PlayerId];     // nominated by the experience
  winningDuelistId?: PlayerId;
  losingDuelistId?: PlayerId;
  resolutionReason?: string;
}

export interface GameResult {
  gameId: string;
  instanceId: string;
  status: ResultStatus;
  mode: MatchMode;
  duel?: DuelResolution;
  winners: PlayerId[];
  losers: PlayerId[];
  placements?: { playerId: PlayerId; rank: number }[];
  performance?: { playerId: PlayerId; score?: number; contribution?: number }[];
  /** Experience rewards the game recommends (e.g. the winning team's share). The game never applies
   *  them: the experience does (here: Party points, in handoff), separately from chips and gifts. */
  rewards?: { playerId: PlayerId; points: number; reason?: string }[];
  metrics?: Record<string, unknown>;
  scoringPolicyVersion?: string;
  reason: string;
}

/** What a definition is told about the match it runs in. */
export interface MatchCtx {
  instanceId: string;
  mode: MatchMode;
  participants: PlayerId[];           // everyone playing
  duelists?: [PlayerId, PlayerId];    // the two whose duel this settles (direct_duel, party_with_duelists)
  connected: (pid: PlayerId) => boolean;
  now: number;
  rng: Rng;
}

/** A server deadline, in server time. The runtime keeps them; a pause stores what's left. */
export interface Deadline { id: string; at: number }

/** What a step of the game produces. `deadlines` replaces the pending ones (omit: unchanged).
 *  State is stored like the rest of the game: empty objects vanish, so read it defensively. */
export interface Transition<S> {
  state: S;
  events?: string[];
  deadlines?: Deadline[];
  result?: GameResult;                // set once: the game is over
}

export type Check<A> = { ok: true; action: A } | { ok: false; error: string };

export interface GameDefinition<C = unknown, S = unknown, A = unknown> {
  id: string;
  version: string;
  label: string;
  emoji: string;
  rules: string;
  categories: GameCategory[];
  supportedModes: MatchMode[];
  compatibility: CompatibilityManifest;
  validateConfig(raw: unknown): C;
  /** With gift stakes, may this mode always produce one winning duelist (directly or by its tiebreak)? */
  canAlwaysResolve(mode: MatchMode, config: C, players: number): { ok: true } | { ok: false; reason: string };
  createInitialState(ctx: MatchCtx, config: C): Transition<S>;
  /** Shape, actor and timing checks for an action from `who` (never trust a client's score or result). */
  validateAction(ctx: MatchCtx, state: S, who: PlayerId, action: unknown): Check<A>;
  reduce(ctx: MatchCtx, state: S, who: PlayerId, action: A): Transition<S>;
  onDeadline(ctx: MatchCtx, state: S, deadlineId: string): Transition<S>;
  /** Example actions `who` could take now (the simulator and the table's offers press these). */
  offers(ctx: MatchCtx, state: S, who: PlayerId): A[];
  projectForPlayer(state: S, pid: PlayerId, now: number): unknown;
  projectForTv(state: S, now: number): unknown;
  projectForHost(state: S, now: number): unknown;
}
