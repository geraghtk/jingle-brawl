import type { DuelGameChoice, WheelSpin } from './duelgame';
// The game state as stored at games/{roomCode} in the Realtime Database.
// Every field is optional where the database may omit it (RTDB drops empty objects/nulls).

export type PlayerId = string;
export type GiftId = string;
export const MISFIT = 'misfit' as const;
export type Owner = PlayerId | typeof MISFIT;

export type GamePhase = 'lobby' | 'main' | 'endgame' | 'ended';
export type DividendRule = 'everyone' | 'defenders' | 'nobody';
// Ties are always settled by a whole-table party game. 'random' / 'draw_off' / 'minigame' are only
// read from older saved games (config() maps them).
export type TiebreakerMethod = 'party_mix' | 'sketch_off' | 'whack_elf' | 'random' | 'draw_off' | 'minigame';
export type DuelTableGames = 'teams' | 'all' | 'off';
export type PartyPrize = 'points_chips' | 'points' | 'none';

export type GremlinChallenges = 'mixed' | 'quick_math' | 'memory_flash' | 'perfect_timing';

export interface Settings {
  publicChips?: boolean;
  startingChips?: number;          // absent = variable (10, or 12 for 11+ players)
  maxReprisalDepth?: number;
  maxBid?: number;
  maxWinnerPayout?: number;
  loserDividendRule?: DividendRule;
  taxThreshold?: number;
  baseCost?: number;
  tiebreakerMethod?: TiebreakerMethod;
  partyPrize?: PartyPrize;
  spectatorBets?: number;          // 0 = off, 1 = each vote stakes 1 chip
  voteWindow?: number;             // seconds the host waits for duel votes (0 = no wait)
  duelGames?: string | string[];   // out-of-game duels the host listed, one per line ("" = none: RTDB drops empty arrays)
  duelAppGames?: import('./duelgame').DuelApp[];  // in-app mini-games that may come up as the duel
  duelPick?: 'defender' | 'wheel' | 'off';
  /** Which whole-table games come up for a regular duel when the host ticked none themselves: the team games
   *  (the default: everyone is on a duelist's team), every whole-table game, or none. */
  duelTableGames?: DuelTableGames;
  /** Pass the Gremlin's challenges: all three in turn, or just one kind. */
  gremlinChallenges?: GremlinChallenges;
  aiCriticEnabled?: boolean;       // (games from before AI Draw Off became a duel game)
  aiCriticUrl?: string;
  /** AI Draw Off's critic: the host's own OpenAI / Gemini key (given to the room server, never stored), or the local critic. */
  aiCriticBackend?: 'openai' | 'gemini' | 'local' | 'ollama';
}

export interface Player {
  name: string;
  chips: number;
  giftId?: GiftId | null;
  ownerUid?: string;
  connected?: boolean;
  joinedAt?: number;
  age?: number;                    // only for under-16s (the Under-16 Exception); adults aren't asked
  avatar?: string;                 // the avatar they picked (core/avatars.ts); else one follows from the name
  isBot?: boolean;
  duelWins?: number;
  duelLosses?: number;
  votingWins?: number;
  duelsFought?: number;           // duels this player was in (each is +1 Party Score: playing counts)
  votingLosses?: number;
  partyPoints?: number;
}

export interface Gift {
  name: string;
  ownerId: Owner;
  naughtyLevel: number;
  misfitOrder?: number;
  createdAt?: number;
}

/** A sealed bid; in a projection, someone else's bid before the reveal is { amount: 0, sealed: true }. */
export interface Bid { amount: number; submittedAt?: number; sealed?: true }
export interface DuelVote { playerId?: PlayerId; votedFor: PlayerId; stake?: number; timestamp?: number }

export type TurnPhase =
  | 'bidding' | 'reveal' | 'yield_choice' | 'duel_pending' | 'yield_pending' | 'tiebreaker_pending'
  | 'duel' | 'reprisal' | 'minigame_party' | 'minigame_draw_off' | 'minigame_arena'
  | 'endgame_choice' | 'endgame_steal_select' | 'endgame_bidding' | 'endgame_reveal' | 'endgame_nobids';

/** One turn record; which fields are set depends on `phase` (mirrors what the pages write). */
export interface Turn {
  phase: TurnPhase;
  // main-game turn
  openerId?: PlayerId;
  activePlayerId?: PlayerId;
  giftId?: GiftId;
  bids?: Record<PlayerId, Bid>;
  openerChoice?: 'fight' | 'yield';
  selectedC1?: PlayerId; selectedC2?: PlayerId;       // yield_choice
  yieldC1?: PlayerId; yieldC2?: PlayerId;             // yield_pending
  tiebreakerC1?: PlayerId; tiebreakerC2?: PlayerId;   // tiebreaker_pending
  tiedBidders?: PlayerId[];
  minigameGoal?: 'pick_one' | 'pick_two';
  minigameStatus?: string;
  // duel
  challengerId?: PlayerId;
  defenderId?: PlayerId;
  pot?: number;
  tax?: number;
  isYieldDuel?: boolean;
  yieldingOpener?: PlayerId;
  isTiebreak?: boolean;
  tiebreakerBids?: Record<PlayerId, number>;
  isGambit?: boolean;
  originalGiftId?: GiftId;
  isReprisal?: boolean;
  reprisalDepth?: number;
  hostPeeked?: true;               // the host looked at the sealed bids (admin_peek_bids; logged)
  duelVotes?: Record<PlayerId, DuelVote>;
  voteEndsAt?: number;             // duel voting window (host clock); see voting.ts
  voteClosed?: boolean;            // host closed voting early
  // which duel is played (duelgame.ts)
  duelGame?: DuelGameChoice;
  wheel?: WheelSpin;
  duelAppStartedAt?: number;
  duelGameWinner?: PlayerId;       // an in-app duel game decided it; the host resolves
  duelGameOutcome?: { status: 'no_contest'; reason: string } | null; // it couldn't decide: the host picks
  arenaHandedOff?: string;         // a runtime mini-game handed its result back (once)
  /** A Game Lab game in the lobby (minigames/lab.ts): practice, nothing at stake; how it was set up. */
  lab?: { gameId: string; players: PlayerId[]; mode: string; duelists?: PlayerId[] };
  // double-tap guards (claimTurn)
  duelStartedBy?: PlayerId;
  resolvedBy?: PlayerId;
  // reprisal offer
  reprisalPlayerId?: PlayerId;
  excludedGiftId?: GiftId;
  // endgame
  misfitId?: GiftId;
  misfitName?: string;
  isMisfitDuel?: boolean;
  isEndgameSteal?: boolean;
  isEndgameAuction?: boolean;
  isNoBidsDuel?: boolean;
  bidderHadGift?: boolean;
  bidderOldGift?: GiftId | null;
}

export interface LastDuel {
  winnerId: PlayerId;
  loserId: PlayerId;
  pot: number;
  dividend: number;
  loserPaid: number;
  at: number | ServerTime;
  /** Spectator bets on this duel: the pool, what each winning chip got back, what went back to the Fund,
   *  how many backed the winner, what waits for the next duel; `returned`: nobody backed the winner. */
  bets?: { pool: number; each: number; toBank: number; winners: number; carry: number; returned?: boolean };
}

export interface Game {
  hostId?: string;
  hostUid?: string;
  createdAt?: number;
  phase: GamePhase;
  settings?: Settings;
  players?: Record<PlayerId, Player>;
  gifts?: Record<GiftId, Gift>;
  currentTurn?: Turn | null;
  minigame?: import('./party').PartyGame | import('./drawoff').DrawOff | null;
  lastDuel?: LastDuel | null;
  /** Spectator bets: chips left in the pool by earlier duels, waiting for the next one (scoring.ts). */
  betPool?: number | null;
  /** players + gifts as they were before the last duel was resolved (Admin → Undo); never shown. */
  undo?: Pick<Game, 'players' | 'gifts'> | null;
  /** A runtime mini-game (core/minigames): server only; each screen gets arenaView instead. */
  arena?: import('./minigames/runtime').ArenaInstance | null;
  /** Who has been teammates in a team game, and how often (so the next deal splits them up). */
  teamPairs?: Record<string, number>;
  arenaView?: import('./minigames/runtime').ArenaView | null;
}

/** RTDB server timestamp placeholder ({".sv": "timestamp"} is what serverTimestamp() sends). */
export interface ServerTime { '.sv': 'timestamp' }
export const SERVER_TIME: ServerTime = { '.sv': 'timestamp' };

/** A multi-path update relative to games/{room}: 'players/p1/chips' → 9, 'currentTurn' → null … */
export type Updates = Record<string, unknown>;

export type EventKind = 'gift' | 'bid' | 'duel';
export interface GameEvent { text: string; kind: EventKind }

/** What an action does: one atomic write, log lines for the host, and anything to do next. */
export interface Plan {
  updates: Updates;
  events: GameEvent[];
  /** Follow-ups the host runs after the write lands (they need the new state). */
  then?: 'drawEndgamePlayer' | 'checkEndgameContinue';
  /** Set when the action was refused (not enough chips, wrong phase …). */
  error?: string;
  /**
   * Win this first (a set-if-empty transaction at `path`) or don't apply the plan at all:
   * a double-tap, a second host tab, or a timer racing a button can't apply it twice.
   */
  claim?: { path: string; value: unknown };
}
