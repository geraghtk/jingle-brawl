// Duel voting window: once a duel starts, the spectators get to call it before the host can
// declare a winner — until every connected spectator has voted, the window runs out, or the
// host closes voting. (The host is often a player too; without this a quick click on "X wins"
// used to end the duel before anyone could vote.)
import { config } from './settings';
import { refuse, Tx } from './tx';
import type { Game, Plan, PlayerId } from './types';

export interface VoteStatus {
  /** Spectators who may vote (connected, not duelling). Tiebreak duels have none. */
  eligible: PlayerId[];
  voted: PlayerId[];
  waiting: PlayerId[];
  /** When the window closes (host clock), or null = not opened yet / no window. */
  endsAt: number | null;
  /** Votes are still being taken (with the window off they are, until the host calls it). */
  accepting: boolean;
  /** The host has to wait: votes are being taken and someone hasn't voted yet. */
  blocking: boolean;
}

export function voteStatus(game: Game, now: number): VoteStatus {
  const t = game.currentTurn;
  const none: VoteStatus = { eligible: [], voted: [], waiting: [], endsAt: null, accepting: false, blocking: false };
  if (!t || t.phase !== 'duel' || t.isTiebreak) return none;
  const eligible = Object.entries(game.players || {})
    .filter(([id, p]) => id !== t.challengerId && id !== t.defenderId && p.connected !== false)
    .map(([id]) => id);
  const voted = eligible.filter((id) => t.duelVotes?.[id]);
  const waiting = eligible.filter((id) => !t.duelVotes?.[id]);
  const windowMs = config(game.settings).voteWindow * 1000;
  const endsAt = typeof t.voteEndsAt === 'number' ? t.voteEndsAt : null;
  const accepting = !t.voteClosed && (endsAt == null || now < endsAt);
  return { eligible, voted, waiting, endsAt, accepting, blocking: accepting && waiting.length > 0 && windowMs > 0 };
}

/** The host opens the window when it sees a duel start (once: a claim on voteEndsAt). */
export function openVoting(game: Game, now: number): Plan {
  const t = game.currentTurn;
  if (!t || t.phase !== 'duel' || t.isTiebreak || t.voteEndsAt != null) return refuse('No vote to open');
  const windowMs = config(game.settings).voteWindow * 1000;
  if (windowMs <= 0) return refuse('Voting window is off');
  return new Tx(game).claim('currentTurn/voteEndsAt', now + windowMs).plan();
}

/** Host override: stop waiting for the stragglers. */
export function closeVoting(game: Game): Plan {
  const t = game.currentTurn;
  if (!t || t.phase !== 'duel') return refuse('No duel');
  return new Tx(game).set('currentTurn/voteClosed', true).event('🗳️ Voting closed', 'duel').plan();
}
