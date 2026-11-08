// Scoring policies: how a game's raw outcome becomes a result. Versioned, so a result says which
// rules made it. Only the two the first games need are built.
import type { PlayerId } from '../types';
import type { GameResult, MatchMode } from './contracts';

export interface ScoringPolicy<Input> {
  id: string;
  version: string;
  result(gameId: string, instanceId: string, mode: MatchMode, input: Input, duelists?: [PlayerId, PlayerId]): GameResult;
}

/** One loser (e.g. holding the Gremlin at doom). A duel's other duelist wins; a free-for-all has survivors, no winner. */
export const singleLoser: ScoringPolicy<{ loser: PlayerId; participants: PlayerId[]; reason: string }> = {
  id: 'single_loser', version: '1',
  result(gameId, instanceId, mode, { loser, participants, reason }, duelists) {
    const survivors = participants.filter((p) => p !== loser);
    const base = { gameId, instanceId, mode, losers: [loser], scoringPolicyVersion: 'single_loser@1', reason, metrics: { survivors } };
    if (duelists && duelists.includes(loser)) {
      const winner = duelists[0] === loser ? duelists[1] : duelists[0];
      return { ...base, status: 'resolved', winners: [winner], duel: { duelists, winningDuelistId: winner, losingDuelistId: loser, resolutionReason: reason } };
    }
    if (duelists) return { ...base, status: 'requires_tiebreaker', winners: [], duel: { duelists, resolutionReason: 'a bystander lost; the duelists are still level' } };
    return { ...base, status: 'resolved', winners: [] };     // free-for-all: no invented champion
  },
};

/** Two scores head to head: the higher wins; equal is a tie the caller must break. */
export const headToHead: ScoringPolicy<{ scores: Record<PlayerId, number>; reason: string }> = {
  id: 'head_to_head', version: '1',
  result(gameId, instanceId, mode, { scores, reason }, duelists) {
    if (!duelists) throw new Error('head_to_head needs two duelists');
    const [a, b] = duelists;
    const base = { gameId, instanceId, mode, scoringPolicyVersion: 'head_to_head@1', reason, performance: duelists.map((p) => ({ playerId: p, score: scores[p] || 0 })) };
    if ((scores[a] || 0) === (scores[b] || 0)) return { ...base, status: 'requires_tiebreaker', winners: [], losers: [], duel: { duelists, resolutionReason: 'a tie' } };
    const [w, l] = (scores[a] || 0) > (scores[b] || 0) ? [a, b] : [b, a];
    return { ...base, status: 'resolved', winners: [w], losers: [l], duel: { duelists, winningDuelistId: w, losingDuelistId: l, resolutionReason: reason } };
  },
};
