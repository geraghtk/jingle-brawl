// Party Score + spectator bets.
//   Party Score = party points + 1 per correct duel call + 1 per duel you fought; the top score is the
//   Party MVP, and the final standings are in this order (leftover chips are worth nothing: a player who
//   never spent one used to top the table).
//   Spectator bets (experimental setting): each vote stakes 1 chip; wrong callers' chips are
//   split between right callers (odd chip → North Pole Fund; nobody right → all to the Fund).
import type { DuelVote, Player, PlayerId } from './types';

export const partyScore = (p: Partial<Player> | undefined) => (p?.partyPoints || 0) + (p?.votingWins || 0) + (p?.duelsFought || 0);

export function partyMvps(players: Record<PlayerId, Partial<Player>> | undefined): PlayerId[] {
  const scores = Object.entries(players || {}).map(([id, p]) => [id, partyScore(p)] as const);
  const best = Math.max(0, ...scores.map(([, s]) => s));
  return best > 0 ? scores.filter(([, s]) => s === best).map(([id]) => id) : [];
}

/** The Party Score table, best first; ties share a place (1, 2, 2, 4). */
export function partyStandings(players: Record<PlayerId, Partial<Player>> | undefined) {
  const rows = Object.entries(players || {})
    .map(([id, p]) => ({ id, name: p.name || '?', points: p.partyPoints || 0, calls: p.votingWins || 0, duels: p.duelsFought || 0, score: partyScore(p) }))
    .sort((a, b) => b.score - a.score || a.name.localeCompare(b.name));
  return rows.map((r) => ({ ...r, place: 1 + rows.findIndex((x) => x.score === r.score) }));
}

export function settleBets(votes: Record<PlayerId, DuelVote> | undefined, winnerId: PlayerId) {
  const delta: Record<PlayerId, number> = {};
  const staked = Object.entries(votes || {}).filter(([, v]) => (v?.stake || 0) > 0);
  const right = staked.filter(([, v]) => v.votedFor === winnerId).map(([id]) => id);
  const wrong = staked.filter(([, v]) => v.votedFor !== winnerId).map(([id]) => id);
  const pool = wrong.length;
  for (const id of wrong) delta[id] = -1;
  let toBank = pool;
  if (right.length) {
    const share = Math.floor(pool / right.length);
    for (const id of right) delta[id] = share;
    toBank = pool - share * right.length;
  }
  return { delta, pool, toBank, winners: right.length };
}
