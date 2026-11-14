// Party Score + spectator bets.
//   Party Score = party points + 1 per correct duel call + 1 per duel you fought; the top score is the
//   Party MVP, and the final standings are in this order (leftover chips are worth nothing: a player who
//   never spent one used to top the table).
//   Spectator bets (an optional setting): see "the pool" below.
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

// --- spectator bets: the pool -------------------------------------------------------------------
// An optional house setting. Each call stakes 1 chip into one pool for the duel, with 1 chip from the
// North Pole Fund laid at the house's line (the duelists' records) and whatever Santa Tax that duel
// paid. The winner's backers share the pool, at most CAP chips back for each chip staked; what is left
// rolls into the next duel's pool. Nobody backed the winner: stakes are returned.
// Why this shape (tools/economy-review, 2026-11-13): with no bonus a careful player never bets; with
// no cap a player who always backs the underdog farms the honest ones (12 to 23 chips a game at a table
// of twelve). With both, betting what you believe beats sitting out and no habit takes much more.
export const BETS = {
  stake: 1, seed: 1, cap: 3,
  /** Fewer spectators than this is not a market (two bettors only trade with each other): calls are free. */
  minSpectators: 4,
  /** The line shrinks each record towards even: this many wins and losses added to both. */
  pad: 3,
} as const;

type DuelRecord = { duelWins?: number; duelLosses?: number } | undefined;
/** The house's opening line: the chance `a` beats `b`, from their duel records this game (even with none). */
export function betLine(a: DuelRecord, b: DuelRecord): number {
  const rate = (p: DuelRecord) => ((p?.duelWins || 0) + BETS.pad) / ((p?.duelWins || 0) + (p?.duelLosses || 0) + 2 * BETS.pad);
  const ra = rate(a), rb = rate(b);
  return ra / (ra + rb);
}

/** What a duel's pool starts from: the chips waiting from earlier duels, this duel's tax, the line on duelist `a`. */
export interface PoolTerms { a: PlayerId; line: number; carry: number; tax: number }

const stakesOn = (votes: Record<PlayerId, DuelVote> | undefined) => Object.entries(votes || {}).filter(([, v]) => (v?.stake || 0) > 0);

/** The pool as it stands, and what one more chip on each duelist would get back if they won (stake included). */
export function poolOdds(votes: Record<PlayerId, DuelVote> | undefined, a: PlayerId, b: PlayerId, terms: PoolTerms) {
  const staked = stakesOn(votes);
  const on = (id: PlayerId) => staked.filter(([, v]) => v.votedFor === id).length;
  const pool = BETS.seed + terms.tax + terms.carry + staked.length * BETS.stake;
  const house = (id: PlayerId) => BETS.seed * (id === terms.a ? terms.line : 1 - terms.line);
  const pays = (id: PlayerId) => Math.min(BETS.cap, (pool + BETS.stake) / (on(id) + house(id) + 1));
  return { pool, backers: { [a]: on(a), [b]: on(b) } as Record<PlayerId, number>, pays: { [a]: pays(a), [b]: pays(b) } as Record<PlayerId, number> };
}

/** Settle a duel's bets: each staked vote's change in chips, and what waits for the next duel. */
export function settleBets(votes: Record<PlayerId, DuelVote> | undefined, winnerId: PlayerId, terms: PoolTerms) {
  const delta: Record<PlayerId, number> = {};
  const staked = stakesOn(votes);
  const right = staked.filter(([, v]) => v.votedFor === winnerId).map(([id]) => id);
  const none = { delta, pool: 0, each: 0, toBank: 0, winners: right.length, carry: terms.carry, returned: false };
  if (!staked.length) return none;                                     // no bets: the pool waits as it was
  if (!right.length) return { ...none, returned: true };               // nobody backed the winner: stakes go back
  const pool = BETS.seed + terms.tax + terms.carry + staked.length * BETS.stake;
  const house = BETS.seed * (winnerId === terms.a ? terms.line : 1 - terms.line);
  const share = pool / (right.length + house);
  const each = Math.min(BETS.cap, Math.floor(share + 1e-9));           // whole chips to each winning chip
  const toBank = Math.floor(Math.min(BETS.cap, share) * house + 1e-9); // the Fund's own winnings go back to it
  for (const [id, v] of staked) delta[id] = v.votedFor === winnerId ? each - BETS.stake : -BETS.stake;
  return { delta, pool, each, toBank, winners: right.length, carry: pool - each * right.length - toBank, returned: false };
}
