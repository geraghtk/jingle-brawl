// How a finished mini-game is announced: the same words and the same list on every screen, for every
// game, worked out from its GameResult alone (ui/Ceremony.tsx draws it). It decides nothing: the
// result is the game's, the points are paid by handoff().
import type { PlayerId } from '../types';
import type { GameResult } from './contracts';

/** The game's own ending plays for this long before the ceremony comes up over it. */
export const CEREMONY_DELAY_MS = 1800;

export interface CeremonyRow { id: PlayerId; points: number; score?: number; rank?: number }
export interface Ceremony {
  /** `win`: someone won; `level`: nobody could be picked; `over`: it ended without a winner (a free-for-all with one loser). */
  tone: 'win' | 'level' | 'over';
  headline: string;
  sub: string;
  /** The people the moment belongs to (the winner, the winning team), captain first. */
  stars: CeremonyRow[];
  /** Everyone else who earned Party points (a bystander who never got caught, the best of the rest). */
  also: CeremonyRow[];
  /** The scores, best first, when the game has any (never more than `SCORES`). */
  scores: CeremonyRow[];
}

const SCORES = 5;
const cap = (s: string) => (s ? s[0].toUpperCase() + s.slice(1) : s);
const list = (names: string[]) => (names.length <= 1 ? names.join('') : `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`);

/** What to announce for `r` (null: nothing, e.g. the host cancelled it). `practice`: a Game Lab round, where no points are paid. */
export function ceremonyFor(r: GameResult | null | undefined, name: (id: PlayerId) => string, practice = false): Ceremony | null {
  if (!r || r.status === 'cancelled') return null;
  const pts: Record<PlayerId, number> = {};
  if (!practice) for (const w of r.rewards || []) if (w.points > 0) pts[w.playerId] = (pts[w.playerId] || 0) + Math.round(w.points);
  const score: Record<PlayerId, number> = {}, rank: Record<PlayerId, number> = {};
  for (const p of r.performance || []) if (p.score != null) score[p.playerId] = p.score;
  for (const p of r.placements || []) rank[p.playerId] = p.rank;
  const row = (id: PlayerId): CeremonyRow => ({ id, points: pts[id] || 0, ...(score[id] != null ? { score: score[id] } : {}), ...(rank[id] != null ? { rank: rank[id] } : {}) });
  const scored = Object.keys(score).sort((a, b) => (rank[a] ?? 99) - (rank[b] ?? 99) || score[b] - score[a]);
  const scores = scored.length > 1 ? scored.slice(0, SCORES).map(row) : [];
  const reason = cap(r.duel?.resolutionReason || r.reason || '');

  if (r.status === 'no_contest') return { tone: 'level', headline: 'No contest', sub: reason, stars: [], also: [], scores };
  if (r.status === 'requires_tiebreaker') return { tone: 'level', headline: 'Too close to call!', sub: r.duel ? 'The host decides the duel.' : reason, stars: [], also: Object.keys(pts).map(row), scores };

  const lead = r.duel?.winningDuelistId;
  const winners = [...(lead && r.winners.includes(lead) ? [lead] : []), ...r.winners.filter((id) => id !== lead)];
  const also = Object.keys(pts).filter((id) => !winners.includes(id)).sort((a, b) => pts[b] - pts[a]).map(row);
  if (!winners.length) {
    const out = r.losers.length === 1 ? name(r.losers[0]) : '';
    return { tone: 'over', headline: out ? `${out} is out!` : 'Game over!', sub: reason, stars: [], also, scores };
  }
  const headline = r.mode === 'captain_teams' && winners.length > 1 ? `${name(winners[0])}'s team wins!`
    : winners.length === 1 ? `${name(winners[0])} wins!`
    : winners.length <= 3 ? `${list(winners.map(name))} win!` : 'A tie at the top!';
  return { tone: 'win', headline, sub: reason, stars: winners.map(row), also, scores };
}

/** How it went for the person holding this phone. */
export function myEnding(r: GameResult | null | undefined, me: PlayerId, participants: PlayerId[]): 'won' | 'lost' | 'safe' | 'level' | 'watched' {
  if (!r || !participants.includes(me)) return 'watched';
  if (r.status !== 'resolved') return 'level';
  if (r.winners.includes(me)) return 'won';
  if (r.losers.includes(me)) return 'lost';
  return r.winners.length ? 'lost' : 'safe';       // nobody won and it was not me who lost
}
