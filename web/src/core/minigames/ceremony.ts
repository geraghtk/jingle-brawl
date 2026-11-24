// How a finished mini-game is announced: the same words and the same list on every screen, for every
// game, worked out from its GameResult alone (ui/Ceremony.tsx draws it). It decides nothing: the
// result is the game's, the points are paid by handoff().
import type { PlayerId } from '../types';
import type { GameResult } from './contracts';
import type { PartyGame } from '../party';

/** The game's own ending plays for this long before the ceremony comes up over it. */
export const CEREMONY_DELAY_MS = 1800;

export interface CeremonyRow {
  id: PlayerId; points: number; score?: number; rank?: number;
  /** A note beside a score ("tied", "+1 chip Elf Bonus"). */
  tag?: string;
}
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
  /** What a score counts, one and many (["vote", "votes"]); none: a bare number. */
  unit?: [string, string];
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
  // (two scores are the duel itself, said already in the headline; a board of noughts says nothing)
  const scores = scored.length > 2 && scored.some((id) => score[id] > 0) ? scored.slice(0, SCORES).map(row) : [];
  // a reason that tells what the loser did ("holding the Gremlin when it went berserk") is said of the loser, by name
  const about = r.duel?.losingDuelistId ?? (r.losers.length === 1 ? r.losers[0] : undefined);
  const why = r.duel?.resolutionReason || r.reason || '';
  const reason = about && /^holding\b/i.test(why) ? `${name(about)} was ${why}` : cap(why);

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
    : winners.length === 1 ? (r.mode === 'party_with_duelists' ? `${name(winners[0])} wins the duel!` : `${name(winners[0])} wins!`)
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

/** A party game's results (Sketch-Off, Whack-an-Elf) as the same ceremony: who won the tie (or the duel) and
 *  goes on to duel, the Party points, everyone's score. `next`: who the tie's winner now duels and for what,
 *  when the screen knows. Null until the results are in. */
export function partyCeremony(mg: PartyGame | null | undefined, name: (id: PlayerId) => string, next: { opener?: string; gift?: string } = {}): Ceremony | null {
  const r = mg?.results;
  if (!mg || !r) return null;
  const goal = mg.minigameGoal, tied = mg.tiedBidders || [], winners = r.winners || [], ranking = r.ranking || [];
  const pts = (id: PlayerId) => (goal === 'lab' ? 0 : Math.round(r.points?.[id] || 0));
  const row = (id: PlayerId, i?: number): CeremonyRow => {
    const s = r.scoreOf?.[id];
    const tag = goal === 'lab' ? '' : [r.bonus === id ? '+1 chip Elf Bonus' : '', goal !== 'duel' && tied.includes(id) && !winners.includes(id) ? 'tied' : ''].filter(Boolean).join(' · ');
    return { id, points: pts(id), ...(s != null ? { score: s } : {}), ...(i != null ? { rank: i + 1 } : {}), ...(tag ? { tag } : {}) };
  };
  const scores = ranking.map((id, i) => row(id, i));
  const unit: [string, string] = mg.type === 'sketch_off' ? ['vote', 'votes'] : ['pt', 'pts'];
  const base = { also: [] as CeremonyRow[], scores, unit };
  if (goal === 'lab') {
    const top = ranking[0];
    const any = top != null && (r.scoreOf?.[top] ?? 0) > 0;
    return any ? { tone: 'win', headline: `${name(top)} comes top!`, sub: '', stars: [row(top)], ...base }
      : { tone: 'over', headline: 'Nobody scored!', sub: '', stars: [], ...base };
  }
  if (!winners.length) return { tone: 'level', headline: 'Too close to call!', sub: '', stars: [], ...base };
  const stars = winners.map((id) => row(id));
  const forGift = next.gift ? ` for ${next.gift}` : ' for the gift';
  if (goal === 'duel') return { tone: 'win', headline: `${name(winners[0])} wins the duel!`, sub: '', stars, ...base };
  if (goal === 'pick_two') return { tone: 'win', headline: `${list(winners.map(name))} win the tie!`, sub: `Now they duel each other${forGift}.`, stars, ...base };
  return { tone: 'win', headline: `${name(winners[0])} wins the tie!`, sub: `Next: a duel with ${next.opener || 'the Opener'}${forGift}.`, stars, ...base };
}

/** How a party game went for the person holding this phone. */
export function myPartyEnding(mg: PartyGame | null | undefined, me: PlayerId): 'won' | 'lost' | 'watched' {
  const r = mg?.results;
  if (!mg || !r) return 'watched';
  if (mg.minigameGoal === 'lab') return r.ranking?.[0] === me && (r.scoreOf?.[me] ?? 0) > 0 ? 'won' : 'watched';
  if ((r.winners || []).includes(me)) return 'won';
  return (mg.tiedBidders || []).includes(me) ? 'lost' : 'watched';       // the rest played for Party points
}
