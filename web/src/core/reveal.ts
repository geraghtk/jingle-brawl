// The sealed bids once they are opened, as a list a screen can draw: everyone's bid highest first,
// who is on top, whether the top is a tie. Before the reveal there is nothing to list (a screen's
// view holds other people's bids sealed until then: core/projection.ts), and this says so with null.
import type { Game, PlayerId } from './types';

/** The phases in which the table has seen the bids. */
const OPEN_PHASES = new Set(['reveal', 'yield_choice', 'tiebreaker_pending', 'duel_pending', 'yield_pending', 'endgame_reveal']);

export interface BidRow {
  id: PlayerId;
  /** Chips bid; 0 for a pass or for someone who never sent a bid. */
  amount: number;
  /** One of the highest bids (more than one: a tie). */
  top: boolean;
  /** Place among the bids, ties sharing one (1, 1, 3 ...); null for a pass. */
  place: number | null;
  /** The reveal came before they sent anything. */
  missing?: true;
}

export interface RevealedBids {
  rows: BidRow[];
  /** The highest bid (0: nobody bid). */
  top: number;
  /** Who bid it. */
  leaders: PlayerId[];
  tie: boolean;
  /** Whose gift it is (the opener, or the Misfit auction's active player): they do not bid. */
  sitter?: PlayerId;
}

export function revealedBids(game: Game): RevealedBids | null {
  const t = game.currentTurn;
  if (!t || !OPEN_PHASES.has(t.phase)) return null;
  const bids = Object.entries(t.bids || {});
  if (bids.some(([, b]) => b.sealed)) return null;       // a sealed bid is never listed as a pass
  const sitter = (t.phase === 'endgame_reveal' ? t.activePlayerId : t.openerId) || undefined;
  const name = (id: PlayerId) => game.players?.[id]?.name || '';
  const sent = bids.filter(([id]) => id !== sitter)
    .map(([id, b]) => ({ id, amount: Math.max(0, b.amount || 0), at: b.submittedAt ?? 0 }))
    .sort((a, b) => b.amount - a.amount || a.at - b.at || name(a.id).localeCompare(name(b.id)));
  const top = sent.length ? sent[0].amount : 0;
  const rows: BidRow[] = sent.map((r) => ({
    id: r.id, amount: r.amount, top: top > 0 && r.amount === top,
    place: r.amount > 0 ? 1 + sent.filter((o) => o.amount > r.amount).length : null,
  }));
  for (const id of Object.keys(game.players || {})) {
    if (id !== sitter && !game.players![id].isBot && !rows.some((r) => r.id === id)) rows.push({ id, amount: 0, top: false, place: null, missing: true });
  }
  const leaders = rows.filter((r) => r.top).map((r) => r.id);
  return { rows, top, leaders, tie: leaders.length > 1, ...(sitter ? { sitter } : {}) };
}

/** "Alice", "Alice and Bob", "Alice, Bob and Cara". */
export function nameList(names: string[]): string {
  if (names.length <= 1) return names[0] || '';
  return `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`;
}

/** The reveal in one plain sentence, from `me`'s seat ("You have the top bid: 4 chips."). */
export function revealSummary(game: Game, r: RevealedBids, me?: PlayerId): string {
  const chips = `${r.top} ${r.top === 1 ? 'chip' : 'chips'}`;
  const who = (id: PlayerId) => (id === me ? 'you' : game.players?.[id]?.name || 'Someone');
  if (!r.leaders.length) return 'Nobody bid.';
  if (r.tie) {
    // "you" goes last so the sentence reads "Alice and you"
    const names = [...r.leaders].sort((a, b) => Number(a === me) - Number(b === me)).map(who);
    const kind = r.leaders.length === 2 ? 'A tie' : `A ${r.leaders.length}-way tie`;
    return `${kind} at ${chips}: ${nameList(names)}.`;
  }
  return r.leaders[0] === me ? `You have the top bid: ${chips}.` : `${who(r.leaders[0])} has the top bid: ${chips}.`;
}
