// Rule helpers shared by host and phone: who can be targeted, ties, and the money maths.
import type { Config } from './settings';
import type { Game, Gift, GiftId, PlayerId, Turn } from './types';
import { MISFIT } from './types';

const entries = <T>(o: Record<string, T> | undefined | null) => Object.entries(o || {});

/** Player ids tied at the highest positive bid of a turn (empty if no bids). */
export function topTiedBidders(turn: Turn | null | undefined): PlayerId[] {
  const bids = entries(turn?.bids).filter(([, b]) => b.amount > 0);
  if (!bids.length) return [];
  const top = Math.max(...bids.map(([, b]) => b.amount));
  return bids.filter(([, b]) => b.amount === top).map(([id]) => id);
}

/** Positive bids, highest first. */
export function rankedBids(turn: Turn | null | undefined): [PlayerId, number][] {
  return entries(turn?.bids)
    .filter(([, b]) => b.amount > 0)
    .map(([id, b]) => [id, b.amount] as [PlayerId, number])
    .sort((a, b) => b[1] - a[1]);
}

/** Reindeer Reprisal targets: any gift except the one just lost and never your own. Misfits allowed. */
export function reprisalTargets(game: Game, excludedGiftId: GiftId | undefined, playerId: PlayerId): [GiftId, Gift][] {
  return entries(game.gifts).filter(([id, g]) =>
    id !== excludedGiftId && g.ownerId !== playerId && (g.ownerId === MISFIT || !!game.players?.[g.ownerId]));
}

/** Grinch's Gambit targets: another PLAYER's gift (the Misfit pile has no one to duel). */
export function gambitTargets(game: Game, openedGiftId: GiftId | undefined, openerId: PlayerId): [GiftId, Gift][] {
  return entries(game.gifts).filter(([id, g]) =>
    id !== openedGiftId && g.ownerId !== openerId && g.ownerId !== MISFIT && !!game.players?.[g.ownerId]);
}

/** Endgame Path A: gifts held by other players. */
export function stealTargets(game: Game, activePlayerId: PlayerId): [GiftId, Gift][] {
  return entries(game.gifts).filter(([, g]) => g.ownerId !== MISFIT && g.ownerId !== activePlayerId && !!game.players?.[g.ownerId]);
}

/** Oldest Misfit first (FIFO). */
export function misfitPile(game: Game): [GiftId, Gift][] {
  return entries(game.gifts).filter(([, g]) => g.ownerId === MISFIT)
    .sort((a, b) => (a[1].misfitOrder || 0) - (b[1].misfitOrder || 0));
}

export const giftless = (game: Game): PlayerId[] => entries(game.players).filter(([, p]) => !p.giftId).map(([id]) => id);
export const wrappedGifts = (game: Game) => Math.max(0, Object.keys(game.players || {}).length - Object.keys(game.gifts || {}).length);

/** Santa Tax: once the paid-in chips reach the threshold, the winner gets at most maxWinnerPayout. */
export function potAndTax(cfg: Config, paidIn: number): { pot: number; tax: number } {
  if (paidIn >= cfg.taxThreshold) {
    const pot = Math.min(paidIn, cfg.maxWinnerPayout);
    return { pot, tax: paidIn - pot };
  }
  return { pot: paidIn, tax: 0 };
}

/** Yield duel: both challengers pay half their bid, rounded down. */
export const yieldPayment = (bid: number) => Math.floor(bid / 2);

/** Loser's Dividend for this duel (none for yield, misfit and tie-break duels). */
export function dividendFor(cfg: Config, turn: Turn, loserId: PlayerId): number {
  if (turn.isYieldDuel || turn.isMisfitDuel || turn.isTiebreak) return 0;
  const pays = cfg.loserDividendRule === 'everyone' || (cfg.loserDividendRule === 'defenders' && loserId === turn.defenderId);
  return pays ? cfg.loserDividend : 0;
}

/** First sealed-bid duel sets Naughty Level to the highest bid; every duel after that adds 1. */
export function naughtyAfterDuel(gift: Gift, turn: Turn): number {
  if ((gift.naughtyLevel || 0) === 0 && turn.bids) {
    return Math.max(0, ...Object.values(turn.bids).map((b) => b.amount || 0));
  }
  return (gift.naughtyLevel || 0) + 1;
}

/** Can this duel's loser take a Reindeer Reprisal? (mirrors the phone's help text) */
export function reprisalAllowed(cfg: Config, turn: Turn): boolean {
  return !turn.isYieldDuel && !turn.isMisfitDuel && !turn.isGambit && !turn.isTiebreak &&
    (turn.reprisalDepth || 0) < cfg.maxReprisalDepth;
}
