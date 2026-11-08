// Is the game waiting on the host? When the host is playing too, their phone shows their own game,
// with the console in a tab next to it; this lights a dot on that tab. Only the moves that move
// the game on count (not overrides, skips or the Peek), and Reveal only once every bid is in.
import { HOST, offers } from './table';
import type { Game } from './types';

const MOVES = new Set([
  'start', 'draw', 'endgame', 'start_duel', 'start_yield_duel', 'host_spin', 'start_duel_app',
  'duel_winner', 'reveal_auction', 'complete_drawoff', 'auction_duel',
]);

export function hostsMove(game: Game, now: number): boolean {
  const mine = offers(game, HOST, now);
  if (mine.some((o) => MOVES.has(o.type))) return true;
  const t = game.currentTurn;
  if (t?.phase === 'bidding' && mine.some((o) => o.type === 'reveal')) {
    const bidders = Object.keys(game.players || {}).filter((id) => id !== t.openerId && game.players![id].connected !== false);
    return bidders.length > 0 && bidders.every((id) => t.bids?.[id]);
  }
  return false;
}
