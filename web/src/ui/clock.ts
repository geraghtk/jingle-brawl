// A countdown in whole seconds, as every screen shows it (the room's time: net/hooks useServerNow).

/** Seconds left until `endsAt`, rounded up (1 until the very end), never below 0. */
export const secsLeft = (endsAt: number, now: number): number => Math.max(0, Math.ceil((endsAt - now) / 1000));

/** The last duel's verdict has this long on the screens before the night's finale takes over. */
export const VERDICT_FIRST_MS = 7000;
/** May the finale start: the game is over, and the verdict of the duel that ended it (if one did) has been seen.
 *  (It used to cut straight to the MVP: the TV never showed who won the last duel, and phones had three layers at once.) */
export function finaleDue(game: { phase?: string; lastDuel?: { at?: unknown; announced?: unknown } | null }, now: number): boolean {
  if (game.phase !== 'ended') return false;
  // a duel an in-app game's ending announced has no verdict to wait for: that ending (with what the winner
  // takes) had its time on the screens before the duel was settled (RESULT_HOLD_MS, a party game's results)
  if (game.lastDuel?.announced) return true;
  const at = game.lastDuel?.at;
  return !(typeof at === 'number' && now < at + VERDICT_FIRST_MS);
}

