// Which mascot line is up, and which waits: the rules, with no DOM (ui/mascots.ts draws what this decides).
// A line belongs to the turn it was said about. Two things used to go wrong: a line waited its turn behind
// another and came up during the NEXT turn, and a new moment waited behind a bubble about something that
// had already passed. So: a waiting line is dropped once its turn is over (or it has waited too long), only
// the newest moment waits at all, and a bubble gives way to it (after a moment to be read) instead of
// running its full time.

export interface Line<T = unknown> {
  who: string;
  text: string;
  /** The turn it is about (a counter the observer keeps: it goes up when a new turn or phase starts). */
  turn: number;
  /** When it was said. */
  at: number;
  /** Flavour, not a game moment: only when nothing else is up or waiting, and never right after another. */
  chatter?: boolean;
  opts?: T;
}

export interface Stage<T = unknown> { showing: Line<T> | null; shownAt: number; hiddenAt: number; queue: Line<T>[] }

/** A bubble stays up this long. */
export const SHOW_MS = 4200;
/** A bubble that has to give way (its turn is over, a newer moment waits) is up at least this long: long enough to read. */
export const MIN_SHOW_MS = 1800;
/** A line that has waited this long is old news. */
export const STALE_MS = 6000;
/** Quiet time after a bubble before idle chatter may come. */
export const CHATTER_GAP_MS = 5000;

export const emptyStage = <T = unknown>(): Stage<T> => ({ showing: null, shownAt: 0, hiddenAt: 0, queue: [] });

/** Someone has something to say. Chatter never waits; a moment replaces whatever was waiting (the newest wins). */
export function offer<T>(s: Stage<T>, line: Line<T>): Stage<T> {
  if (line.chatter) return s.showing || s.queue.length ? s : { ...s, queue: [line] };
  return { ...s, queue: [line] };
}

/** Where things stand at `now`, in turn `turn`: whether the bubble comes down, which line (if any) goes up, and
 *  in how many ms to look again (null: nothing is pending). */
export function step<T>(s: Stage<T>, turn: number, now: number): { stage: Stage<T>; hide: boolean; show: Line<T> | null; again: number | null } {
  let queue = s.queue.filter((l) => l.turn === turn && now - l.at <= STALE_MS);
  let { showing, shownAt, hiddenAt } = s;
  let hide = false;
  if (showing) {
    const age = now - shownAt;
    const giveWay = showing.turn !== turn || queue.some((l) => !l.chatter);
    if (age >= SHOW_MS || (giveWay && age >= MIN_SHOW_MS)) { hide = true; showing = null; hiddenAt = now; }
    else return { stage: { showing, shownAt, hiddenAt, queue }, hide: false, show: null, again: (giveWay ? MIN_SHOW_MS : SHOW_MS) - age };
  }
  // chatter right after a bubble is dropped, not held back for later
  queue = queue.filter((l) => !l.chatter || now - hiddenAt >= CHATTER_GAP_MS);
  const next = queue[0] ?? null;
  if (next) { showing = next; shownAt = now; queue = queue.slice(1); }
  return { stage: { showing, shownAt, hiddenAt, queue }, hide, show: next, again: next ? SHOW_MS : null };
}

interface Box { left: number; right: number; top: number; height: number }
/** The TV: the free stretch of its top bar between the name and the phase badge, if it is wide enough for a
 *  bubble (else null: nobody speaks rather than cover something). */
export function tvGap(brand: Box | null | undefined, badge: Box | null | undefined, bar: Box | null | undefined): { left: number; top: number; width: number; height: number } | null {
  if (!brand || !badge || !bar || bar.height <= 0) return null;
  const pad = bar.height * 0.25;
  const left = brand.right + pad, width = badge.left - pad - left;
  if (width < bar.height * 3.2) return null;
  // (a little taller than the bar: its margins are free too, and three lines of a size the room can read need it)
  const height = bar.height * 1.3;
  return { left, top: bar.top - (height - bar.height) / 2, width, height };
}
