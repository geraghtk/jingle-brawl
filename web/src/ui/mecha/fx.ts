// Mecha Santa Mayhem: the sums behind the TV's effects (ui/mecha/scene3d.ts), kept apart from three.js
// so they can be tested. They read the room's view and decide nothing: a footfall is the moment a move
// the screen was watching comes down, a camera punch is a bounce that dies away.
import type { Busy, MoveKind } from '../../core/minigames/engines/walker';

/** A move the screen has seen in the air. */
export interface Flight { kind: MoveKind; leg: number; at: number; lands: number }

/** How long a camera punch lasts. */
export const PUNCH_MS = 520;
/** How far off its mark (world units, up is positive) the camera is `ageMs` after a thump of size `amp`:
    down first, a couple of smaller bounces, still by PUNCH_MS. */
export function punchAt(amp: number, ageMs: number): number {
  if (!(amp > 0) || ageMs < 0 || ageMs >= PUNCH_MS) return 0;
  const s = ageMs / 1000;
  return -amp * Math.exp(-8 * s) * Math.cos(36 * s) * (1 - ageMs / PUNCH_MS);
}

/** A landing more than this long ago is old news (the screen was away): no dust for it. */
export const LANDING_FRESH_MS = 500;
/** One frame's look at a robot's move. `air`: the move the screen was watching (null: none); `move`: the
    view's move now. Gives the move to watch next, and the one that has just come down (null: none has).
    A move cut short by a topple does not land. */
export function touchdown(air: Flight | null, move: Flight | null | undefined, now: number, busy?: Busy | null): { air: Flight | null; landed: Flight | null } {
  const flying = move && now < move.lands ? { kind: move.kind, leg: move.leg, at: move.at, lands: move.lands } : null;
  if (!air || (flying && flying.at === air.at)) return { air: flying, landed: null };
  const down = now >= air.lands && now - air.lands < LANDING_FRESH_MS && busy !== 'topple';
  return { air: flying, landed: down ? air : null };
}

/** The two robots are this far apart (world units) when the picture splits in two, and this near when it joins again. */
export const SPLIT_AT = 15, JOIN_AT = 12;
/** Should the picture be split: it splits when they are too far apart for one camera to hold both at a
    readable size, and only joins once they are well within range again (so it does not flicker at the edge). */
export function splitFor(gap: number, wasSplit: boolean): boolean {
  return wasSplit ? Math.abs(gap) > JOIN_AT : Math.abs(gap) > SPLIT_AT;
}
/** The strips of a split picture, as fractions of the stage's height from the top: clear of the team panels
    above and the course bar below. Team A's is the upper one. */
export const STRIPS: [{ top: number; height: number }, { top: number; height: number }] = [{ top: 0.265, height: 0.32 }, { top: 0.585, height: 0.32 }];

/** Where a robot hanging on a rope is: the rope hangs from over the middle of the gap and swings through
    `swing` (-1 right back to 1 right forward). In world units, from the pivot. */
export const ROPE = { length: 3.3, reach: 0.72, pivotY: 5.85, hands: 2.6, tilt: 0.35 };
/** `dx`, `y`: where its feet are (so that its raised hands are on the end of the rope: it hangs tilted `lean`, a
    part of the rope's own `angle`, and the hands are `ROPE.hands` up its body from the feet). */
export function hangAt(swing: number): { dx: number; y: number; angle: number; lean: number } {
  const angle = Math.max(-1, Math.min(1, swing)) * ROPE.reach, lean = angle * ROPE.tilt;
  const endX = Math.sin(angle) * ROPE.length, endY = ROPE.pivotY - Math.cos(angle) * ROPE.length;
  return { dx: endX - Math.sin(lean) * ROPE.hands, y: endY - Math.cos(lean) * ROPE.hands, angle, lean };
}

/** Over the top of a climb and down the far side, `f` (0 to 1) of the way through: first across the top, clear of
    it (`over` 0 to 1 of the way along, `y` above the top), then straight down the far side. `top`: how high the
    thing stands. (Kevin, 2026-11-23: it used to sink through the stack on its way across.) */
export const VAULT_ACROSS = 0.6, VAULT_CLEAR = 0.5;
export function vaultAt(f: number, top: number): { over: number; y: number } {
  const g = Math.max(0, Math.min(1, f));
  if (g < VAULT_ACROSS) { const k = g / VAULT_ACROSS; return { over: k, y: top + VAULT_CLEAR + Math.sin(k * Math.PI) * 0.35 }; }
  const d = (g - VAULT_ACROSS) / (1 - VAULT_ACROSS);
  return { over: 1, y: (top + VAULT_CLEAR) * (1 - d * d) };
}
