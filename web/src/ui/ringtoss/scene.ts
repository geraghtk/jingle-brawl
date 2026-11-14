// Ring Toss's scene as the screens draw it: the picture's size and perspective (toScreen), where the
// reindeer's art sits and how high its tips are, how big a ring is and where a ringer rests, and the
// timings (how long a ring is in the air on screen, how long a landing's effects last). Shared by the
// TV / console stage and the phone (which times its feedback to the ring landing on the TV). Drawing
// only: the room's landing and score are core/minigames/ringtoss.
import { arcFor } from '../../core/minigames/engines/projectile';
import type { Peg } from '../../core/minigames/engines/targets';

/** The scene: world x (across) and z (into the scene) to the picture, with a little perspective. */
export const W = 1000, H = 620;
// The view: SX across, SZ into the scene (squeezed: the reindeer stands at depth 1.65, still big, in a
// snowdrift; the longest throw lands just behind it), HZ up. The physics is in
// world units (x, z) only; heights are only for drawing.
const SX = 1290, SZ = 86, HZ = 420;
export function toScreen(x: number, z: number, h = 0): { sx: number; sy: number; k: number } {
  const k = 1 / (1 + 0.35 * z);                       // further away: smaller
  return { sx: W / 2 + x * SX * k, sy: H - 30 - z * SZ * k - h * HZ * k, k };
}

/** The reindeer (tools/art/reindeer.py, variant g seed 4, its left half mirrored so both sides score
 *  alike), standing chest-deep in a snowdrift (host/RingTossStage.tsx Drift, in front of it) at depth 1.65: the 1088 x 791
 *  art at 0.58 from (184.5, 127.6). In the art the far prongs' tips are at (84, 32), the tall ones at
 *  (197, 100), the hat's tip at (544, 30); the pegs (PEGS in core/minigames/ringtoss/ringtoss.ts) are under
 *  them, TIP_H is how high. All three are measured from the art: a new reindeer moves DEER, TIP_H and PEGS
 *  together. */
export const DEER = { src: 'art/ringtoss/reindeer.webp', x: 184.5, y: 127.6, w: 631, h: 458.8 };
export const TIP_H: Record<string, number> = { 'left-outer': 1.329, 'right-outer': 1.329, 'left-inner': 1.181, 'right-inner': 1.181, golden: 1.333 };
export const BACKDROP = 'art/ringtoss/backdrop.webp';
/** A ring on the hat slides down the cone to where the cone fills it (px below the tip); on a prong it
 *  slides a little way down. */
const HAT_DROP = 67;
const PRONG_DROP = 10;
/** Rings on the ground nearer than this are in front of the reindeer; further ones behind it. */
export const DEER_Z = 1.65;
/** A ring this high is as far up as rings go (its shadow is faintest then). */
export const HIGH = 1.55;
/** A ring's size: RING_R as it comes up from the bottom of the screen, about 22 at the reindeer (k^2.5: it
 *  shrinks quickly, so it reads as going away). */
const RING_R = 70;
export const ringR = (k: number) => RING_R * k ** 2.5;
/** A ringer bursts into stars in its thrower's colour as it lands, and is gone after this long. */
export const RINGER_MS = 1300;

/** The reindeer's nose in the scene (the art's (543, 541)): it lights up for a ringer. */
export const NOSE = { x: DEER.x + 543 * 0.58, y: DEER.y + 541 * 0.58, rx: 40, ry: 24 };

/** Where a ringer rests: a little down its prong (stacking further down), or down the hat's cone to its
 *  brim (the ring is wider than the cone). */
export function restOn(p: Peg, stack: number): { sx: number; sy: number; k: number } {
  const tip = toScreen(p.x, p.z, TIP_H[p.id] ?? 1.15);
  return { ...tip, sy: tip.sy + (p.id === 'golden' ? HAT_DROP + stack * 6 : PRONG_DROP + stack * 7) };
}
/** The same, as a height above the ground at the peg (a ringer's flight ends there, not on the ground). */
export function restHeight(p: Peg, stack: number): number {
  const ground = toScreen(p.x, p.z, 0), r = restOn(p, stack);
  return (ground.sy - r.sy) / (HZ * ground.k);
}

/** How long a ring is in the air on screen: its arc's time (engines/projectile arcFor: real gravity, so a
 *  longer, higher throw is up longer), but down before the room moves on (it waits the flight's ms + 300).
 *  A ringer's arc ends on its tip. Drawing only: the room's landing and score don't change. */
export const visMs = (t: { x: number; z: number; ms: number; pegId?: string | null }) =>
  Math.min(t.ms + 250, arcFor(Math.hypot(t.x, t.z), t.pegId ? TIP_H[t.pegId] ?? 1.15 : 0).ms);

/** A ring in the air spins and wobbles as it flies (a thrown ring is never quite flat): how far it's tipped
 *  (degrees) and how open it looks (its height over its width), `ms` into its flight. Each ring its own. */
export function tumble(n: number, ms: number, angle: number): { deg: number; open: number } {
  const s = ms / 1000, ph = n * 1.7;
  return { deg: angle * 40 + 12 * Math.sin(s * 9.5 + ph), open: 0.42 + 0.13 * Math.sin(s * 15 + ph * 0.6) };
}

/** A missed ring goes up in a puff of snow dust as it lands: gone after this long. */
export const MISS_FADE_MS = 1300;
