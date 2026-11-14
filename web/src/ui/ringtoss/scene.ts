// Ring Toss's scene as the screens draw it: where the reindeer's art sits, how high its tips are,
// and how long a ring is in the air on screen. Shared by the TV / console stage and the phone (which
// times its feedback to the ring landing on the TV). Drawing only: the room's landing and score are
// core/minigames/ringtoss.
import { arcFor } from '../../core/minigames/engines/projectile';

/** The reindeer (tools/art/reindeer.py, variant g seed 4, its left half mirrored so both sides score
 *  alike), standing chest-deep in a snowdrift (host/RingTossStage.tsx Drift, in front of it) at depth 1.65: the 1088 x 791
 *  art at 0.58 from (184.5, 127.6). In the art the far prongs' tips are at (84, 32), the tall ones at
 *  (197, 100), the hat's tip at (544, 30); the pegs (PEGS in core/minigames/ringtoss/ringtoss.ts) are under
 *  them, TIP_H is how high. All three are measured from the art: a new reindeer moves DEER, TIP_H and PEGS
 *  together. */
export const DEER = { src: 'art/ringtoss/reindeer.webp', x: 184.5, y: 127.6, w: 631, h: 458.8 };
export const TIP_H: Record<string, number> = { 'left-outer': 1.329, 'right-outer': 1.329, 'left-inner': 1.181, 'right-inner': 1.181, golden: 1.333 };

/** How long a ring is in the air on screen: its arc's time (engines/projectile arcFor: real gravity, so a
 *  longer, higher throw is up longer), but down before the room moves on (it waits the flight's ms + 300).
 *  A ringer's arc ends on its tip. Drawing only: the room's landing and score don't change. */
export const visMs = (t: { x: number; z: number; ms: number; pegId?: string | null }) =>
  Math.min(t.ms + 250, arcFor(Math.hypot(t.x, t.z), t.pegId ? TIP_H[t.pegId] ?? 1.15 : 0).ms);
