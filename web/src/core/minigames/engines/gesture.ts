// Gestures, for any throw game: a phone sends the path of a swipe (screen fractions and ms, as the
// finger moved); the server turns it into a launch (aim and power) with this one mapping. The
// phone never says where its throw lands, and every phone is measured in its own screen heights, so
// a big phone or a fast touch screen is no advantage. Nothing here is about rings or reindeer.

/** A swipe as the phone saw it: [x, y, t] with x and y in 0..1 of its screen (y grows downward), t in ms. */
export type SwipePath = [number, number, number][];

export interface Launch {
  angle: number;   // radians from straight ahead (negative: left), within ±MAX_ANGLE
  power: number;   // 0..MAX_POWER, about 1 for a brisk flick
}

export const GESTURE_VERSION = '1';
export const MAX_ANGLE = 0.5;
export const MAX_POWER = 1.3;
/** Screen heights per second that make power 1 (a brisk flick). */
export const FLICK_SPEED = 3;
/** The end of the swipe counts: how a flick leaves the finger, not how it started. Measured over
 *  the last stretch of at least this long (enough samples to be steady on any touch screen). */
export const RELEASE_WINDOW_MS = 40;
const MAX_POINTS = 64;

/** Check and tidy a path from a phone: null if it isn't a swipe at all. */
export function parsePath(raw: unknown): SwipePath | null {
  if (!Array.isArray(raw) || raw.length < 2 || raw.length > MAX_POINTS) return null;
  const out: SwipePath = [];
  for (const p of raw) {
    if (!Array.isArray(p) || p.length !== 3 || !p.every((n) => typeof n === 'number' && Number.isFinite(n))) return null;
    const [x, y, t] = p as number[];
    if (x < -0.5 || x > 1.5 || y < -0.5 || y > 1.5 || t < 0 || t > 5000) return null;
    if (out.length && t < out[out.length - 1][2]) return null;          // time only moves on
    out.push([x, y, t]);
  }
  return out;
}

/** A swipe's launch: upward is forward; its direction aims, its speed at release is the power.
 *  null: not a throw (downward, or too slow to leave the hand). */
export function launchFrom(path: SwipePath): Launch | null {
  const end = path[path.length - 1];
  // the shortest stretch at the end lasting RELEASE_WINDOW_MS (or the whole swipe, if shorter)
  let i = path.length - 2;
  while (i > 0 && end[2] - path[i][2] < RELEASE_WINDOW_MS) i--;
  const start = path[i];
  const dt = Math.max(16, end[2] - start[2]);               // one frame at least: a 1 ms "swipe" isn't a superpower
  const dx = end[0] - start[0], dy = start[1] - end[1];      // dy > 0: upward
  if (dy <= 0.01) return null;
  const speed = Math.hypot(dx, dy) / (dt / 1000);            // screen heights (roughly) per second
  const power = Math.min(MAX_POWER, speed / FLICK_SPEED);
  if (power < 0.1) return null;
  const angle = Math.max(-MAX_ANGLE, Math.min(MAX_ANGLE, Math.atan2(dx, dy)));
  return { angle, power };
}

/** The accessible way to throw (aim, then a hold-to-charge meter): the same launch, within the same limits. */
export function launchFromControls(raw: unknown): Launch | null {
  const r = raw as { aim?: unknown; power?: unknown } | null;
  if (!r || typeof r.aim !== 'number' || typeof r.power !== 'number' || !Number.isFinite(r.aim) || !Number.isFinite(r.power)) return null;
  return { angle: Math.max(-1, Math.min(1, r.aim)) * MAX_ANGLE, power: Math.max(0.1, Math.min(MAX_POWER, r.power)) };
}

/** A swipe that makes `launch` (for tests and the simulated hands): straight, steady, 100 ms. */
export function pathFor(launch: Launch, steps = 6): SwipePath {
  const dist = launch.power * FLICK_SPEED * 0.1;
  const dx = Math.sin(launch.angle) * dist, dy = Math.cos(launch.angle) * dist;
  return Array.from({ length: steps + 1 }, (_, k): [number, number, number] => {
    const f = k / steps;
    return [0.5 + dx * f, 0.9 - dy * f, Math.round(100 * f)];
  });
}
