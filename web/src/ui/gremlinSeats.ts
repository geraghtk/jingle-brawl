// Pass the Gremlin on the big screen: where each player sits round the table. Drawing only (no DOM):
// the seats are spread evenly along an oval (by distance along it, not by angle, so they don't bunch
// at the ends), the first at the top, clockwise. Positions are percentages of the scene.

export interface Seat { x: number; y: number }

/** The oval the seats are on: its centre and radii, in percent of the scene's width and height. */
export const OVAL = { cx: 50, cy: 51, rx: 40, ry: 34 };
/** The scene's shape the spacing is worked out for (width over height). */
const ASPECT = 16 / 9;

export function seats(n: number): Seat[] {
  if (n <= 0) return [];
  // walk the oval in small steps, measuring in units of the scene's height
  const STEPS = 720;
  const at = (i: number) => {
    const t = -Math.PI / 2 + (i / STEPS) * Math.PI * 2;
    return { x: OVAL.cx + OVAL.rx * Math.cos(t), y: OVAL.cy + OVAL.ry * Math.sin(t) };
  };
  const len: number[] = [0];
  for (let i = 1; i <= STEPS; i++) {
    const a = at(i - 1), b = at(i);
    len.push(len[i - 1] + Math.hypot((b.x - a.x) * ASPECT, b.y - a.y));
  }
  const out: Seat[] = [];
  let j = 0;
  for (let k = 0; k < n; k++) {
    const want = (k / n) * len[STEPS];
    while (j < STEPS && len[j + 1] < want) j++;
    out.push(at(j));
  }
  return out;
}

/** How big a seat's picture is (percent of the scene's height): smaller at a fuller table. */
export const seatSize = (n: number): number => (n <= 6 ? 17 : n <= 9 ? 14.5 : n <= 13 ? 12 : n <= 16 ? 10.5 : 9);

/** A pass drawn across the table: a curve from one seat to another, bowed towards the middle. */
export function passPath(from: Seat, to: Seat): string {
  const mx = (from.x + to.x) / 2, my = (from.y + to.y) / 2;
  const bx = mx + (OVAL.cx - mx) * 0.45, by = my + (OVAL.cy - my) * 0.45;
  return `M${from.x.toFixed(2)},${from.y.toFixed(2)} Q${bx.toFixed(2)},${by.toFixed(2)} ${to.x.toFixed(2)},${to.y.toFixed(2)}`;
}
