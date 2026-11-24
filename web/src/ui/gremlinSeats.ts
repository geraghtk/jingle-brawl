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

/** The part of the scene that has words on it (who has it, what they are doing, the mood, the rule): the
 *  middle of the table, in percent of the scene. A pass is never drawn through it. */
export const WORDS = { x0: 25, x1: 75, y0: 32, y1: 72 };

/** A pass drawn on the table: from one seat to another the short way ROUND the table, along the ring the
 *  seats are on (it used to cut straight across, through the words in the middle). The seats are drawn on
 *  top of it, so it shows between them and never crosses a name. `inset` (0..0.1) pulls it a little towards
 *  the middle, so an older pass on the same stretch lies beside a newer one, not under it. */
export function passPath(from: Seat, to: Seat, inset = 0): string {
  const angle = (p: Seat) => Math.atan2((p.y - OVAL.cy) / OVAL.ry, (p.x - OVAL.cx) / OVAL.rx);
  const a = angle(from);
  let turn = angle(to) - a;
  while (turn > Math.PI) turn -= Math.PI * 2;
  while (turn < -Math.PI) turn += Math.PI * 2;
  const k = 1 - Math.max(0, Math.min(0.1, inset));
  const steps = Math.max(2, Math.ceil(Math.abs(turn) / (Math.PI / 24)));
  const pts: string[] = [];
  for (let i = 0; i <= steps; i++) {
    const t = a + (turn * i) / steps;
    // it leaves and arrives at the seats themselves, and runs on the pulled-in ring between them
    const pull = 1 - (1 - k) * Math.sin((Math.PI * i) / steps);
    pts.push(`${(OVAL.cx + OVAL.rx * pull * Math.cos(t)).toFixed(2)},${(OVAL.cy + OVAL.ry * pull * Math.sin(t)).toFixed(2)}`);
  }
  return `M${pts.join(' L')}`;
}
