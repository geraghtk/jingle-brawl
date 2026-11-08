// Targets, for any throw game: pegs (or holes, or faces) at points in the scene, each worth some
// points within an oval zone (wider into the scene than across, since depth is the harder thing to
// judge). A landing scores the best peg it falls on; ties of position go to the higher value.
// Where the pegs are and what they are worth belong to each game.

export interface Peg { id: string; x: number; z: number; rx: number; rz: number; points: number }

export interface Hit { pegId: string | null; points: number; /** how far off the nearest peg's centre, in its own zone radii */ off: number }

/** Distance from a peg's centre measured in its zone (1 = the edge). */
export const zoneDistance = (peg: Peg, x: number, z: number) => Math.hypot((x - peg.x) / peg.rx, (z - peg.z) / peg.rz);

export function score(pegs: Peg[], x: number, z: number): Hit {
  let best: Hit = { pegId: null, points: 0, off: Infinity };
  for (const p of pegs) {
    const d = zoneDistance(p, x, z);
    const inside = d <= 1;
    if (inside && (p.points > best.points || (p.points === best.points && d < best.off))) best = { pegId: p.id, points: p.points, off: d };
    else if (!best.pegId && d < best.off) best = { pegId: null, points: 0, off: d };
  }
  return best;
}

/** Where a landing was, against the nearest peg, in words a thrower can act on. */
export function miss(pegs: Peg[], x: number, z: number): { dx: number; dz: number; nearest: string } {
  let near = pegs[0], d = Infinity;
  for (const p of pegs) { const k = zoneDistance(p, x, z); if (k < d) { d = k; near = p; } }
  return { dx: x - near.x, dz: z - near.z, nearest: near.id };
}
