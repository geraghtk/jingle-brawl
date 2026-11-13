// A thrown object's flight, for any throw game: deterministic 2.5D (across, into the scene, and a
// height arc), so the server decides where it lands and every screen draws the same flight from
// the same numbers. Versioned: a result says which physics made it.
import type { Launch } from './gesture';

export interface FlightParams {
  version: string;
  baseRange: number;     // how far a launch of power 0 would go (into the scene)
  rangePerPower: number; // how much further each unit of power carries it
  arc: number;           // peak height as a share of the distance
  flightMs: number;      // how long a flight of baseRange + rangePerPower takes (others scale)
}

/** v2: a slower, higher arc you can follow on a TV (about 1 to 1.3 s). v3: a flick's speed matters a
 *  little less (0.85 of distance per unit of power, from 0.75): a brisk flick (power about 1.06) reaches
 *  the Ring Toss reindeer, 10% too fast or slow is about 0.09 off, and the hardest throw lands just
 *  behind it (1.86), not in the distance. v4: a harder throw is quicker in the air too. */
export const DEFAULT_FLIGHT: FlightParams = { version: '4', baseRange: 0.75, rangePerPower: 0.85, arc: 0.5, flightMs: 1300 };

/** Where it comes down: x across (negative left), z into the scene, from a launch at (0, 0). */
export interface Landing { x: number; z: number; ms: number }

export function landing(l: Launch, p: FlightParams = DEFAULT_FLIGHT): Landing {
  const range = p.baseRange + p.rangePerPower * l.power;
  // a harder throw goes further AND faster (it looks it): about 1.4 s for a soft lob, 1 s for the hardest
  const ms = Math.round(p.flightMs * (1.15 - 0.3 * Math.min(1.3, Math.max(0, l.power))));
  return { x: range * Math.sin(l.angle), z: range * Math.cos(l.angle), ms };
}

/** How a flight looks (drawing only: where it lands is landing()'s). Real gravity, the same for every
 *  throw, and a little air drag: it leaves the hand quick and slows along the ground, so its top comes
 *  late and it drops steeply at the end (a long throw clears the reindeer's antlers, then falls behind).
 *  A longer throw goes higher (loft x its distance, between minApex and maxApex), so it's up longer;
 *  how long falls out of the gravity. Heights are in the scene's height units (the top of a TV is about 1.55). */
export interface ArcParams { gravity: number; drag: number; loft: number; minApex: number; maxApex: number }
export const DEFAULT_ARC: ArcParams = { gravity: 7.9, drag: 3, loft: 0.83, minApex: 0.45, maxApex: 1.55 };

/** One flight's arc: how long it's in the air, how high it gets, how fast it starts upward. */
export interface Arc { ms: number; apex: number; vz: number }

/** The arc of a throw that travels `range` along the ground and comes down at height `endH` (0: the
 *  ground; a ringer ends on its tip, so it tops out above it and drops onto it). */
export function arcFor(range: number, endH = 0, p: ArcParams = DEFAULT_ARC): Arc {
  const apex = Math.max(endH + 0.25, Math.min(p.maxApex, Math.max(p.minApex, p.loft * range)));
  const up = Math.sqrt((2 * apex) / p.gravity), down = Math.sqrt((2 * (apex - endH)) / p.gravity);
  return { ms: Math.round((up + down) * 1000), apex, vz: p.gravity * up };
}

/** `u` of the way through the flight (by time): how far along the ground it is (0 to 1) and how high. */
export function arcAt(a: Arc, u: number, p: ArcParams = DEFAULT_ARC): { along: number; h: number } {
  const s = Math.max(0, Math.min(1, u)), t = (s * a.ms) / 1000;
  const along = p.drag > 0 ? (1 - Math.exp(-p.drag * s)) / (1 - Math.exp(-p.drag)) : s;
  return { along, h: Math.max(0, a.vz * t - (p.gravity * t * t) / 2) };
}

/** Where it is `t` ms after launch (for drawing): x, z and its height above the ground. */
export function flightAt(l: Launch, t: number, p: FlightParams = DEFAULT_FLIGHT): { x: number; z: number; h: number; done: boolean } {
  const end = landing(l, p);
  const s = Math.max(0, Math.min(1, t / end.ms));
  const range = Math.hypot(end.x, end.z);
  return { x: end.x * s, z: end.z * s, h: 4 * p.arc * range * s * (1 - s), done: s >= 1 };
}
