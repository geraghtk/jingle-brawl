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

export const DEFAULT_FLIGHT: FlightParams = { version: '1', baseRange: 0.6, rangePerPower: 1.1, arc: 0.35, flightMs: 900 };

/** Where it comes down: x across (negative left), z into the scene, from a launch at (0, 0). */
export interface Landing { x: number; z: number; ms: number }

export function landing(l: Launch, p: FlightParams = DEFAULT_FLIGHT): Landing {
  const range = p.baseRange + p.rangePerPower * l.power;
  const ms = Math.round(p.flightMs * (0.6 + 0.4 * range / (p.baseRange + p.rangePerPower)));
  return { x: range * Math.sin(l.angle), z: range * Math.cos(l.angle), ms };
}

/** Where it is `t` ms after launch (for drawing): x, z and its height above the ground. */
export function flightAt(l: Launch, t: number, p: FlightParams = DEFAULT_FLIGHT): { x: number; z: number; h: number; done: boolean } {
  const end = landing(l, p);
  const s = Math.max(0, Math.min(1, t / end.ms));
  const range = Math.hypot(end.x, end.z);
  return { x: end.x * s, z: end.z * s, h: 4 * p.arc * range * s * (1 - s), done: s >= 1 };
}
