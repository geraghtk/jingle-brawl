// Reindeer Ring Toss as a screen receives it (ringtoss.ts publicView / projectForPlayer): the types
// the TV, the console and the phone share, and the normaliser for a view as it arrives. No DOM, and
// nothing from the screens.
import type { FlightParams } from '../engines/projectile';
import type { Peg } from '../engines/targets';
import type { Stage, Throw } from './ringtoss';

/** A throw as the screens get it: all of the room's Throw but `off` (the last tiebreak's measure). */
export type RingThrow = Omit<Throw, 'off'>;

export interface RingView {
  stage: Stage;
  stageEndsAt: number; pool: string[]; left: Record<string, number>; scores: Record<string, number>; tiebreakRound: number;
  /** The scored round: rings come `batch` at a time; a used-up batch is back at `reloadAt[pid]`. */
  batch: number; reloadMs: number; reloadAt: Record<string, number>;
  /** Each player's targets since their last full set, and their full sets (each worth setBonus). */
  got: Record<string, string[]>; sets: Record<string, number>; setBonus: number;
  practiceThrows: number; flight: FlightParams; pegs: Peg[]; throws: RingThrow[];
  me?: string; myLeft?: number; myScore?: number; nextAt?: number;
}

/** A view as it arrives: empty lists and maps don't travel (the finished game has no rings left). */
export const ringView = (raw: unknown): RingView => {
  const v = (raw || {}) as RingView;
  return { ...v, pool: v.pool || [], left: v.left || {}, scores: v.scores || {}, throws: v.throws || [], pegs: v.pegs || [],
    reloadAt: v.reloadAt || {}, got: v.got || {}, sets: v.sets || {} };
};
