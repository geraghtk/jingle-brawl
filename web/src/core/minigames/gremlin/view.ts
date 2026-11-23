// Pass the Gremlin as a screen receives it (gremlin.ts publicView / projectForPlayer). Never the
// answer or the deadline: the room does not send them. No DOM, and nothing from the screens.
import type { Danger, Pass } from './gremlin';

/** The Party points of the bystander caught `i`-th (0 = first): the later, the more, and always less than
 *  never being caught (with the default of 1 for that: nothing). */
export const outPoints = (i: number, survive: number) => Math.max(0, Math.min(i, survive - 1));

export interface GremlinView {
  stage: 'holding' | 'pass_ready' | 'out_intro' | 'sudden_death_intro' | 'doomed';
  round: 'main' | 'sudden_death';
  pool: string[]; holder: string; prevHolder: string | null;
  challengeKind: string; challengeLabel?: string; solved: boolean; danger: Danger; trail: Pass[];
  bystanderDoomed: string | null; doomed: string | null;
  /** Elimination: which round this is, and the bystanders it has caught so far, first out first (views drop an empty list). */
  roundNo?: number; out?: string[];
  /** How many bystanders can be caught before sudden death, and the Party points for one never caught. */
  maxOut?: number; survive?: number;
  /** The table on the big screen (a view from an older room has none of these). */
  holdSince?: number; misses?: number; held?: Record<string, number>;
  // the holder's own challenge
  issuedAt?: number; cooldownUntil?: number | null;
  prompt?: string;                                     // quick_math
  sequence?: number[]; showMs?: number; length?: number; // memory_flash
  startAt?: number; periodMs?: number; band?: [number, number]; // perfect_timing
  canPassAt?: number; targets?: string[];
}
