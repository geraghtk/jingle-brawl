// Pass the Gremlin as a screen receives it (gremlin.ts publicView / projectForPlayer). Never the
// answer or the deadline: the room does not send them. No DOM, and nothing from the screens.
import type { Danger, Pass } from './gremlin';

export interface GremlinView {
  stage: 'holding' | 'pass_ready' | 'sudden_death_intro' | 'doomed';
  round: 'main' | 'sudden_death';
  pool: string[]; holder: string; prevHolder: string | null;
  challengeKind: string; challengeLabel?: string; solved: boolean; danger: Danger; trail: Pass[];
  bystanderDoomed: string | null; doomed: string | null;
  // the holder's own challenge
  issuedAt?: number; cooldownUntil?: number | null;
  prompt?: string;                                     // quick_math
  sequence?: number[]; showMs?: number; length?: number; // memory_flash
  startAt?: number; periodMs?: number; band?: [number, number]; // perfect_timing
  canPassAt?: number; targets?: string[];
}
