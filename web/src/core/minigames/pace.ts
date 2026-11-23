// A relaxed pace (settings.pace, the host's choice in Setup): for a table with small children, older
// relatives or anyone who finds the games a rush. Each game says here what "more time" means for it:
// a slower belt, a robot that tips over more slowly, a longer clock. Nothing else about a game changes,
// and a game not listed plays the same at either pace (a boxing match is no easier slowed down).
import type { Settings } from '../types';

export type Pace = 'normal' | 'relaxed';
export const paceOf = (s: Settings | undefined): Pace => (s?.pace === 'relaxed' ? 'relaxed' : 'normal');

/** What each runtime game changes at a relaxed pace (laid over its defaults; anything the caller sets wins). */
export const RELAXED: Record<string, Record<string, unknown>> = {
  // the belt starts slower, speeds up less, and there is longer to fill the sleigh
  gift_conveyor: { baseSpeed: 0.08, speedStep: 0.1, roundMs: 110_000 },
  // the robot tips over more slowly, a mistimed step is forgiven for longer, two legs have longer to jump together
  mecha_santa: { roundMs: 120_000, walk: { leanRate: 0.42, graceMs: 220, jumpWindowMs: 1300 } },
  // longer to practise, longer to throw
  reindeer_ring_toss: { practiceMs: 30_000, roundMs: 55_000 },
  // a wrong answer holds you up for less time
  pass_the_gremlin: { wrongCooldownMs: 400 },
};

/** A runtime game's config at the room's pace: the relaxed changes under whatever the caller already chose. */
export function pacedConfig(settings: Settings | undefined, gameId: string, config?: unknown): unknown {
  const slow = paceOf(settings) === 'relaxed' ? RELAXED[gameId] : undefined;
  if (!slow) return config;
  return { ...slow, ...(config && typeof config === 'object' ? config : {}) };
}

/** A thinking clock (drawing, voting) at the room's pace: half as long again when relaxed. */
export const pacedMs = (settings: Settings | undefined, ms: number): number => (paceOf(settings) === 'relaxed' ? Math.round(ms * 1.5) : ms);
