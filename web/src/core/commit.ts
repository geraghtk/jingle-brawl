// Committing a plan to a game in memory: the one place claim semantics live (the room server,
// the simulator and the tests all commit through here).
import { applyUpdates, getPath } from './tx';
import type { Game, Plan } from './types';

/** Apply a plan. Refused plans and lost claims (the path is already set) change nothing. */
export function commitPlan(game: Game, plan: Plan, now: number): { game: Game; applied: boolean } {
  if (plan.error) return { game, applied: false };
  if (plan.claim && getPath(game, plan.claim.path) != null) return { game, applied: false };
  const claimed = plan.claim ? applyUpdates(game, { [plan.claim.path]: plan.claim.value }, now) : game;
  return { game: applyUpdates(claimed, plan.updates, now), applied: true };
}
