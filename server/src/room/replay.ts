// A room rebuilt from its log: the game it started as, then every committed plan in order. If
// this doesn't give the saved state, something changed the game outside the log.
import { commitPlan } from '../../../web/src/core/commit';
import type { Game, Plan } from '../../../web/src/core/types';
import type { Store } from '../store/store';

export function replay(store: Store, code: string): Game | null {
  const origin = store.meta(`origin:${code}`);
  if (!origin) return null;
  let game: Game = JSON.parse(origin);
  for (const e of store.events(code, 0, Number.MAX_SAFE_INTEGER)) {
    const plan = (e.body as { plan?: Pick<Plan, 'updates' | 'claim'> } | null)?.plan;
    if (e.ok && plan) game = commitPlan(game, { updates: plan.updates, events: [], ...(plan.claim ? { claim: plan.claim } : {}) }, e.at).game;
  }
  return game;
}
