// Test helpers: play a game in memory by applying plans exactly as Firebase would.
import { commitPlan } from './commit';
import { rng } from './rng';
import type { Game, Plan, Settings } from './types';

export const NOW = 1_700_000_000_000;
export const ctx = (seed = 1) => ({ now: NOW, rng: rng(seed) });

/** Apply a plan; fails the test on a refused plan; honours claims like the Firebase adapter. */
export function apply(game: Game, plan: Plan): Game {
  if (plan.error) throw new Error(`plan refused: ${plan.error}`);
  return commitPlan(game, plan, NOW).game; // claim lost → nothing happens
}

/** A started game with players p1..pN (10 chips each unless told otherwise). */
export function table(n: number, settings: Settings = {}, chips = 10): Game {
  const players: Game['players'] = {};
  for (let i = 1; i <= n; i++) players[`p${i}`] = { name: `P${i}`, chips };
  // duelPick off: tests call duels directly unless they test picking the duel game
  return { phase: 'main', settings: { duelPick: 'off', ...settings }, players, gifts: {} };
}

export const chipsOf = (g: Game) => Object.fromEntries(Object.entries(g.players || {}).map(([id, p]) => [id, p.chips]));
