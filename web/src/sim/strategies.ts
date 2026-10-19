// Strategy profiles for the economic simulator. Each is a small, explicit decision policy, so a
// result can be traced back to a rule ("passive never bids and Gambits instead").
import type { Rng } from '../core/rng';

export interface View {
  id: string;
  chips: number;
  maxBid: number;
  baseCost: number;
  holding: number;        // value (to me) of the gift I hold now, 0 if none
  hasGift: boolean;
  progress: number;       // share of gifts opened so far
  endgame: boolean;
  rng: Rng;
  value: (giftId: string) => number; // what a gift is worth to me (0–12)
}
export interface Target { id: string; cost: number }

export interface Strategy {
  name: string;
  bid(v: View, giftId: string): number;
  /** Nobody bid on my opened gift: Gambit a target (giftId) or keep (null). */
  gambit(v: View, targets: Target[]): string | null;
  /** Two or more bids on my opened gift: fight (true) or yield (false). */
  fight(v: View, giftId: string): boolean;
  reprisal(v: View, targets: Target[]): string | null;
  /** Endgame: steal a gift (giftId) or go to the Misfit auction (null). */
  steal(v: View, targets: Target[], misfitId: string): string | null;
}

/** Best target by gain over what I hold, if the gain beats `costWeight` × cost. */
function bestTarget(v: View, targets: Target[], costWeight: number, over = v.holding): string | null {
  let best: string | null = null;
  let bestNet = 0;
  for (const t of targets) {
    const net = v.value(t.id) - over - costWeight * t.cost;
    if (net > bestNet) { bestNet = net; best = t.id; }
  }
  return best;
}

const common = (costWeight: number): Pick<Strategy, 'fight' | 'reprisal' | 'steal'> => ({
  // keep a good gift; let them fight over a poor one (I go back in the draw bag)
  fight: (v, giftId) => v.value(giftId) >= 5,
  reprisal: (v, targets) => bestTarget(v, targets, costWeight),
  steal: (v, targets, misfitId) => bestTarget(v, targets, costWeight, v.value(misfitId)),
});

/** Bids `k` chips on anything better than what it holds. */
export const aggressive = (k: number): Strategy => ({
  name: `bid ${k}`,
  bid: (v, giftId) => (v.value(giftId) > v.holding + 0.5 ? Math.min(k, v.maxBid, v.chips) : 0),
  gambit: (v, targets) => bestTarget(v, targets, 0.5),
  ...common(0.5),
});

/** Never bids. Waits to open a gift nobody wants, then Gambits for a better one. */
export const passive: Strategy = {
  name: 'passive',
  bid: () => 0,
  gambit: (v, targets) => bestTarget(v, targets, 0.3),
  ...common(0.3),
};

/** Bids in proportion to how much better the gift is (1 chip per 2 points of gain). */
export const valueScaled: Strategy = {
  name: 'value-scaled',
  bid: (v, giftId) => Math.max(0, Math.min(Math.round((v.value(giftId) - v.holding) / 2), v.maxBid, v.chips)),
  gambit: (v, targets) => bestTarget(v, targets, 0.5),
  ...common(0.5),
};

/** Coin-flip everything: a baseline. */
export const random: Strategy = {
  name: 'random',
  bid: (v) => (v.rng() < 0.5 ? Math.floor(v.rng() * (v.maxBid + 1)) : 0),
  gambit: (v, targets) => (targets.length && v.rng() < 0.5 ? targets[Math.floor(v.rng() * targets.length)].id : null),
  fight: (v) => v.rng() < 0.5,
  reprisal: (v, targets) => (targets.length && v.rng() < 0.5 ? targets[Math.floor(v.rng() * targets.length)].id : null),
  steal: (v, targets) => (targets.length && v.rng() < 0.5 ? targets[Math.floor(v.rng() * targets.length)].id : null),
};

export const ALL = { 'bid 1': aggressive(1), 'bid 2': aggressive(2), 'bid 3': aggressive(3), 'bid 4': aggressive(4), passive, valueScaled, random };
