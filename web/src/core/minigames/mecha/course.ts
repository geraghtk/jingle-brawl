// A new rooftop course for every race (Kevin, 2026-11-23). Both teams run the same one. Every course
// has the same amount of trouble on it (a budget in rough seconds lost), laid out by the same rules, so
// one race is about as long as another and no course is unfair: what changes is which obstacle comes
// when. Pure: the same numbers from `rng` give the same course.
import type { Rng } from '../../rng';
import type { Obstacle, ObstacleKind } from '../engines/walker';

/** What is in the way, and roughly what each costs a team that does it well (seconds). */
export const TROUBLE: Record<Exclude<ObstacleKind, 'boost'>, number> = { wall: 2, gap: 2, ice: 1, climb: 3, rope: 3 };
/** The two that need three jobs to talk: never one straight after the other. */
const HARD = new Set<ObstacleKind>(['climb', 'rope']);
export const WIDTH: Partial<Record<ObstacleKind, number>> = { gap: 4, rope: 7, ice: 9 };

export const COURSE_RULES = {
  /** Nothing before this, so every team gets walking first; nothing after `length - tail`. */
  head: 10, tail: 9,
  /** How many things are in the way, and the trouble they add up to. */
  features: 6, budget: 13,
  /** Rocket candles to grab. */
  boosts: 3,
  /** Clear roof after one thing ends before the next begins. */
  space: 4,
};

/** The kinds in the way, in order: one of each, the rest to make up the budget, no two hard ones in a row. */
function lineUp(rng: Rng): Exclude<ObstacleKind, 'boost'>[] {
  const R = COURSE_RULES;
  const kinds: Exclude<ObstacleKind, 'boost'>[] = ['wall', 'gap', 'ice', 'climb', 'rope'];
  let left = R.budget - kinds.reduce((n, k) => n + TROUBLE[k], 0);
  while (kinds.length < R.features) {
    const slots = R.features - kinds.length;
    // what still fits: enough left for the cheapest in every later slot, and not more than is left
    const fits = (Object.keys(TROUBLE) as (keyof typeof TROUBLE)[]).filter((k) => TROUBLE[k] <= left - (slots - 1) && (slots > 1 || TROUBLE[k] === left));
    const pick = fits.length ? fits[Math.floor(rng() * fits.length)] : 'ice';
    kinds.push(pick);
    left -= TROUBLE[pick];
  }
  // shuffle, then mend: a hard one straight after a hard one swaps with the next easy one further on (or back)
  for (let i = kinds.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [kinds[i], kinds[j]] = [kinds[j], kinds[i]]; }
  for (let pass = 0; pass < kinds.length; pass++) {
    const i = kinds.findIndex((k, n) => n > 0 && HARD.has(k) && HARD.has(kinds[n - 1]));
    if (i < 0) break;
    let j = kinds.findIndex((k, n) => n > i && !HARD.has(k) && !HARD.has(kinds[n + 1] ?? 'wall'));
    if (j < 0) j = kinds.findIndex((k, n) => n < i - 1 && !HARD.has(k) && (n === 0 || !HARD.has(kinds[n - 1])));
    if (j < 0) break;
    [kinds[i], kinds[j]] = [kinds[j], kinds[i]];
  }
  return kinds;
}

/** A course `length` long. */
export function makeCourse(rng: Rng, length: number): Obstacle[] {
  const R = COURSE_RULES;
  const kinds = lineUp(rng);
  const widths = kinds.map((k) => WIDTH[k] || 0);
  // the roof that is left over once everything is laid end to end with its clear space is shared out at random
  const room = length - R.head - R.tail - widths.reduce((a, b) => a + b, 0) - R.space * (kinds.length - 1);
  const shares = kinds.map(() => 0.4 + rng());
  const total = shares.reduce((a, b) => a + b, 0) + 0.7;                 // (the rest is left before the line)
  const out: Obstacle[] = [];
  let at = R.head;
  kinds.forEach((kind, i) => {
    at += (Math.max(0, room) * shares[i]) / total;
    const o: Obstacle = { id: `${kind[0]}${i + 1}`, kind, at: Math.round(at * 2) / 2, ...(widths[i] ? { width: widths[i] } : {}) };
    out.push(o);
    at = o.at + widths[i] + R.space;
  });
  // the candles: one in the run-up to the first hard thing, the others in the longest clear runs
  const runs = out.map((o, i) => { const from = i ? out[i - 1].at + (out[i - 1].width || 0) : 2; return { from, to: o.at, hard: HARD.has(o.kind) }; }).filter((r) => r.to - r.from >= 3.5);
  const firstHard = runs.find((r) => r.hard);
  const picked = [...(firstHard ? [firstHard] : []), ...runs.filter((r) => r !== firstHard).sort((a, b) => (b.to - b.from) - (a.to - a.from))].slice(0, R.boosts);
  picked.sort((a, b) => a.from - b.from).forEach((r, i) => out.push({ id: `b${i + 1}`, kind: 'boost', at: Math.round((r.from + (r.to - r.from) * 0.4) * 2) / 2 }));
  return out.sort((a, b) => a.at - b.at);
}

/** Does a course keep the rules (for the tests, and for a course given by hand)? Empty: it does. */
export function courseFaults(obstacles: Obstacle[], length: number): string[] {
  const R = COURSE_RULES, faults: string[] = [];
  const way = obstacles.filter((o) => o.kind !== 'boost').sort((a, b) => a.at - b.at);
  way.forEach((o, i) => {
    if (o.at < R.head) faults.push(`${o.id} is too near the start`);
    if (o.at + (o.width || 0) > length - R.tail + 1e-6) faults.push(`${o.id} is too near the line`);
    const next = way[i + 1];
    if (next && next.at - (o.at + (o.width || 0)) < R.space - 1e-6) faults.push(`${o.id} and ${next.id} are too close`);
    if (next && HARD.has(o.kind) && HARD.has(next.kind)) faults.push(`${o.id} and ${next.id}: two hard ones in a row`);
  });
  if (new Set(obstacles.map((o) => o.id)).size !== obstacles.length) faults.push('two with the same id');
  return faults;
}
