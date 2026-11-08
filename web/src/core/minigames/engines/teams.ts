// Teams, for any captain-led game: the two duelists captain opposing teams and everyone else is
// shared out by the engine (not picked socially): sizes within one of each other, and, where it can,
// people who were teammates before are split up. The experience keeps who-played-with-whom
// (`pairs`) between games. Nothing here knows what the teams will do.
import type { Rng } from '../../rng';
import type { PlayerId } from '../../types';

export type TeamPolicy = 'balanced_random' | 'pure_random';
export interface Team { teamId: 'A' | 'B'; captainId: PlayerId; memberIds: PlayerId[] }   // memberIds include the captain
export interface TeamAssignment { teams: [Team, Team]; policy: TeamPolicy; version: string }

export const TEAMS_VERSION = '1';
const TRIES = 40;

/** A stable key for two people (order doesn't matter). */
export const pairKey = (a: PlayerId, b: PlayerId) => (a < b ? `${a}|${b}` : `${b}|${a}`);

/** How often these teams put past teammates together again. */
export function repeats(teams: Team[], pairs: Record<string, number> = {}): number {
  let n = 0;
  for (const t of teams) {
    for (let i = 0; i < t.memberIds.length; i++) {
      for (let j = i + 1; j < t.memberIds.length; j++) n += pairs[pairKey(t.memberIds[i], t.memberIds[j])] || 0;
    }
  }
  return n;
}

function shuffled<T>(xs: T[], rng: Rng): T[] {
  const a = [...xs];
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}

export function assignTeams(captains: [PlayerId, PlayerId], others: PlayerId[], rng: Rng,
  opts: { policy?: TeamPolicy; pairs?: Record<string, number> } = {}): TeamAssignment {
  const policy = opts.policy || 'balanced_random';
  const rest = others.filter((p) => !captains.includes(p));
  const make = (order: PlayerId[]): [Team, Team] => {
    const a: PlayerId[] = [captains[0]], b: PlayerId[] = [captains[1]];
    for (const p of order) {
      if (policy === 'pure_random') (rng() < 0.5 ? a : b).push(p);
      else (a.length <= b.length ? a : b).push(p);       // balanced: the smaller team takes the next one
    }
    return [{ teamId: 'A', captainId: captains[0], memberIds: a }, { teamId: 'B', captainId: captains[1], memberIds: b }];
  };
  if (policy === 'pure_random') return { teams: make(shuffled(rest, rng)), policy, version: TEAMS_VERSION };
  // balanced: several random deals, the one that repeats the fewest past pairings
  let best = make(shuffled(rest, rng)), bestRepeats = repeats(best, opts.pairs);
  for (let k = 1; k < TRIES && bestRepeats > 0; k++) {
    const t = make(shuffled(rest, rng));
    const r = repeats(t, opts.pairs);
    if (r < bestRepeats) { best = t; bestRepeats = r; }
  }
  return { teams: best, policy, version: TEAMS_VERSION };
}

/** Remember who played together (the experience stores this between games). */
export function recordPairs(teams: Team[], pairs: Record<string, number> = {}): Record<string, number> {
  const out = { ...pairs };
  for (const t of teams) {
    for (let i = 0; i < t.memberIds.length; i++) {
      for (let j = i + 1; j < t.memberIds.length; j++) { const k = pairKey(t.memberIds[i], t.memberIds[j]); out[k] = (out[k] || 0) + 1; }
    }
  }
  return out;
}

/** Validate an assignment handed to a game (from config): two teams, the captains, nobody twice. */
export function checkAssignment(raw: unknown, captains?: [PlayerId, PlayerId]): TeamAssignment | null {
  const a = raw as TeamAssignment | null;
  if (!a || !Array.isArray(a.teams) || a.teams.length !== 2) return null;
  const seen = new Set<PlayerId>();
  for (const t of a.teams) {
    if (!t || typeof t.captainId !== 'string' || !Array.isArray(t.memberIds) || !t.memberIds.includes(t.captainId)) return null;
    for (const m of t.memberIds) { if (typeof m !== 'string' || seen.has(m)) return null; seen.add(m); }
  }
  if (captains && !(a.teams[0].captainId === captains[0] && a.teams[1].captainId === captains[1])) return null;
  return a;
}
