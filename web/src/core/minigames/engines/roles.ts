// Jobs on a team (theme-free): a team game has a fixed list of jobs (the stations of a conveyor, the
// limbs of a robot, the roles of a defence) and whoever turned up. Everyone gets a job, in turn; a
// small team doubles up, a big one shares; a job whose whole crew has dropped out is open to the team.
import type { PlayerId } from '../../types';

/** Who does each of `jobs` jobs: the members in turn; a job nobody drew goes to a member who already has one. */
export function crewFor(members: PlayerId[], jobs: number): PlayerId[][] {
  return Array.from({ length: jobs }, (_, k) => {
    const drawn = members.filter((_m, i) => i % jobs === k);
    return drawn.length ? drawn : [members[k % members.length]];
  });
}

/** May `who` do job `k`: it is theirs, or its whole crew has dropped out and they are on the team. */
export function mayWork(connected: (p: PlayerId) => boolean, crew: PlayerId[][], members: PlayerId[], who: PlayerId, k: number): boolean {
  const mine = crew[k] || [];
  if (mine.includes(who)) return true;
  return members.includes(who) && mine.every((p) => !connected(p));
}

/** The jobs `who` may do right now (their own, and any whose crew has all dropped out). */
export function jobsOf(connected: (p: PlayerId) => boolean, crew: PlayerId[][], members: PlayerId[], who: PlayerId): number[] {
  return crew.map((_c, k) => k).filter((k) => mayWork(connected, crew, members, who, k));
}

/** A job shared by two or more is done in turns (so two on a job are no quicker than one): whose go it
 *  is at job `k`, given how many times each job has been done. */
export const onDuty = (crew: PlayerId[][], turn: number[], k: number): PlayerId => crew[k][(turn[k] || 0) % crew[k].length];
/** May `who` do job `k` now: it is their go; or whoever's go it is has dropped out and they share the
 *  job; or its whole crew has dropped out and they are on the team. */
export function mayTakeTurn(connected: (p: PlayerId) => boolean, crew: PlayerId[][], turn: number[], members: PlayerId[], who: PlayerId, k: number): boolean {
  const due = onDuty(crew, turn, k);
  if (who === due) return true;
  if (connected(due) || !members.includes(who)) return false;
  return crew[k].includes(who) || crew[k].every((p) => !connected(p));
}
