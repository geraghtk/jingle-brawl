// Mecha Santa Mayhem as a screen receives it (mecha.ts publicView / projectForPlayer). Everything is
// public: both robots, the course, who has which job. A view drops empty lists and maps, so a screen
// reads it through `mechaView`. No DOM, and nothing from the screens.
import { wholeWalker } from '../engines/walker';
import type { Course, ObstacleKind, WalkParams, Walker } from '../engines/walker';

/** The jobs on a robot, in order (a team's `crew[k]` does job k). */
export const JOBS = ['left', 'right', 'balance', 'arms'] as const;
export type Job = (typeof JOBS)[number];
export const JOB_NAMES: Record<Job, string> = { left: 'Left leg', right: 'Right leg', balance: 'Balance', arms: 'Arms' };
/** Which jobs each thing on the course takes (JOBS order): whose phones are told it is coming. */
export const NEEDS: Record<ObstacleKind, number[]> = { wall: [3], boost: [3], gap: [0, 1], climb: [3, 0, 1, 2], rope: [3, 0, 1, 2], ice: [2] };

export interface MechaTeamView {
  teamId: 'A' | 'B'; captainId: string; members: string[]; crew: string[][];
  /** This robot's own numbers. */
  walk: WalkParams;
  /** The robot as it stood at the moment of this view (a screen works out now from its times). */
  walker: Walker;
  /** What it is coming up to (or is in the middle of), once that is close. */
  ahead?: { id: string; kind: ObstacleKind } | null;
}
export interface MechaView {
  stage: 'running' | 'overtime' | 'done';
  startedAt: number; endsAt: number;
  winner: 'A' | 'B' | null;
  course: Course;
  teams: MechaTeamView[];
  // a phone's own
  myTeam?: 'A' | 'B' | null;
  myJobs?: number[];
}

/** A view as sent, with the lists and maps a view drops put back. */
export function mechaView(raw: unknown): MechaView {
  const v = raw as MechaView;
  return {
    ...v, winner: v.winner ?? null, course: { ...v.course, obstacles: v.course?.obstacles || [] },
    teams: (v.teams || []).map((t) => ({ ...t, members: t.members || [], crew: JOBS.map((_, k) => t.crew?.[k] || []), walker: wholeWalker(t.walker) })),
    myJobs: v.myJobs || [],
  };
}
