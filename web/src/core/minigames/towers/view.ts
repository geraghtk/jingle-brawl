// Elf Tower Wars as a screen receives it (towers.ts publicView / projectForPlayer). Everything is
// public: both workshops, every toy and tower, what each team has lined up for the next march, every
// player's energy. A view drops empty lists and maps, so a screen reads it through `towerView`. No DOM,
// and nothing from the screens.
import { wholeField } from '../engines/lanes';
import type { Field, LaneParams, Purse } from '../engines/lanes';

/** A toy lined up at a workshop door while everyone plans: whose side, which lane, what, who played it. */
export interface Queued { side: number; lane: number; unit: string; by: string }

/** What a team's energy looks like from outside: each player's own purse, and who is here to share a round's energy. */
export interface TeamEnergy {
  purses: Record<string, Purse>;
  /** Who is here (a phone that has dropped out gets no share: it goes to the rest). */
  active: string[];
}
export interface TowerTeamView extends TeamEnergy {
  teamId: 'A' | 'B'; captainId: string; members: string[];
  /** Who has said they are done planning this round. */
  done: string[];
}
export interface TowerView {
  stage: 'running' | 'overtime' | 'done';
  /** `plan`: the field stands still and everyone spends; `march`: the armies are out and everyone watches. */
  phase: 'plan' | 'march';
  round: number; rounds: number;
  phaseAt: number; phaseEndsAt: number;
  startedAt: number;
  winner: 'A' | 'B' | null;
  rules: LaneParams;
  /** The battle as of `field.asOf`: side 0 is team A (its workshop on the left), side 1 team B. */
  field: Field;
  queue: Queued[];
  teams: TowerTeamView[];
  // a phone's own
  myTeam?: 'A' | 'B' | null;
}

/** How much energy `pid` has (it comes a round at a time and only goes down when spent: no clock in it). */
export function energyOf(t: TeamEnergy, pid: string): number {
  return t.purses?.[pid]?.energy || 0;
}
/** The moment a screen should draw the field at: while everyone plans it stands still (as it was when the last
 *  march ended), while it marches it is `now`. */
export const fieldClock = (v: Pick<TowerView, 'phase' | 'stage' | 'field'>, now: number) => (v.phase === 'plan' || v.stage === 'done' ? v.field.asOf : now);

/** A view as sent, with the lists and maps a view drops put back. */
export function towerView(raw: unknown): TowerView {
  const v = raw as TowerView;
  return {
    ...v, winner: v.winner ?? null, phase: v.phase === 'march' ? 'march' : 'plan', round: v.round || 1, rounds: v.rounds || 1, field: wholeField(v.field, v.rules), queue: v.queue || [],
    teams: (v.teams || []).map((t) => ({ ...t, members: t.members || [], purses: t.purses || {}, active: t.active || [], done: t.done || [] })),
  };
}
