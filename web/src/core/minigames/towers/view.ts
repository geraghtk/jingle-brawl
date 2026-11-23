// Elf Tower Wars as a screen receives it (towers.ts publicView / projectForPlayer). Everything is
// public: both workshops, every toy and tower, every player's energy. A view drops empty lists and
// maps, so a screen reads it through `towerView`. No DOM, and nothing from the screens.
import { energyAt, wholeField } from '../engines/lanes';
import type { Field, LaneParams, Purse } from '../engines/lanes';

/** What a team's energy looks like from outside: each player's own purse, who is here to earn, and at what rate. */
export interface TeamEnergy {
  /** Each player's own energy, as of its `asOf` (a screen works out now from the rate). */
  purses: Record<string, Purse>;
  /** Who is earning (a phone that has dropped out earns nothing: its share goes to the rest). */
  active: string[];
  /** Energy a second for each of them. */
  rate: number;
}
export interface TowerTeamView extends TeamEnergy { teamId: 'A' | 'B'; captainId: string; members: string[] }
export interface TowerView {
  stage: 'running' | 'overtime' | 'done';
  startedAt: number; endsAt: number;
  /** From here to the end everyone's energy comes `rush` times as fast (the finale). */
  rushAt: number; rush: number;
  winner: 'A' | 'B' | null;
  rules: LaneParams;
  /** The battle as of `field.asOf`: side 0 is team A (its workshop on the left), side 1 team B. */
  field: Field;
  teams: TowerTeamView[];
  // a phone's own
  myTeam?: 'A' | 'B' | null;
}

/** How much energy `pid` has at `now`. */
export function energyOf(t: TeamEnergy, pid: string, now: number, rules: LaneParams, finale?: { rushAt: number; rush: number }): number {
  const u = t.purses?.[pid];
  if (!u) return 0;
  const rate = (t.active || []).includes(pid) ? t.rate : 0;
  // the time since the purse was last counted, with whatever of it is in the finale counted `rush` times
  const extra = finale && finale.rush > 1 ? Math.max(0, now - Math.max(u.asOf, finale.rushAt)) * (finale.rush - 1) : 0;
  return energyAt(u, now + extra, { ...rules.purse, rate });
}

/** A view as sent, with the lists and maps a view drops put back. */
export function towerView(raw: unknown): TowerView {
  const v = raw as TowerView;
  return {
    ...v, winner: v.winner ?? null, rushAt: v.rushAt ?? v.endsAt, rush: v.rush || 1, field: wholeField(v.field, v.rules),
    teams: (v.teams || []).map((t) => ({ ...t, members: t.members || [], purses: t.purses || {}, active: t.active || [], rate: t.rate || 0 })),
  };
}
