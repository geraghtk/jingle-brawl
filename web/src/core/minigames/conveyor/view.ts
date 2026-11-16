// The Great Gift Conveyor as a screen receives it (conveyor.ts publicView / projectForPlayer): both
// belts for everyone, and a phone's own team and stations. A toy's position is as of the belt's
// `asOf`; a screen draws it where it is now (engines/belt posAt) with no message per frame. A view
// drops empty lists and maps: normalise with conveyorView(). No DOM, and nothing from the screens.
import type { Zone } from '../engines/belt';
import type { PartKind } from './conveyor';

export interface ConveyorToyView { id: string; pos: number; want: number[]; fitted: number[]; stowaway: boolean }
export interface ConveyorTeamView {
  teamId: 'A' | 'B'; captainId: string; members: string[];
  /** Who works each station, by `parts`. */
  crew: string[][];
  delivered: number; rejected: number; stolen: number;
  /** How fast the belt runs: its level (shown), and belt lengths a second as of `asOf`. */
  level: number; speed: number; asOf: number;
  toys: ConveyorToyView[];
}
export interface ConveyorView {
  stage: 'running' | 'overtime' | 'done'; endsAt: number; quota: number; winner: 'A' | 'B' | null;
  parts: readonly PartKind[]; variants: number; zones: Zone[];
  teams: ConveyorTeamView[];
  myTeam?: 'A' | 'B' | null;
  /** The stations this phone works (indexes into `parts`). */
  myStations?: number[];
}

/** A view as it arrives: empty lists don't always travel. */
export function conveyorView(raw: unknown): ConveyorView {
  const v = (raw || {}) as ConveyorView;
  return {
    ...v, parts: v.parts || [], zones: v.zones || [], myStations: v.myStations || [],
    teams: (v.teams || []).map((t) => ({
      ...t, members: t.members || [], crew: (v.parts || []).map((_, k) => t.crew?.[k] || []),
      toys: (t.toys || []).map((y) => ({ ...y, want: y.want || [], fitted: y.fitted || [], stowaway: !!y.stowaway })),
    })),
  };
}
