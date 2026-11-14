// The Great Gift Conveyor as a screen receives it (conveyor.ts publicView / projectForPlayer): both
// factories for everyone, and a phone's own team, station and job. A view drops nulls: compare with
// `!= null`. No DOM, and nothing from the screens.
import type { Gift, Order, StepKind } from './conveyor';

export interface ConveyorTeamView {
  teamId: 'A' | 'B'; captainId: string; members: string[]; delivered: number; wrong: number;
  stations: { steps: StepKind[]; workers: string[] }[];
  gifts: { id: string; step: number; station: number; working: boolean; stowaway: number | null }[];
}
export interface ConveyorView {
  stage: 'running' | 'overtime' | 'done'; endsAt: number; quota: number; winner: 'A' | 'B' | null; steps: readonly StepKind[];
  teams: ConveyorTeamView[];
  myTeam?: 'A' | 'B' | null; myStation?: number;
  job?: { item: string; step: StepKind; order: Omit<Order, 'options'>; options: Order['options']; gift: Gift } | null;
}
