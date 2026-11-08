// Future engines, as typed interfaces only (spec M1: declare them now, build them when a game needs
// them). Ring Toss (M4) brings gestures, physics and the shared arena; the Gift Conveyor (M5)
// brings teams and cross-device handoffs.
import type { MatchMode } from './contracts';
import type { PlayerId } from '../types';

/** What a game needs from the event, the devices and the room (checked before it starts). */
export interface CompatibilityManifest {
  version: 1;
  environments: ('in_person' | 'remote' | 'hybrid')[];
  displayRole: 'none' | 'optional_status' | 'required_shared_world';
  topology: ('independent_phone' | 'phone_to_phone' | 'phone_to_shared_display' | 'shared_display_to_phone' | 'physical_world_verified' | 'shared_object_multi_controller')[];
  devices: { feature: 'touch' | 'pointer' | 'camera' | 'motion' | 'haptics' | 'audio' | 'microphone'; required: boolean; fallback?: string }[];
  players: { min: number; max: number };
  modes: MatchMode[];
}

export interface PreflightInput { environment: 'in_person' | 'remote' | 'hybrid'; sharedDisplay: boolean; players: number; mode: MatchMode }
export type Preflight = { ok: true } | { ok: false; reasons: string[] };

/** Preflight: the host is told why a game can't run here, before it starts (not mid-duel). */
export function preflight(m: CompatibilityManifest, input: PreflightInput): Preflight {
  const reasons: string[] = [];
  if (!m.environments.includes(input.environment)) reasons.push(`not for ${input.environment} events`);
  if (m.displayRole === 'required_shared_world' && !input.sharedDisplay) reasons.push('needs the TV screen');
  if (input.players < m.players.min) reasons.push(`needs at least ${m.players.min} players`);
  if (input.players > m.players.max) reasons.push(`at most ${m.players.max} players`);
  if (!m.modes.includes(input.mode)) reasons.push(`can't be played as ${input.mode.replace(/_/g, ' ')}`);
  return reasons.length ? { ok: false, reasons } : { ok: true };
}

// --- not built yet --------------------------------------------------------------------------------

export type TeamPolicy = 'balanced_random' | 'pure_random' | 'captain_draft';
export interface TeamAssignment { teams: { teamId: string; captainId: PlayerId; memberIds: PlayerId[] }[]; policy: TeamPolicy; seed: number }

/** A normalized gesture from a phone (Ring Toss): the server simulates, never trusts a hit. */
export interface GestureIntent { kind: 'swipe'; angle: number; power: number; durationMs: number; screen: { w: number; h: number } }

export interface ArenaPhysics<World, Input> {
  version: string;
  step(world: World, inputs: Input[], dtMs: number): World;
}

export interface SubmissionVoting<Sub> {
  submit(pid: PlayerId, sub: Sub): void;
  vote(voter: PlayerId, target: PlayerId): void;
}

/** A server-owned object moving between screens (the Gift Conveyor): one owner at a time. */
export interface TransferableObject { objectId: string; owner: { kind: 'phone'; pid: PlayerId } | { kind: 'arena' } | { kind: 'transit'; to: string }; at: number }
