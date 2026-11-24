// What the screens are sent of a Nutcracker Knockout fight (robots.ts publicView). Views drop nulls
// and empty maps on the way: normalise with robotView() before reading.
import type { PlayerId } from '../../types';
import type { Busy, FightEvent, PunchKind } from '../engines/fight';
import type { Part, Stage } from './robots';

export interface RobotFighterView {
  health: number;
  /** The parts knocked off so far, first to last (the head is the last there is). */
  lost: Part[];
  /** The part that has been knocked loose and goes with the next blow. */
  loose?: Part;
  busyUntil: number;
  busy?: Busy;
  punch?: { kind: PunchKind; at: number; lands: number };
  guard?: { since: number };
  /** Dodging: punches miss until `until`. */
  dodge?: { at: number; until: number };
}
export interface RobotView {
  stage: Stage;
  stageEndsAt: number;
  round: number;
  roundsToWin: number;
  /** Fighter 0 and fighter 1 (each keeps its colour), and which of them stands on the left of the TV this round. */
  order: [PlayerId, PlayerId];
  left: 0 | 1;
  wins: Record<PlayerId, number>;
  sudden: boolean;
  maxHealth: number;
  /** How many parts a robot has (its health is so many blows for each). */
  parts: number;
  guardMaxMs: number;
  fighters: [RobotFighterView, RobotFighterView];
  events: FightEvent[];
  last?: { round: number; winner: PlayerId; loser: PlayerId; how: 'knockout' | 'time' };
  me?: PlayerId;
  mySeat?: 0 | 1 | null;
}

/** A robot's health as a share of what it started the round with, 0 to 100 (the bars on the TV and the phone). */
export const healthPct = (health: number, max: number) => Math.max(0, Math.min(100, max > 0 ? (health / max) * 100 : 0));

/** A view as it arrives: empty lists and maps don't travel. */
export function robotView(raw: unknown): RobotView {
  const v = (raw || {}) as RobotView;
  const fighter = (f: RobotFighterView | undefined): RobotFighterView => ({ ...(f || { health: 0, busyUntil: 0 }), lost: f?.lost || [], busyUntil: f?.busyUntil || 0 });
  return { ...v, left: v.left === 1 ? 1 : 0, wins: v.wins || {}, events: v.events || [], order: v.order || ['', ''], fighters: [fighter(v.fighters?.[0]), fighter(v.fighters?.[1])] };
}
