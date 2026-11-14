// What the screens are sent of a Nutcracker Knockout fight (robots.ts publicView). Views drop nulls
// and empty maps on the way: normalise with robotView() before reading.
import type { PlayerId } from '../../types';
import type { Busy, FightEvent, Hand } from '../engines/fight';
import type { Part, Stage } from './robots';

export interface RobotFighterView {
  health: number;
  /** The parts knocked off so far, first to last (the head is the last there is). */
  lost: Part[];
  busyUntil: number;
  busy?: Busy;
  punch?: { hand: Hand; at: number; lands: number };
  guard?: { since: number };
}
export interface RobotView {
  stage: Stage;
  stageEndsAt: number;
  round: number;
  roundsToWin: number;
  /** Fighter 0 (left of the TV) and fighter 1. */
  order: [PlayerId, PlayerId];
  wins: Record<PlayerId, number>;
  sudden: boolean;
  maxHealth: number;
  windupMs: number;
  guardMaxMs: number;
  fighters: [RobotFighterView, RobotFighterView];
  events: FightEvent[];
  last?: { round: number; winner: PlayerId; loser: PlayerId; how: 'knockout' | 'time' };
  me?: PlayerId;
  mySeat?: 0 | 1 | null;
}

/** A view as it arrives: empty lists and maps don't travel. */
export function robotView(raw: unknown): RobotView {
  const v = (raw || {}) as RobotView;
  const fighter = (f: RobotFighterView | undefined): RobotFighterView => ({ ...(f || { health: 0, busyUntil: 0 }), lost: f?.lost || [], busyUntil: f?.busyUntil || 0 });
  return { ...v, wins: v.wins || {}, events: v.events || [], order: v.order || ['', ''], fighters: [fighter(v.fighters?.[0]), fighter(v.fighters?.[1])] };
}
