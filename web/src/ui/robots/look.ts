// Nutcracker Knockout: what both the TV and the phones show the same way (no three.js here: the
// phone never loads the 3D scene).
import type { Part } from '../../core/minigames/robots/robots';
import type { Busy } from '../../core/minigames/engines/fight';

/** The two robots' coats: fighter 0 (left of the TV) and fighter 1. */
export const ROBOT_COLORS: [string, string] = ['#ff3366', '#18b88a'];

export const PART_NAMES: Record<Part, string> = {
  hat: 'hat', left_shoulder: 'left shoulder plate', right_shoulder: 'right shoulder plate', chest_door: 'chest door', key: 'wind-up key', head: 'head',
};

/** Why a robot can't do anything for a moment, in a word for its driver. */
export const BUSY_WORDS: Record<Busy, string> = {
  recover: '', blocked: 'Blocked! You are open', stunned: 'Stunned!', flinch: 'Ouch!', guard_break: 'Guard broken!',
};
