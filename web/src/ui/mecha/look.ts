// Mecha Santa Mayhem: what the TV and the phones show the same way (no three.js here: a phone never
// loads the 3D scene).
import type { Busy, WalkEventKind } from '../../core/minigames/engines/walker';
import type { Job } from '../../core/minigames/mecha/view';

/** Team A's and team B's coats (the same two colours as every team game). */
export const MECHA_COLORS: [string, string] = ['#ff3366', '#18b88a'];
/** A job in a word or two, and its sign on a chip. */
export const JOB_SHORT: Record<Job, string> = { left: 'L leg', right: 'R leg', balance: 'Balance', arms: 'Arms' };
export const JOB_SIGN: Record<Job, string> = { left: 'L', right: 'R', balance: '⇄', arms: '✊' };

/** The big word for what a robot has just done (nothing for the ordinary steps). */
export const EVENT_WORDS: Partial<Record<WalkEventKind, string>> = {
  stumble: 'Stumble!', topple: 'Timber!', smashed: 'Smash!', jump: 'Jump!', misjump: 'Not together!', fell: 'Down the gap!', boost: 'Whoosh!', saved: 'Saved!', stride: '', finish: 'Over the line!',
};
/** Why a robot can do nothing for a moment, for its crew. */
export const BUSY_WORDS: Record<Busy, string> = { stumble: 'Stumbled! Steady…', topple: 'It fell over! Getting up…', misjump: 'Jump together!', climb: 'Climbing out of the gap…' };
