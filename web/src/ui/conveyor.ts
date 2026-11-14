// The Great Gift Conveyor: the colours and names its phone card and its TV stage share.
import type { StepKind } from '../core/minigames/conveyor/conveyor';

export const PAPER_COLORS: Record<string, string> = { red: '#e23b3b', green: '#1fa36b', gold: '#f2b705', blue: '#2f6fde', purple: '#8a4bd6' };
export const TEAM_COLORS: Record<string, string> = { A: '#ff3366', B: '#18b88a' };
/** A station's work, as its sign says. */
export const STEP_NAMES: Record<StepKind, string> = { pick: 'Pick', wrap: 'Wrap', label: 'Label', check: 'Check & load' };
