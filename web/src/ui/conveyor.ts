// The Great Gift Conveyor: the names and colours its phone card and its TV stage share.
import type { PartKind } from '../core/minigames/conveyor/conveyor';

export const TEAM_COLORS: Record<string, string> = { A: '#ff3366', B: '#18b88a' };
/** A station's sign. */
export const PART_NAMES: Record<PartKind, string> = { wheels: 'Wheels', lights: 'Lights', paint: 'Paint', bow: 'Bow' };
/** The kinds of each piece: a name and a colour (and a mark, so colour is never the only clue). */
export const VARIANT_NAMES: Record<PartKind, string[]> = {
  wheels: ['Black', 'Red', 'Gold'], lights: ['Yellow', 'Blue', 'Green'], paint: ['Red', 'Green', 'Blue'], bow: ['Gold', 'Silver', 'Pink'],
};
export const VARIANT_COLORS: Record<PartKind, string[]> = {
  wheels: ['#1d2a3a', '#e23b3b', '#f2b705'], lights: ['#ffd84a', '#38b6e6', '#7cc242'], paint: ['#e23b3b', '#1fa36b', '#2f6fde'], bow: ['#f2b705', '#b9c2d0', '#ff7ab6'],
};
export const VARIANT_MARKS = ['●', '■', '▲'];
/** A toy nobody has painted yet. */
export const PRIMER = '#d5dbe4';
