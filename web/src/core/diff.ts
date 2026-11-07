// What changed between two versions of a game, as the same multi-path update a plan uses
// ({ 'players/p2/connected': true, 'currentTurn/bids': null, … }). The room server sends each
// screen only this, not its whole view again: a drawing goes over the wire once, not on every vote.
// applyUpdates(prev, diff(prev, next)) equals next.
import type { Updates } from './types';

const isObj = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);

export function diff(prev: unknown, next: unknown, path = '', out: Updates = {}): Updates {
  if (isObj(prev) && isObj(next)) {
    for (const k of new Set([...Object.keys(prev), ...Object.keys(next)])) {
      diff(prev[k], next[k], path ? `${path}/${k}` : k, out);
    }
    return out;
  }
  if (prev === next) return out;
  if (prev !== undefined && next !== undefined && JSON.stringify(prev) === JSON.stringify(next)) return out;
  if (!path) throw new Error('diff: the whole game changed shape');
  out[path] = next === undefined ? null : next;
  return out;
}
