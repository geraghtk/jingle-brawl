import { describe, expect, it } from 'vitest';
import type { ConveyorToyView } from '../core/minigames/conveyor/view';
import { binned, lineChanges } from './conveyorFactory';

const toy = (id: string, pos: number, fitted: number[], want = [0, 1, 2, 0], stowaway = false): ConveyorToyView => ({ id, pos, want, fitted, stowaway });
const NONE = -1;

describe('the Gift Conveyor on the TV: what just happened, read off two views', () => {
  it('a piece that has gone on: which toy, which station, right or wrong', () => {
    const was = { toys: [toy('A-1', 0.3, [NONE, NONE, NONE, NONE]), toy('A-2', 0.1, [NONE, NONE, NONE, NONE])], stolen: 0 };
    const now = { toys: [toy('A-1', 0.3, [0, NONE, NONE, NONE]), toy('A-2', 0.1, [NONE, 2, NONE, NONE])], stolen: 0 };
    expect(lineChanges(was, now)).toEqual([{ kind: 'good', k: 0, toy: 'A-1' }, { kind: 'bad', k: 1, toy: 'A-2' }]);
    expect(lineChanges(now, now)).toEqual([]);
  });
  it('a raccoon turns up; a new toy is nothing to shout about', () => {
    const was = { toys: [toy('A-1', 0.3, [])], stolen: 0 };
    const now = { toys: [toy('A-1', 0.3, []), toy('A-2', 0, []), toy('A-3', 0, [], [], true)], stolen: 0 };
    expect(lineChanges(was, now)).toEqual([{ kind: 'raccoon', toy: 'A-3' }]);
  });
  it('a raccoon that is gone was shooed, unless it got to the end and took a toy', () => {
    const racc = toy('A-3', 0.5, [], [], true);
    const was = { toys: [toy('A-1', 0.8, []), racc], stolen: 1 };
    expect(lineChanges(was, { toys: [toy('A-1', 0.8, [])], stolen: 1 })).toEqual([{ kind: 'shooed', toy: 'A-3', ghost: racc }]);
    expect(lineChanges(was, { toys: [toy('A-1', 0.8, [])], stolen: 2 })).toEqual([]);
    // a toy that left the belt is not a raccoon
    expect(lineChanges(was, { toys: [racc], stolen: 1 })).toEqual([]);
  });
  it('the toy that went into the bin is the one that left the belt (never the raccoon)', () => {
    const a = toy('A-1', 0.97, [0, 0, 0, 0]), b = toy('A-2', 0.6, []), r = toy('A-3', 0.99, [], [], true);
    expect(binned([a, b, r], [b])).toEqual(a);
    expect(binned([a, b], [a, b])).toBeNull();
    expect(binned([r], [])).toBeNull();
  });
});
