// Ring Toss's show: runs of ringers, the leader, the last seconds, the one target a set still needs.
import { describe, expect, it } from 'vitest';
import { finalCount, lastNeeded, leaderOf, streakAt, streakCall, streakPitch } from './flair';
import { PEGS } from './ringtoss';

const t = (n: number, pid: string, pegId: string | null, stage = 'throwing') => ({ n, pid, stage, pegId });

describe('Ring Toss flair', () => {
  it('a run is your own scored ringers in a row: a miss ends it, others and practice do not', () => {
    const throws = [t(1, 'a', 'golden', 'practice'), t(2, 'a', null), t(3, 'a', 'golden'), t(4, 'b', null), t(5, 'a', 'left-inner'), t(6, 'a', 'golden', 'practice'), t(7, 'a', 'right-outer')];
    expect(streakAt(throws, 'a')).toBe(3);
    expect(streakAt(throws, 'a', 5)).toBe(2);            // as it stood when ring 5 landed
    expect(streakAt(throws, 'b')).toBe(0);
    expect(streakAt([...throws, t(8, 'a', null)], 'a')).toBe(0);
    expect(streakAt([], 'a')).toBe(0);
  });

  it('a run is called from three, louder from five, and its chime climbs to a cap', () => {
    expect(streakCall(2)).toBe('');
    expect(streakCall(3)).toBe('3 in a row!');
    expect(streakCall(6)).toBe('On fire! 6 in a row!');
    expect(streakPitch(0)).toBe(1);
    expect(streakPitch(1)).toBe(1);
    expect(streakPitch(3)).toBeGreaterThan(streakPitch(2));
    expect(streakPitch(50)).toBe(streakPitch(8));
  });

  it('the leader is the one player ahead: nobody while level or before anyone scores', () => {
    expect(leaderOf({ a: 2, b: 1 }, ['a', 'b', 'c'])).toBe('a');
    expect(leaderOf({ a: 2, b: 2 }, ['a', 'b'])).toBeNull();
    expect(leaderOf({}, ['a', 'b'])).toBeNull();
  });

  it('only the scored round counts down, and only its last five seconds', () => {
    expect(finalCount('throwing', 10_000, 4_000)).toBeNull();
    expect(finalCount('throwing', 10_000, 5_000)).toBe(5);
    expect(finalCount('throwing', 10_000, 9_100)).toBe(1);
    expect(finalCount('throwing', 10_000, 10_000)).toBeNull();
    expect(finalCount('practice', 10_000, 9_000)).toBeNull();
    expect(finalCount('tiebreak', 10_000, 9_000)).toBeNull();
  });

  it('one target short of a set: which one', () => {
    const all = PEGS.map((p) => p.id);
    expect(lastNeeded(PEGS, all.filter((id) => id !== 'golden'))).toBe('golden');
    expect(lastNeeded(PEGS, all.slice(0, 3))).toBeNull();
    expect(lastNeeded(PEGS, [])).toBeNull();
  });
});
