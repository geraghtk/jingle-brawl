// The team engines on their own: balanced, captain-led teams that split past teammates where they
// can; a production line where each item has one holder, a station works one at a time, and a
// dropped phone's item goes to the next person.
import { describe, expect, it } from 'vitest';
import { rng } from '../../rng';
import { assignTeams, checkAssignment, pairKey, recordPairs, repeats } from './teams';
import { add, complete, dispatch, holding, layout, release } from './workflow';

const people = (n: number) => Array.from({ length: n }, (_, i) => `p${i + 1}`);

describe('teams', () => {
  it('captains on opposite sides; everyone placed once; sizes within one, for any table', () => {
    for (let n = 2; n <= 13; n++) {
      const all = people(n);
      const { teams } = assignTeams(['p1', 'p2'], all, rng(n));
      expect(teams.map((t) => t.captainId)).toEqual(['p1', 'p2']);
      expect([...teams[0].memberIds, ...teams[1].memberIds].sort()).toEqual([...all].sort());
      expect(Math.abs(teams[0].memberIds.length - teams[1].memberIds.length)).toBeLessThanOrEqual(1);
    }
  });

  it("splits up last game's teammates where it can", () => {
    const all = people(8);
    const first = assignTeams(['p1', 'p2'], all, rng(1)).teams;
    const pairs = recordPairs(first);
    const second = assignTeams(['p1', 'p2'], all, rng(2), { pairs }).teams;
    // many deals are possible: the one picked repeats fewer pairings than a fixed reshuffle typically would
    const naive = assignTeams(['p1', 'p2'], all, rng(2)).teams;
    expect(repeats(second, pairs)).toBeLessThanOrEqual(repeats(naive, pairs));
    expect(repeats(second, pairs)).toBeLessThan(repeats(first, pairs));
    expect(pairs[pairKey('p1', first[0].memberIds[1])]).toBe(1);
  });

  it('a deal handed to a game is checked: two teams, captains in place, nobody twice', () => {
    const deal = assignTeams(['p1', 'p2'], people(5), rng(3));
    expect(checkAssignment(deal, ['p1', 'p2'])).toBe(deal);
    expect(checkAssignment(deal, ['p2', 'p1'])).toBeNull();
    const twice = { ...deal, teams: [deal.teams[0], { ...deal.teams[1], memberIds: [...deal.teams[1].memberIds, 'p1'] }] };
    expect(checkAssignment(twice)).toBeNull();
    expect(checkAssignment({ teams: [] })).toBeNull();
  });
});

describe('the production line', () => {
  const steps = ['a', 'b', 'c', 'd'];

  it('lays steps over stations evenly; with more people than stations they share, never one person on two', () => {
    expect(layout(steps, 4, people(4)).stations).toEqual([[0], [1], [2], [3]]);
    expect(layout(steps, 2, people(2)).stations).toEqual([[0, 1], [2, 3]]);
    expect(layout(steps, 1, people(1)).stations).toEqual([[0, 1, 2, 3]]);
    expect(layout(steps, 2, people(3)).workers).toEqual([['p1', 'p3'], ['p2']]);
  });

  it('one holder per item; only the holder can do its step; a resend finds it moved on', () => {
    const line = layout(steps, 2, ['p1', 'p2']);
    let floor = dispatch(line, add({ items: {}, rr: [] }, 'x-0', { v: 0 }));
    expect(floor.items['x-0'].holder).toBe('p1');
    expect(complete(line, floor, 'x-0', 'p2', { v: 1 })).toEqual({ error: "That one isn't on your table" });
    let r = complete(line, floor, 'x-0', 'p1', { v: 1 });
    floor = (r as { floor: typeof floor }).floor;
    expect(floor.items['x-0']).toMatchObject({ step: 1, holder: 'p1' });          // the same station's next step: same hands
    floor = (complete(line, floor, 'x-0', 'p1', { v: 2 }) as { floor: typeof floor }).floor;
    expect(floor.items['x-0']).toMatchObject({ step: 2, holder: 'p2' });          // on to the next station
    r = complete(line, floor, 'x-0', 'p1', { v: 3 });
    expect(r).toEqual({ error: "That one isn't on your table" });
  });

  it('a station works one item at a time; the next waits in its tray, oldest first', () => {
    const line = layout(steps, 4, people(4));
    let floor = dispatch(line, add(add({ items: {}, rr: [] }, 'x-0', {}), 'x-1', {}));
    expect(Object.values(floor.items).filter((it) => it.holder).length).toBe(1);
    expect(floor.items['x-0'].holder).toBe('p1');
    floor = (complete(line, floor, 'x-0', 'p1', {}) as { floor: typeof floor }).floor;
    expect(floor.items['x-1'].holder).toBe('p1');
    expect(holding(floor, 'p2')!.id).toBe('x-0');
  });

  it("a dropped phone's item goes back to its station and on to someone there who's still here", () => {
    const line = layout(steps, 2, ['p1', 'p2', 'p3']);        // station 0: p1 and p3
    let floor = dispatch(line, add({ items: {}, rr: [] }, 'x-0', {}));
    expect(floor.items['x-0'].holder).toBe('p1');
    floor = release(line, floor, (p) => p === 'p1', (p) => p !== 'p1');
    expect(floor.items['x-0'].holder).toBe('p3');
    // nobody left at the station: it waits
    floor = release(line, floor, (p) => p === 'p3', () => false);
    expect(floor.items['x-0'].holder).toBeNull();
  });
});
