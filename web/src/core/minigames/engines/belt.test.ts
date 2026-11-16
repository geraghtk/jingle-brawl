// The belt engine: things enter a fixed distance apart, ride at the belt's speed, and fall off the
// end; the room knows where everything is at any moment from a position, a time and a speed.
import { describe, expect, it } from 'vitest';
import { advance, atSpeed, evenZones, newBelt, next, nextAt, posAt, zoneOf } from './belt';
import type { Belt, BeltEvent } from './belt';

const make = (n: number) => ({ id: `t${n}`, data: n });
/** Run a belt to `now`, collecting what happened. */
function run(b: Belt<number>, now: number, gap = 0.5): { belt: Belt<number>; events: BeltEvent<number>[] } {
  const events: BeltEvent<number>[] = [];
  for (let i = 0; i < 1000; i++) {
    const r = next(b, now, gap, make);
    b = r.belt;
    if (!r.event) break;
    events.push(r.event);
  }
  return { belt: b, events };
}

describe('the belt engine', () => {
  it('the first thing enters at once; the rest a fixed distance apart, so a faster belt brings them sooner', () => {
    const slow = run(newBelt<number>(0, 0.1), 20_000);               // 0.1 belt a second: 0.5 apart is 5 s apart
    expect(slow.events.filter((e) => e.kind === 'enter').map((e) => e.at)).toEqual([0, 5000, 10_000, 15_000, 20_000]);
    const fast = run(newBelt<number>(0, 0.2), 20_000);
    expect(fast.events.filter((e) => e.kind === 'enter').map((e) => e.at)).toEqual([0, 2500, 5000, 7500, 10_000, 12_500, 15_000, 17_500, 20_000]);
  });

  it('a thing rides to the end in 1 / speed seconds and falls off, oldest first', () => {
    const { belt, events } = run(newBelt<number>(0, 0.1), 16_000);
    const off = events.filter((e) => e.kind === 'off');
    expect(off.map((e) => [e.item.id, e.at])).toEqual([['t0', 10_000], ['t1', 15_000]]);
    expect(belt.items.map((it) => it.id)).toEqual(['t2', 't3']);
    expect(belt.items.map((it) => Math.round(it.pos * 100) / 100)).toEqual([0.6, 0.1]);
    expect(belt.asOf).toBe(16_000);
  });

  it('where a thing is now needs no new message: position as of a time, plus speed x time', () => {
    const { belt } = run(newBelt<number>(0, 0.1), 1000);
    expect(belt.items[0].pos).toBeCloseTo(0.1);
    expect(posAt(belt, belt.items[0].pos, 4000)).toBeCloseTo(0.4);
    expect(posAt(belt, belt.items[0].pos, 60_000)).toBe(1);           // never drawn past the end
    expect(posAt(belt, belt.items[0].pos, 0)).toBeCloseTo(0.1);        // nor backwards
  });

  it('a speed change keeps everything where it is and moves it faster from then on', () => {
    let { belt } = run(newBelt<number>(0, 0.1), 2000);
    expect(belt.items[0].pos).toBeCloseTo(0.2);
    belt = atSpeed(belt, 0.4);
    expect(belt.items[0].pos).toBeCloseTo(0.2);
    expect(nextAt(belt)).toBeCloseTo(2000 + (0.3 / 0.4) * 1000);       // the next one enters when the belt has run 0.5 in all
    const later = run(belt, 4000);
    expect(later.events.map((e) => [e.kind, e.item.id, Math.round(e.at)])).toEqual([['enter', 't1', 2750], ['off', 't0', 4000], ['enter', 't2', 4000]]);
  });

  it('settling in one go or in many small steps gives the same belt', () => {
    const once = run(newBelt<number>(0, 0.13), 30_000, 0.36);
    let b = newBelt<number>(0, 0.13);
    const events: BeltEvent<number>[] = [];
    for (let now = 37; now <= 30_000; now += 37) { const r = run(b, now, 0.36); b = r.belt; events.push(...r.events); }
    b = run(b, 30_000, 0.36).belt;
    expect(events.map((e) => [e.kind, e.item.id])).toEqual(once.events.map((e) => [e.kind, e.item.id]));
    expect(b.items.map((it) => it.id)).toEqual(once.belt.items.map((it) => it.id));
    b.items.forEach((it, i) => expect(it.pos).toBeCloseTo(once.belt.items[i].pos, 6));
  });

  it('awkward speeds and times never make a thing enter twice or get stuck at the end (rounding)', () => {
    // speeds whose arithmetic does not come out even, changed mid-run, stepped at odd moments
    const start = 1_700_000_000_123;
    let b = newBelt<number>(start, 0.1);
    let entered = 0, left = 0, steps = 0;
    for (let now = start; now < start + 120_000; now += 61.7) {
      if (steps++ % 40 === 0) b = atSpeed(b, 0.1 * (1 + 0.14 * ((steps / 40) % 8)));
      for (let i = 0; i < 50; i++) {
        const r = next(b, now, 0.36, make);
        b = r.belt;
        if (!r.event) break;
        if (r.event.kind === 'enter') entered++; else left++;
        expect(i).toBeLessThan(5);                                   // never a burst of events at one moment
      }
      expect(b.items.length).toBeLessThanOrEqual(4);                 // a belt holds about 1 / gap things
      expect(b.items.every((it) => it.pos >= 0 && it.pos <= 1 + 1e-6)).toBe(true);
    }
    expect(entered).toBeGreaterThan(30);
    expect(entered - left).toBe(b.items.length);
    const pos = b.items.map((it) => it.pos).sort((x, y) => y - x);
    for (let i = 1; i < pos.length; i++) expect(pos[i - 1] - pos[i]).toBeCloseTo(0.36, 6);
  });

  it('a stopped belt does nothing; advancing never moves time backwards', () => {
    const still = newBelt<number>(0, 0);
    expect(nextAt(still)).toBe(Infinity);
    expect(run(still, 60_000).events).toEqual([]);
    const b = advance(run(newBelt<number>(0, 0.1), 3000).belt, 1000);
    expect(b.asOf).toBe(3000);
  });

  it('stations are stretches of the belt, evenly spread, with gaps between them', () => {
    const zs = evenZones(4, 0.16);
    expect(zs.map((z) => [Math.round(z.from * 100), Math.round(z.to * 100)])).toEqual([[12, 28], [32, 48], [52, 68], [72, 88]]);
    expect([0.05, 0.12, 0.2, 0.28, 0.3, 0.6, 0.9].map((p) => zoneOf(zs, p))).toEqual([-1, 0, 0, 0, -1, 2, -1]);
  });
});
