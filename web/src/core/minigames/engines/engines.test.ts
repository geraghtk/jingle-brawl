// The throw engines, on their own: a swipe becomes a launch the same way on any phone; a launch
// lands in the same place every time; a landing scores the best peg it falls on.
import { describe, expect, it } from 'vitest';
import { AIM_GAIN, FLICK_SPEED, MAX_ANGLE, MAX_POWER, launchFrom, launchFromControls, parsePath, pathFor } from './gesture';
import type { SwipePath } from './gesture';
import { DEFAULT_FLIGHT, flightAt, landing } from './projectile';
import { miss, score } from './targets';
import type { Peg } from './targets';

describe('gestures', () => {
  it('upward is forward: the direction aims, the speed at release is the power', () => {
    const straight = launchFrom(pathFor({ angle: 0, power: 1 }))!;
    expect(straight.angle).toBeCloseTo(0);
    expect(straight.power).toBeCloseTo(1);
    const left = launchFrom(pathFor({ angle: -0.2, power: 0.8 }))!;
    expect(left.angle).toBeCloseTo(-0.2);
    expect(left.power).toBeCloseTo(0.8);
  });

  it("aim: the whole stroke, both directions in screen heights, turned down (a wobble isn't a wild throw)", () => {
    const at = (deg: number): SwipePath => { const r = (deg * Math.PI) / 180; return [[0.25, 0.9, 0], [0.25 + 0.3 * Math.sin(r), 0.9 - 0.3 * Math.cos(r), 100]]; };
    expect(launchFrom(at(10))!.angle).toBeCloseTo(AIM_GAIN * (10 * Math.PI) / 180);           // 10 degrees of flick: 4.5 of throw
    expect(launchFrom(at(-25))!.angle).toBeCloseTo(-AIM_GAIN * (25 * Math.PI) / 180);
    // a curl in the last few ms doesn't swing it: the stroke's direction aims
    const curl: SwipePath = [[0.25, 0.9, 0], [0.25, 0.75, 50], [0.25, 0.62, 90], [0.3, 0.6, 100]];
    expect(Math.abs(launchFrom(curl)!.angle)).toBeLessThan(0.1);                    // its last 10 ms alone would say 0.5
  });

  it('measured in screen heights and ms, so screen size and touch sample rate make no difference', () => {
    const swipe = pathFor({ angle: 0.1, power: 0.9 });
    // the same swipe sampled 4 times as often
    const fine: SwipePath = [];
    for (let k = 0; k <= 24; k++) {
      const f = k / 24, a = swipe[0], b = swipe[swipe.length - 1];
      fine.push([a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, Math.round(100 * f)]);
    }
    expect(launchFrom(fine)!.power).toBeCloseTo(launchFrom(swipe)!.power, 1);   // within 0.05 (ms rounding)
    expect(launchFrom(fine)!.angle).toBeCloseTo(launchFrom(swipe)!.angle, 2);
  });

  it('the release counts: a slow start and a fast flick is a fast throw', () => {
    const slowThenFast: SwipePath = [[0.5, 0.9, 0], [0.5, 0.88, 300], [0.5, 0.86, 600], [0.5, 0.86 - 0.4, 700]];
    expect(launchFrom(slowThenFast)!.power).toBeCloseTo(Math.min(MAX_POWER, 0.4 / 0.1 / FLICK_SPEED), 1);
  });

  it('limits: no superpowers from a 1 ms swipe, aim within bounds, downward or a crawl is not a throw', () => {
    expect(launchFrom([[0.5, 0.9, 0], [0.5, 0.1, 1]])!.power).toBe(MAX_POWER);
    expect(Math.abs(launchFrom([[0.5, 0.9, 0], [1.4, 0.8, 100]])!.angle)).toBe(MAX_ANGLE);
    expect(launchFrom([[0.5, 0.5, 0], [0.5, 0.8, 100]])).toBeNull();
    expect(launchFrom([[0.5, 0.9, 0], [0.5, 0.88, 2000]])).toBeNull();
  });

  it('a path from a phone is checked: shape, numbers, time order, size', () => {
    expect(parsePath([[0.5, 0.9, 0], [0.5, 0.5, 80]])).toHaveLength(2);
    for (const bad of [null, 'x', [[0.5, 0.9, 0]], [[0.5, 0.9, 50], [0.5, 0.5, 10]], [[0.5, NaN, 0], [0.5, 0.5, 10]], [[9, 0.9, 0], [0.5, 0.5, 10]], Array(80).fill([0.5, 0.5, 0])]) {
      expect(parsePath(bad)).toBeNull();
    }
  });

  it('the controls (aim, hold-to-charge) make a launch within the same limits', () => {
    expect(launchFromControls({ aim: 1, power: 9 })).toEqual({ angle: MAX_ANGLE, power: MAX_POWER });
    expect(launchFromControls({ aim: 0, power: 0.7 })).toEqual({ angle: 0, power: 0.7 });
    expect(launchFromControls({ aim: 'x', power: 1 })).toBeNull();
  });
});

describe('flight', () => {
  it('the same launch always lands in the same place; more power goes further; aim goes sideways', () => {
    const a = landing({ angle: 0, power: 1 });
    expect(landing({ angle: 0, power: 1 })).toEqual(a);
    expect(a.x).toBeCloseTo(0);
    expect(a.z).toBeCloseTo(DEFAULT_FLIGHT.baseRange + DEFAULT_FLIGHT.rangePerPower);
    expect(landing({ angle: 0, power: 0.5 }).z).toBeLessThan(a.z);
    expect(landing({ angle: 0.2, power: 1 }).x).toBeGreaterThan(0);
  });

  it('the drawn flight starts at the thrower, arcs up and ends on the landing', () => {
    const l = { angle: 0.1, power: 0.9 };
    const end = landing(l);
    expect(flightAt(l, 0)).toMatchObject({ x: 0, z: 0, h: 0, done: false });
    expect(flightAt(l, end.ms / 2).h).toBeGreaterThan(0.2);
    const last = flightAt(l, end.ms + 50);
    expect(last.done).toBe(true);
    expect(last.x).toBeCloseTo(end.x);
    expect(last.z).toBeCloseTo(end.z);
  });
});

describe('targets', () => {
  const pegs: Peg[] = [
    { id: 'a', x: 0, z: 1, rx: 0.1, rz: 0.2, points: 1 },
    { id: 'b', x: 0.05, z: 1, rx: 0.05, rz: 0.05, points: 5 },
  ];
  it('the best peg a landing falls on; the zone is an oval; just outside scores nothing', () => {
    expect(score(pegs, 0.05, 1)).toMatchObject({ pegId: 'b', points: 5 });
    expect(score(pegs, -0.05, 1)).toMatchObject({ pegId: 'a', points: 1 });
    expect(score(pegs, 0, 1.19)).toMatchObject({ pegId: 'a', points: 1 });     // deep but in the oval
    expect(score(pegs, 0.11, 0.8)).toMatchObject({ pegId: null, points: 0 });
  });
  it('a miss says which way it was off the nearest peg', () => {
    expect(miss(pegs, 0, 1.4)).toMatchObject({ nearest: 'a' });
    expect(miss(pegs, 0, 1.4).dz).toBeCloseTo(0.4);
  });
});
