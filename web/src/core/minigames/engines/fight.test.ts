// The fight engine: a punch lands a moment after it starts, and what it does depends on the other
// fighter at that moment. Two punches, two defences, each good against something.
import { describe, expect, it } from 'vitest';
import { canDodge, canGuard, canPunch, DEFAULT_FIGHT as P, dodge, guard, newFight, nextAt, punch, settle, windup } from './fight';
import type { Fight, FightParams } from './fight';

const at = (x: Fight, now: number) => settle(x, now, P);
const kinds = (x: Fight) => x.events.map((e) => e.kind);

describe('the fight engine', () => {
  it('a jab does nothing until it lands, then takes one health and makes the other flinch', () => {
    let x = punch(newFight(P), 0, 'jab', 1000, P);
    expect(nextAt(x, P)).toBe(1000 + P.jabMs);
    expect(at(x, 1000 + P.jabMs - 1).fight.f[1].health).toBe(P.health);
    x = at(x, 1000 + P.jabMs).fight;
    expect(x.f[1].health).toBe(P.health - P.jabDamage);
    expect(x.events.at(-1)).toMatchObject({ kind: 'hit', by: 0, punch: 'jab', damage: 1, at: 1000 + P.jabMs });
    expect(x.f[1]).toMatchObject({ busy: 'flinch', busyUntil: 1000 + P.jabMs + P.flinchMs });
    expect(canPunch(x, 1, 1000 + P.jabMs + P.flinchMs - 1)).toBe(false);
    expect(canPunch(x, 1, 1000 + P.jabMs + P.flinchMs)).toBe(true);
    expect(canPunch(x, 0, 1000 + P.jabMs + P.recoverMs - 1)).toBe(false);          // the puncher recovers too
    expect([x.thrown, x.landed]).toEqual([[1, 0], [1, 0]]);
  });

  it('a heavy takes much longer to land and hits harder', () => {
    expect(windup('heavy', P)).toBeGreaterThan(windup('jab', P) * 1.8);
    const x = at(punch(newFight(P), 0, 'heavy', 0, P), P.heavyMs).fight;
    expect(x.f[1].health).toBe(P.health - P.heavyDamage);
    expect(P.heavyDamage).toBeGreaterThan(P.jabDamage);
    expect(at(punch(newFight(P), 0, 'heavy', 0, P), P.heavyMs - 1).fight.f[1].health).toBe(P.health);
  });

  it('a guard stops a jab, and the jabber is open for longer than a jab: the blocker can always hit back', () => {
    let x = guard(newFight(P), 1, true, 0);
    x = punch(x, 0, 'jab', 500, P);
    x = at(x, 500 + P.jabMs).fight;
    expect(x.f[1].health).toBe(P.health);
    expect(x.events.at(-1)).toMatchObject({ kind: 'blocked', by: 0, punch: 'jab' });
    expect(x.f[0].busy).toBe('blocked');
    expect(P.blockedMs).toBeGreaterThan(P.jabMs);
    const t = 500 + P.jabMs;
    x = punch(guard(x, 1, false, t), 1, 'jab', t, P);                // drops the guard and jabs at once
    expect(canGuard(x, 0, t + P.jabMs - 1)).toBe(false);
    expect(canDodge(x, 0, t + P.jabMs - 1)).toBe(false);
    x = at(x, t + P.jabMs).fight;
    expect(x.f[0].health).toBe(P.health - 1);
  });

  it('a heavy smashes through a guard: the guard breaks, some of it lands, and the blocker is open', () => {
    let x = guard(newFight(P), 1, true, 0);
    x = at(punch(x, 0, 'heavy', 100, P), 100 + P.heavyMs).fight;
    expect(x.events.at(-1)).toMatchObject({ kind: 'smash', by: 0, punch: 'heavy', damage: P.smashDamage });
    expect(x.f[1].health).toBe(P.health - P.smashDamage);
    expect(x.f[1].guard).toBeUndefined();
    expect(x.f[1]).toMatchObject({ busy: 'guard_break', busyUntil: 100 + P.heavyMs + P.guardBreakMs });
    // and the smasher is free again in time to land a jab before the blocker can do anything
    expect(P.recoverMs + P.jabMs).toBeLessThan(P.guardBreakMs);
  });

  it('a dodge makes any punch miss and leaves the puncher badly open', () => {
    for (const kind of ['jab', 'heavy'] as const) {
      let x = punch(newFight(P), 0, kind, 0, P);
      const lands = windup(kind, P);
      x = dodge(x, 1, lands - 100, P);                               // timed to the punch
      x = at(x, lands).fight;
      expect(x.events.at(-1), kind).toMatchObject({ kind: 'dodged', by: 0, punch: kind });
      expect(x.f[1].health, kind).toBe(P.health);
      expect(x.f[0], kind).toMatchObject({ busy: 'whiff', busyUntil: lands + P.whiffMs });
    }
    // the dodger is free again while the puncher is still stumbling, with time for a heavy
    expect(P.whiffMs - (P.dodgeMs + P.dodgeRestMs)).toBeGreaterThanOrEqual(P.jabMs);            // even a last-moment dodge earns a jab
    expect(P.whiffMs - P.dodgeRestMs - 60).toBeGreaterThanOrEqual(P.heavyMs);                       // a well-timed one earns a heavy
  });

  it('a dodge only lasts a moment, and its fighter is exposed afterwards: it cannot be spammed', () => {
    let x = dodge(newFight(P), 1, 0, P);
    expect(x.f[1]).toMatchObject({ busy: 'dodge', busyUntil: P.dodgeMs + P.dodgeRestMs });
    expect([canDodge(x, 1, P.dodgeMs + 10), canGuard(x, 1, P.dodgeMs + 10), canPunch(x, 1, P.dodgeMs + 10)]).toEqual([false, false, false]);
    // a jab that lands just after the dodge is over hits
    x = at(punch(x, 0, 'jab', P.dodgeMs - P.jabMs + 20, P), 5000).fight;
    expect(kinds(x)).toEqual(['hit']);
    expect(x.f[1].health).toBe(P.health - 1);
    // dodged too early against a heavy: it lands
    let y = punch(newFight(P), 0, 'heavy', 0, P);
    y = at(dodge(y, 1, 50, P), 5000).fight;
    expect(kinds(y)).toEqual(['hit']);
  });

  it('a punch into a wind-up is a counter: extra damage and their punch never lands (a jab beats a heavy)', () => {
    let x = punch(newFight(P), 0, 'heavy', 0, P);
    x = punch(x, 1, 'jab', 100, P);                                  // the jab lands long before the heavy would
    x = at(x, 5000).fight;
    expect(kinds(x)).toEqual(['counter']);
    expect(x.events[0]).toMatchObject({ by: 1, punch: 'jab', damage: P.jabDamage + P.counter });
    expect([x.f[0].health, x.f[1].health]).toEqual([P.health - P.jabDamage - P.counter, P.health]);
    expect(x.f[0].punch).toBeUndefined();
  });

  it('two punches landing in the same instant are a trade: one each, neither a counter', () => {
    let x = punch(punch(newFight(P), 0, 'jab', 0, P), 1, 'jab', 0, P);
    x = at(x, P.jabMs).fight;
    expect(kinds(x)).toEqual(['hit', 'hit']);
    expect([x.f[0].health, x.f[1].health]).toEqual([P.health - 1, P.health - 1]);
  });

  it('a guard held too long breaks and leaves its fighter open', () => {
    let x = guard(newFight(P), 0, true, 0);
    expect(nextAt(x, P)).toBe(P.guardMaxMs);
    x = at(x, P.guardMaxMs).fight;
    expect(x.f[0].guard).toBeUndefined();
    expect(x.f[0]).toMatchObject({ busy: 'guard_break', busyUntil: P.guardMaxMs + P.guardBreakMs });
    expect(canGuard(x, 0, P.guardMaxMs + 1)).toBe(false);
    x = at(punch(x, 1, 'jab', P.guardMaxMs, P), P.guardMaxMs + P.jabMs).fight;
    expect(x.f[0].health).toBe(P.health - 1);
  });

  it('nothing can be started while punching, guarding or busy', () => {
    let x = punch(newFight(P), 0, 'jab', 0, P);
    expect([canPunch(x, 0, 10), canGuard(x, 0, 10), canDodge(x, 0, 10)]).toEqual([false, false, false]);
    x = guard(newFight(P), 0, true, 0);
    expect([canPunch(x, 0, 10), canGuard(x, 0, 10), canDodge(x, 0, 10)]).toEqual([false, false, false]);
  });

  it('settling stops at a knockout and says who went down; settling again changes nothing', () => {
    const one: FightParams = { ...P, health: 1 };
    const x = punch(newFight(one), 1, 'jab', 0, one);
    const out = settle(x, 10_000, one);
    expect(out.ko).toBe(0);
    expect(out.fight.f[0].health).toBe(0);
    expect(settle(out.fight, 20_000, one)).toEqual({ fight: out.fight, ko: 0 });
  });

  // --- each habit has an answer (why the timings are what they are) ---
  type Brain = (x: Fight, me: 0 | 1, now: number) => Fight;
  /** Two habits fight until one is down (or a minute passes); returns the health each has left. */
  function bout(a: Brain, b: Brain): [number, number] {
    let x = newFight(P);
    for (let now = 0; now < 60_000 && x.f[0].health > 0 && x.f[1].health > 0; now += 20) {
      x = settle(x, now, P).fight;
      x = a(x, 0, now);
      x = b(x, 1, now);
    }
    return [x.f[0].health, x.f[1].health];
  }
  const them = (me: 0 | 1): 0 | 1 => (me === 0 ? 1 : 0);
  const jabMasher: Brain = (x, me, now) => (canPunch(x, me, now) ? punch(x, me, 'jab', now, P) : x);
  const heavyMasher: Brain = (x, me, now) => (canPunch(x, me, now) ? punch(x, me, 'heavy', now, P) : x);
  const turtle: Brain = (x, me, now) => (canGuard(x, me, now) ? guard(x, me, true, now) : x);
  /** Blocks a jab it sees coming, then jabs while the jabber is stuck. */
  const blocker: Brain = (x, me, now) => {
    const o = x.f[them(me)];
    if (o.punch && canGuard(x, me, now)) return guard(x, me, true, now);
    if (!o.punch && x.f[me].guard) x = guard(x, me, false, now);
    return !o.punch && o.busy === 'blocked' && o.busyUntil > now && canPunch(x, me, now) ? punch(x, me, 'jab', now, P) : x;
  };
  /** Jabs whenever it can: the answer to a slow wind-up. */
  const jabber = jabMasher;
  /** Dodges a punch just before it lands, then hits the stumbling puncher with the biggest punch there is time for. */
  const dodger: Brain = (x, me, now) => {
    const o = x.f[them(me)];
    if (o.punch && o.punch.lands - now <= P.dodgeMs - 60 && canDodge(x, me, now)) return dodge(x, me, now, P);
    if (o.punch || o.busy !== 'whiff' || !canPunch(x, me, now)) return x;
    const open = o.busyUntil - now;                                  // how long the puncher is still stumbling
    return open >= P.heavyMs ? punch(x, me, 'heavy', now, P) : open >= P.jabMs ? punch(x, me, 'jab', now, P) : x;
  };
  /** Dodges over and over, whatever is happening. */
  const dodgeSpammer: Brain = (x, me, now) => (canDodge(x, me, now) ? dodge(x, me, now, P) : x);
  /** Waits for a dodge to run out, then jabs into the gap after it. */
  const patient: Brain = (x, me, now) => {
    const o = x.f[them(me)];
    return o.dodge && canPunch(x, me, now) && now + P.jabMs >= o.dodge.until && now + P.jabMs < o.busyUntil ? punch(x, me, 'jab', now, P) : x;
  };

  it('mashing jabs loses to block-then-jab', () => {
    const [masher, answer] = bout(jabMasher, blocker);
    expect(masher).toBe(0);
    expect(answer).toBe(P.health);
  });
  it('mashing heavies loses to jabs (every heavy is caught in its wind-up)', () => {
    const [masher, answer] = bout(heavyMasher, jabber);
    expect(masher).toBe(0);
    expect(answer).toBe(P.health);
  });
  it('holding a block for ever loses to heavies', () => {
    const [blocking, answer] = bout(turtle, heavyMasher);
    expect(blocking).toBe(0);
    expect(answer).toBe(P.health);
  });
  it('a well-timed dodge beats heavies and jabs alike', () => {
    for (const masher of [heavyMasher, jabMasher]) {
      const [m, d] = bout(masher, dodger);
      expect(m).toBe(0);
      expect(d).toBe(P.health);
    }
  });
  it('dodging over and over loses to a jab timed for when it runs out', () => {
    const [spam, answer] = bout(dodgeSpammer, patient);
    expect(spam).toBe(0);
    expect(answer).toBe(P.health);
  });
});
