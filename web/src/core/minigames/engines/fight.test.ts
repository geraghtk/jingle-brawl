// The fight engine: a punch lands a moment after it starts, and what it does depends on the other
// fighter at that moment. Mashing loses to blocking, blocking for ever loses to waiting.
import { describe, expect, it } from 'vitest';
import { canGuard, canPunch, DEFAULT_FIGHT as P, guard, newFight, nextAt, punch, settle } from './fight';
import type { Fight } from './fight';

const at = (x: Fight, now: number) => settle(x, now, P);
const kinds = (x: Fight) => x.events.map((e) => e.kind);

describe('the fight engine', () => {
  it('a punch does nothing until it lands, then takes one health and makes the other flinch', () => {
    let x = punch(newFight(P), 0, 'L', 1000, P);
    expect(nextAt(x, P)).toBe(1000 + P.windupMs);
    expect(at(x, 1000 + P.windupMs - 1).fight.f[1].health).toBe(P.health);
    x = at(x, 1000 + P.windupMs).fight;
    expect(x.f[1].health).toBe(P.health - 1);
    expect(x.events.at(-1)).toMatchObject({ kind: 'hit', by: 0, hand: 'L', damage: 1, at: 1000 + P.windupMs });
    expect(x.f[1]).toMatchObject({ busy: 'flinch', busyUntil: 1000 + P.windupMs + P.flinchMs });
    expect(canPunch(x, 1, 1000 + P.windupMs + P.flinchMs - 1)).toBe(false);
    expect(canPunch(x, 1, 1000 + P.windupMs + P.flinchMs)).toBe(true);
    expect(canPunch(x, 0, 1000 + P.windupMs + P.recoverMs - 1)).toBe(false);       // the puncher recovers too
    expect([x.thrown, x.landed]).toEqual([[1, 0], [1, 0]]);
  });

  it('a guard stops it, and the puncher is open for longer than a wind-up: the blocker can always hit back', () => {
    let x = guard(newFight(P), 1, true, 0);
    x = punch(x, 0, 'R', 500, P);                            // the guard has been up a while: a plain block
    x = at(x, 500 + P.windupMs).fight;
    expect(x.f[1].health).toBe(P.health);
    expect(x.events.at(-1)).toMatchObject({ kind: 'blocked', by: 0 });
    expect(x.f[0].busy).toBe('blocked');
    expect(P.blockedMs).toBeGreaterThan(P.windupMs);
    // the blocker drops the guard and swings at once: it lands before the puncher can do anything
    const t = 500 + P.windupMs;
    x = punch(guard(x, 1, false, t), 1, 'L', t, P);
    expect(canGuard(x, 0, t + P.windupMs - 1)).toBe(false);
    x = at(x, t + P.windupMs).fight;
    expect(x.f[0].health).toBe(P.health - 1);
  });

  it('a guard raised just before the punch lands is a parry: the puncher is stunned', () => {
    let x = punch(newFight(P), 0, 'L', 0, P);
    x = guard(x, 1, true, P.windupMs - P.parryMs);           // as late as still counts
    x = at(x, P.windupMs).fight;
    expect(x.events.at(-1)).toMatchObject({ kind: 'parried', by: 0 });
    expect(x.f[0]).toMatchObject({ busy: 'stunned', busyUntil: P.windupMs + P.stunMs });
    // one ms earlier is only a block
    let y = punch(newFight(P), 0, 'L', 0, P);
    y = at(guard(y, 1, true, P.windupMs - P.parryMs - 1), P.windupMs).fight;
    expect(y.events.at(-1)!.kind).toBe('blocked');
  });

  it('a punch into a wind-up is a counter: double damage and their punch never lands', () => {
    let x = punch(newFight(P), 0, 'L', 0, P);
    x = punch(x, 1, 'R', 100, P);                            // started later: still winding up when the first lands
    x = at(x, 5000).fight;
    expect(kinds(x)).toEqual(['counter']);
    expect([x.f[0].health, x.f[1].health]).toEqual([P.health, P.health - P.counter]);
    expect(x.f[1].punch).toBeUndefined();
  });

  it('two punches landing in the same instant are a trade: one each, neither a counter', () => {
    let x = punch(punch(newFight(P), 0, 'L', 0, P), 1, 'L', 0, P);
    x = at(x, P.windupMs).fight;
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
    // a punch thrown as it breaks lands on an open fighter
    x = at(punch(x, 1, 'R', P.guardMaxMs, P), P.guardMaxMs + P.windupMs).fight;
    expect(x.f[0].health).toBe(P.health - 1);
  });

  it('nothing can be started while punching, guarding or busy', () => {
    let x = punch(newFight(P), 0, 'L', 0, P);
    expect([canPunch(x, 0, 10), canGuard(x, 0, 10)]).toEqual([false, false]);
    x = guard(newFight(P), 0, true, 0);
    expect([canPunch(x, 0, 10), canGuard(x, 0, 10)]).toEqual([false, false]);
  });

  it('settling stops at a knockout and says who went down', () => {
    let x = newFight({ ...P, health: 1 });
    x = punch(x, 1, 'L', 0, P);
    const out = settle(x, 10_000, P);
    expect(out.ko).toBe(0);
    expect(out.fight.f[0].health).toBe(0);
    // settling again changes nothing (the room may be asked twice)
    expect(settle(out.fight, 20_000, P)).toEqual({ fight: out.fight, ko: 0 });
  });

  it('mashing loses to block-then-punch (the reason the timings are what they are)', () => {
    // fighter 0 punches whenever it can; fighter 1 guards while a punch is coming, then swings while 0 is stuck
    let x = newFight(P);
    for (let now = 0; now < 60_000 && x.f[0].health > 0 && x.f[1].health > 0; now += 20) {
      x = settle(x, now, P).fight;
      if (canPunch(x, 0, now)) x = punch(x, 0, 'L', now, P);
      const coming = !!x.f[0].punch;
      if (coming && canGuard(x, 1, now) && x.f[0].punch!.lands - now > P.parryMs + 40) x = guard(x, 1, true, now);   // a plain block, not a parry
      else if (!coming && x.f[1].guard) x = guard(x, 1, false, now);
      if (!coming && canPunch(x, 1, now) && x.f[0].busy === 'blocked' && x.f[0].busyUntil > now) x = punch(x, 1, 'R', now, P);
    }
    expect(x.f[0].health).toBe(0);
    expect(x.f[1].health).toBe(P.health);
  });
});
