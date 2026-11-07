// A clock for tests: time moves only when the test says, and timers fire in order.
import type { Clock } from '../room/actor';

export const T0 = 1_700_000_000_000;

export class FakeClock implements Clock {
  t = T0;
  private timers: { at: number; fn: () => void; id: number }[] = [];
  private next = 0;
  now() { return this.t; }
  at(when: number, fn: () => void) {
    const id = ++this.next;
    this.timers.push({ at: when, fn, id });
    return () => { this.timers = this.timers.filter((x) => x.id !== id); };
  }
  /** Let time pass, firing timers in order. */
  advance(ms: number) {
    const until = this.t + ms;
    for (;;) {
      const due = this.timers.filter((x) => x.at <= until).sort((a, b) => a.at - b.at)[0];
      if (!due) break;
      this.timers = this.timers.filter((x) => x !== due);
      this.t = Math.max(this.t, due.at);
      due.fn();
    }
    this.t = until;
  }
  /** Time passes but no timer gets to run (a busy or sleeping server). */
  jump(ms: number) { this.t += ms; }
}

