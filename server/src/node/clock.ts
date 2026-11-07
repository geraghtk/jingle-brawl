// The real clock for the Node server (tests use a fake one).
import type { Clock } from '../room/actor';

export const realClock: Clock = {
  now: () => Date.now(),
  at(when, fn) {
    const id = setTimeout(fn, Math.max(0, when - Date.now()));
    return () => clearTimeout(id);
  },
};
