// Which automatic host step runs next. Steps come from state (hostAuto), so whoever keeps the
// clock (the room server, the simulator's virtual clock) only remembers which keys already fired.
import type { AutoStep } from './hostauto';

/** The step due soonest by `until` that hasn't fired yet (each key fires once while it stands). */
export function dueStep(steps: AutoStep[], fired: Set<string>, now: number, until: number): AutoStep | null {
  const live = new Set(steps.map((s) => s.key));
  for (const k of fired) if (!live.has(k)) fired.delete(k);
  const due = steps.filter((s) => !fired.has(s.key) && (s.at || now) <= until)
    .sort((a, b) => (a.at || now) - (b.at || now));
  return due[0] || null;
}
