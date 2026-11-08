// A production line across phones, for any team game: server-owned items move through a line of
// steps; the steps are grouped into stations, each worked by one or more people. An item has exactly
// one holder at a time (one person, or a station's in-tray); a step is done only by its holder, and
// done once (a resent tap finds the item already moved on). A station works one item at a time, so
// a bigger team doesn't get a faster line by having more hands. What the steps are, and what makes
// an item right, belong to the game.
import type { PlayerId } from '../../types';

export interface Line {
  steps: string[];                 // in order
  stations: number[][];            // stations[s] = the step indexes station s does (contiguous, in order)
  workers: PlayerId[][];           // workers[s] = who works station s (round robin per item)
}

export interface Item<D> {
  id: string;
  step: number;                    // the step it's at (steps.length: finished)
  holder: PlayerId | null;         // who is working it now (null: waiting in its station's tray)
  data: D;
  touched: { by: PlayerId; step: number }[];   // who did which step (for contribution)
}

export interface Floor<D> {
  items: Record<string, Item<D>>;
  rr: number[];                    // per station: the next worker in turn
}

export const WORKFLOW_VERSION = '1';

/** Lay a line of `steps` out over `stationCount` stations (as even as can be) and these people. */
export function layout(steps: string[], stationCount: number, people: PlayerId[]): Line {
  const s = Math.max(1, Math.min(stationCount, steps.length));
  const stations: number[][] = Array.from({ length: s }, () => []);
  steps.forEach((_, i) => stations[Math.min(s - 1, Math.floor((i * s) / steps.length))].push(i));
  const workers: PlayerId[][] = Array.from({ length: s }, () => []);
  people.forEach((p, i) => workers[i % s].push(p));
  // fewer people than stations: someone works more than one
  for (let k = 0; k < s; k++) if (!workers[k].length) workers[k].push(people[k % people.length]);
  return { steps, stations, workers };
}

export const stationOf = (line: Line, step: number) => line.stations.findIndex((st) => st.includes(step));

/** Is this station's table free (nobody at it is holding an item at one of its steps)? */
function busy<D>(line: Line, floor: Floor<D>, s: number): boolean {
  return Object.values(floor.items).some((it) => it.holder !== null && stationOf(line, it.step) === s);
}

/** Hand waiting items to free stations, oldest first (call after every change). Only to people who are
 *  `available` (connected) if the station has any; with nobody there, the item waits in the tray. */
export function dispatch<D>(line: Line, floor: Floor<D>, available: (pid: PlayerId) => boolean = () => true): Floor<D> {
  const items = { ...floor.items };
  const rr = [...floor.rr];
  for (let s = 0; s < line.stations.length; s++) {
    if (busy(line, { items, rr }, s)) continue;
    const waiting = Object.values(items).filter((it) => it.holder === null && it.step < line.steps.length && stationOf(line, it.step) === s)
      .sort((a, b) => Number(a.id.split('-').pop()) - Number(b.id.split('-').pop()));
    if (!waiting.length) continue;
    const crew = line.workers[s];
    let who: PlayerId | null = null;
    for (let k = 0; k < crew.length && !who; k++) {
      const p = crew[((rr[s] || 0) + k) % crew.length];
      if (available(p)) { who = p; rr[s] = (rr[s] || 0) + k + 1; }
    }
    if (!who) continue;
    items[waiting[0].id] = { ...waiting[0], holder: who };
  }
  return { items, rr };
}

/** A new item at the start of the line, waiting for the first station. */
export function add<D>(floor: Floor<D>, id: string, data: D): Floor<D> {
  return { ...floor, items: { ...floor.items, [id]: { id, step: 0, holder: null, data, touched: [] } } };
}

/** `by` finishes the step their item is at, with its new data: on to the next step (the same person if
 *  it's the same station) or the next station's tray; past the last step it is finished. */
export function complete<D>(line: Line, floor: Floor<D>, itemId: string, by: PlayerId, data: D, to?: number, available?: (pid: PlayerId) => boolean): { floor: Floor<D> } | { error: string } {
  const it = floor.items[itemId];
  if (!it) return { error: 'No such item' };
  if (it.holder !== by) return { error: "That one isn't on your table" };
  const next = to ?? it.step + 1;
  const sameStation = next < line.steps.length && stationOf(line, next) === stationOf(line, it.step);
  const moved: Item<D> = { ...it, data, step: next, holder: sameStation ? by : null, touched: [...it.touched, { by, step: it.step }] };
  return { floor: dispatch(line, { ...floor, items: { ...floor.items, [itemId]: moved } }, available) };
}

/** Someone dropped out: what they held goes back to its station's tray (to the next person there). */
export function release<D>(line: Line, floor: Floor<D>, gone: (pid: PlayerId) => boolean, available?: (pid: PlayerId) => boolean): Floor<D> {
  const items = { ...floor.items };
  for (const it of Object.values(items)) if (it.holder && gone(it.holder)) items[it.id] = { ...it, holder: null };
  return dispatch(line, { ...floor, items }, available);
}

/** An item leaves the line (shipped, or scrapped). */
export function remove<D>(floor: Floor<D>, itemId: string): Floor<D> {
  const items = { ...floor.items };
  delete items[itemId];
  return { ...floor, items };
}

/** What `pid` is holding now (at most one item: a station works one at a time). */
export const holding = <D>(floor: Floor<D>, pid: PlayerId) => Object.values(floor.items).find((it) => it.holder === pid) ?? null;
