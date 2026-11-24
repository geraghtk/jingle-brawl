// The mascots' queue: a line never shows during a turn it was not about, and a new moment does not wait
// behind a bubble about something that has passed.
import { describe, expect, it } from 'vitest';
import { CHATTER_GAP_MS, emptyStage, MIN_SHOW_MS, offer, SHOW_MS, STALE_MS, step, tvGap } from './mascotQueue';
import type { Line, Stage } from './mascotQueue';

const line = (text: string, turn: number, at: number, chatter = false): Line => ({ who: 'elf', text, turn, at, chatter });
/** Offer the lines, then look at `now` in `turn`: what is up. */
const run = (s: Stage, turn: number, now: number) => step(s, turn, now);

describe('the mascots: what shows, given what was said about which turn', () => {
  it('a moment shows at once and stays its time', () => {
    let s = offer(emptyStage(), line('A tie!', 1, 1000));
    let r = run(s, 1, 1000);
    expect(r.show?.text).toBe('A tie!');
    expect(r.again).toBe(SHOW_MS);
    s = r.stage;
    r = run(s, 1, 1000 + SHOW_MS - 1);
    expect(r.hide).toBe(false);
    r = run(s, 1, 1000 + SHOW_MS);
    expect(r.hide).toBe(true);
    expect(r.stage.showing).toBeNull();
  });

  it('a line still waiting when its turn has passed is dropped: it never shows during the next turn', () => {
    let s = offer(emptyStage(), line('first', 1, 0));
    s = run(s, 1, 0).stage;                                   // "first" is up
    s = offer(s, line('about turn 1', 1, 500));               // waits behind it
    const r = run(s, 2, 2500);                                // the game moved on to turn 2
    expect(r.hide).toBe(true);                                // the bubble about turn 1 comes down
    expect(r.show).toBeNull();                                // and the waiting line about turn 1 is gone
    expect(r.stage.queue).toEqual([]);
  });

  it('a new moment replaces a bubble about a turn that has passed (after a moment to read it), instead of waiting its full time', () => {
    let s = run(offer(emptyStage(), line('old turn', 1, 0)), 1, 0).stage;
    s = offer(s, line('new turn', 2, 800));
    let r = run(s, 2, 800);
    expect(r.hide).toBe(false);                               // up for under MIN_SHOW_MS: not yanked away unread
    expect(r.again).toBe(MIN_SHOW_MS - 800);
    r = run(r.stage, 2, MIN_SHOW_MS);
    expect(r.hide).toBe(true);
    expect(r.show?.text).toBe('new turn');                    // well before SHOW_MS
    expect(MIN_SHOW_MS).toBeLessThan(SHOW_MS);
  });

  it('a bubble whose turn is over comes down even with nothing new to say', () => {
    const s = run(offer(emptyStage(), line('about the bidding', 1, 0)), 1, 0).stage;
    expect(run(s, 2, MIN_SHOW_MS - 1).hide).toBe(false);
    expect(run(s, 2, MIN_SHOW_MS).hide).toBe(true);
  });

  it('only the newest moment waits: the one before it is not kept to show late', () => {
    let s = run(offer(emptyStage(), line('up', 1, 0)), 1, 0).stage;
    s = offer(s, line('second', 1, 100));
    s = offer(s, line('third', 1, 200));
    expect(s.queue.map((l) => l.text)).toEqual(['third']);
    const r = run(s, 1, MIN_SHOW_MS);
    expect(r.show?.text).toBe('third');
    expect(r.stage.queue).toEqual([]);
  });

  it('a line that has waited too long is old news', () => {
    const s = { ...emptyStage(), queue: [line('late', 1, 0)] };
    expect(run(s, 1, STALE_MS).show?.text).toBe('late');
    expect(run(s, 1, STALE_MS + 1).show).toBeNull();
  });

  it('chatter never waits and never follows straight after a bubble', () => {
    const up = run(offer(emptyStage(), line('moment', 1, 0)), 1, 0).stage;
    expect(offer(up, line('psst', 1, 100, true)).queue).toEqual([]);            // someone is talking: dropped
    const down = run(up, 1, SHOW_MS).stage;                                       // hidden at SHOW_MS
    const soon = run(offer(down, line('psst', 1, SHOW_MS + 100, true)), 1, SHOW_MS + 100);
    expect(soon.show).toBeNull();
    expect(soon.stage.queue).toEqual([]);                                         // dropped, not held back
    const later = SHOW_MS + CHATTER_GAP_MS;
    expect(run(offer(down, line('psst', 1, later, true)), 1, later).show?.text).toBe('psst');
    // a moment pushes waiting chatter out
    const q = offer(offer(emptyStage(), line('psst', 1, 0, true)), line('moment', 1, 0));
    expect(q.queue.map((l) => l.text)).toEqual(['moment']);
  });
});

describe('the mascots on the TV: only in the free stretch of the top bar', () => {
  const bar = { left: 30, right: 1890, top: 24, height: 76 };
  it('between the name and the phase badge, a little taller than the bar', () => {
    const g = tvGap({ left: 30, right: 330, top: 24, height: 76 }, { left: 810, right: 1110, top: 30, height: 60 }, bar)!;
    expect(g.left).toBeGreaterThan(330);
    expect(g.left + g.width).toBeLessThan(810);
    expect(g.top + g.height / 2).toBe(bar.top + bar.height / 2);
  });
  it('no room (a long badge, a narrow screen): nobody speaks rather than cover something', () => {
    expect(tvGap({ left: 30, right: 330, top: 24, height: 76 }, { left: 500, right: 1400, top: 30, height: 60 }, bar)).toBeNull();
    expect(tvGap(null, null, bar)).toBeNull();
  });
});
