import { describe, expect, it } from 'vitest';
import { secsLeft } from './clock';
import { finaleDue, VERDICT_FIRST_MS } from './clock';

describe('secsLeft', () => {
  it('rounds up, so the last second reads 1 until it is over', () => {
    expect(secsLeft(10_000, 0)).toBe(10);
    expect(secsLeft(10_000, 1)).toBe(10);
    expect(secsLeft(10_000, 9_001)).toBe(1);
    expect(secsLeft(10_000, 9_999)).toBe(1);
    expect(secsLeft(10_000, 10_000)).toBe(0);
  });
  it('never goes below zero', () => {
    expect(secsLeft(10_000, 25_000)).toBe(0);
  });
});

describe('the finale waits for the last verdict', () => {
  it('a game that a duel has just ended shows that duel\'s verdict first; one ended any other way goes straight to the finale', () => {
    const at = 1_000_000;
    expect(finaleDue({ phase: 'ended', lastDuel: { at } }, at + 1000)).toBe(false);
    expect(finaleDue({ phase: 'ended', lastDuel: { at } }, at + VERDICT_FIRST_MS)).toBe(true);
    expect(finaleDue({ phase: 'ended', lastDuel: null }, at)).toBe(true);
    expect(finaleDue({ phase: 'ended' }, at)).toBe(true);
    expect(finaleDue({ phase: 'ended', lastDuel: { at: { '.sv': 'timestamp' } } }, at)).toBe(true);   // (a time not yet stamped: do not wait for ever)
    expect(finaleDue({ phase: 'main', lastDuel: null }, at)).toBe(false);
  });
});

describe('the finale after a duel an in-app game announced', () => {
  it('there is no verdict to wait for: the ending of the game (with what the winner takes) had its time before the duel was settled', () => {
    const at = 1_000_000;
    expect(finaleDue({ phase: 'ended', lastDuel: { at, announced: true } }, at)).toBe(true);
    expect(finaleDue({ phase: 'ended', lastDuel: { at } }, at)).toBe(false);          // a duel settled at the table still gets its verdict first
    expect(finaleDue({ phase: 'endgame', lastDuel: { at, announced: true } }, at)).toBe(false);
  });
});
