import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { cueNotes, play } from './sound';
import type { Cue } from './sound';

const ALL: Cue[] = ['jump', 'right', 'wrong', 'pass', 'doom', 'tick', 'fanfare', 'ding', 'thud', 'go', 'horn', 'clank', 'bell', 'whoosh'];

describe('the sound effects', () => {
  it('every cue is a short phrase on real instruments, within hearing', () => {
    for (const cue of ALL) {
      const notes = cueNotes(cue);
      expect(notes.length, cue).toBeGreaterThan(0);
      for (const [voice, f, at, len, level] of notes) {
        expect(['chime', 'mallet', 'pluck', 'brass', 'drum', 'air', 'tin'], cue).toContain(voice);
        expect(f, cue).toBeGreaterThan(40); expect(f, cue).toBeLessThan(4000);
        expect(at, cue).toBeGreaterThanOrEqual(0); expect(at + len, cue).toBeLessThan(2.2);
        expect(level, cue).toBeGreaterThan(0); expect(level, cue).toBeLessThanOrEqual(1);
      }
    }
  });
  it('no cue is a bare square or sawtooth beep (the old 8-bit sound): those waves only ever go through a filter', () => {
    const src = readFileSync(new URL('./sound.ts', import.meta.url), 'utf8');
    expect(src).not.toMatch(/'square'/);
    // a saw is only ever connected to a low-pass filter (`lp`), never straight to the output
    for (const m of src.matchAll(/osc\(r, 'sawtooth',[^)]*\)/g)) expect(m[0]).toMatch(/, lp(, -?\d+)?\)$/);
  });
  it('a pitch lifts every note of a cue by the same step', () => {
    const base = cueNotes('ding'), up = cueNotes('ding', 1.5);
    up.forEach((n, i) => { expect(n[1]).toBeCloseTo(base[i][1] * 1.5); expect(n[2]).toBe(base[i][2]); });
  });
  it('playing with no audio (a test, a server) is silent, not an error', () => {
    expect(() => play('fanfare')).not.toThrow();
  });
});
