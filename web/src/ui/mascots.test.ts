// SVG paints in source order (later = on top), so a hat must come AFTER the face it sits on.
// Santa's and the elf's hats used to be drawn first, so the face covered the brim.
import { beforeAll, describe, expect, it } from 'vitest';

let svg: Record<string, string>;
beforeAll(async () => {
  (globalThis as { location?: unknown }).location ??= { search: '' }; // mascots.ts reads ?mascots=0
  svg = (await import('./mascots')).svg;
});

const at = (s: string, fill: string) => {
  const i = s.indexOf(`fill="${fill}"`);
  expect(i, `${fill} is drawn`).toBeGreaterThan(-1);
  return i;
};

describe('mascots: hats sit on top of heads', () => {
  it("Santa: the face is drawn first, then the hat, its white brim and the bobble", () => {
    const face = at(svg.santa, '#f6c7a6');
    expect(at(svg.santa, '#d7263d')).toBeGreaterThan(face); // red hat
    expect(svg.santa.indexOf('<rect')).toBeGreaterThan(face); // white brim
  });
  it('the elf: the face is drawn first, then the hat and its band', () => {
    const face = at(svg.elf, '#f8d2b5');
    expect(at(svg.elf, '#1f9d55')).toBeGreaterThan(face); // green hat
    expect(at(svg.elf, '#d7263d')).toBeGreaterThan(face); // red band
  });
});

describe('mascots: what they say', () => {
  type Line = string | ((x: string | number) => string);
  let lines: Record<string, Line[]>;
  beforeAll(async () => { lines = (await import('./mascots')).lines; });
  // Each line with a long name / amount filled in, as the bubble would show it
  const render = (l: Line, moment: string) =>
    typeof l === 'function' ? l(/bid|tax/i.test(moment) ? 4 : 'Alexandra-Jo') : l;

  it('every moment has plenty of lines, so they rarely repeat', () => {
    expect(Object.keys(lines).length).toBeGreaterThanOrEqual(15);
    for (const [moment, pool] of Object.entries(lines)) expect(pool.length, moment).toBeGreaterThanOrEqual(6);
  });
  it('no line repeats within a moment', () => {
    for (const [moment, pool] of Object.entries(lines)) {
      const said = pool.map((l) => render(l, moment));
      expect(new Set(said).size, moment).toBe(said.length);
    }
  });
  it('lines fit a speech bubble and use no em dashes', () => {
    for (const [moment, pool] of Object.entries(lines)) {
      for (const l of pool) {
        const s = render(l, moment);
        expect(s.length, `${moment}: ${s}`).toBeLessThanOrEqual(90);
        expect(s, moment).not.toMatch(/\u2014|\s\u2013\s/);
        if (typeof l === 'function') expect(s, moment).toMatch(/Alexandra-Jo|4/);
      }
    }
  });
});
