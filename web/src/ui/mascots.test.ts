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
