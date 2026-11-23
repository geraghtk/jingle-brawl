import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { TOWER_ART, TOWER_COLORS, artSrc } from './art';

describe('Elf Tower Wars pictures', () => {
  it('every picture the game draws is in web/public/art/towers', () => {
    for (const name of TOWER_ART) expect(existsSync(resolve(__dirname, `../../../public/art/towers/${name}.webp`)), name).toBe(true);
  });
  it('a piece is drawn in its own team colour, and an unknown piece still gets a picture', () => {
    expect(artSrc('tank', TOWER_COLORS[0])).toBe('art/towers/tank-a.webp');
    expect(artSrc('tank', TOWER_COLORS[1])).toBe('art/towers/tank-b.webp');
    expect(artSrc('pine')).toBe('art/towers/pine.webp');
    expect(artSrc('dragon', TOWER_COLORS[1])).toBe('art/towers/soldier-b.webp');
  });
});
