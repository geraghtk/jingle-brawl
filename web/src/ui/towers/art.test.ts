import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { TOWER_ART, TOWER_COLORS, artSrc, towerPiece } from './art';

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
  it('a tower has a picture for each level: built again it looks it', () => {
    for (const kind of ['snow', 'cannon', 'frost']) {
      expect(towerPiece(kind)).toBe(kind);
      expect(towerPiece(kind, 2)).toBe(`${kind}2`);
      expect(towerPiece(kind, 3)).toBe(`${kind}3`);
      expect(towerPiece(kind, 9)).toBe(`${kind}3`);
      expect(artSrc(towerPiece(kind, 2), TOWER_COLORS[1])).toBe(`art/towers/${kind}2-b.webp`);
    }
    expect(towerPiece('dragon', 2)).toBe('dragon');                      // no picture of its own: the plain one
  });
});
