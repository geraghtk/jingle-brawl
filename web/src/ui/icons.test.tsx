import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { SYMBOLS } from '../core/minigames/gremlin/challenges';
import { Glyphs, ICONS, Icon } from './icons';

describe('icons', () => {
  it('text keeps its words; the emoji it knows become pictures whose alt is the emoji', () => {
    const html = renderToStaticMarkup(<Glyphs>{'⚔️ Duel: Ada vs Bo 🫥'}</Glyphs>);
    expect(html).toBe('<img class="yd-icon" src="art/icons/duel.webp" alt="⚔️" draggable="false"/> Duel: Ada vs Bo 🫥');
    expect(renderToStaticMarkup(<Glyphs>{'no icons here'}</Glyphs>)).toBe('no icons here');
    expect(renderToStaticMarkup(<Icon name="gambit" />)).toContain('alt="😈"');
  });
  it('every icon has its picture', () => {
    for (const name of [...new Set(Object.values(ICONS)), 'gambit']) {
      expect(existsSync(resolve(__dirname, `../../public/art/icons/${name}.webp`)), name).toBe(true);
    }
  });
  it("the Gremlin has a picture for each mood, and memory_flash's symbols are all stickers", () => {
    for (const mood of ['calm', 'uneasy', 'unstable']) expect(existsSync(resolve(__dirname, `../../public/art/gremlin/${mood}.webp`)), mood).toBe(true);
    for (const s of SYMBOLS) expect(renderToStaticMarkup(<Glyphs>{s}</Glyphs>), s).toContain('<img');
  });
});
