import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { WHACK_ART, WhackHole } from './whack';

describe('Whack-an-Elf art', () => {
  it('every picture is there', () => {
    for (const name of WHACK_ART) expect(existsSync(resolve(__dirname, `../../public/art/whack/${name}.webp`)), name).toBe(true);
  });
  it('a hit shows the bonked face and the mallet; an empty hole shows no pop-up', () => {
    const hit = renderToStaticMarkup(<WhackHole kind="grinch" state="hit" />);
    expect(hit).toContain('art/whack/grinch-bonk.webp');
    expect(hit).toContain('art/whack/mallet.webp');
    const up = renderToStaticMarkup(<WhackHole kind="gold" state="up" />);
    expect(up).toContain('src="art/whack/gold.webp" alt="golden elf"');
    expect(up).not.toContain('mallet');
    expect(renderToStaticMarkup(<WhackHole state="down" />)).not.toContain('wh-pop');
  });
});
