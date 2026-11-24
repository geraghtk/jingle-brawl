// The TV's main-game stage (host/TvStage.tsx): the sums behind a picture that never scrolls.
// Everything is in "u": one hundredth of the picture's height (the picture is 16:9, so 177.78u wide).
// Drawing only, no DOM: how big a name may be set so it fits its box without breaking a word, and how
// many columns and rows fill an area best with n tiles.

/** A 16:9 picture: its width in u. */
export const PIC_W = 177.78;

/** The average width of a letter, in ems, of the display face (bold Space Grotesk). */
const EM = 0.63, EM_UPPER = 0.72;

export interface Fit { size: number; lines: number; /** it cannot fit even at the smallest size: the box may break the word */ tight: boolean }

/** The pieces a line may break between: at spaces, and after a hyphen (as a browser does). */
export function breakable(text: string): string[] {
  return text.trim().split(/\s+/).flatMap((w) => w.split(/(?<=-)(?=[^-])/)).filter(Boolean);
}

/** How many lines `text` takes in a box `width` u wide at `size` u (greedy, never inside a word); Infinity if a word is wider than the box. */
export function linesAt(text: string, width: number, size: number, upper = false): number {
  const em = (upper ? EM_UPPER : EM) * size;
  const perLine = width / em;
  let lines = 1, used = 0;
  const words = text.trim().split(/\s+/).filter(Boolean);
  for (const word of words) {
    const parts = breakable(word);
    parts.forEach((part, i) => {
      if (part.length > perLine) { lines = Infinity; return; }
      const gap = used > 0 && i === 0 ? 1 : 0;                  // a space before a new word, none inside a hyphenated one
      if (used + gap + part.length <= perLine) used += gap + part.length;
      else { lines += 1; used = part.length; }
    });
  }
  return lines;
}

/** The biggest size (u, in steps of a quarter) from `max` down to `min` at which `text` fits `width` on at most `maxLines` lines. */
export function fitText(text: string, o: { width: number; max: number; min: number; maxLines?: number; upper?: boolean }): Fit {
  const maxLines = o.maxLines ?? 1;
  for (let size = o.max; size >= o.min - 1e-9; size -= 0.25) {
    const lines = linesAt(text, o.width, size, o.upper);
    if (lines <= maxLines) return { size: Math.round(size * 100) / 100, lines, tight: false };
  }
  return { size: o.min, lines: maxLines, tight: true };
}

export interface Grid { cols: number; rows: number; /** a tile's width, u */ tile: number }

/** n tiles, each `ratio` times as tall as wide, as big as fit into w x h (u) with `gap` between them; never wider than `maxTile`. */
export function bestGrid(n: number, w: number, h: number, o: { ratio?: number; gap?: number; maxTile?: number } = {}): Grid {
  const ratio = o.ratio ?? 1, gap = o.gap ?? 0, count = Math.max(1, n);
  let best: Grid = { cols: count, rows: 1, tile: 0 };
  for (let cols = 1; cols <= count; cols++) {
    const rows = Math.ceil(count / cols);
    const tile = Math.min((w - gap * (cols - 1)) / cols, (h - gap * (rows - 1)) / rows / ratio);
    if (tile > best.tile + 1e-9) best = { cols, rows, tile };
  }
  const tile = Math.max(0, Math.min(best.tile, o.maxTile ?? Infinity));
  return { ...best, tile: Math.floor(tile * 100) / 100 };
}

/** The players' rail along the bottom: one row while the tiles stay readable, then two rows of smaller ones. */
export function railShape(n: number, width = PIC_W - 6): { rows: 1 | 2; cols: number; compact: boolean; tileW: number } {
  const gap = 1.2, count = Math.max(1, n);
  const one = (width - gap * (count - 1)) / count;
  if (one >= 17) return { rows: 1, cols: count, compact: false, tileW: Math.min(30, Math.floor(one * 100) / 100) };
  const cols = Math.ceil(count / 2);
  return { rows: 2, cols, compact: true, tileW: Math.floor(((width - gap * (cols - 1)) / cols) * 100) / 100 };
}
