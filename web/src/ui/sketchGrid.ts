// Sketch-Off on the big screen: how many columns and rows of square drawings fill an area best.
// Drawing only (no DOM): the biggest square tile that fits `n` of them into a width-over-height `aspect`.

export function sketchGrid(n: number, aspect = 2): { cols: number; rows: number } {
  let best = { cols: 1, rows: Math.max(1, n) }, size = 0;
  for (let cols = 1; cols <= Math.max(1, n); cols++) {
    const rows = Math.ceil(Math.max(1, n) / cols);
    const tile = Math.min(aspect / cols, 1 / rows);    // a tile's side, in units of the area's height
    if (tile > size + 1e-9) { size = tile; best = { cols, rows }; }
  }
  return best;
}

/** The wall's margins, in percent of its height: `pad` between the drawings and the edge of the wall (so a
 *  drawing's frame, its "done" ring and its tilt never touch the screen's frame or the prompt above), `gap`
 *  between two drawings (so two rings never touch), `cap` the caption under each. The stage hands them to
 *  the css (ui/sketch.css .sk-grid), which does the same sum as `sketchTile`. */
export const WALL = { pad: 3.5, gap: 4.5, cap: 5 };

/** The side of one drawing on a wall `w` by `h` (any unit), `n` of them, with or without captions. */
export function sketchTile(n: number, w: number, h: number, captions = true): { cols: number; rows: number; tile: number; width: number; height: number } {
  const u = h / 100, pad = WALL.pad * u, gap = WALL.gap * u, cap = captions ? WALL.cap * u : 0;
  const { cols, rows } = sketchGrid(n, w / h);
  const tile = Math.max(0, Math.min((w - 2 * pad - (cols - 1) * gap) / cols, (h - 2 * pad - (rows - 1) * gap) / rows - cap));
  return { cols, rows, tile, width: cols * tile + (cols - 1) * gap, height: rows * (tile + cap) + (rows - 1) * gap };
}
