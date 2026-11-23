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
