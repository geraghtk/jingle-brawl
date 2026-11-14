# Game art: avatars and icons

The sticker avatars (`web/public/art/avatars/`) and icons (`web/public/art/icons/`) are made with
Flux schnell in a local ComfyUI, then cut out of their plain background.

```bash
# ComfyUI running on :8188 with models/checkpoints/flux1-schnell-fp8.safetensors
# (here: C:\Users\Kevin\Apps\ComfyUI, `.venv\Scripts\python main.py --port 8188`)
PY=/c/Users/Kevin/Apps/ComfyUI/.venv/Scripts/python.exe     # has numpy, scipy, Pillow
$PY tools/art/gen.py tools/art/prompts.json out --seeds 2 [--only av-fox,duel]
$PY tools/art/grid.py sheet.png out '^av-'                  # a labelled sheet to pick from
```

`prompts.json` holds the shared style (die-cut vinyl sticker, thick outline, flat bright colours)
and one subject per item: `av-<id>` for avatars, the rest are icons. Each item renders once per
seed into `out/raw/`, and `out/cut/` has it cut out, trimmed and squared. Pick the best take, then
save it as WebP: avatars at 256 px, icons at 160 px (`<name>.webp`).

- A new avatar also needs a line in `web/src/core/avatars.ts` (id, label, tile colour).
- A new icon needs its emoji mapped in `web/src/ui/icons.tsx`.
- Unit tests check that every avatar and icon has its picture.
- `naughty.webp` is the imp avatar, reused as the icon for a gift's naughty level.
- The Gremlin (Pass the Gremlin): `gremlin-calm` / `-uneasy` / `-unstable`, saved as
  `web/public/art/gremlin/{calm,uneasy,unstable}.webp` (256 px; one seed's take of each mood so they match).
  The prompt never says "gremlin": Flux draws the film's creatures for that word.

## Reindeer Ring Toss

`reindeer.py` renders the Ring Toss reindeer wide (1536 x 896), flat cartoon style with no sticker border,
and cuts it out without squaring (`--variant bg`: the winter backdrop). The chosen take (variant g, seed 4:
whimsical but battle-ready, a party hat whose tip is the 5-point peg, very wide antlers) had its left half
mirrored, so both sides score alike, and the grey ground shadow under its beard removed; it and
`backdrop.webp` (bg seed 2) are in `web/public/art/ringtoss/`. The snowdrift it stands in is drawn in
`ui/ringtoss/scene.ts`. A new reindeer means measuring its antler tips and hat tip and moving the pegs
(`ringtoss.ts` PEGS) and the placement (`DEER`, `TIP_H` and the view's `SX`/`SZ` in
`ui/ringtoss/scene.ts`) together. `reindeer_blender.py` builds one in Blender instead (exact tip
positions, but it looked less finished than the Flux art; kept for reference).

## Whack-an-Elf (Blender)

`whack.py` builds the Whack-an-Elf pieces from primitives in Blender (cel shading + inverted-hull
outline, to sit with the stickers) and renders them with one camera, so the layers line up:

```bash
/c/Users/Kevin/Apps/blender-5.2.3-windows-x64/blender.exe -b --python tools/art/whack.py -- out --size 384
```

`hole-back` + `hole-front` (the near lip, drawn over the pop-up), `elf` / `gold` / `grinch` and their
`-bonk` faces, `mallet`. Save them as WebP in `web/public/art/whack/` (384 px, mallet 256 px). The
script prints `CLIP_LINE_PCT`, where a pop-up disappears into the hole; `.wh-pit` in
`web/src/ui/party.css` clips at it (`inset(... 19% ...)` for 81).
