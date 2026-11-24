"""Elf Tower Wars' pictures (web/public/art/towers/): Flux schnell in the local ComfyUI, in the game's flat
2D look (as whack2d.py), cut out of a plain background.

    python towers.py out_dir [--seeds 5] [--only soldier,tank]      takes of each piece + out/sheet.png
    python towers.py out_dir --place soldier=2,tank=0,...           the chosen takes into web/public/art/towers

Everything a team owns is drawn in RED and saved twice: `<name>-a.webp` as drawn (pushed to team A's
pink-red) and `<name>-b.webp` with its reds turned to team B's green (`recolor`), so the two armies are
the same toys in two colours. The field (`field`) and the scenery (pine, snowman, gifts) are saved once.
"""
import colorsys, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from gen import cut, render, workflow, call  # noqa: F401
import gen

STYLE = ("A flat vector cartoon illustration of {subject}. Clean bold shapes, thick dark outline, smooth soft cel "
         "shading, bright friendly colors, simple and polished like a modern mobile game asset. A single object, "
         "centered, whole and fully inside the picture. No sticker border, no white rim, no frame. Isolated on a "
         "plain solid flat medium grey background, no ground, no shadow, no text, no letters.")
PIECES = {
    "soldier": "a cute chunky wooden toy soldier seen exactly from the side, marching towards the right, one leg "
               "stepping forward, big round head with rosy cheek, a tall black fur hat with a gold badge, a bright "
               "red coat with gold buttons, white trousers, black boots, carrying a little wooden popgun over its "
               "shoulder",
    "tank": "a cute brown teddy bear riding in a chunky little toy tank seen exactly from the side, driving towards "
            "the right, the tank made of bright red painted wood with big round grey wheels and a short stubby "
            "barrel pointing right, the smiling bear's head and paws poking out of the top wearing a small red helmet",
    "mouse": "a cute chubby grey wind-up toy mouse seen exactly from the side, scurrying towards the right, big round "
             "pink ears, a tiny black nose, a long curly tail, a big bright red wind-up key sticking out of its back, "
             "a little bright red scarf round its neck",
    "snow": "a small round toy castle turret tower in three-quarter view, grey stone blocks, a bright red pointed "
            "cone roof with a little flag, a wooden catapult arm on top loaded with a big white snowball, a neat "
            "pile of white snowballs at its foot",
    "cannon": "a chunky toy cannon in three-quarter view, a big fat barrel painted in red and white candy cane "
              "stripes pointing up and to the right, on a bright red wooden carriage with two round gold wheels, a "
              "few round peppermint candies stacked beside it",
    "frost": "a magical frost lamp in three-quarter view: a short ornate silver lamp post on a round stone base, "
             "holding a big glowing icy light-blue crystal lantern, a bright red ribbon bow tied round the post, "
             "snowflakes and sparkles of frost drifting around it",
    # a tower built again on its spot (level 2, level 3): the same tower, visibly more of it
    "snow2": "a tall sturdy toy castle turret tower in three-quarter view, two storeys of grey stone blocks with a "
             "crenellated stone balcony halfway up, a bright red pointed cone roof with a little flag, two wooden "
             "catapult arms on top each loaded with a big white snowball, a bright red banner hanging on its wall, "
             "a neat pile of white snowballs at its foot",
    "snow3": "a grand tall toy castle keep with three round turrets in three-quarter view, grey stone blocks with "
             "gold trim, bright red pointed cone roofs each with a gold star on top and little flags, three wooden "
             "catapult arms loaded with big white snowballs, bright red banners with gold edging, glowing warm "
             "yellow windows, a big neat pile of white snowballs at its foot",
    "cannon2": "a chunky toy cannon with two barrels side by side in three-quarter view, both big fat barrels painted "
               "in red and white candy cane stripes pointing up and to the right, on a sturdy bright red wooden "
               "carriage with two round gold wheels and a steel shield plate at the front, a neat pyramid of round "
               "peppermint candies stacked beside it",
    "cannon3": "a mighty toy cannon battery in three-quarter view, three big fat barrels painted in red and white "
               "candy cane stripes with gold rings pointing up and to the right, on a grand bright red wooden "
               "carriage with gold trim, a gold star crest and four round gold wheels, a little red flag on top, a "
               "big pyramid of round peppermint candies stacked beside it",
    "frost2": "a magical frost lamp in three-quarter view: a tall ornate silver lamp post on a round stone base, "
              "with two curling arms holding two big glowing icy light-blue crystal lanterns, a bright red ribbon "
              "bow tied round the post, icicles hanging from the arms, snowflakes and sparkles of frost drifting "
              "around it",
    "frost3": "a grand magical frost beacon in three-quarter view: a tall ornate silver and gold lamp post on a wide "
              "round stone base, with three curling arms holding three big glowing icy light-blue crystal lanterns "
              "and a large glowing ice crystal star on top, bright red ribbon bows, long icicles, a ring of frost "
              "and snowflakes swirling around it",
    "workshop": "a cozy little elf toy workshop cottage in three-quarter view, warm cream walls with wooden beams, a "
                "steep bright red roof heaped with thick white snow, a brick chimney, a round arched wooden door "
                "with a wreath, two glowing warm yellow windows, a bright red striped awning, strings of little "
                "lights, toys and wrapped presents stacked by the door",
    "pine": "a plump little cartoon pine tree, dark green layered branches each tipped with thick white snow, a short "
            "brown trunk",
    "snowman": "a small friendly snowman with a carrot nose, coal eyes, stick arms, a striped purple scarf and a "
               "little black top hat",
    "gifts": "a small neat stack of three wrapped Christmas presents in yellow, blue and purple wrapping with ribbon "
             "bows",
    "cane": "a short candy cane signpost stuck upright in a small mound of snow, red and white stripes, with a tiny "
            "golden bell hanging from its hook",
}
TEAM = ["soldier", "tank", "mouse", "snow", "cannon", "frost", "snow2", "snow3", "cannon2", "cannon3", "frost2", "frost3", "workshop"]
FIELD = ("A flat vector cartoon illustration, a top-down bird's-eye view of a smooth empty snowy field, soft white "
         "and very pale blue snow with gentle rounded drifts, a few faint sparkles and tiny footprints, soft "
         "shadows. Uniform and calm, nothing else on it: no trees, no houses, no objects, no characters, no text. "
         "Like the ground of a modern mobile strategy game map.")
SIZES = {"soldier": 256, "tank": 288, "mouse": 192, "snow": 288, "cannon": 288, "frost": 288, "workshop": 512, "pine": 224,
         "snowman": 224, "gifts": 192, "cane": 192,
         "snow2": 320, "snow3": 352, "cannon2": 320, "cannon3": 352, "frost2": 320, "frost3": 352}
# how close to the background a pixel must be to be cut away: white snow against the near-white backdrop
# needs a tight one, or the roof gets holes where its snow was (and the tower's loaded snowball is hollow)
TOL = {"workshop": 4, "snow": 4, "snow2": 4, "snow3": 4}
# drawn facing left however it is asked: turned round when placed (every toy walks to the right)
MIRROR = {"mouse"}
A_HUE, B_HUE = 345 / 360, 163 / 360          # team A #ff3366, team B #18b88a


def recolor(im, hue, sat=1.0, val=1.0):
    """The picture with its reds (the team's paint) turned to `hue`; everything else as it was."""
    a = np.asarray(im.convert("RGBA")).astype(np.float32) / 255
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx, mn = a[..., :3].max(-1), a[..., :3].min(-1)
    d = mx - mn
    s = np.where(mx > 0, d / np.maximum(mx, 1e-6), 0)
    h = np.zeros_like(mx)
    m = d > 1e-6
    rc = m & (mx == r); gc = m & (mx == g) & ~rc; bc = m & ~rc & ~gc
    h[rc] = ((g - b)[rc] / d[rc]) % 6
    h[gc] = (b - r)[gc] / d[gc] + 2
    h[bc] = (r - g)[bc] / d[bc] + 4
    h = h / 6
    red = ((h > 0.93) | (h < 0.035)) & (s > 0.55) & (mx > 0.25)       # paint red: not skin, wood or brick
    out = a.copy()
    vec = np.vectorize(colorsys.hsv_to_rgb)
    rr, gg, bb = vec(np.full(red.sum(), hue), np.clip(s[red] * sat, 0, 1), np.clip(mx[red] * val, 0, 1))
    out[red, 0], out[red, 1], out[red, 2] = rr, gg, bb
    return Image.fromarray((out * 255).round().astype(np.uint8), "RGBA")


# a piece that is grey all over (the wind-up mice) gets its team's colour in its coat too: its key and bow alone
# were too small to tell the two packs apart on the TV
COAT = {"mouse"}


def coat(im, hue, sat=0.42):
    """The picture with its plain greys (a grey coat: not the ink lines, not white) tinted to `hue`. Tinted
    pixels are no longer grey, so doing it twice changes nothing."""
    a = np.asarray(im.convert("RGBA")).astype(np.float32) / 255
    mx, mn = a[..., :3].max(-1), a[..., :3].min(-1)
    s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    grey = (s < 0.13) & (mx > 0.32) & (mx < 0.9) & (a[..., 3] > 0.3)
    out = a.copy()
    vec = np.vectorize(colorsys.hsv_to_rgb)
    rr, gg, bb = vec(np.full(grey.sum(), hue), np.full(grey.sum(), sat), np.clip(mx[grey] * 1.22, 0, 1))
    out[grey, 0], out[grey, 1], out[grey, 2] = rr, gg, bb
    return Image.fromarray((out * 255).round().astype(np.uint8), "RGBA")


def recoat():
    """Give the placed COAT pieces their team coats (no ComfyUI needed): python tools/art/towers.py --coat"""
    dest = Path(__file__).resolve().parents[2] / "web" / "public" / "art" / "towers"
    for name in COAT:
        for team, hue in (("a", A_HUE), ("b", B_HUE)):
            f = dest / f"{name}-{team}.webp"
            coat(Image.open(f), hue).save(f, quality=90)
            print("coated", f.name)


def trim(im, size):
    box = im.getchannel("A").point(lambda v: 255 if v > 40 else 0).getbbox()      # (a faint haze is not the picture's edge)
    im = im.crop(box) if box else im
    k = size / max(im.size)
    return im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS)


def place(out, picks):
    dest = Path(__file__).resolve().parents[2] / "web" / "public" / "art" / "towers"
    dest.mkdir(parents=True, exist_ok=True)
    for name, seed in picks.items():
        if name == "field":
            Image.open(out / "raw" / f"field-{seed}.png").convert("RGB").resize((1280, 720), Image.LANCZOS).save(dest / "field.webp", quality=82)
            continue
        im = trim(Image.open(out / "cut" / f"{name}-{seed}.png"), SIZES[name])
        if name in MIRROR:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        if name in TEAM:
            a, b = recolor(im, A_HUE), recolor(im, B_HUE, 0.9, 0.82)
            if name in COAT:
                a, b = coat(a, A_HUE), coat(b, B_HUE)
            a.save(dest / f"{name}-a.webp", quality=90)
            b.save(dest / f"{name}-b.webp", quality=90)
        else:
            im.save(dest / f"{name}.webp", quality=90)
        print("placed", name, seed)


def main():
    args = sys.argv[1:]
    if "--coat" in args:
        return recoat()
    out = Path(args[0]); (out / "raw").mkdir(parents=True, exist_ok=True); (out / "cut").mkdir(exist_ok=True)
    if "--place" in args:
        return place(out, {k: int(v) for k, v in (kv.split("=") for kv in args[args.index("--place") + 1].split(","))})
    seeds = int(args[args.index("--seeds") + 1]) if "--seeds" in args else 5
    only = args[args.index("--only") + 1].split(",") if "--only" in args else list(PIECES) + ["field"]
    tiles = []
    for name in only:
        for s in range(seeds):
            raw = out / "raw" / f"{name}-{s}.png"
            if name == "field":
                if not raw.exists():
                    wf = workflow(FIELD, 9000 + s); wf["4"]["inputs"].update(width=1344, height=768)
                    gen.workflow = lambda p, sd, _wf=wf: _wf
                    render(FIELD, 9000 + s).save(raw)
                    gen.workflow = workflow
                tiles.append((f"field-{s}", Image.open(raw).convert("RGBA")))
                continue
            if not raw.exists():
                render(STYLE.format(subject=PIECES[name]), 9000 + s).save(raw)
                print("rendered", name, s, flush=True)
            im = cut(Image.open(raw).convert("RGB"), tol=TOL.get(name, 28), out=384, pad=0.02)
            im.save(out / "cut" / f"{name}-{s}.png")
            tiles.append((f"{name}-{s}", im))
            if name in TEAM:
                tiles.append((f"{name}-{s} B", recolor(im, B_HUE, 0.9, 0.82)))
    cols = 10
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 200, rows * 220), (225, 238, 250))
    d = ImageDraw.Draw(sheet)
    for i, (name, im) in enumerate(tiles):
        x, y = (i % cols) * 200, (i // cols) * 220
        t = im.resize((192, round(192 * im.height / im.width)) if im.width >= im.height else (round(192 * im.width / im.height), 192))
        sheet.paste(t, (x + 4, y + 4), t)
        d.text((x + 6, y + 200), name, fill=(20, 30, 50))
    sheet.save(out / "sheet.png")
    print("sheet", out / "sheet.png")


if __name__ == "__main__":
    main()
