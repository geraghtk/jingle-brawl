"""Turn the chosen whack2d.py takes into Whack-an-Elf's six pop-up pictures (web/public/art/whack/).

    python whack2d_place.py cut_dir --elf elf-up-3 --grump grump-up-2

Every picture is 384 x 384 and shares the hole's camera (hole-back / hole-front, still from whack.py):
a pop-up is centred on the hole and its cut-off waist sits below the line where the front lip hides it
(CLIP, 81% down), so it rises out of the hole. From one take of each character:
  - only the top of the figure is kept (head, arms, chest: the hole hides the rest);
  - the golden elf is the elf itself turned to gold (same pose, so the two read as kin);
  - the bonked face is the same picture knocked askew with stars round its head (two takes of "the
    same character, dazed" never matched closely enough to swap in mid-animation).
"""
import math, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

SIZE, CLIP = 384, 0.81
OUT = Path(__file__).resolve().parents[2] / "web/public/art/whack"
#       share of the figure kept (from the top: enough that the cut is well below the lip), its width on the
#       384 canvas. The cut edge sits at BOTTOM; how high the head comes follows from the take's own shape.
# a squat figure (the Grump) is sized by height instead (width 0): its head reaches TOP, wherever that puts its width
SHAPE = {"elf": (0.82, 250), "grump": (0.92, 0)}
TOP = 0.13
BOTTOM = CLIP + 0.06


def bust(cut_png, keep, width):
    """The top of the figure on the hole's canvas; returns it and how far down its head starts (0 to 1)."""
    im = Image.open(cut_png).convert("RGBA")
    im = im.crop(im.getbbox())
    im = im.crop((0, 0, im.width, int(im.height * keep)))
    im = im.crop(im.getbbox())                 # trimmed again: a ground shadow under the feet was wider than the figure
    if not width: width = min(340, int(im.width * (BOTTOM - TOP) * SIZE / im.height))
    h = int(im.height * width / im.width)
    im = im.resize((width, h), Image.LANCZOS)
    y = int(SIZE * BOTTOM) - h
    canvas = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    canvas.alpha_composite(im, ((SIZE - im.width) // 2, max(0, y)))
    return canvas, max(0, y) / SIZE


def golden(im):
    """The same picture in gold: its light and dark kept, its colours gone to a gold ramp (lines stay dark)."""
    a = np.asarray(im).astype(np.float32)
    lum = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]) / 255.0
    stops = [(0.0, (70, 40, 5)), (0.25, (150, 95, 10)), (0.55, (240, 175, 20)), (0.8, (255, 220, 70)), (1.0, (255, 250, 200))]
    out = np.zeros_like(a[..., :3])
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        m = (lum >= t0) & (lum <= t1)
        f = ((lum - t0) / (t1 - t0))[..., None]
        out = np.where(m[..., None], np.array(c0) + (np.array(c1) - np.array(c0)) * f, out)
    g = Image.fromarray(np.dstack([out, a[..., 3]]).astype(np.uint8), "RGBA")
    d = ImageDraw.Draw(g)
    for x, y, r in [(80, 150, 11), (306, 190, 8), (290, 96, 6)]:      # a few sparkles
        d.polygon([(x, y - r), (x + r * 0.28, y - r * 0.28), (x + r, y), (x + r * 0.28, y + r * 0.28), (x, y + r),
                   (x - r * 0.28, y + r * 0.28), (x - r, y), (x - r * 0.28, y - r * 0.28)], fill=(255, 255, 255, 255), outline=(200, 140, 0, 255))
    return g


def bonked(im, top):
    """Knocked askew, with stars circling where the head is."""
    tilt = im.rotate(-11, resample=Image.BICUBIC, center=(SIZE // 2, int(SIZE * CLIP)))
    d = ImageDraw.Draw(tilt)
    cx, cy = SIZE // 2 + 26, int(SIZE * top) + 26
    for k in range(5):
        ang = math.pi * (1.05 + 0.9 * k / 4)
        x, y, r = cx + math.cos(ang) * 96, cy + math.sin(ang) * 46, 15 - (k % 2) * 4
        pts = []
        for i in range(10):
            rr = r if i % 2 == 0 else r * 0.45
            t = -math.pi / 2 + i * math.pi / 5
            pts.append((x + math.cos(t) * rr, y + math.sin(t) * rr))
        d.polygon(pts, fill=(255, 214, 40, 255), outline=(29, 42, 58, 255))
    return tilt


def main():
    args = sys.argv[1:]
    cut = Path(args[0])
    pick = {k: args[args.index(f"--{k}") + 1] for k in ("elf", "grump")}
    made = {}
    for who, (keep, width) in SHAPE.items():
        made[who] = bust(cut / f"{pick[who]}.png", keep, width)
    made["gold"] = (golden(made["elf"][0]), made["elf"][1])
    for who, (im, top) in made.items():
        im.save(OUT / f"{who}.webp", quality=92, method=6)
        bonked(im, top).save(OUT / f"{who}-bonk.webp", quality=92, method=6)
        print("saved", who, "and", f"{who}-bonk")


if __name__ == "__main__":
    main()
