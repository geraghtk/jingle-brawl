"""Frames from tools/shots/burst.mjs side by side, in order, so a movement can be read off one picture:

    python tools/shots/sheet.py burst_dir 12000 15000 [--prefix tv] [--cols 5] [--max 20] [--crop x0,y0,x1,y1] [--out sheet.png]

Takes the frames whose time (ms, in the file name) is between the two numbers, at most `--max` of them (evenly
spread), `--cols` to a row, each labelled with its time. `--crop` (fractions of the frame, e.g. 0.2,0.3,0.8,0.9)
looks closer at one part of the screen. Needs Pillow.
"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw


def main():
    a = sys.argv[1:]
    opt = lambda k, d: a[a.index(f"--{k}") + 1] if f"--{k}" in a else d
    src, lo, hi = Path(a[0]), int(a[1]), int(a[2])
    prefix, cols, most = opt("prefix", "tv"), int(opt("cols", 5)), int(opt("max", 20))
    crop = [float(x) for x in opt("crop", "0,0,1,1").split(",")]
    frames = sorted((int(p.stem.split("-")[1]), p) for p in src.glob(f"{prefix}-*.jpg"))
    frames = [f for f in frames if lo <= f[0] <= hi]
    if not frames:
        sys.exit("no frames in that range")
    if len(frames) > most:
        frames = [frames[round(i * (len(frames) - 1) / (most - 1))] for i in range(most)]
    first = Image.open(frames[0][1])
    box = (round(crop[0] * first.width), round(crop[1] * first.height), round(crop[2] * first.width), round(crop[3] * first.height))
    w = 1920 // cols
    h = round(w * (box[3] - box[1]) / (box[2] - box[0]))
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h), (20, 24, 32))
    d = ImageDraw.Draw(sheet)
    for i, (ms, p) in enumerate(frames):
        im = Image.open(p).convert("RGB").crop(box).resize((w, h), Image.LANCZOS)
        x, y = (i % cols) * w, (i // cols) * h
        sheet.paste(im, (x, y))
        d.rectangle((x, y, x + 62, y + 14), fill=(0, 0, 0))
        d.text((x + 3, y + 1), f"{ms} ms", fill=(255, 255, 255))
    out = Path(opt("out", str(src / f"sheet-{prefix}-{lo}-{hi}.png")))
    sheet.save(out)
    print(out)


if __name__ == "__main__":
    main()
