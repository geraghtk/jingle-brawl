"""Whack-an-Elf's pop-ups in the game's flat 2D look (Flux schnell in the local ComfyUI), replacing the
Blender ones (which looked like a different game beside the sticker avatars and the Ring Toss reindeer).

    python whack2d.py out_dir [--seeds 6] [--only elf,grump] [--bonk]

Three characters, each seen from the waist up (the hole hides the rest): the elf, the golden elf, and
the Grump (an original sour little fellow in a top hat: nobody's character). `--bonk` renders the
dazed face of each with the SAME seed and the same description, so a take and its bonked twin match as
closely as the model allows. Raw takes go to out/raw, cut-outs to out/cut and a labelled sheet to
out/sheet.png. Pick a seed per character, then `place.py`-style by hand: see README (Whack-an-Elf).
"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw
from gen import cut, render

STYLE = ("A flat vector cartoon illustration of {subject}. Clean bold shapes, thick dark outline, smooth soft cel "
         "shading, bright friendly colors, simple and polished like a modern mobile game character. Seen straight "
         "from the front, centered, the body cut off at the waist by the bottom edge of the picture. No sticker "
         "border, no white rim, no frame. Isolated on a plain solid flat medium grey background, no shadow, no "
         "text, no letters.")
WHO = {
    "elf": "a cheeky Christmas elf from the waist up, big head, pointy ears, rosy cheeks, a green pointed hat with "
           "a yellow bell on its tip, a green tunic with a red collar, both little arms raised and waving",
    "gold": "a shiny solid gold statue of a cheeky Christmas elf from the waist up, entirely golden yellow from hat "
            "to tunic, big head, pointy ears, a pointed hat with a bell on its tip, both little arms raised, small "
            "sparkles around it",
    "grump": "a sour grumpy little old troll from the waist up, big head, dark purple fur, thick angry frowning "
             "eyebrows, a big downturned scowling mouth, a tall black top hat with a sprig of holly, a dark grey "
             "scarf, little arms folded crossly",
}
FACE = {
    "up": "",
    "bonk": ", dazed and dizzy after a bonk on the head: eyes turned into spirals, tongue sticking out, a big red "
            "bump on the forehead, the hat knocked crooked, little yellow stars circling the head",
}


def main():
    args = sys.argv[1:]
    out = Path(args[0]); (out / "raw").mkdir(parents=True, exist_ok=True); (out / "cut").mkdir(exist_ok=True)
    seeds = int(args[args.index("--seeds") + 1]) if "--seeds" in args else 6
    only = args[args.index("--only") + 1].split(",") if "--only" in args else list(WHO)
    faces = ["up", "bonk"] if "--bonk" in args else ["up"]
    pick = [int(x) for x in args[args.index("--pick") + 1].split(",")] if "--pick" in args else range(seeds)
    tiles = []
    for who in only:
        for face in faces:
            for s in pick:
                name = f"{who}-{face}-{s}"
                raw = out / "raw" / f"{name}.png"
                if not raw.exists():
                    render(STYLE.format(subject=WHO[who] + FACE[face]), 7000 + s).save(raw)
                    print("rendered", name, flush=True)
                im = cut(Image.open(raw).convert("RGB"), out=384, pad=0.02)
                im.save(out / "cut" / f"{name}.png")
                tiles.append((name, im))
    # a labelled sheet on a snowy blue (how they will sit in the game)
    cols = len(list(pick)) or 1
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 200, rows * 220), (205, 225, 244))
    d = ImageDraw.Draw(sheet)
    for i, (name, im) in enumerate(tiles):
        x, y = (i % cols) * 200, (i // cols) * 220
        sheet.paste(im.resize((192, 192)), (x + 4, y + 4), im.resize((192, 192)))
        d.text((x + 6, y + 200), name, fill=(20, 30, 50))
    sheet.save(out / "sheet.png")
    print("sheet", out / "sheet.png")


if __name__ == "__main__":
    main()
