import sys, re
from pathlib import Path
from PIL import Image, ImageDraw
# grid.py out.png dir prefix-regex : every cut image matching, labelled, 8 per row, on the app background
fs = sorted(f for f in Path(sys.argv[2], 'cut').glob('*.png') if re.match(sys.argv[3], f.stem))
cols, cell = 8, 150
rows = (len(fs) + cols - 1) // cols
sheet = Image.new('RGB', (cols * cell, rows * (cell + 18)), (236, 238, 242))
d = ImageDraw.Draw(sheet)
for i, f in enumerate(fs):
    x, y = (i % cols) * cell, (i // cols) * (cell + 18)
    im = Image.open(f).resize((130, 130), Image.LANCZOS)
    sheet.paste(im, (x + 10, y + 4), im)
    d.text((x + 8, y + cell), f.stem, fill=(20, 20, 30))
sheet.save(sys.argv[1])
