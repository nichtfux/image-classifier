"""Coloured contact sheet: synthetic doodles coloured the way training will colour them."""
import argparse, os, sys
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth_doodles as sd
import colour

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "notes", "colour-preview.png"))
ap.add_argument("--only")
a = ap.parse_args()
names = a.only.split(",") if a.only else list(sd.THINGS)
tile, label_w = 100, 110
img = Image.new("RGB", (label_w + tile * a.n, tile * len(names)), (246, 241, 230))
dr = ImageDraw.Draw(img)
rng = np.random.default_rng(a.seed)
for r, name in enumerate(names):
    dr.text((6, r * tile + tile // 2 - 6), name, fill="black")
    for c in range(a.n):
        d = sd.make(name, rng)
        cols = colour.colourise(name, colour.stroke_ink(d), rng)
        ox, oy = label_w + c * tile + 8, r * tile + 8
        k = (tile - 16) / 255
        for (xs, ys), ci in zip(d, cols):
            pts = [(ox + x * k, oy + y * k) for x, y in zip(xs, ys)]
            if len(pts) > 1:
                dr.line(pts, fill=colour.PALETTE[ci].rgb, width=3, joint="curve")
        dr.rectangle([label_w + c * tile, r * tile, label_w + (c + 1) * tile - 1, (r + 1) * tile - 1], outline=(215, 208, 195))
img.save(a.out)
print(a.out)
