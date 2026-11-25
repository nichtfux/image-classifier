"""Contact sheet of synth_doodles output: one row per thing, N samples each.

    python -I synth_preview.py --n 8 --out ../notes/synth-preview.png [--only bomb,tank]
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth_doodles as sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "notes", "synth-preview.png"))
    ap.add_argument("--only")
    a = ap.parse_args()
    names = a.only.split(",") if a.only else list(sd.THINGS)
    tile, label_w = 110, 120
    img = Image.new("RGB", (label_w + tile * a.n, tile * len(names)), "white")
    dr = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    rng = np.random.default_rng(a.seed)
    for r, name in enumerate(names):
        dr.text((6, r * tile + tile // 2 - 6), name, fill="black", font=font)
        for c in range(a.n):
            d = sd.make(name, rng)
            ox, oy = label_w + c * tile + 8, r * tile + 8
            k = (tile - 16) / 255
            for xs, ys in d:
                pts = [(ox + x * k, oy + y * k) for x, y in zip(xs, ys)]
                if len(pts) > 1:
                    dr.line(pts, fill="black", width=2)
            dr.rectangle([label_w + c * tile, r * tile, label_w + (c + 1) * tile - 1, (r + 1) * tile - 1], outline=(220, 220, 220))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    img.save(a.out)
    print(a.out)


if __name__ == "__main__":
    main()
