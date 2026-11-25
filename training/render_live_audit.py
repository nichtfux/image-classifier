"""Diagnostic figure from captured client strokes, not training examples."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from colour import planes, PALETTE
from colour_raster import reference

ROOT = Path(__file__).resolve().parent.parent


def main():
    font = lambda n: ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', n)
    image = Image.new('RGB', (760, 1190), '#faf8f3')
    draw = ImageDraw.Draw(image)
    draw.text((24, 12), 'Live drawing: what the model actually sees', font=font(25), fill='#181b24')
    draw.text((24, 50), 'Captured in DrawTest; current 40px model, unchanged weights.', font=font(18), fill='#444444')
    draw.text((24, 77), 'Canvas strokes (close view)               40px network input (enlarged)', font=font(18), fill='#444444')
    for row, index in enumerate((1, 2, 4, 6)):
        record = json.loads((ROOT / f'notes/live-paced-after-{index:02d}.json').read_text())
        cap, case = record['capture'], record['case']
        strokes = cap['strokes']
        y = 115 + row * 260
        pred = record['top'][0]
        points = np.concatenate([s['points'] for s in strokes])
        extent = round(float(np.ptp(points,axis=0).max()))
        label = f"{case['label']}, painted, {extent}px across: {pred['label']} {pred['prob']:.0%}"
        draw.text((24, y), label, font=font(21), fill='#181b24')
        paper = (246, 241, 230)
        lo, hi = points.min(0) - 12, points.max(0) + 12
        span = max(hi-lo)
        centre = (lo+hi)/2
        scale = 200/span
        canvas = Image.new('RGB', (215, 215), paper)
        cd = ImageDraw.Draw(canvas)
        for s in strokes:
            rgb = PALETTE[s['colour']-1].rgb
            rgb = tuple(round(v*255) for v in rgb) if max(rgb) <= 1 else tuple(rgb)
            pts = [tuple((np.asarray(p)-centre)*scale+107) for p in s['points']]
            width = max(1, round(s['width']*scale))
            if len(pts)>1:
                cd.line(pts, fill=rgb, width=width)
            r = width/2
            for x,z in pts:
                cd.ellipse((x-r,z-r,x+r,z+r), fill=rgb)
        drawing = [[[p[0] for p in s['points']], [p[1] for p in s['points']]] for s in strokes]
        ink, owner = reference(drawing, 40)
        x = planes(ink, owner, [s['colour']-1 for s in strokes])
        rgb = x[1:].transpose(1,2,0)+(1-x[0,...,None])*np.asarray(paper)/255
        tile = Image.fromarray(np.rint(rgb*255).clip(0,255).astype(np.uint8)).resize((215,215),Image.Resampling.NEAREST)
        image.paste(canvas, (75,y+35))
        image.paste(tile, (450,y+35))
    draw.text((24, 1153), 'Self-authored diagnostic cases; not a real-player accuracy estimate.', font=font(15), fill='#444444')
    image.save(ROOT / 'notes/live-drawing-input-comparison.png')


if __name__ == '__main__':
    main()
