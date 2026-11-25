"""Compiled, bounded stroke rasterization for building large paired corpora.

This implements the Luau rasterizer's segment bounding boxes in float64. The
independent NumPy Renderer remains the export/parity reference. No fast-math.
"""
import itertools
import math
import numpy as np
from numba import njit


@njit(cache=True)
def render_flat(xs, ys, offsets, size, box, radius):
    minx, miny = np.min(xs), np.min(ys)
    maxx, maxy = np.max(xs), np.max(ys)
    extent = max(maxx - minx, maxy - miny)
    scale = box / extent if extent > 1e-6 else 1.0
    cx, cy = (minx + maxx) * 0.5, (miny + maxy) * 0.5
    half, reach = size / 2, radius + 0.5
    dmin = np.full((size, size), np.inf, np.float64)
    for si in range(len(offsets) - 1):
        start, end = offsets[si], offsets[si + 1]
        for ai in range(start, max(start + 1, end - 1)):
            bi = min(ai + 1, end - 1)
            ax, ay = (xs[ai] - cx) * scale + half, (ys[ai] - cy) * scale + half
            bx, by = (xs[bi] - cx) * scale + half, (ys[bi] - cy) * scale + half
            dx, dy = bx - ax, by - ay
            ll = dx * dx + dy * dy
            x0, x1 = max(0, math.ceil(min(ax, bx) - reach - 0.5)), min(size-1, math.floor(max(ax, bx) + reach - 0.5))
            y0, y1 = max(0, math.ceil(min(ay, by) - reach - 0.5)), min(size-1, math.floor(max(ay, by) + reach - 0.5))
            for py in range(y0, y1 + 1):
                fy = py + 0.5
                for px in range(x0, x1 + 1):
                    fx = px + 0.5
                    t = 0.0
                    if ll > 1e-12:
                        t = min(1.0, max(0.0, ((fx-ax)*dx+(fy-ay)*dy)/ll))
                    ex, ey = fx - (ax+t*dx), fy - (ay+t*dy)
                    d2 = ex*ex + ey*ey
                    if d2 < dmin[py, px]:
                        dmin[py, px] = d2
    out = np.zeros((size, size), np.uint8)
    for y in range(size):
        for x in range(size):
            if dmin[y, x] < reach * reach:
                v = min(1.0, reach - math.sqrt(dmin[y, x]))
                out[y, x] = np.uint8(np.rint(v * 255.0))
    return out


def flatten(strokes):
    strokes = [s for s in strokes if len(s[0])]
    counts = [len(s[0]) for s in strokes]
    assert counts, "empty drawing"
    offsets = np.concatenate((np.array([0], np.int64), np.cumsum(counts, dtype=np.int64)))
    xs = np.fromiter(itertools.chain.from_iterable(s[0] for s in strokes), np.float64, count=int(offsets[-1]))
    ys = np.fromiter(itertools.chain.from_iterable(s[1] for s in strokes), np.float64, count=int(offsets[-1]))
    return xs, ys, offsets


def render(strokes, size=28):
    return render_flat(*flatten(strokes), size, 22.0 * size / 28, size / 28)
