"""Strokes -> 28x28 network input. luau/DrawRecognizer does exactly the same steps,
so keep the two in sync (constants are written into the exported Luau module).

Steps:
 1. Bounding box of all points; the longer side is scaled to BOX pixels, aspect kept,
    and the box centre goes to the image centre (14, 14).
 2. Each pixel's value comes from its centre's distance d to the nearest stroke
    segment: v = clamp(RADIUS + 0.5 - d, 0, 1). So the line width is fixed in
    28x28 space (about 2*RADIUS + 1 px) no matter how big the player drew.
"""
import numpy as np

SIZE = 28
BOX = 22.0
RADIUS = 1.0

_yy, _xx = np.mgrid[0:SIZE, 0:SIZE]
_PX = (_xx + 0.5).reshape(-1, 1).astype(np.float64)
_PY = (_yy + 0.5).reshape(-1, 1).astype(np.float64)


class Renderer:
    """Same stroke geometry at different resolutions; the 28px path is unchanged.

    Scale the bounding box and ink radius with the canvas. Anti-alias coverage
    still spans one output pixel, so 40px resolves edges instead of enlarging 28px.
    """
    def __init__(self, size=28):
        self.size = int(size)
        self.box = BOX * self.size / SIZE
        self.radius = RADIUS * self.size / SIZE
        yy, xx = np.mgrid[0:self.size, 0:self.size]
        self.px = (xx + 0.5).reshape(-1, 1).astype(np.float64)
        self.py = (yy + 0.5).reshape(-1, 1).astype(np.float64)

    def render(self, strokes):
        pts = [np.stack([np.asarray(xs, np.float64), np.asarray(ys, np.float64)], 1) for xs, ys in strokes if len(xs)]
        allp = np.concatenate(pts, 0)
        mn, mx = allp.min(0), allp.max(0)
        extent = max(mx - mn)
        scale = self.box / extent if extent > 1e-6 else 1.0
        pts = [(p - (mn + mx) * 0.5) * scale + self.size / 2 for p in pts]
        a, b = segments(pts)
        ax, ay = a[:, 0][None], a[:, 1][None]
        dx, dy = (b[:, 0] - a[:, 0])[None], (b[:, 1] - a[:, 1])[None]
        ll = dx * dx + dy * dy
        with np.errstate(invalid="ignore", divide="ignore"):
            t = ((self.px - ax) * dx + (self.py - ay) * dy) / ll
        t = np.where(ll > 1e-12, np.clip(t, 0.0, 1.0), 0.0)
        ex, ey = self.px - (ax + t * dx), self.py - (ay + t * dy)
        d = np.sqrt((ex * ex + ey * ey).min(1))
        return np.clip(self.radius + 0.5 - d, 0.0, 1.0).reshape(self.size, self.size)


def normalise(strokes):
    """strokes: list of (xs, ys) sequences. Returns list of (N,2) float64 arrays in 28x28 space."""
    pts = [np.stack([np.asarray(xs, np.float64), np.asarray(ys, np.float64)], 1) for xs, ys in strokes if len(xs)]
    allp = np.concatenate(pts, 0)
    mn = allp.min(0)
    mx = allp.max(0)
    size = max(mx[0] - mn[0], mx[1] - mn[1])
    scale = BOX / size if size > 1e-6 else 1.0
    c = (mn + mx) * 0.5
    return [(p - c) * scale + SIZE / 2 for p in pts]


def segments(npts):
    a, b = [], []
    for p in npts:
        if len(p) == 1:
            a.append(p)
            b.append(p)
        else:
            a.append(p[:-1])
            b.append(p[1:])
    return np.concatenate(a, 0), np.concatenate(b, 0)


def render_norm(npts):
    a, b = segments(npts)
    ax, ay = a[:, 0][None], a[:, 1][None]
    dx, dy = (b[:, 0] - a[:, 0])[None], (b[:, 1] - a[:, 1])[None]
    ll = dx * dx + dy * dy
    with np.errstate(invalid="ignore", divide="ignore"):
        t = ((_PX - ax) * dx + (_PY - ay) * dy) / ll
    t = np.where(ll > 1e-12, np.clip(t, 0.0, 1.0), 0.0)
    ex = _PX - (ax + t * dx)
    ey = _PY - (ay + t * dy)
    d = np.sqrt((ex * ex + ey * ey).min(1))
    v = np.clip(RADIUS + 0.5 - d, 0.0, 1.0)
    return v.reshape(SIZE, SIZE)


def render(strokes):
    return render_norm(normalise(strokes))


def augment(strokes, rng):
    """Random rotation / stretch / shear / jitter, applied to the raw strokes before render."""
    ang = rng.uniform(-0.25, 0.25)
    sx, sy = rng.uniform(0.8, 1.2, 2)
    sh = rng.uniform(-0.2, 0.2)
    c, s = np.cos(ang), np.sin(ang)
    m = np.array([[c, -s], [s, c]]) @ np.array([[sx, sh], [0, sy]])
    out = []
    for xs, ys in strokes:
        p = np.stack([np.asarray(xs, np.float64), np.asarray(ys, np.float64)], 1) @ m.T
        p += rng.normal(0, 1.5, p.shape)
        out.append((p[:, 0], p[:, 1]))
    return out
