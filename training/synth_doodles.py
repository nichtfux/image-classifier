"""Hand-written doodle programs for the things Quick, Draw! doesn't have.

Each thing has a few drawing programs (variants) built from simple strokes, the way people
draw them in a 20-second round. `humanise` then wobbles every stroke, misses closures,
overshoots ends, drops details and distorts the whole drawing, so no two come out alike.
Output is Quick, Draw!'s simplified format: per stroke [[xs], [ys]], integers 0-255.

    python -I synth_doodles.py --per-class 20 --out ../data/synth/review.json   # review sample
    python -I synth_doodles.py --per-class 5000 --ndjson ../data/synth          # training set
"""
import argparse
import json
import math
import os

import numpy as np

TAU = 2 * math.pi


# ---------------------------------------------------------------- primitives (y points down)
def P(*pts):
    return np.array(pts, float)


def arc(cx, cy, rx, ry, a0, a1, n=None):
    n = n or max(8, int(abs(a1 - a0) / TAU * 40))
    t = np.linspace(a0, a1, n)
    return np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], 1)


def ell(cx, cy, rx, ry, rng, n=36):
    a0 = rng.uniform(0, TAU)
    return arc(cx, cy, rx, ry, a0, a0 + TAU * rng.uniform(0.97, 1.08), n)


def rect(x, y, w, h):
    return P((x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y))


def bez(p0, p1, p2, p3, n=24):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = map(np.asarray, (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def qbez(p0, p1, p2, n=18):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2 = map(np.asarray, (p0, p1, p2))
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def spiral(cx, cy, r, turns, cw=True, r0=0.0, n=None):
    n = n or int(40 * turns)
    t = np.linspace(0, 1, n)
    a = t * turns * TAU * (1 if cw else -1)
    rr = r0 + (r - r0) * t
    return np.stack([cx + rr * np.cos(a), cy + rr * np.sin(a)], 1)


def bumpy(cx, cy, rx, ry, bumps, depth, rng, a0=0.0, a1=TAU, n=None):
    """Cloud-like outline: a sequence of outward bumps along an ellipse."""
    n = n or bumps * 10
    t = np.linspace(a0, a1, n)
    k = np.abs(np.sin((t - a0) / (a1 - a0) * bumps * math.pi))
    r = 1 + depth * k
    return np.stack([cx + rx * r * np.cos(t), cy + ry * r * np.sin(t)], 1)


def wave(x0, x1, y, amp, periods, n=None):
    n = n or int(periods * 16) + 2
    x = np.linspace(x0, x1, n)
    return np.stack([x, y + amp * np.sin((x - x0) / (x1 - x0) * periods * TAU)], 1)


def zig(x0, x1, y, amp, teeth):
    xs = np.linspace(x0, x1, teeth * 2 + 1)
    ys = [y + (amp if i % 2 else -amp) * 0.5 for i in range(len(xs))]
    return np.stack([xs, ys], 1)


def star_poly(cx, cy, r_out, r_in, k, rot=-math.pi / 2):
    pts = []
    for i in range(2 * k + 1):
        r = r_out if i % 2 == 0 else r_in
        a = rot + i * math.pi / k
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return np.array(pts)


def sparks(cx, cy, r, rng, k=None):
    k = k or rng.integers(3, 7)
    out = []
    for i in range(k):
        a = rng.uniform(0, TAU)
        r0 = r * rng.uniform(0.2, 0.5)
        out.append(P((cx + r0 * math.cos(a), cy + r0 * math.sin(a)),
                     (cx + r * math.cos(a), cy + r * math.sin(a))))
    return out


def eye_pair(cx, cy, dx, r, rng, style=None):
    style = style or rng.choice(["dot", "circle", "x"])
    out = []
    for x in (cx - dx, cx + dx):
        if style == "dot":
            out.append(ell(x, cy, r * 0.5, r * 0.5, rng, 10))
        elif style == "circle":
            out.append(ell(x, cy, r, r, rng, 16))
            if rng.random() < 0.5:
                out.append(ell(x, cy, r * 0.3, r * 0.3, rng, 8))
        else:
            out += [P((x - r, cy - r), (x + r, cy + r)), P((x + r, cy - r), (x - r, cy + r))]
    return out


def maybe(rng, p):
    return rng.random() < p


# ---------------------------------------------------------------- the things
def bomb(rng):
    r = rng.uniform(26, 34)
    s = [ell(50, 58, r, r * rng.uniform(0.9, 1.05), rng)]
    # People often put the cap straight on top. Diagonal-only attachment left
    # that common pose outside the corpus, even after the small global rotation.
    a = (-math.pi / 2 + rng.uniform(-0.35, 0.35) if maybe(rng, 0.45)
         else -math.pi / 4 + rng.uniform(-0.4, 0.3))
    nx, ny = 50 + r * math.cos(a), 58 + r * math.sin(a)
    # What separates a bomb from an apple: a boxy cap, a LONG wiggly fuse and a big spark burst.
    # (The first mouse test read a short-fused bomb as an apple.)
    w = rng.uniform(8, 13)
    ux, uy = math.cos(a), math.sin(a)
    vx, vy = -uy, ux
    h = rng.uniform(6, 10)
    c = [(nx + vx * w / 2 - ux * 2, ny + vy * w / 2 - uy * 2), (nx + vx * w / 2 + ux * h, ny + vy * w / 2 + uy * h),
         (nx - vx * w / 2 + ux * h, ny - vy * w / 2 + uy * h), (nx - vx * w / 2 - ux * 2, ny - vy * w / 2 - uy * 2)]
    s.append(np.array(c + [c[0]]))
    nx, ny = nx + ux * h, ny + uy * h
    L = rng.uniform(22, 38)
    t = np.linspace(0, 1, 24)
    wig = rng.uniform(2, 6) * np.sin(t * TAU * rng.uniform(1.0, 2.0) + rng.uniform(0, TAU))
    fx = nx + ux * L * t + vx * wig + rng.uniform(-0.3, 0.3) * L * t ** 2
    fy = ny + uy * L * t + vy * wig - rng.uniform(0.2, 0.5) * L * t ** 2
    s.append(np.stack([fx, fy], 1))
    ex, ey = fx[-1], fy[-1]
    if maybe(rng, 0.9):
        if maybe(rng, 0.5):
            s += sparks(ex, ey, rng.uniform(8, 13), rng, k=int(rng.integers(5, 9)))
        else:
            s.append(star_poly(ex, ey, rng.uniform(8, 12), rng.uniform(3, 5), int(rng.integers(5, 8)), rot=rng.uniform(0, TAU)))
    if maybe(rng, 0.4):  # shine
        s.append(arc(50, 58, r * 0.65, r * 0.65, math.pi * 1.1, math.pi * 1.45, 8))
    return s


def nuke(rng):
    v = rng.choice(["mushroom", "mushroom", "mushroom", "trefoil"])
    if v == "trefoil":
        s = [ell(50, 50, 40, 40, rng)] if maybe(rng, 0.7) else []
        s.append(ell(50, 50, 5, 5, rng, 12))
        for i in range(3):
            a = -math.pi / 2 + i * TAU / 3
            inner = arc(50, 50, 9, 9, a - 0.45, a + 0.45, 6)
            outer = arc(50, 50, 33, 33, a + 0.5, a - 0.5, 12)
            s.append(np.concatenate([inner, outer, inner[:1]]))
        return s
    capw, caph = rng.uniform(30, 40), rng.uniform(16, 22)
    cy = 30
    cap = bumpy(50, cy, capw, caph, int(rng.integers(5, 9)), rng.uniform(0.08, 0.18), rng, math.pi * 0.95, math.pi * 2.05)
    capb = qbez((50 + capw, cy + 4), (50, cy + caph * rng.uniform(0.6, 1.1)), (50 - capw, cy + 4))
    s = [np.concatenate([cap, capb])]
    sw = rng.uniform(7, 12)
    base_y = rng.uniform(82, 90)
    s.append(qbez((50 - sw, cy + caph * 0.75), (50 - sw * 0.6, 60), (50 - sw * 1.6, base_y - 4)))
    s.append(qbez((50 + sw, cy + caph * 0.75), (50 + sw * 0.6, 60), (50 + sw * 1.6, base_y - 4)))
    if maybe(rng, 0.8):
        s.append(bumpy(50, base_y, rng.uniform(26, 38), rng.uniform(7, 10), int(rng.integers(4, 7)), 0.15, rng,
                       math.pi, math.pi * 2))
        s.append(P((50 - 38, base_y + 2), (50 + 38, base_y + 2)))
    if maybe(rng, 0.35):
        s.append(ell(50, 58, sw * 2.2, 4, rng, 20))
    return s


def meteor(rng):
    r = rng.uniform(13, 20)
    cx, cy = 68, 66
    s = [bumpy(cx, cy, r, r, int(rng.integers(5, 9)), rng.uniform(0.0, 0.12), rng) if maybe(rng, 0.5) else ell(cx, cy, r, r, rng)]
    if maybe(rng, 0.5):
        for _ in range(rng.integers(1, 4)):
            s.append(ell(cx + rng.uniform(-r / 2, r / 2), cy + rng.uniform(-r / 2, r / 2), r * 0.18, r * 0.15, rng, 10))
    d = np.array([-1.0, -1.0]) / math.sqrt(2)
    nrm = np.array([1.0, -1.0]) / math.sqrt(2)
    if maybe(rng, 0.55):  # streak lines
        for i in range(int(rng.integers(3, 6))):
            off = (i - 2) * r * 0.45
            st = np.array([cx, cy]) + nrm * off * 0.9 + d * r * 0.8
            ln = rng.uniform(25, 50)
            s.append(P(st, st + d * ln))
    else:  # flame tail outline
        p1 = np.array([cx, cy]) + nrm * r
        p2 = np.array([cx, cy]) - nrm * r
        tip = np.array([cx, cy]) + d * rng.uniform(55, 70)
        s.append(np.concatenate([qbez(p1, p1 + d * 25 + nrm * 6, tip), qbez(tip, p2 + d * 25 - nrm * 6, p2)]))
    return s


def volcano(rng):
    base = 92
    top_y = rng.uniform(36, 48)
    tw = rng.uniform(8, 16)
    s = [P((6, base), (50 - tw, top_y), (50 + tw, top_y), (94, base))]
    if maybe(rng, 0.5):
        s.append(P((4, base), (96, base)))
    # the eruption is what makes it a volcano and not a mountain: always draw it
    k = int(rng.integers(3, 6))
    for i in range(k):
        x0 = 50 - tw + (2 * tw) * (i + 0.5) / k
        s.append(qbez((x0, top_y - 1), (x0 + rng.uniform(-10, 10), top_y - 20), (x0 + rng.uniform(-22, 22), top_y - rng.uniform(18, 34))))
    if maybe(rng, 0.6):  # lava down the side
        side = rng.choice([-1, 1])
        x0 = 50 + side * tw * 0.6
        s.append(P((x0, top_y), (x0 + side * 6, top_y + 14), (x0 + side * 3, top_y + 22), (x0 + side * 10, top_y + 34)))
    if maybe(rng, 0.35):
        s.append(bumpy(50 + rng.uniform(-15, 15), top_y - 38, 14, 7, 4, 0.25, rng))
    return s


def tsunami(rng):
    """A breaking wave: the back rises from the left, the crest curls over to the right and down."""
    s = []
    base = 90
    j = lambda v: v + rng.uniform(-4, 4)
    # one outline: up the back on the right, over the crest, the lip hangs forward to the left and
    # curls in, then the hollow face runs back under the lip and down to the ground
    outline = np.concatenate([
        qbez((j(96), base), (j(92), j(22)), (j(52), j(12))),
        qbez((52, 12), (j(18), j(10)), (j(18), j(42))),
        qbez((18, 42), (j(26), 50), (j(34), 38)),
        qbez((34, 38), (j(40), j(26)), (j(54), j(30))),
        qbez((54, 30), (j(64), j(62)), (j(30), base)),
    ])
    s.append(outline)
    if maybe(rng, 0.6):
        s.append(spiral(28, 36, rng.uniform(4, 6), rng.uniform(1, 1.6), r0=0.5))
    if maybe(rng, 0.7):
        s.append(P((0, base), (100, base)))
    if maybe(rng, 0.5):  # spray drops off the lip
        for _ in range(int(rng.integers(2, 5))):
            s.append(ell(rng.uniform(8, 40), rng.uniform(2, 10), 1.8, 1.8, rng, 6))
    if maybe(rng, 0.5):  # a water line on the back of the wave
        s.append(qbez((80, base - 14), (78, 40), (58, 22)))
    if maybe(rng, 0.3):  # tiny house to show scale
        hx = rng.uniform(4, 12)
        s.append(rect(hx - 4, base - 8, 8, 8))
        s.append(P((hx - 5, base - 8), (hx, base - 13), (hx + 5, base - 8)))
    return s


def black_hole(rng):
    s = []
    """Never a plain spiral (that is its own class): always a scribbled-in black centre, plus a
    flat glowing disk around it or curved arms pulled into it, often with stars."""
    s = []
    cr = rng.uniform(11, 17)
    s.append(ell(50, 50, cr, cr * rng.uniform(0.9, 1.1), rng))
    s.append(hatch_circle(50, 50, cr * 0.9, rng))
    v = rng.choice(["disk", "disk", "arms", "both"])
    if v in ("disk", "both"):
        tilt = rng.uniform(0.25, 0.45)
        for k, rr in enumerate((cr * 1.9, cr * 2.8)):
            if k == 0 or maybe(rng, 0.7):
                s.append(ell(50, 50, rr * 1.15, rr * tilt, rng))
    if v in ("arms", "both"):
        n = int(rng.integers(2, 5))
        a0 = rng.uniform(0, TAU)
        cw = 1 if maybe(rng, 0.5) else -1
        for i in range(n):
            a = a0 + i * TAU / n
            t = np.linspace(0, 1, 16)
            rr = cr * 1.1 + t * rng.uniform(22, 30)
            aa = a + cw * t * rng.uniform(1.2, 2.0)
            s.append(np.stack([50 + rr * np.cos(aa), 50 + rr * np.sin(aa)], 1))
    if maybe(rng, 0.6):  # stars around it
        for _ in range(int(rng.integers(2, 6))):
            x, y = rng.uniform(5, 95), rng.uniform(5, 95)
            if math.hypot(x - 50, y - 50) > cr * 2.2:
                s += [P((x - 2.5, y), (x + 2.5, y)), P((x, y - 2.5), (x, y + 2.5))]
    return s


def hatch_circle(cx, cy, r, rng):
    """One back-and-forth scribble that fills a circle, the way people blacken something in."""
    k = int(rng.integers(5, 9))
    ang = rng.uniform(0, math.pi)
    u, v = np.array([math.cos(ang), math.sin(ang)]), np.array([-math.sin(ang), math.cos(ang)])
    pts = []
    for i in range(k):
        o = -r + 2 * r * (i + 0.5) / k
        h = math.sqrt(max(0.0, r * r - o * o))
        a, b = np.array([cx, cy]) + v * o - u * h, np.array([cx, cy]) + v * o + u * h
        pts += [a, b] if i % 2 == 0 else [b, a]
    return np.array(pts)


def pistol(rng):
    s = []
    bl = rng.uniform(40, 55)
    bh = rng.uniform(10, 15)
    x0, y0 = 22, 30
    s.append(P((x0, y0), (x0 + bl, y0), (x0 + bl, y0 + bh), (x0 + 16, y0 + bh)))
    gx = x0 + rng.uniform(2, 8)
    gw = rng.uniform(13, 18)
    gl = rng.uniform(28, 36)
    slope = rng.uniform(4, 10)
    s.append(P((x0 + 16, y0 + bh), (gx + gw + slope * 0.4, y0 + bh), (gx + gw + slope, y0 + bh + gl),
               (gx + slope, y0 + bh + gl), (gx, y0 + bh), (x0, y0 + bh), (x0, y0)))
    tg_x = gx + gw + slope * 0.4
    s.append(qbez((tg_x, y0 + bh), (tg_x + 12, y0 + bh + 2), (tg_x + 8, y0 + bh + 12)))
    s.append(qbez((tg_x + 8, y0 + bh + 12), (tg_x + 2, y0 + bh + 13), (tg_x - 1, y0 + bh + 9)))
    if maybe(rng, 0.6):
        s.append(P((tg_x + 3, y0 + bh), (tg_x + 5, y0 + bh + 7)))
    if maybe(rng, 0.35):  # revolver cylinder
        s.append(ell(x0 + 22, y0 + bh * 0.5, 7, 7, rng, 14))
    if maybe(rng, 0.4):
        s.append(P((x0 + bl - 4, y0), (x0 + bl - 4, y0 - 3)))
    if maybe(rng, 0.25):  # bang
        s += sparks(x0 + bl + 10, y0 + bh / 2, 8, rng)
    return s


def tank(rng):
    s = []
    tl, tr = 8, 92
    ty, tb = 62, 82
    s.append(np.concatenate([P((tl + 8, ty), (tr - 8, ty)), arc(tr - 10, (ty + tb) / 2, 10, (tb - ty) / 2, -math.pi / 2, math.pi / 2, 10),
                             P((tr - 8, tb), (tl + 8, tb)), arc(tl + 10, (ty + tb) / 2, 10, (tb - ty) / 2, math.pi / 2, math.pi * 1.5, 10)]))
    if maybe(rng, 0.8):
        k = int(rng.integers(3, 7))
        for i in range(k):
            x = tl + 14 + (tr - tl - 28) * i / max(1, k - 1)
            s.append(ell(x, (ty + tb) / 2, 5, 5, rng, 12))
    hull_top = ty - rng.uniform(10, 16)
    s.append(P((tl + 6, ty), (tl + 14, hull_top), (tr - 10, hull_top), (tr - 4, ty)))
    tx = rng.uniform(40, 55)
    tw = rng.uniform(16, 24)
    th = rng.uniform(10, 15)
    if maybe(rng, 0.5):
        s.append(arc(tx, hull_top, tw, th, math.pi, TAU, 14))
    else:
        s.append(P((tx - tw, hull_top), (tx - tw + 4, hull_top - th), (tx + tw - 4, hull_top - th), (tx + tw, hull_top)))
    by = hull_top - th * 0.55
    bx = tx + tw - 2
    blen = rng.uniform(26, 38)
    ang = rng.uniform(-0.35, 0.05)
    ux, uy = math.cos(ang), math.sin(ang)
    s.append(P((bx, by - 2), (bx + ux * blen, by - 2 + uy * blen)))
    s.append(P((bx, by + 2), (bx + ux * blen, by + 2 + uy * blen)))
    if maybe(rng, 0.5):
        s.append(P((bx + ux * blen, by - 3 + uy * blen), (bx + ux * blen, by + 3 + uy * blen)))
    if maybe(rng, 0.3):
        s.append(P((tx - 4, hull_top - th), (tx - 4, hull_top - th - 8)))
    return s


def rocket(rng):
    s = []
    w = rng.uniform(11, 16)
    top, bot = 10, rng.uniform(68, 76)
    nose = rng.uniform(16, 24)
    s.append(np.concatenate([P((50 - w, bot), (50 - w, top + nose)), qbez((50 - w, top + nose), (50 - w * 0.6, top + 2), (50, top)),
                             qbez((50, top), (50 + w * 0.6, top + 2), (50 + w, top + nose)), P((50 + w, top + nose), (50 + w, bot), (50 - w, bot))]))
    if maybe(rng, 0.6):
        s.append(ell(50, top + nose + 12, w * 0.45, w * 0.45, rng, 14))
    fl = rng.uniform(10, 16)
    s.append(P((50 - w, bot - fl * 1.4), (50 - w - fl, bot + 4), (50 - w, bot)))
    s.append(P((50 + w, bot - fl * 1.4), (50 + w + fl, bot + 4), (50 + w, bot)))
    if maybe(rng, 0.85):
        fy = bot + rng.uniform(12, 22)
        if maybe(rng, 0.5):
            s.append(P((50 - w * 0.7, bot), (50 - w * 0.35, fy - 5), (50, fy), (50 + w * 0.35, fy - 5), (50 + w * 0.7, bot)))
        else:
            s.append(np.concatenate([qbez((50 - w * 0.7, bot), (50 - w * 0.6, fy - 4), (50, fy)), qbez((50, fy), (50 + w * 0.6, fy - 4), (50 + w * 0.7, bot))]))
        if maybe(rng, 0.4):
            s.append(P((50, bot), (50, fy - 8)))
    if maybe(rng, 0.25):
        s.append(P((50 - w, top + nose + 2), (50 + w, top + nose + 2)))
    return s


def grenade(rng):
    rx, ry = rng.uniform(20, 26), rng.uniform(25, 31)
    cy = 60
    s = [ell(50, cy, rx, ry, rng)]
    if maybe(rng, 0.75):
        for k in (-1, 0, 1):
            if k == 0 or maybe(rng, 0.8):
                s.append(P((50 - rx * math.sqrt(max(0, 1 - (k * 0.45) ** 2)), cy + k * ry * 0.45),
                           (50 + rx * math.sqrt(max(0, 1 - (k * 0.45) ** 2)), cy + k * ry * 0.45)))
        s.append(qbez((50, cy - ry), (50 + rng.uniform(-4, 4), cy), (50, cy + ry)))
    top = cy - ry
    s.append(rect(43, top - 8, 14, 9))
    s.append(np.concatenate([P((57, top - 6), (70, top - 2)), qbez((70, top - 2), (75, top + 14), (68, top + 26))]))
    if maybe(rng, 0.85):
        s.append(ell(rng.uniform(35, 40), top - 12, rng.uniform(5, 8), rng.uniform(5, 8), rng, 14))
    return s


def dynamite(rng):
    s = []
    n = int(rng.choice([1, 3, 3, 3]))
    w, h = (14, 54) if n == 1 else (12, 50)
    xs = [50 - w / 2] if n == 1 else [50 - w * 1.5, 50 - w / 2, 50 + w / 2]
    top = 30
    for x in xs:
        s.append(rect(x, top, w, h))
    if n == 3 and maybe(rng, 0.7):
        for yy in (top + h * 0.25, top + h * 0.7):
            s.append(P((xs[0] - 1, yy), (xs[-1] + w + 1, yy)))
    fx = 50
    ex, ey = fx + rng.uniform(-18, 18), top - rng.uniform(16, 26)
    s.append(qbez((fx, top), (fx + rng.uniform(-14, 14), top - 12), (ex, ey)))
    if maybe(rng, 0.75):
        s += sparks(ex, ey, rng.uniform(5, 9), rng)
    if n == 1 and maybe(rng, 0.4):  # TNT
        y0, y1 = top + h * 0.35, top + h * 0.6
        cx = 50
        s += [P((cx - 5, y0), (cx + 5, y0)), P((cx, y0), (cx, y1))]
    return s


def ninja_star(rng):
    k = int(rng.choice([4, 4, 4, 3, 6]))
    ro, ri = rng.uniform(38, 44), rng.uniform(9, 16)
    s = [star_poly(50, 50, ro, ri, k, rot=-math.pi / 2 + rng.uniform(-0.4, 0.4))]
    if maybe(rng, 0.75):
        s.append(ell(50, 50, ri * 0.55, ri * 0.55, rng, 12))
    return s


def bow_and_arrow(rng):
    s = []
    r = rng.uniform(34, 42)
    half = rng.uniform(1.0, 1.25)
    bow = arc(40, 50, r * 0.55, r, -half, half, 24)
    s.append(bow)
    pull = rng.uniform(0, 14)
    a, b = bow[0], bow[-1]
    if pull > 3:
        nock = (a[0] - pull, 50)
        s.append(P(a, nock, b))
    else:
        nock = (a[0], 50)
        s.append(P(a, b))
    if True:  # the arrow always goes with the bow (a bow alone was rejected in review)
        x0 = nock[0]
        x1 = x0 + rng.uniform(60, 75)
        s.append(P((x0, 50), (x1, 50)))
        s.append(P((x1 - 8, 45), (x1, 50), (x1 - 8, 55)))
        for i in range(int(rng.integers(1, 3))):
            s.append(P((x0 + 3 + i * 4, 50), (x0 - 2 + i * 4, 45)))
            s.append(P((x0 + 3 + i * 4, 50), (x0 - 2 + i * 4, 55)))
    return s


def shield(rng):
    s = []
    if maybe(rng, 0.7):
        w = rng.uniform(30, 38)
        top, bot = 12, 90
        s.append(np.concatenate([P((50 - w, top), (50 + w, top)), qbez((50 + w, top), (50 + w, top + 50), (50, bot)),
                                 qbez((50, bot), (50 - w, top + 50), (50 - w, top))]))
        e = rng.choice(["cross", "chevron", "line", "none", "inner"])
        if e == "cross":
            s += [P((50, top + 8), (50, bot - 12)), P((50 - w * 0.6, top + 30), (50 + w * 0.6, top + 30))]
        elif e == "chevron":
            s.append(P((50 - w * 0.6, top + 22), (50, top + 40), (50 + w * 0.6, top + 22)))
        elif e == "line":
            s.append(P((50, top), (50, bot)))
        elif e == "inner":
            s.append(np.concatenate([P((50 - w * 0.75, top + 7), (50 + w * 0.75, top + 7)),
                                     qbez((50 + w * 0.75, top + 7), (50 + w * 0.75, top + 46), (50, bot - 10)),
                                     qbez((50, bot - 10), (50 - w * 0.75, top + 46), (50 - w * 0.75, top + 7))]))
    else:
        r = rng.uniform(38, 44)
        s.append(ell(50, 50, r, r, rng))
        s.append(ell(50, 50, r * rng.uniform(0.6, 0.8), r * rng.uniform(0.6, 0.8), rng))
        s.append(ell(50, 50, r * 0.18, r * 0.18, rng, 12))
    return s


def dinosaur(rng):
    v = rng.choice(["trex", "longneck", "longneck", "longneck", "stego", "stego"])
    s = []
    if v == "longneck":
        body = ell(52, 60, 24, 14, rng)
        s.append(body)
        s.append(np.concatenate([qbez((34, 52), (24, 30), (22, 12)), ell(18, 10, 7, 4.5, rng, 14)[:10]]))
        s.append(qbez((32, 58), (30, 38), (26, 16)))
        s.append(qbez((75, 56), (88, 58), (98, 70)))
        s.append(qbez((74, 64), (86, 66), (98, 70)))
        for x in (38, 46, 60, 68):
            s.append(P((x, 70), (x, 86)))
            if maybe(rng, 0.5):
                s.append(P((x + 4, 70), (x + 4, 86), (x, 86)))
        if maybe(rng, 0.6):
            s.append(ell(16, 9, 1, 1, rng, 6))
    elif v == "stego":
        body = np.concatenate([arc(50, 66, 34, 24, math.pi, TAU, 24), P((84, 66), (16, 66))])
        s.append(body)
        for i in range(int(rng.integers(4, 7))):
            a = math.pi + (i + 1) * math.pi / 7
            x, y = 50 + 34 * math.cos(a), 66 + 24 * math.sin(a)
            nx, ny = math.cos(a), math.sin(a)
            s.append(P((x - ny * 4 - 1, y + nx * 4), (x + nx * 12, y + ny * 12), (x + ny * 4 + 1, y - nx * 4)))
        s.append(np.concatenate([P((16, 66), (8, 58)), ell(8, 56, 6, 4, rng, 12)]))
        s.append(P((84, 62), (98, 70), (84, 66)))
        for x in (26, 38, 62, 74):
            s.append(P((x, 66), (x, 80)))
    else:  # T-rex
        s.append(np.concatenate([P((30, 40), (30, 14), (56, 12), (60, 22), (46, 26), (44, 32), (56, 34), (52, 40))]))
        s.append(np.concatenate([qbez((30, 38), (12, 50), (16, 68)), qbez((16, 68), (30, 80), (48, 72)),
                                 qbez((48, 72), (70, 74), (96, 84)), qbez((96, 84), (70, 62), (50, 52)), P((50, 52), (48, 40))]))
        s.append(P((22, 72), (22, 92), (30, 92)))
        s.append(P((38, 76), (38, 92), (46, 92)))
        s.append(P((44, 46), (50, 50), (52, 46)))
        if maybe(rng, 0.7):
            s.append(ell(40, 18, 1.5, 1.5, rng, 6))
        if maybe(rng, 0.5):
            s.append(zig(44, 58, 23, 3, 3))
    return s


def zombie(rng):
    s = []
    if maybe(rng, 0.6):  # full body, arms forward
        s.append(ell(40, 20, 11, 12, rng))
        s += eye_pair(40, 18, 4.5, 2.2, rng, style=rng.choice(["x", "dot", "circle"]))
        s.append(zig(34, 46, 26, 2.5, 3) if maybe(rng, 0.6) else P((34, 26), (46, 27)))
        s.append(P((40, 32), (42, 66)))
        s.append(P((41, 40), (78, 40)))
        s.append(P((41, 46), (76, 47)))
        s.append(P((42, 66), (32, 92)))
        s.append(P((42, 66), (54, 92)))
        if maybe(rng, 0.5):  # torn shirt
            s.append(zig(30, 52, 60, 4, 4))
        if maybe(rng, 0.4):
            s.append(P((36, 8), (38, 12), (40, 8), (42, 12)))
    else:  # head close-up
        s.append(np.concatenate([qbez((22, 40), (24, 6), (52, 8)), qbez((52, 8), (80, 8), (78, 42)),
                                 qbez((78, 42), (76, 84), (50, 88)), qbez((50, 88), (24, 86), (22, 40))]))
        s += eye_pair(50, 38, 14, 6, rng, style=rng.choice(["x", "circle"]))
        s.append(P((34, 66), (66, 66)))
        for x in np.linspace(38, 62, int(rng.integers(3, 6))):
            s.append(P((x, 62), (x, 70)))
        if maybe(rng, 0.7):  # stitches
            s.append(P((28, 20), (44, 26)))
            for x in (32, 36, 40):
                s.append(P((x, 19 + (x - 28) * 0.37), (x + 1, 25 + (x - 28) * 0.37)))
    return s


def ghost(rng):
    w = rng.uniform(26, 32)
    top, bot = 10, rng.uniform(80, 88)
    head = arc(50, top + w, w, w, math.pi, TAU, 20)
    k = int(rng.integers(3, 6))
    xs = np.linspace(50 + w, 50 - w, 2 * k + 1)
    ys = [bot if i % 2 == 0 else bot - rng.uniform(6, 12) for i in range(2 * k + 1)]
    hem = np.stack([xs, ys], 1) if maybe(rng, 0.5) else wave(50 + w, 50 - w, bot - 4, 4, k)
    s = [np.concatenate([head, P((50 + w, top + w), (50 + w, bot)), hem, P((50 - w, bot), (50 - w, top + w))])]
    ey = top + w * rng.uniform(0.8, 1.1)
    s.append(ell(50 - w * 0.35, ey, 4, 6, rng, 12))
    s.append(ell(50 + w * 0.35, ey, 4, 6, rng, 12))
    if maybe(rng, 0.6):
        s.append(ell(50, ey + 14, 5, 6, rng, 12))
    if maybe(rng, 0.35):
        s.append(qbez((50 - w, top + w + 10), (50 - w - 14, top + w + 4), (50 - w - 16, top + w + 18)))
        s.append(qbez((50 + w, top + w + 10), (50 + w + 14, top + w + 4), (50 + w + 16, top + w + 18)))
    return s


def robot(rng):
    s = []
    hw, hh = rng.uniform(13, 18), rng.uniform(12, 16)
    s.append(rect(50 - hw, 14, 2 * hw, 2 * hh))
    if maybe(rng, 0.75):
        s.append(P((50, 14), (50, 4)))
        s.append(ell(50, 3, 2.5, 2.5, rng, 8))
    sq = maybe(rng, 0.5)
    for x in (50 - hw * 0.45, 50 + hw * 0.45):
        s.append(rect(x - 3, 14 + hh * 0.6, 6, 6) if sq else ell(x, 14 + hh * 0.75, 3.5, 3.5, rng, 10))
    s.append(rect(50 - hw * 0.5, 14 + hh * 1.4, hw, 4) if maybe(rng, 0.5) else P((50 - hw * 0.5, 14 + hh * 1.5), (50 + hw * 0.5, 14 + hh * 1.5)))
    by = 14 + 2 * hh + 3
    bw, bh = rng.uniform(18, 24), rng.uniform(28, 34)
    s.append(P((50 - 4, 14 + 2 * hh), (50 - 4, by)))
    s.append(rect(50 - bw, by, 2 * bw, bh))
    if maybe(rng, 0.6):
        s.append(rect(50 - bw * 0.5, by + 6, bw, bh * 0.4))
    s.append(P((50 - bw, by + 5), (50 - bw - 14, by + 22)))
    s.append(P((50 + bw, by + 5), (50 + bw + 14, by + 22)))
    if maybe(rng, 0.5):
        s.append(ell(50 - bw - 15, by + 25, 3, 3, rng, 8))
        s.append(ell(50 + bw + 15, by + 25, 3, 3, rng, 8))
    for x in (50 - bw * 0.5, 50 + bw * 0.5):
        s.append(rect(x - 4, by + bh, 8, 16) if maybe(rng, 0.5) else P((x, by + bh), (x, by + bh + 16)))
    return s


def alien(rng):
    s = []
    hw = rng.uniform(22, 28)
    top = 8
    head = np.concatenate([arc(50, top + hw, hw, hw, math.pi, TAU, 20),
                           qbez((50 + hw, top + hw), (50 + hw * 0.8, top + hw * 1.9), (50, top + hw * 2.3)),
                           qbez((50, top + hw * 2.3), (50 - hw * 0.8, top + hw * 1.9), (50 - hw, top + hw))])
    s.append(head)
    ey = top + hw * 1.25
    for side in (-1, 1):
        cx = 50 + side * hw * 0.42
        ang = side * 0.5
        e = arc(0, 0, hw * 0.32, hw * 0.16, 0, TAU, 16)
        c, sn = math.cos(ang), math.sin(ang)
        s.append(e @ np.array([[c, sn], [-sn, c]]) + (cx, ey))
    if maybe(rng, 0.5):
        s.append(P((46, top + hw * 1.85), (54, top + hw * 1.85)))
    if maybe(rng, 0.5):
        for side in (-1, 1):
            s.append(qbez((50 + side * 8, top + 2), (50 + side * 14, top - 8), (50 + side * 20, top - 10)))
            s.append(ell(50 + side * 21, top - 11, 2.5, 2.5, rng, 8))
    if maybe(rng, 0.5):  # small body
        by = top + hw * 2.3
        s.append(P((44, by), (40, by + 26), (60, by + 26), (56, by)))
        s.append(P((42, by + 8), (28, by + 18)))
        s.append(P((58, by + 8), (72, by + 18)))
        s.append(P((45, by + 26), (44, by + 38)))
        s.append(P((55, by + 26), (56, by + 38)))
    return s


def poop(rng):
    s = []
    tiers = int(rng.choice([2, 3, 3, 3, 4]))
    y = 88
    w = rng.uniform(36, 42)
    pts = []
    for i in range(tiers):
        h = rng.uniform(13, 17)
        pts.append(arc(50, y - h / 2, w, h / 2, math.pi / 2, math.pi * 1.5, 10))
        y -= h
        w *= rng.uniform(0.65, 0.78)
    tip = P((50 - w * 0.5, y + 4), (50 + rng.uniform(-6, 6), y - 10))
    # the left side climbs the tiers to the tip; the right side mirrors it back down
    left = np.concatenate(pts + [tip])
    right = left[::-1].copy()
    right[:, 0] = 100 - right[:, 0]
    s.append(np.concatenate([left, right[1:]]))
    s.append(P((10, 88), (90, 88)))
    if maybe(rng, 0.55):  # stink lines
        for x in (30, 50, 70):
            if maybe(rng, 0.8):
                s.append(np.stack([x + 3 * np.sin(np.linspace(0, TAU * 1.2, 14)), np.linspace(y - 12, y - 32, 14)], 1))
    if maybe(rng, 0.4):
        s += eye_pair(50, 66, 9, 3, rng, style="circle")
        s.append(arc(50, 72, 8, 5, 0.2, math.pi - 0.2, 10))
    return s


def heart(rng):
    v = rng.choice(["plain", "plain", "plain", "arrow", "double"])
    w, h = rng.uniform(36, 44), rng.uniform(36, 44)
    cy = 46
    hrt = np.concatenate([bez((50, cy - h * 0.45), (50 - w * 0.2, cy - h * 1.15), (50 - w * 1.25, cy - h * 0.6), (50, cy + h)),
                          bez((50, cy + h), (50 + w * 1.25, cy - h * 0.6), (50 + w * 0.2, cy - h * 1.15), (50, cy - h * 0.45))])
    s = [hrt]
    if v == "arrow":
        s.append(P((8, 80), (92, 16)))
        s.append(P((84, 16), (92, 16), (90, 25)))
        s.append(P((8, 80), (4, 72)))
        s.append(P((8, 80), (16, 82)))
    elif v == "double":
        s.append(hrt * 0.55 + np.array([50 * 0.45 + 20, 50 * 0.45 - 18]))
    return s


def taco(rng):
    """Like the taco emoji: a deep half-moon shell with spots, the folded edge as a line, and
    scalloped lettuce bulging out of it. Tilted in most drawings (see MAX_ROT), so it doesn't read as a bowl."""
    s = []
    r = rng.uniform(36, 42)
    top = 38
    shell = arc(50, top, r, r * rng.uniform(0.95, 1.08), 0.0, math.pi, 26)
    s.append(np.concatenate([shell, P((50 - r, top), (50 + r, top))]) if maybe(rng, 0.6) else shell)
    k = int(rng.integers(4, 8))
    w = 2 * r / k
    h = rng.uniform(5, 9)
    lettuce = np.concatenate([arc(50 - r + w * (i + 0.5), top, w / 2, h, math.pi, TAU, 8) for i in range(k)])
    s.append(lettuce)
    if maybe(rng, 0.5):  # inner edge of the shell
        s.append(arc(50, top, r * 0.82, r * 0.82, 0.25, math.pi - 0.25, 18))
    if maybe(rng, 0.65):  # spots on the shell
        for _ in range(int(rng.integers(3, 7))):
            a = rng.uniform(0.3, math.pi - 0.3)
            d = rng.uniform(0.4, 0.85) * r
            x, y = 50 + d * math.cos(a), top + d * math.sin(a)
            s.append(P((x, y), (x + 2, y + 1)))
    if maybe(rng, 0.4):  # tomato bits
        for _ in range(int(rng.integers(2, 4))):
            x = rng.uniform(50 - r * 0.6, 50 + r * 0.6)
            s.append(rect(x, top - h - 5, 4, 4))
    return s


def cheese(rng):
    s = []
    if maybe(rng, 0.75):  # wedge
        a, b, c = (14, 44), (86, 30), (60, 18)
        s.append(P(a, b, c, a))
        s.append(P(a, (14, 78), (86, 66), b))
        if maybe(rng, 0.4):
            s.append(P(c, (60, 24)))
    else:
        s.append(rect(14, 30, 72, 44))
        s.append(P((14, 30), (26, 18), (96, 18), (86, 30)))
        s.append(P((96, 18), (96, 62), (86, 74)))
    for _ in range(int(rng.integers(2, 6))):
        r = rng.uniform(3, 7)
        s.append(ell(rng.uniform(24, 76), rng.uniform(50, 68), r, r * rng.uniform(0.8, 1.1), rng, 12))
    return s


def straw_hat(rng):
    s = []
    bw, bh = rng.uniform(42, 48), rng.uniform(9, 14)
    cy = 62
    s.append(ell(50, cy, bw, bh, rng))
    cw = bw * rng.uniform(0.45, 0.6)
    ch = rng.uniform(22, 30)
    s.append(np.concatenate([qbez((50 - cw, cy - 2), (50 - cw, cy - ch * 1.25), (50, cy - ch)), qbez((50, cy - ch), (50 + cw, cy - ch * 1.25), (50 + cw, cy - 2))]))
    if maybe(rng, 0.85):  # band
        s.append(qbez((50 - cw, cy - 7), (50, cy - 3), (50 + cw, cy - 7)))
        s.append(qbez((50 - cw, cy - 12), (50, cy - 8), (50 + cw, cy - 12)))
    if maybe(rng, 0.35):
        for x in np.linspace(50 - bw + 8, 50 + bw - 8, 6):
            s.append(P((x, cy + bh * 0.3), (x + 3, cy + bh * 0.8)))
    return s


def kunai(rng):
    s = []
    L = rng.uniform(80, 92)
    x0 = 50
    tip, bw = 6, rng.uniform(8, 11)
    blade_end = tip + L * 0.5
    s.append(P((x0, tip), (x0 + bw, blade_end - 10), (x0 + 3, blade_end), (x0 - 3, blade_end), (x0 - bw, blade_end - 10), (x0, tip)))
    if maybe(rng, 0.6):
        s.append(P((x0, tip + 6), (x0, blade_end - 4)))
    h_end = blade_end + L * 0.32
    s.append(rect(x0 - 3, blade_end, 6, h_end - blade_end))
    if maybe(rng, 0.7):
        for y in np.linspace(blade_end + 4, h_end - 4, int(rng.integers(3, 6))):
            s.append(P((x0 - 3, y), (x0 + 3, y + 4)))
    s.append(ell(x0, h_end + 7, 7, 7, rng, 14))
    return s


def spiral_cls(rng):
    turns = rng.uniform(2.0, 4.0)
    return [spiral(50, 50, rng.uniform(40, 46), turns, cw=bool(rng.integers(2)), r0=rng.uniform(0, 3))]


def cursed_face(rng):
    s = [ell(50, 50, rng.uniform(34, 38), rng.uniform(40, 44), rng)]
    style = rng.choice(["circle", "dot", "slit"])
    for ey, sc in ((36, 1.0), (50, 0.8)):
        for side in (-1, 1):
            x = 50 + side * 15
            if style == "slit":
                s.append(np.concatenate([qbez((x - 6 * sc, ey), (x, ey - 4 * sc), (x + 6 * sc, ey)), qbez((x + 6 * sc, ey), (x, ey + 4 * sc), (x - 6 * sc, ey))]))
            else:
                s.append(ell(x, ey, 5 * sc, 4 * sc, rng, 12))
                if style == "dot":
                    s.append(ell(x, ey, 1.2, 1.2, rng, 6))
    if maybe(rng, 0.8):  # marks under the eyes / on cheeks
        for side in (-1, 1):
            x = 50 + side * 28
            s.append(P((x, 56), (x - side * 8, 62)))
            if maybe(rng, 0.6):
                s.append(P((x, 62), (x - side * 8, 68)))
    if maybe(rng, 0.5):
        s.append(P((44, 14), (50, 22), (56, 14)))
    if maybe(rng, 0.6):
        s.append(zig(38, 62, 74, 4, 4))
    else:
        s.append(qbez((36, 72), (50, 80), (64, 72)))
    return s


def x2(rng):
    s = []
    order = rng.choice(["x2", "x2", "x2", "2x"])
    xs, ts = (12, 54) if order == "x2" else (58, 10)
    xw = rng.uniform(26, 32)
    s.append(P((xs, 34), (xs + xw, 78)))
    s.append(P((xs + xw, 34), (xs, 78)))
    t = ts
    s.append(np.concatenate([arc(t + 18, 34, 16, 14, math.pi * 1.05, math.pi * 2.25, 18), P((t + 30, 46), (t + 2, 80), (t + 36, 80))]))
    return s


def undo_arrow(rng):
    s = []
    r = rng.uniform(26, 34)
    a0 = rng.uniform(-0.2, 0.4)
    a1 = math.pi + rng.uniform(0.0, 0.5)
    body = arc(52, 56, r, r * rng.uniform(0.75, 1.0), a0, -a1, 22)  # counter-clockwise over the top
    s.append(body)
    end = body[-1]
    d = body[-1] - body[-3]
    d = d / (np.linalg.norm(d) + 1e-9)
    nrm = np.array([-d[1], d[0]])
    hl = rng.uniform(15, 20)
    head = P(end - d * hl + nrm * hl * 0.75, end, end - d * hl - nrm * hl * 0.75)
    s.append(head)
    if maybe(rng, 0.3):
        s[0] = np.concatenate([body, head])
        s.pop()
    return s


THINGS = {
    "bomb": bomb, "nuke": nuke, "meteor": meteor, "volcano": volcano, "tsunami": tsunami, "black hole": black_hole,
    "pistol": pistol, "tank": tank, "rocket": rocket, "grenade": grenade, "dynamite": dynamite, "ninja star": ninja_star,
    "bow and arrow": bow_and_arrow, "shield": shield, "dinosaur": dinosaur, "zombie": zombie, "ghost": ghost, "robot": robot,
    "alien": alien, "poop": poop, "heart": heart, "taco": taco, "cheese": cheese, "straw hat": straw_hat, "kunai": kunai,
    "spiral": spiral_cls, "cursed face": cursed_face, "x2": x2, "undo arrow": undo_arrow,
}
NO_MIRROR = {"x2", "undo arrow"}
MAX_ROT = {"rocket": 0.8, "kunai": 0.9, "ninja star": 0.8, "spiral": math.pi, "black hole": math.pi, "meteor": 0.5,
           "pistol": 0.35, "bow and arrow": 0.4, "taco": 0.55}


# ---------------------------------------------------------------- humanising
def resample(p, step):
    if len(p) < 2:
        return p
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    if d[-1] < 1e-6:
        return p[:1]
    n = max(2, int(d[-1] / step) + 1)
    t = np.linspace(0, d[-1], n)
    return np.stack([np.interp(t, d, p[:, 0]), np.interp(t, d, p[:, 1])], 1)


def humanise(strokes, rng, name, mess):
    out = []
    for s in strokes:
        s = np.asarray(s, float)
        if len(s) < 2:
            continue
        s = resample(s, 1.0)
        closed = len(s) > 8 and np.hypot(*(s[0] - s[-1])) < 2.0
        n = len(s)
        if closed and maybe(rng, 0.8):  # start somewhere else, close sloppily
            k = int(rng.integers(0, n))
            s = np.concatenate([s[k:-1], s[:k + 1]])
            extra = int(rng.integers(-6, 8) * mess)
            if extra > 0:
                s = np.concatenate([s, s[1:extra + 1]])
            elif extra < 0 and len(s) > 12:
                s = s[:extra]
        n = len(s)
        t = np.linspace(0, 1, n)
        length = max(1.0, n)
        noise = np.zeros_like(s)
        # smooth, low-frequency hand drift: a few sine components per axis, gentler on short strokes
        for k in range(1, 4):
            freq = rng.uniform(0.4, 1.1) * k * (1 + length / 250)
            for axis in (0, 1):
                amp = rng.normal(0, 1.3 * mess / k) * min(1.0, length / 30)
                noise[:, axis] += amp * np.sin(TAU * freq * t + rng.uniform(0, TAU))
        s = s + noise + rng.normal(0, 0.12 * mess, s.shape)
        if not closed and n > 4 and maybe(rng, 0.5):  # overshoot or stop short at the ends
            d = s[-1] - s[-2]
            s = np.concatenate([s, [s[-1] + d * rng.uniform(0, 4)]]) if maybe(rng, 0.5) else s[: max(2, n - int(rng.integers(0, 3)))]
        out.append(s)
    if maybe(rng, 0.12 * mess) and len(out) > 3:  # forget one small detail (never the first or a big stroke)
        lens = [len(p) for p in out]
        small = [i for i in range(1, len(out)) if lens[i] < 0.15 * sum(lens)
                 and not (name == "bomb" and i in (1, 2))]  # keep its cap and fuse
        if small:
            out.pop(int(rng.choice(small)))
    # whole-drawing distortion
    rot = rng.uniform(-1, 1) * MAX_ROT.get(name, 0.18)
    sx, sy = rng.uniform(0.82, 1.18, 2)
    sh = rng.uniform(-0.12, 0.12)
    c, sn = math.cos(rot), math.sin(rot)
    m = np.array([[c, -sn], [sn, c]]) @ np.array([[sx, sh], [0, sy]])
    if name not in NO_MIRROR and maybe(rng, 0.5):
        m = m @ np.array([[-1, 0], [0, 1]])
    return [p @ m.T for p in out]


# ---------------------------------------------------------------- Quick, Draw! simplification
def rdp(p, eps):
    if len(p) < 3:
        return p
    a, b = p[0], p[-1]
    ab = b - a
    nrm = np.hypot(*ab)
    if nrm < 1e-9:
        d = np.hypot(*(p - a).T)
    else:
        d = np.abs(ab[0] * (p[:, 1] - a[1]) - ab[1] * (p[:, 0] - a[0])) / nrm
    i = int(np.argmax(d))
    if d[i] <= eps:
        return np.array([a, b])
    return np.concatenate([rdp(p[: i + 1], eps)[:-1], rdp(p[i:], eps)])


def to_quickdraw(strokes):
    allp = np.concatenate(strokes)
    mn = allp.min(0)
    k = 255.0 / max(1e-6, (allp.max(0) - mn).max())
    out = []
    for s in strokes:
        q = rdp(resample((s - mn) * k, 1.0), 2.0)
        q = np.clip(np.round(q), 0, 255).astype(int)
        keep = np.r_[True, np.any(np.diff(q, axis=0) != 0, axis=1)]
        q = q[keep]
        if len(q) == 1:
            q = np.concatenate([q, q])
        out.append([q[:, 0].tolist(), q[:, 1].tolist()])
    return out


def make(name, rng):
    mess = rng.uniform(0.4, 1.3)
    return to_quickdraw(humanise(THINGS[name](rng), rng, name, mess))


def write_class(job):
    """One thing's training file; its own seed, so classes can run in parallel and stay reproducible."""
    ci, name, n, seed, folder = job
    rng = np.random.default_rng([seed, ci])
    path = os.path.join(folder, f"{name}.ndjson")
    with open(path + ".part", "w", encoding="utf-8", newline="\n") as f:
        for i in range(n):
            row = {"word": name, "key_id": f"synth{seed}-{ci:02d}-{i:06d}", "recognized": True, "drawing": make(name, rng)}
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    os.replace(path + ".part", path)
    return name, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", help="one JSON list with ids, for review")
    ap.add_argument("--ndjson", help="folder for <thing>.ndjson training files (generated in parallel)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--only", help="comma-separated things to (re)generate; others are left as they are")
    a = ap.parse_args()
    if a.ndjson and not a.out:
        from multiprocessing import Pool
        os.makedirs(a.ndjson, exist_ok=True)
        only = set(a.only.split(",")) if a.only else None
        jobs = [(ci, name, a.per_class, a.seed, a.ndjson) for ci, name in enumerate(THINGS) if not only or name in only]
        with Pool(a.workers) as pool:
            for name, n in pool.imap_unordered(write_class, jobs):
                print(f"{name:14s} {n}", flush=True)
        return
    rng = np.random.default_rng(a.seed)
    rows = []
    for name in THINGS:
        for i in range(a.per_class):
            rows.append({"id": f"s{a.seed}-{name.replace(' ', '_')}-{i}", "cls": name, "drawing": make(name, rng)})
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rows, f, separators=(",", ":"))
    if a.ndjson:
        os.makedirs(a.ndjson, exist_ok=True)
        for name in THINGS:
            with open(os.path.join(a.ndjson, f"{name}.ndjson"), "w", encoding="utf-8", newline="\n") as f:
                for r in rows:
                    if r["cls"] == name:
                        f.write(json.dumps({"word": name, "key_id": r["id"], "recognized": True, "drawing": r["drawing"]},
                                           separators=(",", ":")) + "\n")
    print(len(rows), "drawings")


if __name__ == "__main__":
    main()
