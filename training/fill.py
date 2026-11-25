"""Filled-in drawings: teach the network that a coloured-in shape is the same thing.

2026-11-09, after drawing an apple as a solid red blob that read as "beard":
"we need to train with filled out stuff". Quick, Draw! drawings are outlines, so
training fills some of them in, on the finished 28x28 planes:

1. Ink above INK_T is a wall. The walls are thickened by one pixel first, so an
   outline with a small gap still counts as closed.
2. Everything the paper outside can reach from the image border is outside;
   whatever it cannot reach is inside the drawing (lake fill, all closed regions).
3. Inside pixels become ink in the fill colour. Some fills are lazy: they stop one
   pixel short of the outline or leave a few holes, like a quick scribble.

An outline in the same colour as its fill gives a plain blob, which is what a
player gets who only scribbles the shape. Drawings with no closed outline stay as
they are. Works on a batch of 4-plane inputs (ink, ink*r, ink*g, ink*b) on any device.
"""
import torch
import torch.nn.functional as F

P_FILL = 0.25   # share of training drawings that get filled
P_LAZY = 0.30   # of those: fill stops one pixel short of the outline
P_HOLES = 0.30  # of those: a few pixels left out
HOLE_RATE = 0.05
P_OWN = 0.70    # fill colour = colour of the drawing's longest stroke (if not black), else a class main colour
P_BLACK_FILL = 0.25  # an all-black drawing filled in black (details then vanish, like a black scribble)
INK_T = 0.35
STEPS = 48      # flood steps; enough to reach every outside pixel of a 28x28 image


def fill_colour(cls, colours, ink, rng):
    """Palette index for the fill of one drawing (reference rule; training may port it).

    colours: palette index per stroke (from colour.colourise), ink: per-stroke ink
    lengths. Players who colour in usually pick a colour, and black detail lines on a
    coloured fill stay visible through the colour planes.
    """
    import numpy as np
    from colour import _PARSED, PALETTE
    own = colours[int(np.argmax(ink))]
    if own != 0 and rng.random() < P_OWN:
        return own
    if all(c == 0 for c in colours) and rng.random() < P_BLACK_FILL:
        return 0
    kind, v = _PARSED[cls][0]
    if kind == "cycle":
        return v[int(rng.integers(len(v)))]
    c = int(rng.choice(len(PALETTE), p=v))
    return c


def _grow(mask):
    return F.max_pool2d(mask.float().unsqueeze(1), 3, 1, 1).squeeze(1) > 0


def inside(ink):
    """(B,H,W) ink -> (B,H,W) bool: pixels enclosed by the drawing, ink excluded."""
    ink_mask = ink > INK_T
    wall = _grow(ink_mask)
    outside = torch.zeros_like(ink_mask)
    outside[:, 0, :] = outside[:, -1, :] = True
    outside[:, :, 0] = outside[:, :, -1] = True
    outside &= ~wall
    # Same one-pixel closure tolerance at both sizes. Rounding 40/28 still gives
    # one pixel; two would seal noticeably larger gaps. Flood travel scales up.
    for _ in range(max(STEPS, round(STEPS * ink.shape[-1] / 28))):
        outside = _grow(outside) & ~wall
    outside |= _grow(outside) & wall & ~ink_mask  # give back the thickening ring
    return ~outside & ~ink_mask


def fill_planes(x, fill_rgb, do_fill, generator=None, region=None):
    """Fill closed regions of chosen drawings in place.

    x: (B,4,H,W) float planes. fill_rgb: (B,3) net colour of each fill (black = 0).
    do_fill: (B,) bool, which drawings to fill (draw it with probability P_FILL).
    """
    if not bool(do_fill.any()):
        return x
    B = x.shape[0]
    dev = x.device
    if region is None:
        region = torch.zeros_like(x[:, 0], dtype=torch.bool)
        region[do_fill] = inside(x[do_fill, 0])
    region = region & do_fill[:, None, None]
    lazy = torch.rand(B, device=dev, generator=generator) < P_LAZY
    shrunk = region & ~_grow(~region)
    region = torch.where(lazy[:, None, None], shrunk, region)
    holes = torch.rand(B, device=dev, generator=generator) < P_HOLES
    keep = torch.rand(region.shape, device=dev, generator=generator) >= HOLE_RATE
    region &= ~holes[:, None, None] | keep
    r = region.unsqueeze(1)
    x[:, :1] = torch.where(r, torch.ones_like(x[:, :1]), x[:, :1])
    x[:, 1:] = torch.where(r, fill_rgb[:, :, None, None].to(x.dtype).expand(-1, -1, *x.shape[2:]), x[:, 1:])
    return x
