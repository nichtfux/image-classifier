"""Training-only paint/brush variation. Never used by the serving rasterizer.

Operates on our recognized Quick, Draw! inputs or our procedural drawings only.
Keeps a substantial untouched group, rather than making colour/fill compulsory.
"""
import torch
import torch.nn.functional as F


def augment(x, region, fill_rgb, strength=1., generator=None, preserve_details=False):
    original = x if preserve_details else None
    b, _, h, w = x.shape
    dev = x.device
    def rand(*shape):
        return torch.rand(shape, device=dev, generator=generator)
    style = rand(b)
    # 35% filled: solid, incomplete painted, parallel scribble and cross-hatch.
    selected = style < .35 * strength
    # Paint can cover interior detail strokes too, while retaining the silhouette.
    area = region | (x[:,0] > .35)
    interior = -F.max_pool2d(-area.float()[:,None],3,1,1)[:,0] > .5
    overpaint = rand(b) < .25
    r = (region | (interior & overpaint[:,None,None])) & selected[:, None, None]
    yy, xx = torch.meshgrid(torch.arange(h, device=dev), torch.arange(w, device=dev), indexing='ij')
    angle = rand(b) * 6.2831853
    u = xx[None] * angle.cos()[:, None, None] + yy[None] * angle.sin()[:, None, None]
    spacing = 2. + rand(b) * 8.
    phase = rand(b) * spacing
    stripe = torch.remainder(u + phase[:, None, None], spacing[:, None, None])
    # Real scribbles leave long gaps; pixelwise random holes alone cannot model them.
    keep = stripe < (spacing * (.25 + rand(b) * .65))[:, None, None]
    cross = torch.remainder(-xx[None] * angle.sin()[:, None, None] + yy[None] * angle.cos()[:, None, None], spacing[:, None, None]) < 1.5
    kind = rand(b)
    keep = torch.where((kind < .38)[:, None, None], torch.ones_like(keep), keep)
    keep |= (kind > .80)[:, None, None] & cross
    lazy = rand(b) < .25
    eroded = 1 - F.max_pool2d((~r).float()[:, None], 3, 1, 1)[:, 0]
    r &= torch.where(lazy[:, None, None], eroded > .5, torch.ones_like(r)) & keep
    paint = torch.cat([torch.ones((b, 1), device=dev), fill_rgb], 1)[:, :, None, None]
    # Some fills sit underneath details, others paint over existing interior lines.
    under = rand(b) < .7
    r &= (~under[:, None, None]) | (x[:, 0] < .4)
    x = torch.where(r[:, None], paint, x)
    # Some people paint the silhouette directly, without tracing an outline first.
    # Remove only the exterior contour on a small filled subset; interior parts stay.
    substantial = region.sum((1,2)) > area.sum((1,2)) * .50
    direct = selected & substantial & (rand(b) < .20)
    border = area & ~interior
    x = torch.where((direct[:,None,None] & border)[:,None], torch.zeros_like(x), x)

    if preserve_details:
        # Apply before the shared width/affine transform. Overlaying the original
        # afterwards would create doubled, differently rotated outlines.
        x = torch.where((original[:, 0] > .35)[:, None], original, x)

    # Variable width, including thin careful outlines and a broad soft brush.
    width = rand(b)
    thick = F.max_pool2d(x, 3, 1, 1)
    broad = F.max_pool2d(x, 5, 1, 2)
    thin = -F.max_pool2d(-x, 3, 1, 1)
    x = torch.where((width < .12 * strength)[:, None, None, None], .5*x+.5*thin, x)
    x = torch.where(((width >= .12) & (width < .29 * strength))[:, None, None, None], thick, x)
    x = torch.where(((width >= .29) & (width < .36 * strength))[:, None, None, None], broad, x)
    # Soft paint edges; all four channels share geometry.
    blurred = F.avg_pool2d(x, 3, 1, 1)
    x = torch.where((rand(b) < .08 * strength)[:, None, None, None], .65*x+.35*blurred, x)
    # Small device-independent coordinate/style changes; no reflection of text/arrows.
    theta = torch.zeros((b, 2, 3), device=dev)
    a = (rand(b)-.5)*.36*strength
    sx, sy = 1+(rand(b)-.5)*.15*strength, 1+(rand(b)-.5)*.15*strength
    theta[:, 0, 0] = a.cos()*sx; theta[:, 0, 1] = -a.sin()*sy
    theta[:, 1, 0] = a.sin()*sx; theta[:, 1, 1] = a.cos()*sy
    theta[:, :, 2] = (rand(b, 2)-.5)*.06*strength
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), align_corners=False)
    return x.contiguous(memory_format=torch.channels_last)
