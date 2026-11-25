"""Generate an uninstalled runtime copy for cosine-head candidates.

The installed luau/DrawRecognizer.luau remains untouched. Existing first four
recognize returns and the image layout stay identical. Fifth return = raw norm;
beginRecognize results add qualityNorm. No Rough/Good/Detailed tiers are assigned.
"""
from pathlib import Path


def build(out_dir):
    root=Path(__file__).resolve().parent.parent
    s=(root/'luau/DrawRecognizer.luau').read_text(encoding='utf-8')
    s=s.replace('M.modelName = Weights.name or "legacy"',
        'M.modelName = Weights.name or "legacy"\nM.qualityNormScale = Weights.qualityNormScale')
    s=s.replace('kind = kind,\n', 'kind = kind,\n\t\t\tnormalizeInput = spec.normalizeInput or false,\n')
    s=s.replace('local n = #layers\n\tfor li, layer', 'local n = #layers\n\tlocal qualityNorm = nil\n\tfor li, layer')
    s=s.replace('x = denseLayer(x, layer, li < n, checkpoint)', '''if layer.normalizeInput then
                local sum = 0
                for i = 1, layer.nper do sum += x[i] * x[i] end
                qualityNorm = math.sqrt(sum)
                local inv = 1 / math.max(qualityNorm, 1e-8)
                for i = 1, layer.nper do
                    x[i] *= inv
                    if checkpoint and i % 256 == 0 then checkpoint() end
                end
            end
            x = denseLayer(x, layer, li < n, checkpoint)''')
    s=s.replace('\treturn z\nend', '\treturn z, qualityNorm\nend')
    s=s.replace('local z = logits(image, workspaceAcc, checkpoint)', 'local z, qualityNorm = logits(image, workspaceAcc, checkpoint)')
    s=s.replace('\treturn p\nend', '\treturn p, qualityNorm\nend')
    s=s.replace('local p = forward(image, table.create(SIZE * SIZE, Vector3.zero), checkpoint)',
                'local p, qualityNorm = forward(image, table.create(SIZE * SIZE, Vector3.zero), checkpoint)')
    s=s.replace('networkMs = t2 - t1, image = image }', 'networkMs = t2 - t1, image = image, qualityNorm = qualityNorm }')
    s=s.replace('local p = M.forward(image)', 'local p, qualityNorm = M.forward(image)')
    s=s.replace('return top, (t1 - t0) * 1000, (t2 - t1) * 1000, image',
                'return top, (t1 - t0) * 1000, (t2 - t1) * 1000, image, qualityNorm')
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    (out/'DrawRecognizer.luau').write_text(s,encoding='utf-8',newline='\n')
    for name in ['DrawRasterizer','DrawJob']:
        (out/(name+'.luau')).write_bytes((root/'luau'/(name+'.luau')).read_bytes())


if __name__=='__main__':
    import sys
    build(sys.argv[1])
