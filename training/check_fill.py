"""Check cached CPU/GPU region equivalence and the compiled fill-colour policy."""
import numpy as np
import torch
from numba import set_num_threads
from categories import CATEGORIES
from colour import colourise,NET
from colour_batch import sample_fills
from colour_raster import render
from colour_probes import cases
from fill import inside,fill_planes,fill_colour,STEPS,INK_T
from prepare_fill import regions

if __name__=='__main__':
    set_num_threads(4);torch.set_num_threads(4)
    for size in (28,40):
        a=[]
        for c in cases():
            a.append(render([[[p[0] for p in s],[p[1] for p in s]] for s in c['strokes']],size)[0])
        a=np.stack(a);cpu=regions(a,max(STEPS,round(STEPS*size/28)),INK_T*255)
        ink=torch.from_numpy(a).float()/255
        assert np.array_equal(cpu,inside(ink).numpy())
        x=torch.zeros(len(a),4,size,size);x[:,0]=ink
        gen=lambda:torch.Generator().manual_seed(73)
        rgb=torch.from_numpy(NET[[3]*len(a)]);on=torch.ones(len(a),dtype=torch.bool)
        one=fill_planes(x.clone(),rgb,on,generator=gen())
        two=fill_planes(x.clone(),rgb,on,generator=gen(),region=torch.from_numpy(cpu))
        assert torch.equal(one,two)
        assert torch.equal(fill_planes(x.clone(),rgb,~on),x)
        assert torch.all(one[:,0]>=x[:,0]) and torch.all(one[:,1:]<=one[:,:1])
    for name in ('apple','bomb','rainbow','heart','black hole'):
        n=20000;ci=CATEGORIES.index(name);ink=np.array([10.,3.,1.]);rng=np.random.default_rng(54)
        cs=np.array([colourise(name,ink,rng) for _ in range(n)],np.uint8)
        ref=np.array([fill_colour(name,c,ink,rng) for c in cs]);offset=np.arange(n+1,dtype=np.int64)*3
        compiled=sample_fills(np.full(n,ci,np.int64),offset,np.tile(ink,n),cs.ravel(),73)
        assert np.max(np.abs(np.bincount(ref,minlength=10)/n-np.bincount(compiled,minlength=10)/n))<.025
    print('28/40 cached regions, deterministic fills, ink preservation and reference fill-colour distributions agree.')
