"""Independent ownership, ties, palette, recolouring and four-plane invariants."""
import json
from pathlib import Path
import numpy as np
import torch
import colour
from colour_raster import render,reference
from colour_batch import sample
from raster import Renderer
from categories import CATEGORIES
from train import Net
from train import quantise,fold
from train_colour import quantise_colour,quant_net
from colour_data import inputs


if __name__=='__main__':
    root=Path(__file__).resolve().parent.parent
    cases=[[[[0,100],[50,50]],[[0,100],[50,50]]],
           [[[0,100],[0,100]],[[0,100],[100,0]]],
           [[[],[]],[[3],[7]]],[[[0,0],[0,0]]]]
    cases += [r['drawing'] for r in json.loads((root/'data/expanded299-28/test_strokes.json').read_text())[::7]]
    rng=np.random.default_rng(64)
    for _ in range(50):
        cases.append([[rng.integers(-200,400,n).tolist(),rng.integers(-200,400,n).tolist()]
                      for n in rng.integers(1,18,int(rng.integers(1,6)))])
    for size in (28,40):
        for drawing in cases:
            ink,owner,length=render(drawing,size)
            ref,ro=reference(drawing,size)
            assert np.array_equal(ink,np.rint(ref*255).astype(np.uint8))
            assert np.array_equal(ink,np.rint(Renderer(size).render(drawing)*255).astype(np.uint8))
            assert np.array_equal(owner[ref>0],ro[ref>0])
            assert np.allclose(length,colour.stroke_ink(drawing),atol=1e-10)
            black=colour.planes(ref,ro,[0]*len(drawing))
            assert np.array_equal(black[0],ref) and not black[1:].any()
    _,tie,_=render(cases[0]);assert np.all(tie[14,4:24]==1),'later stroke did not win'
    labels=np.tile(np.arange(299,dtype=np.int64),400)
    offsets=np.arange(len(labels)+1,dtype=np.int64)*3
    lengths=np.tile(np.array([10.,3.,1.]),len(labels))
    a=sample(labels,offsets,lengths,3);b=sample(labels,offsets,lengths,4)
    assert np.array_equal(a,sample(labels,offsets,lengths,3)) and not np.array_equal(a,b)
    assert a.max()<10 and not sample(labels,offsets,lengths,2,2).any()
    # Exact starting function before fine-tuning, including the original 288-wide head.
    checkpoint=torch.load(root/'training/out/wide299_28_r4.pt',map_location='cpu',weights_only=True)
    old=Net([24,48,96],288,299).eval();old.load_state_dict(checkpoint['state_dict'])
    new=Net([24,48,96],288,299,in_channels=4).eval()
    state=new.state_dict()
    for k,v in checkpoint['state_dict'].items():
        if k=='c1.weight':state[k].zero_();state[k][:,:1].copy_(v)
        else:state[k].copy_(v)
    new.load_state_dict(state)
    x=torch.rand(3,1,28,28);x4=torch.cat([x,torch.zeros(3,3,28,28)],1)
    with torch.no_grad():assert torch.allclose(old(x),new(x4),atol=1e-5,rtol=1e-6)
    with torch.no_grad():new.c1.weight[:,1:].normal_(0,.4)
    oq,nq=quantise(fold(old)),quantise_colour(new)
    assert np.array_equal(nq[0][0][:,:1],oq[0][0]) and np.array_equal(nq[0][1][:,0],oq[0][1])
    for i in range(1,5):
        assert all(np.array_equal(a,b) for a,b in zip(oq[i],nq[i]))
    with torch.no_grad():assert torch.allclose(quant_net(oq,'cpu')(x),quant_net(nq,'cpu')(x4),atol=2e-5,rtol=1e-6)
    chunk=(torch.full((2,28,28),255,dtype=torch.uint8),torch.tensor([0,1]),
           torch.zeros((2,28,28),dtype=torch.int32),torch.tensor([0,1]),torch.tensor([0,3],dtype=torch.uint8))
    inp,_=inputs(chunk)
    assert inp.shape==(2,4,28,28) and inp.is_contiguous(memory_format=torch.channels_last)
    assert not inp[0,1:].any() and torch.allclose(inp[1,1:,0,0],torch.from_numpy(colour.NET[3]))
    print('Ownership/ink reference, crossing/overlap ties, empty strokes, black planes, sampler and zero-channel initialization passed.')
