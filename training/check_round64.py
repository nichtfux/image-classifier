"""CPU smoke checks for the queued resolution change. Uses training rows only."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from colour import planes
from colour_raster import render, reference
from gacl import QualityNet
from long_run_data64 import prepare_index, INDEX, ROOT, Data, inside64, raw_path
from round64_model import initialize
from train_next import load


def main():
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    prepare_index([0,290])
    drawings=[]
    for label in [0,290]:
        with np.load(INDEX/f'{label:03d}.npz') as z:
            offset=int(z['train'][0])
        with raw_path(label).open('rb') as handle:
            handle.seek(offset);drawing=json.loads(handle.readline())['drawing']
        drawings.append(drawing)
        ink,owner,length=render(drawing,64)
        exact,exact_owner=reference(drawing,64)
        assert np.max(np.abs(ink.astype(np.float64)/255-exact))<=.5/255+1e-10
        assert np.array_equal(owner[ink>0],exact_owner[ink>0])
    ink40,owner40,_=render(drawings[1],40)
    small=np.rint(planes(ink40/255,owner40,[0]*len(drawings[1]))*255).astype(np.uint8)
    ink64,owner64,_=render(drawings[1],64)
    large=np.rint(planes(ink64/255,owner64,[0]*len(drawings[1]))*255).astype(np.uint8)
    data=Data.__new__(Data)
    data.corpus=SimpleNamespace(y=np.arange(299,dtype=np.int64))
    data.custom=[np.asarray([i]) for i in range(270,299)]
    data.source=SimpleNamespace(offsets=np.arange(299),read=lambda label,offset:drawings[label>=270])
    data.proc={'x':np.stack([small]*29),'y':np.arange(270,299),
               'region':np.ones((29,40,40),bool),'fill':np.full(29,3,np.uint8)}
    data.proc_by_class=[np.asarray([i]) for i in range(29)]
    data.game={'x':np.stack([small]*4),'y':np.arange(4)}
    generated={'x':np.stack([large]*8),'y':np.r_[np.arange(4),np.arange(270,274)],
               'teacher':np.zeros((8,299),np.float32)}
    data.bank=SimpleNamespace(data=generated,real=np.arange(4),
              custom_by_class=[np.asarray([i+4]) if i<4 else np.asarray([],np.int64) for i in range(29)])
    as_tensor=torch.as_tensor
    zeros=torch.zeros
    def cpu_tensor(*args,**kwargs):
        if kwargs.get('device')=='cuda':kwargs['device']='cpu'
        return as_tensor(*args,**kwargs)
    def cpu_zeros(*args,**kwargs):
        if kwargs.get('device')=='cuda':kwargs['device']='cpu'
        return zeros(*args,**kwargs)
    with patch.object(torch.Tensor,'cuda',lambda self,*args,**kwargs:self), \
         patch('torch.as_tensor',cpu_tensor),patch('torch.zeros',cpu_zeros):
        x,y,kd_start,soft=data.batch(np.random.default_rng(20261111),16)
    assert x.shape==(16,4,64,64) and y.shape==(16,)
    assert soft.shape==(2,299) and kd_start==14
    assert torch.isfinite(x).all() and x.min()>=0 and x.max()<=1
    pointer=json.loads((ROOT/'training/out/long299_40_v4/best-so-far.json').read_text(encoding='utf-8'))
    teacher,_=load(Path(pointer['path'])/'checkpoint.pt','cpu')
    student=initialize(teacher)
    assert student.f1.weight.shape==(768,12288)
    optimizer=torch.optim.AdamW(student.parameters(),lr=1e-5)
    output=student(x[:2])
    assert output.shape==(2,299) and torch.isfinite(output).all()
    loss=F.cross_entropy(output,y[:2]);loss.backward();optimizer.step()
    target=ROOT/'training/out/round64/cpu-smoke-checkpoint.pt'
    target.parent.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(20261111)
    torch.save({'model':student.state_dict(),'optimizer':optimizer.state_dict(),
                'rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state()},target)
    restored=torch.load(target,map_location='cpu',weights_only=False)
    assert restored['rng']==rng.bit_generator.state
    assert len(restored['optimizer']['state'])==len(optimizer.state)
    for key,value in student.state_dict().items():assert torch.equal(value,restored['model'][key])
    target.unlink()
    assert not torch.cuda.is_initialized()
    result={'size':64,'widths':[48,96,192],'fc':768,'batch_shape':list(x.shape),
            'raster_reference':'passed','real_and_custom_index':'passed',
            'teacher_transfer':'passed','optimizer_rng_checkpoint':'passed','cuda_initialized':False}
    (ROOT/'training/out/round64/cpu-checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
