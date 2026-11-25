"""64px model and CPU export preparation, shared by timing and training."""
import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

from candidate_runtime import build
from categories import CATEGORIES
from gacl import QualityNet
from long_run import qnpz, atomic

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'training/out'
WIDTHS = (48, 96, 192)
FC = 768
SIZE = 64


def initialize(teacher):
    """Transfer convolution filters and resize the dense spatial weight grid."""
    model = QualityNet(WIDTHS, FC, SIZE)
    state = teacher.state_dict()
    adapted = model.state_dict()
    for key in adapted:
        if key != 'f1.weight':
            adapted[key].copy_(state[key].detach().cpu())
    spatial = state['f1.weight'].detach().cpu().reshape(FC, WIDTHS[-1], 5, 5)
    adapted['f1.weight'].copy_((F.interpolate(spatial, size=(8,8), mode='bilinear',
                                      align_corners=False) * 25/64).reshape(FC,-1))
    model.load_state_dict(adapted)
    return model.eval()


def export_arrays(q, name, destination, vectors=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    qnpz(OUT / (name+'.npz'), q)
    meta = {'name':name, 'widths':list(WIDTHS), 'fc':FC, 'size':SIZE,
            'input_channels':4, 'params':sum(int(w.size+b.size) for w,s,b in q),
            'categories':CATEGORIES, 'corpus':'expanded299-28', 'feature_normalize':False,
            'int8_top1':0., 'int8_top3':0., 'installed':False,
            'generated_data':{'generator':'FLUX.1-schnell', 'teacher':'SigLIP SO400M/384',
                              'licence':'Apache-2.0'}}
    atomic(OUT/(name+'.json'), meta)
    command = [sys.executable, str(ROOT/'training/export.py'), '--name',name,
               '--samples','0','--out-dir',str(destination)]
    if vectors:
        command += ['--extra-vectors',str(vectors)]
    subprocess.run(command, check=True)
    build(destination)
    return meta


def prepare_timing():
    torch.set_num_threads(2)
    name='timing299_64_untrained'
    destination=OUT/name
    rng=np.random.default_rng(20261111)
    pattern=rng.integers(-100,101,257,dtype=np.int16).astype(np.int8)
    shapes=[(48,4,3,3),(96,48,3,3),(192,96,3,3),(FC,192*64),(299,FC)]
    arrays=[]
    for layer,shape in enumerate(shapes):
        count=math.prod(shape)
        weights=np.resize(pattern,count).reshape(shape)
        scale=1/(127*math.sqrt(math.prod(shape[1:])))
        scales=np.full((shape[0],shape[1]) if layer==0 else (shape[0],),scale,np.float32)
        bias=np.arange(shape[0],dtype=np.float32)*.0001 if layer==4 else np.zeros(shape[0],np.float32)
        arrays.append((weights,scales,bias))
    vectors=OUT/'long299_40_v4/exports/long299_40_v4_s000306013/train-only-vectors.json'
    export_arrays(arrays,name,destination,vectors)
    original=(destination/'Weights.luau').read_text()
    header=original[:original.index('\tdata = table.concat')]
    spec=[]
    for weights,scales,bias in arrays:
        spec.append({'count':int(weights.size),'scales':int(scales.size),
                     'scale':float(scales.flat[0]),'bias':int(bias.size),'bias_values':bias.tolist()})
    # Only this disposable timing module uses a compact raw-buffer constructor.
    # Inference kernels and the normal int8 export path are identical.
    pieces=['string.char('+','.join(str(int(v)%256) for v in pattern[i:i+64])+')'
            for i in range(0,len(pattern),64)]
    definitions='local pattern='+ '..'.join(pieces)+'\n'
    definitions+='local parts={}\n'
    for row in spec:
        definitions+=f'parts[#parts+1]=string.sub(string.rep(pattern,{math.ceil(row["count"]/257)}),1,{row["count"]})\n'
        definitions+=f'parts[#parts+1]=string.rep(string.pack("<f",{row["scale"]!r}),{row["scales"]})\n'
        definitions+='do local biases={'+','.join(repr(v) for v in row['bias_values'])+'};for i,v in biases do biases[i]=string.pack("<f",v) end;parts[#parts+1]=table.concat(biases) end\n'
    definitions+='local raw=buffer.fromstring(table.concat(parts))\n'
    compact=definitions+header+'\traw=raw,\n}\n'
    runtime=(destination/'DrawRecognizer.luau').read_text().replace(
        'local buf = decode64(Weights.data)','local buf = Weights.raw or decode64(Weights.data)')
    assert 'Weights.raw or' in runtime
    folder=destination/'studio'
    folder.mkdir(exist_ok=True)
    for filename in ['DrawRasterizer.luau','DrawJob.luau','DrawModelInfo.luau','TestVectors.luau']:
        (folder/filename).write_bytes((destination/filename).read_bytes())
    (folder/'Weights.luau').write_text(compact,encoding='utf-8',newline='\n')
    (folder/'DrawRecognizer.luau').write_text(runtime,encoding='utf-8',newline='\n')
    atomic(destination/'timing-manifest.json',{'name':name,'trained':False,'widths':list(WIDTHS),
           'fc':FC,'size':64,'weight_bytes':sum(w.size+s.nbytes+b.nbytes for w,s,b in arrays),
           'purpose':'Private server cost/parity only. Never selected for player guesses.'})
    print('CPU timing model ready',folder,flush=True)


if __name__=='__main__':
    prepare_timing()
