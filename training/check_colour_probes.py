"""Predict the same independently authored strokes later run in Studio."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from colour import planes
from colour_probes import cases,ROOT
from colour_raster import reference
from train import dequant,forward_plain

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);a=ap.parse_args();torch.set_num_threads(4)
    out=ROOT/'training/out';meta=json.loads((out/(a.name+'.json')).read_text())
    with np.load(out/(a.name+'.npz')) as z:q=[(z[f'w{i}'],z[f's{i}'],z[f'b{i}']) for i in range(5)]
    result=[]
    for c in cases():
        raw=[[[p[0] for p in st],[p[1] for p in st]] for st in c['strokes']]
        ink,owner=reference(raw,meta.get('size',28))
        colours=c['colours'] or [1]*len(raw)
        image=planes(ink,owner,[v-1 for v in colours])
        if q[0][0].shape[1]==1:image=image[:1]
        with torch.no_grad():v,i=forward_plain(dequant(q),torch.from_numpy(image)[None].double()).softmax(1)[0].topk(3)
        result.append({'name':c['name'],'top':[{'label':meta['categories'][int(j)],'prob':float(p)} for j,p in zip(i,v)]})
    p=ROOT/'notes'/f'colour-probes-{a.name}.json';p.write_text(json.dumps(result,indent=1),encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=1))
