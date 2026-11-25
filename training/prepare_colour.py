"""Add disk-backed ownership/length metadata without changing any pinned raster.

Per-category restart markers and streaming raw JSON keep memory bounded. uint16
ownership preserves drawings with more than 255 strokes. No four-plane corpus.
"""
import argparse
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path
import time
import numpy as np
from categories import CATEGORIES
from colour_raster import render
from prepare_pair import write_json

DATA=Path(__file__).resolve().parent.parent/'data'


def category_job(job):
    ci,name,outname,size=job
    out=DATA/outname; src=DATA/('simplified' if ci<270 else 'synth')/(name+'.ndjson')
    base=DATA/(f'expanded270-{size}' if ci<270 else 'expanded299-28')
    manifest=json.loads((DATA/'expanded299-28/splits.json').read_text())
    done=out/f'{ci:03d}-done.json'
    identity={'name':name,'size':size,'bytes':src.stat().st_size,'mtime_ns':src.stat().st_mtime_ns,
              'renderer_sha256':hashlib.sha256(Path(__file__).with_name('colour_raster.py').read_bytes()).hexdigest()}
    if done.exists():
        previous=json.loads(done.read_text())
        extra_ok=size==28 or ci<270 or all((out/f'{ci:03d}-{s}-ink.npy').exists() for s in ('train','validation','test'))
        if previous['identity']==identity and extra_ok and all((out/f'{ci:03d}-{s}-{k}.npy').exists() for s in ('train','validation','test') for k in ('owner','offset','length')):
            return previous
    testids=set(manifest['test'][name]);valids=set(manifest['validation'][name])
    trainids=None
    if ci<270:
        ids=np.load(DATA/'expanded270-selections'/f'{ci:03d}-train-ids.npy')
        trainids={str(int(v)):i for i,v in enumerate(ids)}
    expected={s:np.load(base/f'{ci:03d}-{s}.npy',mmap_mode='r') for s in ('train','validation','test')}
    owners={s:np.lib.format.open_memmap(out/f'{ci:03d}-{s}-owner.npy',mode='w+',dtype=np.uint16,shape=(len(expected[s]),size,size)) for s in expected}
    new_ink={s:np.lib.format.open_memmap(out/f'{ci:03d}-{s}-ink.npy',mode='w+',dtype=np.uint8,shape=(len(expected[s]),size,size)) for s in expected} if size!=28 and ci>=270 else {}
    lens={s:[None]*len(expected[s]) for s in expected};position={s:0 for s in expected}
    maximum=0
    with src.open('rb') as f:
        for line in f:
            row=json.loads(line)
            if not row.get('recognized') or not any(len(st[0]) for st in row.get('drawing',[])):continue
            key=str(row['key_id'])
            if key in testids:split='test';i=position[split]
            elif key in valids:split='validation';i=position[split]
            elif trainids is not None:
                if key not in trainids:continue
                split='train';i=trainids[key]
            else:
                if position['train']>=len(expected['train']):continue
                split='train';i=position[split]
            ink,owner,length=render(row['drawing'],size)
            if new_ink:new_ink[split][i]=ink
            else:assert np.array_equal(ink,expected[split][i]),(name,split,i,'ink changed')
            owners[split][i]=owner;lens[split][i]=length
            maximum=max(maximum,len(length));position[split]+=1
    result={'identity':identity,'name':name,'max_strokes':maximum,'counts':{},'ink_pixels_exact':True}
    for s in expected:
        assert position[s]==len(expected[s]) and all(v is not None for v in lens[s]),(name,s,position[s])
        lengths=np.concatenate(lens[s]); offsets=np.r_[np.int64(0),np.cumsum([len(v) for v in lens[s]],dtype=np.int64)]
        np.save(out/f'{ci:03d}-{s}-length.npy',lengths)
        np.save(out/f'{ci:03d}-{s}-offset.npy',offsets)
        owners[s].flush()
        if new_ink:new_ink[s].flush()
        result['counts'][s]={'drawings':len(expected[s]),'strokes':len(lengths)}
    write_json(done,result)
    return result


def combine(out,results,size=28):
    original=DATA/'expanded299-28'
    for split in ('train','validation','test'):
        total=sum(r['counts'][split]['drawings'] for r in results)
        nstroke=sum(r['counts'][split]['strokes'] for r in results)
        own=np.lib.format.open_memmap(out/f'{split}-owner.npy',mode='w+',dtype=np.uint16,shape=(total,size,size))
        newx=np.lib.format.open_memmap(out/f'{split}-x.npy',mode='w+',dtype=np.uint8,shape=(total,size,size)) if size!=28 else None
        off=np.lib.format.open_memmap(out/f'{split}-offset.npy',mode='w+',dtype=np.int64,shape=(total+1,))
        length=np.lib.format.open_memmap(out/f'{split}-length.npy',mode='w+',dtype=np.float64,shape=(nstroke,))
        labels=np.load(original/f'{split}-y.npy',mmap_mode='r')
        arrays=[tuple(np.load(out/f'{i:03d}-{split}-{k}.npy',mmap_mode='r') for k in ('owner','offset','length')) for i in range(len(CATEGORIES))]
        inks=[np.load(DATA/f'expanded270-{size}'/f'{i:03d}-{split}.npy' if i<270 else out/f'{i:03d}-{split}-ink.npy',mmap_mode='r') for i in range(299)] if size!=28 else None
        rowpos,stroke_pos=0,0
        rounds=(max(len(a[0]) for a in arrays)+255)//256 if split=='train' else 1
        for ri in range(rounds):
            for ci,(o,ix,le) in enumerate(arrays):
                lo,hi=len(o)*ri//rounds,len(o)*(ri+1)//rounds
                n=hi-lo
                assert np.all(labels[rowpos:rowpos+n]==ci),(split,ci,rowpos)
                s0,s1=int(ix[lo]),int(ix[hi])
                own[rowpos:rowpos+n]=o[lo:hi]
                if newx is not None:newx[rowpos:rowpos+n]=inks[ci][lo:hi]
                off[rowpos:rowpos+n+1]=ix[lo:hi+1]-s0+stroke_pos
                length[stroke_pos:stroke_pos+s1-s0]=le[s0:s1]
                rowpos+=n;stroke_pos+=s1-s0
        assert rowpos==total and stroke_pos==nstroke
        own.flush();off.flush();length.flush();del own,off,length,arrays
        if newx is not None:
            newx.flush();del newx,inks
            np.save(out/f'{split}-y.npy',labels)
        print('combined',split,total,'drawings',nstroke,'strokes',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='colour299-28');ap.add_argument('--workers',type=int,default=8)
    ap.add_argument('--size',type=int,choices=[28,40],default=28);a=ap.parse_args()
    out=DATA/a.out;out.mkdir(exist_ok=True);t=time.time();results=[]
    with Pool(a.workers,maxtasksperchild=4) as pool:
        for result in pool.imap_unordered(category_job,[(i,n,a.out,a.size) for i,n in enumerate(CATEGORIES)]):
            results.append(result);print(f"{len(results)}/299 {result['name']} max strokes {result['max_strokes']} ({time.time()-t:.0f}s)",flush=True)
    results.sort(key=lambda r:CATEGORIES.index(r['name']))
    combine(out,results,a.size)
    write_json(out/'corpus.json',{'categories':CATEGORIES,'base':'expanded299-28' if a.size==28 else a.out,'size':a.size,'owner_dtype':'uint16',
        'tie_epsilon':1e-12,'ink_pixels_exact':True,'same_row_order':True,'category_records':results})
    print('colour metadata ready',flush=True)
