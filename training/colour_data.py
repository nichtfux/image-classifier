"""Bounded colour input construction from ink/owner/length memory maps."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from categories import CATEGORIES
from colour import NET,colourise
from colour_batch import sample,sample_fills
from fill import fill_colour

DATA=Path(__file__).resolve().parent.parent/'data'


class ColourData:
    def __init__(self,split,folder='colour299-28'):
        self.root=DATA/folder
        meta=json.loads((self.root/'corpus.json').read_text())
        self.base=DATA/meta['base'];self.split=split
        self.size=meta['size']
        self.x=np.load(self.base/f'{split}-x.npy',mmap_mode='r')
        self.y=np.load(self.base/f'{split}-y.npy',mmap_mode='r')
        self.owner=np.load(self.root/f'{split}-owner.npy',mmap_mode='r')
        self.offset=np.load(self.root/f'{split}-offset.npy',mmap_mode='r')
        self.length=np.load(self.root/f'{split}-length.npy',mmap_mode='r')
        assert len(self.x)==len(self.owner)==len(self.offset)-1==len(self.y)

    def fixed(self,mode,seed):
        path=self.root/f'{self.split}-colours-{mode}-{seed}.npy'
        signature={'colour_sha256':hashlib.sha256(Path(__file__).with_name('colour.py').read_bytes()).hexdigest(),
                   'length_mtime':(self.root/f'{self.split}-length.npy').stat().st_mtime_ns,
                   'strokes':len(self.length),'mode':mode,'seed':seed}
        marker=path.with_suffix('.json')
        if path.exists() and marker.exists() and json.loads(marker.read_text())==signature:
            return np.load(path,mmap_mode='r')
        result=np.lib.format.open_memmap(path,mode='w+',dtype=np.uint8,shape=(len(self.length),))
        rng=np.random.default_rng(seed)
        for row,label in enumerate(self.y):
            lo,hi=int(self.offset[row]),int(self.offset[row+1])
            if mode=='policy':result[lo:hi]=colourise(CATEGORIES[int(label)],self.length[lo:hi],rng)
            elif mode=='random':result[lo:hi]=rng.integers(0,len(NET),hi-lo,dtype=np.uint8)
            elif mode=='black':result[lo:hi]=0
            else:raise ValueError(mode)
        result.flush();del result
        marker.write_text(json.dumps(signature,indent=1),encoding='utf-8',newline='\n')
        return np.load(path,mmap_mode='r')

    def fixed_fills(self,colours,seed):
        path=self.root/f'{self.split}-fills-{seed}.npy'
        signature={'fill_sha256':hashlib.sha256(Path(__file__).with_name('fill.py').read_bytes()).hexdigest(),
                   'colour_sha256':hashlib.sha256(Path(__file__).with_name('colour.py').read_bytes()).hexdigest(),
                   'drawings':len(self.y),'seed':seed}
        marker=path.with_suffix('.json')
        if path.exists() and marker.exists() and json.loads(marker.read_text())==signature:
            return np.load(path,mmap_mode='r')
        result=np.empty(len(self.y),np.uint8);rng=np.random.default_rng(seed)
        for row,label in enumerate(self.y):
            lo,hi=int(self.offset[row]),int(self.offset[row+1])
            result[row]=fill_colour(CATEGORIES[int(label)],colours[lo:hi],self.length[lo:hi],rng)
        np.save(path,result);marker.write_text(json.dumps(signature,indent=1),encoding='utf-8',newline='\n')
        return np.load(path,mmap_mode='r')

    def chunk(self,lo,hi,seed=0,mode=0,device='cuda',fixed=None,black=False,fill=False,fixed_fill=None):
        x=torch.from_numpy(np.array(self.x[lo:hi])).to(device)
        y=torch.from_numpy(np.array(self.y[lo:hi])).to(device)
        if black:return x,y,None,None,None
        offset=np.array(self.offset[lo:hi+1]);start=int(offset[0]);end=int(offset[-1]);offset-=start
        labels=np.array(self.y[lo:hi]);length=np.array(self.length[start:end])
        colours=np.array(fixed[start:end]) if fixed is not None else sample(labels,offset,length,seed,mode)
        owner=torch.from_numpy(np.array(self.owner[lo:hi],dtype=np.int32)).to(device)
        result=(x,y,owner,torch.from_numpy(offset[:-1]).to(device),torch.from_numpy(colours).to(device))
        if fill:
            fi=np.array(fixed_fill[lo:hi]) if fixed_fill is not None else sample_fills(labels,offset,length,colours,seed+700000)
            result+= (torch.from_numpy(fi).to(device),)
            region=self.root/f'{self.split}-regions.npy'
            if region.with_suffix('.json').exists():
                packed=np.load(region,mmap_mode='r')
                masks=np.unpackbits(np.array(packed[lo:hi]),axis=1,count=self.size**2).reshape(hi-lo,self.size,self.size)
                result+=(torch.from_numpy(masks.astype(bool)).to(device),)
        return result


def inputs(chunk,indices=None,black=False):
    x,y,owner,offset,colours=chunk[:5]
    if indices is not None:x,y=x[indices],y[indices]
    ink=x.float().div(255)
    if black or owner is None:
        out=torch.zeros((len(x),*x.shape[1:],4),device=x.device)
        out[...,0]=ink
    else:
        o=owner if indices is None else owner[indices]
        off=offset if indices is None else offset[indices]
        palette=torch.as_tensor(NET,device=x.device)
        rgb=palette[colours[off[:,None,None]+o.long()].long()]
        out=torch.cat([ink[...,None],ink[...,None]*rgb],3)
    return out.permute(0,3,1,2).contiguous(memory_format=torch.channels_last),y


def cohort_metrics(truth,top):
    result={}
    for name,mask in [('all',np.ones(len(truth),bool)),('real',truth<270),('synthetic',truth>=270)]:
        result[name]={'drawings':int(mask.sum()),'top1':float(np.mean(top[mask,0]==truth[mask])),
                      'top3':float(np.any(top[mask]==truth[mask,None],axis=1).mean())}
    real=truth<270;alarms=top[real,0]>=270
    counts=np.bincount(top[real,0][alarms],minlength=299)
    result['false_alarms']={'count':int(alarms.sum()),'denominator':int(real.sum()),'rate':float(alarms.mean()),
       'worst':[{'label':CATEGORIES[int(i)],'count':int(counts[i]),'rate':float(counts[i]/real.sum())}
                for i in np.argsort(-counts)[0:10] if counts[i] and i>=270]}
    return result
