"""Native 64px strokes with the 40px round's exact split membership.

Only validation rasters are cached as full arrays. Training uses a compact byte
offset index and renders source strokes on demand. Existing 40px raster-only
rehearsal banks are resized, while QD and generated strokes are rendered at 64.
"""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
import os

import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F

from categories import CATEGORIES
from colour import NET, planes
from colour_batch import sample, sample_fills
from colour_data import ColourData, inputs
from colour_raster import render
from generated_splits import group_key
from long_run_data import rows, Data as Data40
from fill import fill_planes
from style_augment64 import augment

ROOT = Path(__file__).resolve().parent.parent
GENERATED = ROOT/'data/generated'
FOLDER = ROOT/'data/colour299-64'
INDEX = FOLDER/'stroke-index'
OLD_RUN = ROOT/'training/out/long299_40_v4'
KEY = re.compile(rb'"key_id"\s*:\s*("[^"]+"|[^,}\s]+)')


def inside64(ink):
    mask = ink > round(.35*255)
    # 40px's one-pixel closure tolerance scales to two at 64px. Border flood
    # converges completely through scipy, so it has no fixed step limit.
    wall = ndimage.binary_dilation(mask, structure=np.ones((3,3),bool), iterations=2)
    outside = ~ndimage.binary_fill_holes(wall)
    outside |= ndimage.binary_dilation(outside, structure=np.ones((3,3),bool), iterations=2) & wall & ~mask
    return ~outside & ~mask


def raw_path(label):
    return ROOT/'data'/('simplified' if label<270 else 'synth')/(CATEGORIES[label]+'.ndjson')


class StrokeSource:
    def __init__(self):
        self.files = OrderedDict()
        self.offsets = np.load(INDEX/'train-global-offsets.npy',mmap_mode='r')
        self.validation = np.load(INDEX/'validation-offsets.npy',mmap_mode='r')

    def read(self,label,offset):
        handle = self.files.pop(label,None)
        if handle is None:
            handle = raw_path(label).open('rb')
        self.files[label] = handle
        while len(self.files)>24:
            _,old = self.files.popitem(last=False)
            old.close()
        handle.seek(int(offset))
        row = json.loads(handle.readline())
        assert row.get('recognized') and any(len(s[0]) for s in row['drawing'])
        return row['drawing']


def prepare_index(labels=None):
    INDEX.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'data/expanded299-28/splits.json').read_text())
    old=ColourData('train','colour299-40')
    for label in range(299) if labels is None else labels:
        if (ROOT/'training/out/round64/STOP').exists():
            raise InterruptedError('64px preparation stopped')
        name=CATEGORIES[label]
        marker=INDEX/f'{label:03d}.json'
        source=raw_path(label)
        identity={'bytes':source.stat().st_size,'mtime_ns':source.stat().st_mtime_ns}
        bank=INDEX/f'{label:03d}.npz'
        if marker.exists() and bank.exists():
            try:
                if json.loads(marker.read_text(encoding='utf-8'))['source']==identity:
                    with np.load(bank) as stored:
                        assert len(stored['validation'])==1000
                    continue
            except (OSError,ValueError,KeyError,AssertionError):
                pass
        valid=set(map(str,manifest['validation'][name]))
        test=set(map(str,manifest['test'][name]))
        if label<270:
            ids=np.load(ROOT/'data/expanded270-selections'/f'{label:03d}-train-ids.npy')
            desired={str(int(key)):i for i,key in enumerate(ids)}
            train=np.full(len(ids),-1,np.int64)
        else:
            desired=None
            train=[]
        held_hashes=set()
        if desired is None:
            with source.open('rb') as handle:
                for line in handle:
                    match=KEY.search(line)
                    if match and match.group(1).strip(b'"').decode('ascii') in valid|test:
                        drawing=json.loads(line)['drawing']
                        held_hashes.add(hashlib.sha256(json.dumps(drawing,separators=(',',':')).encode()).digest())
        validation=[]
        seen=set()
        train_ids=[]
        with source.open('rb') as handle:
            while True:
                offset=handle.tell()
                line=handle.readline()
                if not line:break
                key_match=KEY.search(line)
                if not key_match:continue
                key=key_match.group(1).strip(b'"').decode('ascii')
                if key in test:continue
                if desired is not None and key not in desired and key not in valid:continue
                row=json.loads(line)
                if not row.get('recognized') or not any(len(s[0]) for s in row.get('drawing',[])):continue
                if key in valid:
                    validation.append(offset)
                elif desired is not None:
                    train[desired[key]]=offset
                else:
                    signature=hashlib.sha256(json.dumps(row['drawing'],separators=(',',':')).encode()).digest()
                    if signature in seen or signature in held_hashes:continue
                    seen.add(signature)
                    train.append(offset);train_ids.append(key)
        assert len(validation)==1000,(name,len(validation))
        if desired is not None:
            assert np.all(train>=0),name
        else:
            prior=json.loads((ROOT/'data/expanded299-28'/f'{label:03d}-done.json').read_text())
            assert hashlib.sha256('\n'.join(train_ids).encode()).hexdigest()==prior['train_ids_sha256'],name
        temporary=bank.with_suffix('.part.npz')
        np.savez(temporary,train=np.asarray(train,np.int64),validation=np.asarray(validation,np.int64))
        temporary.replace(bank)
        temporary_marker=marker.with_suffix('.part.json')
        temporary_marker.write_text(json.dumps({'source':identity,'train':len(train),'validation':len(validation)}),encoding='utf-8',newline='\n')
        temporary_marker.replace(marker)
        print('STROKE_INDEX',label,name,len(train),flush=True)
    if labels is not None:return
    destination=INDEX/'train-global-offsets.npy'
    if not destination.exists():
        temporary=destination.with_suffix('.part.npy')
        mapping=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.int64,shape=(len(old.y),))
        yy=np.asarray(old.y)
        boundaries=np.r_[0,np.flatnonzero(yy[1:]!=yy[:-1])+1,len(yy)]
        positions=np.zeros(299,np.int64)
        per_class=[np.load(INDEX/f'{i:03d}.npz')['train'] for i in range(299)]
        for lo,hi in zip(boundaries[:-1],boundaries[1:]):
            ci=int(yy[lo]);start=int(positions[ci]);count=int(hi-lo)
            mapping[lo:hi]=per_class[ci][start:start+count]
            positions[ci]+=count
        assert all(positions[i]==len(per_class[i]) for i in range(299))
        mapping.flush();del mapping
        temporary.replace(destination)
    validation=np.concatenate([np.load(INDEX/f'{i:03d}.npz')['validation'] for i in range(299)])
    np.save(INDEX/'validation-offsets.npy',validation)


def prepare_validation():
    marker=FOLDER/'validation-ready.json'
    if marker.exists():return
    old=ColourData('validation','colour299-40')
    source=StrokeSource()
    count=len(old.y)
    ink=np.lib.format.open_memmap(FOLDER/'validation-x.npy',mode='w+',dtype=np.uint8,shape=(count,64,64))
    owner=np.lib.format.open_memmap(FOLDER/'validation-owner.npy',mode='w+',dtype=np.uint16,shape=(count,64,64))
    region=np.lib.format.open_memmap(FOLDER/'validation-regions.npy',mode='w+',dtype=np.uint8,shape=(count,512))
    for index,label in enumerate(old.y):
        if index%1000==0 and (ROOT/'training/out/round64/STOP').exists():
            raise InterruptedError('64px validation preparation stopped')
        drawing=source.read(int(label),source.validation[index])
        image,ownership,length=render(drawing,64)
        ink[index]=image;owner[index]=ownership;region[index]=np.packbits(inside64(image))
        if index%1000==0:
            check,_,_=render(drawing,40)
            assert np.array_equal(check,old.x[index]),'Validation identity/order differs from 40px'
        if index%10000==0:print('VALIDATION64',index,count,flush=True)
    ink.flush();owner.flush();region.flush()
    del ink,owner,region
    for name in ['y','offset','length']:
        shutil.copyfile(old.root/f'validation-{name}.npy',FOLDER/f'validation-{name}.npy')
    (FOLDER/'validation-regions.json').write_text(json.dumps({'size':64,'closure_pixels':2}),encoding='utf-8',newline='\n')
    (FOLDER/'corpus.json').write_text(json.dumps({'base':'colour299-64','size':64,'categories':CATEGORIES}),encoding='utf-8',newline='\n')
    marker.write_text(json.dumps({'drawings':count,'same_split_and_order':True,'closure_pixels':2}),encoding='utf-8',newline='\n')


def load_rows64(selected):
    cache=FOLDER/'generated'
    cache.mkdir(exist_ok=True)
    images,labels,teachers,identities=[],[],[],[]
    for row in selected:
        target=cache/(row['id']+'.npz')
        if not target.exists():
            original=GENERATED/row['sample']
            assert hashlib.sha256(original.read_bytes()).hexdigest()==row['sample_sha256']
            geometry=json.loads((GENERATED/'samples'/(row['id']+'.json')).read_text())
            image,owner,_=render(geometry['strokes'],64)
            colours=[c-1 for c in geometry['colours']]
            x=np.rint(planes(image.astype(np.float32)/255,owner,colours)*255).astype(np.uint8)
            with np.load(original) as old:
                temporary=target.with_suffix('.part.npz')
                np.savez(temporary,x=x,y=old['y'],teacher=old['teacher_logits'])
            temporary.replace(target)
        with np.load(target) as z:
            images.append(z['x']);labels.append(int(z['y']));teachers.append(z['teacher'])
        identities.append(row['id'])
    return {'x':np.stack(images),'y':np.asarray(labels,np.int64),
            'teacher':np.asarray(teachers,np.float32),'ids':np.asarray(identities)}


def pin_validation(folder):
    marker=folder/'generated-validation.json'
    bank=folder/'generated-validation.npz'
    if marker.exists():
        try:
            meta=json.loads(marker.read_text(encoding='utf-8'))
            if bank.exists() and hashlib.sha256(bank.read_bytes()).hexdigest()==meta['sha256']:
                return meta
        except (OSError,ValueError,KeyError):
            pass
    meta=json.loads((OLD_RUN/'generated-validation.json').read_text(encoding='utf-8'))
    temporary=bank.with_suffix('.part.npz')
    np.savez_compressed(temporary,**load_rows64(meta['rows']))
    temporary.replace(bank)
    meta['size']=64
    meta['sha256']=hashlib.sha256(bank.read_bytes()).hexdigest()
    temporary_marker=marker.with_suffix('.part.json')
    temporary_marker.write_text(json.dumps(meta),encoding='utf-8',newline='\n')
    temporary_marker.replace(marker)
    return meta


class GrowingBank64:
    def __init__(self,folder):
        self.validation=pin_validation(folder)
        self.count=0
        self.refresh()

    def refresh(self):
        chosen=[];seen=set()
        forbidden=set(self.validation['groups'])
        for row in rows():
            if row['split']!='train' or row['input_sha256'] in seen:continue
            assert group_key(row) not in forbidden
            seen.add(row['input_sha256']);chosen.append(row)
        if len(chosen)==self.count:return False
        if len(chosen)>self.count>0 and all(row['id']==identity for row,identity in zip(chosen,self.data['ids'])):
            additions=load_rows64(chosen[self.count:])
            updated={key:np.concatenate([self.data[key],additions[key]]) for key in self.data}
        else:
            updated=load_rows64(chosen)
        self.data=updated
        self.count=len(chosen)
        self.real=np.flatnonzero(self.data['y']<270)
        self.custom_by_class=[np.flatnonzero(self.data['y']==i) for i in range(270,299)]
        self.signature=hashlib.sha256('\n'.join(self.data['ids']).encode()).hexdigest()
        return True


def tensor_bank(bank,indices):
    x=torch.from_numpy(bank['x'][indices].astype(np.float32)/255).cuda()
    if x.shape[-1]!=64:
        x=F.interpolate(x,size=(64,64),mode='bilinear',align_corners=False)
    return x,torch.from_numpy(bank['y'][indices].astype(np.int64)).cuda()


class Data(Data40):
    def __init__(self,folder):
        self.corpus=ColourData('train','colour299-40')
        custom=np.flatnonzero(self.corpus.y>=270)
        cy=np.asarray(self.corpus.y[custom])
        self.custom=[custom[cy==i] for i in range(270,299)]
        self.source=StrokeSource()
        self.bank=GrowingBank64(folder)
        with np.load(ROOT/'data/synth-diverse40/train.npz') as z:
            self.proc={k:z[k] for k in z.files}
        self.proc_by_class=[np.flatnonzero(self.proc['y']==i) for i in range(270,299)]
        with np.load(ROOT/'data/generated/rounds/balanced-v4-20261110/qd-game/train.npz') as z:
            self.game={k:z[k] for k in ['x','y']}

    def qd(self,ids,rng):
        labels=np.asarray(self.corpus.y[ids])
        images,owners,lengths,regions=[],[],[],[]
        for index,label in zip(ids,labels):
            drawing=self.source.read(int(label),self.source.offsets[index])
            image,owner,length=render(drawing,64)
            images.append(image);owners.append(owner);lengths.append(length);regions.append(inside64(image))
        offset=np.r_[0,np.cumsum([len(x) for x in lengths])].astype(np.int64)
        length=np.concatenate(lengths)
        seed=int(rng.integers(1,2**30))
        colours=sample(labels,offset,length,seed,0)
        fills=sample_fills(labels,offset,length,colours,seed+700000)
        chunk=(torch.from_numpy(np.stack(images)).cuda(),torch.from_numpy(labels).cuda(),
               torch.from_numpy(np.stack(owners).astype(np.int32)).cuda(),
               torch.from_numpy(offset[:-1]).cuda(),torch.from_numpy(colours).cuda())
        x,y=inputs(chunk)
        region=torch.from_numpy(np.stack(regions)).cuda()
        rgb=torch.as_tensor(NET,device='cuda')[torch.from_numpy(fills).cuda().long()]
        return styles(x,region,rgb,y,rng),y

    def batch(self,rng,size=32):
        unit=size//16
        pieces=[self.qd(self.random_ids(6*unit,False,rng),rng),
                self.qd(self.random_ids(5*unit,True,rng),rng)]
        ix=np.asarray([rng.choice(self.proc_by_class[c]) for c in rng.integers(0,29,2*unit)])
        x,y=tensor_bank(self.proc,ix)
        region=F.interpolate(torch.from_numpy(self.proc['region'][ix].astype(np.float32)).cuda()[:,None],size=(64,64),mode='nearest')[:,0]>.5
        rgb=torch.as_tensor(NET,device='cuda')[torch.from_numpy(self.proc['fill'][ix]).cuda().long()]
        pieces.append((styles(x,region,rgb,y,rng),y))
        pieces.append(tensor_bank(self.game,rng.integers(0,len(self.game['y']),unit)))
        start=sum(len(y) for _,y in pieces)
        bank=self.bank
        present=[v for v in bank.custom_by_class if len(v)]
        chosen=np.r_[rng.choice(bank.real,unit),[rng.choice(present[c]) for c in rng.integers(0,len(present),unit)]]
        pieces.append(tensor_bank(bank.data,chosen))
        x=torch.cat([p[0] for p in pieces]).contiguous(memory_format=torch.channels_last)
        y=torch.cat([p[1] for p in pieces])
        black=torch.from_numpy(rng.random(len(x))<.30).cuda()
        x[black,1:]=0
        return x,y,start,torch.from_numpy(bank.data['teacher'][chosen]).cuda()


def styles(x,region,rgb,y,rng):
    """Explicit black/colour/filled/brush rehearsal; do not fill text into blobs.

    Training-only overpainting is suppressed for detail-dependent custom classes.
    Canonical filled/brush validation deliberately remains unchanged.
    """
    original = x.clone()
    choice = torch.from_numpy(rng.random(len(x))).cuda()
    no_blob = torch.zeros(299,device='cuda',dtype=torch.bool)
    for name in ['spiral','x2','undo arrow','cursed face','robot','straw hat']:
        no_blob[CATEGORIES.index(name)] = True
    region = region & ~no_blob[y][:,None,None]
    filled = choice < .30
    fill_planes(x,rgb,filled,region=region)
    brush = choice >= .70
    if brush.any():
        bx = augment(original[brush],region[brush],rgb[brush],preserve_details=True)
        x[brush] = bx
    return x.contiguous(memory_format=torch.channels_last)


def prepare_all():
    FOLDER.mkdir(parents=True,exist_ok=True)
    import msvcrt
    lock=(FOLDER/'preparation.lock').open('a+b')
    while True:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
            break
        except OSError:
            if (ROOT/'training/out/round64/STOP').exists():
                raise InterruptedError('64px preparation stopped while waiting for its lock')
            time.sleep(5)
    try:
        ready=FOLDER/'PREPARATION_DONE.json'
        if ready.exists():return
        # Per-class files are independent. Three CPU workers overlap indexing
        # with calibration while one preparation lock protects final assembly.
        from concurrent.futures import ProcessPoolExecutor
        groups=[list(range(270,299)),list(range(0,135)),list(range(135,270))]
        with ProcessPoolExecutor(max_workers=3) as pool:
            for result in pool.map(prepare_index,groups):
                pass
        prepare_index()
        prepare_validation()
        temporary=ready.with_suffix('.part.json')
        temporary.write_text(json.dumps({'size':64,'validation':299000,'complete':True}),encoding='utf-8',newline='\n')
        temporary.replace(ready)
    finally:
        lock.close()


if __name__=='__main__':
    prepare_all()
