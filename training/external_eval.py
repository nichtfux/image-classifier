"""Evaluation only. Never trains, writes weights, installs or talks to Studio.

All raster conversions and label aliases are fixed before scoring. Source images
are read directly from parquet/zip; only official DomainNet test members are read.
"""
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_k,'1')
import argparse, collections, hashlib, io, json, re, time, zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from skimage.feature import canny
from numba import njit
from categories import CATEGORIES
from colour import NET, PALETTE

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'data/external_eval'
WORK=256
ALIASES={
 'armchair':'chair','bear (animal)':'bear','beer mug':'mug','car (sedan)':'car',
 'flower with stem':'flower','flying bird':'bird','head phones':'headphones',
 'ice cream cone':'ice cream','lightbulb':'light bulb','mouse (animal)':'mouse',
 'pigeon':'bird','potted plant':'house plant','race car':'car','revolver':'pistol',
 'rooster':'bird','seagull':'bird','socks':'sock','speed boat':'speedboat',
 'standing bird':'bird','suv':'car','teacup':'cup','tennis racket':'tennis racquet',
 'trousers':'pants','tv':'television','wineglass':'wine glass','wrist watch':'wristwatch',
}
def normal(name): return re.sub(r'[ _-]+',' ',name.strip()).casefold()
CANON={normal(n):i for i,n in enumerate(CATEGORIES)}
def mapping(name):
    key=normal(name)
    if key in CANON:return CANON[key],'exact normalized'
    if key in ALIASES:return CATEGORIES.index(ALIASES[key]),'explicit synonym/subclass'
    return -1,'unmapped; no fuzzy matching'
def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=1,allow_nan=False)+'\n',encoding='utf-8',newline='\n')

@njit(cache=True)
def stroke_ink(xs,ys,offsets,size):
    # Independent point-segment rendering, same normalization/radius as serving.
    loX,loY,hiX,hiY=xs.min(),ys.min(),xs.max(),ys.max()
    span=max(hiX-loX,hiY-loY);scale=(22*size/28)/span if span>1e-6 else 1.
    cx,cy=(loX+hiX)/2,(loY+hiY)/2
    dist=np.full((size,size),np.inf)
    reach=size/28+.5
    for s in range(len(offsets)-1):
        a0,a1=offsets[s],offsets[s+1]
        for a in range(a0,max(a0+1,a1-1)):
            if a0==a1:continue
            b=min(a+1,a1-1)
            ax,ay=(xs[a]-cx)*scale+size/2,(ys[a]-cy)*scale+size/2
            bx,by=(xs[b]-cx)*scale+size/2,(ys[b]-cy)*scale+size/2
            dx,dy=bx-ax,by-ay;ll=dx*dx+dy*dy
            for y in range(max(0,int(np.ceil(min(ay,by)-reach-.5))),min(size-1,int(np.floor(max(ay,by)+reach-.5)))+1):
                for x in range(max(0,int(np.ceil(min(ax,bx)-reach-.5))),min(size-1,int(np.floor(max(ax,bx)+reach-.5)))+1):
                    t=min(1.,max(0.,((x+.5-ax)*dx+(y+.5-ay)*dy)/ll)) if ll>1e-12 else 0.
                    d=(x+.5-ax-t*dx)**2+(y+.5-ay-t*dy)**2
                    if d<dist[y,x]:dist[y,x]=d
    return np.minimum(1.,np.maximum(0.,reach-np.sqrt(dist))).astype(np.float32)

@njit(cache=True)
def skeleton_ink(xs,ys,size):
    x0,x1,y0,y1=xs.min(),xs.max(),ys.min(),ys.max()
    span=max(x1-x0,y1-y0);scale=(22*size/28)/span if span else 1.
    reach=size/28+.5;dist=np.full((size,size),np.inf)
    for k in range(len(xs)):
        px=(xs[k]-(x0+x1)/2)*scale+size/2
        py=(ys[k]-(y0+y1)/2)*scale+size/2
        for y in range(max(0,int(np.ceil(py-reach-.5))),min(size-1,int(np.floor(py+reach-.5)))+1):
            for x in range(max(0,int(np.ceil(px-reach-.5))),min(size-1,int(np.floor(px+reach-.5)))+1):
                d=(x+.5-px)**2+(y+.5-py)**2
                if d<dist[y,x]:dist[y,x]=d
    return np.minimum(1.,np.maximum(0.,reach-np.sqrt(dist))).astype(np.float32)

def heuristics(mask,rgb=None):
    sk=skeletonize(mask)
    neighbours=ndi.convolve(sk.astype(np.int16),np.ones((3,3),np.int16),mode='constant')-sk
    y,x=np.nonzero(mask)
    span=max(np.ptp(x),np.ptp(y),1) if len(x) else 1
    components=ndi.label(mask,np.ones((3,3)))[1]
    # Image proxies, NOT pen strokes or a quality scale.
    return [float(sk.sum()/span),float(ndi.label(sk&(neighbours>=3),np.ones((3,3)))[1]),
            float((sk&(neighbours==1)).sum()),float(components),float(mask.mean()),
            float((ndi.binary_fill_holes(mask)&~mask).mean())]

def raster_views(data,domain):
    image=Image.open(io.BytesIO(data)).convert('RGBA')
    image.thumbnail((1600,1600)) # fixed decode/work ceiling, independent of label
    a=np.asarray(image,dtype=np.float32)/255
    rgb=a[:,:,:3];alpha=a[:,:,3]
    if np.mean(alpha<.99)>.01:
        mask=alpha>.10
    else:
        border=np.concatenate((rgb[0],rgb[-1],rgb[:,0],rgb[:,-1]))
        bg=np.median(border,axis=0)
        # Same channel-wise maximum, without a slow 3-wide full-image reduction.
        distance=np.maximum(np.maximum(abs(rgb[:,:,0]-bg[0]),abs(rgb[:,:,1]-bg[1])),abs(rgb[:,:,2]-bg[2]))
        mask=distance>.10
    ys,xs=np.nonzero(mask)
    if not len(xs):raise ValueError('blank_foreground')
    x0,x1,y0,y1=xs.min(),xs.max()+1,ys.min(),ys.max()+1
    rgb=rgb[y0:y1,x0:x1];mask=mask[y0:y1,x0:x1]
    h,w=mask.shape;scale=WORK/max(h,w)
    shape=(max(1,round(w*scale)),max(1,round(h*scale)))
    coverage=np.asarray(Image.fromarray(mask.astype(np.uint8)*255).resize(shape,Image.Resampling.BOX),dtype=np.float32)/255
    rgb=np.asarray(Image.fromarray((rgb*255).astype(np.uint8)).resize(shape,Image.Resampling.BILINEAR),dtype=np.float32)/255
    solid=coverage>.12
    if domain in ('clipart','painting'):
        gray=rgb.mean(2);gray=np.where(solid,gray,1.)
        line=canny(gray,sigma=1.,low_threshold=.10,high_threshold=.20)
        line|=solid&~ndi.binary_erosion(solid)
        line=skeletonize(line)
    else:line=skeletonize(solid)
    yy,xx=np.nonzero(line)
    if not len(xx):raise ValueError('blank_skeleton')
    pal=np.asarray([p.rgb for p in PALETTE],np.float32)/255
    # Closest actual game ink, never the target class's colour prior.
    distance=((rgb[:,:,None,0]-pal[None,None,:,0])**2+
              (rgb[:,:,None,1]-pal[None,None,:,1])**2+
              (rgb[:,:,None,2]-pal[None,None,:,2])**2)
    inds=np.argmin(distance,axis=2)
    colour=NET[inds].transpose(2,0,1)
    source=np.concatenate((coverage[None],colour*coverage[None]),axis=0)
    if domain=='tu':source[1:]=0 # monochrome sketches have no ink-colour metadata
    result={}
    for size in (40,28):
        # Foreground includes unknown original brush radius. Fit its outside to
        # the expected outside extent of a trained stroke, with matching padding.
        target=24*size/28
        fit=4*target/max(source.shape[1:])
        shape=(max(1,round(source.shape[2]*fit)),max(1,round(source.shape[1]*fit)))
        preserve=[]
        for p in source:
            canvas=Image.new('F',(size*4,size*4))
            canvas.paste(Image.fromarray(p).resize(shape,Image.Resampling.BOX),((size*4-shape[0])//2,(size*4-shape[1])//2))
            preserve.append(np.asarray(canvas.resize((size,size),Image.Resampling.BOX),dtype=np.float32))
        preserve=np.stack(preserve)
        thin=np.zeros((4,size,size),np.float32)
        thin[0]=skeleton_ink(xx.astype(np.float64),yy.astype(np.float64),size)
        result[f'preserve{size}']=preserve.astype(np.float16)
        result[f'line{size}']=thin.astype(np.float16) # black line/edge control
    result['features']=np.asarray(heuristics(solid),np.float32)
    return result

_ZIPS={}
def convert_job(job):
    index,domain,payload=job
    try:
        if domain=='tu':data=payload
        else:
            path,member=payload
            if path not in _ZIPS:_ZIPS[path]=zipfile.ZipFile(path)
            data=_ZIPS[path].read(member)
        return index,raster_views(data,domain),None
    except Exception as e:return index,None,type(e).__name__+': '+str(e)[:150]

def tu_names():
    text=(ROOT/'data/external/tu-berlin/README.md').read_text(encoding='utf-8')
    rows=re.findall(r"^          '(\d+)': (.+)$",text,re.M)
    return [name for _,name in sorted(rows,key=lambda r:int(r[0]))]
def definitions():
    import pyarrow.parquet as pq
    tu=[]
    for p in sorted((ROOT/'data/external/tu-berlin').glob('*.parquet')):
        tu.append((p,pq.ParquetFile(p).metadata.num_rows))
    return tu
def label_report():
    maps={}
    maps['tu']=[dict(source=n,target=CATEGORIES[i] if i>=0 else None,index=i,rule=rule) for n in tu_names() for i,rule in [mapping(n)]]
    for domain in ('clipart','painting','sketch'):
        names=sorted(set(s.split('/')[1] for s in (ROOT/f'data/external/domainnet/{domain}_test.txt').read_text().splitlines()))
        maps[domain]=[dict(source=n,target=CATEGORIES[i] if i>=0 else None,index=i,rule=rule) for n in names for i,rule in [mapping(n)]]
    write_json(OUT/'label-mapping.json',maps)
    lines=['# External label mappings','', 'Fixed before scoring. Exact spelling normalization plus explicit synonyms/subclasses; no fuzzy label matching.','']
    for domain,rows in maps.items():
        lines+=['## '+domain,'',f'{sum(r["index"]>=0 for r in rows)} / {len(rows)} source labels mapped.','', '| Source | Our label | Rule |','|---|---|---|']
        lines += [f'| {r["source"]} | {r["target"] or "UNMAPPED"} | {r["rule"]} |' for r in rows]
        lines+=['']
    (ROOT/'notes/external-label-mapping.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    return maps

def prepare_images(domain,workers):
    import pyarrow.parquet as pq
    folder=OUT/domain;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return
    labels=[];ids=[];jobs=[];source_labels=[]
    if domain=='tu':
        names=tu_names()
        for path,_ in definitions():
            offset=0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
                for row in batch.to_pylist():
                    labels.append(mapping(names[row['label']])[0]);ids.append(f'{path.name}:{offset}')
                    source_labels.append(row['label'])
                    jobs.append((len(jobs),domain,row['image']['bytes']));offset+=1
    else:
        archive=str(ROOT/f'data/external/domainnet/{domain}.zip')
        for line in (ROOT/f'data/external/domainnet/{domain}_test.txt').read_text().splitlines():
            member,_=line.rsplit(' ',1)
            labels.append(mapping(member.split('/')[1])[0]);ids.append(member)
            jobs.append((len(jobs),domain,(archive,member)))
    n=len(jobs)
    arrays={k:np.lib.format.open_memmap(folder/(k+'.npy'),mode='w+',dtype=np.float16,shape=(n,4,size,size)) for size in (40,28) for k in (f'preserve{size}',f'line{size}')}
    features=np.zeros((n,6),np.float32);valid=np.zeros(n,bool);errors=[]
    start=time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done,(i,result,error) in enumerate(pool.map(convert_job,jobs,chunksize=12),1):
            if error:
                errors.append({'id':ids[i],'error':error})
                for arr in arrays.values():arr[i]=0
            else:
                valid[i]=True;features[i]=result.pop('features')
                for key,arr in arrays.items():arr[i]=result[key]
            if done%2000==0:print(domain,done,'/',n,'seconds',round(time.time()-start),flush=True)
    for arr in arrays.values():arr.flush()
    np.save(folder/'truth.npy',np.asarray(labels,np.int16));np.save(folder/'valid.npy',valid);np.save(folder/'features.npy',features)
    if source_labels:np.save(folder/'source-labels.npy',np.asarray(source_labels,np.int16))
    write_json(folder/'ids.json',ids);write_json(folder/'errors.json',errors)
    write_json(folder/'complete.json',{'n':n,'valid':int(valid.sum()),'mapped':int((np.asarray(labels)>=0).sum()),'seconds':time.time()-start,'evaluation_only':True})

_MANIFEST=None
def qd_job(args):
    global _MANIFEST
    ci,limit=args;name=CATEGORIES[ci];rows=[];seen=set()
    train=set(map(int,np.load(ROOT/f'data/expanded270-selections/{ci:03d}-train-ids.npy')))
    if _MANIFEST is None:_MANIFEST=json.loads((ROOT/'data/expanded270-40/splits.json').read_text())
    manifest=_MANIFEST
    held=set(map(int,manifest['test'][name]+manifest['validation'][name]))
    with (ROOT/f'data/simplified/{name}.ndjson').open('rb') as f:
        for line in f:
            d=json.loads(line)
            if d.get('recognized') is not False or not any(len(s[0]) for s in d.get('drawing',[])):continue
            key=int(d['key_id'])
            if key in seen:continue
            assert key not in train and key not in held,(name,key,'sample ID reused')
            seen.add(key);rows.append(d)
            if len(rows)>=limit:break
    arrays={size:np.zeros((len(rows),size,size),np.float32) for size in (28,40)}
    for i,d in enumerate(rows):
        strokes=[s for s in d['drawing'] if len(s[0])]
        xs=np.concatenate([s[0] for s in strokes]).astype(np.float64);ys=np.concatenate([s[1] for s in strokes]).astype(np.float64)
        offsets=np.r_[np.int64(0),np.cumsum([len(s[0]) for s in strokes])]
        for size in arrays:arrays[size][i]=stroke_ink(xs,ys,offsets,size)
    return ci,arrays,[str(d['key_id']) for d in rows]
def prepare_qd(limit,workers):
    folder=OUT/'qd_unrecognized';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return
    output={s:np.lib.format.open_memmap(folder/f'native{s}.npy',mode='w+',dtype=np.float16,shape=(270*limit,1,s,s)) for s in (28,40)}
    truth=[];ids=[];offset=0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for ci,arrays,keys in pool.map(qd_job,[(ci,limit) for ci in range(270)],chunksize=1):
            n=len(keys)
            for size,arr in output.items():arr[offset:offset+n,0]=arrays[size]
            truth.extend([ci]*n);ids.extend(keys);offset+=n
            if ci%20==0:print('qd_unrecognized',ci+1,'classes',offset,'drawings',flush=True)
    for arr in output.values():arr.flush()
    np.save(folder/'truth.npy',np.asarray(truth,np.int16));np.save(folder/'valid.npy',np.ones(offset,bool))
    write_json(folder/'ids.json',ids)
    write_json(folder/'complete.json',{'n':offset,'limit_per_class':limit,'training_flag_excluded':True,'sample_ids_overlap_train_validation_test':0,'selection':'first unique recognized=false records in each downloaded class file; prefix selection, not a random full-file sample','evaluation_only':True})

def load_net(name,device):
    from train_colour import quant_net
    with np.load(ROOT/f'training/out/{name}.npz') as z:q=[(z[f'w{i}'],z[f's{i}'],z[f'b{i}']) for i in range(5)]
    return quant_net(q,device)
def metrics(truth,top,valid):
    use=truth>=0;real=use&(truth<270);custom=use&(truth>=270)
    hit=(top[:,0]==truth)&valid;hit3=(top==truth[:,None]).any(1)&valid
    def cohort(mask):
        return {'n':int(mask.sum()),'top1':float(hit[mask].mean()) if mask.any() else None,'top3':float(hit3[mask].mean()) if mask.any() else None}
    classes=[]
    for ci in np.unique(truth[use]):
        mask=truth==ci;classes.append({'label':CATEGORIES[ci],**cohort(mask)})
    alarms=real&(top[:,0]>=270)&valid
    destinations=collections.Counter(CATEGORIES[i] for i in top[alarms,0])
    return {'all':cohort(use),'real':cohort(real),'custom':cohort(custom),'macro_top1':float(np.mean([c['top1'] for c in classes])),'macro_top3':float(np.mean([c['top3'] for c in classes])),
            'classes':classes,'worst_classes':sorted(classes,key=lambda r:(r['top1'],r['top3']))[:15],
            'real_to_custom':{'n':int(alarms.sum()),'denominator':int(real.sum()),'rate':float(alarms.sum()/real.sum()) if real.any() else None,'destinations':dict(destinations.most_common())},
            'unmapped':{'n':int((~use).sum()),'custom_routing':int(((~use)&(top[:,0]>=270)&valid).sum())},'conversion_failures_mapped':int((use&~valid).sum())}
def evaluate(domain,name):
    import torch
    folder=OUT/domain;output=folder/(name+'.json')
    if output.exists():return json.loads(output.read_text())
    net=load_net(name,'cuda');size=40 if name=='colour299_40' else 28
    truth=np.load(folder/'truth.npy');valid=np.load(folder/'valid.npy')
    modes=('native',) if domain=='qd_unrecognized' else ('line','preserve')
    result={}
    for mode in modes:
        x=np.load(folder/f'{mode}{size}.npy',mmap_mode='r')
        top=np.full((len(truth),3),-1,np.int16);prob=np.zeros(len(truth),np.float32);confidence=np.zeros(len(truth),np.float32)
        with torch.inference_mode():
            for lo in range(0,len(truth),512):
                hi=min(len(truth),lo+512);batch=np.array(x[lo:hi],dtype=np.float32)
                if net.c1.in_channels==1:batch=batch[:,:1]
                elif batch.shape[1]==1:batch=np.pad(batch,((0,0),(0,3),(0,0),(0,0)))
                z=net(torch.from_numpy(batch).cuda().contiguous(memory_format=torch.channels_last)).float();p=z.softmax(1)
                top[lo:hi]=p.topk(3,1).indices.cpu().numpy();confidence[lo:hi]=p.max(1).values.cpu().numpy()
                target=np.maximum(0,truth[lo:hi]);prob[lo:hi]=p.gather(1,torch.from_numpy(target.astype(np.int64)).cuda()[:,None])[:,0].cpu().numpy()
        top[~valid]=-1;prob[~valid]=0;prob[truth<0]=0
        result[mode]=metrics(truth,top,valid)
        np.savez_compressed(folder/f'{name}-{mode}-predictions.npz',top3=top,true_probability=prob,max_probability=confidence,truth=truth)
        print('SCORE',domain,name,mode,result[mode]['all'],flush=True)
    write_json(output,result);del net
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);ap.add_argument('--qd-per-class',type=int,default=300);ap.add_argument('--skip-prepare',action='store_true');args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    config={'evaluation_only':True,'work_size':WORK,'foreground_distance':.10,'coverage_threshold':.12,'line_canny':{'sigma':1,'low':.10,'high':.20},'palette':'nearest game RGB palette per pixel, no class priors','primary':{'tu':'line','sketch':'line','clipart':'preserve','painting':'preserve','qd_unrecognized':'native'},'float16_cache':True,'qd_per_class':args.qd_per_class}
    write_json(OUT/'conversion-plan.json',config);label_report()
    import torch
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=True
    hashes={name:hashlib.sha256((ROOT/f'training/out/{name}.npz').read_bytes()).hexdigest() for name in ('colour299_40','wide299_28_r4')}
    results={}
    for domain in ('tu','clipart','painting','sketch','qd_unrecognized'):
        if not args.skip_prepare:
            if domain=='qd_unrecognized':prepare_qd(args.qd_per_class,args.workers)
            else:prepare_images(domain,args.workers)
        results[domain]={name:evaluate(domain,name) for name in hashes}
        write_json(ROOT/'notes/external-results.json',results)
    assert all(hashlib.sha256((ROOT/f'training/out/{name}.npz').read_bytes()).hexdigest()==digest for name,digest in hashes.items()),'weights changed'
    write_json(OUT/'complete.json',{'weights_sha256':hashes,'no_training':True,'no_studio':True,'results':'notes/external-results.json'})

if __name__=='__main__':main()
