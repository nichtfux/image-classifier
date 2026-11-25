"""Small image-compatible sanity check, not an artistic-quality calibration."""
import json,io
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from scipy.stats import spearmanr
from skimage.morphology import skeletonize
from external_eval import ROOT,OUT,CATEGORIES,load_net,heuristics,write_json,tu_names,mapping

FEATURES=['skeleton_length_per_span','junction_clusters','endpoints','connected_components','ink_area_share','enclosed_paper_share']


def main():
    torch.set_num_threads(2)
    tu=OUT/'tu'
    truth=np.load(tu/'truth.npy');valid=np.load(tu/'valid.npy')
    source_labels=np.load(tu/'source-labels.npy')
    names=tu_names()
    exact=np.asarray([mapping(n)[1]=='exact normalized' for n in names])
    use=valid&(truth>=0)&(truth<270)&exact[source_labels]
    if (tu/'include.npy').exists():use&=np.load(tu/'include.npy')
    targets=np.unique(truth[use]);rng=np.random.default_rng(20261110)
    native=[];labels=[]
    for ci in targets:
        data=np.load(ROOT/f'data/expanded270-40/{ci:03d}-test.npy',mmap_mode='r')
        for i in rng.choice(len(data),min(80,len(data)),replace=False):
            x=np.zeros((4,40,40),np.float32);x[0]=data[i].astype(np.float32)/255
            native.append(x);labels.append(ci)
    x=np.stack(native);labels=np.asarray(labels,np.int16)
    net=load_net('colour299_40','cuda')
    q=np.zeros(len(x),np.float32);top=np.zeros(len(x),np.int16)
    with torch.inference_mode():
        for lo in range(0,len(x),256):
            z=net(torch.from_numpy(x[lo:lo+256]).cuda().contiguous(memory_format=torch.channels_last)).softmax(1)
            top[lo:lo+256]=z.argmax(1).cpu().numpy()
            q[lo:lo+256]=z.gather(1,torch.from_numpy(labels[lo:lo+256].astype(np.int64)).cuda()[:,None])[:,0].cpu().numpy()
    tq=np.load(tu/'colour299_40-line-predictions.npz')['true_probability']
    # Common 40px input view: strokes/time cannot be recovered from TU rasters.
    tx=np.load(tu/'line40.npy',mmap_mode='r')
    th=np.asarray([heuristics(tx[i,0]>.5) for i in np.flatnonzero(use)],np.float32)
    qh=np.asarray([heuristics(row[0]>.5) for row in x],np.float32)
    tl=truth[use];tq=tq[use]
    classes=[]
    for ci in targets:
        a=tl==ci;b=labels==ci
        classes.append({'label':CATEGORIES[ci],'tu_n':int(a.sum()),'quickdraw_n':int(b.sum()),
            'tu_true_probability':float(tq[a].mean()),'quickdraw_true_probability':float(q[b].mean()),
            'tu_features':dict(zip(FEATURES,th[a].mean(0).tolist())),
            'quickdraw_features':dict(zip(FEATURES,qh[b].mean(0).tolist()))})
    def macro(dataset,key):return float(np.mean([row[dataset+'_'+key] for row in classes]))
    result={'caveat':'No human quality labels here. Exact-normalized, shared real target classes only, equally weighted by target class. TU uses the line adapter; Quick Draw uses native prepared 40px black held-out inputs. All heuristics recomputed at common model-input resolution. Image/stroke conversion remains a confound. No score or thresholds fitted.',
        'shared_classes':len(targets),'tu_n':int(use.sum()),'quickdraw_n':len(q),
        'macro_true_probability':{'tu':macro('tu','true_probability'),'quickdraw':macro('quickdraw','true_probability')},
        'macro_features':{dataset:{feature:float(np.mean([r[dataset+'_features'][feature] for r in classes])) for feature in FEATURES} for dataset in ('tu','quickdraw')},
        'classes':classes,'unavailable':['pen-stroke count','drawing time','stroke order','last-30%-ink ablation','human quality rating','GACL quality norm']}
    for key,values in [('tu',(tl,tq,th)),('quickdraw',(labels,q,qh))]:
        lab,p,feat=values
        centered_p=p.copy();centered_f=feat.copy()
        for ci in targets:
            m=lab==ci;centered_p[m]-=p[m].mean();centered_f[m]-=feat[m].mean(0)
        result[key+'_within_class_spearman']={name:float(spearmanr(centered_p,centered_f[:,j]).statistic) for j,name in enumerate(FEATURES)}
    write_json(ROOT/'notes/external-detail-sanity.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='classes'},indent=1),flush=True)


if __name__=='__main__':main()
