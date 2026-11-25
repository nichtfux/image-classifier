"""Frozen, test-only comparison after selecting on allowed validation.

QD recognized pinned tests, failed QD, TU and DomainNet are never optimized here.
All external adapters and mappings are reused exactly from external_eval.py.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr,mannwhitneyu
from categories import CATEGORIES
from colour import NET
from colour_data import ColourData,inputs,cohort_metrics
from fill import fill_planes
from style_eval import augment
from gacl import from_quant
from external_eval import metrics,tu_names,mapping

ROOT=Path(__file__).resolve().parent.parent;OUT=ROOT/'training/out';CACHE=ROOT/'data/external_eval'


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8',newline='\n')


def load(name,device='cuda'):
    info=json.loads((OUT/f'{name}.json').read_text())
    with np.load(OUT/f'{name}.npz') as z:q=[(z[f'w{i}'],z[f's{i}'],z[f'b{i}']) for i in range(5)]
    return from_quant(q,device,info.get('feature_normalize',False),info.get('quality_norm_scale',100.)),info


def captured_replays(net):
    """Retained self-authored MCP captures, not the author’s private held-out set."""
    from colour_raster import reference
    from colour import planes
    records=[];images=[]
    for path in sorted((ROOT/'notes').glob('live-paced-after-*.json')):
        row=json.loads(path.read_text(encoding='utf-8'));strokes=row['capture']['strokes']
        raw=[[[p[0] for p in st['points']],[p[1] for p in st['points']]] for st in strokes]
        ink,owner=reference(raw,40);images.append(planes(ink,owner,[st['colour']-1 for st in strokes]))
        records.append({'file':path.name,'label':row['case']['label'],'paint':row['case']['paint'],'nominal_scale':row['case']['size']})
    with torch.inference_mode():
        x=torch.from_numpy(np.stack(images)).float().to(next(net.parameters()).device).contiguous(memory_format=torch.channels_last)
        p=net(x).float().softmax(1).cpu().numpy()
    for row,values in zip(records,p):
        top=np.argsort(-values)[:3]
        row['top3']=[{'label':CATEGORIES[int(i)],'prob':float(values[i])} for i in top]
        row['correct']=row['label']==row['top3'][0]['label']
    return {'drawings':len(records),'correct':sum(r['correct'] for r in records),'records':records,
            'caveat':'Twelve retained self-authored MCP developer gestures, replayed offline. Never trained on or used for selection. Not real-player accuracy.'}


def qd(net,data,mode,seed,batch=512):
    filled=mode=='filled';paint=mode=='painted'
    fixed=None if mode=='black' else data.fixed('policy' if filled or paint else mode,seed)
    fixed_fill=data.fixed_fills(fixed,seed+300000) if filled or paint else None
    top=np.empty((len(data.y),3),np.int16);norms=np.empty(len(data.y),np.float32)
    with torch.inference_mode():
        for lo in range(0,len(data.y),batch):
            hi=min(lo+batch,len(data.y))
            chunk=data.chunk(lo,hi,device='cuda',fixed=fixed,black=mode=='black',fill=filled or paint,fixed_fill=fixed_fill)
            x,y=inputs(chunk,black=mode=='black')
            if filled or paint:
                gen=torch.Generator(device='cuda').manual_seed(seed+lo)
                rgb=torch.as_tensor(NET,device='cuda')[chunk[5].long()]
                if filled:fill_planes(x,rgb,torch.ones(len(x),device='cuda',dtype=torch.bool),generator=gen,region=chunk[6])
                else:x=augment(x,chunk[6],rgb,generator=gen)
            z,norm=net(x,details=True);top[lo:hi]=z.topk(3,1).indices.cpu().numpy();norms[lo:hi]=norm.cpu().numpy()
    truth=np.asarray(data.y);result=cohort_metrics(truth,top)
    classes=[]
    for i in range(299):
        m=truth==i;classes.append({'label':CATEGORIES[i],'n':int(m.sum()),'top1':float((top[m,0]==i).mean()),'top3':float((top[m]==i).any(1).mean())})
    real=truth<270;bh=(top[:,0]==275)&real
    result['black_hole_false_alarms']={'count':int(bh.sum()),'denominator':int(real.sum()),'rate':float(bh.sum()/real.sum())}
    result['classes']=classes;result['worst_real']=sorted(classes[:270],key=lambda r:r['top1'])[:15]
    return result,top,norms


def external(net,domain,mode):
    folder=CACHE/domain;truth=np.load(folder/'truth.npy');valid=np.load(folder/'valid.npy')
    include=np.load(folder/'include.npy') if (folder/'include.npy').exists() else np.ones(len(truth),bool)
    x=np.load(folder/f'{mode}40.npy',mmap_mode='r')
    top=np.full((len(truth),3),-1,np.int16);norms=np.zeros(len(truth),np.float32);prob=np.zeros(len(truth),np.float32)
    with torch.inference_mode():
        for lo in range(0,len(truth),512):
            batch=np.array(x[lo:lo+512],dtype=np.float32)
            if batch.shape[1]==1:batch=np.pad(batch,((0,0),(0,3),(0,0),(0,0)))
            z,n=net(torch.from_numpy(batch).cuda().contiguous(memory_format=torch.channels_last),details=True)
            top[lo:lo+len(batch)]=z.topk(3,1).indices.cpu().numpy();norms[lo:lo+len(batch)]=n.cpu().numpy()
            target=np.maximum(0,truth[lo:lo+len(batch)]).astype(np.int64)
            prob[lo:lo+len(batch)]=z.softmax(1).gather(1,torch.from_numpy(target).cuda()[:,None])[:,0].cpu().numpy()
    top[~valid]=-1
    result=metrics(truth[include],top[include],valid[include])
    real=include&(truth>=0)&(truth<270);bh=real&valid&(top[:,0]==275)
    result['black_hole_false_alarms']={'count':int(bh.sum()),'denominator':int(real.sum()),'rate':float(bh.sum()/real.sum())}
    return result,top,norms,prob


def procedural(net,split='test'):
    data=np.load(ROOT/f'data/synth-diverse40/{split}.npz');truth=data['y'];top=[]
    with torch.inference_mode():
        for lo in range(0,len(truth),512):
            x=torch.from_numpy(data['x'][lo:lo+512].astype(np.float32)/255).cuda().contiguous(memory_format=torch.channels_last)
            top.append(net(x).topk(3,1).indices.cpu().numpy())
    top=np.concatenate(top);result=metrics(truth,top,np.ones(len(top),bool))
    result['caveat']='Own code-generated view=2 drawings, disjoint seeds; some family branches share geometry. Upper bound, not player accuracy.'
    return result


def probes(net):
    from colour_probes import cases
    from colour_raster import reference
    from colour import planes
    result=[]
    with torch.inference_mode():
        for c in cases():
            raw=[[[p[0] for p in st],[p[1] for p in st]] for st in c['strokes']]
            ink,owner=reference(raw,40);colours=c['colours'] or [1]*len(raw)
            x=planes(ink,owner,[v-1 for v in colours])
            z,norm=net(torch.from_numpy(x)[None].float().cuda().contiguous(memory_format=torch.channels_last),details=True)
            p=z.softmax(1)[0]
            values,ids=p.topk(3)
            result.append({'name':c['name'],'feature_norm':float(norm[0]),'top3':[{'label':CATEGORIES[int(i)],'prob':float(v)} for v,i in zip(values,ids)]})
    return result


def quality(net,qd_norms,tu_norms):
    # Same class-matched sample as external_quality.py, 80 drawings in each class.
    folder=CACHE/'tu';truth=np.load(folder/'truth.npy');valid=np.load(folder/'valid.npy')
    sources=np.load(folder/'source-labels.npy');names=tu_names()
    exact=np.asarray([mapping(n)[1]=='exact normalized' for n in names])
    use=valid&(truth>=0)&(truth<270)&exact[sources]&np.load(folder/'include.npy')
    targets=np.unique(truth[use]);rng=np.random.default_rng(20261110)
    tx=np.load(folder/'line40.npy',mmap_mode='r')
    data=ColourData('test','colour299-40');qy=np.asarray(data.y);classes=[];alltu=[];allqd=[];tcenter=[];qcenter=[];tinkcenter=[];qinkcenter=[]
    for ci in targets:
        ix=np.flatnonzero(qy==ci);chosen=rng.choice(len(ix),80,replace=False)
        # Prepared global class test blocks preserve the per-class source ordering.
        a=tu_norms[use&(truth==ci)];b=qd_norms[ix[chosen]]
        ai=(np.array(tx[use&(truth==ci),0])>.5).mean((1,2))
        bi=(np.array(data.x[ix[chosen]])>127).mean((1,2))
        tcenter.extend((a-a.mean()).tolist());qcenter.extend((b-b.mean()).tolist())
        tinkcenter.extend((ai-ai.mean()).tolist());qinkcenter.extend((bi-bi.mean()).tolist())
        auc=float(mannwhitneyu(a,b).statistic/(len(a)*len(b)))
        classes.append({'label':CATEGORIES[ci],'tu_n':len(a),'qd_n':len(b),'tu_mean_norm':float(a.mean()),'qd_mean_norm':float(b.mean()),'tu_over_qd_auc':auc})
        alltu.extend(a.tolist());allqd.extend(b.tolist())
    result={'shared_classes':len(classes),'tu_n':len(alltu),'quickdraw_n':len(allqd),
        'tu_mean_norm':float(np.mean([r['tu_mean_norm'] for r in classes])),
        'qd_mean_norm':float(np.mean([r['qd_mean_norm'] for r in classes])),
        'tu_higher_classes':sum(r['tu_mean_norm']>r['qd_mean_norm'] for r in classes),
        'macro_tu_over_qd_auc':float(np.mean([r['tu_over_qd_auc'] for r in classes])),
        'within_class_norm_ink_spearman':{'tu':float(spearmanr(tcenter,tinkcenter).statistic),'quickdraw':float(spearmanr(qcenter,qinkcenter).statistic)},
        'classes':classes,'caveat':'Dataset identity is not a human quality label. TU raster thinning vs QD native strokes remains a confound. No thresholds/tiers are fitted. The author Rough/Good/Detailed ratings are the ground truth.'}
    return result,np.array(alltu),np.array(allqd)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',default='gacl299_40_wide');ap.add_argument('--baseline-only',action='store_true')
    a=ap.parse_args();torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    report_path=ROOT/'notes/training-round-next.json'
    report=json.loads(report_path.read_text()) if report_path.exists() else {}
    models=['colour299_40'] if a.baseline_only else [a.name]
    data=ColourData('test','colour299-40')
    for name in models:
        net,info=load(name);result={'quickdraw':{},'external':{},'info':info}
        for mode,seed in [('black',100),('policy',20261110),('random',20261111),('filled',20261110),('painted',20261112)]:
            r,top,norm=qd(net,data,mode,seed)
            result['quickdraw'][mode]=r
            np.savez(OUT/f'{name}-next-{mode}-predictions.npz',top3=top,norm=norm)
            print(name,mode,{k:v for k,v in r.items() if k not in ['classes','worst_real']},flush=True)
            if mode=='black':qd_norms=norm
        for domain,primary in [('tu','line'),('qd_unrecognized','native'),('clipart','preserve'),('painting','preserve'),('sketch','line')]:
            modes=[primary] if domain=='qd_unrecognized' else ['line','preserve']
            result['external'][domain]={'primary_mode':primary,'modes':{}}
            for mode in modes:
                r,top,norm,prob=external(net,domain,mode)
                result['external'][domain]['modes'][mode]=r
                np.savez(OUT/f'{name}-next-{domain}-{mode}.npz',top3=top,norm=norm,true_probability=prob)
                print(name,domain,mode,r['all'],r['real_to_custom'],flush=True)
                if domain=='tu' and mode=='line':tu_norms=norm
        result['diverse_synthetic']=procedural(net)
        result['scripted_probes']=probes(net)
        result['captured_replays']=captured_replays(net)
        result['quality'],tuq,qdq=quality(net,qd_norms,tu_norms)
        result['quality']['trained_with_gacl']=bool(info.get('feature_normalize'))
        np.savez(OUT/f'{name}-quality-comparison.npz',tu=tuq,quickdraw=qdq)
        if name!='colour299_40':
            b=report['colour299_40']['quickdraw']['black'];current=result['quickdraw']['black']
            r4=json.loads((OUT/'colour299_40.json').read_text())['baseline_black_test']
            passed=all(current[g][k]>=max(b[g][k],r4[g][k])-.002 for g in ['real','synthetic'] for k in ['top1','top3'])
            result['black_acceptance_guard']=passed
            info.update({'int8_top1':current['all']['top1'],'int8_top3':current['all']['top3'],
                         'accepted_black':passed,'test_modes':result['quickdraw']})
            write(OUT/f'{name}.json',info)
        report[name]=result;write(report_path,report)
        del net;torch.cuda.empty_cache()
    print('DONE',flush=True)


if __name__=='__main__':main()
