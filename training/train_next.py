"""Separate 2026-11-10 candidate. Recognized QD + own procedural data only.

Background-safe, atomic checkpoints; never writes the installed export or Studio.
Model selection uses pinned allowed validation, never outside-set test scores.
"""
import argparse, copy, json, math, os, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from categories import CATEGORIES
from colour import NET
from colour_data import ColourData, inputs, cohort_metrics
from train_colour import quant_net, quantise_colour
from gacl import QualityNet, widen_from, loss as gacl_loss
from style_augment import augment

HERE=Path(__file__).resolve().parent
OUT=HERE/'out'; NAME='gacl299_40_wide'


def save(net, path, info):
    temp=path.with_suffix('.part')
    torch.save({'state_dict':net.state_dict(),'categories':CATEGORIES,'cosine':net.cosine,
                'head_scale_folded':net.quantized_head,
                'widths':[net.c1.out_channels,net.c2.out_channels,net.c3.out_channels],
                'fc':net.f1.out_features,'size':40,**info},temp)
    os.replace(temp,path)


def load(path, dev='cuda'):
    ck=torch.load(path,map_location=dev,weights_only=True)
    net=QualityNet(ck['widths'],ck['fc'],ck['size']).to(dev).to(memory_format=torch.channels_last)
    net.load_state_dict(ck['state_dict']);net.cosine=ck['cosine']
    net.quantized_head=ck.get('head_scale_folded',False)
    return net.eval(),ck


def validation(net, data, rows, dev='cuda'):
    from fill import fill_planes
    result={}
    for mode in ('black','filled'):
        fixed=None if mode=='black' else data.fixed('policy',20261110)
        fills=None if mode=='black' else data.fixed_fills(fixed,20561110)
        truth=[];top=[]
        with torch.inference_mode():
            # Rows are contiguous 100-per-class, no random external validation.
            for lo,hi in rows:
                chunk=data.chunk(lo,hi,device=dev,fixed=fixed,black=mode=='black',fill=mode=='filled',fixed_fill=fills)
                x,y=inputs(chunk,black=mode=='black')
                if mode=='filled':
                    gen=torch.Generator(device=dev).manual_seed(20261110+lo)
                    fill_planes(x,torch.as_tensor(NET,device=dev)[chunk[5].long()],torch.ones(len(x),device=dev,dtype=torch.bool),generator=gen,region=chunk[6])
                z=net(x).float();top.append(z.topk(3,1).indices.cpu().numpy());truth.append(y.cpu().numpy())
        result[mode]=cohort_metrics(np.concatenate(truth),np.concatenate(top))
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--hours',type=float,default=2.6)
    ap.add_argument('--epochs',type=int,default=3);ap.add_argument('--warm-fraction',type=float,default=.5)
    ap.add_argument('--bs',type=int,default=512);ap.add_argument('--chunk',type=int,default=50000)
    ap.add_argument('--resume');a=ap.parse_args()
    torch.manual_seed(20261110);torch.set_num_threads(5);torch.backends.cudnn.benchmark=True
    dev='cuda';start=time.time();deadline=start+a.hours*3600
    base=torch.load(OUT/'colour299_40.pt',map_location=dev,weights_only=True)
    if a.resume:net,ck=load(a.resume,dev)
    else:net=widen_from(base['state_dict'],dev)
    with np.load(OUT/'colour299_40.npz') as z:q=[(z[f'w{i}'],z[f's{i}'],z[f'b{i}']) for i in range(5)]
    teacher=quant_net(q,dev)
    train=ColourData('train','colour299-40');val=ColourData('validation','colour299-40')
    ranges=[(i,min(i+a.chunk,len(train.y))) for i in range(0,len(train.y),a.chunk)]
    vy=np.asarray(val.y);rows=[]
    for ci in range(299):
        ix=np.flatnonzero(vy==ci);rows.append((int(ix[0]),int(ix[0])+100))
    baseline=validation(teacher,val,rows);print('baseline validation',json.dumps(baseline),flush=True)
    if not a.resume:
        net.eval();chunk=train.chunk(0,512,device=dev,black=True);x,y=inputs(chunk,black=True)
        # Verify the widening is function-preserving before adding symmetry noise.
        original=QualityNet((24,48,96),288).to(dev).eval();sd=original.state_dict()
        sd.update(base['state_dict']);original.load_state_dict(sd)
        with torch.no_grad():
            error=float((net(x)-original(x)).abs().max())
            exact=float((net.double()(x[:8].double())-original.double()(x[:8].double())).abs().max())
            net.float();original.float()
            print('widening float32 / float64 max error',error,exact,flush=True)
            assert exact<1e-9, 'widening must preserve the mathematical function'
            for n,p in net.named_parameters():
                if n.endswith('weight') and p.ndim>1:p.add_(torch.randn_like(p)*float(p.std())*.002)
        del original,chunk,x,y
    palette=torch.as_tensor(NET,device=dev);rng=np.random.default_rng(20261110)
    pool=ThreadPoolExecutor(1);history=[];best=-float('inf');style_data=None
    phases=[('warm',a.warm_fraction)]+[(f'gacl{i+1}',1.) for i in range(a.epochs)]
    if a.resume:phases=[(f'gacl{i+1}',1.) for i in range(a.epochs)]
    last=time.time();seen_total=int(ck.get('seen',0)) if a.resume else 0
    for pi,(phase,fraction) in enumerate(phases):
        if time.time()>deadline:break
        if phase!='warm' and not net.cosine:
            net.eval()
            chunk=train.chunk(150000,200000,device=dev,black=True)
            calibration=torch.randperm(len(chunk[0]),device=dev)[:4096]
            x,_=inputs(chunk,calibration,black=True)
            with torch.no_grad():
                norm=net.features(x).float().norm(dim=1)
                net.norm_scale.fill_(float(norm.median())/.56)
                net.f2.weight.copy_(F.normalize(net.f2.weight,dim=1));net.f2.bias.zero_()
            net.cosine=True;print('GACL norm scale',float(net.norm_scale),flush=True)
        head=[p for n,p in net.named_parameters() if n.startswith(('f1','f2'))]
        trunk=[p for n,p in net.named_parameters() if not n.startswith(('f1','f2'))]
        opt=torch.optim.AdamW([{'params':trunk,'lr':.00006 if phase=='warm' else .000035},
                              {'params':head,'lr':.00015 if phase=='warm' else .00018}],weight_decay=.0001)
        order=rng.permutation(len(ranges))[:math.ceil(len(ranges)*fraction)]
        initial_lr=[g['lr'] for g in opt.param_groups]
        phase_seen=0;loss_sum=0.;q_sum=0.;steps=0
        pending=pool.submit(train.chunk,*ranges[order[0]],seed=20261110+pi*10000,device=dev,fill=True)
        for ci,ri in enumerate(order):
            chunk=pending.result()
            if ci+1<len(order):pending=pool.submit(train.chunk,*ranges[order[ci+1]],seed=20261110+pi*10000+ci+1,device=dev,fill=True)
            style_path=HERE.parent/'data/synth-diverse40/train.npz'
            if style_data is None and style_path.exists():
                with np.load(style_path) as z:style_data={k:torch.from_numpy(z[k]).to(dev) for k in ['x','y','region','fill']}
                print('additional independent procedural styles loaded',len(style_data['y']),flush=True)
            perm=torch.randperm(len(chunk[0]),device=dev)
            net.train()
            # Preserve pretrained normalization; mixed paint styles do not redefine serving BN.
            for module in net.modules():
                if isinstance(module,(nn.BatchNorm2d,nn.Dropout)):module.eval()
            for pos in range(0,len(perm),a.bs):
                ix=perm[pos:pos+a.bs];x,y=inputs(chunk,ix);black=torch.zeros_like(x);black[:,0]=x[:,0]
                x=augment(x,chunk[6][ix],palette[chunk[5][ix].long()])
                if style_data is not None:
                    n=max(1,len(ix)//16);j=torch.randint(len(style_data['y']),(n,),device=dev)
                    sx=style_data['x'][j].float()/255
                    sx=augment(sx,style_data['region'][j].bool(),palette[style_data['fill'][j].long()])
                    x=torch.cat([x,sx]);y=torch.cat([y,style_data['y'][j].long()])
                # Black anchors remain exactly the serving input, not augmented geometry.
                nanchor=max(1,len(ix)//4)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    z,norm=net(x,details=True)
                    if net.cosine:
                        loss,details=gacl_loss(z.float(),norm,y,net.norm_scale,soft_bin=phase=='gacl1' and ci<len(order)//4)
                    else:loss=F.cross_entropy(z.float(),y);details={'q':0.}
                    bz,bnorm=net(black[:nanchor],details=True)
                    with torch.no_grad():tz=teacher(black[:nanchor]).float()
                    T=2.;kd=F.kl_div(F.log_softmax(bz.float()/T,1),F.softmax(tz/T,1),reduction='batchmean')*T*T
                    loss=loss+(.7 if net.cosine else 1.5)*kd+.25*F.cross_entropy(bz.float(),chunk[1][ix[:nanchor]])
                    if net.cosine:
                        anchor,_=gacl_loss(bz.float(),bnorm,chunk[1][ix[:nanchor]],net.norm_scale)
                        loss=loss+.15*anchor
                    real=y<270
                    if bool(real.any()):
                        # Cost-sensitive hard negatives: a generic dark mass must not become black hole.
                        target=z[real].gather(1,y[real,None]).squeeze(1)
                        bh=z[real,275]
                        loss=loss+.08*F.softplus(bh-target+2).mean()
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step()
                count=len(ix);phase_seen+=count;seen_total+=count;steps+=1;loss_sum+=float(loss.detach())*count;q_sum+=details['q']
            progress=(ci+1)/len(order)
            for group,lr in zip(opt.param_groups,initial_lr):group['lr']=lr*(.25+.75*.5*(1+math.cos(math.pi*progress)))
            if time.time()-last>50:
                print(json.dumps({'phase':phase,'chunk':ci+1,'chunks':len(order),'seen':phase_seen,'total_seen':seen_total,'seconds':round(time.time()-start),'loss':loss_sum/phase_seen,'q':q_sum/max(1,steps)}),flush=True);last=time.time()
                save(net,OUT/f'{NAME}-last.pt',{'phase':phase,'seen':seen_total,'seconds':time.time()-start})
            if time.time()>deadline:break
        net.eval();metrics=validation(net,val,rows)
        record={'phase':phase,'seen':phase_seen,'seconds':time.time()-start,'metrics':metrics,'cosine':net.cosine,'norm_scale':float(net.norm_scale)}
        history.append(record);print('validation',json.dumps(record),flush=True)
        acceptable=all(metrics['black'][g][k]>=baseline['black'][g][k]-.003 for g in ['real','synthetic'] for k in ['top1','top3'])
        score=metrics['filled']['all']['top1']+metrics['black']['real']['top1']
        save(net,OUT/f'{NAME}-{phase}.pt',{'phase':phase,'seen':seen_total,'seconds':time.time()-start,'validation':metrics})
        if net.cosine and score>best:
            best=score;save(net,OUT/f'{NAME}.pt',{'phase':phase,'seen':seen_total,'seconds':time.time()-start,'validation':metrics,'guard':acceptable})
        (OUT/f'{NAME}-progress.json').write_text(json.dumps({'history':history,'baseline_validation':baseline,'args':vars(a)},indent=2),encoding='utf-8',newline='\n')
        if time.time()>deadline:break
    print('DONE',time.time()-start,flush=True)


if __name__=='__main__':main()
