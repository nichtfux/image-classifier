"""4-plane colour learning, black retention guards, int8 evaluation and checkpoints.

RGB warm-up can freeze all existing weights. Whole-model refinement uses black
distillation, fixed BN statistics, disabled dropout and a smaller shape LR.
First-conv scales are per output/input channel to avoid RGB changing ink rounding.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train import Net,fold,quantise,dequant
from colour_data import ColourData,inputs,cohort_metrics
from colour import PALETTE,NET
from fill import fill_planes,P_FILL

HERE=Path(__file__).resolve().parent;OUT=HERE/'out'


def quantise_colour(net):
    layers=fold(net);q=quantise(layers)
    w,b=layers[0]
    scale=np.abs(w).max((2,3))/127.
    scale=np.where(scale>0,scale,1.).astype(np.float32)
    wi=np.clip(np.rint(w/scale[:,:,None,None].astype(np.float64)),-127,127).astype(np.int8)
    q[0]=(wi,scale,b.astype(np.float32))
    return q


def quant_net(q,dev):
    inc=q[0][0].shape[1];fc=q[3][0].shape[0]
    size=28 if q[3][0].shape[1]==96*9 else 40
    net=Net([24,48,96],fc,299,size=size,in_channels=inc).to(dev).to(memory_format=torch.channels_last)
    plain=dequant(q)
    with torch.no_grad():
        for i,(w,b) in enumerate(plain[:3],1):
            c=getattr(net,f'c{i}');c.weight.copy_(torch.from_numpy(w).to(dev))
            c.bias=nn.Parameter(torch.from_numpy(b).float().to(dev));setattr(net,f'b{i}',nn.Identity())
        for target,(w,b) in zip((net.f1,net.f2),plain[3:]):
            target.weight.copy_(torch.from_numpy(w).to(dev));target.bias.copy_(torch.from_numpy(b).to(dev))
    return net.eval().requires_grad_(False)


def evaluate(net,data,mode,seed,dev,batch=1024,save=None):
    filled=mode=='filled'
    fixed=None if mode=='black' else data.fixed('policy' if filled else mode,seed)
    fixed_fill=data.fixed_fills(fixed,seed+300000) if filled else None
    top=np.empty((len(data.y),3),np.int16)
    inc=net.c1.in_channels
    with torch.no_grad():
        for lo in range(0,len(data.y),batch):
            hi=min(lo+batch,len(data.y));chunk=data.chunk(lo,hi,device=dev,fixed=fixed,black=mode=='black',fill=filled,fixed_fill=fixed_fill)
            x,y=inputs(chunk,black=mode=='black')
            if filled:
                gen=torch.Generator(device=dev).manual_seed(seed+lo)
                fill_planes(x,torch.as_tensor(NET,device=dev)[chunk[5].long()],torch.ones(len(x),device=dev,dtype=torch.bool),generator=gen,region=chunk[6] if len(chunk)>6 else None)
            if inc==1:x=x[:,:1].contiguous(memory_format=torch.channels_last)
            z=net(x).float();top[lo:hi]=z.topk(3,1).indices.cpu().numpy()
    truth=np.array(data.y,dtype=np.int16)
    if save:np.savez_compressed(save,truth=truth,top3=top)
    return cohort_metrics(truth,top)


def guard(metrics,baseline):
    return all(metrics[group][key]>=baseline[group][key]-.002
               for group in ('real','synthetic') for key in ('top1','top3'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',default='colour299_28')
    ap.add_argument('--epochs',type=int,default=1);ap.add_argument('--rgb-only',action='store_true')
    ap.add_argument('--resume');ap.add_argument('--chunk',type=int,default=100000)
    ap.add_argument('--bs',type=int,default=1024);ap.add_argument('--rgb-lr',type=float,default=.0005)
    ap.add_argument('--shape-lr',type=float,default=.00002);ap.add_argument('--keep-weight',type=float,default=2.)
    ap.add_argument('--colour-seed',type=int,default=20261109)
    ap.add_argument('--size',type=int,choices=[28,40],default=28)
    ap.add_argument('--p-fill',type=float,default=P_FILL)
    a=ap.parse_args();torch.manual_seed(0);torch.set_num_threads(6);torch.backends.cudnn.benchmark=True
    dev='cuda' if torch.cuda.is_available() else 'cpu';start=time.time()
    base=torch.load(OUT/'wide299_28_r4.pt',map_location=dev,weights_only=True)
    labels=base['categories'];fc=base['state_dict']['f1.weight'].shape[0]
    net=Net([24,48,96],fc,299,size=a.size,in_channels=4).to(dev).to(memory_format=torch.channels_last)
    state=net.state_dict()
    for key,value in base['state_dict'].items():
        if key=='c1.weight':state[key].zero_();state[key][:,:1].copy_(value)
        elif key=='f1.weight' and a.size==40:
            value=F.interpolate(value.reshape(fc,96,3,3),size=(5,5),mode='bilinear',align_corners=False)*9/25
            state[key].copy_(value.reshape(fc,-1))
        else:state[key].copy_(value)
    if a.size==40 and not a.resume:
        # Retain the trained 40px real-class model, then append r4's 32 custom
        # features/head as a starting point. All layers are refined afterwards.
        forty=torch.load(OUT/'wide270_40.pt',map_location=dev,weights_only=True)['state_dict']
        for key,value in forty.items():
            if key=='c1.weight':state[key][:,:1].copy_(value)
            elif key=='f1.weight':state[key][:256].copy_(value)
            elif key=='f1.bias':state[key][:256].copy_(value)
            elif key=='f2.weight':state[key][:270,:256].copy_(value);state[key][:270,256:].zero_()
            elif key=='f2.bias':state[key][:270].copy_(value)
            else:state[key].copy_(value)
    net.load_state_dict(state)
    if a.resume:
        resumed=torch.load(a.resume,map_location=dev,weights_only=True)
        assert resumed['categories']==labels
        net.load_state_dict(resumed['state_dict'])
    ink_kernel=net.c1.weight[:,:1].detach().clone()
    with np.load(OUT/'wide299_28_r4.npz') as z:oldq=[(z[f'w{i}'],z[f's{i}'],z[f'b{i}']) for i in range(5)]
    teacher=quant_net(oldq,dev)
    folder=f'colour299-{a.size}'
    train,validation=ColourData('train',folder),ColourData('validation',folder)
    baseline=evaluate(teacher,ColourData('validation'),'black',100,dev)
    paired_ink=np.load(train.base.parent/'expanded299-28/train-x.npy',mmap_mode='r') if a.size==40 else None
    print('baseline black validation',json.dumps(baseline),flush=True)
    if a.rgb_only:
        for p in net.parameters():p.requires_grad_(False)
        net.c1.weight.requires_grad_(True)
    groups=[{'params':[net.c1.weight],'lr':a.rgb_lr}]
    if not a.rgb_only:groups.append({'params':[p for n,p in net.named_parameters() if n!='c1.weight'],'lr':a.shape_lr})
    opt=torch.optim.AdamW(groups,weight_decay=1e-4)
    ranges=[(lo,min(lo+a.chunk,len(train.y))) for lo in range(0,len(train.y),a.chunk)]
    total_steps=a.epochs*sum((hi-lo+a.bs-1)//a.bs for lo,hi in ranges)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,[g['lr'] for g in groups],total_steps=total_steps)
    amp={'device_type':'cuda','dtype':torch.bfloat16,'enabled':dev=='cuda'}
    rng=torch.Generator().manual_seed(0);loader=ThreadPoolExecutor(1)
    best_score=-1.;best_state=None;selected=0;history=[];last=time.time()
    any_score=-1.;any_state=None;any_epoch=0
    palette=torch.as_tensor(NET,device=dev)
    if a.resume:
        qn=quant_net(quantise_colour(net.eval()),dev)
        initial={m:evaluate(qn,validation,m,s,dev) for m,s in [('black',100),('policy',20261110),('filled',20261110)]}
        acceptable=guard(initial['black'],baseline)
        history.append({'epoch':0,**initial,'guard':acceptable})
        if acceptable:
            best_score=(initial['policy']['all']['top1']+initial['filled']['all']['top1'])/2
            best_state=copy.deepcopy(net.state_dict())
        del qn
    for ep in range(a.epochs):
        net.train()
        for module in net.modules():
            if isinstance(module,(nn.BatchNorm2d,nn.Dropout)):module.eval()
        order=[ranges[i] for i in torch.randperm(len(ranges),generator=rng).tolist()]
        pending=loader.submit(train.chunk,*order[0],seed=a.colour_seed+ep*10000,device=dev,fill=True)
        loss_total,seen=0.,0
        for ci,(lo,hi) in enumerate(order):
            chunk=pending.result()
            if ci+1<len(order):pending=loader.submit(train.chunk,*order[ci+1],seed=a.colour_seed+ep*10000+ci+1,device=dev,fill=True)
            teacher_ink=torch.from_numpy(np.array(paired_ink[lo:hi])).to(dev) if paired_ink is not None else chunk[0]
            permutation=torch.randperm(hi-lo,device=dev)
            for pos in range(0,hi-lo,a.bs):
                idx=permutation[pos:pos+a.bs];x,y=inputs(chunk,idx)
                black=torch.zeros_like(x);black[:,0]=x[:,0]
                fill_planes(x,palette[chunk[5][idx].long()],torch.rand(len(idx),device=dev)<a.p_fill,region=chunk[6][idx] if len(chunk)>6 else None)
                with torch.autocast(**amp):
                    logits=net(x);loss=F.cross_entropy(logits.float(),y)
                    if not a.rgb_only:
                        black_logits=net(black).float()
                        with torch.no_grad():target=teacher(teacher_ink[idx,None].float().div(255).contiguous(memory_format=torch.channels_last)).float()
                        T=2.;loss=loss+a.keep_weight*T*T*F.kl_div(F.log_softmax(black_logits/T,1),F.softmax(target/T,1),reduction='batchmean')
                        loss=loss+.25*F.cross_entropy(black_logits,y)
                opt.zero_grad(set_to_none=True);loss.backward()
                net.c1.weight.grad[:,:1].zero_()
                opt.step();sched.step()
                with torch.no_grad():net.c1.weight[:,:1].copy_(ink_kernel)
                loss_total+=float(loss.item())*len(idx);seen+=len(idx)
            del chunk,permutation,x,y,logits,loss
            if time.time()-last>=45 or ci+1==len(order):
                print(f'epoch {ep+1}/{a.epochs} chunk {ci+1}/{len(order)} {seen/len(train.y):.1%} ({time.time()-start:.0f}s)',flush=True);last=time.time()
        net.eval();q=quantise_colour(net);qn=quant_net(q,dev)
        black=evaluate(qn,validation,'black',100,dev)
        policy=evaluate(qn,validation,'policy',20261110,dev)
        random=evaluate(qn,validation,'random',20261111,dev)
        filled=evaluate(qn,validation,'filled',20261110,dev)
        acceptable=guard(black,baseline)
        metrics={'epoch':ep+1,'loss':loss_total/seen,'black':black,'policy':policy,'random':random,'filled':filled,'guard':acceptable}
        history.append(metrics);print('validation',json.dumps(metrics),flush=True)
        score=(policy['all']['top1']+filled['all']['top1'])/2
        torch.save({'state_dict':{k:v.cpu() for k,v in net.state_dict().items()},'categories':labels,'args':vars(a),'input_channels':4},OUT/(a.name+'.last.pt'))
        if score>any_score:any_score=score;any_state=copy.deepcopy(net.state_dict());any_epoch=ep+1
        if acceptable and score>best_score:
            best_score=score;best_state=copy.deepcopy(net.state_dict());selected=ep+1
            torch.save({'state_dict':{k:v.cpu() for k,v in best_state.items()},'categories':labels,'args':vars(a),'input_channels':4},OUT/(a.name+'.pt'))
    loader.shutdown()
    if best_state is None:
        print('No checkpoint passed validation retention; export comparison only, never install.',flush=True)
        best_state=any_state;selected=any_epoch
    assert best_state is not None
    net.load_state_dict(best_state);net.eval();q=quantise_colour(net);qn=quant_net(q,dev)
    torch.save({'state_dict':{k:v.cpu() for k,v in best_state.items()},'categories':labels,'args':vars(a),'input_channels':4},OUT/(a.name+'.pt'))
    if a.rgb_only:
        assert np.array_equal(q[0][0][:,:1],oldq[0][0]) and np.array_equal(q[0][1][:,0],oldq[0][1])
        for i in range(1,5):assert all(np.array_equal(u,v) for u,v in zip(q[i],oldq[i]))
    arr={}
    for i,(w,s,b) in enumerate(q):arr.update({f'w{i}':w,f's{i}':s,f'b{i}':b})
    np.savez(OUT/(a.name+'.npz'),**arr)
    test=ColourData('test',folder);modes={}
    for mode,seed in [('black',100),('policy',20261112),('random',20261113),('filled',20261112)]:
        modes[mode]=evaluate(qn,test,mode,seed,dev,save=OUT/(a.name+'.'+mode+'.predictions.npz'))
        print('test',mode,json.dumps(modes[mode]),flush=True)
    with np.load(OUT/'wide299_28_r4.predictions.npz') as old:
        oldtruth,oldtop=old['truth'],old['top3']
    baseline_test=cohort_metrics(oldtruth,oldtop)
    accepted=guard(modes['black'],baseline_test)
    results={'name':a.name,'categories':labels,'widths':[24,48,96],'fc':fc,'size':a.size,'input_channels':4,
             'params':sum(p.numel() for p in net.parameters()),'corpus':'expanded299-28','owner_corpus':folder,
             'int8_top1':modes['black']['all']['top1'],'int8_top3':modes['black']['all']['top3'],
             'baseline':'wide299_28_r4','baseline_black_test':baseline_test,'test_modes':modes,'accepted':accepted,
             'selected_epoch':selected,'history':history,'training_args':vars(a),'train_seconds':time.time()-start,
             'palette':[{'name':p.name,'rgb':p.rgb,'net':p.net} for p in PALETTE],
             'colour_caveat':'Colourise gains primarily measure the synthetic colour table, not real coloured player drawings.',
             'first_conv_quantization':'per output/input plane; other layers per output channel'}
    (OUT/(a.name+'.json')).write_text(json.dumps(results,indent=1),encoding='utf-8',newline='\n')
    print('ACCEPTED',accepted,'selected',selected,flush=True)


if __name__=='__main__':main()
