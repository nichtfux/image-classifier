"""Matched continued-control / generated-hard / generated+SigLIP experiments.

Frozen generated IDs before training. All branches start from the same selected
GACL control and use the same QD batches, update count and optimiser settings.
No outside test input is opened by this trainer.
"""
import argparse,json,math,time,copy,hashlib
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from train_next import load,save,OUT
from gacl import loss as gacl_loss
from colour_data import ColourData,inputs
from colour import NET
from style_augment import augment
from generate_dataset import read_rows,ROOT
from generated_splits import finalize,group_key,METHOD


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=1500);ap.add_argument('--bs',type=int,default=256)
    ap.add_argument('--gen-per-batch',type=int,default=16)
    ap.add_argument('--only-distill',action='store_true',help='rerun corrected teacher branch on the exact frozen pilot IDs')
    a=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    frozen=json.loads((ROOT/'pilot-training-snapshot.json').read_text()) if a.only_distill else None
    if frozen:
        accepted=set(frozen['all_kept_ids']);all_kept=[r for r in read_rows() if r['id'] in accepted]
        assert all(r['keep'] for r in all_kept) and len(all_kept)==len(accepted)
    else:
        all_kept=[r for r in finalize(ROOT) if r['keep']]
        (ROOT/'pilot-generation-status.json').write_bytes((ROOT/'generation-status.json').read_bytes())
    rows=[r for r in all_kept if r['split']=='train']
    if len(rows)<20:raise RuntimeError('Too few kept training images for a meaningful first experiment (<20).')
    ids=[r['id'] for r in rows]
    manifest={'generated':True,'rows':ids,'validation_ids':[r['id'] for r in all_kept if r['split']=='validation'],
              'test_ids':[r['id'] for r in all_kept if r['split']=='test'],'all_kept_ids':[r['id'] for r in all_kept],
              'sha256':hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
              'source':'FLUX.1-schnell, frozen zero-shot SigLIP filter; train split only','args':vars(a)}
    groups={split:{group_key(r) for r in all_kept if r['split']==split} for split in ['train','validation','test']}
    assert not (groups['train']&groups['validation'] or groups['train']&groups['test'] or groups['validation']&groups['test'])
    manifest['split_method']=METHOD;manifest['prompt_groups']={k:sorted(v) for k,v in groups.items()}
    if frozen:
        assert ids==frozen['rows'] and manifest['test_ids']==frozen['test_ids'] and manifest['validation_ids']==frozen['validation_ids']
        manifest=frozen
    else:(ROOT/'pilot-training-snapshot.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8',newline='\n')
    images=[];labels=[];targets=[]
    for row in rows:
        with np.load(ROOT/row['sample']) as z:
            images.append(z['x']);labels.append(int(z['y']));targets.append(z['teacher_logits'])
    gx=torch.from_numpy(np.stack(images)).cuda().float()/255
    gy=torch.tensor(labels,device='cuda')
    gz=torch.from_numpy(np.stack(targets).astype(np.float32)).cuda()
    group_counts=Counter((r['label'],r['style']) for r in rows)
    styles_per_class=Counter(ci for ci,style in group_counts)
    real_classes=sum(ci<270 for ci in styles_per_class);custom_classes=sum(ci>=270 for ci in styles_per_class)
    probabilities=[]
    for row in rows:
        ci=row['label'];mass=(.2/max(1,real_classes)) if ci<270 else (.8/max(1,custom_classes))
        probabilities.append(mass/styles_per_class[ci]/group_counts[(ci,row['style'])])
    gw=torch.tensor(probabilities,device='cuda');gw/=gw.sum()
    del images,labels,targets
    source=OUT/'gacl299_40_wide.pt'
    control,ck=load(source);control.eval().requires_grad_(False)
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    protocol={'args':vars(a),'base_checkpoint_sha256':source_hash,'seed':20261110,
              'split_method':METHOD,'training_sources':{},'outside_test_files_opened':False}
    protocol['generated_sampling']='80% custom / 20% real, uniform classes within each cohort, uniform available styles within each class'
    protocol['teacher_reduction']='KL averaged over generated examples, then multiplied by generated_count / full_batch_count'
    protocol['teacher_full_batch_coefficient']=.25*a.gen_per_batch/a.bs
    if frozen:protocol['unchanged_control_hard_protocol']='experiments/unscaled-teacher-pilot/ablation-protocol.json'
    for filename in ['train_generated_ablation.py','generated_splits.py','gacl.py','style_augment.py','colour_data.py','colour_batch.py','colour.py','fill.py']:
        protocol['training_sources'][filename]=hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest()
    (ROOT/'ablation-protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8',newline='\n')
    data=ColourData('train','colour299-40');palette=torch.as_tensor(NET,device='cuda')
    procedural_path=OUT.parent.parent/'data/synth-diverse40/train.npz'
    with np.load(procedural_path) as z:procedural={k:torch.from_numpy(z[k]).cuda() for k in ['x','y','region','fill']}
    protocol['procedural_source_sha256']=hashlib.sha256(procedural_path.read_bytes()).hexdigest()
    protocol['common_procedural_per_batch']=16
    (ROOT/'ablation-protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8',newline='\n')
    experiments=[('gacl299_40_continued','control'),('gacl299_40_genhard','hard'),('gacl299_40_genkd','distill')]
    history=[]
    if a.only_distill:
        experiments=[experiments[-1]]
        history=[r for r in json.loads((ROOT/'ablation-training.json').read_text()) if r['name']!='gacl299_40_genkd']
    for name,mode in experiments:
        torch.manual_seed(20261110);rng=np.random.default_rng(20261110)
        net,_=load(source);net.train()
        for module in net.modules():
            if isinstance(module,(nn.BatchNorm2d,nn.Dropout)):module.eval()
        opt=torch.optim.AdamW(net.parameters(),lr=.00002,weight_decay=.0001)
        step=0;started=time.time();total=0.;last=started
        while step<a.steps:
            lo=int(rng.integers(0,len(data.y)-50000));hi=lo+50000
            chunk=data.chunk(lo,hi,seed=30261110+step,device='cuda',fill=True)
            permutation=torch.randperm(hi-lo,device='cuda')
            for pos in range(0,len(permutation)-a.bs,a.bs):
                ix=permutation[pos:pos+a.bs];x,y=inputs(chunk,ix)
                black=torch.zeros_like(x[:64]);black[:,0]=x[:64,0]
                x=augment(x,chunk[6][ix],palette[chunk[5][ix].long()])
                # Retain the control's broader handwritten custom programs in
                # every branch; generated data must not erase that replay mix.
                sj=torch.randint(len(procedural['y']),(16,),device='cuda')
                sx=augment(procedural['x'][sj].float()/255,procedural['region'][sj].bool(),palette[procedural['fill'][sj].long()])
                x[-a.gen_per_batch-16:-a.gen_per_batch]=sx
                y=y.clone();y[-a.gen_per_batch-16:-a.gen_per_batch]=procedural['y'][sj].long()
                # Random stream advances identically even for the continued control.
                j=torch.multinomial(gw,a.gen_per_batch,replacement=True)
                if mode!='control':
                    x[-a.gen_per_batch:]=gx[j];y=y.clone();y[-a.gen_per_batch:]=gy[j]
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    z,norm=net(x,details=True);loss,_=gacl_loss(z.float(),norm,y,net.norm_scale)
                    bz,bnorm=net(black,details=True)
                    with torch.no_grad():tz=control(black).float()
                    T=2.;loss=loss+.7*T*T*F.kl_div(F.log_softmax(bz.float()/T,1),F.softmax(tz/T,1),reduction='batchmean')
                    loss=loss+.25*F.cross_entropy(bz.float(),chunk[1][ix[:64]])
                    anchor,_=gacl_loss(bz.float(),bnorm,chunk[1][ix[:64]],net.norm_scale);loss=loss+.15*anchor
                    real=y<270
                    if bool(real.any()):
                        target=z[real].gather(1,y[real,None]).squeeze(1)
                        loss=loss+.08*F.softplus(z[real,275]-target+2).mean()
                    if mode=='distill':
                        soft=F.softmax(gz[j]/2.,1)
                        loss=loss+.25*(a.gen_per_batch/a.bs)*4*F.kl_div(F.log_softmax(z[-a.gen_per_batch:].float()/2.,1),soft,reduction='batchmean')
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step()
                total+=float(loss.detach());step+=1
                for group in opt.param_groups:group['lr']=.00002*(.25+.75*.5*(1+math.cos(math.pi*step/a.steps)))
                if time.time()-last>45:
                    print(name,'step',step,'/',a.steps,'loss',total/step,'seconds',time.time()-started,flush=True);last=time.time()
                    save(net,OUT/(name+'-last.pt'),{'phase':mode,'step':step,'generated_snapshot':manifest['sha256']})
                if step>=a.steps:break
        net.eval();info={'phase':mode,'step':step,'generated_snapshot':manifest['sha256'],'generated_train_images':len(rows),
                        'base_checkpoint_sha256':source_hash,'seconds':time.time()-started,'distillation':{'teacher':'frozen zero-shot SigLIP SO400M','weight':.25,'temperature':2.,'scope':'converted generated inputs only','full_batch_coefficient':.25*a.gen_per_batch/a.bs} if mode=='distill' else None}
        save(net,OUT/(name+'.pt'),info);history.append({'name':name,**info});del net;torch.cuda.empty_cache()
        (ROOT/'ablation-training.json').write_text(json.dumps(history,indent=2),encoding='utf-8',newline='\n')
        print('BRANCH DONE',name,info,flush=True)
    print('ABLATION TRAINING DONE',flush=True)


if __name__=='__main__':main()
