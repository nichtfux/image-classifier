"""One 60-hour, restartable, validation-only 40px experiment. No Studio access.

STOP_LONG_RUN is graceful; checkpoints include optimizer, all RNGs and counters.
Outside sets, test splits, and the twelve developer gestures are never opened.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from balanced_validation import MODES, cohort, top3 as compact_top3
from candidate_runtime import build
from categories import CATEGORIES
from colour import NET
from colour_data import ColourData, inputs, cohort_metrics
from fill import fill_planes
from gacl import from_quant, quantize, QualityNet
from long_run_data import Data, pin_validation, GENERATED
from style_eval import augment as eval_brush
from train_next import load, save, OUT

ROOT = Path(__file__).resolve().parent.parent
RUN = OUT / 'long299_40_v4'
NAME = 'long299_40_v4'
STATUS = ROOT / 'notes/long-run-status.md'
SOURCE = 'balanced299_40_branch768_rehearsal_s10000_calibrated'
STOP = RUN / 'STOP_LONG_RUN'


def atomic(path, value, binary=False):
    path = Path(path); tmp = path.with_suffix(path.suffix+'.part')
    if binary:
        torch.save(value,tmp)
    else:
        tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8',newline='\n')
    for attempt in range(30):
        try:
            os.replace(tmp,path); return
        except PermissionError:
            time.sleep(.1)
    raise RuntimeError('Cannot replace '+str(path))


class Share:
    """Bound our active work, not FLUX's. No hard GPU-utilization claim."""
    def __init__(self,duty):
        self.duty=duty; self.active=0.; self.sleep=0.; self.start=time.perf_counter()
    def begin(self):
        self.mark=time.perf_counter()
    def end(self):
        torch.cuda.synchronize()
        used=time.perf_counter()-self.mark
        pause=min(1.,used*(1/self.duty-1))
        self.active+=used
        if pause>0:
            time.sleep(pause); self.sleep+=pause


def cpu_state(net):
    return {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}


def qnpz(path,q):
    temp=path.with_suffix('.part.npz')
    np.savez(temp,**{f'{key}{i}':v for i,row in enumerate(q) for key,v in zip(['w','s','b'],row)})
    os.replace(temp,path)


def metrics(y,top):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        result=cohort_metrics(y,top)
    real=y<270
    result['black_hole_rate']=float((top[real,0]==275).mean()) if real.any() else 0.
    for key in ['all','real','synthetic']:
        if not result[key]['drawings']:
            result[key]['top1']=result[key]['top3']=None
    result['custom_classes']={}
    for label in range(270,299):
        mask=y==label
        if mask.any():
            result['custom_classes'][CATEGORIES[label]]={
                'drawings':int(mask.sum()),'top1':float((top[mask,0]==label).mean()),
                'top3':float((top[mask]==label).any(1).mean())}
    return result


def forward_top(net,x,share,bs=32):
    values=[]
    for part in x.split(bs):
        share.begin()
        with torch.inference_mode():
            values.append(net(part).topk(3,1).indices.cpu().numpy())
        share.end()
    return np.concatenate(values)


def evaluate(net,data,share,phase):
    result={}
    for mode,seed in MODES:
        print('VALIDATION',phase,mode,flush=True)
        painted=mode in ['filled','painted']
        fixed=None if mode=='black' else data.fixed('policy' if painted else mode,seed)
        fills=data.fixed_fills(fixed,seed+300000) if painted else None
        top=[]
        for lo in range(0,len(data.y),512):
            hi=min(lo+512,len(data.y))
            chunk=data.chunk(lo,hi,device='cuda',fixed=fixed,black=mode=='black',fill=painted,fixed_fill=fills)
            x,_=inputs(chunk,black=mode=='black')
            if painted:
                generator=torch.Generator(device='cuda').manual_seed(seed+lo)
                rgb=torch.as_tensor(NET,device='cuda')[chunk[5].long()]
                if mode=='filled':
                    fill_planes(x,rgb,torch.ones(len(x),device='cuda',dtype=torch.bool),generator=generator,region=chunk[6])
                else:
                    x=eval_brush(x,chunk[6],rgb,generator=generator)
            top.append(forward_top(net,x,share))
            del x,chunk
        result[mode]=metrics(np.asarray(data.y),np.concatenate(top))
    for bank,path in [('qd-game',ROOT/'data/generated/rounds/balanced-v4-20261110/qd-game/validation.npz'),
                      ('generated',RUN/'generated-validation.npz')]:
        with np.load(path) as z:
            x,y=z['x'],z['y']
        for black in [False,True]:
            key=bank+('-black' if black else '')
            top=[]
            for lo in range(0,len(y),32):
                image=torch.from_numpy(x[lo:lo+32].astype(np.float32)/255).cuda()
                if black:image[:,1:]=0
                top.append(forward_top(net,image.contiguous(memory_format=torch.channels_last),share))
            result[key]=metrics(y,np.concatenate(top))
    return result


def reference_metrics(baseline,share):
    path=RUN/'references.json'
    if path.exists():
        result=json.loads(path.read_text())
        if 'custom_classes' not in result['installed']['filled']:
            for mode,_ in MODES:
                with np.load(OUT/'colour299_40-validation'/f'{mode}.npz') as z:
                    result['installed'][mode]=metrics(z['y'],compact_top3(z['z']))
            atomic(path,result)
        return result
    result={'installed':{},'previous_real_style':{}}
    for mode,_ in MODES:
        with np.load(OUT/'colour299_40-validation'/f'{mode}.npz') as z:
            result['installed'][mode]=metrics(z['y'],compact_top3(z['z']))
        # Apply the frozen previous offsets exactly (no reselection/calibration).
        checkpoint=torch.load(OUT/f'{SOURCE}.pt',map_location='cpu',weights_only=True)
        with np.load(OUT/'balanced299_40_branch768_rehearsal_s10000-validation'/f'{mode}.npz') as z:
            compact=z['z'].copy()
            compact[:,6:]-=np.asarray(checkpoint['calibration']['custom_offsets'])[None]
            result['previous_real_style'][mode]=cohort(compact,z['y'])
    for bank,bank_path in [('qd-game',ROOT/'data/generated/rounds/balanced-v4-20261110/qd-game/validation.npz'),
                      ('generated',RUN/'generated-validation.npz')]:
        with np.load(bank_path) as z:x,y=z['x'],z['y']
        for black in [False,True]:
            tops=[]
            for lo in range(0,len(y),32):
                image=torch.from_numpy(x[lo:lo+32].astype(np.float32)/255).cuda()
                if black:image[:,1:]=0
                tops.append(forward_top(baseline,image.contiguous(memory_format=torch.channels_last),share))
            result['installed'][bank+('-black' if black else '')]=metrics(y,np.concatenate(tops))
    atomic(path,result)
    return result


def guards(scores,reference):
    failures=[]; rows=[]; penalty=0.
    def minimum(label,value,limit,scale=.01):
        nonlocal penalty
        gap=max(0.,limit-value)
        penalty+=gap/scale
        rows.append({'guard':label,'value':value,'minimum':limit,'pass':gap<1e-10})
        if gap>1e-10:failures.append(label)
    def maximum(label,value,limit,denominator):
        nonlocal penalty
        gap=max(0.,value-limit)
        # Counts are exact on the same pinned drawings; no confidence-band waiver.
        penalty+=gap/max(1/denominator,.0001)
        rows.append({'guard':label,'value':value,'maximum':limit,'pass':gap<1e-10})
        if gap>1e-10:failures.append(label)
    for mode,base in reference['installed'].items():
        new=scores[mode]
        for group in ['real','synthetic']:
            if not base[group]['drawings']:continue
            for metric in ['top1','top3']:
                tolerance=.002
                limit=base[group][metric]-tolerance
                if group=='real' and mode in ['filled','painted']:
                    limit=max(base[group][metric]+.01,
                              reference['previous_real_style'][mode][group][metric]-.002)
                # Generated custom styles must improve on installed by >=5pp.
                if mode=='generated' and group=='synthetic':limit=base[group][metric]+.05
                minimum(f'{mode}/{group}/{metric}',new[group][metric],limit)
        if base['real']['drawings']:
            n=base['real']['drawings']
            maximum(f'{mode}/custom-false-alarm',new['false_alarms']['rate'],base['false_alarms']['rate'],n)
            maximum(f'{mode}/black-hole-false-alarm',new['black_hole_rate'],base['black_hole_rate'],n)
        if mode in ['black','filled','painted']:
            for label,old_class in base.get('custom_classes',{}).items():
                for metric in ['top1','top3']:
                    minimum(f'{mode}/{label}/{metric}',
                            new['custom_classes'][label][metric],old_class[metric]-.002)
    utility=float(np.mean([r[g]['top1'] for r in scores.values() for g in ['real','synthetic']
                           if r[g]['top1'] is not None]))
    return {'eligible':not failures,'failures':failures,'rows':rows,'penalty':penalty,
            'utility':utility,'rank':utility-penalty}


def export(net,q,scores,guard,step):
    # Each export is immutable until parity passes; then the pointer is replaced.
    tag=f'{NAME}_s{step:09d}'
    destination=RUN/'exports'/tag
    destination.mkdir(parents=True,exist_ok=True)
    qnpz(OUT/f'{tag}.npz',q)
    meta={'name':tag,'widths':[net.c1.out_channels,net.c2.out_channels,net.c3.out_channels],
          'fc':net.f1.out_features,'size':40,'input_channels':4,'params':sum(p.numel() for p in net.parameters()),
          'categories':CATEGORIES,'corpus':'expanded299-28','feature_normalize':False,
          'int8_top1':scores['black']['all']['top1'],'int8_top3':scores['black']['all']['top3'],
          'generated_data':{'generator':'FLUX.1-schnell','teacher':'SigLIP SO400M/384','licence':'Apache-2.0'},
          'installed':False,'quality_tiers':'parked','validation_guards':guard}
    atomic(OUT/f'{tag}.json',meta)
    # TRAIN-only vectors, spread across styles; no outside/dev/test file is opened.
    from long_run_data import rows
    picked=[]
    for style in ['marker','outline','flat','painted','kids']:
        row=next((r for r in rows() if r['split']=='train' and r['style']==style),None)
        if row:
            geometry=json.loads((GENERATED/'samples'/(row['id']+'.json')).read_text())
            picked.append({'label':int(row['label']),'drawing':geometry['strokes'],
                           'colours':[c-1 for c in geometry['colours']]})
    assert len(picked)>=3
    vectors=destination/'train-only-vectors.json'; atomic(vectors,picked)
    subprocess.run([sys.executable,str(ROOT/'training/export.py'),'--name',tag,'--samples','0',
                    '--out-dir',str(destination),'--extra-vectors',str(vectors)],check=True)
    build(destination)
    from offline_native import run
    parity=run(destination,tag)
    cost=parity['native']['FORWARD'][1]  # p95, measured offline native CLI
    passed=cost<=100
    atomic(destination/'parity.json',{'runtime':parity,'native_p95_ms':cost,
            'within_native_budget':passed,'fresh_studio_measurement':False})
    atomic(destination/'guards.json',guard)
    save(net,destination/'checkpoint.pt',{'phase':'unattended-v4','quality_tier':'parked','step':step})
    shutil.copyfile(OUT/f'{tag}.npz',destination/'weights-int8.npz')
    atomic(destination/'model.json',meta)
    return {'path':str(destination),'step':step,'guard_eligible':guard['eligible'],
            'native_p95_ms':cost,'within_native_budget':passed,'parity_passed':True,
            'rank':guard['rank'],'utility':guard['utility'],
            'scores':{mode:{'real':s['real'],'custom':s['synthetic'],
                           'custom_false_alarm':s['false_alarms']['rate'],
                           'black_hole_false_alarm':s['black_hole_rate']} for mode,s in scores.items()}}


def status(state,phase,share=None):
    elapsed=state.get('elapsed',0)
    remaining=max(0.,state.get('hours',60)*3600-elapsed)
    current=state.get('last_guards',{})
    lines=['# Unattended 40px training', '',f'Updated UTC: {time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime())}',
           f'Phase: {phase}. PID: {os.getpid()}. No auto-install; Studio is untouched.',
           f'Elapsed: {elapsed/3600:.2f} h / {state.get("hours",60):.0f} h. ETA: {remaining/3600:.2f} h remaining.',
           f'Step: {state.get("step",0)}. Examples: {state.get("seen",0):,}. '
           f'Equivalent epoch: {state.get("seen",0)/27046888:.4f}. Resume count: {state.get("resumes",0)}.',
           f'Generated v4 train rows: {state.get("bank_count",0):,}. Refresh every 15 minutes; prompt groups kept apart.',
           'GPU share: training work capped at 40% duty, 60% voluntary gaps; memory cap 9% (~1.1 GB). '
           'This is a work/sleep throttle, not a hardware utilization guarantee. FLUX remains running.',
           f'Last checkpoint: `{RUN / "last.pt"}`.',
           f'Best passing export: {(state.get("best_passing") or {}).get("path","none yet")}.',
           f'Best provisional export: {(state.get("best_provisional") or {}).get("path","not exported yet")}.',
           f'Current guards: {len(current.get("failures",[])) if current else "not evaluated"} failed. Full QD validation: 299,000 drawings × 5 modes; '
           'plus pinned generated and converted-QD validation, colour and black counterfactuals.',
           '']
    best=state.get('best_passing') or state.get('best_provisional') or {}
    if best.get('scores'):
        lines+=['Best-so-far validation scores (provisional until every guard passes):',
                '| Style | Real top-1 / top-3 | Custom top-1 / top-3 | Custom FA | Black-hole FA |',
                '|---|---:|---:|---:|---:|']
        def pair(row):
            return '-' if row['top1'] is None else f'{100*row["top1"]:.2f}% / {100*row["top3"]:.2f}%'
        for mode,s in best['scores'].items():
            lines.append(f'| {mode} | {pair(s["real"])} | {pair(s["custom"])} | '
                         f'{s["custom_false_alarm"]*100:.4f}% | {s["black_hole_false_alarm"]*100:.4f}% |')
    lines+=['','Current checkpoint guards:','| Guard | Score | Required | Pass |','|---|---:|---:|---|']
    for row in current.get('rows',[]):
        constraint=('≥ '+f'{row["minimum"]*100:.3f}%') if 'minimum' in row else ('≤ '+f'{row["maximum"]*100:.3f}%')
        lines.append(f'| {row["guard"]} | {row["value"]*100:.3f}% | {constraint} | {row["pass"]} |')
    lines+=['',f'Stop gracefully: create `{STOP}`. Restart: `training/restart-long-run.ps1`.',
            'STOP prevents restart. Remove it explicitly before resuming. No Scheduled Task or startup entry is installed.',
            'Outside sets and developer gestures remain final tests only. Quality tiers remain parked.',
            'Training: recognized Quick, Draw! (CC BY 4.0), own procedural generation, and marked FLUX/SigLIP v4 data only.']
    tmp=STATUS.with_suffix('.part.md'); tmp.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    os.replace(tmp,STATUS)
    atomic(RUN/'progress.json',{**state,'phase':phase,'pid':os.getpid(),'updated':time.time()})


def checkpoint(net,opt,rng,state,validation,share,reference,first=False):
    def payload():
        return {'format':1,'state_dict':cpu_state(net),'optimizer':opt.state_dict(),
                'numpy_rng':rng.bit_generator.state,'python_rng':random.getstate(),
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),
                'state':copy.deepcopy(state),'source':SOURCE,'categories':CATEGORIES,
                'widths':[48,96,192],'fc':768,'size':40,'cosine':False}
    # Recovery copy exists before the lengthy validation/export stage begins.
    state['validation_complete']=False
    state['elapsed']+=time.monotonic()-state.pop('_mark')
    atomic(RUN/'pending.pt',payload(),True)
    status(state,'full permitted validation',share)
    before=time.monotonic()
    q=quantize(net)
    evalnet=from_quant(q,cosine=False)
    scores=evaluate(evalnet,validation,share,f'step {state["step"]}')
    del evalnet
    guard=guards(scores,reference)
    state['last_scores']=scores;state['last_guards']=guard
    state['guard_version']=2
    atomic(RUN/f'validation-s{state["step"]:09d}.json',{'scores':scores,'guards':guard})
    current=state.get('best_provisional')
    passing=state.get('best_passing')
    better=current is None or guard['rank']>current['rank']
    eligible=guard['eligible'] and (passing is None or guard['utility']>passing['utility'])
    if better or eligible:
        artifact=export(net,q,scores,guard,state['step'])
        if better:
            state['best_provisional']=artifact
            atomic(RUN/'best-so-far.json',artifact)
        if eligible and artifact['within_native_budget']:
            state['best_passing']=artifact
            atomic(RUN/'best-passing.json',artifact)
    state['elapsed']+=time.monotonic()-before
    state['validation_complete']=True
    last=RUN/'last.pt'
    if last.exists():shutil.copyfile(last,RUN/'previous.pt')
    atomic(last,payload(),True)
    # Read back full checkpoint, not only a state-dict or export.
    restored=torch.load(last,map_location='cpu',weights_only=False)
    assert restored['state']['step']==state['step']
    assert restored['optimizer']['state'] and restored['numpy_rng']==rng.bit_generator.state
    for key,value in net.state_dict().items():
        assert torch.equal(value.cpu(),restored['state_dict'][key]),key
    atomic(RUN/'checkpoint-verified.json',{'step':state['step'],'sha256':hashlib.sha256(last.read_bytes()).hexdigest(),
            'optimizer_slots':len(restored['optimizer']['state']),'rng_verified':True})
    status(state,'training (checkpoint verified)',share)
    state['_mark']=time.monotonic()
    print('CHECKPOINT',state['step'],'failed guards',len(guard['failures']),'elapsed hours',state['elapsed']/3600,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--hours',type=float,default=60.)
    parser.add_argument('--first-leg',action='store_true')
    parser.add_argument('--check-minutes',type=float,default=60.)
    args=parser.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    # A held Windows byte-range lock prevents duplicate trainers.
    lock=(RUN/'worker.lock').open('a+b');lock.seek(0)
    import msvcrt
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    if STOP.exists():return
    torch.set_num_threads(3);torch.backends.cudnn.benchmark=False
    torch.cuda.set_per_process_memory_fraction(.09)
    torch.manual_seed(2026111101);random.seed(2026111101)
    rng=np.random.default_rng(2026111101)
    share=Share(.40)
    state={'step':0,'seen':0,'elapsed':0.,'hours':args.hours,'resumes':0}
    teachers=RUN/'teachers';teachers.mkdir(exist_ok=True)
    frozen_source=teachers/'previous-style-model.pt'
    frozen_installed=teachers/'installed-colour299_40.npz'
    for original,destination in [(OUT/f'{SOURCE}.pt',frozen_source),
                                 (OUT/'colour299_40.npz',frozen_installed)]:
        if not destination.exists():
            temporary=destination.with_suffix(destination.suffix+'.part')
            shutil.copyfile(original,temporary);os.replace(temporary,destination)
    atomic(teachers/'sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [frozen_source,frozen_installed]})
    net,source=load(frozen_source)
    net.requires_grad_(True).eval()
    groups=[{'params':[p for n,p in net.named_parameters() if not n.startswith('f2.')], 'lr':1e-5},
            {'params':list(net.f2.parameters()),'lr':8e-5}]
    opt=torch.optim.AdamW(groups,weight_decay=1e-5)
    resume=None
    for path in [RUN/'pending.pt',RUN/'last.pt',RUN/'previous.pt']:
        if path.exists():
            try:
                candidate=torch.load(path,map_location='cpu',weights_only=False)
                if resume is None or candidate['state']['step']>=resume['state']['step']:resume=candidate
            except Exception as error:print('Ignoring damaged checkpoint',path,error,flush=True)
    if resume:
        net.load_state_dict(resume['state_dict']);opt.load_state_dict(resume['optimizer'])
        state=resume['state'];state['resumes']+=1
        rng.bit_generator.state=resume['numpy_rng'];random.setstate(resume['python_rng'])
        torch.set_rng_state(resume['torch_rng']);torch.cuda.set_rng_state(resume['cuda_rng'])
        assert state['step']>0 and len(opt.state)>0
        atomic(RUN/'resume-proof.json',{'resumed_step':state['step'],'resume_count':state['resumes'],
               'optimizer_slots':len(opt.state),'numpy_rng_verified':rng.bit_generator.state==resume['numpy_rng'],
               'torch_rng_verified':torch.equal(torch.get_rng_state(),resume['torch_rng']),
               'cuda_rng_verified':torch.equal(torch.cuda.get_rng_state().cpu(),resume['cuda_rng'].cpu()),
               'pid':os.getpid(),'utc':time.time()})
        print('RESUMED',state['step'],'optimizer slots',len(opt.state),flush=True)
    else:
        # Keep the previous real-style gains as an explicit validation/teacher target.
        atomic(RUN/'protocol.json',{'hours':args.hours,'source':SOURCE,'all_parameters_trainable':True,
              'frozen_batchnorm_statistics':True,'colour_strength':'learned, no fixed post-training attenuation',
              'style_rehearsal':'explicit custom outline, colour, fill, brush every batch; no text-to-blob fills',
              'teacher':'previous candidate for real; installed for correct custom; frozen SigLIP soft targets',
              'selection':'QD full validation + pinned generated prompt validation only; outside and dev final-only',
              'guards':'black/policy/random within .2pp; real filled/brush retain previous gains within .2pp; '
                       'custom black/filled/brush within .2pp installed; no increased custom/BH FA on every cohort',
              'gpu_duty':.40,'memory_fraction':.09,'native_p95_budget_ms':100,'auto_install':False})
    data=Data(RUN)
    state['bank_count']=data.bank.count
    validation=ColourData('validation','colour299-40')
    with np.load(frozen_installed) as z:
        installed=from_quant([(z[f'w{i}'],z[f's{i}'],z[f'b{i}']) for i in range(5)],cosine=False)
    prior,_=load(frozen_source);prior.requires_grad_(False).eval()
    reference=reference_metrics(installed,share)
    # Constructing frozen teachers consumes Torch RNG. The training stream must
    # resume from its saved state after setup, not from those initializations.
    if resume:
        rng.bit_generator.state=resume['numpy_rng'];random.setstate(resume['python_rng'])
        torch.set_rng_state(resume['torch_rng']);torch.cuda.set_rng_state(resume['cuda_rng'])
    sources=RUN/'source-code'
    sources.mkdir(exist_ok=True)
    hashes={}
    for filename in ['long_run.py','long_run_data.py','long_run_supervisor.py','colour.py','colour_batch.py',
                     'fill.py','style_augment.py','style_eval.py','gacl.py','export.py','candidate_runtime.py']:
        source_path=ROOT/'training'/filename
        hashes[filename]=hashlib.sha256(source_path.read_bytes()).hexdigest()
        shutil.copyfile(source_path,sources/filename)
    atomic(RUN/'source-hashes.json',hashes)
    state['_mark']=time.monotonic()
    last_check=last_refresh=last_status=time.monotonic()
    if resume and state.get('guard_version',0)<2:
        # The initial smoke leg predates per-class custom guards. Do not compare
        # its aggregate-only provisional rank with the now stricter protocol.
        state.pop('best_provisional',None);state.pop('best_passing',None)
        state['validation_complete']=False
    if resume and not state.get('validation_complete',False):
        checkpoint(net,opt,rng,state,validation,share,reference)
    status({k:v for k,v in state.items() if k!='_mark'},'training',share)
    while state['elapsed']+time.monotonic()-state['_mark']<state['hours']*3600:
        if STOP.exists():
            checkpoint(net,opt,rng,state,validation,share,reference)
            status({k:v for k,v in state.items() if k!='_mark'},'stopped gracefully',share)
            return
        if time.monotonic()-last_refresh>900:
            changed=data.bank.refresh();state['bank_count']=data.bank.count
            state['bank_signature']=data.bank.signature
            print('BANK_REFRESH',data.bank.count,changed,flush=True)
            last_refresh=time.monotonic()
        x,y,start,teacher=data.batch(rng,32)
        share.begin()
        opt.zero_grad(set_to_none=True)
        logits=net(x)
        loss=F.cross_entropy(logits,y)
        real=y<270;custom=~real
        # Margin penalties are relative to the true class, not all-real logit size.
        correct=logits.gather(1,y[:,None])[:,0]
        loss+=.6*F.softplus(logits[real,270:].amax(1)-correct[real]+1.).mean()
        loss+=.25*F.softplus(logits[custom,:270].amax(1)-correct[custom]+1.).mean()
        with torch.no_grad():
            target_real=prior(x[real])
            target_custom=installed(x[custom])
        temperature=3.
        loss+=.7*temperature**2*F.kl_div(F.log_softmax(logits[real]/temperature,1),
                    F.softmax(target_real/temperature,1),reduction='batchmean')
        usable=target_custom.argmax(1)==y[custom]
        if usable.any():
            loss+=1.5*temperature**2*F.kl_div(F.log_softmax(logits[custom][usable]/temperature,1),
                    F.softmax(target_custom[usable]/temperature,1),reduction='batchmean')
        # Only accepted, prompt-disjoint generated training rows are distilled.
        loss+=.10*F.kl_div(F.log_softmax(logits[start:]/2.,1),F.softmax(teacher/2.,1),reduction='batchmean')*4
        if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),3.)
        opt.step();share.end()
        state['step']+=1;state['seen']+=len(y);state['last_loss']=float(loss.detach())
        del x,y,logits,loss
        if state['step']%100==0:
            print('TRAIN',state['step'],'loss',state['last_loss'],'bank',data.bank.count,flush=True)
        due=(args.first_leg and state['step']>=32) or time.monotonic()-last_check>args.check_minutes*60
        if due:
            checkpoint(net,opt,rng,state,validation,share,reference)
            last_check=time.monotonic()
            if args.first_leg:
                atomic(RUN/'first-leg-complete.json',{'step':state['step'],'pid':os.getpid()})
                sys.exit(75)  # supervisor proves a real checkpoint -> process restart
        if time.monotonic()-last_status>60:
            view={k:v for k,v in state.items() if k!='_mark'}
            view['elapsed']+=time.monotonic()-state['_mark']
            status(view,'training',share);last_status=time.monotonic()
    checkpoint(net,opt,rng,state,validation,share,reference)
    status({k:v for k,v in state.items() if k!='_mark'},'finished 60-hour training; awaiting human review',share)
    atomic(RUN/'DONE.json',{'step':state['step'],'elapsed':state['elapsed'],'pid':os.getpid()})


if __name__=='__main__':
    try:main()
    except Exception:
        RUN.mkdir(parents=True,exist_ok=True)
        (RUN/'last-error.txt').write_text(traceback.format_exc(),encoding='utf-8',newline='\n')
        raise
