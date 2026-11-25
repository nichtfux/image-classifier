"""Queued after normal 40px completion. Validation-only teacher calibration."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import torch

import balanced_validation as validation
from balanced_validation import top3
from constrained_bias import fit_offsets
from gacl import from_quant
from long_run import atomic, guards, metrics, qnpz, Share
from train_next import load, save

ROOT=Path(__file__).resolve().parent.parent
OLD=ROOT/'training/out/long299_40_v4'
DEST=ROOT/'training/out/round64/post40'


def main():
    assert (OLD/'DONE.json').exists() or (OLD/'HANDOFF_TO64.json').exists(), '40px training has no completed checkpoint handoff'
    if (DEST/'DONE.json').exists():return
    DEST.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.cuda.set_per_process_memory_fraction(.09)
    share=Share(.40)
    original_compact=validation.compact
    def paced_compact(model,images):
        values=[]
        for batch in images.split(32):
            if (ROOT/'training/out/round64/STOP').exists():
                raise InterruptedError('64px queue stopped during calibration')
            share.begin()
            values.append(original_compact(model,batch))
            share.end()
        return np.concatenate(values)
    validation.compact=paced_compact
    selected=json.loads((OLD/'best-passing.json' if (OLD/'best-passing.json').exists()
                         else OLD/'best-so-far.json').read_text())
    artifact=Path(selected['path'])
    net,checkpoint=load(artifact/'checkpoint.pt')
    with np.load(artifact/'weights-int8.npz') as z:
        arrays=[(z[f'w{i}'],z[f's{i}'],z[f'b{i}']) for i in range(5)]
    workspace=DEST/'cohorts'
    (workspace/'qd-game').mkdir(parents=True,exist_ok=True)
    (workspace/'generated').mkdir(exist_ok=True)
    shutil.copyfile(ROOT/'data/generated/rounds/balanced-v4-20261110/qd-game/validation.npz',workspace/'qd-game/validation.npz')
    shutil.copyfile(OLD/'generated-validation.npz',workspace/'generated/validation.npz')
    meta=json.loads((OLD/'generated-validation.json').read_text())
    atomic(workspace/'generated/manifest.json',{'source_rows':meta['rows']})
    validation.ROUND=workspace
    baseline_name='round64_post40_baseline'
    candidate_name='round64_post40_candidate'
    with np.load(ROOT/'training/out/colour299_40.npz') as z:
        baseline=from_quant([(z[f'w{i}'],z[f's{i}'],z[f'b{i}']) for i in range(5)],cosine=False)
    baseline_path=validation.cache(baseline,baseline_name)
    candidate_path=validation.cache(from_quant(arrays,cosine=False),candidate_name)
    keys=[p.stem for p in sorted(candidate_path.glob('*.npz'))]
    offsets,solver=fit_offsets(baseline_path,candidate_path,keys,top3)
    if offsets is None:offsets=np.zeros(29,np.float32)
    references={'installed':{},'previous_real_style':{}}
    raw_guard_scores={};calibrated_guard_scores={}
    raw_full={};calibrated_full={}
    for key in keys:
        with np.load(baseline_path/f'{key}.npz') as b, np.load(candidate_path/f'{key}.npz') as c:
            use=c['role']==1
            references['installed'][key]=metrics(b['y'][use],top3(b['z'][use]))
            raw_guard_scores[key]=metrics(c['y'][use],top3(c['z'][use]))
            calibrated_guard_scores[key]=metrics(c['y'][use],top3(c['z'][use],offsets))
            raw_full[key]=metrics(c['y'],top3(c['z']))
            calibrated_full[key]=metrics(c['y'],top3(c['z'],offsets))
    prior=torch.load(ROOT/'training/out/balanced299_40_branch768_rehearsal_s10000_calibrated.pt',map_location='cpu',weights_only=True)
    for mode,_ in validation.MODES:
        with np.load(ROOT/'training/out/balanced299_40_branch768_rehearsal_s10000-validation'/f'{mode}.npz') as z:
            use=z['role']==1
            references['previous_real_style'][mode]=metrics(z['y'][use],top3(z['z'][use],np.asarray(prior['calibration']['custom_offsets'])))
    raw_guard=guards(raw_guard_scores,references)
    calibrated_guard=guards(calibrated_guard_scores,references)
    # Retain the 40px round's improved custom shapes as well as legacy guards.
    retained=True
    for mode in ['black','filled','painted']:
        for label,old in raw_guard_scores[mode]['custom_classes'].items():
            for score in ['top1','top3']:
                if calibrated_guard_scores[mode]['custom_classes'][label][score]<old[score]-.002:
                    retained=False
    use_calibrated=retained and calibrated_guard['rank']>raw_guard['rank']
    if not use_calibrated:offsets=np.zeros(29,np.float32)
    with torch.no_grad():net.f2.bias[270:]-=torch.as_tensor(offsets,device='cuda')
    arrays[4]=(arrays[4][0],arrays[4][1],arrays[4][2].copy())
    arrays[4][2][270:]-=offsets
    save(net,DEST/'teacher.pt',{'phase':'post40-calibration','source':str(artifact),
                              'custom_offsets':offsets.tolist(),'quality_tier':'parked'})
    qnpz(DEST/'teacher-weights.npz',arrays)
    atomic(DEST/'teacher-scores.json',calibrated_full if use_calibrated else raw_full)
    chosen_guard_scores=calibrated_guard_scores if use_calibrated else raw_guard_scores
    atomic(DEST/'guard-scores.json',chosen_guard_scores)
    fallback=DEST/'fallback-export'
    fallback.mkdir(exist_ok=True)
    original_meta=json.loads((artifact/'model.json').read_text(encoding='utf-8'))
    fallback_name=original_meta['name']+'_post40'
    qnpz(ROOT/'training/out'/(fallback_name+'.npz'),arrays)
    fallback_meta={**original_meta,'name':fallback_name,'installed':False,
                   'checkpoint':str(DEST/'teacher.pt'),'source':str(artifact),
                   'calibration':{'custom_offsets':offsets.tolist(),'independent_guard_half':True}}
    atomic(ROOT/'training/out'/(fallback_name+'.json'),fallback_meta)
    atomic(fallback/'model.json',fallback_meta)
    shutil.copyfile(DEST/'teacher.pt',fallback/'checkpoint.pt')
    shutil.copyfile(DEST/'teacher-weights.npz',fallback/'weights-int8.npz')
    subprocess.run([sys.executable,str(ROOT/'training/export.py'),'--name',fallback_name,
                    '--samples','0','--extra-vectors',str(artifact/'train-only-vectors.json'),
                    '--out-dir',str(fallback)],check=True)
    from candidate_runtime import build
    from offline_native import run
    build(fallback)
    parity=run(fallback,fallback_name)
    atomic(fallback/'parity.json',parity)
    subprocess.run([sys.executable,str(ROOT/'training/package_studio_model.py'),
                    '--export',str(fallback),'--out',str(DEST/'fallback-studio')],check=True)
    atomic(DEST/'DONE.json',{'source':str(artifact),'calibrated':use_calibrated,
           'offsets':offsets.tolist(),'solver':solver,'retained_custom_styles':retained,
           'raw_guards':raw_guard,'calibrated_guards':calibrated_guard,
           'teacher':str(DEST/'teacher.pt'),'fallback_export':str(fallback),
           'fallback_name':fallback_name,'outside_sets_opened':False})


if __name__=='__main__':main()
