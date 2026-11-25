"""Frozen pilot generated test only: teacher-filtered upper bound, not humans."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from generate_dataset import ROOT,read_rows
from evaluate_next import load,write,OUT
from external_eval import metrics
from generated_splits import assign,group_key

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda',choices=['cuda','cpu']);args=ap.parse_args()
    torch.set_num_threads(3)
    snapshot=json.loads((ROOT/'pilot-training-snapshot.json').read_text())
    ids=set(snapshot['test_ids']);rows=[]
    with (ROOT/'manifest.jsonl').open(encoding='utf-8') as f:
        for line in f:
            if not line.endswith('\n'):break
            row=json.loads(line)
            if row['id'] in ids:rows.append(row)
            if len(rows)==len(ids):break
    assert len(rows)==len(ids),'Incomplete frozen generated hold-out'
    assert all(r['split']=='test' and r['id'] not in snapshot['rows'] for r in rows)
    assert all(assign(r)=='test' and group_key(r) not in snapshot['prompt_groups']['train'] for r in rows)
    if not rows:raise RuntimeError('No held-out generated images yet')
    images=[];labels=[]
    for row in rows:
        with np.load(ROOT/row['sample']) as z:images.append(z['x']);labels.append(int(z['y']))
    x=torch.from_numpy(np.stack(images)).float().to(args.device)/255
    black=x.clone();black[:,1:]=0
    y=np.asarray(labels,np.int16)
    path=OUT.parent.parent/'notes/training-round-next.json';report=json.loads(path.read_text())
    for name in ['colour299_40','gacl299_40_wide','gacl299_40_continued','gacl299_40_genhard','gacl299_40_genkd']:
        net,_=load(name,args.device)
        for key,inputs in [('generated_holdout',x),('generated_holdout_black',black)]:
            top=[]
            with torch.inference_mode():
                for lo in range(0,len(inputs),128):top.append(net(inputs[lo:lo+128].contiguous(memory_format=torch.channels_last)).topk(3,1).indices.cpu().numpy())
            result=metrics(y,np.concatenate(top),np.ones(len(y),bool))
            result['caveat']='Tiny prompt-disjoint generated hold-out, selected by the same frozen teacher on original previews. Black counterfactual zeros RGB only. Upper bound; not human/player accuracy.'
            report[name][key]=result;print(name,key,result['all'],flush=True)
        del net
        if args.device=='cuda':torch.cuda.empty_cache()
    write(path,report)
