"""Select int8 candidates on permitted validation only, before outside testing."""
import argparse,json,time,shutil,os
from pathlib import Path
import numpy as np
import torch
from colour_data import ColourData
from train_next import load,OUT
from gacl import quantize,from_quant
from train_colour import evaluate
from evaluate_next import procedural


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',default='gacl299_40_wide');a=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    data=ColourData('validation','colour299-40')
    with np.load(OUT/'colour299_40.npz') as z:q=[(z[f'w{i}'],z[f's{i}'],z[f'b{i}']) for i in range(5)]
    baseline_net=from_quant(q,cosine=False)
    baseline={m:evaluate(baseline_net,data,m,s,'cuda',batch=512) for m,s in [('black',100),('filled',20261110)]}
    baseline['diverse']=procedural(baseline_net,'validation');del baseline_net
    candidates=[]
    progress=json.loads((OUT/f'{a.name}-progress.json').read_text())
    allowed={r['phase'] for r in progress['history']}
    for phase in ['gacl1','gacl2','gacl3']:
        path=OUT/f'{a.name}-{phase}.pt'
        if phase not in allowed or not path.exists():continue
        net,ck=load(path);q=quantize(net);qn=from_quant(q,cosine=net.cosine,norm_scale=float(net.norm_scale));del net
        result={m:evaluate(qn,data,m,s,'cuda',batch=512) for m,s in [('black',100),('filled',20261110)]}
        result['diverse']=procedural(qn,'validation')
        guard=all(result['black'][g][k]>=baseline['black'][g][k]-.002 for g in ['real','synthetic'] for k in ['top1','top3'])
        score=(result['black']['real']['top1']+result['filled']['real']['top1']+.15*result['diverse']['all']['top1']
               -5*result['black']['false_alarms']['rate']-5*result['filled']['false_alarms']['rate'])
        row={'phase':phase,'path':str(path),'validation':result,'guard':guard,'score':score};candidates.append(row)
        print('CANDIDATE',phase,'guard',guard,'black',result['black']['real']['top1'],'filled',result['filled']['real']['top1'],'diverse',result['diverse']['all']['top1'],'score',score,flush=True)
        del qn;torch.cuda.empty_cache()
    eligible=[r for r in candidates if r['guard']] or candidates
    selected=max(eligible,key=lambda r:r['score'])
    temp=OUT/f'{a.name}-selected.part'
    shutil.copyfile(selected['path'],temp);os.replace(temp,OUT/f'{a.name}.pt')
    report={'baseline':baseline,'candidates':candidates,'selected':selected,'rule':'Full pinned recognized-QD/custom validation only. Black real and old-synthetic top1/top3 within .2pp of installed validation preferred. Rank black-real + filled-real + .15 diverse-synthetic-validation minus 5x both custom false-alarm rates. No outside test scores used.'}
    (OUT/f'{a.name}-selection.json').write_text(json.dumps(report,indent=2),encoding='utf-8',newline='\n')
    (OUT/f'{a.name}-selected.txt').write_text(str(OUT/f'{a.name}.pt'),encoding='utf-8',newline='\n')
    print('SELECTED',selected['phase'],selected['guard'],flush=True)


if __name__=='__main__':main()
