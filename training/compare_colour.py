"""Paired int8 colour/fill report. Studio measurements are recorded separately."""
import json
from pathlib import Path
import numpy as np
from categories import CATEGORIES

ROOT=Path(__file__).resolve().parent.parent;OUT=ROOT/'training/out'


def read(name):return json.loads((OUT/(name+'.json')).read_text())


def main():
    selected=read('colour-selection')['selected'];names=[selected,'colour299_40']
    result={'baseline':'wide299_28_r4','selected_28':selected,'models':{},'baseline_modes':read('r4.colour-baseline'),
            'seeds':{'policy_and_filled':20261112,'random':20261113,'fill_colour':20561112},
            'caveats':['Colourise gains mostly measure the supplied class-colour table, not coloured player drawings.',
                'Synthetic tests share their drawing programs with training and remain upper bounds.',
                'Random test inks are independent for each stroke. Filled tests attempt a fill on every drawing; open shapes may be unchanged.',
                'Both sizes use identical source rows and colour assignments. Their native grids, flood travel and random pixel holes differ.',
                '28px starts from r4; 40px retains the trained wide270_40 real path and appends custom features before full fine-tuning. This compares complete candidates, not resolution alone.']}
    for name in names:
        m=read(name);result['models'][name]={k:m[k] for k in ('size','input_channels','params','test_modes','accepted','training_args','selected_epoch','train_seconds')}
        per={}
        for mode in ('black','filled'):
            with np.load(OUT/(name+'.'+mode+'.predictions.npz')) as z:
                truth,top=z['truth'],z['top3'];per[mode]={}
                for ci in range(270,299):
                    mask=truth==ci
                    per[mode][CATEGORIES[ci]]={'top1':float(np.mean(top[mask,0]==ci)),'top3':float(np.any(top[mask]==ci,axis=1).mean())}
        result['models'][name]['per_synthetic']=per
    result['paired_40_minus_28']={}
    for mode in ('black','policy','random','filled'):
        loaded=[]
        for name in names:
            with np.load(OUT/(name+'.'+mode+'.predictions.npz')) as z:loaded.append((z['truth'],z['top3']))
        truth=loaded[0][0];assert np.array_equal(truth,loaded[1][0])
        groups={}
        for group,mask in [('all',np.ones(len(truth),bool)),('real',truth<270),('synthetic',truth>=270)]:
            groups[group]={}
            for key,k in [('top1',1),('top3',3)]:
                a,b=[np.any(p[:,:k]==truth[:,None],axis=1)[mask] for _,p in loaded]
                diff=b.astype(float)-a;mean=diff.mean();se=diff.std(ddof=1)/np.sqrt(len(diff))
                groups[group][key]={'delta_points':float(mean*100),'paired_ci95_points':[float((mean-1.96*se)*100),float((mean+1.96*se)*100)],'only_28':int((a&~b).sum()),'only_40':int((b&~a).sum())}
        result['paired_40_minus_28'][mode]=groups
    p=ROOT/'notes/colour-comparison.json';p.write_text(json.dumps(result,indent=1),encoding='utf-8',newline='\n')
    for name,m in result['models'].items():
        print(name,'accepted',m['accepted'])
        for mode,s in m['test_modes'].items():print(mode,*(f'{g} {s[g]["top1"]:.4%}/{s[g]["top3"]:.4%}' for g in ('all','real','synthetic')))


if __name__=='__main__':main()
