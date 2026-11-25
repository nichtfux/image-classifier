"""Quantize/export a separate candidate; never writes installed/default weights."""
import argparse,json,subprocess,sys,shutil
from pathlib import Path
import numpy as np
import torch
from train_next import load,OUT
from gacl import quantize
from categories import CATEGORIES
from candidate_runtime import build


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--name',default='gacl299_40_wide');ap.add_argument('--out-dir')
    ap.add_argument('--frozen-quant',help='Export these already evaluated int8 arrays without requantizing')
    a=ap.parse_args();torch.set_num_threads(4)
    net,ck=load(a.checkpoint,'cpu')
    if a.frozen_quant:
        with np.load(a.frozen_quant) as frozen:
            q=[(frozen[f'w{i}'],frozen[f's{i}'],frozen[f'b{i}']) for i in range(5)]
    else:
        q=quantize(net)
    owner_corpus=f"colour299-{ck['size']}"
    corpus=json.loads((OUT.parent.parent/'data'/owner_corpus/'corpus.json').read_text())['base']
    destination=OUT/f'{a.name}.npz'
    if a.frozen_quant:
        if Path(a.frozen_quant).resolve()!=destination.resolve():shutil.copyfile(a.frozen_quant,destination)
    else:
        np.savez(destination,**{f'{k}{i}':v for i,row in enumerate(q) for k,v in zip(['w','s','b'],row)})
    old=OUT/f'{a.name}.json';res=json.loads(old.read_text()) if old.exists() else {}
    res.update({'name':a.name,'widths':ck['widths'],'fc':ck['fc'],'size':ck['size'],
        'input_channels':4,'params':sum(p.numel() for p in net.parameters()),'categories':CATEGORIES,
        'corpus':'expanded299-28','ink_corpus':corpus,'owner_corpus':owner_corpus,'feature_normalize':net.cosine,
        'quality_norm_scale':float(net.norm_scale),'int8_top1':res.get('int8_top1',0.),'int8_top3':res.get('int8_top3',0.),
        'checkpoint':str(Path(a.checkpoint).resolve()),'training_phase':ck.get('phase'),'installed':False})
    if ck.get('phase') in ['hard','distill'] or str(ck.get('phase', '')).startswith('balanced-v4'):
        from local_flux import REVISION as flux_revision
        from local_teacher import REVISION as teacher_revision
        res['generated_data']={'generator':'FLUX.1-schnell','revision':flux_revision,'licence':'Apache-2.0',
            'teacher':'SigLIP SO400M/384','teacher_revision':teacher_revision,'snapshot':ck['generated_snapshot']}
    for key in ['calibration', 'rgb_strength', 'raw_feature_scale', 'quality_tier', 'source']:
        if key in ck:res[key]=ck[key]
    old.write_text(json.dumps(res,indent=2),encoding='utf-8',newline='\n')
    out=Path(a.out_dir) if a.out_dir else OUT.parent.parent/'luau/models'/a.name
    assert out.resolve()!= (OUT.parent.parent/'luau').resolve()
    extra=[]
    if str(ck.get('phase', '')).startswith('balanced-v4'):
        root=OUT.parent.parent/'data/generated'
        manifest=Path(ck['generated_snapshot'])/'manifest.json'
        if not manifest.exists():manifest=manifest.parent/'generated/manifest.json'
        rows=[row for row in json.loads(manifest.read_text())['source_rows'] if row['split']=='train']
        selected=sorted(rows,key=lambda row:row['point_count'],reverse=True)[:3]
        for style in ['marker','outline','flat','painted','kids']:
            row=next((row for row in rows if row['style']==style and row not in selected),None)
            if row:selected.append(row)
        vectors=[]
        for row in selected:
            geometry=json.loads((root/'samples'/(row['id']+'.json')).read_text())
            vectors.append({'label':row['label'],'drawing':geometry['strokes'],
                            'colours':[colour-1 for colour in geometry['colours']],
                            'generated':True,'source_id':row['id']})
        path=OUT/f'{a.name}-runtime-probes.json'
        path.write_text(json.dumps(vectors),encoding='utf-8',newline='\n')
        extra=['--extra-vectors',str(path.resolve())]
    if a.name in ['gacl299_40_continued','gacl299_40_genhard','gacl299_40_genkd']:
        root=OUT.parent.parent/'data/generated';snapshot=json.loads((root/'pilot-training-snapshot.json').read_text())
        train_ids=set(snapshot['rows']);rows=[json.loads(l) for l in (root/'manifest.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
        rows=[r for r in rows if r['id'] in train_ids and r['keep']]
        selected=sorted(rows,key=lambda r:r['point_count'],reverse=True)[:3]
        for style in ['marker','outline','flat','painted','kids']:
            candidate=next((r for r in rows if r['style']==style and r not in selected),None)
            if candidate:selected.append(candidate)
        vectors=[]
        for row in selected:
            g=json.loads((root/'samples'/(row['id']+'.json')).read_text())
            vectors.append({'label':row['label'],'drawing':g['strokes'],'colours':[c-1 for c in g['colours']],
                            'generated':True,'source_id':row['id']})
        path=root/'pilot-runtime-probes.json';path.write_text(json.dumps(vectors),encoding='utf-8',newline='\n')
        extra=['--extra-vectors',str(path.resolve())]
    subprocess.run([sys.executable,'export.py','--name',a.name,'--out-dir',str(out.resolve()),'--samples','16','--probes',*extra],cwd=OUT.parent,check=True)
    build(out)
    print('UNINSTALLED',out.resolve(),flush=True)


if __name__=='__main__':main()
