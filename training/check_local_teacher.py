"""Allowed procedural validation smoke check, not an outside-set evaluation."""
import json,time
from pathlib import Path
import torch,numpy as np
from local_teacher import Teacher,ROOT
from generated_convert import preview
from categories import CATEGORIES

if __name__=='__main__':
    torch.set_num_threads(3)
    while not (ROOT/'siglip-so400m-download.json').exists():time.sleep(3)
    teacher=Teacher();data=np.load(ROOT.parent/'synth-diverse40/validation.npz')
    selected=np.arange(0,5800,100);images=[preview(data['x'][i]) for i in selected]
    records=[]
    for lo in range(0,len(images),4):
        z=teacher.score(images[lo:lo+4])
        for j,row in enumerate(z):
            ci=int(data['y'][selected[lo+j]]);decision=teacher.decision(row,ci)
            records.append({'word':CATEGORIES[ci],**decision})
    result={'source':'Our procedural validation only; no teacher fine-tuning. Fixed threshold .05, strict intended top word among 299 plus six reject captions.',
            'text_signature':teacher.text_signature,'n':len(records),'keep_rate':sum(r['keep'] for r in records)/len(records),'records':records}
    (ROOT/'teacher-procedural-smoke.json').write_text(json.dumps(result,indent=2),encoding='utf-8',newline='\n')
    print('TEACHER SMOKE',result['n'],result['keep_rate'],flush=True)
    print('REJECTED',[(r['word'],r['top_label'],round(r['intended_score'],3)) for r in records if not r['keep']],flush=True)
